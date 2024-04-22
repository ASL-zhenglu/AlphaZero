# -*- coding: utf-8 -*-
 
from __future__ import print_function
import random
import numpy as np
import os
import time
from collections import defaultdict, deque
from game_board import Board,Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net import PolicyValueNet
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

data_train = {"batch": [], "value": [], "criterion": []}
data_eval = {"batch": [], "win_ratio": []}



class TrainPipeline():
    def __init__(self, init_model=None,transfer_model=None):
        self.resnet_block = 19  # resnet 残差块数量
        # 五子棋逻辑和棋盘UI的参数
        self.board_width = 11
        self.board_height = 11
        self.n_in_row = 5
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)
        # 训练参数
        self.learn_rate = 1e-3  # 学习率
        self.n_playout = 400  # 每一步模拟次数
        self.c_puct = 5
        self.buffer_size = 500000 # 缓冲区大小，避免内存过大
        self.batch_size = 512  # 小批量训练
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1 # 每下多少局训练一次网络
        self.check_freq = 50
        self.game_batch_num = 50000000 # 总的游戏训练次数
        self.best_win_ratio = 0.0  # 最佳胜率
       
        self.pure_mcts_playout_num = 200 # # 用于纯粹的mcts的模拟数量，用作评估训练策略的对手
        if (init_model is not None) and os.path.exists(init_model+'.index'):
            # 初始化的策略开始训练
            self.policy_value_net = PolicyValueNet(self.board_width,self.board_height,block=self.resnet_block,init_model=init_model,cuda=True)
        elif (transfer_model is not None) and os.path.exists(transfer_model+'.index'):
            # 以前面策略开始训练
            self.policy_value_net = PolicyValueNet(self.board_width,self.board_height,block=self.resnet_block,transfer_model=transfer_model,cuda=True)
        else:
            # 以一个新的策略开始训练
            self.policy_value_net = PolicyValueNet(self.board_width,self.board_height,block=self.resnet_block,cuda=True)
        # 定义训练机器人
        self.mcts_player = MCTSPlayer(policy_value_function=self.policy_value_net.policy_value_fn_random,
                                       action_fc=self.policy_value_net.action_fc_test,
                                       evaluation_fc=self.policy_value_net.evaluation_fc2_test,
                                       c_puct=self.c_puct,
                                       n_playout=self.n_playout,
                                       is_selfplay=True)

    def get_equi_data(self, play_data):
        '''
        通过旋转和翻转来增加数据集
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        '''
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # 逆时针旋转
                equi_state = np.array([np.rot90(s, i) for s in state])
                #rotate counterclockwise 90*i
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i)
                #np.flipud like A[::-1,...]
                #https://docs.scipy.org/doc/numpy-1.6.0/reference/generated/numpy.flipud.html
                # change the reshaped numpy
                # 0,1,2,
                # 3,4,5,
                # 6,7,8,
                # as
                # 6 7 8
                # 3 4 5
                # 0 1 2
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # 水平翻转
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                #这个np.fliplr like m[:, ::-1]
                #https://docs.scipy.org/doc/numpy/reference/generated/numpy.fliplr.html
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        '''
        收集自我博弈数据进行训练， n_games：次数
        '''
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player,is_shown=False)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # 增加数据
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data) # 添加

    def policy_update(self):
        '''
        更新策略价值网络
        '''
        # play_data: [(state, mcts_prob, winner_z), ..., ...]
        # train an epoch

        tmp_buffer = np.array(self.data_buffer)
        np.random.shuffle(tmp_buffer)
        steps = len(tmp_buffer)//self.batch_size
        print('tmp buffer: {}, steps: {}'.format(len(tmp_buffer),steps))
        for i in range(steps):
            mini_batch = tmp_buffer[i*self.batch_size:(i+1)*self.batch_size] # 当前批数据
            state_batch = [data[0] for data in mini_batch]
            mcts_probs_batch = [data[1] for data in mini_batch]
            winner_batch = [data[2] for data in mini_batch]

            old_probs, old_v = self.policy_value_net.policy_value(state_batch=state_batch,
                                                                  actin_fc=self.policy_value_net.action_fc_test,
                                                                  evaluation_fc=self.policy_value_net.evaluation_fc2_test)
            # 旧的策略概率和旧的价值估计
            loss, entropy = self.policy_value_net.train_step(state_batch,
                                                             mcts_probs_batch,
                                                             winner_batch,
                                                             self.learn_rate)
            # 获取损失值和熵
            new_probs, new_v = self.policy_value_net.policy_value(state_batch=state_batch,
                                                                  actin_fc=self.policy_value_net.action_fc_test,
                                                                  evaluation_fc=self.policy_value_net.evaluation_fc2_test)
            # 获取新的策略概率和新的价值估计
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            # 衡量旧的策略概率与新的策略概率之间的差异
            explained_var_old = (1 -
                                 np.var(np.array(winner_batch) - old_v.flatten()) /
                                 np.var(np.array(winner_batch)))
            
            # 衡量旧的价值估计对胜利者结果的解释能力
            explained_var_new = (1 -
                                 np.var(np.array(winner_batch) - new_v.flatten()) /
                                 np.var(np.array(winner_batch)))

            if steps<10 or (i%(steps//10)==0):
                # print some information, not too much
                print('batch: {},length: {}'
                      'kl:{:.5f},'
                      'loss:{},'
                      'entropy:{},'
                      'explained_var_old:{:.3f},'
                      'explained_var_new:{:.3f}'.format(i,
                                                        len(mini_batch),
                                                        kl,
                                                        loss,
                                                        entropy,
                                                        explained_var_old,
                                                        explained_var_new))

        return loss, entropy

    def policy_evaluate(self, n_games=10):
        '''
        通过与纯的MCTS算法对抗来评估训练的策略
        注意：这仅用于监控训练进度
        '''
        current_mcts_player = MCTSPlayer(policy_value_function=self.policy_value_net.policy_value_fn_random,
                                       action_fc=self.policy_value_net.action_fc_test,
                                       evaluation_fc=self.policy_value_net.evaluation_fc2_test,
                                       c_puct=5,
                                       n_playout=400,
                                       is_selfplay=False)

        test_player = MCTS_Pure(c_puct=5,
                                n_playout=self.pure_mcts_playout_num)

        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(player1=current_mcts_player,
                                          player2=test_player,
                                          start_player=i % 2,
                                          is_shown=0,
                                          print_prob=False)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def run(self):
        '''
        开始训练
        '''
        # 创造文件
        if not os.path.exists('tmp'):
            os.makedirs('tmp')
        if not os.path.exists('model'):
            os.makedirs('model')

        # 记录每一部分时间
        start_time = time.time()
        collect_data_time = 0
        train_data_time = 0
        evaluate_time = 0

        try:
            for i in range(self.game_batch_num):
                # 收集自我博弈的时间
                collect_data_start_time = time.time()
                self.collect_selfplay_data(self.play_batch_size)
                collect_data_time += time.time()-collect_data_start_time
                print("batch i:{}, episode_len:{}".format(
                        i+1, self.episode_len))

                if len(self.data_buffer) > self.batch_size*5: 
                    # 收集训练数据
                    train_data_start_time = time.time()
                    loss, entropy = self.policy_update()
                    train_data_time += time.time()-train_data_start_time
    
                    
                    data_train["batch"] += ([i+1, i+1])
                    data_train["value"] += [float(loss), float(entropy)]
                    data_train["criterion"] += ["loss", "entropy"]

                    # 输出
                    print('now time : {}'.format((time.time() - start_time) / 3600))
                    print('collect_data_time : {}, train_data_time : {},evaluate_time : {}'.format(
                        collect_data_time / 3600, train_data_time / 3600,evaluate_time/3600))

                if (i+1) % self.check_freq == 0 :

                    # 保存当前的模型用于评估
                    self.policy_value_net.save_model('tmp/current_policy.model')
                    if (i+1) % (self.check_freq*2) == 0: 
                        print("current self-play batch: {}".format(i + 1))
                        evaluate_start_time = time.time()

                        # 评估当前模型
                        win_ratio = self.policy_evaluate(n_games=10)   

                        data_eval["batch"].append(i+1)
                        data_eval["win_ratio"].append(win_ratio)

                        evaluate_time += time.time()-evaluate_start_time
                        if win_ratio > self.best_win_ratio:
                           
                            print("New best policy!!!!!!!!")
                            self.best_win_ratio = win_ratio
                            self.policy_value_net.save_model('model/best_policy.model')

                            if (self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 5000):
                                # 增加模拟次数
                                self.pure_mcts_playout_num += 100 # pure_mcts_playout_num 纯mcts模拟次数
                                self.best_win_ratio = 0.0
                            if self.pure_mcts_playout_num ==5000:
                                # 重置模拟次数
                                self.pure_mcts_playout_num = 1000
                                self.best_win_ratio = 0.0

        except KeyboardInterrupt: 
            print('\n\rquit')

def draw(data_train, data_eval):
    data_train = DataFrame(data_train)
    data_eval = DataFrame(data_eval)

    plt.figure(figsize=(10, 5))

    # 绘制训练数据的图表
    plt.subplot(1, 2, 1) # 一行两列，第一列
    ax = sns.lineplot(x="batch", y="value", hue="criterion", style="criterion", data=data_train)
    plt.xlabel("batch", fontsize=12)
    plt.ylabel("value", fontsize=12)
    plt.title("AlphaZero Gomoku Performance (train)", fontsize=14)

    # 绘制评估数据的图表
    plt.subplot(1, 2, 2)
    ax = sns.lineplot(x="batch", y="win_ratio", data=data_eval)
    plt.xlabel("batch", fontsize=12)
    plt.ylabel("win ratio", fontsize=12)
    plt.title("AlphaZero Gomoku Performance (eval)", fontsize=14)

    plt.tight_layout()  # 调整子图的布局，防止重叠
    plt.show()



if __name__ == '__main__':
    training_pipeline = TrainPipeline(init_model='model/best_policy.model',transfer_model=None)
    # training_pipeline = TrainPipeline(init_model=None, transfer_model='transfer_model/best_policy.model')
    # training_pipeline = TrainPipeline()
    training_pipeline.run()

    draw(data_train, data_eval)