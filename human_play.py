# -*- coding: utf-8 -*-
 
from __future__ import print_function
from game_board import Board, Game
from mcts_pure import MCTSPlayer as MCTS_pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net import PolicyValueNet
import time
from os import path
import os
from collections import defaultdict

class Human(object):
    """
    玩家
    """
    def __init__(self): #  初始化人类玩家对象。
        self.player = None

    def set_player_ind(self, p): #  设置玩家编号。
        self.player = p

    def get_action(self, board,is_selfplay=False,print_probs_value=0):
        # 获取人类玩家的移动输入，要求输入格式为行列坐标，如"2 3"。
        try:
            location = input("Your move: ")
            if isinstance(location, str):  
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location) # 二维转一维
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move,_ = self.get_action(board) # 再次输入
        return move,None

    def __str__(self): # 返回玩家的字符串表示。
        return "Human {}".format(self.player)

def run(start_player=0,is_shown=1): # 人和电脑打
   
    n = 5
    width, height = 11, 11
    model_file = 'model_11_11_5/best_policy.model'
    p = os.getcwd()
    model_file = path.join(p,model_file) 

    board = Board(width=width, height=height, n_in_row=n)
    game = Game(board)

    mcts_player = MCTS_pure(5,400)

    best_policy = PolicyValueNet(board_width=width,board_height=height,block=19,init_model=model_file,cuda=True)

 
    alpha_zero_player = MCTSPlayer(policy_value_function=best_policy.policy_value_fn_random,
                                   action_fc=best_policy.action_fc_test,
                                   evaluation_fc=best_policy.evaluation_fc2_test,
                                   c_puct=5,
                                   n_playout=400,
                                   is_selfplay=False)
    # AI vs AI
    # alpha_zero_player_oppo = MCTSPlayer(policy_value_function=best_policy.policy_value_fn_random,
    #                                     action_fc=best_policy.action_fc_test_oppo,
    #                                     evaluation_fc=best_policy.evaluation_fc2_test_oppo,
    #                                     c_puct=5,
    #                                     n_playout=400,
    #                                     is_selfplay=False)

    # 玩家输入: 2,3
    # 使用界面玩
    game.start_play_with_UI(alpha_zero_player)


if __name__ == '__main__':
    run(start_player=0,is_shown=True)