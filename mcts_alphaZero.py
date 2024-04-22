# -*- coding: utf-8 -*-
"""
蒙特卡罗树搜索AlphaGo Zero形式，使用策略值网络引导树搜索和评估叶节点
"""

import numpy as np
import copy

def softmax(x):
    probs = np.exp(x - np.max(x))
    # https://mp.weixin.qq.com/s/2xYgaeLlmmUfxiHCbCa8dQ
    # avoid float overflow and underflow
    probs /= np.sum(probs)
    return probs

class TreeNode(object):
  
    """MCTS树中的节点。
    每个节点跟踪其自身的值Q，先验概率P及其访问次数调整的先前得分u。
    """
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}   
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p  

    def expand(self, action_priors,add_noise):
 
        """通过创建新子项来展开树。
        action_priors：一系列动作元组及其先验概率根据策略函数.
        """
        # when train by self-play, add dirichlet noises in each node

        # should note it's different from paper that only add noises in root node
        # i guess alphago zero discard the whole tree after each move and rebuild a new tree, so it's no conflict
        # while here i contained the Node under the chosen action, it's a little different.
        # there's no idea which is better
        # in addition, the parameters should be tried
        # for 11x11 board,
        # dirichlet parameter :0.3 is ok, should be smaller with a bigger board,such as 20x20 with 0.03
        # weights between priors and noise: 0.75 and 0.25 in paper and i don't change it here,
        # but i think maybe 0.8/0.2 or even 0.9/0.1 is better because i add noise in every node
        # rich people can try some other parameters
        if add_noise:
            action_priors = list(action_priors)
            length = len(action_priors)
            dirichlet_noise = np.random.dirichlet(0.3 * np.ones(length))
            for i in range(length):
                if action_priors[i][0] not in self._children:
                    self._children[action_priors[i][0]] = TreeNode(self,0.75*action_priors[i][1]+0.25*dirichlet_noise[i])
        else:
            for action, prob in action_priors:
                if action not in self._children:
                    self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
 
        """在子节点中选择能够提供最大行动价值Q的行动加上奖金u（P）。
        return：（action，next_node）的元组
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
    
        """从叶节点评估中更新节点值
        leaf_value: 这个子树的评估值来自从当前玩家的视角
        """
        self._n_visits += 1
        # 统计访问次数
        # update visit count
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits
        # 更新Q值， 取对于所有访问次数的平均数
        # Update Q, a running average of values for all visits.
        # there is just: (v-Q)/(n+1)+Q = (v-Q+(n+1)*Q)/(n+1)=(v+n*Q)/(n+1)

    def update_recursive(self, leaf_value):
 
        """就像调用update（）一样，但是对所有祖先进行递归应用。
        """
        # 如果它不是根节点，则应首先更新此节点的父节点。
        if self._parent:
            self._parent.update_recursive(-leaf_value)
 
        self.update(leaf_value)

    def get_value(self, c_puct):
         
        """计算并返回此节点的值。它是叶评估Q和此节点的先验的组合
        调整了访问次数，u。
        c_puct：控制相对影响的（0，inf）中的数字，该节点得分的值Q和先验概率P.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        '''
        check if leaf node (i.e. no nodes below this have been expanded).
        '''
        """检查叶节点（即没有扩展的节点）。"""
        return self._children == {}

    def is_root(self):
        '''
        check if it's root node
        '''
        return self._parent is None


class MCTS(object):
 
    """对蒙特卡罗树搜索的一个简单实现"""
    def __init__(self, policy_value_fn,action_fc,evaluation_fc, is_selfplay,c_puct=5, n_playout=400):
     
        """
        policy_value_fn：一个接收板状态和输出的函数（动作，概率）元组列表以及[-1,1]中的分数
        （即来自当前的最终比赛得分的预期值玩家的观点）对于当前的玩家。
        c_puct：（0，inf）中的数字，用于控制探索的速度收敛于最大值政策。 更高的价值意味着
        依靠先前的更多。
        """
        self._root = TreeNode(None, 1.0)
        # root node do not have parent ,and sure with prior probability 1

        self._policy_value_fn = policy_value_fn
        self._action_fc = action_fc
        self._evaluation_fc = evaluation_fc

        self._c_puct = c_puct
        # it's 5 in paper and don't change here,but maybe a better number exists in gomoku domain
        self._n_playout = n_playout # times of tree search
        self._is_selfplay = is_selfplay

    def _playout(self, state):
    
        """从根到叶子运行单个播出，获取值
         叶子并通过它的父母传播回来。
         State已就地修改，因此必须提供副本。
        """
        node = self._root
        # print('============node visits:',node._n_visits)
        # deep = 0
        while(1):
            if node.is_leaf():
                break
            # 贪心算法选择下一步行动
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            # print('move in tree...',action)
            state.do_move(action)
            # deep+=1
        # print('-------------deep is :',deep)

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        # 使用网络评估叶子，该网络输出（动作，概率）元组p的列表以及当前玩家的[-1,1]中的分数v。
        action_probs, leaf_value = self._policy_value_fn(state,self._action_fc,self._evaluation_fc)
        # 查看游戏是否结束
        end, winner = state.game_end()
        if not end:
            # print('expand move:',state.width*state.height-len(state.availables),node._n_visits)
            node.expand(action_probs,add_noise=self._is_selfplay)
        else:
            # 对于结束状态,将叶子节点的值换成"true"
            # for end state，return the "true" leaf_value
            # print('end!!!',node._n_visits)
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player() else -1.0
                )
        # 在本次遍历中更新节点的值和访问次数
        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)
        # no rollout here

    def get_move_visits(self, state):
 
        """按顺序运行所有播出并返回可用的操作及其相应的概率。
        state: 当前游戏的状态
        temp: 介于(0,1]之间的临时参数控制探索的概率
        """
        for n in range(self._n_playout):
            # print('playout:',n)
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # calc the visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)

        return acts, visits

    def update_with_move(self, last_move):
 
        """在当前的树上向前一步，保持我们已经知道的关于子树的一切.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"

class MCTSPlayer(object):
     
    """基于MCTS的AI玩家"""
    def __init__(self, policy_value_function,action_fc,evaluation_fc,c_puct=5, n_playout=400, is_selfplay=0):
        '''
        init some parameters
        '''
        self._is_selfplay = is_selfplay # 自我博弈
        self.policy_value_function = policy_value_function # 策略价值函数，用于评估棋局状态的动作价值和策略概率
        self.action_fc = action_fc # 用于生成可行动作
        self.evaluation_fc = evaluation_fc # 用于评估棋局状态的胜负情况
        self.first_n_moves = 12
        # For the first n moves of each game, the temperature is set to τ = 1,
        # For the remainder of the game, an infinitesimal temperature is used, τ→ 0.
        # in paper n=30, here i choose 12 for 11x11, entirely by feel
        self.mcts = MCTS(policy_value_fn = policy_value_function,
                         action_fc = action_fc,
                         evaluation_fc = evaluation_fc,
                         is_selfplay = self._is_selfplay,
                         c_puct = c_puct,
                         n_playout = n_playout)

    def set_player_ind(self, p):
        '''
        设置玩家索引
        '''
        self.player = p

    def reset_player(self):
        '''
        重置玩家状态
        '''
        self.mcts.update_with_move(-1)

    def get_action(self,board,is_selfplay,print_probs_value):
        '''
        下一步
        '''
        sensible_moves = board.availables
        # 像alphaGo Zero论文一样使用MCTS算法返回的pi向量
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(board.width * board.height)
        if len(sensible_moves) > 0:
            if is_selfplay:
                acts, visits = self.mcts.get_move_visits(board)
                if board.width * board.height - len(board.availables) <= self.first_n_moves:
                    # For the first n moves of each game, the temperature is set to τ = 1
                    temp = 1
                    probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
                    move = np.random.choice(acts, p=probs)
                else:
                    # For the remainder of the game, an infinitesimal temperature is used, τ→ 0
                    temp = 1e-3
                    probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
                    move = np.random.choice(acts, p=probs)

                self.mcts.update_with_move(move)
                # update the tree with self move
            else:
                self.mcts.update_with_move(board.last_move)
                # update the tree with opponent's move and then do mcts from the new node

                acts, visits = self.mcts.get_move_visits(board)
                temp = 1e-3
                # always choose the most visited move
                probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
                move = np.random.choice(acts, p=probs)

                self.mcts.update_with_move(move)
                # update the tree with self move

            p = softmax(1.0 / 1.0 * np.log(np.array(visits) + 1e-10))
            move_probs[list(acts)] = p
            # return the prob with temp=1

            if print_probs_value and move_probs is not None:
                act_probs, value = self.policy_value_function(board,self.action_fc,self.evaluation_fc)
                print('-' * 10)
                print('value',value)
                # print the probability of each move
                probs = np.array(move_probs).reshape((board.width, board.height)).round(3)[::-1, :]
                for p in probs:
                    for x in p:
                        print("{0:6}".format(x), end='')
                    print('\r')

            return move,move_probs

        else:
            print("棋盘已满")

    def __str__(self):
        return "Alpha {}".format(self.player)


