# -*- coding: utf-8 -*-

'''
蒙特卡罗树搜索（MCTS）的纯实现
'''

import numpy as np
import copy
from operator import itemgetter
from collections import defaultdict


def rollout_policy_fn(board):
    # 初次展示时使用随机方式
    action_probs = np.random.rand(len(board.availables)) # rollout randomly
    return zip(board.availables, action_probs)

def policy_value_fn(board):
    """
    接受状态并输出（动作，概率）列表的函数元组和状态的分数"""
    # 返回统一概率和0分的纯MCTS
    action_probs = np.ones(len(board.availables))/len(board.availables) 
    return zip(board.availables, action_probs), 0

class TreeNode(object):
    """MCTS树中的节点。 每个节点都跟踪自己的值Q，
    先验概率P及其访问次数调整的先前得分u。
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # 一个字典，将动作映射到子节点。
        self._n_visits = 0 # 访问次数
        self._Q = 0 # 平均分数
        self._u = 0 # 节点的奖励值，用于平衡探索和利用,  Q+u 才是ucb
        self._P = prior_p # 节点的先验概率，表示选择该节点的动作的先验概率

    def expand(self, action_priors):
        """通过创建新子项来展开树。
        action_priors：一系列动作元组及其先验概率根据策略函数.
        所有合法动作极其概率
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)
        # expand all children that under this state

    def select(self, c_puct):
        """在子节点中选择能够提供最大行动价值Q的行动加上奖励u（P）。
        return：（action，next_node）的元组
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))
        # self._children is a dict
        # act_node[1].get_value will return the action with max Q+u and corresponding state

    def update(self, leaf_value):
        """从叶节点评估中更新节点值
        leaf_value: 这个子树的评估值来自从当前玩家的视角
        """
        self._n_visits += 1
        # 统计访问次数
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits
        # 更新Q值,取对于所有访问次数的平均数
        # (v-Q)/(n+1)+Q = (v-Q+(n+1)*Q)/(n+1)=(v+n*Q)/(n+1)

    def update_recursive(self, leaf_value):
        """就像调用update（）一样，但是对所有祖先进行递归应用。
        """
        # 如果它不是根节点，则应首先更新此节点的父节点。
        if self._parent:
            self._parent.update_recursive(-leaf_value)
            # 因为你的父节点是队友的回合，你的收益是他的损失
        self.update(leaf_value)

    def get_value(self, c_puct):
        '''计算并返回此节点的值。它是叶评估Q和此节点的先验的组合
        调整了访问次数，u。
        c_puct：控制相对影响的（0，inf）中的数字，该节点得分的值Q和先验概率P. c 一般取常数2
        self._parent._n_visits 是总探索次数
        '''
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """检查叶节点（即没有扩展的节点）。
        """
        return self._children == {}

    def is_root(self):
        """检查根节点
        """
        return self._parent is None

class MCTS(object):
    """对蒙特卡罗树搜索的一个简单实现"""
    def __init__(self, policy_value_fn, c_puct=5, n_playout=400):
        """
        policy_value_fn：一个接收板状态和输出的函数
        （动作，概率）元组列表以及[-1,1]中的分数
        （即来自当前的最终比赛得分的预期值
        玩家的观点）对于当前的玩家。
        c_puct：（0，inf）中的数字，用于控制探索的速度
        收敛于最大值政策。 更高的价值意味着
        依靠先前的更多。
        """
        self._root = TreeNode(parent=None, prior_p=1.0)
        # 该节点没有父节点，并且具有先验概率为1.0。根节点表示当前的游戏状态。
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout # 树搜索次数

    def _playout(self, state):
       
        """
        从根到叶子运行单个播出，获取值
        叶子并通过它的父母传播回来。
        State已就地修改，因此必须提供副本。
        """
        node = self._root
        while(1):
            # 选择动作
            
            if node.is_leaf():
                # break if the node is leaf node
                # print('breaking...................................')
                break
            # Greedily select next move.
            # 贪心算法选择下一步行动
            action, node = node.select(self._c_puct)   
            # print('select action is ...',action)
            # print(action,state.availables)
            state.do_move(action)   
            # this state should be the same state with current node

        action_probs, _ = self._policy(state)
        # 查询游戏是否终结
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        # 通过随机的rollout评估叶子结点 
        
        leaf_value = self._evaluate_rollout(state)
        # 在本次遍历中更新节点的值和访问次数
        node.update_recursive(-leaf_value)
        # print('after update...', node._n_visits, node._Q)

    def _evaluate_rollout(self, state, limit=1000):

        """使用推出策略直到游戏结束，
        如果当前玩家获胜则返回+1，如果对手获胜则返回-1，
        如果是平局则为0。
        """
        player = state.get_current_player()
        for i in range(limit):
            end, winner = state.game_end()
            if end:
                break
            action_probs = rollout_policy_fn(state)
            max_action = max(action_probs, key=itemgetter(1))[0]
            # 返回[('action1', 0.3), ('action2', 0.5), ('action3', 0.2)] 中概率最大的值的动作
            # itemgetter
            # a = [1,2,3] 
            # >>> b=operator.itemgetter(1)      //定义函数b，获取对象的第1个域的值
            # >>> b(a) 
            # 2
            # https://www.cnblogs.com/zhoufankui/p/6274172.html
            state.do_move(max_action)
        else:
            # 如果没有从循环中断，请发出警告。
            print("WARNING: rollout reached move limit")
        if winner == -1:  # tie
            return 0
        else:
            return 1 if winner == player else -1

    def get_move(self, state):
 
        """按顺序运行所有播出并返回访问量最大的操作。
        state：当前的比赛状态
        return ：所选操作
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
            # use deepcopy and playout on the copy state

        # some statistics just for check
        # visits_count = defaultdict(int)
        # visits_count_dic = defaultdict(int)
        # self.sum = 0
        # Q_U_dic = defaultdict(int)
        # for act,node in self._root._children.items():
        #     visits_count[act] += node._n_visits
        #     visits_count_dic[str(state.move_to_location(act))] += node._n_visits
        #     self.sum += node._n_visits
        #     Q_U_dic[str(state.move_to_location(act))] = node.get_value(5)

        # print(Q_U_dic)
        # print(self.sum,visits_count_dic)

        return max(self._root._children.items(),
                   key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        '''保留我们已经知道的关于子树的信息
        '''
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"

class MCTSPlayer(object):
    """基于MCTS的AI玩家"""
    def __init__(self, c_puct=5, n_playout=400):
        '''
        初始化类
        '''
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)

    def set_player_ind(self, p):
        '''
        设置玩家编号
        '''
        self.player = p

    def reset_player(self):
        '''
        重置玩家状态
        '''
        self.mcts.update_with_move(-1) # reset the node

    def get_action(self, board,is_selfplay=False,print_probs_value=0): 
        '''
        用于根据当前棋盘状态获取下一步的动作。
        '''
        sensible_moves = board.availables
        if board.last_move!=-1: # 说明之前有落子
            self.mcts.update_with_move(last_move=board.last_move)
            # 更新蒙特卡罗树，以便重用之前的搜索结果。
            # retain the tree that can continue to use
            # so update the tree with opponent's move and do mcts from the current node

        if len(sensible_moves) > 0:
            move = self.mcts.get_move(board)
            self.mcts.update_with_move(move)
            # every time when get a move, update the tree
        else:
            print("WARNING: the board is full")

        return move, None

    def __str__(self):
        return "MCTS {}".format(self.player)








