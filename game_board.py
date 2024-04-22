# -*- coding: utf-8 -*-

import numpy as np
from collections import deque
from GUI_v1_4 import GUI

class Board(object):
     
    def __init__(self, **kwargs):   # **kwargs表示输入的变量不定
        self.width = int(kwargs.get('width', 11))
        self.height = int(kwargs.get('height', 11))
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.states = {}
        self.players = [1, 2]
        # player1 and player2

        self.feature_planes = 8
        # 在AlphaGo Zero中，使用17个特征平面，而这里设置为8个。
        # 特征平面是用于表示棋局状态的二进制平面，用于输入到神经网络中。
        # 这里的8个特征平面包括历史特征和当前玩家的颜色特征。
        self.states_sequence = deque(maxlen=self.feature_planes)
        self.states_sequence.extendleft([[-1,-1]] * self.feature_planes)
        #  创建一个deque对象，用于存储最近8次移动的位置，第一个是move，第二个是player
        #  初始时，队列中填充了self.feature_planes个[-1, -1]元素，表示游戏开始时没有移动

    def init_board(self, start_player=0):
        '''
        初始化棋盘状态和一些变量
        '''
        # if self.width < self.n_in_row or self.height < self.n_in_row:
        #     raise Exception('board width and height can not be '
        #                     'less than {}'.format(self.n_in_row))

        self.current_player = self.players[start_player]  # 先手
        self.availables = list(range(self.width * self.height)) # 11*11=121
        # 初始化所有的位置，开始时所有的位置都可以用
        self.states = {} # 状态
        self.last_move = -1 # 最后一步

        self.states_sequence = deque(maxlen=self.feature_planes)
        self.states_sequence.extendleft([[-1, -1]] * self.feature_planes)

    def move_to_location(self, move):  # 将一维转成二维
        '''
        比如3*3:
        6 7 8
        3 4 5
        0 1 2
        5 对应的是（1，2）
        '''
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location): # 二维转一维
        if len(location) != 2: # 没有两个值
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height): # 超出范围
            return -1
        return move

    def current_state(self):
        '''
        从当前玩家返回
        状态形式: (self.feature_planes+1) x width x height
        '''
        square_state = np.zeros((self.feature_planes+1, self.width, self.height))
        if self.states:  # 棋盘存在
            moves, players = np.array(list(zip(*self.states.items())))
            # 获取到玩家key，value对应到move，player
            # moves = np.array([1, 2, 3])
            # players = np.array([1, 1, 2])
            move_curr = moves[players == self.current_player] # 当前玩家移动
            move_oppo = moves[players != self.current_player]

            # to construct the binary feature planes as alphazero did
            for i in range(self.feature_planes):
                # put all moves on planes
                if i%2 == 0: # 说明该平面是对手的棋子位置平面
                    square_state[i][move_oppo // self.width,move_oppo % self.height] = 1.0
                else:
                    square_state[i][move_curr // self.width,move_curr % self.height] = 1.0
            # 删除一些移动去构造历史平面特征
            for i in range(0,len(self.states_sequence)-2,2): # 遍历偶数
                for j in range(i+2,len(self.states_sequence),2): # 遍历奇数
                    if self.states_sequence[i][1]!= -1: # 有效移动
                        assert square_state[j][self.states_sequence[i][0] // self.width,self.states_sequence[i][0] % self.height] == 1.0, 'wrong oppo number'
                        square_state[j][self.states_sequence[i][0] // self.width, self.states_sequence[i][0] % self.height] = 0.
            for i in range(1,len(self.states_sequence)-2,2):
                for j in range(i+2,len(self.states_sequence),2):
                    if self.states_sequence[i][1] != -1:
                        assert square_state[j][self.states_sequence[i][0] // self.width,self.states_sequence[i][0] % self.height] ==1.0, 'wrong player number'
                        square_state[j][self.states_sequence[i][0] // self.width, self.states_sequence[i][0] % self.height] = 0.

        if len(self.states) % 2 == 0:
            #  如果棋子数量为偶数，表示轮到玩家1落子，将最后一个平面 square_state[self.feature_planes] 全部设置为 1.0，表示轮到玩家1。否则，将该平面全部设置为 0.
            square_state[self.feature_planes][:, :] = 1.0 
        return square_state[:, ::-1, :]
        # 反转
        # 0,1,2,
        # 3,4,5,
        # 6,7,8,
        # we will change it like
        # 6 7 8
        # 3 4 5
        # 0 1 2
        

    def do_move(self, move):
        '''
        更新棋盘
        '''
        self.states[move] = self.current_player # 每一步对应的玩家
        self.states_sequence.appendleft([move,self.current_player])
        self.availables.remove(move)
        # 从棋盘中移除
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        # 切换玩家
        self.last_move = move # 保存最后一步

    def has_a_winner(self): # 判断胜者
     
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        # moves have been played
        if len(moved) < self.n_in_row + 2:
            # too few moves to get 5-in-a-row
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                # 向右遍历
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                # 向左遍历
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                # 向右上方遍历
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                # 向左上方遍历
                return True, player

        return False, -1

    def game_end(self): # 检查游戏是否结束
        end, winner = self.has_a_winner()
        if end:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self): # 获得当前玩家编号
        return self.current_player

class Game(object):
     
    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, player1, player2):
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print(board.states)
        print()
        print(' ' * 2, end='')
        # rjust()
        # http://www.runoob.com/python/att-string-rjust.html
        for x in range(width):
            print("{0:4}".format(x), end='')
        # print('\r\n')
        print('\r')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(4), end='')
                elif p == player2:
                    print('O'.center(4), end='')
                else:
                    print('-'.center(4), end='')
            # print('\r\n')
            print('\r')

    def start_play(self, player1, player2, start_player=0, is_shown=1,print_prob =True):
        '''
        开始游戏
        '''
        if start_player not in (0, 1):
            raise Exception('玩家编号不合理')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        # print(p1,p2)
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}

        if is_shown: 
            self.graphic(self.board, player1.player, player2.player)

        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move,move_probs = player_in_turn.get_action(self.board,is_selfplay=False,print_probs_value=print_prob)

            self.board.do_move(move)

            if is_shown:
                print('player %r move : %r' % (current_player, [move // self.board.width, move % self.board.width]))
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()

            if end:
                if is_shown:
                    if winner != -1:
                        print("胜者是", players[winner])
                    else:
                        print("游戏结束")
                return winner

    def start_play_with_UI(self, AI, start_player=0):
        '''
        使用界面开始
        '''
        AI.reset_player() # 重置AI的状态
        self.board.init_board()
        current_player = SP = start_player # SP是起始玩家
        UI = GUI(self.board.width)
        end = False
        while True:
            print('当前玩家', current_player)  
            if current_player == 0:
                UI.show_messages('Your ')
            else:
                UI.show_messages('AI ')
            # print_probs_value 打印每个位置的概率
            if current_player == 1 and not end: 
                move, move_probs = AI.get_action(self.board, is_selfplay=False, print_probs_value=1)
            else:
                inp = UI.get_input()
                if inp[0] == 'move' and not end:
                    if type(inp[1]) != int:
                        move = UI.loc_2_move(inp[1])
                    else:
                        move = inp[1]
                elif inp[0] == 'RestartGame':
                    end = False
                    current_player = SP
                    self.board.init_board()
                    UI.restart_game()
                    AI.reset_player()
                    continue
                elif inp[0] == 'ResetScore':
                    UI.reset_score()
                    continue
                elif inp[0] == 'quit':
                    exit()
                elif inp[0] == 'SwitchPlayer':
                    end = False
                    self.board.init_board()
                    UI.restart_game(False)
                    UI.reset_score()
                    AI.reset_player()
                    SP = (SP+1) % 2
                    current_player = SP
                    continue
            if not end:
                # print(move, type(move), current_player)
                UI.render_step(move, self.board.current_player)
                self.board.do_move(move)
                # print('move', move)
                # print(2, self.board.get_current_player())
                current_player = (current_player + 1) % 2
                # UI.render_step(move, current_player)
                end, winner = self.board.game_end()
                if end:
                    if winner != -1:
                        print("Game end. Winner is player", winner)
                        UI.add_score(winner)
                        UI.show_message_box(winner)
                    else:
                        print("Game end. Tie")
                    print(UI.score)
                    print()

    def start_self_play(self, player, is_shown=0):
        '''
        使用了一个MCTS（蒙特卡洛树搜索）玩家来模拟自我对弈的过程
        存储数据: (state, mcts_probs, z) 用于训练
        '''
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], [] # mcts_probs对应概率值
        while True:
            move, move_probs = player.get_action(self.board,
                                                 is_selfplay=True,
                                                 print_probs_value=False)
            # 存储数据
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player) 
            
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0 # 胜者为1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)

