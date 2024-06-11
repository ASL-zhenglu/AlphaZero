# -*- coding: utf-8 -*-

import pygame
from pygame.locals import *
from pygame.locals import QUIT, MOUSEBUTTONDOWN, MOUSEMOTION
import time
class GUI:

    def __init__(self, board_size=11):
        pygame.init()

        self.score = [0, 0]
        self.BoardSize = board_size # 棋盘尺寸
        self.UnitSize = 40      # 所有元素的基本大小，用于计算界面元素的位置和大小
        self.TestSize = int(self.UnitSize * 0.625) # 文本大小
        self.state = {}         # 字典的键是棋盘上的位置，值是移动的玩家。
        self.areas = {}         # 字典的键是区域的名称，值是对应的矩形区域
        self.ScreenSize = None  # 屏幕的尺寸，用于计算界面元素的位置和大小
        self.screen = None # 用于显示游戏界面
        self.last_action_player = None # 上次操作的玩家
        self.round_counter = 0 # 回合
        self.messages = ''
        self._background_color = (197, 227, 205) # 背景颜色
        self._board_color = (254, 185, 120) # 棋盘的颜色

        self.reset(board_size) # 重置

         
        self.restart_game(False) # 重置游戏
        self.reset_score() # 重置得分

    def show_message_box(self, player):
        font = pygame.font.Font(None, 36)
        message = f"player {player} win"
        text = font.render(message, True, (0, 0, 0))
        text_rect = text.get_rect(center=(self.screen.get_width() / 2, self.screen.get_height() / 2))

        # start_time = time.time()
        # display_time = 5  # 显示时间（秒）

        # while time.time() - start_time < display_time:
        #     for event in pygame.event.get():
        #         if event.type == pygame.QUIT:  # 检测到退出事件时退出循环
        #             return

            # 在原本界面上绘制提示框
        self.screen.blit(text, text_rect)
        pygame.display.update(text_rect)  # 只更新提示框的区域，不覆盖其他部分


    def reset(self, bs): # bs： boardsize, 功能:重置棋盘
 
        self.BoardSize = bs
        self.ScreenSize = (self.BoardSize * self.UnitSize + 2 * self.UnitSize,
                           self.BoardSize * self.UnitSize + 3 * self.UnitSize)
        # 设置宽和高
        self.screen = pygame.display.set_mode(self.ScreenSize, 0, 32) # 一个对象
        pygame.display.set_caption('AlphaZero_Gomuku') # 设置标题

        self.areas['SwitchPlayer'] = Rect(self.ScreenSize[0]/2-self.UnitSize*1.5, self.ScreenSize[1] - self.UnitSize, self.UnitSize*3, self.UnitSize)
        # x坐标、y坐标、宽度和高度， Rect画矩形
        self.areas['RestartGame'] = Rect(self.ScreenSize[0] - self.UnitSize*3, self.ScreenSize[1] - self.UnitSize, self.UnitSize*3, self.UnitSize)
        self.areas['ResetScore'] = Rect(0, self.ScreenSize[1] - self.UnitSize, self.UnitSize*2.5, self.UnitSize)

        board_lenth = self.UnitSize * self.BoardSize
        self.areas['board'] = Rect(self.UnitSize, self.UnitSize, board_lenth, board_lenth)
        # 左上角坐标，宽和高

    def restart_game(self, button_down=True): # 重启游戏
        
        self.round_counter += 1
        self._draw_static()
        if button_down:
            self._draw_button('RestartGame', 1) # 绘制成高亮
        self.state = {}
        self.last_action_player = None
        pygame.display.update()

    def reset_score(self): # 重置得分和回合
        
        self.score = [0, 0]
        self.round_counter = 1
        self.show_messages()

    def add_score(self, winner): # 加分

        if winner == 1:
            self.score[0] += 1
        elif winner == 2:
            self.score[1] += 1
        else:
            raise ValueError('player number error')
        self.show_messages()

    def render_step(self, action, player): 
        '''
        功能: 下一步
        action: 表示执行的操作，位置
        player: 玩家编号
        '''
        try:
            action = int(action)
        except Exception:
            pass
        if type(action) != int:
            move = self.loc_2_move(action)
        else:
            move = action

        for event in pygame.event.get():
            if event.type == QUIT:
                exit()

        if self.last_action_player: # 把上一步的叉去掉
            self._draw_pieces(self.last_action_player[0], self.last_action_player[1], False)

        self._draw_pieces(action, player, True) # 在当前画叉
        self.state[move] = player
        self.last_action_player = move, player

    def move_2_loc(self, move): # 一维转二维
        return move % self.BoardSize, move // self.BoardSize

    def loc_2_move(self, loc):  # 二维转一维
        return loc[0] + loc[1] * self.BoardSize

    def get_input(self):
        '''
        获取鼠标输入，返回鼠标发生事件
        '''
        while True:
            event = pygame.event.wait()
            if event.type == QUIT:
                return 'quit',
            
            if event.type == MOUSEBUTTONDOWN:   # 鼠标按下
                if event.button == 1: # 左键
                    mouse_pos = event.pos # 鼠标位置
                    print(mouse_pos,"########")
                    for name, rec in self.areas.items(): 
                        if self._in_area(mouse_pos, rec): 
                            if name != 'board':
                                self._draw_button(name, 2, True)
                                pygame.time.delay(100)
                                self._draw_button(name, 1, True)
                                return name,
                            else:
                                x = (mouse_pos[0] - self.UnitSize)//self.UnitSize
                                y = self.BoardSize - (mouse_pos[1] - self.UnitSize)//self.UnitSize - 1
                                move = self.loc_2_move((x, y))
                                if move not in self.state:
                                    return 'move', move

            if event.type == MOUSEMOTION:       # 鼠标移动，碰到按钮高亮
                mouse_pos = event.pos
                for name, rec in self.areas.items():
                    if name != 'board':
                        if self._in_area(mouse_pos, rec):
                            self._draw_button(name, 1, True)
                        else:
                            self._draw_button(name, update=True)

    def deal_with_input(self, inp, player):
        '''
        处理输入事件
        inp: 名字
        player: 玩家编号
        '''
        if inp[0] == 'RestartGame':
            self.restart_game()
        elif inp[0] == 'ResetScore':
            self.reset_score()
        elif inp[0] == 'quit':
            exit()
        elif inp[0] == 'move':
            self.render_step(inp[1], player)
        elif inp[0] == 'SwitchPlayer':
            UI.restart_game(False)
            UI.reset_score()
           

    def show_messages(self, messages=None): # 绘制按钮上面的文本

        if messages: # AI 或者 玩家
            self.messages = messages
        pygame.draw.rect(self.screen, self._background_color, (0, self.ScreenSize[1]-self.UnitSize*2, self.ScreenSize[0], self.UnitSize)) # 画长条矩形
        self._draw_round(False)
        self._draw_text(self.messages, (self.ScreenSize[0]/2, self.ScreenSize[1]-self.UnitSize*1.5), text_height=self.TestSize)
        self._draw_score()

    def _draw_score(self, update=True):
        score = 'Score: ' + str(self.score[0]) + ' : ' + str(self.score[1])
        self._draw_text(score, (self.ScreenSize[0] * 0.11, self.ScreenSize[1] - self.UnitSize*1.5),
                        backgroud_color=self._background_color, text_height=self.TestSize)
        if update:
            pygame.display.update()

    def _draw_round(self, update=True):
        self._draw_text('Round: ' + str(self.round_counter), (self.ScreenSize[0]*0.88, self.ScreenSize[1] - self.UnitSize*1.5),
                        backgroud_color=self._background_color, text_height=self.TestSize)
        if update:
            pygame.display.update()

    def _draw_pieces(self, loc, player, last_step=False): 
        '''
        功能：在最后一步画叉
        loc:  棋子位置
        player: 玩家编号
        last_step: 是否为最后一步
        '''
        try:
            loc = int(loc)
        except Exception:
            pass

        if type(loc) is int:
            x, y = self.move_2_loc(loc)
        else:
            x, y = loc
        
        pos = int(self.UnitSize * 1.5 + x * self.UnitSize), int(self.UnitSize * 1.5 + (self.BoardSize - y - 1) * self.UnitSize)
        if player == 1: 
            c = (0, 0, 0) # 黑色
        elif player == 2:
            c = (255, 255, 255) # 白色
        else:
            raise ValueError('num input ValueError')
        
        pygame.draw.circle(self.screen, c, pos, int(self.UnitSize * 0.45))
        if last_step:
            if player == 1:
                c = (255, 255, 255)
            elif player == 2:
                c = (0, 0, 0)

            start_p1 = pos[0] - self.UnitSize * 0.3, pos[1]
            end_p1 = pos[0] + self.UnitSize * 0.3, pos[1]
            pygame.draw.line(self.screen, c, start_p1, end_p1)

            start_p2 = pos[0], pos[1] - self.UnitSize * 0.3
            end_p2 = pos[0], pos[1] + self.UnitSize * 0.3
            pygame.draw.line(self.screen, c, start_p2, end_p2)

    def _draw_static(self):  # 绘制棋盘内容
        '''
        绘制棋盘的静态内容
        '''
        self.screen.fill(self._background_color) # 背景填充颜色
        board_lenth = self.UnitSize * self.BoardSize
        pygame.draw.rect(self.screen, self._board_color, self.areas['board']) # 画大框
        for i in range(self.BoardSize):
            #
            start = self.UnitSize * (i + 0.5)
            pygame.draw.line(self.screen, (0, 0, 0), (start + self.UnitSize, self.UnitSize*1.5),
                             (start + self.UnitSize, board_lenth + self.UnitSize*0.5))
            pygame.draw.line(self.screen, (0, 0, 0), (self.UnitSize*1.5, start + self.UnitSize),
                             (board_lenth + self.UnitSize*0.5, start + self.UnitSize))
            pygame.draw.rect(self.screen, (0, 0, 0), (self.UnitSize, self.UnitSize, board_lenth, board_lenth), 1)
       
            # self._draw_text(self.BoardSize - i - 1, (self.UnitSize / 2, start + self.UnitSize), text_height=self.TestSize)  # 竖的
            # self._draw_text(i, (start + self.UnitSize, self.UnitSize / 2), text_height=self.TestSize)  # 横的

        # 绘制按钮
        for name in self.areas.keys():
            if name != 'board':
                self._draw_button(name)

        self.show_messages()

    def _draw_text(self, text, position, text_height=25, font_color=(0, 0, 0), backgroud_color=None, pos='center',
                   angle=0):  # 绘制文本
        '''
        text：内容。
        position：表示文本的位置，是一个包含两个元素的元组，表示文本的左上角的 x 和 y 坐标。
        text_height：表示文本的高度，是一个整数值，默认为 25。
        font_color：表示文本的字体颜色，是一个包含三个元素的元组，默认为黑色 (0, 0, 0)。
        background_color：表示文本的背景颜色，是一个包含三个元素的元组，表示红、绿、蓝三个颜色通道的值。默认为 None，表示不设置背景颜色，即透明背景。
        pos：表示文本在文本矩形框中的位置。可选的取值包括 'center'（居中对齐）、'top'（顶部对齐）、'bottom'（底部对齐）、'left'（左对齐）、'right'（右对齐）以及它们的组合，例如 'topleft'（左上对齐）。默认为 'center'。
        angle：表示文本的旋转角度，默认为 0，表示不旋转。
        '''
        posx, posy = position
        font_obj = pygame.font.Font(None, int(text_height)) # 字体对象
        text_surface_obj = font_obj.render(str(text), True, font_color, backgroud_color) # 文本对象
        text_surface_obj = pygame.transform.rotate(text_surface_obj, angle) # 按指定角度旋转
        text_rect_obj = text_surface_obj.get_rect() # 文本矩形框
        exec('text_rect_obj.' + pos + ' = (posx, posy)') # text_rect_obj.pos = (posx, posy)
        self.screen.blit(text_surface_obj, text_rect_obj)   # 绘制文本

    def _draw_button(self, name, high_light=0, update=False):
        rec = self.areas[name] 
        if not high_light:
            color = (225, 225, 225) # 灰色
        elif high_light == 1:
            color = (245, 245, 245) # 灰白色
        elif high_light == 2:
            color = (255, 255, 255) # 白色
        else:
            raise ValueError('高亮显示错误')
        pygame.draw.rect(self.screen, color, rec) # 画矩形，画在screen上
        pygame.draw.rect(self.screen, (0, 0, 0), rec, 1) # 1表示被填充
        self._draw_text(name, rec.center, text_height=self.TestSize)
        if update:
            pygame.display.update()

    @staticmethod
    def _in_area(loc, area):
        '''
        判断是否在区域里面
        '''
        return True if area[0] < loc[0] < area[0] + area[2] and area[1] < loc[1] < area[1] + area[3] else False


if __name__ == '__main__':
    # test
    UI = GUI()
    action = 22
    player = 1
    i = 1
    UI.add_score(1)
    while True:
        if i == 1:
            UI.show_messages('first player\'s turn')
        else:
            UI.show_messages('second player\'s turn')
        inp = UI.get_input()
        print(inp)
        UI.deal_with_input(inp, i)
        if inp[0] == 'move':
            i %= 2
            i += 1
        elif inp[0] == 'RestartGame':
            i = 1
        elif inp[0] == 'SwitchPlayer':
            i = 1
