import matplotlib.pyplot as plt

# 假设我们有以下两个列表分别代表 AI 和玩家的胜局数量
game_results = [(16, 8), (2, 1), (10, 1), (9, 1), (16, 4),(5, 1), (6, 2)]

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制柱状图
bar_width = 1
ai_wins = [result[0] for result in game_results]
player_wins = [result[1] for result in game_results]
ax.bar(range(0, len(game_results)*2, 2), ai_wins, width=bar_width, color='blue', label='AI Wins')
ax.bar(range(1, len(game_results)*2+1, 2), player_wins, width=bar_width, color='orange', label='Player Wins')

# 设置 x 轴标签和刻度
ax.set_xticks(range(0, len(game_results)*2, 2))
ax.set_xticklabels([f'Game {i+1}' for i in range(len(game_results))])
ax.set_xlabel('Game Number')
ax.set_ylabel('Wins')
ax.set_title('Game Results')

# 添加图例
ax.legend(loc='upper left')

# 调整子图之间的间距
plt.subplots_adjust(wspace=0.2)

# 显示图表
plt.show()