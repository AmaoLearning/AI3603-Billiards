"""
绘制 BayesMCTSAgent 发展历程折线图
展示与 BasicAgent 和 BasicAgentPro 的对战胜率及用时变化
"""

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# x轴标签（改进版本）
versions = [
    "残差水平角",
    "+残差速度", 
    "+噪声采样",
    "+最终检查",
    "+最差分加权",
    "-噪声采样"
]
x = np.arange(len(versions))

# 胜率数据 (None 表示无数据)
win_rate_basic = [0.800, 0.833, None, 0.950, 0.933, 0.925]
win_rate_pro = [0.358, 0.567, 0.617, 0.708, 0.750, 0.650]

# 用时数据 (小时, None 表示无数据)
time_basic = [None, None, None, 5.43, 7.07, 4.267]
time_pro = [None, 3.15, 6.1, 5.67, 5.90, 3.617]

# 创建两个子图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# ==================== 图1: 胜率 ====================
ax1.set_ylabel('胜率 (Win Rate)', fontsize=12, fontweight='bold')
ax1.set_title('BayesMCTSAgent 改进历程 - 胜率变化', fontsize=14, fontweight='bold', pad=10)

# 绘制胜率线 - vs BasicAgent
valid_idx_basic = [i for i, v in enumerate(win_rate_basic) if v is not None]
valid_vals_basic = [v for v in win_rate_basic if v is not None]
line1, = ax1.plot([x[i] for i in valid_idx_basic], valid_vals_basic, 
                   'o-', color='royalblue', linewidth=2.5, markersize=10, 
                   label='vs BasicAgent')

# 绘制胜率线 - vs BasicAgentPro
valid_idx_pro = [i for i, v in enumerate(win_rate_pro) if v is not None]
valid_vals_pro = [v for v in win_rate_pro if v is not None]
line2, = ax1.plot([x[i] for i in valid_idx_pro], valid_vals_pro, 
                   's-', color='coral', linewidth=2.5, markersize=10,
                   label='vs BasicAgentPro')

ax1.set_ylim(0.25, 1.05)
ax1.set_xticks(x)

# 在胜率点上标注数值
for i in valid_idx_basic:
    ax1.annotate(f'{win_rate_basic[i]:.1%}', (x[i], win_rate_basic[i]), 
                 textcoords="offset points", xytext=(0, 14), ha='center', 
                 fontsize=15, color='royalblue', fontweight='bold')
for i in valid_idx_pro:
    ax1.annotate(f'{win_rate_pro[i]:.1%}', (x[i], win_rate_pro[i]), 
                 textcoords="offset points", xytext=(0, -22), ha='center',
                 fontsize=15, color='coral', fontweight='bold')

ax1.legend(loc='upper left', fontsize=11)
ax1.grid(True, alpha=0.3, linestyle='--')

# ==================== 图2: 用时 ====================
ax2.set_ylabel('120局用时 (小时)', fontsize=12, fontweight='bold')
ax2.set_title('BayesMCTSAgent 改进历程 - 用时变化', fontsize=14, fontweight='bold', pad=10)

# 绘制用时线 - vs BasicAgent
valid_idx_time_basic = [i for i, v in enumerate(time_basic) if v is not None]
valid_vals_time_basic = [v for v in time_basic if v is not None]
line3, = ax2.plot([x[i] for i in valid_idx_time_basic], valid_vals_time_basic, 
                   '^-', color='royalblue', linewidth=2.5, markersize=10,
                   label='vs BasicAgent')

# 绘制用时线 - vs BasicAgentPro  
valid_idx_time_pro = [i for i, v in enumerate(time_pro) if v is not None]
valid_vals_time_pro = [v for v in time_pro if v is not None]
line4, = ax2.plot([x[i] for i in valid_idx_time_pro], valid_vals_time_pro, 
                   'v-', color='coral', linewidth=2.5, markersize=10,
                   label='vs BasicAgentPro')

ax2.set_ylim(2, 8)
ax2.set_xticks(x)
ax2.set_xticklabels(versions, rotation=15, ha='right', fontsize=15)

# 在用时点上标注数值
for i in valid_idx_time_basic:
    ax2.annotate(f'{time_basic[i]:.2f}h', (x[i], time_basic[i]), 
                 textcoords="offset points", xytext=(0, 20), ha='center',
                 fontsize=15, color='royalblue', fontweight='bold')
for i in valid_idx_time_pro:
    ax2.annotate(f'{time_pro[i]:.2f}h', (x[i], time_pro[i]), 
                 textcoords="offset points", xytext=(0, -28), ha='center',
                 fontsize=15, color='coral', fontweight='bold')

ax2.legend(loc='upper left', fontsize=11)
ax2.grid(True, alpha=0.3, linestyle='--')

# 调整布局
fig.tight_layout()

# 保存图片
plt.savefig('bayes_mcts_progress.png', dpi=150, bbox_inches='tight')
print("图表已保存为 bayes_mcts_progress.png")

# 显示图表
plt.show()
