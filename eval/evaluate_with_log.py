import logging
from pathlib import Path
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# 导入必要的模块
from poolenv import PoolEnv
from utils import set_random_seed
from agent_with_log import BasicAgent, BasicAgentPro, NewAgent

# 设置随机种子，enable=True 时使用固定种子，enable=False 时使用完全随机
# 根据需求，我们在这里统一设置随机种子，确保 agent 双方的全局击球扰动使用相同的随机状态
set_random_seed(enable=True, seed=42)

env = PoolEnv()
results = {'AGENT_A_WIN': 0, 'AGENT_B_WIN': 0, 'SAME': 0}
n_games = 120  # 对战局数 自己测试时可以修改 扩充为120局为了减少随机带来的扰动



def _safe_agent_method(agent):
    """Return agent.method() string if available, otherwise class name."""
    if hasattr(agent, "method") and callable(getattr(agent, "method")):
        try:
            return str(agent.method())
        except Exception:
            return agent.__class__.__name__
    return agent.__class__.__name__


def _build_logger(agent_a, agent_b):
    logs_dir = Path("./eval/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    filename = f"evaluate_{_safe_agent_method(agent_a)}_{_safe_agent_method(agent_b)}v4.log"
    log_path = logs_dir / filename
    logger = logging.getLogger("evaluate")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.info("Logging to %s", log_path.as_posix())
    return logger

agent_a, agent_b = BasicAgentPro(), NewAgent()
logger = _build_logger(agent_a, agent_b)

players = [agent_a, agent_b]  # 用于切换先后手
target_ball_choice = ['solid', 'solid', 'stripe', 'stripe']  # 轮换球型

for i in range(n_games): 
    logger.info("------- 第 %d 局比赛开始 -------", i)
    env.reset(target_ball=target_ball_choice[i % 4])
    logger.info("本局 Player A: %s, 目标球型: %s", players[i % 2].__class__.__name__, target_ball_choice[i % 4])
    while True:
        player = env.get_curr_player()
        logger.info("[第%d次击球] player: %s", env.hit_count, player)
        obs = env.get_observation(player)
        if player == 'A':
            action = players[i % 2].decision(*obs)
        else:
            action = players[(i + 1) % 2].decision(*obs)
        step_info = env.take_shot(action)
        
        done, info = env.get_done()
        if not done:
            if step_info.get('FOUL_FIRST_HIT'):
                logger.info("本杆判罚：首次接触对方球或黑8，直接交换球权。")
            if step_info.get('NO_POCKET_NO_RAIL'):
                logger.info("本杆判罚：无进球且母球或目标球未碰库，直接交换球权。")
            if step_info.get('NO_HIT'):
                logger.info("本杆判罚：白球未接触任何球，直接交换球权。")
            if step_info.get('ME_INTO_POCKET'):
                logger.info("我方球入袋：%s", step_info['ME_INTO_POCKET'])
            if step_info.get('ENEMY_INTO_POCKET'):
                logger.info("对方球入袋：%s", step_info['ENEMY_INTO_POCKET'])
        if done:
            # 统计结果（player A/B 转换为 agent A/B） 
            if info['winner'] == 'SAME':
                results['SAME'] += 1
            elif info['winner'] == 'A':
                results[['AGENT_A_WIN', 'AGENT_B_WIN'][i % 2]] += 1
            else:
                results[['AGENT_A_WIN', 'AGENT_B_WIN'][(i+1) % 2]] += 1
            logger.info("本局结束 winner=%s | 当前统计=%s", info['winner'], results)
            break

# 计算分数：胜1分，负0分，平局0.5
results['AGENT_A_SCORE'] = results['AGENT_A_WIN'] * 1 + results['SAME'] * 0.5
results['AGENT_B_SCORE'] = results['AGENT_B_WIN'] * 1 + results['SAME'] * 0.5

logger.info("最终结果：%s", results)
