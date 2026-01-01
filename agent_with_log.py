"""
agent.py - Agent 决策模块

定义 Agent 基类和具体实现：
- Agent: 基类，定义决策接口
- BasicAgent: 基于贝叶斯优化的参考实现
- NewAgent: 学生自定义实现模板
- analyze_shot_for_reward: 击球结果评分函数
"""

import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
import copy
import os
from datetime import datetime
import random
import logging
import signal
import functools
# from poolagent.pool import Pool as CuetipEnv, State as CuetipState
# from poolagent import FunctionAgent

from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import torch
from utils import calculate_ghost_ball_params, evaluate_state, get_pockets
from train.train_fast import AimNet
from agents import Agent

logger = logging.getLogger("evaluate")

# ============ 超时安全模拟机制 ============
class SimulationTimeoutError(Exception):
    """物理模拟超时异常"""
    pass

def _timeout_handler(signum, frame):
    """超时信号处理器"""
    raise SimulationTimeoutError("物理模拟超时")

def simulate_with_timeout(shot, timeout=3):
    """带超时保护的物理模拟
    
    参数：
        shot: pt.System 对象
        timeout: 超时时间（秒），默认3秒
    
    返回：
        bool: True 表示模拟成功，False 表示超时或失败
    
    说明：
        使用 signal.SIGALRM 实现超时机制（仅支持 Unix/Linux）
        超时后自动恢复，不会导致程序卡死
    """
    # 设置超时信号处理器
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)  # 设置超时时间
    
    try:
        pt.simulate(shot, inplace=True)
        signal.alarm(0)  # 取消超时
        return True
    except SimulationTimeoutError:
        print(f"[WARNING] 物理模拟超时（>{timeout}秒），跳过此次模拟")
        return False
    except Exception as e:
        signal.alarm(0)  # 取消超时
        raise e
    finally:
        signal.signal(signal.SIGALRM, old_handler)  # 恢复原处理器

# ============================================



def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """
    分析击球结果并计算奖励分数（完全对齐台球规则）
    
    参数：
        shot: 已完成物理模拟的 System 对象
        last_state: 击球前的球状态，{ball_id: Ball}
        player_targets: 当前玩家目标球ID，['1', '2', ...] 或 ['8']
    
    返回：
        float: 奖励分数
            +50/球（己方进球）, +100（合法黑8）, +10（合法无进球）
            -100（白球进袋）, -500（非法黑8/白球+黑8）, -30（首球/碰库犯规）
    
    规则核心：
        - 清台前：player_targets = ['1'-'7'] 或 ['9'-'15']，黑8不属于任何人
        - 清台后：player_targets = ['8']，黑8成为唯一目标球
    """
    
    # 1. 基本分析
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
    
    # 根据 player_targets 判断进球归属（黑8只有在清台后才算己方球）
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. 分析首球碰撞（定义合法的球ID集合）
    first_contact_ball_id = None
    foul_first_hit = False
    valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            # 过滤掉 'cue' 和非球对象（如 'cue stick'），只保留合法的球ID
            other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break
    
    # 首球犯规判定：完全对齐 player_targets
    if first_contact_ball_id is None:
        # 未击中任何球（但若只剩白球和黑8且已清台，则不算犯规）
        if len(last_state) > 2 or player_targets != ['8']:
            foul_first_hit = True
    else:
        # 首次击打的球必须是 player_targets 中的球
        if first_contact_ball_id not in player_targets:
            foul_first_hit = True
    
    # 3. 分析碰库
    cue_hit_cushion = False
    target_hit_cushion = False
    foul_no_rail = False
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if 'cushion' in et:
            if 'cue' in ids:
                cue_hit_cushion = True
            if first_contact_ball_id is not None and first_contact_ball_id in ids:
                target_hit_cushion = True

    if len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion):
        foul_no_rail = True
        
    # 计算奖励分数
    score = 0
    
    if cue_pocketed and eight_pocketed:
        score -= 500
    elif cue_pocketed:
        score -= 100
    elif eight_pocketed:
        is_targeting_eight_ball_legally = (len(player_targets) == 1 and player_targets[0] == "8")
        score += 150 if is_targeting_eight_ball_legally else -500
            
    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30
        
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20
    
    if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
        score = 10
        
    return score


class BasicAgent(Agent):
    """基于贝叶斯优化的智能 Agent"""
    
    def __init__(self, target_balls=None):
        """初始化 Agent
        
        参数：
            target_balls: 保留参数，暂未使用
        """
        super().__init__()
        
        # 搜索空间
        self.pbounds = {
            'V0': (0.5, 8.0),
            'phi': (0, 360),
            'theta': (0, 90), 
            'a': (-0.5, 0.5),
            'b': (-0.5, 0.5)
        }
        
        # 优化参数
        self.INITIAL_SEARCH = 20
        self.OPT_SEARCH = 10
        self.ALPHA = 1e-2
        
        # 模拟噪声（可调整以改变训练难度）
        self.noise_std = {
            'V0': 0.1,
            'phi': 0.1,
            'theta': 0.1,
            'a': 0.003,
            'b': 0.003
        }
        self.enable_noise = False
        
        print("BasicAgent (贝叶斯优化版) 已初始化。")

    
    def _create_optimizer(self, reward_function, seed):
        """创建贝叶斯优化器
        
        参数：
            reward_function: 目标函数，(V0, phi, theta, a, b) -> score
            seed: 随机种子
        
        返回：
            BayesianOptimization对象
        """
        gpr = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=self.ALPHA,
            n_restarts_optimizer=10,
            random_state=seed
        )
        
        bounds_transformer = SequentialDomainReductionTransformer(
            gamma_osc=0.8,
            gamma_pan=1.0
        )
        
        optimizer = BayesianOptimization(
            f=reward_function,
            pbounds=self.pbounds,
            random_state=seed,
            verbose=0,
            bounds_transformer=bounds_transformer
        )
        optimizer._gp = gpr
        
        return optimizer


    def decision(self, balls=None, my_targets=None, table=None):
        """使用贝叶斯优化搜索最佳击球参数
        
        参数：
            balls: 球状态字典，{ball_id: Ball}
            my_targets: 目标球ID列表，['1', '2', ...]
            table: 球桌对象
        
        返回：
            dict: 击球动作 {'V0', 'phi', 'theta', 'a', 'b'}
                失败时返回随机动作
        """
        if balls is None:
            print(f"[BasicAgent] Agent decision函数未收到balls关键信息，使用随机动作。")
            return self._random_action()
        try:
            
            # 保存一个击球前的状态快照，用于对比
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ["8"]
                logger.info("[BasicAgent] 我的目标球已全部清空，自动切换目标为：8号球")

            # 1.动态创建"奖励函数" (Wrapper)
            # 贝叶斯优化器会调用此函数，并传入参数
            def reward_fn_wrapper(V0, phi, theta, a, b):
                # 创建一个用于模拟的沙盒系统
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                cue = pt.Cue(cue_ball_id="cue")

                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
                
                try:
                    if self.enable_noise:
                        V0_noisy = V0 + np.random.normal(0, self.noise_std['V0'])
                        phi_noisy = phi + np.random.normal(0, self.noise_std['phi'])
                        theta_noisy = theta + np.random.normal(0, self.noise_std['theta'])
                        a_noisy = a + np.random.normal(0, self.noise_std['a'])
                        b_noisy = b + np.random.normal(0, self.noise_std['b'])
                        
                        V0_noisy = np.clip(V0_noisy, 0.5, 8.0)
                        phi_noisy = phi_noisy % 360
                        theta_noisy = np.clip(theta_noisy, 0, 90)
                        a_noisy = np.clip(a_noisy, -0.5, 0.5)
                        b_noisy = np.clip(b_noisy, -0.5, 0.5)
                        
                        shot.cue.set_state(V0=V0_noisy, phi=phi_noisy, theta=theta_noisy, a=a_noisy, b=b_noisy)
                    else:
                        shot.cue.set_state(V0=V0, phi=phi, theta=theta, a=a, b=b)
                    
                    # 关键：使用带超时保护的物理模拟（3秒上限）
                    if not simulate_with_timeout(shot, timeout=3):
                        return 0  # 超时是物理引擎问题，不惩罚agent
                except Exception as e:
                    # 模拟失败，给予极大惩罚
                    return -500
                
                # 使用我们的"裁判"来打分
                score = analyze_shot_for_reward(
                    shot=shot,
                    last_state=last_state_snapshot,
                    player_targets=my_targets
                )


                return score

            logger.info(f"[BasicAgent] 正在为 Player (targets: {my_targets}) 搜索最佳击球...")
            
            seed = np.random.randint(1e6)
            optimizer = self._create_optimizer(reward_fn_wrapper, seed)
            optimizer.maximize(
                init_points=self.INITIAL_SEARCH,
                n_iter=self.OPT_SEARCH
            )
            
            best_result = optimizer.max
            best_params = best_result['params']
            best_score = best_result['target']

            if best_score < 10:
                logger.info(f"[BasicAgent] 未找到好的方案 (最高分: {best_score:.2f})。使用随机动作。")
                return self._random_action()
            action = {
                'V0': float(best_params['V0']),
                'phi': float(best_params['phi']),
                'theta': float(best_params['theta']),
                'a': float(best_params['a']),
                'b': float(best_params['b']),
            }

            logger.info(f"[BasicAgent] 决策 (得分: {best_score:.2f}): "
                  f"V0={action['V0']:.2f}, phi={action['phi']:.2f}, "
                  f"θ={action['theta']:.2f}, a={action['a']:.3f}, b={action['b']:.3f}")
            return action

        except Exception as e:
            logger.info(f"[BasicAgent] 决策时发生严重错误，使用随机动作。原因: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()

# ============ BasicAgentPro: 基于MCTS的进阶 Agent ============
class BasicAgentPro(Agent):
    """基于MCTS（蒙特卡洛树搜索）的进阶 Agent"""
    
    def __init__(self,
                 n_simulations=50,       # 仿真次数
                 c_puct=1.414):          # 探索系数
        super().__init__()
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.ball_radius = 0.028575
        
        # 定义噪声水平 (与 poolenv 保持一致或略大)
        self.sim_noise = {
            'V0': 0.1, 'phi': 0.15, 'theta': 0.1, 'a': 0.005, 'b': 0.005
        }
        
        logger.info("BasicAgentPro (MCTS版) 已初始化。")

    def _calc_angle_degrees(self, v):
        angle = math.degrees(math.atan2(v[1], v[0]))
        return angle % 360

    def _get_ghost_ball_target(self, cue_pos, obj_pos, pocket_pos):
        vec_obj_to_pocket = np.array(pocket_pos) - np.array(obj_pos)
        dist_obj_to_pocket = np.linalg.norm(vec_obj_to_pocket)
        if dist_obj_to_pocket == 0: return 0, 0
        unit_vec = vec_obj_to_pocket / dist_obj_to_pocket
        ghost_pos = np.array(obj_pos) - unit_vec * (2 * self.ball_radius)
        vec_cue_to_ghost = ghost_pos - np.array(cue_pos)
        dist_cue_to_ghost = np.linalg.norm(vec_cue_to_ghost)
        phi = self._calc_angle_degrees(vec_cue_to_ghost)
        return phi, dist_cue_to_ghost

    def generate_heuristic_actions(self, balls, my_targets, table):
        """
        生成候选动作列表
        """
        actions = []
        
        cue_ball = balls.get('cue')
        if not cue_ball: return [self._random_action()]
        cue_pos = cue_ball.state.rvw[0]

        # 获取所有目标球的ID
        target_ids = [bid for bid in my_targets if balls[bid].state.s != 4]
        
        # 如果没有目标球了（理论上外部会处理转为8号，这里兜底）
        if not target_ids:
            target_ids = ['8']
        
        logger.info("[BasicAgentPro] 正在为 Player (targets: %s) 搜索最佳击球...", target_ids)

        # 遍历每一个目标球
        for tid in target_ids:
            obj_ball = balls[tid]
            obj_pos = obj_ball.state.rvw[0]

            # 遍历每一个袋口
            for pocket_id, pocket in table.pockets.items():
                pocket_pos = pocket.center

                # 1. 计算理论进球角度
                phi_ideal, dist = self._get_ghost_ball_target(cue_pos, obj_pos, pocket_pos)

                # 2. 根据距离简单的估算力度 (距离越远力度越大，基础力度2.0)
                v_base = 1.5 + dist * 1.5
                v_base = np.clip(v_base, 1.0, 6.0)

                # 3. 生成几个变种动作加入候选池
                # 变种1：精准一击
                actions.append({
                    'V0': v_base, 'phi': phi_ideal, 'theta': 0, 'a': 0, 'b': 0
                })
                # 变种2：力度稍大
                actions.append({
                    'V0': min(v_base + 1.5, 7.5), 'phi': phi_ideal, 'theta': 0, 'a': 0, 'b': 0
                })
                # 变种3：角度微调 (左右偏移 0.5 度，应对噪声)
                actions.append({
                    'V0': v_base, 'phi': (phi_ideal + 0.5) % 360, 'theta': 0, 'a': 0, 'b': 0
                })
                actions.append({
                    'V0': v_base, 'phi': (phi_ideal - 0.5) % 360, 'theta': 0, 'a': 0, 'b': 0
                })

        # 如果通过启发式没有生成任何动作（极罕见），补充随机动作
        if len(actions) == 0:
            for _ in range(5):
                actions.append(self._random_action())
        
        # 随机打乱顺序
        random.shuffle(actions)
        return actions[:30]

    def simulate_action(self, balls, table, action):
        """
        [修改点1] 执行带噪声的物理仿真
        让 Agent 意识到由于误差的存在，某些"极限球"是不可打的
        """
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
        
        try:
            # --- 注入高斯噪声 ---
            noisy_V0 = np.clip(action['V0'] + np.random.normal(0, self.sim_noise['V0']), 0.5, 8.0)
            noisy_phi = (action['phi'] + np.random.normal(0, self.sim_noise['phi'])) % 360
            noisy_theta = np.clip(action['theta'] + np.random.normal(0, self.sim_noise['theta']), 0, 90)
            noisy_a = np.clip(action['a'] + np.random.normal(0, self.sim_noise['a']), -0.5, 0.5)
            noisy_b = np.clip(action['b'] + np.random.normal(0, self.sim_noise['b']), -0.5, 0.5)

            cue.set_state(V0=noisy_V0, phi=noisy_phi, theta=noisy_theta, a=noisy_a, b=noisy_b)
            pt.simulate(shot, inplace=True)
            return shot
        except Exception:
            return None

    def decision(self, balls=None, my_targets=None, table=None):
        if balls is None: return self._random_action()
        
        # 预处理
        remaining = [bid for bid in my_targets if balls[bid].state.s != 4]
        if len(remaining) == 0: my_targets = ["8"]
        last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

        # 生成候选动作
        candidate_actions = self.generate_heuristic_actions(balls, my_targets, table)
        n_candidates = len(candidate_actions)
        
        N = np.zeros(n_candidates)
        Q = np.zeros(n_candidates)
        
        # MCTS 循环
        for i in range(self.n_simulations):
            # Selection (UCB)
            if i < n_candidates:
                idx = i
            else:
                total_n = np.sum(N)
                # 使用归一化后的 Q 进行计算
                ucb_values = (Q / (N + 1e-6)) + self.c_puct * np.sqrt(np.log(total_n + 1) / (N + 1e-6))
                idx = np.argmax(ucb_values)
            
            # Simulation (带噪声)
            shot = self.simulate_action(balls, table, candidate_actions[idx])

            # Evaluation
            if shot is None:
                raw_reward = -500.0
            else:
                raw_reward = analyze_shot_for_reward(shot, last_state_snapshot, my_targets)
            
            # 映射公式: (val - min) / (max - min)
            normalized_reward = (raw_reward - (-500)) / 650.0
            # 截断一下防止越界
            normalized_reward = np.clip(normalized_reward, 0.0, 1.0)

            # Backpropagation
            N[idx] += 1
            Q[idx] += normalized_reward # 累加归一化后的分数

        # Final Decision
        # 选平均分最高的 (Robust Child)
        avg_rewards = Q / (N + 1e-6)
        best_idx = np.argmax(avg_rewards)
        best_action = candidate_actions[best_idx]
        
        # 简单打印一下当前最好的预测胜率
        logger.info(f"[BasicAgentPro] Best Avg Score: {avg_rewards[best_idx]:.3f} (Sims: {self.n_simulations})")
        
        return best_action

class NewAgent(Agent):
    """自定义 Agent 模板（待学生实现）"""
    
    def __init__(self):
        model_path = os.path.join('train', 'checkpoints', 'aim_model.pth')
        #self.agent = HybridLearningAgent(model_path)
        # self.agent = MCTSAgent()
        self.agent = BayesMCTSAgent(True)
    
    def decision(self, balls=None, my_targets=None, table=None):
        """决策方法
        
        参数：
            observation: (balls, my_targets, table)
        
        返回：
            dict: {'V0', 'phi', 'theta', 'a', 'b'}
        """
        return self.agent.decision(balls=balls, my_targets=my_targets, table=table)
    
    def method(self):
        """
        实际所用方法的名称
        """
        return self.agent.__class__.__name__

class MCTSAgent(Agent):
    """
    MCTS-Lite Agent
    特点：
    1. 几何求解 + 微调模拟
    2. 在评估决策时多看一步
    3. 没有学习过程, 比较省时

    成绩: 29.0/40.0, 0.725
          换用analyze_shot_for_reward后为 32.0/40.0, 0.800
          走位评分版本1 25.0/40.0, 0.625
          完全不截断版本 22.0/40.0, 0.550
          角度放宽到90/candidates放多到6个版本 20.0/40.0, 0.500
    """
    def __init__(self):
        super().__init__()
        logger.info("ImprovedMCTSAgent 已初始化 - 包含防守逻辑与微调瞄准")

    def _generate_safety_shot(self, balls, candidates):
        """
        防守策略：当没有好机会时，轻轻碰一下离得最近的球，避免犯规
        """
        logger.info("[MCTSAgent] 启动防守模式 (Safety Mode)")
        cue_pos = balls['cue'].state.rvw[0]
        min_dist = float('inf')
        best_target = None
        
        # 找最近的自己的球
        for bid in candidates:
            obj_pos = balls[bid].state.rvw[0]
            dist = np.linalg.norm(np.array(obj_pos[:2]) - np.array(cue_pos[:2]))
            if dist < min_dist:
                min_dist = dist
                best_target = bid
        
        if best_target:
            obj_pos = balls[best_target].state.rvw[0]
            dx = obj_pos[0] - cue_pos[0]
            dy = obj_pos[1] - cue_pos[1]
            phi = np.degrees(np.arctan2(dy, dx)) % 360
            # 极轻的力度，只要碰到就行
            logger.info(f'[MCTSAgent] 轻轻触碰己方球 {best_target}')
            return {'V0': 0.8, 'phi': phi, 'theta': 0, 'a': 0, 'b': 0}
        
        return self._random_action()

    def decision(self, balls=None, my_targets=None, table=None):
        if balls is None:
            return self._random_action()
        
        cue_ball = balls['cue']
        cue_pos = cue_ball.state.rvw[0]
        R = cue_ball.params.R

        last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        # --- 1. 生成几何候选 (Candidates) ---
        candidates = [] 
        remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
        if not remaining_own: remaining_own = ['8']

        logger.info("[MCTSAgent] 正在为 Player (targets: %s) 搜索最佳击球...", remaining_own)

        for ball_id in remaining_own:
            obj_pos = balls[ball_id].state.rvw[0]
            for pid, pocket in table.pockets.items():
                phi_ideal, cut_angle, dist = calculate_ghost_ball_params(cue_pos, obj_pos, pocket.center, R)
                
                # 只有非常难打的球才会被过滤 (阈值 85度)
                if abs(cut_angle) > 85: continue
                
                candidates.append({
                    'target_id': ball_id,
                    'phi_center': phi_ideal,
                    'cut_angle': cut_angle,
                    'distance': dist
                })

        # 如果真的没有进攻机会，执行防守
        if not candidates:
            logger.info("[MCTSAgent] 无几何进攻线路，尝试防守。")
            return self._generate_safety_shot(balls, remaining_own)

        # 排序：优先考虑切角小、距离近的球
        candidates.sort(key=lambda x: x['cut_angle'] + x['distance']*10)
        top_candidates = candidates[:4] # 只看前4个最好的选择

        best_action = None
        best_score = -float('inf')
        best_tag = ""

        # --- 2. 模拟与微调 (Simulation) ---
        logger.info("[MCTSAgent] 评估 %d 个进攻线路...", len(top_candidates))
        
        for cand in top_candidates: # 到只剩8时经常只有1个cand，这说明白球位置不好
            # 微调逻辑：不仅尝试理论角度，还要尝试左右偏差
            # 物理引擎中，球的碰撞会有偏差，必须通过微调来修正
            phi_offsets = [0, -0.5, 0.5, -1.0, 1.0] 
            speeds = [0.8, 2.0, 4.0, 6.5] # 轻推、慢、中、快
            
            for V0 in speeds:
                for offset in phi_offsets:
                    phi_try = cand['phi_center'] + offset
                    
                    # 构建模拟环境
                    sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                    sim_table = copy.deepcopy(table)
                    cue = pt.Cue(cue_ball_id="cue")
                    cue.set_state(V0=V0, phi=phi_try, theta=0, a=0, b=0)
                    shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
                    
                    try:
                        pt.simulate(shot, inplace=True)
                        score = evaluate_state(shot, last_state_snapshot, my_targets)
                        
                        if score > best_score:
                            best_score = score
                            best_action = {'V0': V0, 'phi': phi_try, 'theta': 0, 'a': 0, 'b': 0}
                            # 如果找到了必进球且走位不错的解，可以提前剪枝，也相当于倾向于速度更小的动作
                            if score >= 60: 
                                logger.info("[MCTSAgent] 找到绝佳线路！决策: V0=%.1f, phi=%.1f (ExpScore:%.1f)",
                                                best_action['V0'],
                                                best_action['phi'],
                                                best_score,)
                                return best_action
                                
                    except Exception as e:
                        # 打印错误但不中断程序
                        logger.error("[MCTSAgent ERROR] Sim failed: %s", e)
                        continue

        if (best_action is None) or (best_score < 0): # 更激进地采取防守策略
            logger.info("[MCTSAgent] 模拟后未发现可行进攻方案，转为防守。")
            return self._generate_safety_shot(balls, remaining_own)
            
        logger.info(
            "[MCTSAgent] 决策: V0=%.1f, phi=%.1f (ExpScore:%.1f)",
            best_action['V0'],
            best_action['phi'],
            best_score,
        )
        return best_action

    def _random_action(self):
        return {'V0': 1.0, 'phi': np.random.uniform(0,360), 'theta':0, 'a':0, 'b':0}

class BankAgent(Agent):
    """
    BankAgent (MCTS + Bank Shots)
    特点：
    1. 继承了 MCTSAgent 的几何求解 + 微调模拟能力
    2. 新增：翻袋 (Bank Shot) 路径规划能力
    3. 能够识别直打困难的局面，自动寻找撞库解法
    """
    def __init__(self):
        super().__init__()
        logger.info("BankAgent 已初始化 - 具备翻袋攻击能力")

    def _get_table_rails(self, table):
        """获取4个库边的坐标位置"""
        # table.w 是宽 (y轴方向), table.l 是长 (x轴方向)
        # 假设球桌中心在 (0,0)
        right = table.l / 2
        left = -table.l / 2
        top = table.w / 2
        bottom = -table.w / 2
        return {'left': left, 'right': right, 'top': top, 'bottom': bottom}

    def _get_virtual_pocket(self, pocket_pos, rail_name, rails):
        """计算虚拟袋口坐标 (镜像点)"""
        px, py = pocket_pos[0], pocket_pos[1]
        
        if rail_name == 'top':   # y = top, 镜像点 y' = 2*top - y
            return np.array([px, 2 * rails['top'] - py, 0])
        elif rail_name == 'bottom':
            return np.array([px, 2 * rails['bottom'] - py, 0])
        elif rail_name == 'left': # x = left, 镜像点 x' = 2*left - x
            return np.array([2 * rails['left'] - px, py, 0])
        elif rail_name == 'right':
            return np.array([2 * rails['right'] - px, py, 0])
        return None

    def _generate_safety_shot(self, balls, my_targets):
        """防守策略"""
        logger.info("[BankAgent] 启动防守模式 (Safety Mode)")
        cue_pos = balls['cue'].state.rvw[0]
        min_dist = float('inf')
        best_target = None
        
        candidates = [b for b in my_targets if balls[b].state.s != 4]
        if not candidates: candidates = ['8']
        
        for bid in candidates:
            obj_pos = balls[bid].state.rvw[0]
            dist = np.linalg.norm(np.array(obj_pos[:2]) - np.array(cue_pos[:2]))
            if dist < min_dist:
                min_dist = dist
                best_target = bid
        
        if best_target:
            obj_pos = balls[best_target].state.rvw[0]
            dx = obj_pos[0] - cue_pos[0]
            dy = obj_pos[1] - cue_pos[1]
            phi = np.degrees(np.arctan2(dy, dx)) % 360
            return {'V0': 0.8, 'phi': phi, 'theta': 0, 'a': 0, 'b': 0}
        
        return self._random_action()

    def _generate_bank_candidates(self, balls, my_targets, table, cue_pos, R):
        """生成翻袋击球候选"""
        candidates = []
        rails = self._get_table_rails(table)
        
        remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
        if not remaining_own: remaining_own = ['8']

        for ball_id in remaining_own:
            obj_pos = balls[ball_id].state.rvw[0]
            
            # 遍历4条库边
            for rail_name in ['top', 'bottom', 'left', 'right']:
                for pid, pocket in table.pockets.items():
                    # 计算虚拟袋口
                    virtual_pos = self._get_virtual_pocket(pocket.center, rail_name, rails)
                    
                    # 几何计算：把虚拟袋口当做目标
                    phi_ideal, cut_angle, dist_cue_obj = calculate_ghost_ball_params(
                        cue_pos, obj_pos, virtual_pos, R
                    )
                    
                    # 翻袋难度较大，切角阈值设低一点 (如 60度)
                    if abs(cut_angle) > 60: continue
                    
                    # 估算总距离：母球->目标球 + 目标球->虚拟袋口
                    dist_obj_virtual = np.linalg.norm(virtual_pos - obj_pos)
                    total_dist = dist_cue_obj + dist_obj_virtual
                    
                    candidates.append({
                        'type': 'bank', # 标记类型
                        'target_id': ball_id,
                        'phi_center': phi_ideal,
                        'cut_angle': cut_angle,
                        'distance': total_dist,
                        'rail': rail_name,
                        'pocket_id': pid
                    })
        return candidates

    def decision(self, balls=None, my_targets=None, table=None):
        if balls is None: return self._random_action()
        
        cue_ball = balls['cue']
        cue_pos = cue_ball.state.rvw[0]
        R = cue_ball.params.R

        # --- 1. 生成直打候选 (Direct) ---
        candidates = [] 
        remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
        if not remaining_own: remaining_own = ['8']

        for ball_id in remaining_own:
            obj_pos = balls[ball_id].state.rvw[0]
            for pid, pocket in table.pockets.items():
                phi_ideal, cut_angle, dist = calculate_ghost_ball_params(cue_pos, obj_pos, pocket.center, R)
                
                if abs(cut_angle) > 85: continue
                
                candidates.append({
                    'type': 'direct',
                    'target_id': ball_id,
                    'phi_center': phi_ideal,
                    'cut_angle': cut_angle,
                    'distance': dist
                })

        # --- 2. 生成翻袋候选 (Bank) ---
        bank_candidates = self._generate_bank_candidates(balls, my_targets, table, cue_pos, R)
        candidates.extend(bank_candidates)

        if not candidates:
            logger.info("[BankAgent] 无几何进攻线路，尝试防守。")
            return self._generate_safety_shot(balls, my_targets)

        # --- 3. 混合排序 ---
        # 略微降低翻袋惩罚，鼓励在直打困难时尝试翻袋
        def sort_key(c):
            penalty = 0 if c['type'] == 'direct' else 25
            return c['cut_angle'] + c['distance']*10 + penalty

        candidates.sort(key=sort_key)
        top_candidates = candidates[:5]  # 只关注前5个机会

        logger.info(
            "[BankAgent] 评估 %d 个线路 (含 %d 个翻袋)...",
            len(top_candidates),
            sum(1 for c in top_candidates if c['type']=='bank'),
        )

        best_action = None
        best_score = 0  # 只接受得分>0的方案
        best_tag = ""

        # --- 4. 模拟与高精度微调 ---
        for cand in top_candidates:
            # 高密度角度微调：直打[-1.5,1.5]分21份；翻袋[-2.5,2.5]分31份
            phi_offsets = np.linspace(-1.5, 1.5, 21)
            
            if cand['type'] == 'bank':
                phi_offsets = np.linspace(-2.5, 2.5, 31)
                base_speeds = [4.0, 6.0, 8.0]
            else:
                base_speeds = [2.5, 4.5, 6.5]
            
            for V0 in base_speeds:
                # 如果已有较高得分，当前线路不再尝试更多力度以节省时间
                if best_score > 80:
                    break

                for offset in phi_offsets:
                    phi_try = cand['phi_center'] + offset
                    
                    sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                    sim_table = copy.deepcopy(table)
                    cue = pt.Cue(cue_ball_id="cue")
                    cue.set_state(V0=V0, phi=phi_try, theta=0, a=0, b=0)
                    shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
                    
                    try:
                        pt.simulate(shot, inplace=True)
                        score = evaluate_state(shot, my_targets, cand['target_id'])
                        
                        if score > best_score:
                            best_score = score
                            best_action = {'V0': V0, 'phi': phi_try, 'theta': 0, 'a': 0, 'b': 0}
                            best_tag = "[翻袋]" if cand['type'] == 'bank' else "[直打]"

                            if score > 120:
                                logger.info("[BankAgent] 锁定%s绝佳线路！Score: %.1f", best_tag, score)
                                return best_action
                    except Exception:
                        continue

        if best_action is None:
            logger.info("[BankAgent] 模拟显示无进球机会 (BestScore: %.1f)，智能转为防守。", best_score)
            return self._generate_safety_shot(balls, my_targets)
            
        logger.info(
            "[BankAgent] 决策: V0=%.1f, phi=%.1f, score=%.1f %s",
            best_action['V0'],
            best_action['phi'],
            best_score,
            best_tag,
        )
        return best_action

    def _random_action(self):
        return {'V0': 1.0, 'phi': np.random.uniform(0,360), 'theta':0, 'a':0, 'b':0}

class HybridLearningAgent(Agent):
    """
    混合智能体：神经网络引导 + 局部搜索验证
    策略：
    1. 使用神经网络预测偏差，大幅缩小搜索范围。
    2. 在预测值附近进行微小范围的模拟验证（解决物理噪声）。
    3. 如果进攻模拟全部失败，严格执行防守（解决乱打问题）。
   
    成绩：换用analyze_shot_for_reward后 23.0/40.0 0.575
    """
    def __init__(self, model_path='checkpoints/aim_model.pth'):
        super().__init__()
        self.model = AimNet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # 加载模型
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"[HybridAgent] 神经网络加载成功: {model_path}")
            self.use_nn = True
        except Exception as e:
            logger.info(f"[HybridAgent] 模型加载失败 ({e})，回退到纯几何搜索模式。")
            self.use_nn = False

    def _predict_correction(self, cut_angle, distance, V0):
        """调用神经网络预测修正量"""
        if not self.use_nn: return 0.0

        c_norm = cut_angle / 90.0
        d_norm = distance / 2.05
        v_norm = V0 / 8.0
        
        inputs = torch.tensor([[c_norm, d_norm, v_norm]], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            delta = self.model(inputs).item()
        return delta

    def _generate_safety_shot(self, balls, my_targets):
        """防守策略：必须碰库"""
        logger.info("[HybridAgent] 进攻不可行，切换强力防守。")
        cue_pos = balls['cue'].state.rvw[0]
        min_dist = float('inf')
        best_target = None
        
        candidates = [b for b in my_targets if balls[b].state.s != 4]
        if not candidates: candidates = ['8']
        
        # 找最近的球
        for bid in candidates:
            obj_pos = balls[bid].state.rvw[0]
            dist = np.linalg.norm(np.array(obj_pos[:2]) - np.array(cue_pos[:2]))
            if dist < min_dist:
                min_dist = dist
                best_target = bid
        
        if best_target:
            obj_pos = balls[best_target].state.rvw[0]
            dx = obj_pos[0] - cue_pos[0]
            dy = obj_pos[1] - cue_pos[1]
            phi = np.degrees(np.arctan2(dy, dx)) % 360
            # 力度 3.0，确保碰库不犯规
            return {'V0': 3.0, 'phi': phi, 'theta': 0, 'a': 0, 'b': 0}
        
        # 实在没办法，随机打
        return {'V0': 1.0, 'phi': 0, 'theta':0, 'a':0, 'b':0}

    def decision(self, balls=None, my_targets=None, table=None):
        if balls is None: return self._generate_safety_shot(balls, my_targets)
        
        cue_ball = balls['cue']
        cue_pos = cue_ball.state.rvw[0]
        R = cue_ball.params.R

        
        last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        # --- 1. 筛选候选球 ---
        candidates = []
        remaining = [bid for bid in my_targets if balls[bid].state.s != 4]
        if not remaining: remaining = ['8']
        
        for ball_id in remaining:
            obj_pos = balls[ball_id].state.rvw[0]
            for pid, pocket in table.pockets.items():
                phi_geo, cut_angle, dist = calculate_ghost_ball_params(cue_pos, obj_pos, pocket.center, R)
                # 严格过滤 > 85 度的球
                if abs(cut_angle) > 85: continue
                
                candidates.append({
                    'target_id': ball_id,
                    'phi_geo': phi_geo,
                    'cut_angle': cut_angle,
                    'distance': dist
                })
        
        # 排序：优先切角小、距离近
        candidates.sort(key=lambda x: x['cut_angle'] + x['distance']*10)
        top_candidates = candidates[:4] # 只看最好的4个

        best_action = None
        best_score = 0 # 【关键修复】: 初始分数设为0，任何负分(没进)都不会被选中

        # --- 2. 混合搜索 (Neural Guide + Local Search) ---
        for cand in top_candidates:
            # 尝试几种力度
            speeds = [3.0, 5.0, 7.0]
            
            for V0 in speeds:
                # Step A: 神经网络预测偏差
                delta_pred = self._predict_correction(cand['cut_angle'], cand['distance'], V0)
                phi_center = cand['phi_geo'] + delta_pred
                
                # Step B: 局部微调 (Local Grid Search)
                # 既然网络可能有误差，我们在预测值左右 0.5度 内再扫 5 个点
                # 这样既利用了网络的先验，又解决了物理噪声
                offsets = np.linspace(-0.5, 0.5, 5) 
                
                for off in offsets:
                    phi_try = phi_center + off
                    
                    # 模拟
                    sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                    sim_table = copy.deepcopy(table)
                    cue = pt.Cue(cue_ball_id="cue")
                    cue.set_state(V0=V0, phi=phi_try, theta=0, a=0, b=0)
                    shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
                    
                    try:
                        pt.simulate(shot, inplace=True)
                        score = evaluate_state(shot, last_state_snapshot, my_targets)
                        
                        # 如果是好结果
                        if score > best_score:
                            best_score = score
                            best_action = {'V0': V0, 'phi': phi_try, 'theta': 0, 'a': 0, 'b': 0}
                            
                            # 提前剪枝：如果已经稳进球了，不用再搜了
                            if score > 80: 
                                logger.info(f"[Hybrid] 🎯 命中! NN偏:{delta_pred:.2f} | 微调:{off:.2f} | Score:{score:.1f}")
                                return best_action
                                
                    except: continue

        # --- 3. 决策 ---
        if best_action is not None and best_score > 0:
            logger.info(f"[Hybrid] 决策: V0={best_action['V0']:.1f}, phi={best_action['phi']:.1f} (ExpScore:{best_score:.1f})")
            return best_action
        
        # 如果模拟了一圈，分数全是 -50 或 -1000，说明根本进不去
        # 【关键修复】: 绝对不选那些 -50 的动作，转为防守
        logger.info(f"[Hybrid] 进攻模拟全失败 (BestScore: {best_score})，转为防守。")
        return self._generate_safety_shot(balls, my_targets)

class BayesMCTSAgent(Agent):
    """基于贝叶斯优化的智能MCTS Agent"""
    
    def __init__(self, enable_noise=False):
        """初始化 Agent
        
        参数：
            target_balls: 保留参数，暂未使用
        
        成绩：
            单打8号球: 31.0/40.0
            analyze_shot_for_reward: 32.0/40.0
            evaluate_state: 28.0/40.0
            v1 vs BasicPro: 43.0/120.0
            v2 vs Basic: 100.0/120.0
            v2 with Noise vs Basic: 106.0/120.0
        """
        super().__init__()
        
        # 搜索空间 - 扩大角度搜索范围以适应切球
        self.pbounds = {            'd_V0': (-2.0, 2.0),
            'd_phi': (-3.0, 3.0),  # 从 ±0.5 扩大到 ±3.0，关键改进！
            'theta': (0, 30),      # 限制跳球角度，减少无效搜索
            'a': (-0.2, 0.2),      # 缩小塞球范围，减少复杂度
            'b': (-0.2, 0.2)
        }
        
        # 优化参数 - 增加初始探索
        self.INITIAL_SEARCH = 10
        self.OPT_SEARCH = 5
        self.NOISE_SAMPLES = 3  # 多次采样取平均
        self.EARLY_STOP_SCORE = 1000
        self.ALPHA = 1e-2
        
        # 模拟噪声（与 BasicAgentPro 保持一致）
        self.noise_std = {
            'V0': 0.1,
            'phi': 0.15,
            'theta': 0.1,
            'a': 0.005,
            'b': 0.005
        }
        self.enable_noise = enable_noise
        
        logger.info("[BayesMCTS] (Enhanced v2) 已初始化。")
    
    def _create_optimizer(self, reward_function, seed):
        """创建贝叶斯优化器
        
        参数：
            reward_function: 目标函数，(V0, phi, theta, a, b) -> score
            seed: 随机种子
        
        返回：
            BayesianOptimization对象
        """
        gpr = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=self.ALPHA,
            n_restarts_optimizer=10,
            random_state=seed
        )
        
        bounds_transformer = SequentialDomainReductionTransformer(
            gamma_osc=0.8,
            gamma_pan=1.0
        )
        
        optimizer = BayesianOptimization(
            f=reward_function,
            pbounds=self.pbounds,
            random_state=seed,
            verbose=0,
            bounds_transformer=bounds_transformer
        )
        optimizer._gp = gpr
        
        return optimizer
    
    def _evaluate_action(self, d_V0, d_phi, theta, a, b, base_phi, base_v, balls, table, my_targets, last_state_snapshot):
        """
        带多次噪声采样的动作评估 (核心改进)
        
        改进点：
        1. 多次采样取平均，提高稳健性（与MCTS思想对齐）
        2. 使用 analyze_shot_for_reward（与 BasicAgentPro 对齐）
        """
        # 1. 还原绝对参数
        phi_base = (base_phi + d_phi) % 360
        V0_base = np.clip(base_v + d_V0, 0.8, 7.5)
        
        # 2. 多次噪声采样
        n_samples = self.NOISE_SAMPLES if self.enable_noise else 1
        scores = []
        
        for _ in range(n_samples):
            # 注入噪声（如果启用）
            if self.enable_noise:
                V0 = np.clip(V0_base + np.random.normal(0, self.noise_std['V0']), 0.5, 8.0)
                phi = (phi_base + np.random.normal(0, self.noise_std['phi'])) % 360
                theta_n = np.clip(theta + np.random.normal(0, self.noise_std['theta']), 0, 90)
                a_n = np.clip(a + np.random.normal(0, self.noise_std['a']), -0.5, 0.5)
                b_n = np.clip(b + np.random.normal(0, self.noise_std['b']), -0.5, 0.5)
            else:
                V0, phi, theta_n, a_n, b_n = V0_base, phi_base, theta, a, b
            
            # 构建模拟环境
            sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            sim_table = copy.deepcopy(table)
            cue = pt.Cue(cue_ball_id="cue")
            sim_shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
            sim_shot.cue.set_state(V0=V0, phi=phi, theta=theta_n, a=a_n, b=b_n)
            
            try:
                pt.simulate(sim_shot, inplace=True)
                # 使用与 BasicAgentPro 相同的评估函数！
                score = evaluate_state(sim_shot, last_state_snapshot, my_targets)
            except:
                score = -500.0
            
            scores.append(score)
        
        # 返回平均分（更稳健）
        return np.mean(scores)

    def decision(self, balls=None, my_targets=None, table=None):
        """使用贝叶斯优化搜索最佳击球参数
        
        参数：
            balls: 球状态字典，{ball_id: Ball}
            my_targets: 目标球ID列表，['1', '2', ...]
            table: 球桌对象
        
        返回：
            dict: 击球动作 {'V0', 'phi', 'theta', 'a', 'b'}
                失败时返回随机动作
        """
        if balls is None:
            logger.info("[BayesMCTS] Agent decision函数未收到balls关键信息，使用随机动作。")
            return self._random_action()
        try:
            
            # 保存一个击球前的状态快照，用于对比
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                remaining_own = ["8"]
                logger.info("[BayesMCTS] 我的目标球已全部清空，自动切换目标为：8号球")
            
            cue_ball = balls['cue']
            cue_pos = cue_ball.state.rvw[0]
            R = cue_ball.params.R

            # ========== 开球特殊处理 ==========
            # 判断是否为开球局面：所有目标球都在台上（7个）
            is_break_shot = balls['1'].state.t == 0
            
            if is_break_shot:
                # 开球策略：直接大力冲击球堆，不做复杂搜索
                # 标准开球角度：瞄准1号球（球堆顶端）
                one_pos = balls['1'].state.rvw[0]
                dx = one_pos[0] - cue_pos[0]
                dy = one_pos[1] - cue_pos[1]
                phi_break = np.degrees(np.arctan2(dy, dx)) % 360
                
                logger.info("[BayesMCTS] 检测到开球局面，使用快速开球策略 (phi=%.1f)", phi_break)
                return {
                    'V0': 7.0,  # 大力开球
                    'phi': phi_break,
                    'theta': 0,
                    'a': 0,
                    'b': 0
                }
            # ====================================
            
            candidates = []

            logger.info("[BayesMCTS] 正在为 Player (targets: %s) 搜索最佳击球...", remaining_own)
            for ball_id in remaining_own:
                obj_pos = balls[ball_id].state.rvw[0]
                for pid, pocket in table.pockets.items():
                    phi_ideal, cut_angle, dist = calculate_ghost_ball_params(cue_pos, obj_pos, pocket.center, R)
                    
                    # 放宽阈值到 80 度（与其他 Agent 一致）
                    if abs(cut_angle) > 80: continue
                    
                    candidates.append({
                        'target_id': ball_id,
                        'phi_center': phi_ideal,
                        'cut_angle': cut_angle,
                        'distance': dist
                    })
            
            # 改进排序：降低距离权重，优先考虑切角
            # 原公式 cut_angle + 10*dist 对远台惩罚过重
            candidates.sort(key=lambda x: x['cut_angle'] * 1.5 + x['distance'] * 5)
            
            # 只取前 3 个候选，平衡速度与质量
            top_candidates = candidates[:3]

            top_action = None
            top_score = -float('inf')
            top_base_phi = 0
            top_base_v = 0
            
            # 早停标志
            found_good_shot = False

            for cand in top_candidates:
                # 早停：已找到进球方案，不再继续搜索
                if found_good_shot:
                    break
                    
                # 1.动态创建“奖励函数” (Wrapper)
                # 贝叶斯优化器会调用此函数，并传入参数

                # 几何先验角度
                base_phi = cand['phi_center']
                # 粗略估计速度
                base_v = 1.5 + cand['distance'] * 1.5
                
                # 使用 partial 绑定参数，确保变量隔离！
                # 这样优化器调用的函数就只剩 (V0, d_phi, theta, a, b) 这5个参数了
                target_func = functools.partial(
                    self._evaluate_action,
                    base_phi=base_phi,       # 绑定当前的几何角
                    base_v=base_v,           # 绑定当前的估计速度
                    balls=balls,             # 绑定当前球状态
                    table=table,
                    my_targets=remaining_own,
                    last_state_snapshot=last_state_snapshot
                )
                
                # 创建优化器
                seed = np.random.randint(1e6)
                optimizer = self._create_optimizer(target_func, seed)
                optimizer.maximize(
                    init_points=self.INITIAL_SEARCH,
                    n_iter=self.OPT_SEARCH
                )
                
                best_result = optimizer.max
                best_params = best_result['params']
                best_score = best_result['target']

                final_phi = (float(best_params['d_phi']) + base_phi) % 360
                final_v = np.clip((float(best_params['d_V0']) + base_v), 0.8, 7.5)

                action = {
                    'V0': final_v,
                    'phi': final_phi,
                    'theta': float(best_params['theta']),
                    'a': float(best_params['a']),
                    'b': float(best_params['b']),
                }

                if best_score > top_score:
                    top_score = best_score
                    top_action = action
                    top_base_phi = base_phi
                    top_base_v = base_v
                    
                    # 早停：找到进球方案后不再搜索其他候选
                    if best_score >= self.EARLY_STOP_SCORE:
                        logger.info("[BayesMCTS] 早停：找到进球方案 (score=%.1f)", best_score)
                        found_good_shot = True

            if top_score < 0:  # 减少误打黑8
                logger.info("[BayesMCTS] 未找到好的方案 (最高分: %.2f)。使用随机动作。", top_score)
                return self._random_action()
            
            logger.info(
                "[BayesMCTS] 决策 (得分: %.2f): V0=%.2f, V0_base=%.2f, phi=%.2f, phi_base=%.2f θ=%.2f, a=%.3f, b=%.3f",
                top_score,
                top_action['V0'],
                top_base_v,
                top_action['phi'],
                top_base_phi,
                top_action['theta'],
                top_action['a'],
                top_action['b'],
            )
            return top_action

        except Exception as e:
            logger.error("[BayesMCTS] 决策时发生严重错误，使用随机动作。原因: %s", e)
            import traceback
            logger.exception(e)
            return self._random_action()
