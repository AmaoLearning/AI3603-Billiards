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
import torch
# from poolagent.pool import Pool as CuetipEnv, State as CuetipState
# from poolagent import FunctionAgent

from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from train.train_fast import AimNet
from utils import calculate_ghost_ball_params, get_pockets, evaluate_state

import functools


logger = logging.getLogger("evaluate")


def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """
    分析击球结果并计算奖励分数
    
    参数：
        shot: 已完成物理模拟的 System 对象
        last_state: 击球前的球状态，{ball_id: Ball}
        player_targets: 当前玩家目标球ID，['1', '2', ...]
    
    返回：
        float: 奖励分数
            +50/球（己方进球）, +100（合法黑8）, +10（合法无进球）
            -100（白球进袋）, -150（非法黑8）, -30（首球/碰库犯规）
    """
    
    # 1. 基本分析
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
    
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. 分析首球碰撞
    first_contact_ball_id = None
    foul_first_hit = False
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            other_ids = [i for i in ids if i != 'cue']
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break
    
    if first_contact_ball_id is None:
        if len(last_state) > 2:  # 只有白球和8号球时不算犯规
             foul_first_hit = True
    else:
        remaining_own_before = [bid for bid in player_targets if last_state[bid].state.s != 4]
        opponent_plus_eight = [bid for bid in last_state.keys() if bid not in player_targets and bid not in ['cue']]
        if ('8' not in opponent_plus_eight):
            opponent_plus_eight.append('8')
            
        if len(remaining_own_before) > 0 and first_contact_ball_id in opponent_plus_eight:
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
        score -= 150
    elif cue_pocketed:
        score -= 100
    elif eight_pocketed:
        is_targeting_eight_ball_legally = (len(player_targets) == 1 and player_targets[0] == "8")
        score += 100 if is_targeting_eight_ball_legally else -150
            
    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30
        
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20
    
    if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
        score = 10
        
    return score

class Agent():
    """Agent 基类"""
    def __init__(self):
        pass
    
    def decision(self, *args, **kwargs):
        """决策方法（子类需实现）
        
        返回：dict, 包含 'V0', 'phi', 'theta', 'a', 'b'
        """
        pass
    
    def _random_action(self,):
        """生成随机击球动作
        
        返回：dict
            V0: [0.5, 8.0] m/s
            phi: [0, 360] 度
            theta: [0, 90] 度
            a, b: [-0.5, 0.5] 球半径比例
        """
        action = {
            'V0': round(random.uniform(0.5, 8.0), 2),   # 初速度 0.5~8.0 m/s
            'phi': round(random.uniform(0, 360), 2),    # 水平角度 (0°~360°)
            'theta': round(random.uniform(0, 90), 2),   # 垂直角度
            'a': round(random.uniform(-0.5, 0.5), 3),   # 杆头横向偏移（单位：球半径比例）
            'b': round(random.uniform(-0.5, 0.5), 3)    # 杆头纵向偏移
        }
        return action



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
        
        logger.info("BasicAgent (Smart, pooltool-native) 已初始化。")

    
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
            logger.info("[BasicAgent] Agent decision函数未收到balls关键信息，使用随机动作。")
            return self._random_action()
        try:
            
            # 保存一个击球前的状态快照，用于对比
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ["8"]
                logger.info("[BasicAgent] 我的目标球已全部清空，自动切换目标为：8号球")

            # 1.动态创建“奖励函数” (Wrapper)
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
                    
                    # 关键：使用 pooltool 物理引擎 (世界A)
                    pt.simulate(shot, inplace=True)
                except Exception as e:
                    # 模拟失败，给予极大惩罚
                    return -500
                
                # 使用我们的“裁判”来打分
                score = analyze_shot_for_reward(
                    shot=shot,
                    last_state=last_state_snapshot,
                    player_targets=my_targets
                )


                return score

            logger.info("[BasicAgent] 正在为 Player (targets: %s) 搜索最佳击球...", remaining_own)
            
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
                logger.info("[BasicAgent] 未找到好的方案 (最高分: %.2f)。使用随机动作。", best_score)
                return self._random_action()
            action = {
                'V0': float(best_params['V0']),
                'phi': float(best_params['phi']),
                'theta': float(best_params['theta']),
                'a': float(best_params['a']),
                'b': float(best_params['b']),
            }

            logger.info(
                "[BasicAgent] 决策 (得分: %.2f): V0=%.2f, phi=%.2f, θ=%.2f, a=%.3f, b=%.3f",
                best_score,
                action['V0'],
                action['phi'],
                action['theta'],
                action['a'],
                action['b'],
            )
            return action

        except Exception as e:
            logger.error("[BasicAgent] 决策时发生严重错误，使用随机动作。原因: %s", e)
            import traceback
            logger.exception(e)
            return self._random_action()

class NewAgent(Agent):
    """自定义 Agent 模板（待学生实现）"""
    
    def __init__(self):
        model_path = os.path.join('train', 'checkpoints', 'aim_model.pth')
        #self.agent = HybridLearningAgent(model_path)
        # self.agent = MCTSAgent()
        self.agent = BayesMCTSAgent()
    
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

class BayesMCTSAgent(BasicAgent):
    """基于贝叶斯优化的智能MCTS Agent"""
    
    def __init__(self, target_balls=None):
        """初始化 Agent
        
        参数：
            target_balls: 保留参数，暂未使用
        """
        super().__init__()
        
        # 搜索空间
        self.pbounds = {
            'V0': (0.5, 8.0),
            'd_phi': (-5, 5),
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
            'd_phi': 0.1,
            'theta': 0.1,
            'a': 0.003,
            'b': 0.003
        }
        self.enable_noise = False
        
        logger.info("[BayesMCTS] (Smart, pooltool-native) 已初始化。")
    
    def _evaluate_action(self, V0, d_phi, theta, a, b, base_phi, balls, table, my_targets, last_state_snapshot):
        # 1. 还原绝对角度
        actual_phi = base_phi + d_phi
        
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        
        # 2. 设置状态
        sim_shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
        sim_shot.cue.set_state(V0=V0, phi=actual_phi, theta=theta, a=a, b=b)
        
        try:
            pt.simulate(sim_shot, inplace=True)
        except:
            return -500.0
            
        # 3. 评分
        score = analyze_shot_for_reward(
            shot=sim_shot,
            last_state=last_state_snapshot,
            player_targets=my_targets
        )
        return score

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
                my_targets = ["8"]
                logger.info("[BayesMCTS] 我的目标球已全部清空，自动切换目标为：8号球")
            
            candidates = []
            cue_ball = balls['cue']
            cue_pos = cue_ball.state.rvw[0]
            R = cue_ball.params.R

            logger.info("[BayesMCTS] 正在为 Player (targets: %s) 搜索最佳击球...", remaining_own)
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
            
            candidates.sort(key=lambda x: x['cut_angle'])
            top_candidates = candidates[:2]

            top_action = None
            top_score = -float('inf')
            top_base = 0
            top_delta = 0

            for cand in top_candidates:
                # 1.动态创建“奖励函数” (Wrapper)
                # 贝叶斯优化器会调用此函数，并传入参数

                # 几何先验角度
                base_phi = cand['phi_center']
                
                # 使用 partial 绑定参数，确保变量隔离！
                # 这样优化器调用的函数就只剩 (V0, d_phi, theta, a, b) 这5个参数了
                target_func = functools.partial(
                    self._evaluate_action,
                    base_phi=base_phi,       # 绑定当前的几何角
                    balls=balls,             # 绑定当前球状态
                    table=table,
                    my_targets=my_targets,
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

                final_phi = (float(best_params['phi']) + base_phi) % 360

                action = {
                    'V0': float(best_params['V0']),
                    'phi': final_phi,
                    'theta': float(best_params['theta']),
                    'a': float(best_params['a']),
                    'b': float(best_params['b']),
                }

                if best_score > top_score:
                    top_score = best_score
                    top_action = action
                    top_base = base_phi
                    top_delta = float(best_params['phi'])

            if top_score < 10:
                logger.info("[BayesMCTS] 未找到好的方案 (最高分: %.2f)。使用随机动作。", top_score)
                return self._random_action()
            
            logger.info(
                "[BayesMCTS] 决策 (得分: %.2f): V0=%.2f, phi_base=%.2f, phi_delta=%.2f θ=%.2f, a=%.3f, b=%.3f",
                top_score,
                top_action['V0'],
                top_base,
                top_delta,
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