import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
from datetime import datetime

from .agent import Agent

from utils import calculate_ghost_ball_params, evaluate_state, get_pockets
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import functools
import copy

class NewAgent(Agent):
    """自定义 Agent 模板（待学生实现）"""
    
    def __init__(self):
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
            v4 with Noise/analyze_shot_for_reward vs BasicPro: 68.0/120.0
            v4 with Noise/evaluate_state vs BasicPro: 67.0/120.0 3h09m
            v4 with Noise/evaluate_state/more searchs: 74.0/120.0 3h48m
            v4 with Noise/evaluate_state/more searchs/more cands/banks: 73.0/120.0 7h19m
            v5 with early stop vs BasicPro: 76.0/120.0 3h24m
            v5 with more samples/more enemy pocketed punish: 74.0/120.0 6h06m
            v5 with more enemy pocketed punish/more foul punish: 77.0/120.0 4h11m 进黑球太多
            v5 with more enemy pocketed punish/more foul punish/extra tests: 85.0/120.0 5h40m
            v6 with severe punishment on foul: 85.0/120.0 4h17m
            v6 with more openninng strategy vs pro: 85.0/120.0 4h17m | vs basic: 114.0/120.0 5h26m
            v6 with optimized punishment strategy vs pro: 90.0/120.0 5h54m | vs basic: 112.0/120.0 7h04m
            v6 with above and no sampling in Bayes vs basic: 111.0/120.0 4h16m | vs pro: 78.0/120.0 3h37m
            v7 speed up in codes vs basic: 90.0/98.0 4h13m | vs pro: 77.0/114.0 4h14m
            final experiment on restricting the dphi range (-0.8, 0.8): 107.0/120.0 6h37m
        """
        super().__init__()
        
        # 搜索空间 - 扩大角度搜索范围以适应切球
        self.pbounds = {            
            'd_V0': (-2.0, 2.0),
            'd_phi': (-3.0, 3.0),  # 从 ±0.5 扩大到 ±3.0，关键改进！
            'theta': (0, 30),      # 限制跳球角度，减少无效搜索
            'a': (-0.2, 0.2),      # 缩小塞球范围，减少复杂度
            'b': (-0.2, 0.2)
        }
        
        # 优化参数 - 增加初始探索
        self.INITIAL_SEARCH = 20
        self.OPT_SEARCH = 10
        self.NOISE_SAMPLES = 3  # 多次采样取平均
        self.NOISE_JUDGES = 5 # 对最优结果多次评估
        self.EARLY_STOP_SCORE = 50
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
        
        # 翻袋策略相关参数
        self.MIN_CANDIDATES = 6  # 最少候选数量，不足时添加翻袋
        self.BANK_CUT_ANGLE_THRESHOLD = 60  # 翻袋切角阈值（比直打更严格）
        self.DIRECT_CUT_ANGLE_THRESHOLD = 75
        
        print("[BayesMCTS] (Enhanced v2) 已初始化。")
    
    def _get_table_rails(self, table):
        """获取4个库边的坐标位置"""
        # 使用 utils.py 中定义的常量
        # 球桌原点在左下角 (0,0)，中心在 (TABLE_W/2, TABLE_L/2)
        # x 范围: [0, TABLE_W], y 范围: [0, TABLE_L]
        from utils import TABLE_W, TABLE_L
        left = 0
        right = TABLE_W   # 0.9906
        bottom = 0
        top = TABLE_L     # 1.9812
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
    
    def _generate_bank_candidates(self, balls, my_targets, table, cue_pos, R):
        """生成翻袋击球候选
        
        参数:
            balls: 球状态字典
            my_targets: 目标球列表
            table: 球桌对象
            cue_pos: 母球位置
            R: 球半径
        
        返回:
            list: 翻袋候选列表
        """
        candidates = []
        rails = self._get_table_rails(table)
        
        remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
        if not remaining_own:
            remaining_own = ['8']

        for ball_id in remaining_own:
            obj_pos = balls[ball_id].state.rvw[0]
            
            # 遍历4条库边
            for rail_name in ['top', 'bottom', 'left', 'right']:
                for pid, pocket in table.pockets.items():
                    # 计算虚拟袋口
                    virtual_pos = self._get_virtual_pocket(pocket.center, rail_name, rails)
                    if virtual_pos is None:
                        continue
                    
                    # 几何计算：把虚拟袋口当做目标
                    phi_ideal, cut_angle, dist_cue_obj, dist_obj_virtual = calculate_ghost_ball_params(
                        cue_pos, obj_pos, virtual_pos, R
                    )
                    
                    # 翻袋难度较大，切角阈值设低一点
                    if abs(cut_angle) > self.BANK_CUT_ANGLE_THRESHOLD:
                        continue
                    
                    # 估算总距离：母球->目标球 + 目标球->虚拟袋口
                    total_dist = dist_cue_obj + dist_obj_virtual
                    
                    candidates.append({
                        'type': 'bank',  # 标记类型
                        'target_id': ball_id,
                        'phi_center': phi_ideal,
                        'cut_angle': cut_angle,
                        'dist_obj_pocket': dist_obj_virtual,  # 目标球到虚拟袋口距离
                        'distance': total_dist,
                        'rail': rail_name,
                        'pocket_id': pid
                    })
        
        return candidates
    
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
    
    def _generate_safety_shot(self, balls, my_targets):
        """防守策略：当没有好机会时，轻轻碰一下离得最近的目标球，避免犯规
        
        参数:
            balls: 球状态字典
            my_targets: 目标球列表
        
        返回:
            dict: 防守动作 {'V0', 'phi', 'theta', 'a', 'b'}
        """
        print("[BayesMCTS] 启动防守模式 (Safety Mode)")
        cue_pos = balls['cue'].state.rvw[0]
        min_dist = float('inf')
        best_target = None
        
        # 找最近的己方目标球
        candidates = [bid for bid in my_targets if balls[bid].state.s != 4]
        if not candidates:
            candidates = ['8']  # 如果没有目标球，打8号球
        
        for bid in candidates:
            obj_pos = balls[bid].state.rvw[0]
            dist = np.linalg.norm(obj_pos[:2] - cue_pos[:2])
            if dist < min_dist:
                min_dist = dist
                best_target = bid
        
        if best_target:
            obj_pos = balls[best_target].state.rvw[0]
            dx = obj_pos[0] - cue_pos[0]
            dy = obj_pos[1] - cue_pos[1]
            phi = np.degrees(np.arctan2(dy, dx)) % 360
            # 轻轻碰一下，确保合法但不给对手留好球
            V0 = max(1.0, min_dist * 1.5)
            print(f"[BayesMCTS] 防守: 轻触 {best_target} 号球 (V0={V0:.2f}, phi={phi:.1f})")
            return {'V0': V0, 'phi': phi, 'theta': 0, 'a': 0, 'b': 0}
        
        # 实在找不到，返回随机动作
        return self._random_action()
    
    def _random_action(self):
        """生成随机动作（兜底用）"""
        return {'V0': 1.0, 'phi': np.random.uniform(0, 360), 'theta': 0, 'a': 0, 'b': 0}
    
    def _evaluate_action(self, d_V0, d_phi, theta, a, b, base_phi, base_v, balls, table, my_targets, last_state_snapshot, sample_num):
        """
        带多次噪声采样的动作评估 (核心改进)
        
        改进点：
        1. 多次采样取平均，提高稳健性（与MCTS思想对齐）
        2. 使用 analyze_shot_for_reward
        3. 性能优化：减少不必要的深拷贝
        """
        # 1. 还原绝对参数
        phi_base = (base_phi + d_phi) % 360
        V0_base = np.clip(base_v + d_V0, 0.8, 7.5)
        
        # 2. 多次噪声采样
        n_samples = sample_num if self.enable_noise else 1
        scores = []
        
        # 预计算噪声参数（如果启用）- 批量生成随机数更快
        if self.enable_noise and n_samples > 1:
            noise_V0 = np.random.normal(0, self.noise_std['V0'], n_samples)
            noise_phi = np.random.normal(0, self.noise_std['phi'], n_samples)
            noise_theta = np.random.normal(0, self.noise_std['theta'], n_samples)
            noise_a = np.random.normal(0, self.noise_std['a'], n_samples)
            noise_b = np.random.normal(0, self.noise_std['b'], n_samples)
        
        for i in range(n_samples):
            # 注入噪声（如果启用）
            if self.enable_noise and n_samples > 1:
                V0 = np.clip(V0_base + noise_V0[i], 0.5, 8.0)
                phi = (phi_base + noise_phi[i]) % 360
                theta_n = np.clip(theta + noise_theta[i], 0, 90)
                a_n = np.clip(a + noise_a[i], -0.5, 0.5)
                b_n = np.clip(b + noise_b[i], -0.5, 0.5)
            else:
                V0, phi, theta_n, a_n, b_n = V0_base, phi_base, theta, a, b
            
            # 构建模拟环境 - 使用更轻量的拷贝方式
            sim_balls = {bid: copy.copy(ball) for bid, ball in balls.items()}
            # 对球状态进行深拷贝（状态是可变的）
            for bid in sim_balls:
                sim_balls[bid].state = copy.deepcopy(balls[bid].state)
            
            sim_table = table  # table 在模拟中不会被修改，无需拷贝
            cue = pt.Cue(cue_ball_id="cue")
            sim_shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
            sim_shot.cue.set_state(V0=V0, phi=phi, theta=theta_n, a=a_n, b=b_n)
            
            try:
                pt.simulate(sim_shot, inplace=True)
                # 使用与 BasicAgentPro 相同的评估函数！
                score = evaluate_state(sim_shot, last_state_snapshot, my_targets)
            except:
                score = -500.0
            
            if score <= -500: return score # 增大对犯规的惩罚力度，但是应该尤其惩罚黑8非正常进洞

            scores.append(score)
        
        avg_score = np.mean(scores)
        min_score = min(scores)
        
        # 返回平均分（更稳健）
        if min_score < 0:
            # 负分惩罚：平均分与最差分的加权
            # 权重可调：更重视最差情况则增大 worst 权重
            return avg_score * 0.6 + min_score * 0.4
        else:
            return avg_score

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
            print("[BayesMCTS] Agent decision函数未收到balls关键信息，使用随机动作。")
            return self._random_action()
        try:
            
            # 保存一个击球前的状态快照，用于对比（只拷贝必要的状态信息）
            last_state_snapshot = {}
            for bid, ball in balls.items():
                last_state_snapshot[bid] = copy.copy(ball)
                last_state_snapshot[bid].state = copy.deepcopy(ball.state)

            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                remaining_own = ["8"]
                print("[BayesMCTS] 我的目标球已全部清空，自动切换目标为：8号球")
            
            cue_ball = balls['cue']
            cue_pos = cue_ball.state.rvw[0]
            R = cue_ball.params.R

            # ========== 开球特殊处理 ==========
            # 判断是否为开球局面：所有目标球都在台上（7个）
            is_break_shot = balls['1'].state.t == 0
            
            if is_break_shot:
                if '1' in remaining_own:
                    # 开球策略：直接大力冲击球堆，不做复杂搜索
                    # 标准开球角度：瞄准1号球（球堆顶端）
                    one_pos = balls['1'].state.rvw[0]
                    dx = one_pos[0] - cue_pos[0]
                    dy = one_pos[1] - cue_pos[1]
                    phi_break = np.degrees(np.arctan2(dy, dx)) % 360
                    
                    print(f"[BayesMCTS] 检测到开球局面，使用Solid快速开球策略 (phi={phi_break:.1f})")
                    return {
                        'V0': 7.0,  # 大力开球
                        'phi': phi_break + 0.5,
                        'theta': 0,
                        'a': 0.1,
                        'b': 0
                    }
                else:
                    print("[BayesMCTS] 检测到开球局面，使用Stripe快速开球策略 (phi=90.0)")
                    return {
                        'V0': 7.0,
                        'phi': 90.0,
                        'theta': 0,
                        'a': 0.1,
                        'b': 0
                    }
            # ====================================
            
            candidates = []

            print(f"[BayesMCTS] 正在为 Player (targets: {remaining_own}) 搜索最佳击球...")
            
            # ========== 1. 生成直打候选 ==========
            for ball_id in remaining_own:
                obj_pos = balls[ball_id].state.rvw[0]
                for pid, pocket in table.pockets.items():
                    phi_ideal, cut_angle, dist_cue_obj, dist_obj_pocket = calculate_ghost_ball_params(cue_pos, obj_pos, pocket.center, R)
                    
                    # 放宽阈值到 80 度（与其他 Agent 一致）
                    if abs(cut_angle) > self.DIRECT_CUT_ANGLE_THRESHOLD: continue
                    
                    candidates.append({
                        'type': 'direct',  # 标记为直打
                        'target_id': ball_id,
                        'phi_center': phi_ideal,
                        'cut_angle': cut_angle,
                        'distance': dist_cue_obj,
                        'dist_obj_pocket': dist_obj_pocket  # 新增：目标球到袋口距离
                    })
            
            # ========== 2. 当直打候选不足时，添加翻袋候选 ==========
            if len(candidates) < self.MIN_CANDIDATES:
                print(f"[BayesMCTS] 直打候选不足 ({len(candidates)} < {self.MIN_CANDIDATES})，添加翻袋策略...")
                bank_candidates = self._generate_bank_candidates(balls, remaining_own, table, cue_pos, R)
                candidates.extend(bank_candidates)
                print(f"[BayesMCTS] 添加了 {len(bank_candidates)} 个翻袋候选")
            
            # ========== 3. 混合排序 ==========
            # 改进排序：直打优先，综合考虑切角、母球距离和目标球到袋口距离
            def sort_key(c):
                # 翻袋惩罚：因为翻袋难度更大，给一定的排序惩罚
                penalty = 0 if c.get('type', 'direct') == 'direct' else 20
                # 目标球到袋口距离：距离越远难度越大
                dist_obj_pocket = c.get('dist_obj_pocket', 0.5)
                return c['cut_angle'] * 1.5 + c['distance'] * 3 + dist_obj_pocket * 8 + penalty
            
            candidates.sort(key=sort_key)
            
            # 只取前 3 个候选，平衡速度与质量
            top_candidates = candidates[:self.MIN_CANDIDATES]
            
            # 日志输出当前候选的类型分布
            direct_count = sum(1 for c in top_candidates if c.get('type', 'direct') == 'direct')
            bank_count = len(top_candidates) - direct_count
            if bank_count > 0:
                print(f"[BayesMCTS] 最终候选: {direct_count} 直打 + {bank_count} 翻袋")

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
                # 粗略估计速度：考虑母球到目标球距离 + 目标球到袋口距离
                dist_obj_pocket = cand.get('dist_obj_pocket', 0.5)
                base_v = 1.2 + cand['distance'] * 2.0 + dist_obj_pocket * 1.5
                
                # 使用 partial 绑定参数，确保变量隔离！
                # 这样优化器调用的函数就只剩 (V0, d_phi, theta, a, b) 这5个参数了
                target_func = functools.partial(
                    self._evaluate_action,
                    base_phi=base_phi,       # 绑定当前的几何角
                    base_v=base_v,           # 绑定当前的估计速度
                    balls=balls,             # 绑定当前球状态
                    table=table,
                    my_targets=remaining_own,
                    last_state_snapshot=last_state_snapshot,
                    sample_num=self.NOISE_SAMPLES
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

                best_score = self._evaluate_action(
                    float(best_params['d_V0']),
                    float(best_params['d_phi']),
                    action['theta'],
                    action['a'],
                    action['b'],
                    base_phi,
                    base_v,
                    balls,             
                    table,
                    remaining_own,
                    last_state_snapshot,
                    self.NOISE_JUDGES)

                if best_score > top_score:
                    top_score = best_score
                    top_action = action
                    top_base_phi = base_phi
                    top_base_v = base_v
                    
                    # 早停：找到进球方案后不再搜索其他候选
                    if best_score >= self.EARLY_STOP_SCORE:
                        print(f"[BayesMCTS] 早停：找到进球方案 (score={best_score:.1f})")
                        found_good_shot = True

            if top_score < -50:  # 减少误打黑8
                print(f"[BayesMCTS] 未找到好的方案 (最高分: {top_score:.2f})。切换防守模式。")
                return self._generate_safety_shot(balls, remaining_own)
            
            print(
                f"[BayesMCTS] 决策 (得分: {top_score:.2f}): V0={top_action['V0']:.2f}, V0_base={top_base_v:.2f}, "
                f"phi={top_action['phi']:.2f}, phi_base={top_base_phi:.2f}, θ={top_action['theta']:.2f}, "
                f"a={top_action['a']:.3f}, b={top_action['b']:.3f}"
            )
            return top_action

        except Exception as e:
            print(f"[BayesMCTS] 决策时发生严重错误，使用随机动作。原因: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()