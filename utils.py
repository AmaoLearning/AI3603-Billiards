import numpy as np
import pooltool as pt
import random

try:
    import torch
except ImportError:
    torch = None

def set_random_seed(enable=False, seed=42):
    """
    设置随机种子以确保实验的可重复性
    
    Args:
        enable (bool): 是否启用固定随机种子
        seed (int): 当 enable 为 True 时使用的随机种子
    """
    if enable:
        # 设置 Python 随机种子
        random.seed(seed)
        # 设置 NumPy 随机种子
        np.random.seed(seed)
        
        # 设置 PyTorch 随机种子（如果可用）
        if torch is not None:
            torch.manual_seed(seed)  # CPU 随机种子
            torch.cuda.manual_seed(seed)  # 当前 GPU 随机种子
            torch.cuda.manual_seed_all(seed)  # 所有 GPU 随机种子
            # 确保 CUDA 操作的确定性
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        print(f"随机种子已设置为: {seed}")
    else:
        # 重置为随机性，使用系统时间作为种子
        random.seed()
        np.random.seed(None)
        
        print("随机种子已禁用，使用完全随机模式")

# 物理常数
TABLE_W = 0.9906
TABLE_L = 1.9812
BALL_R = 0.028575

def calculate_ghost_ball_params(cue_pos, obj_pos, pocket_pos, R):
        """计算几何参数 (Ghost Ball)"""
        pocket_vec = np.array([pocket_pos[0], pocket_pos[1], 0])
        obj_vec = np.array([obj_pos[0], obj_pos[1], 0])
        
        # 向量：目标球 -> 袋口
        vec_obj_pocket = pocket_vec - obj_vec # 注意方向：从球指向袋口
        dist_obj_pocket = np.linalg.norm(vec_obj_pocket)
        vec_obj_pocket_unit = vec_obj_pocket / (dist_obj_pocket + 1e-6)
        
        # 假想球位置：目标球中心沿进球线反向延伸 2R
        # 也就是：母球撞击目标球时，母球应该在的位置
        ghost_pos = obj_vec - vec_obj_pocket_unit * (2 * R)
        
        # 瞄准向量：母球 -> 假想球
        cue_vec_3d = np.array([cue_pos[0], cue_pos[1], 0])
        aim_vec = ghost_pos - cue_vec_3d
        
        # 计算角度 phi
        phi = np.degrees(np.arctan2(aim_vec[1], aim_vec[0])) % 360
        
        # 计算切角 (Cut Angle)
        # 向量：母球 -> 目标球
        vec_cue_obj = obj_vec - cue_vec_3d
        norm_co = np.linalg.norm(vec_cue_obj)
        
        if norm_co == 0: return phi, 180, 1000
        
        # 切角是 (母球-目标球) 连线 与 (目标球-袋口) 连线 的夹角
        cos_theta = np.dot(vec_cue_obj, vec_obj_pocket) / (norm_co * dist_obj_pocket + 1e-6)
        cut_angle = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
        
        return phi, cut_angle, norm_co, dist_obj_pocket

def get_pockets(table):
        return table.pockets

def _calculate_shot_quality(cue_pos, target_pos, pocket_pos):
    """
    基于物理尺寸优化的击球质量评估
    数据依据: Table(1.98x0.99), R=0.0285
    """
    # --- 2. 向量计算 ---
    vec_cue_obj = np.array(target_pos) - np.array(cue_pos)
    vec_obj_pocket = np.array(pocket_pos) - np.array(target_pos)
    
    dist_cue_obj = np.linalg.norm(vec_cue_obj)
    dist_obj_pocket = np.linalg.norm(vec_obj_pocket)
    
    # 极近距离保护 (防止除零或物理穿模)
    if dist_cue_obj < 0.001 or dist_obj_pocket < 0.001: return 0.0

    # --- 3. 切角评分 (Angle Score) ---
    # 计算余弦夹角
    cos_theta = np.dot(vec_cue_obj, vec_obj_pocket) / (dist_cue_obj * dist_obj_pocket)
    cut_angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
    
    # 超过 80 度几乎不可打
    if cut_angle_deg > 80:
        angle_score = 0.0
    else:
        # 非线性衰减：小角度(直球)分数高，大角度分数掉得快
        # 理由：在小台子上，切球带来的偏差更容易导致碰壁
        angle_score = (1.0 - (cut_angle_deg / 80.0) ** 1.5)

    # --- 4. 距离评分 (Distance Score) - 针对 1.98m 桌子优化 ---
    # 定义黄金区间：因桌子较小，最佳距离缩短为 0.4m - 0.8m
    # 0.4m ~ 14个球身，0.8m ~ 28个球身
    
    if dist_cue_obj < 0.15: 
        # 太近 (<15cm)：架杆不便，且容易推杆犯规
        dist_score = 0.3
    elif 0.3 <= dist_cue_obj <= 0.9:
        # 黄金区域
        dist_score = 1.0
    else:
        # 远台惩罚
        # 超过 1.5m (桌长的3/4) 视为极难
        dist_score = max(0, 1.0 - (dist_cue_obj - 0.9) * 0.8)

    # --- 5. [新增] 贴库惩罚 (Rail Proximity Penalty) ---
    # 这是一个巨大的战略提升。母球如果贴库，击球难度激增。
    # 边界通常是 [0, W] 和 [0, L]
    # 我们定义“贴库”为距离库边小于 2.5 个半径 (约 7cm)
    rail_margin = BALL_R * 2.5
    
    cx, cy = cue_pos[0], cue_pos[1]
    
    is_rail_bridge = (
        cx < rail_margin or cx > (TABLE_W - rail_margin) or
        cy < rail_margin or cy > (TABLE_L - rail_margin)
    )
    
    rail_penalty = 0.0
    if is_rail_bridge:
        # 如果贴库，质量直接打对折，甚至更低
        # 这会迫使 V0 选择哪怕远一点、但不要贴库的力道
        rail_penalty = 0.4 

    # --- 6. 综合评分 ---
    # 基础分
    base_quality = angle_score * 0.6 + dist_score * 0.4
    
    # 应用惩罚
    final_quality = base_quality - rail_penalty
    
    return max(0.0, final_quality)

def evaluate_state(shot: pt.System, last_state: dict, player_targets: list):
    """
    综合评估击球结果（结合规则判定 + 走位质量）
    
    设计原则：
    1. 基础评分与 analyze_shot_for_reward 完全对齐（确保规则正确性）
    2. 额外加入走位奖励，但权重不能压过进球（走位是锦上添花）
    3. 简化逻辑，移除有争议的惩罚项（如 disturbed_8）
    
    参数：
        shot: 已完成物理模拟的 System 对象
        last_state: 击球前的球状态，{ball_id: Ball}
        player_targets: 当前玩家目标球ID，['1', '2', ...] 或 ['8']
    
    返回：
        float: 综合奖励分数
    """
    
    # ========== 第一部分：规则评分（与 analyze_shot_for_reward 对齐）==========
    
    # 1. 基本分析
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
    
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. 分析首球碰撞
    first_contact_ball_id = None
    foul_first_hit = False
    valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break
    
    # 首球犯规判定
    if first_contact_ball_id is None:
        if len(last_state) > 2 or player_targets != ['8']:
            foul_first_hit = True
    else:
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

    if len(new_pocketed) == 0 and first_contact_ball_id is not None:
        if not cue_hit_cushion and not target_hit_cushion:
            foul_no_rail = True
    
    # 4. 计算基础分数（与 analyze_shot_for_reward 完全一致）
    score = 0.0
    is_foul = False
    
    if cue_pocketed and eight_pocketed:
        return -500.0  # 直接返回，最严重犯规
    elif cue_pocketed:
        return -100.0  # 白球落袋
    elif eight_pocketed:
        is_legal_8 = (len(player_targets) == 1 and player_targets[0] == "8")
        return 150.0 if is_legal_8 else -500.0
            
    if foul_first_hit:
        score -= 50.0
        is_foul = True
    if foul_no_rail:
        score -= 50.0
        is_foul = True
    
    score -= len(enemy_pocketed) * 50.0
    
    if is_foul: # 犯规时不需要考虑进己方球分数
        return score
        
    score += len(own_pocketed) * 50.0
    
    if score == 0:
        score = 10.0  # 合法无进球基础分
    
    # ========== 第二部分：走位质量评估（BayesMCTS 专属增强）==========
    
    
    final_cue_pos = shot.balls['cue'].state.rvw[0]
    
    # 确定下一杆目标球
    remaining_targets = [bid for bid in player_targets if shot.balls[bid].state.s != 4]
    if not remaining_targets:
        remaining_targets = ['8']  # 清完己方球后打8号
    
    # 进球后球权延续，评估下一杆机会
    if len(own_pocketed) > 0:
        best_next_shot_quality = 0.0
        
        for tid in remaining_targets:
            if shot.balls[tid].state.s == 4:
                continue  # 已进袋的球跳过
            target_pos = shot.balls[tid].state.rvw[0]
            
            for pid, pocket in shot.table.pockets.items():
                quality = _calculate_shot_quality(final_cue_pos, target_pos, pocket.center)
                best_next_shot_quality = max(best_next_shot_quality, quality)
        
        # 走位奖励：最高 +25 分（约半个进球的价值）
        # 这样 Agent 会在能进球的基础上，优先选择走位好的方案
        position_bonus = best_next_shot_quality * 25.0
        score += position_bonus
    else:
        # ========== 无进球时评估对手机会 ==========
        # 如果己方未进球，评估是否给对手留下了好球机会
        # 对手目标球：排除己方目标球、cue、8号球后剩余的球
        all_balls = {'1', '2', '3', '4', '5', '6', '7', '9', '10', '11', '12', '13', '14', '15'}
        opponent_targets = [bid for bid in all_balls if bid not in player_targets and shot.balls[bid].state.s != 4]
        
        if opponent_targets:
            best_opponent_quality = 0.0
            
            for tid in opponent_targets:
                target_pos = shot.balls[tid].state.rvw[0]
                
                for pid, pocket in shot.table.pockets.items():
                    quality = _calculate_shot_quality(final_cue_pos, target_pos, pocket.center)
                    best_opponent_quality = max(best_opponent_quality, quality)
            
            # 对手好球惩罚：如果给对手留下了高质量机会，扣分
            # 惩罚幅度：最高 -15 分（不宜过大，避免过于保守）
            # 只有当对手机会质量 > 0.5 时才开始惩罚
            if best_opponent_quality > 0.5:
                opponent_penalty = (best_opponent_quality - 0.5) * 30.0  # 最高约 -15 分
                score -= opponent_penalty
    
    return score

