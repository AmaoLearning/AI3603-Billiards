import numpy as np
import pooltool as pt

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
        
        return phi, cut_angle, norm_co

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
    is_foul = False
    
    if cue_pocketed and eight_pocketed:
        score -= 150
        is_foul = True
    elif cue_pocketed:
        score -= 100
        is_foul = True
    elif eight_pocketed:
        is_targeting_eight_ball_legally = (len(player_targets) == 1 and player_targets[0] == "8")
        return 100 if is_targeting_eight_ball_legally else -150
            
    if foul_first_hit:
        score -= 30
        is_foul = True
    if foul_no_rail:
        score -= 30
        is_foul = True
        
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20
    
    if score == 0 and not is_foul:
        score = 10
    
    if not is_foul and not cue_pocketed:
        
        final_cue_pos = shot.balls['cue'].state.rvw[0]
        
        # 判断球权是否延续：进自己的球且没犯规
        turn_continues = (len(own_pocketed) > 0)
        
        # A. 确定下一杆的目标球列表
        remaining_targets = [bid for bid in player_targets if shot.balls[bid].state.s != 4]
        
        # 如果打完了所有目标球，下一杆目标是黑8
        if not remaining_targets and len(own_pocketed) > 0:
            remaining_targets = ['8']
            
        # B. 评估逻辑
        if turn_continues:
            # === 进攻模式 ===
            # 计算剩下的球里，哪一个最好打（Max Opportunity）
            best_opportunity = 0.0

            if remaining_targets:
                for tid in remaining_targets:
                    target_ball_pos = shot.balls[tid].state.rvw[0]
                    
                    # 遍历所有袋口，找这个球的最佳进球路线
                    for pid, pocket in shot.table.pockets.items():
                        quality = _calculate_shot_quality(final_cue_pos, target_ball_pos, pocket.center)
                        if quality > best_opportunity:
                            best_opportunity = quality
            
                    
                    if tid != '8':
                        # 将走位质量加入总分
                        # 权重建议：走位好坏大约值 20-30 分，相当于半个进球
                        # 这样 Agent 会在能进球的前提下，优先选择 V0 能带来高 quality 的那一杆
                        score += best_opportunity * 10.0
                    else:
                        # 8号球走位权重极高！
                        # 如果能舒服地打8号，给予更高奖励，这会迫使上一杆拼命走到这个位置
                        score += best_opportunity * 15.0
                    
            
        # else:
            # === 防守模式 (可选) ===
            # 如果没进球，我希望把母球停在让对手难受的位置
            # 这需要推断对手的目标球。简单起见，假设对手要打剩下的非我方球
            opponent_balls = [bid for bid in shot.balls if bid not in player_targets and bid != 'cue' and bid != '8' and shot.balls[bid].state.s != 4]
            if not opponent_balls: opponent_balls = ['8']
            
            opponent_best_opportunity = 0.0
            for tid in opponent_balls:
                target_ball_pos = shot.balls[tid].state.rvw[0]
                for pid, pocket in shot.table.pockets.items():
                    quality = _calculate_shot_quality(final_cue_pos, target_ball_pos, pocket.center)
                    if quality > opponent_best_opportunity:
                        opponent_best_opportunity = quality
            
            # 如果留给对手的机会很好，扣分！
            # 这会驱使 Agent 在没把握进球时，选择把球停在对手打不到的地方
            score -= opponent_best_opportunity * 20.0
        
    return score

