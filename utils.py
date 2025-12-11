import numpy as np

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

def evaluate_state(shot, my_targets, original_target_id):
        """改进的评分函数"""
        if not shot.events: 
            return -1000
        
        pocketed_ids = []
        cue_scratch = False
        
        for event in shot.events:
            # 兼容性写法
            if event.event_type.name == 'POCKETED':
                pocketed_ids.extend(event.agents)
        
        if 'cue' in pocketed_ids: cue_scratch = True
        if shot.balls['cue'].state.s == 4: cue_scratch = True

        if cue_scratch: return -500
        
        # 进攻结果判定
        score = 0
        if original_target_id in pocketed_ids:
            score += 100 
        elif any(bid in my_targets for bid in pocketed_ids):
            score += 80 
        # else:
        #     return -50 

        # 走位评估
        final_balls = shot.balls
        final_cue = final_balls['cue']
        
        if original_target_id == '8':
             others = [b for b in my_targets if b != '8' and final_balls[b].state.s != 4]
             if not others: return 10000
             else: return -1000 

        cue_pos = final_cue.state.rvw[0]
        R = final_cue.params.R
        
        remaining = [b for b in my_targets if b not in pocketed_ids and final_balls[b].state.s != 4]
        if not remaining: remaining = ['8']

        best_next_shot = 0
        for bid in remaining:
            b_pos = final_balls[bid].state.rvw[0]
            for pid, pocket in get_pockets(shot.table).items():
                _, cut, dist = calculate_ghost_ball_params(cue_pos, b_pos, pocket.center, R)
                if cut < 50:
                    quality = (60 - cut) + (1.0 - abs(dist - 1.0))*20
                    if quality > best_next_shot: best_next_shot = quality
        
        return score + best_next_shot * 0.5