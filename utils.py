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

def evaluate_state(self, shot, my_targets, original_target_id):
        # 1. 空杆或未发生事件 -> 极刑
        if not shot.events: return -1000
        
        pocketed_ids = []
        cue_scratch = False
        
        # 兼容性写法获取进球事件
        for event in shot.events:
            event_name = event.event_type.name if hasattr(event.event_type, 'name') else str(event.event_type)
            if 'POCKETED' in event_name:
                pocketed_ids.extend(event.agents)
        
        # 母球落袋 -> 极刑
        if 'cue' in pocketed_ids: cue_scratch = True
        if shot.balls['cue'].state.s == 4: cue_scratch = True
        
        if cue_scratch: return -1000 # 绝对不能选会导致母球落袋的线路
        
        # 2. 进攻结果
        score = 0
        if original_target_id in pocketed_ids:
            score += 100
        elif any(bid in my_targets for bid in pocketed_ids):
            score += 60 # 运气球分低一点，鼓励打指定球
        else:
            # 没进球 -> 负分，告诉 Agent 这是一个失败的尝试 # 注释这一部分会导致母球掉袋的情况变多
            return -50 

        # 3. 简单走位 (如果进球了)
        final_cue = shot.balls['cue']
        if final_cue.state.s != 4:
            # 鼓励母球停在台面中央，不要贴库
            # 简单启发式：离中心越近越好
            cue_pos = final_cue.state.rvw[0]
            dist_to_center = np.linalg.norm(cue_pos[:2])
            score += (1.0 - dist_to_center / 1.0) * 10 
            
        return score