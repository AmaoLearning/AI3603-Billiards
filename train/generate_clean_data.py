import pooltool as pt
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

def get_perfect_delta(cue_pos, obj_pos, pocket_pos, R, V0):
    """
    寻找进球区间的中心点，消除袋口宽度带来的噪声
    """
    # 1. 计算几何基准
    vec_obj_pock = np.array(pocket_pos) - np.array(obj_pos)
    vec_obj_pock_unit = vec_obj_pock / np.linalg.norm(vec_obj_pock)
    ghost_pos = np.array(obj_pos) - vec_obj_pock_unit * (2*R)
    vec_cue_ghost = ghost_pos - np.array(cue_pos)
    phi_geo = np.degrees(np.arctan2(vec_cue_ghost[1], vec_cue_ghost[0])) % 360
    
    # 2. 扫描进球区间
    # 真实的物理偏差通常在 [-2, 2] 度之间
    scan_range = np.linspace(-2.5, 2.5, 100) # 0.05度精度
    made_offsets = []
    
    table = pt.Table.from_game_type(pt.GameType.EIGHTBALL)
    
    for offset in scan_range:
        phi_try = phi_geo + offset
        
        # 极简模拟
        cue = pt.Cue(cue_ball_id="cue")
        cue.set_state(V0=V0, phi=phi_try, theta=0, a=0, b=0)
        balls = {'cue': pt.Ball('cue'), '1': pt.Ball('1')}
        balls['cue'].state.rvw[0] = cue_pos
        balls['1'].state.rvw[0] = obj_pos
        
        shot = pt.System(table=table, balls=balls, cue=cue)
        pt.simulate(shot, inplace=True)
        
        # 判定进球
        if shot.balls['1'].state.s == 4:
            made_offsets.append(offset)
            
    if not made_offsets:
        return None, None # 没打进，或者是死球
        
    # 3. 取区间中点作为 Label (最核心的一步)
    # 这能极大降低噪声
    perfect_offset = np.mean(made_offsets)
    
    # 计算切角用于特征
    vec_cue_obj = np.array(obj_pos) - np.array(cue_pos)
    cos_theta = np.dot(vec_cue_obj[:2], vec_obj_pock[:2]) / (np.linalg.norm(vec_cue_obj[:2]) * np.linalg.norm(vec_obj_pock[:2]))
    cut_angle = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
    
    dist = np.linalg.norm(vec_cue_obj[:2])
    
    return perfect_offset, cut_angle, dist

def generate_balanced_data(num_samples=3000):
    data = []
    R = pt.Ball('1').params.R
    table = pt.Table.from_game_type(pt.GameType.EIGHTBALL)
    pockets = list(table.pockets.values())
    
    pbar = tqdm(total=num_samples)
    
    while len(data) < num_samples:
        # 随机生成状态
        cue_pos = [np.random.uniform(-1, 1), np.random.uniform(-0.5, 0.5), R]
        obj_pos = [np.random.uniform(-1, 1), np.random.uniform(-0.5, 0.5), R]
        if np.linalg.norm(np.array(cue_pos) - np.array(obj_pos)) < 0.2: continue
        
        pocket = pockets[np.random.randint(6)]
        V0 = np.random.uniform(1.5, 7.0)
        
        label, cut_angle, dist = get_perfect_delta(cue_pos, obj_pos, pocket.center, R, V0)
        
        if label is None: continue
        
        # 关键：数据平衡策略
        # 简单球(切角<15)丢弃率 80%
        if abs(cut_angle) < 15 and np.random.random() < 0.8:
            continue
            
        data.append({
            'cut_angle': cut_angle,
            'distance': dist,
            'V0': V0,
            'label_delta': label
        })
        pbar.update(1)
        
    os.makedirs('data', exist_ok=True)
    pd.DataFrame(data).to_csv(os.path.join('data', 'clean_aim_data.csv'), index=False)
    print("高质量数据生成完毕！")

if __name__ == "__main__":
    generate_balanced_data()