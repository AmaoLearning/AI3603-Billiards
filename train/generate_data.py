import pooltool as pt
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import os

def generate_fast_data(num_samples=10000, filename='fast_aim_data.csv'):
    data = []
    
    # 初始化标准8球桌
    table = pt.Table.from_game_type(pt.GameType.EIGHTBALL)
    R = pt.Ball('1').params.R
    bounds = { 'min_x': R, 'max_x': table.w - R,
               'min_y': R, 'max_y': table.l - R} 
 
    # 获取所有袋口以便快速随机选择
    pockets = list(table.pockets.values())
    
    start_time = time.time()
    print(f"开始采集 {num_samples} 条数据...")
    
    # 使用 while 循环直到凑够数据
    pbar = tqdm(total=num_samples)
    min_dist = 2 * R 
 
    while len(data) < num_samples:
        # 1. 快速随机摆球
        # 技巧：直接在半场随机生成，提高效率
        cue_pos = [np.random.uniform(bounds['min_x'], bounds['max_x']), np.random.uniform(bounds['min_y'], bounds['max_y']), R]
        obj_pos = [np.random.uniform(bounds['min_x'], bounds['max_x']), np.random.uniform(bounds['min_y'], bounds['max_y']), R]
        
        # 距离太近直接重开
        dist = np.linalg.norm(np.array(cue_pos[:2]) - np.array(obj_pos[:2]))
        if dist <= min_dist: continue
        
        # 随机选袋口
        pocket = pockets[np.random.randint(6)]
        
        # 2. 计算几何基准 (Features)
        vec_obj_pock = np.array(pocket.center) - np.array(obj_pos)
        vec_obj_pock_unit = vec_obj_pock / np.linalg.norm(vec_obj_pock)
        ghost_pos = np.array(obj_pos) - vec_obj_pock_unit * (2*R)
        vec_cue_ghost = ghost_pos - np.array(cue_pos)
        
        # 几何 phi
        phi_geo = np.degrees(np.arctan2(vec_cue_ghost[1], vec_cue_ghost[0])) % 360
        
        # 计算切角
        vec_cue_obj = np.array(obj_pos) - np.array(cue_pos)
        cos_theta = np.dot(vec_cue_obj[:2], vec_obj_pock[:2]) / (np.linalg.norm(vec_cue_obj[:2]) * np.linalg.norm(vec_obj_pock[:2]))
        cut_angle = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
        
        # 3. 简单过滤 (避免全是直线球)
        # 只有当切角很小(<10度)时，我们才以 50% 的概率跳过
        # 这样既保留了简单样本，又不会让数据严重失衡
        if abs(cut_angle) >= 90 or (abs(cut_angle) < 10 and np.random.random() < 0.5):
            continue
            
        # 4. 随机尝试进球 (蒙特卡洛)
        # 我们知道物理偏差通常在 [-2, 2] 之间
        # 随机生成一个力度
        V0 = np.random.uniform(2.0, 8.0)
        
        # 尝试 5 次，每次随机取一个偏移量
        # 只要进球就记录，然后立马 break
        shot_success = False
        
        for _ in range(5):
            # 在几何角度附近随机采样
            # 采用正态分布采样，更容易命中靠近中心的点，减少边缘噪音
            offset_try = np.random.normal(0, 1.0) 
            
            # 限制范围
            if offset_try < -2.5 or offset_try > 2.5: continue
            
            # 极简模拟
            cue = pt.Cue(cue_ball_id="cue")
            cue.set_state(V0=V0, phi=phi_geo + offset_try, theta=0, a=0, b=0)
            
            # 最小化系统：只放两个球
            balls = {'cue': pt.Ball('cue'), '1': pt.Ball('1')}
            balls['cue'].state.rvw[0] = cue_pos
            balls['1'].state.rvw[0] = obj_pos
            
            shot = pt.System(table=table, balls=balls, cue=cue)
            
            try:
                pt.simulate(shot, inplace=True)
                
                # 判定：目标球落袋 (state.s == 4)
                if shot.balls['1'].state.s == 4:
                    # 记录数据
                    data.append({
                        'cut_angle': cut_angle,
                        'distance': dist,
                        'V0': V0,
                        'label_delta': offset_try # 直接记录这个能进的偏移量
                    })
                    shot_success = True
                    
                    print(f"[Shot] cut_angle={cut_angle}, dist={dist}, V0={V0}, label_delta={offset_try}: shot_success={shot_success}")
                    break # 这一杆记录完就跳出，不需要找最佳解
            except:
                continue

            print(f"[Shot] cut_angle={cut_angle}, dist={dist}, V0={V0}, label_delta={offset_try}: shot_success={shot_success}")
        
        if shot_success:
            pbar.update(1)

    pbar.close()
    
    # 保存
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"完成！耗时 {time.time()-start_time:.1f}s, 数据已保存为 {filename}")

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    generate_fast_data(num_samples=10000, filename=os.path.join('data', 'fast_aim_data.csv')) # 先生成 5000 条试试
