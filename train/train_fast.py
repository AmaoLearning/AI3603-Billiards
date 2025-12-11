import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader # 新增
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# 检测是否有 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class AimNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 保持原有架构，BatchNorm 在 Mini-batch 训练中非常重要
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

def train_fast():
    # 读取数据 (注意文件名可能变了)
    csv_path = os.path.join('data', 'fast_aim_data.csv')
    if not os.path.exists(csv_path):
        print(f"错误：找不到文件 {csv_path}，请先运行数据生成脚本。")
        return

    df = pd.read_csv(csv_path)
    print(f"加载数据: {len(df)} 条")
    
    X = df[['cut_angle', 'distance', 'V0']].values.astype(np.float32)
    y = df[['label_delta']].values.astype(np.float32)
    
    # --- 归一化 (Input Scaling) ---
    # 重要提示：这里的参数必须与 Agent 推理时完全一致
    X[:, 0] /= 90.0  # 切角
    X[:, 1] /= 2.0   # 距离
    X[:, 2] /= 8.0   # 力度 (之前是 V0/10.0，这里改为8.0以匹配生成脚本的范围)
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # 转为 Tensor 并移至设备
    X_t = torch.tensor(X_train).to(device)
    y_t = torch.tensor(y_train).to(device)
    X_v = torch.tensor(X_val).to(device)
    y_v = torch.tensor(y_val).to(device)

    # --- 关键修改 1: 使用 DataLoader 进行 Batch 训练 ---
    batch_size = 256 # 较大的 Batch 有助于抵消噪声
    train_ds = TensorDataset(X_t, y_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    model = AimNet().to(device)
    
    # --- 关键修改 2: 换回 MSELoss ---
    # MSE 会强迫模型预测噪声分布的"中心"，这正是我们要的
    criterion = nn.MSELoss() 
    
    # 学习率稍大一点，配合 Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)
    
    print(f"开始训练 (Fast/Batch Mode) - Epochs: 1000, Batch: {batch_size}")
    
    for epoch in range(1001):
        model.train()
        epoch_loss = 0
        
        # Mini-batch 循环
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # 计算平均 Train Loss
        avg_train_loss = epoch_loss / len(train_loader)

        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(X_v)
            val_loss = criterion(val_pred, y_v)
            
        # 更新学习率
        scheduler.step(val_loss)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss.item():.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
            
    os.makedirs('checkpoints', exist_ok=True)
    save_path = os.path.join('checkpoints', 'aim_model.pth') # 注意这里保存名字要和Agent加载的一致
    torch.save(model.state_dict(), save_path)
    print(f"训练完成，模型已保存至: {save_path}")

if __name__ == "__main__":
    train_fast()