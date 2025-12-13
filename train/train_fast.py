import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AimNet(nn.Module):
    def __init__(self):
        super().__init__()
        # [改进1]: 加宽网络，增加拟合能力
        self.net = nn.Sequential(
            nn.Linear(3, 256),        # 输入 -> 256
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, 128),      # 256 -> 128
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, 64),       # 128 -> 64
            nn.ReLU(),
            
            nn.Linear(64, 1)          # 输出
        )
        
        # [改进2]: 初始化权重 (He Initialization)，防止开局梯度消失
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)

def train_fast():
    csv_path = os.path.join('data', 'fast_aim_data.csv')
        
    print(f"读取数据: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # --- [诊断步骤] 检查数据分布 ---
    print("数据统计摘要:")
    print(df.describe())
    label_var = df['label_delta'].var()
    print(f"Label 方差 (Baseline MSE): {label_var:.6f}")
    # 如果训练 Loss 无法低于这个方差，说明模型无效
    
    X = df[['cut_angle', 'distance', 'V0']].values.astype(np.float32)
    y = df[['label_delta']].values.astype(np.float32)
    
    X[:, 0] /= df['cut_angle'].max()  
    X[:, 1] /= df['distance'].max()
    X[:, 2] /= df['V0'].max()
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    
    X_t = torch.tensor(X_train).to(device)
    y_t = torch.tensor(y_train).to(device)
    X_v = torch.tensor(X_val).to(device)
    y_v = torch.tensor(y_val).to(device)

    # [改进3]: 增大 Batch Size
    batch_size = 512 
    train_ds = TensorDataset(X_t, y_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    model = AimNet().to(device)
    criterion = nn.MSELoss()
    
    # [改进4]: 初始 LR 设大，交给 OneCycleLR 管理
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    
    epochs = 500 # 500轮足够了，不用1000
    
    # [改进5]: 使用 OneCycleLR 替代 ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,           # 允许冲到的最大 LR
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,         # 30% 的时间用来热身 (Warmup)
        div_factor=25.0,
        final_div_factor=10000.0
    )
    
    print(f"开始训练 (OneCycleLR Mode) - Epochs: {epochs}")
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            scheduler.step() # 每个 Batch 更新一次 LR
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_v)
            val_loss = criterion(val_pred, y_v)
            
        current_lr = optimizer.param_groups[0]['lr']
        
        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), os.path.join('checkpoints', 'aim_model.pth'))
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:4d} | Train: {avg_train_loss:.6f} | Val: {val_loss.item():.6f} | Best: {best_val_loss:.6f} | LR: {current_lr:.6f}")

    print(f"训练结束。最优 Val Loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    train_fast()