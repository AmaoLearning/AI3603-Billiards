import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

class AimNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 增加一点深度，加入 BatchNorm 加速收敛
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

def train_robust():
    df = pd.read_csv(os.path.join('data', 'clean_aim_data.csv'))
    
    X = df[['cut_angle', 'distance', 'V0']].values.astype(np.float32)
    y = df[['label_delta']].values.astype(np.float32)
    
    # 归一化 (Input Scaling)
    X[:, 0] /= 90.0
    X[:, 1] /= 2.0
    X[:, 2] /= 8.0
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
    
    model = AimNet()
    # 使用 Huber Loss，比 MSE 更抗噪声
    criterion = nn.HuberLoss(delta=0.1) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50)
    
    X_t = torch.tensor(X_train)
    y_t = torch.tensor(y_train)
    X_v = torch.tensor(X_val)
    y_v = torch.tensor(y_val)
    
    print("开始训练 (Robust Mode)...")
    for epoch in range(2000):
        model.train()
        optimizer.zero_grad()
        pred = model(X_t)
        loss = criterion(pred, y_t)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_v), y_v)
            
        scheduler.step(val_loss)
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss.item():.6f} | LR: {optimizer.param_groups[0]['lr']:.5f}")
            
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), os.path.join('checkpoints', 'robust_aim_model.pth'))
    print("训练完成。")

if __name__ == "__main__":
    train_robust()