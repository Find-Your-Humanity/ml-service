# train_autoencoder.py

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import os
import sys
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Hyperparameters via environment (for Kubeflow/Katib)
EPOCHS = int(os.getenv("EPOCHS", "50"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
LR = float(os.getenv("LR", "1e-3"))

# config 모듈 import를 위한 경로 추가 (src를 sys.path에 추가)
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent  # .../src
sys.path.append(str(src_dir))

from config.paths import get_model_file_path, ensure_directories

class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train(csv_path):
    df = pd.read_csv(csv_path)
    drop_cols = ['user_id', 'session_id', 'label'] if 'label' in df.columns else ['user_id', 'session_id']
    df = df.drop(columns=drop_cols, errors='ignore')

    # 컬럼 순서 저장
    feature_columns = df.columns.tolist()
    ensure_directories()
    joblib.dump(feature_columns, get_model_file_path("feature_columns.pkl"))

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    joblib.dump(scaler, get_model_file_path("scaler.pkl"))

    X = torch.tensor(scaled, dtype=torch.float32)
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = AutoEncoder(input_dim=X.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for xb in loader:
            xb = xb[0]
            recon = model(xb)
            loss = criterion(recon, xb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/50, Loss: {avg_loss:.6f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            # 모델 디렉토리 확보 후 저장
            ensure_directories()
            torch.save(model.state_dict(), get_model_file_path("model.pth"))

    model.eval()
    with torch.no_grad():
        recon = model(X)
        mse = torch.mean((X - recon)**2, dim=1).numpy()
    threshold = np.mean(mse) + 3 * np.std(mse)
    
    # threshold 파일 저장
    with open(get_model_file_path("threshold.txt"), "w") as f:
        f.write(str(threshold))
    print(f"✅ 최적 모델 저장 완료: {get_model_file_path('model.pth')}")
    print(f"✅ Threshold 저장 완료: {threshold:.6f}")
    # Katib metric (regex example: BEST_LOSS: ([0-9.]+))
    print(f"BEST_LOSS: {best_loss:.6f}")

if __name__ == "__main__":
    # 학습 데이터는 모델 디렉터리 내 CSV를 기본값으로 사용
    train(get_model_file_path("merged_session_basic_data.csv"))
