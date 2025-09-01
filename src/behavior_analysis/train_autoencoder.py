# train_autoencoder.py

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import itertools
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, latent_dim=16, dropout_rate=0.0):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_with_params(X_train, X_val, train_loader, val_loader, params, verbose=False):
    """주어진 파라미터로 모델 학습"""
    model = AutoEncoder(
        input_dim=X_train.shape[1],
        hidden_dim=params['hidden_dim'],
        latent_dim=params['latent_dim'],
        dropout_rate=params['dropout_rate']
    )
    
    criterion = nn.MSELoss()
    if params['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    elif params['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'])
    else:  # RMSprop
        optimizer = torch.optim.RMSprop(model.parameters(), lr=params['learning_rate'])

    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(params['max_epochs']):
        # Training
        model.train()
        train_loss = 0.0
        for xb in train_loader:
            xb = xb[0]
            recon = model(xb)
            loss = criterion(recon, xb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb in val_loader:
                xb = xb[0]
                recon = model(xb)
                loss = criterion(recon, xb)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        
        if verbose and epoch % 10 == 0:
            print(f"    Epoch {epoch+1}, Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= params['patience']:
            break
    
    # 최적 모델로 복원
    model.load_state_dict(best_model_state)
    
    # Validation threshold 계산
    model.eval()
    with torch.no_grad():
        val_recon = model(X_val)
        val_mse = torch.mean((X_val - val_recon)**2, dim=1).numpy()
    
    threshold = np.mean(val_mse) + params['threshold_factor'] * np.std(val_mse)
    
    return model, best_val_loss, threshold

def grid_search_hyperparameters(csv_path):
    """그리드 서치로 하이퍼파라미터 튜닝"""
    
    # 데이터 로드 및 전처리
    df = pd.read_csv(csv_path)
    drop_cols = ['user_id', 'session_id', 'label'] if 'label' in df.columns else ['user_id', 'session_id']
    df = df.drop(columns=drop_cols, errors='ignore')
    
    # 컬럼 순서 저장
    feature_columns = df.columns.tolist()
    joblib.dump(feature_columns, "feature_columns.pkl")
    
    # 7:3으로 train/validation 분할
    train_df, val_df = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)
    
    # 스케일링
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df)
    val_scaled = scaler.transform(val_df)
    joblib.dump(scaler, "scaler.pkl")
    
    # 텐서 변환
    X_train = torch.tensor(train_scaled, dtype=torch.float32)
    X_val = torch.tensor(val_scaled, dtype=torch.float32)
    
    print(f"📊 데이터 분할: Train {len(train_df)}, Validation {len(val_df)}")
    print(f"📊 입력 차원: {X_train.shape[1]}")
    
    # 하이퍼파라미터 그리드 정의
    param_grid = {
        'hidden_dim': [32, 64, 128],
        'latent_dim': [8, 16, 32],
        'learning_rate': [1e-4, 1e-3, 5e-3],
        'batch_size': [16, 32, 64],
        'dropout_rate': [0.0, 0.1, 0.2],
        'optimizer': ['Adam', 'AdamW'],
        'threshold_factor': [2.5, 3.0, 3.5],
        'max_epochs': [50],
        'patience': [10]
    }
    
    # 모든 조합 생성
    keys = param_grid.keys()
    values = param_grid.values()
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"🔍 총 {len(param_combinations)}개 파라미터 조합 테스트 시작...")
    
    best_val_loss = float('inf')
    best_params = None
    best_model = None
    best_threshold = None
    results = []
    
    for i, params in enumerate(param_combinations):
        print(f"\n[{i+1}/{len(param_combinations)}] 테스트 중...")
        print(f"  파라미터: {params}")
        
        try:
            # 데이터 로더 생성 (배치 크기에 따라)
            train_dataset = TensorDataset(X_train)
            val_dataset = TensorDataset(X_val)
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
            
            # 모델 학습
            model, val_loss, threshold = train_with_params(X_train, X_val, train_loader, val_loader, params)
            
            print(f"  결과: Val Loss = {val_loss:.6f}, Threshold = {threshold:.6f}")
            
            # 결과 기록
            result = params.copy()
            result['val_loss'] = val_loss
            result['threshold'] = threshold
            results.append(result)
            
            # 최적 모델 업데이트
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = params.copy()
                best_model = model
                best_threshold = threshold
                print(f"  ✅ 새로운 최적 모델! (Val Loss: {val_loss:.6f})")
                
        except Exception as e:
            print(f"  ❌ 오류 발생: {e}")
            continue
    
    # 최적 모델 저장
    if best_model is not None:
        torch.save(best_model.state_dict(), "model.pth")
        with open("threshold.txt", "w") as f:
            f.write(str(best_threshold))
        
        print(f"\n🎯 그리드 서치 완료!")
        print(f"📈 최적 성능: Validation Loss = {best_val_loss:.6f}")
        print(f"📊 최적 파라미터:")
        for key, value in best_params.items():
            print(f"   {key}: {value}")
        print(f"🎯 최적 Threshold: {best_threshold:.6f}")
        
        # 결과를 CSV로 저장
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('val_loss')
        results_df.to_csv("grid_search_results.csv", index=False)
        print(f"📁 상세 결과 저장: grid_search_results.csv")
        
        return best_model, best_params, best_threshold
    else:
        print("❌ 모든 파라미터 조합에서 오류 발생")
        return None, None, None

def train(csv_path):
    """기본 학습 함수 (기존 방식)"""
    df = pd.read_csv(csv_path)
    drop_cols = ['user_id', 'session_id', 'label'] if 'label' in df.columns else ['user_id', 'session_id']
    df = df.drop(columns=drop_cols, errors='ignore')

    # 컬럼 순서 저장
    feature_columns = df.columns.tolist()
    joblib.dump(feature_columns, "feature_columns.pkl")

    # 7:3으로 train/validation 분할
    train_df, val_df = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)
    
    # 스케일링 (train set 기준으로 fit)
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df)
    val_scaled = scaler.transform(val_df)
    joblib.dump(scaler, "scaler.pkl")

    # 텐서 변환
    X_train = torch.tensor(train_scaled, dtype=torch.float32)
    X_val = torch.tensor(val_scaled, dtype=torch.float32)
    
    # 데이터 로더
    train_dataset = TensorDataset(X_train)
    val_dataset = TensorDataset(X_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 모델 초기화
    model = AutoEncoder(input_dim=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"📊 데이터 분할: Train {len(train_df)}, Validation {len(val_df)}")
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(50):  # 최대 에포크 증가
        # Training
        model.train()
        train_loss = 0.0
        for xb in train_loader:
            xb = xb[0]
            recon = model(xb)
            loss = criterion(recon, xb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb in val_loader:
                xb = xb[0]
                recon = model(xb)
                loss = criterion(recon, xb)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/50, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Early stopping 및 최적 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "model.pth")
            patience_counter = 0
            print(f"✅ 최적 모델 저장 (Val Loss: {avg_val_loss:.6f})")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"⏹️ Early stopping at epoch {epoch+1}")
            break

    # 최적 모델 로드하여 threshold 계산 (validation set 기준)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    with torch.no_grad():
        val_recon = model(X_val)
        val_mse = torch.mean((X_val - val_recon)**2, dim=1).numpy()
    
    threshold = np.mean(val_mse) + 3 * np.std(val_mse)
    with open("threshold.txt", "w") as f:
        f.write(str(threshold))
    
    print(f"🎯 최종 결과:")
    print(f"   최적 Validation Loss: {best_val_loss:.6f}")
    print(f"   Threshold (μ + 3σ): {threshold:.6f}")
    print(f"   모델 저장: model.pth")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--grid-search":
        print("🔍 그리드 서치 모드로 실행")
        grid_search_hyperparameters("merged_session_basic_data.csv")
    else:
        print("📚 기본 학습 모드로 실행")
        print("💡 그리드 서치를 원한다면: python train_autoencoder.py --grid-search")
        train("merged_session_basic_data.csv")
