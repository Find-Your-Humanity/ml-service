# inference_bot_detector.py
import torch
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.preprocessing import MinMaxScaler
from src.config.paths import get_model_file_path


class AutoEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=32, latent_dim=16, dropout_rate=0.0):
        super(AutoEncoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def extract_features_from_json(path):
    with open(path, "r") as f:
        data = json.load(f)[0]

    mouse = pd.DataFrame(data.get("mouseMovements", []))
    clicks = pd.DataFrame(data.get("mouseClicks", []))

    if len(mouse) == 0:
        return None

    mouse["dt"] = mouse["timestamp"].diff().fillna(0)

    dx = mouse["x"].diff().fillna(0)
    dy = mouse["y"].diff().fillna(0)
    distance = np.sqrt(dx**2 + dy**2)
    speed = distance / mouse["dt"].replace(0, np.nan)

    click_counts = {
        "click_count": len(clicks),
        "mousedown_count": 0,
        "mouseup_count": 0,
    }
    if not clicks.empty and "type" in clicks.columns:
        click_counts["mousedown_count"] = (clicks["type"] == "mousedown").sum()
        click_counts["mouseup_count"] = (clicks["type"] == "mouseup").sum()
        for t in clicks["type"].unique():
            click_counts[f"click_type_{t}"] = (clicks["type"] == t).sum()

    summary = {
        "total_distance": distance.sum(),
        "average_speed": speed.mean(),
        "max_speed": speed.max(),
        "min_speed": speed.min(),
        "std_speed": speed.std(),
        "total_duration": mouse["timestamp"].iloc[-1] - mouse["timestamp"].iloc[0],
        "movement_count": len(mouse),
        "pause_count": (speed < 5).sum()
    }
    summary.update(click_counts)

    # 누락된 클릭 타입 기본 0으로 추가
    for col in ["click_type_click", "click_type_mousedown", "click_type_mouseup"]:
        if col not in summary:
            summary[col] = 0

    return summary

def detect_bot(json_path):
    feat = extract_features_from_json(json_path)
    
    # None 반환 시 처리
    if feat is None:
        return {
            "score": 0.0,
            "mse": float('inf'),
            "threshold": 0.0,
            "dynamic_threshold": 0,
            "is_bot": True,
            "features": {},
            "error": "No mouse movement data found"
        }
    
    df = pd.DataFrame([feat])

    # ✅ feature_columns 불러오기
    feature_columns = joblib.load(get_model_file_path("feature_columns.pkl"))

    # ✅ 누락된 컬럼은 0으로 채우고, 순서 맞추기
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]

    scaler = joblib.load(get_model_file_path("scaler.pkl"))
    scaled = scaler.transform(df)
    x = torch.tensor(scaled, dtype=torch.float32)

    # 그리드 서치 결과의 최적 파라미터 사용
    model = AutoEncoder(input_dim=x.shape[1], hidden_dim=64, latent_dim=32, dropout_rate=0.0)
    model.load_state_dict(torch.load(get_model_file_path("model.pth")))
    model.eval()

    with open(get_model_file_path("threshold.txt"), "r") as f:
        threshold = float(f.read())

    with torch.no_grad():
        recon = model(x)
        mse = torch.mean((x - recon)**2, dim=1).item()

    # 개선된 점수 계산 방식
    # MSE가 매우 클 경우를 대비한 로그 스케일링 적용
    if mse > threshold:
        # MSE가 threshold보다 클 때는 로그 비율로 점수 계산
        ratio = mse / threshold
        if ratio > 1000:  # 매우 큰 차이일 때
            score = max(0, 100 * (1 - np.log10(ratio) / 10))  # 로그 스케일링
        else:
            score = max(0, 100 * (1 - ratio / 100))  # 선형 스케일링 (완화됨)
    else:
        # 원래 공식 사용
        score = max(0, 100 * (1 - (mse / threshold)))
    
    # 간단한 고정 임계값 사용
    is_bot = score < 50

    # NumPy 타입 -> Python 기본 타입으로 변환
    feat_serialized = {k: (v.item() if isinstance(v, (np.integer, np.floating)) else v)
                       for k, v in feat.items()}

    return {
        "score": round(float(score), 2),
        "mse": round(float(mse), 6),
        "threshold": round(float(threshold), 6),
        "is_bot": bool(is_bot),
        "features": feat_serialized
    }

if __name__ == "__main__":
    result = detect_bot("/Users/kang-yeongmo/userdata/behavior_data_20250731_105045.json")
    print(json.dumps(result, indent=2, ensure_ascii=False))
