# inference_bot_detector.py
import torch
import pandas as pd
import numpy as np
import json
import joblib
from src.config.paths import get_model_file_path

# ==============================================
# 기존 AutoEncoder 기반 로직 (주석 처리 - 보존)
# from sklearn.preprocessing import MinMaxScaler
# class AutoEncoder(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim=128, latent_dim=8, dropout_rate=0.1):
#         super(AutoEncoder, self).__init__()
#         self.encoder = torch.nn.Sequential(
#             torch.nn.Linear(input_dim, hidden_dim),
#             torch.nn.ReLU(),
#             torch.nn.Dropout(dropout_rate),
#             torch.nn.Linear(hidden_dim, latent_dim)
#         )
#         self.decoder = torch.nn.Sequential(
#             torch.nn.Linear(latent_dim, hidden_dim),
#             torch.nn.ReLU(),
#             torch.nn.Dropout(dropout_rate),
#             torch.nn.Linear(hidden_dim, input_dim)
#         )
#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded
# ==============================================

# ----------------------------------------------
# 사람/봇 이진분류 MLP (best_model.pt) 로더
# 학습 스크립트의 구조와 동일하게 정의
class MLP(torch.nn.Module):
    def __init__(self, input_features: int):
        super().__init__()
        self.layer1 = torch.nn.Linear(input_features, 64)
        self.layer2 = torch.nn.Linear(64, 32)
        self.output_layer = torch.nn.Linear(32, 1)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        return self.output_layer(x)
# ----------------------------------------------

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
    print(f"🔍 [DEBUG] detect_bot called with: {json_path}")
    
    feat = extract_features_from_json(json_path)
    print(f"🔍 [DEBUG] Extracted features: {feat}")
    
    # None 반환 시 처리
    if feat is None:
        print("❌ [DEBUG] No features extracted")
        return {
            "score": 0.0,
            "mse": 999999.0,  # float('inf') 대신 큰 값 사용
            "threshold": 0.0,
            "dynamic_threshold": 0,
            "is_bot": True,
            "features": {},
            "error": "No mouse movement data found"
        }
    
    df = pd.DataFrame([feat])
    print(f"🔍 [DEBUG] DataFrame created: {df.shape}")

    # ✅ (분류 모델용) 피처 정렬/일치화
    # 주의: 학습 시 사용한 컬럼 목록이 별도 파일로 제공되지 않은 경우
    #       스케일러가 기대하는 피처 수(n_features_in_)에 맞추어 DataFrame 컬럼을 정렬만 수행합니다.
    #       운영에서는 학습 시 feature_columns 를 별도 파일로 저장/로딩하는 것을 권장합니다.

    # 스케일러 적용: 컬럼을 알파벳 순으로 정렬 후 scaler.transform 적용
    df_sorted = df.reindex(sorted(df.columns), axis=1)
    try:
        scaler_path = get_model_file_path("scaler.joblib")
        print(f"🔍 [DEBUG] Loading scaler from: {scaler_path}")
        scaler = joblib.load(scaler_path)
        print("🔍 [DEBUG] Scaler loaded successfully")
        scaled = scaler.transform(df_sorted)
    except Exception as e:
        print(f"❌ [DEBUG] Scaler load/transform failed, fallback to raw features: {e}")
        scaled = df_sorted.values
    x = torch.tensor(scaled, dtype=torch.float32)
    print(f"🔍 [DEBUG] Scaled data shape: {x.shape}")

    # 분류 모델 로딩 (best_model.pt)
    try:
        model_path = get_model_file_path("best_model.pt")
        print(f"🔍 [DEBUG] Loading classifier model from: {model_path}")
        model = MLP(input_features=x.shape[1])
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        print(f"🔍 [DEBUG] Classifier model loaded successfully")
    except Exception as e:
        print(f"❌ [DEBUG] Error loading classifier model: {e}")
        raise

    # 분류 확률 예측 (시그모이드 → 확률)
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).item()  # 0~1
    score = round(float(prob * 100.0), 2)    # 0~100
    # 임계값: 0.5 (필요시 환경변수/설정으로 외부화 가능)
    is_bot = bool(prob >= 0.5)
    print(f"🔍 [DEBUG] Classifier prob={prob:.4f}, score={score}, is_bot={is_bot}")

    # NumPy 타입 -> Python 기본 타입으로 변환 (무한대/NaN 값 처리)
    def safe_convert(value):
        if isinstance(value, (np.integer, np.floating)):
            if np.isinf(value) or np.isnan(value):
                return 0.0
            return value.item()
        return value
    
    feat_serialized = {k: safe_convert(v) for k, v in feat.items()}

    # JSON 직렬화를 위해 무한대 값 처리
    def safe_float(value):
        if np.isinf(value) or np.isnan(value):
            return 0.0
        return round(float(value), 6)

    return {
        "score": score,
        "is_bot": is_bot,
        "features": feat_serialized
    }

if __name__ == "__main__":
    result = detect_bot("/Users/kang-yeongmo/userdata/behavior_data_20250731_105045.json")
    print(json.dumps(result, indent=2, ensure_ascii=False))
