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
    """
    사용자가 제공한 학습/추론 스펙에 맞춘 11개 피처를 계산합니다.
    - duration, movement_count, click_count,
      total_distance, avg_velocity, std_velocity, max_velocity, avg_acceleration,
      straightness, avg_angle_change, std_angle_change
    """
    with open(path, "r") as f:
        data = json.load(f)
    # 사용자 스크립트는 단일 세션 dict를 기대하지만
    # 이 프로젝트의 detect_bot는 [dict] 리스트를 넘기므로 양쪽 모두 허용
    if isinstance(data, list) and len(data) > 0:
        data = data[0]

    movements = data.get("mouseMovements", []) or []
    clicks = data.get("mouseClicks", []) or []
    page_events = data.get("pageEvents", {}) or {}

    if len(movements) < 2:
        return None

    import math
    features = {}
    # duration: 마지막 움직임 시각 - 페이지 입장 혹은 첫 움직임 시각
    features["duration"] = movements[-1].get("timestamp", 0) - (page_events.get("enterTime") or movements[0].get("timestamp", 0))
    features["movement_count"] = len(movements)
    features["click_count"] = len([c for c in clicks if c.get("type") == "click"])

    distances, velocities, accelerations, time_deltas = [], [], [], []
    for i in range(len(movements) - 1):
        p1, p2 = movements[i], movements[i + 1]
        dx = (p2.get("x", 0) - p1.get("x", 0))
        dy = (p2.get("y", 0) - p1.get("y", 0))
        dist = math.sqrt(dx * dx + dy * dy)
        dt = (p2.get("timestamp", 0) - p1.get("timestamp", 0)) / 1000.0
        if dt > 0:
            distances.append(dist)
            time_deltas.append(dt)
            velocities.append(dist / dt)

    if not velocities:
        return None

    for i in range(len(velocities) - 1):
        v1, v2 = velocities[i], velocities[i + 1]
        dt = time_deltas[i + 1] if i + 1 < len(time_deltas) else None
        if dt and dt > 0:
            accelerations.append((v2 - v1) / dt)

    features["total_distance"] = float(sum(distances))
    features["avg_velocity"] = float(np.mean(velocities)) if velocities else 0.0
    features["std_velocity"] = float(np.std(velocities)) if velocities else 0.0
    features["max_velocity"] = float(np.max(velocities)) if velocities else 0.0
    features["avg_acceleration"] = float(np.mean(accelerations)) if accelerations else 0.0

    start_point, end_point = movements[0], movements[-1]
    dx_se = (end_point.get("x", 0) - start_point.get("x", 0))
    dy_se = (end_point.get("y", 0) - start_point.get("y", 0))
    straight_line_dist = math.sqrt(dx_se * dx_se + dy_se * dy_se)
    total_distance = features["total_distance"]
    features["straightness"] = float(straight_line_dist / total_distance) if total_distance > 0 else 1.0

    angles = []
    for i in range(len(movements) - 2):
        p1, p2, p3 = movements[i], movements[i + 1], movements[i + 2]
        v1 = (p2.get("x", 0) - p1.get("x", 0), p2.get("y", 0) - p1.get("y", 0))
        v2 = (p3.get("x", 0) - p2.get("x", 0), p3.get("y", 0) - p2.get("y", 0))
        mag_v1 = math.sqrt(v1[0] * v1[0] + v1[1] * v1[1])
        mag_v2 = math.sqrt(v2[0] * v2[0] + v2[1] * v2[1])
        if mag_v1 * mag_v2 > 0:
            cos_val = (v1[0] * v2[0] + v1[1] * v2[1]) / (mag_v1 * mag_v2)
            cos_val = float(np.clip(cos_val, -1.0, 1.0))
            angle = math.degrees(math.acos(cos_val))
            angles.append(angle)
    features["avg_angle_change"] = float(np.mean(angles)) if angles else 0.0
    features["std_angle_change"] = float(np.std(angles)) if angles else 0.0

    return features

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

    # ✅ (분류 모델용) 피처 일치화: scaler.feature_names_in_ 기준으로 정확히 맞추기

    try:
        scaler_path = get_model_file_path("scaler.joblib")
        print(f"🔍 [DEBUG] Loading scaler from: {scaler_path}")
        scaler = joblib.load(scaler_path)
        print("🔍 [DEBUG] Scaler loaded successfully")
        feature_order = None
        if hasattr(scaler, "feature_names_in_"):
            feature_order = list(scaler.feature_names_in_)
            # 누락된 컬럼은 0으로 채우고, 초과된 컬럼은 제거
            for col in feature_order:
                if col not in df.columns:
                    df[col] = 0
            df_aligned = df[feature_order]
        else:
            # 백업: 알파벳 정렬
            df_aligned = df.reindex(sorted(df.columns), axis=1)
        scaled = scaler.transform(df_aligned)
    except Exception as e:
        print(f"❌ [DEBUG] Scaler load/transform failed, fallback to raw features: {e}")
        scaled = df.values
    x = torch.tensor(scaled, dtype=torch.float32)
    print(f"🔍 [DEBUG] Scaled data shape: {x.shape}")

    # 분류 모델 로딩 (best_model.pt)
    try:
        model_path = get_model_file_path("best_model.pt")
        print(f"🔍 [DEBUG] Loading classifier model from: {model_path}")
        input_dim = scaled.shape[1]
        model = MLP(input_features=input_dim)
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
