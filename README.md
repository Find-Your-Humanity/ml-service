# Real Captcha ML Service

Real Captcha의 **머신러닝 및 AI 기반 사용자 행동 분석 서비스**입니다. AutoEncoder 기반 봇 탐지, 행동 패턴 분석, 필기 OCR, Abstract 이미지 분류(keras .h5) 추론을 제공합니다.

## 🤖 **주요 기능**

### **행동 분석 시스템 (Behavior Analysis)**
- **AutoEncoder 기반 봇 탐지**: 정상 사용자 행동 패턴 학습 및 이상 탐지
- **실시간 행동 패턴 분석**: 마우스 움직임, 클릭, 타이밍 패턴 분석
- **신뢰도 스코어링**: 0-100점 스코어로 사용자 신뢰도 측정
- **동적 캡차 난이도 조절**: 스코어에 따른 적응형 캡차 제시

### **데이터 처리 파이프라인**
- **행동 데이터 수집**: 마우스, 클릭, 스크롤 이벤트 처리
- **특성 추출**: 속도, 가속도, 패턴 일관성 분석
- **데이터 정규화**: MinMaxScaler 기반 특성 정규화
- **모델 추론**: 실시간 봇 탐지 및 스코어 계산

### **모델 관리**
- **AutoEncoder 모델**: PyTorch 기반 이상 탐지 모델
- **임계값 관리**: 동적 임계값 조정 및 최적화
- **모델 재훈련**: 새로운 데이터로 지속적 학습

## 🏗️ **프로젝트 구조**

```
src/
├── behavior_analysis/           # 행동 분석 모듈
│   ├── __init__.py             # 모듈 패키지 설정
│   ├── inference_bot_detector.py   # 실시간 봇 탐지 추론
│   ├── train_autoencoder.py       # AutoEncoder 모델 훈련
│   ├── generate_data.py           # 합성 데이터 생성
│   ├── merge_basic.py             # 데이터 병합 및 전처리
│   └── models/                    # 훈련된 모델 저장소
│       ├── model_v3.pth            # AutoEncoder 모델 가중치
│       ├── scaler_v3.pkl           # 특성 정규화 스케일러
│       ├── feature_columns_v3.pkl  # 특성 컬럼 정보
│       └── threshold_v3.txt        # 봇 탐지 임계값
├── data/                          # 데이터 저장소
│   └── behavior_data/             # 행동 데이터 파일들
│       ├── behavior_data_*.json   # 수집된 행동 데이터
│       └── bot_sessions.json      # 봇 세션 데이터
└── api/                          # API 서비스
    ├── __init__.py
    └── app.py                    # FastAPI 엔드포인트 (/predict-bot, /predict-text, /predict-abstract-proba-batch)
```

## 🚀 **빠른 시작**

### **환경 설정**

#### 1. Python 가상환경 생성
```bash
cd backend/ml-service
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

#### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

**주요 패키지:**
- `torch>=1.9.0` - PyTorch (AutoEncoder 모델)
- `scikit-learn>=1.0.0` - 데이터 전처리 및 스케일링
- `pandas>=1.3.0` - 데이터 분석
- `numpy>=1.21.0` - 수치 계산
- `joblib>=1.1.0` - 모델 직렬화

### **데이터 준비**

#### 1. 행동 데이터 생성 (테스트용)
```bash
python src/behavior_analysis/generate_data.py
```

#### 2. 데이터 병합 및 전처리
```bash
python src/behavior_analysis/merge_basic.py
```

### **모델 훈련**

#### AutoEncoder 모델 훈련
```bash
python src/behavior_analysis/train_autoencoder.py
```

**훈련 과정:**
1. 행동 데이터 로드 및 특성 추출
2. 데이터 정규화 (MinMaxScaler)
3. AutoEncoder 네트워크 훈련
4. 최적 임계값 계산 및 저장
5. 모델 및 전처리기 저장

### **실시간 봇 탐지**

#### 추론 실행
```bash
python src/behavior_analysis/inference_bot_detector.py
```

**탐지 과정:**
1. 새로운 행동 데이터 로드
2. 특성 추출 및 정규화
3. AutoEncoder 재구성 오차 계산
4. 임계값 비교로 봇/인간 분류
5. 신뢰도 스코어 (0-100) 반환

## 📊 **AI 모델 상세**

### **AutoEncoder 아키텍처**
```python
class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)  # 병목층 (Bottleneck)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
```

### **특성 추출 (Feature Engineering)**

#### 마우스 행동 특성
- `avg_speed`: 평균 이동 속도
- `max_speed`: 최대 이동 속도
- `acceleration_changes`: 가속도 변화 횟수
- `direction_changes`: 방향 변화 횟수

#### 클릭 행동 특성
- `click_count`: 총 클릭 횟수
- `avg_click_duration`: 평균 클릭 지속 시간
- `click_intervals`: 클릭 간격 패턴

#### 타이밍 특성
- `total_time`: 총 소요 시간
- `pause_count`: 일시정지 횟수
- `idle_time`: 비활성 시간

### **스코어링 시스템**
```python
def calculate_trust_score(reconstruction_error, threshold):
    """
    재구성 오차를 기반으로 신뢰도 스코어 계산
    - 낮은 오차 → 높은 신뢰도 (인간)
    - 높은 오차 → 낮은 신뢰도 (봇)
    """
    if reconstruction_error <= threshold:
        # 정상 범위: 70-100점
        score = 70 + (threshold - reconstruction_error) / threshold * 30
    else:
        # 이상 범위: 0-70점
        score = max(0, 70 - (reconstruction_error - threshold) / threshold * 70)
    
    return min(100, max(0, score))
```

## 🔧 **설정 및 튜닝**

### **중앙 경로 관리**
모든 파일 경로는 `config/paths.py`를 통해 중앙 관리됩니다:

```python
from config.paths import get_model_file_path, get_data_file_path

# 모델 파일 로드
model_path = get_model_file_path("model_v3.pth")
threshold_path = get_model_file_path("threshold_v3.txt")

# 데이터 파일 로드
data_path = get_data_file_path("behavior_data_001.json")
```

### **환경변수 설정**
```bash
# 선택적 환경변수
export MODEL_DIR="/custom/model/path"    # 커스텀 모델 디렉토리
export DATA_DIR="/custom/data/path"      # 커스텀 데이터 디렉토리
```

## 🔮 **향후 개발 계획**

### **Phase 1: API 서비스 구현 (진행중)**
- FastAPI 기반 REST API 구축
- `/api/detect-bot` 실시간 봇 탐지 엔드포인트
- `/api/train-model` 모델 재훈련 엔드포인트

### **Phase 2: ImageNet 통합 (계획됨)**
- ImageNet 데이터셋 다운로드 및 전처리
- AI 기반 추상적 감정 분석 모델
- 동적 이미지 캡차 생성 API

### **Phase 3: 실시간 학습 (계획됨)**
- 온라인 학습 파이프라인
- 모델 성능 모니터링
- A/B 테스트 프레임워크

## 🧪 **테스트 및 평가**

### **모델 성능 테스트**
```bash
# 테스트 데이터로 정확도 평가
python -c "
from src.behavior_analysis.inference_bot_detector import detect_bot
result = detect_bot('src/data/behavior_data/test_data.json')
print(f'Detection Result: {result}')
"
```

### **벤치마크 메트릭**
- **정확도 (Accuracy)**: 전체 예측 중 정확한 예측 비율
- **정밀도 (Precision)**: 봇으로 예측한 것 중 실제 봇인 비율
- **재현율 (Recall)**: 실제 봇 중 올바르게 탐지한 비율
- **F1-Score**: 정밀도와 재현율의 조화 평균

## 🔒 **보안 및 개인정보 보호**

### **데이터 보안**
- 모든 행동 데이터는 익명화되어 처리
- 개인 식별 정보 수집 금지
- 로컬 처리 우선, 최소한의 서버 전송

### **모델 보안**
- 모델 가중치 암호화 저장
- API 엔드포인트 인증 및 권한 관리
- 요청 제한 (Rate Limiting) 적용

## 📈 **모니터링 및 로깅**

### **성능 모니터링**
- 모델 추론 시간 측정
- 메모리 사용량 모니터링
- 배치 처리 최적화

### **로깅 시스템**
```python
import logging
from config.paths import LOG_DIR

logging.basicConfig(
    filename=LOG_DIR / "ml_service.log",
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## 📚 **API 문서 (향후)**

### **봇 탐지 API**
```http
POST /api/detect-bot
Content-Type: application/json

{
  "behavior_data": {
    "mouse_movements": [...],
    "clicks": [...],
    "timing": {...}
  }
}

Response:
{
  "is_bot": false,
  "confidence_score": 85.6,
  "trust_level": "high",
  "next_captcha_type": "image"
}
```

## 📄 **라이선스**

MIT License - 자세한 내용은 `LICENSE` 파일을 참조하세요.

---

**Real Captcha ML Service v2.0.0**  
© 2025 Find Your Humanity. All rights reserved.
