# Real Captcha ML Service

Real Captcha의 **머신러닝 및 AI 기반 사용자 행동 분석 서비스**입니다. AutoEncoder 기반 봇 탐지, 행동 패턴 분석, 그리고 향후 ImageNet 기반 이미지 생성을 담당합니다.

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
│       ├── model.pth              # AutoEncoder 모델 가중치
│       ├── scaler.pkl             # 특성 정규화 스케일러
│       ├── feature_columns.pkl    # 특성 컬럼 정보
│       └── threshold.txt          # 봇 탐지 임계값
├── data/                          # 데이터 저장소
│   └── behavior_data/             # 행동 데이터 파일들
│       ├── behavior_data_*.json   # 수집된 행동 데이터
│       └── bot_sessions.json      # 봇 세션 데이터
└── api/                          # API 서비스 (향후 구현)
    ├── __init__.py
    ├── bot_detection_api.py      # 봇 탐지 API 엔드포인트
    └── image_generation_api.py   # 이미지 생성 API (ImageNet)
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
model_path = get_model_file_path("model.pth")
threshold_path = get_model_file_path("threshold.txt")

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


---

## 📓 Kubeflow Jupyter에서 행동 분석 모델 학습 가이드 (Behavior Analysis Training on Kubeflow)

본 섹션은 Kubeflow의 Jupyter Notebook에서 backend/ml-service의 행동 분석(AutoEncoder 기반) 모델을 학습하고, 학습된 모델을 서빙하여 외부에서 테스트하는 전 과정을 설명합니다.

### 0) 사전 준비사항
- Kubeflow가 배포되어 있고 Jupyter Notebook 서버를 생성/접근할 권한이 있어야 합니다.
- 클러스터에 다음 리소스가 존재한다고 가정합니다.
  - 네임스페이스: `captcha`
  - 모델 저장용 PVC: `ml-models-pvc` (deploy-manifests/ml-service/pvc-ml-models.yaml)
  - 서빙 매니페스트: Deployment/Service/Ingress (deploy-manifests/ml-service/*)
- Ingress 도메인: `ml.realcatcha.com`

### 1) Jupyter Notebook 서버 생성
1. Kubeflow Central Dashboard 접속 → Notebooks → New Server
2. Image 선택 (Spawner UI 기준)
   - 권장: `ghcr.io/kubeflow/kubeflow/notebook-servers/jupyter-pytorch-full:v1.10.0`
   - GPU가 있으면: `.../jupyter-pytorch-cuda-full:v1.10.0`
3. 자원 설정(예시)
   - CPU: 2 vCPU, Memory: 4~8Gi
   - GPU가 필요 없다면 0
4. Data Volumes에 모델 PVC 마운트 추가
   - Existing PVC: `ml-models-pvc`
   - Mount Path: `/home/jovyan/models` (임의 경로 가능)
5. Create → Notebook 접속

참고: Notebooks UI 스포너 기본값은 manifests-1.10.2/applications/jupyter/.../spawner_ui_config.yaml을 따릅니다. 파이토치 계열 이미지를 선택하세요.

### 2) 코드 준비 및 환경 설치
Notebook 터미널(또는 셀)에서 다음을 실행합니다.

```bash
# 1) 저장소 클론 (이미 볼륨에 있다면 생략)
cd /home/jovyan
git clone https://your.git.server/realcatcha.git
cd realcatcha/backend/ml-service

# 2) 파이썬 의존성 설치
# repo에 requirements.txt가 없을 경우 아래 패키지들을 설치하세요.
pip install fastapi==0.104.1 uvicorn[standard]==0.24.0 \
            torch==2.1.1 scikit-learn==1.3.2 pandas==2.1.4 \
            numpy==1.25.2 joblib==1.3.2
# (requirements.txt가 존재하면)
# pip install -r requirements.txt
```

### 3) 학습 데이터 확인
- 예시 CSV가 저장되어 있습니다.
  - 경로: `backend/ml-service/src/behavior_analysis/models/merged_session_basic_data.csv`
- 학습 스크립트는 중앙 경로(`src/config/paths.py`)를 사용해 위 경로의 CSV를 자동으로 읽습니다. 현재 작업 디렉터리에 의존하지 않습니다.

### 4) 모델 학습 실행
```bash
# 어디서 실행해도 됩니다 (권장: ml-service 루트)
cd /home/jovyan/realcatcha/backend/ml-service
python src/behavior_analysis/train_autoencoder.py
```
예상 산출물(항상 `src/behavior_analysis/models`에 생성):
- model.pth
- threshold.txt
- scaler.pkl
- feature_columns.pkl

정상적으로 완료되면 “✅ 최적 모델 저장 완료”, “✅ Threshold 저장 완료” 로그가 출력됩니다.

### 5) 학습 산출물(PVC 저장) 정리
서빙 컨테이너에서 모델 파일을 읽을 수 있도록, Notebook에서 마운트한 PVC 경로에도 백업을 남기는 것을 권장합니다.

```bash
# Notebook에서 PVC를 /home/jovyan/models 로 마운트했다고 가정
cp model.pth threshold.txt scaler.pkl feature_columns.pkl /home/jovyan/models/
ls -al /home/jovyan/models
```

필요시 버전 관리용으로 하위 폴더를 두고 날짜를 붙여 관리하세요.

---

## 🛰️ 모델 서빙 및 외부 테스트 가이드

현재 저장소에는 FastAPI 기반 API 서버가 포함되어 있습니다.
- 엔드포인트: POST `/predict-bot`
- 구현 파일: `backend/ml-service/src/api/app.py`
- 내부 로직은 `src/behavior_analysis/inference_bot_detector.py`의 `detect_bot`을 호출합니다.

중요: inference 코드는 모델 파일을 `src/behavior_analysis/models` 경로에서 찾습니다(`src/config/paths.py` 참고). 즉, 런타임 컨테이너 내 해당 경로에 파일이 존재해야 합니다.

### A) 기존 배포 매니페스트(Deployment/Service/Ingress) 사용
리포지토리 제공 매니페스트:
- Deployment: `deploy-manifests/ml-service/deployment-ml-service.yaml`
- Service: `deploy-manifests/ml-service/service-ml-service.yaml`
- Ingress: `deploy-manifests/ml-service/ml-service-ingress.yaml` (호스트: `ml.realcatcha.com`)
- PVC: `deploy-manifests/ml-service/pvc-ml-models.yaml` (이름: ml-models-pvc)

기본 Deployment는 PVC를 `/models`에 마운트합니다. 하지만 애플리케이션 코드는 `/app/src/behavior_analysis/models`를 바라봅니다. 다음 중 하나의 방법을 선택하세요.

방법 1. 마운트 경로를 코드 경로에 맞추기(권장)
- deployment-ml-service.yaml의 volumeMounts를 아래처럼 수정:
```yaml
volumeMounts:
  - name: model-storage
    mountPath: /app/src/behavior_analysis/models
    readOnly: true
```
이렇게 하면 컨테이너가 기동될 때, PVC에 저장해 둔 `model.pth`, `threshold.txt`, `scaler.pkl`, `feature_columns.pkl`을 그대로 로딩할 수 있습니다.

방법 2. 기존 `/models` 유지하되, postStart 훅이나 엔트리포인트에서 복사
- 컨테이너 시작 스크립트에서 `/models/*.pth|*.pkl|threshold.txt`를 `/app/src/behavior_analysis/models/`로 복사합니다.
- 예: `cp /models/* /app/src/behavior_analysis/models/`

방법 3. 이미지를 새로 빌드할 때 모델 파일을 이미지에 포함
- CI/CD에서 `src/behavior_analysis/models` 경로로 산출물을 복사하여 이미지를 빌드합니다.

헬스체크 주의사항
- 현재 Deployment의 liveness/readinessProbe는 `/health`를 조회하도록 설정되어 있습니다.
- FastAPI 앱에 `/health` 엔드포인트를 추가하거나(권장), 프로브 경로를 존재하는 경로(`/`, `/docs`)로 변경하세요.

### B) 클러스터에 적용
```bash
# 네임스페이스 확인
kubectl get ns captcha

# PVC(필요시)
kubectl apply -f deploy-manifests/ml-service/pvc-ml-models.yaml

# 모델 파일이 PVC에 있는지 확인(예: node에서 또는 임시 Pod로 마운트해서 확인)
# kubectl -n captcha exec -it <any-pod> -- ls -al /models

# Deployment/Service/Ingress 적용
kubectl apply -f deploy-manifests/ml-service/deployment-ml-service.yaml
kubectl apply -f deploy-manifests/ml-service/service-ml-service.yaml
kubectl apply -f deploy-manifests/ml-service/ml-service-ingress.yaml

# 상태 확인
kubectl -n captcha get deploy,po,svc,ingress | grep ml-service
```

### C) 외부에서 예측 API 테스트
Ingress가 준비되면 다음과 같이 호출합니다.

요청 스키마(요약):
```json
{
  "behavior_data": {
    "mouseMovements": [
      {"x": 100, "y": 230, "timestamp": 1718850001000},
      {"x": 102, "y": 231, "timestamp": 1718850001100}
    ],
    "mouseClicks": [
      {"type": "mousedown", "timestamp": 1718850001200},
      {"type": "mouseup",   "timestamp": 1718850001210}
    ]
  }
}
```

curl 예시:
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "behavior_data": {
      "mouseMovements": [
        {"x": 100, "y": 230, "timestamp": 1718850001000},
        {"x": 102, "y": 231, "timestamp": 1718850001100}
      ],
      "mouseClicks": [
        {"type": "mousedown", "timestamp": 1718850001200},
        {"type": "mouseup",   "timestamp": 1718850001210}
      ]
    }
  }' \
  https://ml.realcatcha.com/predict-bot
```

Python 예시:
```python
import requests

url = "https://ml.realcatcha.com/predict-bot"
payload = {
  "behavior_data": {
    "mouseMovements": [
      {"x": 100, "y": 230, "timestamp": 1718850001000},
      {"x": 102, "y": 231, "timestamp": 1718850001100}
    ],
    "mouseClicks": [
      {"type": "mousedown", "timestamp": 1718850001200},
      {"type": "mouseup",   "timestamp": 1718850001210}
    ]
  }
}
resp = requests.post(url, json=payload, timeout=10)
print(resp.status_code, resp.json())
```

응답 예시:
```json
{
  "confidence_score": 87.3,
  "is_bot": false,
  "mse": 0.000321,
  "threshold": 0.002145,
  "features": {
    "total_distance": 12.3,
    "average_speed": 0.42,
    "max_speed": 2.1,
    "min_speed": 0.0,
    "std_speed": 0.3,
    "total_duration": 2300,
    "movement_count": 2,
    "pause_count": 1,
    "click_count": 2,
    "mousedown_count": 1,
    "mouseup_count": 1,
    "click_type_mousedown": 1,
    "click_type_mouseup": 1,
    "click_type_click": 0
  }
}
```

### D) 문제 해결 팁
- 404 또는 502가 발생하면 Ingress → Service → Pod 경로와 포트를 점검하세요.
- 500 에러 시 컨테이너 로그를 확인하세요: `kubectl -n captcha logs deploy/ml-service -f`
- `model.pth` 또는 `scaler.pkl`을 찾지 못한다면:
  - 컨테이너 내 `/app/src/behavior_analysis/models`에 파일이 있는지 확인
  - 또는 Deployment에서 PVC 마운트 경로를 해당 디렉터리로 바꿨는지 확인
- 학습 시 `feature_columns.pkl`, `scaler.pkl`이 다른 곳에 생성된다면 학습을 `models` 디렉터리에서 실행했는지 확인하세요.
- 프로브 실패 시 `/health` 엔드포인트가 구현되어 있는지, 또는 프로브 경로를 올바르게 설정했는지 확인하세요.

---

## 📦 부록: KServe로의 확장(옵션)
- 현재 FastAPI 서빙(Deployment/Service/Ingress) 구성이 있으나, KServe를 사용할 수도 있습니다.
- Custom Predictor(uvicorn + FastAPI) 컨테이너 이미지를 지정하고, 모델 PVC 또는 MinIO(S3)를 마운트/연결하여 운영할 수 있습니다.
- 본 옵션은 운영 환경 요건에 맞추어 별도 설계가 필요합니다.


---

## ⚙️ Kubeflow Katib 하이퍼파라미터 튜닝 가이드

학습 스크립트는 다음 환경변수로 하이퍼파라미터를 제어할 수 있습니다.
- EPOCHS (기본 50)
- BATCH_SIZE (기본 32)
- LR (기본 1e-3)

또한 학습 종료 시 다음과 같이 Katib가 파싱할 수 있는 메트릭 로그를 출력합니다.
- BEST_LOSS: <float>

예: BEST_LOSS: 0.001234

### 1) Katib Experiment 예시 (YAML)
아래 예시는 위 3개의 파라미터를 탐색하며, 모델 산출물은 PVC를 `/app/src/behavior_analysis/models` 에 마운트하여 저장합니다.

```yaml
apiVersion: kubeflow.org/v1beta1
kind: Experiment
metadata:
  name: ae-hpo
  namespace: captcha
spec:
  maxTrialCount: 9
  parallelTrialCount: 3
  maxFailedTrialCount: 3
  objective:
    type: minimize
    goal: 0.0
    objectiveMetricName: best_loss
    additionalMetricNames: []
  algorithm:
    algorithmName: random
  parameters:
    - name: lr
      parameterType: DOUBLE
      feasibleSpace:
        min: "1e-4"
        max: "1e-2"
    - name: batchSize
      parameterType: INT
      feasibleSpace:
        min: "16"
        max: "128"
    - name: epochs
      parameterType: INT
      feasibleSpace:
        min: "10"
        max: "100"
  trialTemplate:
    primaryContainerName: trainer
    trialParameters:
      - name: lr
        description: learning rate
        reference: lr
      - name: batchSize
        description: batch size
        reference: batchSize
      - name: epochs
        description: epochs
        reference: epochs
    trialSpec:
      apiVersion: v1
      kind: Pod
      spec:
        restartPolicy: Never
        containers:
          - name: trainer
            image: python:3.10-slim
            command: ["bash", "-lc"]
            args:
              - >
                pip install torch==2.1.1 scikit-learn==1.3.2 pandas==2.1.4 numpy==1.25.2 joblib==1.3.2 &&
                git clone https://your.git.server/realcatcha.git &&
                cd realcatcha/backend/ml-service &&
                python -m pip install fastapi uvicorn &&
                export LR={{ .TrialParameters.lr }} &&
                export BATCH_SIZE={{ .TrialParameters.batchSize }} &&
                export EPOCHS={{ .TrialParameters.epochs }} &&
                python src/behavior_analysis/train_autoencoder.py
            env:
              - name: MODEL_DIR
                value: /app/src/behavior_analysis/models
            volumeMounts:
              - name: model-storage
                mountPath: /app/src/behavior_analysis/models
        volumes:
          - name: model-storage
            persistentVolumeClaim:
              claimName: ml-models-pvc
  metricsCollectorSpec:
    collector:
      kind: StdOut
    source:
      filter:
        metricsFormat:
          - {name: best_loss, regex: "BEST_LOSS: ([0-9.]+)"}
```

적용:
```bash
kubectl apply -f ae-hpo.yaml
kubectl -n captcha get experiment ae-hpo -o yaml | less
```

팁:
- Experiment 내 image는 사내 표준 학습 이미지로 교체하세요.
- PVC 이름(ml-models-pvc)과 네임스페이스(captcha)를 환경에 맞게 변경하세요.

---

## 🚀 KServe로 FastAPI 커스텀 Predictor 서빙하기
기존 Deployment/Service/Ingress 대신 KServe를 사용해 동일한 FastAPI 앱을 커스텀 컨테이너로 서빙할 수 있습니다.

### 1) InferenceService 예시 (custom predictor)
```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: ml-bot-detector
  namespace: captcha
spec:
  predictor:
    containers:
      - name: user-container
        image: ghcr.io/yourorg/realcatcha-ml-service:latest
        imagePullPolicy: IfNotPresent
        command: ["uvicorn"]
        args: ["src.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
        env:
          - name: MODEL_DIR
            value: /app/src/behavior_analysis/models
        ports:
          - containerPort: 8080
            name: http1
        volumeMounts:
          - name: model-storage
            mountPath: /app/src/behavior_analysis/models
    volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: ml-models-pvc
```

적용:
```bash
kubectl apply -f inference-service.yaml
kubectl -n captcha get isvc ml-bot-detector -w
```

호출:
- 커스텀 컨테이너 모드에서는 컨테이너의 HTTP 엔드포인트가 그대로 노출됩니다. 즉, `/predict-bot`와 `/health`를 그대로 사용할 수 있습니다.
- 클러스터 내부: `http://ml-bot-detector-predictor-default.captcha.svc.cluster.local/predict-bot`
- 외부: Istio Ingress 설정에 따라 도메인 또는 게이트웨이를 통해 접근합니다.

주의:
- PVC를 `/app/src/behavior_analysis/models`로 마운트해야 inference가 기대하는 경로와 일치합니다.
- 모델 교체 시 PVC의 파일만 바꿔도 즉시 반영됩니다(롤링 업데이트 필요시 재배포).

---

## 🧪 Kubeflow Pipelines로 학습 자동화 (간단 예)
단일 스텝 파이프라인으로 학습을 수행하고, 모델 산출물을 PVC에 저장하는 예시입니다.

```python
from kfp import dsl

def train_op():
    return dsl.ContainerOp(
        name='train-ae',
        image='python:3.10-slim',
        command=['bash', '-lc'],
        arguments=[
            'pip install torch==2.1.1 scikit-learn==1.3.2 pandas==2.1.4 numpy==1.25.2 joblib==1.3.2 && '
            'git clone https://your.git.server/realcatcha.git && '
            'cd realcatcha/backend/ml-service && '
            'export LR=0.001 BATCH_SIZE=64 EPOCHS=30 && '
            'python src/behavior_analysis/train_autoencoder.py'
        ],
        file_outputs={}
    ).add_pvolumes({
        '/app/src/behavior_analysis/models': dsl.PipelineVolume(pvc='ml-models-pvc')
    })

@dsl.pipeline(name='ae-train-pipeline', description='AutoEncoder training')
def pipeline():
    train = train_op()

if __name__ == '__main__':
    import kfp
    from kfp import compiler
    compiler.Compiler().compile(pipeline, 'ae-train-pipeline.yaml')
```

업로드 후 파이프라인 실행 시 파라미터는 env(EPOCHS, BATCH_SIZE, LR)로 제어할 수 있습니다. Katib과 결합하려면 Katib Experiment에서 Trial 템플릿으로 파이프라인 step 이미지를 호출하거나, Katib에서 직접 컨테이너를 실행하는 방식을 사용하세요.

---

## ✅ 참고/정리
- 코드에서 경로는 모두 Linux/Kubeflow 친화적으로 `src/config/paths.py`를 통해 관리됩니다.
- FastAPI 앱에 `/health`가 구현되어 있으며 프로브 경로로 사용할 수 있습니다.
- 학습/추론 모두 모델 파일 경로는 `src/behavior_analysis/models` 기준입니다. 운영에서는 PVC를 해당 경로에 마운트하세요.
- 하이퍼파라미터는 환경변수(EPOCHS, BATCH_SIZE, LR)로 제어되며 Katib가 로그 `BEST_LOSS:`를 파싱하도록 구현되어 있습니다.
