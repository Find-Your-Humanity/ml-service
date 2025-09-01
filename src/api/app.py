# ml_service/app.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
from dotenv import load_dotenv
import tempfile
import json
import os
from pathlib import Path
import sys
import traceback
from src.config.paths import ensure_directories
from typing import List
 

# detect_bot은 메모리 사용 이슈가 있어 요청 시점에 지연 import 합니다.

# FastAPI 인스턴스
app = FastAPI(title="ML Bot Detection API")

load_dotenv()

# CORS 설정
app.add_middleware(
	CORSMiddleware,
	allow_origins=[
		"http://localhost:3000",
		"http://localhost:3001",
		"https://realcatcha.com",
		"https://www.realcatcha.com",
		"https://api.realcatcha.com",
		"https://test.realcatcha.com",
		"https://dashboard.realcatcha.com"
	],
	allow_credentials=True,
	allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
	allow_headers=["*"],
)

# Ensure required directories exist at startup
@app.on_event("startup")
def _init_dirs():
    try:
        ensure_directories()
    except Exception:
        pass

# Health endpoint for probes
@app.get("/health")
def health():
    return {"status": "ok"}

# 요청 스키마
class BehaviorDataRequest(BaseModel):
    behavior_data: Dict[str, Any]

# 엔드포인트
@app.post("/predict-bot")
def predict_bot(req: BehaviorDataRequest):
    try:
        # 지연 import로 서버 기동 시 스키런/사이파이 로드 회피
        from src.behavior_analysis.inference_bot_detector import detect_bot
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump([req.behavior_data], tmp)
            tmp_path = tmp.name

        result = detect_bot(tmp_path)

        os.unlink(tmp_path)  # 임시 파일 삭제

        # 응답 리턴
        return {
            "confidence_score": round(result.get("score", 0), 2),
            "is_bot": result.get("is_bot", False),
            "mse": result.get("mse"),
            "threshold": result.get("threshold"),
            "features": result.get("features")
        }

    except Exception as e:
        print(traceback.format_exc())  # 터미널에 Traceback 강제 출력
        raise HTTPException(status_code=500, detail=f"Bot detection failed: {str(e)}")


# ===== CRNN Handwriting OCR Predict API =====
# 모델/문자셋 경로 설정: 모델 경로는 환경변수로만 지정
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 모델 경로: 환경변수에서만 읽음 (미설정 시 에러)
_ENV_CRNN_MODEL_PATH = os.environ.get("CRNN_MODEL_PATH") or os.environ.get("OCR_MODEL_PATH")
CRNN_MODEL_PATH = Path(_ENV_CRNN_MODEL_PATH).resolve() if _ENV_CRNN_MODEL_PATH else None

# charset 경로는 기존 기본 경로 유지 (원하면 CRNN_CHARSET_PATH 환경변수로 확장 가능)
CRNN_CHARSET_PATH = PROJECT_ROOT / "src" / "crnn" / "model" / "charset.json"

_crnn_predictor = None

def _get_crnn_predictor():
    global _crnn_predictor
    if _crnn_predictor is None:
        # 지연 import로 초기 부하 최소화
        from src.crnn.inference import HandwritingPredictor
        if CRNN_MODEL_PATH is None:
            raise HTTPException(status_code=500, detail="CRNN model path not set. Set environment variable 'CRNN_MODEL_PATH' (or 'OCR_MODEL_PATH').")
        if not CRNN_MODEL_PATH.exists():
            raise HTTPException(status_code=500, detail=f"CRNN model not found: {CRNN_MODEL_PATH}")
        if not CRNN_CHARSET_PATH.exists():
            raise HTTPException(status_code=500, detail=f"CRNN charset not found: {CRNN_CHARSET_PATH}")
        with open(CRNN_CHARSET_PATH, "r", encoding="utf-8") as f:
            charset = json.load(f)
        idx_to_char = charset["idx_to_char"]
        char_to_idx = charset["char_to_idx"]
        _crnn_predictor = HandwritingPredictor(
            str(CRNN_MODEL_PATH),
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
        )
    return _crnn_predictor


@app.post("/predict-text")
async def predict_text(file: UploadFile = File(...)):
    try:
        predictor = _get_crnn_predictor()
        backend = "pil"
        # 업로드 파일을 임시 경로에 저장 후 예측
        suffix = Path(file.filename).suffix or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        try:
            text = predictor.predict(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        return {"text": text, "preprocess": backend}
    except HTTPException:
        raise
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Text prediction failed: {str(e)}")


# ===== Abstract Image Classification (Keras .h5) =====
_abstract_model = None
_abstract_input_shape = None
_abstract_class_list: List[str] = []

# 경로 설정
ML_PROJECT_ROOT = Path(__file__).resolve().parents[2]
ABSTRACT_MODEL_PATH = ML_PROJECT_ROOT / "src" / "abstract" / "abstract_model.h5"
# word_list.txt는 ml-service/src/crnn/word_list.txt를 사용하도록 통일
WORD_LIST_PATH = ML_PROJECT_ROOT / "src" / "crnn" / "word_list.txt"


def _load_word_list(path: Path) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"[abstract] failed to load word_list: {e}")
        return []


def _get_abstract_model():
    global _abstract_model, _abstract_input_shape, _abstract_class_list
    if _abstract_model is None:
        try:
            # TensorFlow keras 경로로 통일 (TF 2.16.2 권장)
            import tensorflow as tf  # type: ignore
            _abstract_model = tf.keras.models.load_model(str(ABSTRACT_MODEL_PATH), compile=False)
            # 입력 크기 확인 (None, H, W, C)
            shape = getattr(_abstract_model, 'input_shape', None)
            if isinstance(shape, tuple) and len(shape) == 4:
                _abstract_input_shape = (shape[1], shape[2], shape[3])
            else:
                # 기본값: 224x224x3
                _abstract_input_shape = (224, 224, 3)
        except Exception as e:
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Failed to load abstract model: {e}")

    if not _abstract_class_list:
        _abstract_class_list = _load_word_list(WORD_LIST_PATH)
    if not _abstract_class_list:
        raise HTTPException(status_code=500, detail="word_list.txt is empty or missing")

    return _abstract_model, _abstract_input_shape, _abstract_class_list


def _preprocess_image_to_tensor(path: Path, input_shape: tuple):
    from PIL import Image
    import numpy as np
    target_h, target_w, target_c = input_shape
    with Image.open(path) as img:
        img = img.convert('RGB') if target_c == 3 else img.convert('L')
        img = img.resize((target_w, target_h))
        arr = np.array(img).astype('float32') / 255.0
        if target_c == 1 and arr.ndim == 2:
            arr = arr[..., None]
        return arr


@app.post("/predict-abstract-proba-batch")
async def predict_abstract_proba_batch(
    target_class: str = Form(...),
    files: List[UploadFile] = File(...),
):
    try:
        print(f"📥 [/predict-abstract-proba-batch] incoming: target_class='{target_class}', files={len(files) if files else 0}")
        t0 = __import__("time").time()
        model, input_shape, class_list = _get_abstract_model()
        t_model = __import__("time").time()
        try:
            class_index = class_list.index(target_class)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"target_class '{target_class}' not in word_list")

        # 임시 파일 저장 후 배치 전처리
        tmp_paths: List[Path] = []
        try:
            for uf in files:
                suffix = Path(uf.filename or "").suffix or ".jpg"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(await uf.read())
                    tmp_paths.append(Path(tmp.name))

            import numpy as np
            batch = np.stack([_preprocess_image_to_tensor(p, input_shape) for p in tmp_paths], axis=0)
            t_prep = __import__("time").time()

            # 예측
            preds = model.predict(batch)
            t_pred = __import__("time").time()
            # 출력 형태: (N, num_classes)
            if preds.ndim != 2 or preds.shape[1] < (class_index + 1):
                raise HTTPException(status_code=500, detail="Model output shape mismatch with class list")

            probs = preds[:, class_index].astype(float).tolist()
            elapsed_model = int((t_model - t0) * 1000)
            elapsed_prep = int((t_prep - t_model) * 1000)
            elapsed_pred = int((t_pred - t_prep) * 1000)
            print(
                f"📦 [/predict-abstract-proba-batch] ok: num_files={len(files)}, class_index={class_index}, "
                f"t_model={elapsed_model}ms, t_prep={elapsed_prep}ms, t_pred={elapsed_pred}ms"
            )
            return {"probs": probs, "num_files": len(files), "class_index": class_index}
        finally:
            for p in tmp_paths:
                try:
                    os.unlink(p)
                except Exception:
                    pass
    except HTTPException:
        raise
    except Exception as e:
        try:
            import time as _t
            print(f"❌ [/predict-abstract-proba-batch] failed: {str(e)}")
        except Exception:
            pass
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Abstract batch prediction failed: {str(e)}")
