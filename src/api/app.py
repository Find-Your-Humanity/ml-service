# ml_service/app.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Dict, Any
import tempfile
import json
import os
from pathlib import Path
import sys
import traceback

# detect_bot은 메모리 사용 이슈가 있어 요청 시점에 지연 import 합니다.

# FastAPI 인스턴스
app = FastAPI(title="ML Bot Detection API")

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
# 모델/문자셋 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CRNN_MODEL_PATH = PROJECT_ROOT / "src" / "crnn" / "model" / "best_model_by_cer.pth"
CRNN_CHARSET_PATH = PROJECT_ROOT / "src" / "crnn" / "model" / "charset.json"

_crnn_predictor = None

def _get_crnn_predictor():
    global _crnn_predictor
    if _crnn_predictor is None:
        # 지연 import로 초기 부하 최소화
        from src.crnn.inference import HandwritingPredictor
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
        return {"text": text}
    except HTTPException:
        raise
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Text prediction failed: {str(e)}")
