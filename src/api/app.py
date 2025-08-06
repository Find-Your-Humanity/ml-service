# ml_service/app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import tempfile
import json
import os
from pathlib import Path
import sys
import traceback

# 기존 detect_bot import
from src.behavior_analysis.inference_bot_detector import detect_bot

# FastAPI 인스턴스
app = FastAPI(title="ML Bot Detection API")

# 요청 스키마
class BehaviorDataRequest(BaseModel):
    behavior_data: Dict[str, Any]

# 엔드포인트
@app.post("/predict-bot")
def predict_bot(req: BehaviorDataRequest):
    try:
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
