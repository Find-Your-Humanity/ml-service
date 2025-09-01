from pathlib import Path
import os

# src/config/ 기준으로 상위(src) 디렉토리 경로
BASE_DIR = Path(__file__).resolve().parent.parent

# 기본 디렉터리
# 사용자 행동 모델 아티팩트 기본 경로를 /root/models 로 지정
DEFAULT_MODEL_DIR = Path("/root/models")
DEFAULT_DATA_DIR = BASE_DIR / "data" / "behavior_data"

# 환경변수로 오버라이드 가능
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(DEFAULT_MODEL_DIR))).resolve()
DATA_DIR = Path(os.environ.get("DATA_DIR", str(DEFAULT_DATA_DIR))).resolve()

# 로그 디렉터리
LOG_DIR = BASE_DIR / "logs"


def ensure_directories() -> None:
    """모델/데이터/로그 디렉터리 생성(없으면)."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_model_file_path(filename: str) -> str:
    """모델 파일 경로 반환"""
    return str(MODEL_DIR / filename)


def get_data_file_path(filename: str) -> Path:
    """데이터 파일 경로(Path) 반환"""
    return DATA_DIR / filename