from pathlib import Path

# src/config/ 기준으로 상위(src) 디렉토리 경로
BASE_DIR = Path(__file__).resolve().parent.parent


def get_model_file_path(filename: str) -> str:
    """모델 파일 경로 반환"""
    return str(BASE_DIR / "behavior_analysis" / "models" / filename)


def get_data_file_path(filename: str) -> str:
    """데이터 파일 경로 반환"""
    return str(BASE_DIR / "data" / "behavior_data" / filename)