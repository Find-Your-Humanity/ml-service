"""
Behavior Analysis Module

사용자 행동 분석 및 봇 탐지를 위한 모듈
"""

# 주요 함수들을 패키지 레벨에서 import 가능하도록 설정
try:
    from .inference_bot_detector import detect_bot
except ImportError:
    # 의존성이 없는 경우에도 패키지 로딩이 실패하지 않도록
    detect_bot = None

__all__ = ['detect_bot'] 