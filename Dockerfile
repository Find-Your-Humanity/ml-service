# 앱 이미지는 프리베이크된 런타임 이미지를 기반으로 빌드
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# 헬스체크/이미지 처리 등에 필요한 최소 패키지는 런타임 이미지에 이미 포함되어 있음

# 애플리케이션 소스만 복사
COPY src/ ./src/

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8001/health || exit 1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8001"]