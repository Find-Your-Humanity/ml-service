FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# 런타임 최소 의존(헬스체크용 curl, 인증서, 이미지 처리용 라이브러리)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       curl \
       ca-certificates \
       libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 애플리케이션 소스만 복사 (의존성은 PVC에서 동기화된 /opt/sitepkgs로 제공)
COPY src/ ./src/

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8001/health || exit 1

# uvicorn은 동기화된 site-packages(/opt/sitepkgs)에 있다고 가정
# (Deployment에서 PYTHONPATH="/opt/sitepkgs:/root/workspace/ml-service:/app"로 덮어써서 사용)
CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8001"]