# 경량 런타임 이미지: PVC의 venv를 활용하여 의존성 설치를 제거
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/root/workspace/venv \
    PATH=/root/workspace/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONPATH=/app:/root/workspace/ml-service

WORKDIR /app

# 런타임 최소 의존 패키지(헬스체크용 curl, 인증서, 일부 이미지 처리 라이브러리)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       curl \
       ca-certificates \
       libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 애플리케이션 소스(가벼운 파일만) 복사
COPY src/ ./src/

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8001/health || exit 1

# PVC의 venv가 존재해야 하며, 없으면 빠르게 실패
CMD ["/bin/sh","-lc","test -x \"$VIRTUAL_ENV/bin/python\" || { echo \"venv not found at $VIRTUAL_ENV\"; exit 1; }; exec python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8001"]