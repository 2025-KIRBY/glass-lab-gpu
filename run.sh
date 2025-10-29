#!/usr/bin/env bash
set -euo pipefail

# 0) (선택) DNS 고정 — 네트워크 흔들릴 때 안정화
cp -L /etc/resolv.conf /etc/resolv.conf.bak 2>/dev/null || true
[ -L /etc/resolv.conf ] && unlink /etc/resolv.conf || true
printf "nameserver 1.1.1.1\nnameserver 8.8.8.8\noptions timeout:1 attempts:2 rotate\n" > /etc/resolv.conf

# 1) HF 캐시(영구 볼륨 권장)
export HF_HOME=${HF_HOME:-/runpod-volume/hf}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
mkdir -p "$HF_HOME"

# 2) (선택) venv 활성화
# [ -d /workspace/glass-lab-gpu/venv ] && source /workspace/glass-lab-gpu/venv/bin/activate

# 3) 앱 디렉토리로 이동
cd /workspace/glass-lab-gpu

# 4) 서버 실행 (로그 파일, 백그라운드)
# 포그라운드로 띄우고 싶으면 nohup 제거하고 exec uvicorn ... 으로 바꿔도 됨
nohup uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --limit-max-request 67108864 \
  > server.log 2>&1 &

# 5) 컨테이너 종료 방지: 백그라운드 작업이 끝날 때까지 대기
wait
