# app/main.py
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.logging import setup_logging
from app.core.config import settings
from app.api.endpoints import generate, inpaint, render3d


# Hugging Face / hy3dgen 캐시 경로를 RunPod 영구 저장소로 고정
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["HY3DGEN_CACHE"] = "/workspace/.cache/hy3dgen"

setup_logging()
app = FastAPI(title="Glass-Lab API")

# ✅ 서버 시작 시 두 파이프라인 모두 미리 로딩
@app.on_event("startup")
async def preload_pipelines():
    # 여기서 import 하는 이유: 상단에서 하면 순환참조 위험 줄이려고
    from app.models.controlnet_sdxl_model import get_pipe as get_controlnet_pipe
    from app.models.inpaint_sdxl_model import get_pipe as get_inpaint_pipe

    print("🚀 FastAPI startup: pre-loading SDXL pipelines...")

    # 기본 generate용 ControlNet SDXL
    _ = get_controlnet_pipe()

    # 인페인트용 SDXL
    _ = get_inpaint_pipe()

    print("✅ All SDXL pipelines are ready.")

# CORS
origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:5173",
    "https://glass-lab.vercel.app", 
    "https://www.glass-lab.site"
],
    allow_credentials=True,

    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우팅
app.include_router(generate.router)
app.include_router(inpaint.router)
# app.include_router(render3d.router)

@app.get("/healthz")
def health():
    return {"ok": True}
