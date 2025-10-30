# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.logging import setup_logging
from app.core.config import settings
from app.api.endpoints import generate, inpaint, render3d

setup_logging()
app = FastAPI(title="Glass-Lab API")

# CORS
origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:5173",
    "https://glass-lab.vercel.app", 
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
