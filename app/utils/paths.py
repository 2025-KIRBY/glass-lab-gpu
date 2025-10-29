# app/utils/paths.py
from pathlib import Path
from app.core.config import settings

def job_root(job_id: str) -> Path:
    return Path(settings.STORAGE_ROOT) / "jobs" / job_id

def input_dir(job_id: str) -> Path:
    p = job_root(job_id) / "input"
    p.mkdir(parents=True, exist_ok=True)
    return p

def stage_dir(job_id: str, stage: str) -> Path:
    p = job_root(job_id) / stage
    p.mkdir(parents=True, exist_ok=True)
    return p

def stage1_dir(job_id: str) -> Path:
    return stage_dir(job_id, "stage1")

def stage2_dir(job_id: str) -> Path:
    return stage_dir(job_id, "stage2")

def stage3_dir(job_id: str) -> Path:
    return stage_dir(job_id, "stage3")

def index_json(job_id: str) -> Path:
    return job_root(job_id) / "index.json"
