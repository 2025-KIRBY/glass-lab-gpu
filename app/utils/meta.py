# app/utils/meta.py
import json, time
from pathlib import Path
from typing import Any
from app.utils.paths import index_json

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def read_or_init(job_id: str) -> dict[str, Any]:
    p = index_json(job_id)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"job_id": job_id, "created_at": _now_iso(), "input": {}, "stages": {}}

def write(job_id: str, data: dict[str, Any]) -> None:
    p = index_json(job_id)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def set_input(job_id: str, init: str, concept: list[str], condition: list[str], mask: str | None = None) -> None:
    meta = read_or_init(job_id)
    meta["input"] = {
        "init": init,
        "concept": concept,
        "condition": condition,
        "mask": mask
    }
    write(job_id, meta)

def append_stage(job_id: str, stage: str, files: list[str], extra: dict[str, Any] | None = None) -> None:
    meta = read_or_init(job_id)
    meta["stages"].setdefault(stage, {})
    meta["stages"][stage].update({
        "files": files,
        "created_at": _now_iso(),
        **(extra or {})
    })
    write(job_id, meta)
