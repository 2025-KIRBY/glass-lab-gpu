# app/api/endpoints/generate.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import StreamingResponse
from typing import List
import io
import logging
log = logging.getLogger(__name__)

# 인증/동시성/업로드 가드 (환경변수 API_KEY 없으면 require_auth는 자동 패스)
from app.core.security import require_auth, concurrency_guard, guard_uploads
from app.services.generate_service import run as run_stage1

router = APIRouter()

@router.post("/generate")
async def generate(
    init_image: UploadFile = File(...),
    concept_images: List[UploadFile] = File(...),
    condition_images: List[UploadFile] = File(...),
    _auth = Depends(require_auth),
    _conc = Depends(concurrency_guard),
):
    # 파일 타입/확장자 등 1차 가드
    guard_uploads([init_image, *concept_images, *condition_images])

    # 파라미터 검증
    if len(concept_images) < 2:
        raise HTTPException(400, "concept_images는 최소 2장")
    if len(condition_images) != 5:
        raise HTTPException(400, "condition_images는 항상 5장")

    try:
        zip_bytes, filename, job_id = await run_stage1(
            init_image, concept_images, condition_images
        )
    except Exception:
        log.exception("generate failed") 
        raise HTTPException(500, "이미지 생성 중 오류가 발생했습니다")

    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "X-Job-Id": job_id,  # 프론트에서 추적/백업 참조에 유용
    }
    return StreamingResponse(io.BytesIO(zip_bytes), media_type="application/zip", headers=headers)
