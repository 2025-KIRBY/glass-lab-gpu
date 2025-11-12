# app/api/endpoints/inpaint.py
from fastapi import APIRouter, Form, UploadFile, File, HTTPException, Depends
from fastapi.responses import StreamingResponse
from typing import List
import io, logging

# 인증/동시성/업로드 가드
from app.core.security import require_auth, concurrency_guard, guard_uploads
from app.services.inpaint_service import run_stage2

log = logging.getLogger(__name__)
router = APIRouter()


@router.post("/inpaint")
async def inpaint_generate(
    # prompt: str = Form(...), # 프론트에서 프롬프트 안 받는걸로
    init_image: UploadFile = File(...),
    mask_image: UploadFile = File(...),
    new_concept_images: List[UploadFile] = File(...),
    condition_images: List[UploadFile] = File(...),
    _auth = Depends(require_auth),
    _conc = Depends(concurrency_guard),
):
    """
    Stage 2 Inpainting API
    - prompt + init + mask + concept + condition 이미지 입력
    - 생성 이미지는 한장씩 스트리밍으로 주기 
    """

    # 1️⃣ 업로드 파일 검증
    guard_uploads([init_image, mask_image, *new_concept_images, *condition_images])

    # 2️⃣ 입력 검증
    if len(new_concept_images) < 1:
        raise HTTPException(400, "new_concept_images는 최소 1장 이상이어야 합니다.")
    if len(condition_images) != 5:
        raise HTTPException(400, "condition_images는 항상 5장이어야 합니다.")

    try:
        # 3️⃣ 인페인팅 실행
        # zip_bytes, filename, job_id = await run_stage2(
        #     prompt=prompt,
        #     init_image=init_image,
        #     mask_image=mask_image,
        #     new_concept_images=new_concept_images,
        #     condition_images=condition_images,
        # )
        gen = run_stage2(
            prompt="",  # 디폴트 프롬프트는 서비스 계층에 넣어줌
            init_image=init_image,
            mask_image=mask_image,
            new_concept_images=new_concept_images,
            condition_images=condition_images,
            num_images=6,
        )

    except Exception:
        log.exception("inpaint_generate failed")
        raise HTTPException(500, "인페인팅 생성 중 오류가 발생했습니다.")

    # 4️⃣ 응답 헤더 구성
    headers = {
        # "Content-Disposition": f'attachment; filename="{filename}"',
        "X-Job-Id": job_id,
    }

    # 5️⃣  스트리밍 응답 반환
    return StreamingResponse(
        gen,
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers=headers,
    )
