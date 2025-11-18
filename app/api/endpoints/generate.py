# app/api/endpoints/generate.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from fastapi.responses import StreamingResponse
from typing import List
import io
import logging
log = logging.getLogger(__name__)

# 인증/동시성/업로드 가드 (환경변수 API_KEY 없으면 require_auth는 자동 패스)
from app.core.security import require_auth, concurrency_guard, guard_uploads
from app.services.generate_service import run_stage1 # 이미지 한장씩 넘기는 스트리밍 방식으로 변경함

router = APIRouter()

@router.post("/generate")
async def generate(
    init_image: UploadFile = File(...),
    concept_images: List[UploadFile] = File(...),
    condition_images: List[UploadFile] = File(...),

    # 슬라이더 값들 -> 괄호 안의 값은 디폴트
    init_image_weight: float = Form(0.6),
    concept_images_weight: float = Form(0.8),
    condition_images_weight: float = Form(0.2),

    controlnet_condition_scale: float = Form(0.35),
    control_guidance_end: float = Form(0.35),

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
        # zip_bytes, filename, job_id = await run_stage1(
        #     init_image, concept_images, condition_images
        # )
        # 3) ZIP을 받는 게 아니라 async generator를 받는다
        gen = run_stage1(
            init_image=init_image,
            concept_images=concept_images,
            condition_images=condition_images,
            num_images=6,  # 필요하면 프론트에서 Form으로 받도록 바꿔도 됨

            # 슬라이더에서 받은 값들 전달
            init_image_weight=init_image_weight,
            concept_images_weight=concept_images_weight,
            condition_images_weight=condition_images_weight,
            controlnet_condition_scale=controlnet_condition_scale,
            control_guidance_end=control_guidance_end,
        )
    except Exception:
        log.exception("generate failed") 
        raise HTTPException(500, "이미지 생성 중 오류가 발생했습니다")

    # 스트리밍 응답으로 감싸서 반환하기
    # headers = {
    #     # "Content-Disposition": f'attachment; filename="{filename}"',
    #     "X-Job-Id": job_id,
    # }
    return StreamingResponse(
        gen,
        media_type="multipart/x-mixed-replace; boundary=frame",
        # headers=headers,
    )
