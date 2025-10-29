# app/services/generate_service.py
from typing import List, Tuple
from fastapi import UploadFile
from PIL import Image
from app.utils.io import new_job_id, save_file, pil_list_to_zip_bytes
from app.utils import paths, meta
from app.models.controlnet_sdxl_model import generate_6  # 스텁 또는 실제 파이프라인

async def run(init_up: UploadFile, concept_up_list: List[UploadFile], condition_up_list: List[UploadFile]) -> Tuple[bytes, str, str]:
    """
    returns: (zip_bytes, filename, job_id)
    """
    job_id = new_job_id()

    # 경로 보장
    inp_dir = paths.input_dir(job_id)
    out_dir = paths.stage1_dir(job_id)

    # 업로드 저장
    init_p = save_file(inp_dir, init_up)
    concept_ps = [save_file(inp_dir, f) for f in concept_up_list]
    cond_ps    = [save_file(inp_dir, f) for f in condition_up_list]

    # 메타 기록 (입력)
    meta.set_input(
        job_id,
        init=init_p.name,
        concept=[p.name for p in concept_ps],
        condition=[p.name for p in cond_ps],
        mask=None
    )

    # ===== 모델 호출 지점 =====
    images = await generate_6(init_p, concept_ps, cond_ps)  # -> list[PIL.Image]
    # ==============================

    # 결과 저장 + 파일명 수집
    names = []
    for i, im in enumerate(images, 1):
        p = out_dir / f"result_{i:03d}.png"
        im.save(p)
        names.append(p.name)

    # 메타 기록 (stage1)
    meta.append_stage(
        job_id,
        "stage1",
        files=[f"stage1/{n}" for n in names],
        extra={"num_outputs": len(names)}
    )

    # 프론트 전송용 ZIP 바이트
    zip_bytes = pil_list_to_zip_bytes(images, names=names)
    return zip_bytes, "stage1_results.zip", job_id
