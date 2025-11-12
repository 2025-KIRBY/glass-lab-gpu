from __future__ import annotations

from pathlib import Path
import io, zipfile, uuid
from typing import List, AsyncGenerator
from PIL import Image
from fastapi import UploadFile
from app.models.inpaint_sdxl_model import InpaintSDXL, get_pipe

def _to_pil(u: UploadFile) -> Image.Image:
    data = u.file.read()
    # .read()는 현재는 동기 방식으로 가정 
    return Image.open(io.BytesIO(data)).convert("RGB")

# async def run_stage2(
#     prompt:  str,
#     init_image: UploadFile,
#     mask_image: UploadFile,
#     new_concept_images: List[UploadFile],
#     condition_images: List[UploadFile],
#     num_images: int = 6
# ) -> Tuple[bytes, str, str]:
#     """
#     Stage 2 인페인팅 서비스: N장 결과 생성 및 서버 저장 + URL 반환
#     """
#     init_pil = _to_pil(init_image)
#     mask_pil = _to_pil(mask_image)
#     concept_pils = [_to_pil(u) for u in new_concept_images]
#     cond_pils = [_to_pil(u) for u in condition_images] 

#     pipe = get_pipe()   # InpaintSDXL 싱글톤

#     # 모델의 generate 메서드가 요구하는 ref_imgs와 weights 구성
#     ref_imgs = [init_pil] + concept_pils + cond_pils
#     weights = [0.6] + [0.8] * len(concept_pils) + [0.2] * len(cond_pils)

#     # ============================================================
#     # 3. 이미지 생성 (num_images 만큼)
#     # ============================================================
#     job_id = uuid.uuid4().hex[:12]
#     gen_images: List[Image.Image] = []

#     for i in range(num_images):
#         current_seed = 1234 + i
#         generated_image = pipe.inpaint_generate(
#             prompt=prompt,             # ✅ 프롬프트 전달
#             base_img=init_pil,
#             mask_img=mask_pil,
#             ref_imgs=ref_imgs,
#             weights=weights,
#             seed=current_seed,
#             steps=40,
#             guidance=7.5,
#         )
#         gen_images.append(generated_image)

#     # ============================================================
#     # 4. ZIP 패키징 + 서버 저장
#     # ============================================================
#     save_dir = Path("/workspace/outputs/inpaint")
#     save_dir.mkdir(parents=True, exist_ok=True)

#     buf = io.BytesIO()
#     with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
#         for i, im in enumerate(gen_images):
#             filename = f"inpaint_{job_id}_{i+1:02d}.png"

#             # 서버 저장
#             file_path = save_dir / filename
#             im.save(file_path)

#             # ZIP 파일에 추가
#             img_bytes = io.BytesIO()
#             im.save(img_bytes, format="PNG")
#             zf.writestr(filename, img_bytes.getvalue())

#     buf.seek(0)

#     filename = f"inpaint_result_{job_id}.zip"
#     return buf.read(), filename, job_id


async def run_stage2(
    prompt: str,
    init_image: UploadFile,
    mask_image: UploadFile,
    new_concept_images: List[UploadFile],
    condition_images: List[UploadFile],
    num_images: int = 3,
) -> AsyncGenerator[bytes, None]:
    """
    Stage 2 인페인팅을 한 장 생성할 때마다 바로바로 내보내는 스트리밍 버전.
    실제 HTTP 헤더(content-type)는 라우터에서 StreamingResponse로 감싸면서 넣는다.
    """
    # 1. 업로드를 PIL로 변환
    init_pil = _to_pil(init_image)
    mask_pil = _to_pil(mask_image)
    concept_pils = [_to_pil(u) for u in new_concept_images]
    cond_pils = [_to_pil(u) for u in condition_images]

    # 2. 모델 가져오기
    pipe = get_pipe()

    # 3. 참조 이미지 & weight 구성
    ref_imgs = [init_pil] + concept_pils + cond_pils
    weights = [0.6] + [0.8] * len(concept_pils) + [0.2] * len(cond_pils)

    # 4. 저장 폴더
    save_dir = Path("/workspace/outputs/inpaint")
    save_dir.mkdir(parents=True, exist_ok=True)

    boundary = "frame"
    job_id = uuid.uuid4().hex[:12]

    # 5. 이미지 생성 루프
    for i in range(num_images):
        current_seed = 1234 + i # 시드값 안쓴다

        generated_image: Image.Image = pipe.inpaint_generate(
            prompt="A photorealistic image of glasses, high detail, white background",
            base_img=init_pil,
            mask_img=mask_pil,
            ref_imgs=ref_imgs,
            weights=weights,
            seed=current_seed,
            steps=40,
            guidance=7.5,
        )

        # 파일명 구성
        filename = f"inpaint_{job_id}_{i+1:02d}.png"
        file_path = save_dir / filename

        # 서버에도 저장
        generated_image.save(file_path)

        # 메모리에 PNG로 저장
        img_buf = io.BytesIO()
        generated_image.save(img_buf, format="PNG")
        img_bytes = img_buf.getvalue()

        # multipart 조각 만들기
        part = (
            f"--{boundary}\r\n"
            "Content-Type: image/png\r\n"
            f"Content-Length: {len(img_bytes)}\r\n"
            f"X-Filename: {filename}\r\n"
            f"X-Job-Id: {job_id}\r\n"
            "\r\n"
        ).encode("utf-8") + img_bytes + b"\r\n"

        # 한 장씩 바로바로 보내기
        yield part

    # 스트림 끝 표시
    yield f"--{boundary}--\r\n".encode("utf-8")