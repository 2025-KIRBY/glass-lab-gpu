from __future__ import annotations

from pathlib import Path
import io, uuid
from typing import List, Optional, AsyncGenerator
from PIL import Image
from fastapi import UploadFile
from app.models.controlnet_sdxl_model import get_pipe

def _to_pil(u: UploadFile) -> Image.Image:
    data = u.file.read()
    # ⭐️ .read()는 동기 방식으로 가정 ⭐️
    return Image.open(io.BytesIO(data)).convert("RGB")

# async def run(
#     init_image: UploadFile,
#     concept_images: List[UploadFile],
#     condition_images: List[UploadFile],
#     num_images: int = 6, # generate_service의 num_images 인자를 처리하기 위해 추가
#     # ⭐️ 참고: prompt, guidance 등 다른 인자들도 필요하면 추가해야 합니다.
# ) -> Tuple[bytes, str, str]:
#     """
#     ControlNetSDXL.generate() 시그니처에 맞추어 인자를 처리하고,
#     다수의 이미지를 생성하여 ZIP으로 패키징합니다.
#     """
    
#     # 1. 파일 PIL 객체로 변환
#     init_pil = _to_pil(init_image)
#     concept_pils = [_to_pil(u) for u in concept_images]
#     cond_pils = [_to_pil(u) for u in condition_images]

#     # 2. 모델 로드 및 인자 구성
#     pipe = get_pipe() 
    
#     # 모델의 generate 메서드가 요구하는 ref_imgs와 weights 구성
#     ref_imgs = [init_pil] + concept_pils + cond_pils
#     # weights는 모델의 내부 로직에 따라 결정: init=0.8, concept=0.8, condition=0.4
#     weights = [0.6] + [0.8] * len(concept_pils) + [0.2] * len(cond_pils)

#     # 3. 이미지 생성 루프
#     # pipe.generate()는 이미지 1장을 반환하므로, num_images만큼 루프를 돌립니다.
#     gen_images: List[Image.Image] = []
#     job_id = uuid.uuid4().hex[:12] # 작업당 고유 ID 생성

#     # ⭐️ 6장의 이미지를 생성하기 위해 generate 메서드를 num_images만큼 호출 ⭐️
#     for i in range(num_images):
#         # 시드는 루프마다 변경되어야 다른 이미지를 생성합니다.
#         current_seed = 1234 + i 
        
#         # ⭐️ pipe.generate 호출 인자를 모델의 시그니처와 일치시킵니다 ⭐️
#         generated_image = pipe.generate(
#             prompt="A photorealistic image of glasses, high detail", # 예시 프롬프트. API에서 받아와야 합니다.
#             base_img=init_pil,
#             ref_imgs=ref_imgs,
#             weights=weights,
#             seed=current_seed,
#             # steps, guidance 등 필요한 인자들도 여기에 명시해야 합니다.
#         )
#         gen_images.append(generated_image)

#     # 4. ZIP 패키징 (기존 로직 유지)
    
#     # ✅ 4-1. 서버 저장용 폴더 생성
#     # save_dir = Path("/workspace/outputs")   # RunPod/EC2용
#     save_dir = Path("app/static/outputs") # 로컬 FastAPI용
#     save_dir.mkdir(parents=True, exist_ok=True)

#     # ✅ 4-2. ZIP으로 묶기 + 서버에도 저장
#     buf = io.BytesIO()
#     with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
#         for i, im in enumerate(gen_images):
#             filename = f"gen_{i:02d}_{job_id}.png"

#             # ⭐️ 백엔드 저장
#             file_path = save_dir / filename
#             im.save(file_path)

#             # ⭐️ ZIP에 추가 (프론트로 보낼용)
#             io_img = io.BytesIO()
#             im.save(io_img, format="PNG")
#             zf.writestr(filename, io_img.getvalue())

#     buf.seek(0)

#     filename = f"result_{job_id}.zip"
#     return buf.read(), filename, job_id

async def run_stage1(
    init_image: UploadFile,
    concept_images: List[UploadFile],
    condition_images: List[UploadFile],

    # 엔드포인트에서 넘어온 슬라이더 값들
    init_image_weight: float,
    concept_images_weight: float,
    condition_images_weight: float,
    controlnet_condition_scale: float,
    control_guidance_end: float,

    num_images: int = 6, # 기본값 맨 뒤로 가야함
) -> AsyncGenerator[bytes, None]:
    """
    이미지를 한 장 생성할 때마다 multipart/x-mixed-replace로 흘려보내는 generator

    이 함수 자체는 '데이터 조각을 만들어서 yield'만 하고,
    실제로 HTTP 응답 헤더(content-type 등)는 FastAPI 라우터에서
    StreamingResponse로 감싸며 설정하는 구조로 사용!!
    """
    # 1. PIL 변환
    init_pil = _to_pil(init_image)
    concept_pils = [_to_pil(u) for u in concept_images]
    cond_pils = [_to_pil(u) for u in condition_images]

    # 2. 모델 로드 및 인자 구성
    pipe = get_pipe()

    ref_imgs = [init_pil] + concept_pils + cond_pils
    weights = [init_image_weight] + [concept_images_weight] * len(concept_pils) + [condition_images_weight] * len(cond_pils)

    # 3. 로컬에 생성이미지들 저장!
    save_dir = Path("/workspace/outputs/generate")
    save_dir.mkdir(parents=True, exist_ok=True)

    boundary = "frame"  # multipart 경계 문자열
    job_id = uuid.uuid4().hex[:12]

    for i in range(num_images):
        current_seed = 1234 + i # 근데 실제로 seed 안쓰긴 할거임

        # ✅ 1장 생성
        generated_image: Image.Image = pipe.generate(
            prompt="A photorealistic image of glasses, high detail, white background",
            base_img=init_pil,
            ref_imgs=ref_imgs,
            weights=weights,
            seed=current_seed,

            # 프론트 슬라이더에서 온 값 사용
            controlnet_condition_scale=controlnet_condition_scale,
            control_guidance_end=control_guidance_end,
        )

        # 서버에 PNG로 저장
        filename = f"gen_{i:02d}_{job_id}.png"
        file_path = save_dir / filename
        generated_image.save(file_path)

        # ✅ 메모리에 PNG로 담기
        img_buf = io.BytesIO()
        generated_image.save(img_buf, format="PNG")
        img_bytes = img_buf.getvalue()

        # ✅ multipart 한 파트 만들기
        # Content-Length를 넣어두면 프론트가 파싱하기 더 편함
        part = (
            f"--{boundary}\r\n"
            "Content-Type: image/png\r\n"
            f"Content-Length: {len(img_bytes)}\r\n"
            f"X-Filename: {filename}\r\n"
            "\r\n"
        ).encode("utf-8") + img_bytes + b"\r\n"

        # 이제 이 파트를 바로바로 내보냄
        yield part

    # 스트림 끝 알리기
    yield f"--{boundary}--\r\n".encode("utf-8")