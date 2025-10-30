# # app/services/generate_service.py
# from __future__ import annotations
# import io, zipfile
# from typing import List, Tuple, Optional
# from PIL import Image
# from fastapi import UploadFile

# from app.models.controlnet_sdxl_model import ControlNetSDXL
# from app.models.controlnet_sdxl_model import get_pipe

# _model: Optional[GlassLabSDXL] = None

# def _get_model() -> GlassLabSDXL:
#     global _model
#     if _model is None:
#         _model = GlassLabSDXL(
#             # dtype=torch.float16  # 기본값(f16). CPU나 낮은 VRAM이면 float32로 변경
#         )
#     return _model

# def _to_pil(u: UploadFile) -> Image.Image:
#     data = u.file.read()
#     return Image.open(io.BytesIO(data)).convert("RGB")

# async def run(
#     init_image: UploadFile,
#     concept_images: List[UploadFile],
#     condition_images: List[UploadFile],
# ) -> Tuple[bytes, str, str]:
#     """
#     엔드포인트 generate.py가 기대하는 시그니처 그대로:
#       - init 1장
#       - concept 2~5장
#       - condition 5장
#       - 결과 6장 ZIP
#     """
#     init_pil = _to_pil(init_image)
#     concept_pils = [_to_pil(u) for u in concept_images]
#     cond_pils = [_to_pil(u) for u in condition_images]

#     pipe = get_pipe() 
#     gen_images, job_id = pipe.generate(
#         init_image=init_pil,
#         concept_images=concept_pils,
#         condition_images=cond_pils,
#         num_images=6,   # 요구: 6장 생성
#     )

#     # ZIP 패키징
    
#     buf = io.BytesIO()
#     with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
#         for i, im in enumerate(gen_images):
#             io_img = io.BytesIO()
#             im.save(io_img, format="PNG")
#             zf.writestr(f"gen_{i:02d}.png", io_img.getvalue())
#     buf.seek(0)

#     filename = "result_images.zip"
#     return buf.read(), filename, job_id
from __future__ import annotations
import io, zipfile
from typing import List, Tuple, Optional
from PIL import Image
from fastapi import UploadFile
import uuid

# ⭐️ 수정: GlassLabSDXL 대신 ControlNetSDXL 임포트 ⭐️
from app.models.controlnet_sdxl_model import ControlNetSDXL, get_pipe

# ⭐️ 수정: _model 변수 제거 (get_pipe 사용) ⭐️

def _to_pil(u: UploadFile) -> Image.Image:
    data = u.file.read()
    # ⭐️ .read()는 async/await가 필요할 수 있으나, 현재는 동기 방식으로 가정 ⭐️
    return Image.open(io.BytesIO(data)).convert("RGB")


async def run(
    init_image: UploadFile,
    concept_images: List[UploadFile],
    condition_images: List[UploadFile],
    num_images: int = 6, # generate_service의 num_images 인자를 처리하기 위해 추가
    # ⭐️ 참고: prompt, guidance 등 다른 인자들도 필요하면 추가해야 합니다.
) -> Tuple[bytes, str, str]:
    """
    ControlNetSDXL.generate() 시그니처에 맞추어 인자를 처리하고,
    다수의 이미지를 생성하여 ZIP으로 패키징합니다.
    """
    
    # 1. 파일 PIL 객체로 변환
    init_pil = _to_pil(init_image)
    concept_pils = [_to_pil(u) for u in concept_images]
    cond_pils = [_to_pil(u) for u in condition_images]

    # 2. 모델 로드 및 인자 구성
    pipe = get_pipe() 
    
    # 모델의 generate 메서드가 요구하는 ref_imgs와 weights 구성
    ref_imgs = [init_pil] + concept_pils + cond_pils
    # weights는 모델의 내부 로직에 따라 결정: init=0.8, concept=0.8, condition=0.4
    weights = [0.8] + [0.8] * len(concept_pils) + [0.4] * len(cond_pils)

    # 3. 이미지 생성 루프
    # pipe.generate()는 이미지 1장을 반환하므로, num_images만큼 루프를 돌립니다.
    gen_images: List[Image.Image] = []
    job_id = uuid.uuid4().hex[:12] # 작업당 고유 ID 생성

    # ⭐️ 6장의 이미지를 생성하기 위해 generate 메서드를 num_images만큼 호출 ⭐️
    for i in range(num_images):
        # 시드는 루프마다 변경되어야 다른 이미지를 생성합니다.
        current_seed = 1234 + i 
        
        # ⭐️ pipe.generate 호출 인자를 모델의 시그니처와 일치시킵니다 ⭐️
        generated_image = pipe.generate(
            prompt="A photorealistic image of glasses, high detail", # 예시 프롬프트. API에서 받아와야 합니다.
            base_img=init_pil,
            ref_imgs=ref_imgs,
            weights=weights,
            seed=current_seed,
            # steps, guidance 등 필요한 인자들도 여기에 명시해야 합니다.
        )
        gen_images.append(generated_image)

    # 4. ZIP 패키징 (기존 로직 유지)
    
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, im in enumerate(gen_images):
            io_img = io.BytesIO()
            im.save(io_img, format="PNG")
            zf.writestr(f"gen_{i:02d}_{job_id}.png", io_img.getvalue())
    buf.seek(0)

    filename = f"result_{job_id}.zip"
    return buf.read(), filename, job_id