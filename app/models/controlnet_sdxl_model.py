# from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
# from diffusers.utils import load_image
# from PIL import Image
# import numpy as np
# import torch, cv2, time, gc, os

# class ControlNetSDXL:
#     def __init__(self, device="cuda"):
#         print("🔹 Loading SDXL ControlNet + IP Adapter pipeline...")
#         self.device = device
#         self.vae = AutoencoderKL.from_pretrained(
#             "madebyollin/sdxl-vae-fp16-fix",
#             torch_dtype=torch.float16
#         )

#         self.controlnet = ControlNetModel.from_pretrained(
#             "xinsir/controlnet-union-sdxl-1.0",
#             torch_dtype=torch.float16
#         )

#         self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
#             "stabilityai/stable-diffusion-xl-base-1.0",
#             controlnet=self.controlnet,
#             vae=self.vae,
#             torch_dtype=torch.float16,
#             use_safetensors=True
#         ).to(self.device)

#         # IP Adapter 로드
#         self.pipe.load_ip_adapter(
#             "h94/IP-Adapter",
#             subfolder="sdxl_models",
#             weight_name="ip-adapter_sdxl.bin"
#         )

#         print("✅ Model load complete.")

#     # --- IP Adapter embedding 가중합 ---
#     def set_adapter(self, image_list, weights):
#         embeds_list = []
#         for img in image_list:
#             embed = self.pipe.prepare_ip_adapter_image_embeds(
#                 ip_adapter_image=img,
#                 ip_adapter_image_embeds=None,
#                 device=self.device,
#                 num_images_per_prompt=1,
#                 do_classifier_free_guidance=True
#             )
#             embeds_list.append(embed[0])

#         pos = torch.stack([e[0] for e in embeds_list], dim=0)
#         neg = torch.stack([e[1] for e in embeds_list], dim=0)
#         w = torch.tensor(weights, device=pos.device, dtype=pos.dtype).view(-1, 1, 1)

#         weighted_pos = (pos * w).sum(dim=0, keepdim=True)
#         weighted_neg = (neg * w).sum(dim=0, keepdim=True)
#         combined = torch.cat([weighted_pos, weighted_neg], dim=0)
#         return combined

#     # --- Canny 엣지 생성 ---
#     def get_control_image(self, init_image):
#         img_np = np.array(init_image)
#         edges = cv2.Canny(img_np, 100, 200)
#         kernel = np.ones((3, 3), np.uint8)
#         edges_thick = cv2.dilate(edges, kernel, iterations=1)
#         control_img = np.concatenate([edges_thick[:, :, None]] * 3, axis=2)
#         control_img = Image.fromarray(control_img)
#         return control_img

#     # --- 메인 생성 함수 ---
#     def generate(self, prompt, base_img, ref_imgs, weights):
#         control_image = self.get_control_image(base_img)
#         embeds = self.set_adapter(ref_imgs, weights)
#         self.pipe.set_ip_adapter_scale(1.1)

#         generator = torch.Generator(device=self.device).manual_seed(1234)

#         out = self.pipe(
#             prompt=prompt,
#             guidance_scale=7.5,
#             num_inference_steps=16,
#             ip_adapter_image_embeds=[embeds],
#             image=control_image,
#             controlnet_conditioning_scale=1.0,
#             control_guidance_start=0.0,
#             control_guidance_end=1.0,
#             generator=generator
#         ).images[0]

#         gc.collect()
#         torch.cuda.empty_cache()

#         return out


# app/models/controlnet_sdxl_model.py
"""
SDXL + ControlNet + IP-Adapter (1차 생성) 파이프라인
- 입력: init_path (1장), concept_paths (N장), condition_paths (M장)
- 출력: PIL.Image 리스트 (기본 6장)
- Colab 코드의 핵심(파이프라인 로딩, IP-Adapter 임베딩 가중합, Canny control 등)을
  서버 용도로 리팩터링한 버전.

필요 패키지 버전(권장):
  pip install git+https://github.com/tencent-ailab/IP-Adapter.git
  pip install diffusers==0.35.1 transformers==4.56.2 accelerate==1.10.1 safetensors opencv-python pillow
  # torch는 CUDA 버전에 맞춰 별도 설치

선택 가속:
  pip install xformers==0.0.27.post2  # (torch cu121과 궁합)
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import numpy as np
from PIL import Image
import torch
import cv2

from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
)

# ===== 기본 하이퍼파라미터(필요시 조정) =====
IMG_SIZE = 1024
NUM_OUTPUTS = 6
GUIDANCE = 7.5
STEPS = 20
SEED: Optional[int] = 1234    # None이면 무작위
ADAPTER_SCALE = 1.2
CONTROLNET_CONDITIONING_SCALE = 0.4
CONTROL_GUIDANCE_START = 0.0
CONTROL_GUIDANCE_END = 0.4
PROMPT = "glasses"            # 필요시 서비스 계층에서 주입 가능

# ===== 모델 체크포인트 =====
VAE_REPO = "madebyollin/sdxl-vae-fp16-fix"
BASE_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
CONTROLNET_REPO = "xinsir/controlnet-union-sdxl-1.0"
IPADAPTER_REPO = "h94/IP-Adapter"
IPADAPTER_SUBFOLDER = "sdxl_models"
IPADAPTER_WEIGHTS = "ip-adapter_sdxl.bin"

# ===== Lazy Singleton 파이프라인 =====
_pipe: Optional[StableDiffusionXLControlNetPipeline] = None
_device = "cuda" if torch.cuda.is_available() else "cpu"
_dtype = torch.float16 if _device == "cuda" else torch.float32


def _load_resize(path: Path, size: int = IMG_SIZE) -> Image.Image:
    """경로에서 이미지를 로드하고 정사각형으로 리사이즈."""
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    return img


def _to_control_image_from_canny(base_img: Image.Image) -> Image.Image:
    """
    ControlNet용 엣지 맵 생성(Canny + dilation).
    condition_paths가 비어있을 때 fallback으로 사용.
    """
    arr = np.array(base_img)
    edges = cv2.Canny(arr, 100, 200)
    kernel = np.ones((3, 3), np.uint8)
    edges_thick = cv2.dilate(edges, kernel, iterations=1)
    control = np.stack([edges_thick] * 3, axis=2)  # (H, W, 3)
    return Image.fromarray(control)


def _get_pipe() -> StableDiffusionXLControlNetPipeline:
    """모델을 1회 로드하고 재사용."""
    global _pipe
    if _pipe is not None:
        return _pipe

    vae = AutoencoderKL.from_pretrained(VAE_REPO, torch_dtype=_dtype)
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_REPO, torch_dtype=_dtype)

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        BASE_REPO,
        controlnet=controlnet,
        vae=vae,
        torch_dtype=_dtype,
        use_safetensors=True,
    )

    # IP-Adapter 로드
    pipe.load_ip_adapter(
        IPADAPTER_REPO,
        subfolder=IPADAPTER_SUBFOLDER,
        weight_name=IPADAPTER_WEIGHTS,
    )

    # 가속 옵션
    if _device == "cuda":
        pipe.to("cuda")
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    else:
        pipe.to("cpu")

    pipe.enable_attention_slicing()  # 메모리 절약(속도-메모리 트레이드오프)
    # 필요시: pipe.enable_model_cpu_offload()

    _pipe = pipe
    return _pipe


def _combine_ipadapter_embeds(
    pipe: StableDiffusionXLControlNetPipeline,
    ref_images: List[Image.Image],
    weights: List[float],
) -> torch.Tensor:
    """
    여러 참조이미지의 IP-Adapter 임베딩을 가중합.
    - prepare_ip_adapter_image_embeds(..., do_classifier_free_guidance=True)가
      (2, N, C) 형태(긍정/부정)로 반환된다고 가정.
    - 출력: (2, N, C) 텐서
    """
    embeds_list = []
    for img in ref_images:
        emb = pipe.prepare_ip_adapter_image_embeds(
            ip_adapter_image=img,
            ip_adapter_image_embeds=None,
            device=_device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )
        # emb[0]: (2, N, C)
        embeds_list.append(emb[0])

    # 긍정/부정 분리 후 weight 가중합
    pos = torch.stack([e[0] for e in embeds_list], dim=0)  # (n, N, C)
    neg = torch.stack([e[1] for e in embeds_list], dim=0)  # (n, N, C)
    w = torch.tensor(weights, device=pos.device, dtype=pos.dtype).view(-1, 1, 1)

    weighted_pos = (pos * w).sum(dim=0, keepdim=True)  # (1, N, C)
    weighted_neg = (neg * w).sum(dim=0, keepdim=True)  # (1, N, C)
    combined = torch.cat([weighted_pos, weighted_neg], dim=0)  # (2, N, C)
    return combined


async def generate_6(
    init_path: Path,
    concept_paths: List[Path],
    condition_paths: List[Path],
    *,
    num_outputs: int = NUM_OUTPUTS,
    prompt: str = PROMPT,
    steps: int = STEPS,
    guidance: float = GUIDANCE,
    adapter_scale: float = ADAPTER_SCALE,
    controlnet_conditioning_scale: float = CONTROLNET_CONDITIONING_SCALE,
    control_guidance_start: float = CONTROL_GUIDANCE_START,
    control_guidance_end: float = CONTROL_GUIDANCE_END,
    seed: Optional[int] = SEED,
) -> List[Image.Image]:
    """
    우리 1차 API에서 호출하는 메인 함수.
    - init_path: 기본 뼈대 이미지 1장
    - concept_paths: 콘셉트 스타일 이미지들
    - condition_paths: (선택) ControlNet용 조건 이미지들 혹은 추가 참조들
        * 첫 번째 이미지를 ControlNet "image"로 사용하려 시도
        * 없으면 init의 Canny를 생성해 ControlNet에 사용
    - 반환: PIL.Image 리스트 (num_outputs 장)
    """

    pipe = _get_pipe()

    # ----- 이미지 로드 & 리사이즈 -----
    init_img = _load_resize(init_path)
    concept_imgs = [_load_resize(p) for p in concept_paths]

    # ControlNet 이미지 선택: condition_paths[0] 있으면 그것, 없으면 Canny(init)
    control_img: Image.Image
    ref_extra: List[Image.Image] = []
    if len(condition_paths) > 0:
        control_img = _load_resize(condition_paths[0])
        # 나머지는 IP-Adapter 참조로 활용(옵션)
        if len(condition_paths) > 1:
            ref_extra = [_load_resize(p) for p in condition_paths[1:]]
    else:
        control_img = _to_control_image_from_canny(init_img)
        ref_extra = []

    # ----- IP-Adapter 참조 구성 -----
    # Colab 코드: ref_images = [init] + concepts + base_list
    # 여기서는 ref_extra를 "base_list/condition-like"로 취급
    ref_images: List[Image.Image] = [init_img] + concept_imgs + ref_extra
    if not ref_images:
        # 극단 케이스 방지: 최소 init_img는 넣어준다
        ref_images = [init_img]

    # 가중치: init=0.8, concept들=0.8, ref_extra=0.4 (원 코드에 맞춤)
    weights: List[float] = (
        [0.8] + [0.8] * len(concept_imgs) + [0.4] * len(ref_extra)
    )

    # 임베딩 가중합
    combined_embeds = _combine_ipadapter_embeds(pipe, ref_images, weights)

    # 시드
    generator = None
    if seed is not None:
        generator = torch.Generator(device=_device).manual_seed(seed)

    # 어댑터 스케일
    pipe.set_ip_adapter_scale(adapter_scale)

    # ----- 생성 루프 -----
    outs: List[Image.Image] = []
    for _ in range(num_outputs):
        result = pipe(
            prompt=prompt,
            guidance_scale=guidance,
            num_inference_steps=steps,
            ip_adapter_image_embeds=[combined_embeds],  # 중요
            image=control_img,  # ControlNet conditioning image
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
            # i2i를 사용할 경우:
            # image=init_img, strength=0.4,
            generator=generator,
        )
        outs.append(result.images[0])

    # 메모리 힌트(필수는 아님)
    if _device == "cuda":
        torch.cuda.empty_cache()

    return outs
