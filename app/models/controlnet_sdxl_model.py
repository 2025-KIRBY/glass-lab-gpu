# # # app/models/controlnet_sdxl_model.py

# from __future__ import annotations
# from typing import List, Tuple, Optional
# from PIL import Image
# import numpy as np
# import torch
# import cv2
# import uuid
# import os

# from diffusers import (
#     StableDiffusionXLControlNetPipeline,
#     ControlNetModel,
#     AutoencoderKL,
# )
# from diffusers.models.attention_processor import AttnProcessor2_0

# # ====== Model Repository IDs ======
# SDXL_BASE_ID = "stabilityai/stable-diffusion-xl-base-1.0"
# VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
# CONTROLNET_UNION_ID = "xinsir/controlnet-union-sdxl-1.0"
# IP_ADAPTER_REPO = "h94/IP-Adapter"
# IP_ADAPTER_SUBFOLDER = "sdxl_models"
# IP_ADAPTER_WEIGHT = "ip-adapter_sdxl.bin"

# # ====== Defaults ======
# PROMPT_DEFAULT = "glasses"
# GUIDANCE = 7.5
# STEPS = 20
# ADAPTER_SCALE = 1.2
# CONTROLNET_CONDITIONING_SCALE = 0.4
# CONTROL_GUIDANCE_START = 0.0
# CONTROL_GUIDANCE_END = 0.4
# IMG_SIZE = 1024


# # ====== Util ======
# def _resize_1024(im: Image.Image) -> Image.Image:
#     if im.size == (IMG_SIZE, IMG_SIZE):
#         return im
#     return im.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)


# def _make_control_from_init(init_rgb_1024: Image.Image) -> Image.Image:
#     """init → Canny Edge → Dilation → RGB PIL"""
#     image_np = np.array(init_rgb_1024)
#     edges = cv2.Canny(image_np, 100, 200)
#     kernel = np.ones((3, 3), np.uint8)
#     edges_thick = cv2.dilate(edges, kernel, iterations=1)
#     control = edges_thick[:, :, None]
#     control = np.concatenate([control] * 3, axis=2)
#     return Image.fromarray(control)


# # ============================================================
# # 🧠 GlassLabSDXL: Pipeline Wrapper
# # ============================================================
# class GlassLabSDXL:
#     def __init__(
#         self,
#         device: Optional[str] = None,
#         torch_dtype: torch.dtype = torch.float16,
#         cache_dir: Optional[str] = None,
#     ) -> None:
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         self.dtype = torch_dtype
#         self.cache_dir = cache_dir

#         print("[GlassLabSDXL] 🚀 Loading base pipeline components...")

#         # --- Load core models ---
#         vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=self.dtype, cache_dir=self.cache_dir)
#         controlnet = ControlNetModel.from_pretrained(CONTROLNET_UNION_ID, torch_dtype=self.dtype, cache_dir=self.cache_dir)

#         self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
#             SDXL_BASE_ID,
#             controlnet=controlnet,
#             vae=vae,
#             torch_dtype=self.dtype,
#             use_safetensors=True,
#             cache_dir=self.cache_dir,
#         ).to(self.device)

#         # --- Load IP-Adapter ---
#         self.pipe.load_ip_adapter(
#             IP_ADAPTER_REPO,
#             subfolder=IP_ADAPTER_SUBFOLDER,
#             weight_name=IP_ADAPTER_WEIGHT,
#             cache_dir=self.cache_dir,
#         )

#         # --- Disable xFormers (conflict with IP-Adapter) ---
#         try:
#             self.pipe.disable_xformers_memory_efficient_attention()
#         except Exception:
#             pass

#         # --- Force SDPA Attention Processor (Torch 2.0 default) ---
#         self.pipe.unet.set_attn_processor(AttnProcessor2_0())
#         if hasattr(self.pipe, "controlnet"):
#             self.pipe.controlnet.set_attn_processor(AttnProcessor2_0())

#         # --- Memory optimization ---
#         self.pipe.enable_attention_slicing("max")
#         self.pipe.enable_vae_slicing()
#         self.pipe.enable_vae_tiling()
#         self.pipe.set_progress_bar_config(disable=True)

#         # --- Debug info ---
#         any_proc = next(iter(self.pipe.unet.attn_processors.values()))
#         print(f"[GlassLabSDXL] ✅ Attention Processor: {type(any_proc).__name__}")
#         print("[GlassLabSDXL] ✅ Pipeline ready for generation.")

#     # ============================================================
#     # Image Generation
#     # ============================================================
#     def generate_batch(
#         self,
#         init_image: Image.Image,
#         concept_images: List[Image.Image],
#         condition_images: List[Image.Image],
#         num_images: int = 6,
#         seed: Optional[int] = None,
#         prompt: str = PROMPT_DEFAULT,
#         guidance_scale: float = GUIDANCE,
#         steps: int = STEPS,
#         adapter_scale: float = ADAPTER_SCALE,
#         controlnet_conditioning_scale: float = CONTROLNET_CONDITIONING_SCALE,
#         control_guidance_start: float = CONTROL_GUIDANCE_START,
#         control_guidance_end: float = CONTROL_GUIDANCE_END,
#     ) -> Tuple[List[Image.Image], str]:
#         """
#         init 1장 + concept n장(2~5) + condition 5장 입력 → n개 이미지 생성
#         """
#         # 1️⃣ Preprocess inputs
#         init_1024 = _resize_1024(init_image.convert("RGB"))
#         concepts_1024 = [_resize_1024(im.convert("RGB")) for im in concept_images]
#         conds_1024 = [_resize_1024(im.convert("RGB")) for im in condition_images]

#         # 2️⃣ ControlNet input from init
#         control_image = _make_control_from_init(init_1024)

#         # 3️⃣ IP-Adapter reference images 및 가중치 정의
#         ref_images = [init_1024] + concepts_1024 + conds_1024
#         weights = [0.8] + [0.8] * len(concepts_1024) + [0.4] * len(conds_1024)
        
#         # 4️⃣ Random generator
#         if seed is not None:
#             # 시드 기반으로 매번 다른 이미지를 생성하려면 루프 밖에서 gen을 한 번 정의하고,
#             # 루프 내에서 .manual_seed_all() 등으로 시드를 변경해야 하나,
#             # 현재는 하나의 gen을 정의하고 파이프라인에 전달합니다.
#             gen = torch.Generator(device=self.device).manual_seed(int(seed))
#         else:
#             gen = torch.Generator(device=self.device) # 시드가 없으면 장치 기본 시드를 사용

#         # =======================================================
#         # ⭐️ 5️⃣ IP-Adapter 임베딩 통합 로직 (수정된 핵심 부분) ⭐️
#         # =======================================================
        
#         # 5-A) 각 이미지별 임베딩 추출 (2, N, C)
#         embeds_list = []
#         for image in ref_images:
#             # 주의: ip_adapter_image에 PIL.Image 객체를 넣어주려면 리스트가 아닌 PIL.Image 객체여야 합니다.
#             # 하지만 파이프라인 내부 로직상 이미지를 리스트로 감싸서 전달하는 것이 안전합니다.
#             # 만약 [image]로 넣어서 에러가 난다면, 이미지를 리스트로 감싸지 않고 PIL.Image 객체 그대로 전달해 보세요.
#             # 여기서는 이전 코드의 1장 리스트 방식을 따릅니다.
#             embed = self.pipe.prepare_ip_adapter_image_embeds(
#                 ip_adapter_image=[image], 
#                 ip_adapter_image_embeds=None,
#                 device=self.device,
#                 num_images_per_prompt=1,
#                 do_classifier_free_guidance=True
#             )
#             # embed: (2, N, C) 텐서 (cond, uncond)
#             # embeds_list.append(embed)  -> 리스트로 받으면 에러남. 명시적으로 Tensor 변환
#             if isinstance(embed, (list, tuple)) and len(embed) > 0 and isinstance(embed[0], torch.Tensor):
#                 # 만약 반환 값이 [Tensor] 형태라면, Tensor를 추출
#                 embeds_list.append(embed[0])
#             elif isinstance(embed, torch.Tensor):
#                 # 반환 값이 바로 Tensor라면, 그대로 저장
#                 embeds_list.append(embed)
#             else:
#                 # 예상치 못한 형식이라면 에러 처리 또는 디버깅 로깅 추가
#                 raise TypeError(f"prepare_ip_adapter_image_embeds returned unexpected type: {type(embed)}")

#         # 5-B) 가중치 정의 및 텐서 변환
#         # (2, N, C) 형태의 임베딩 텐서들을 (num_images, 2, N, C)로 쌓음
#         stacked_embeds = torch.stack(embeds_list, dim=0) 
        
#         # 가중치 텐서를 PyTorch로 변환 및 차원 확장 (num_images, 1, 1, 1)
#         weights_tensor = torch.tensor(weights, device=self.device, dtype=stacked_embeds.dtype)
#         weights_expanded = weights_tensor.view(-1, 1, 1, 1)
        
#         # 5-C) 가중 평균 임베딩 계산
#         # 가중치 곱: (num_images, 2, N, C) * (num_images, 1, 1, 1)
#         weighted_sum = torch.sum(stacked_embeds * weights_expanded, dim=0) 
#         total_weight = torch.sum(weights_tensor)
        
#         # 최종 통합 텐서: (2, N, C) - 조건 임베딩과 비조건 임베딩 쌍
#         combined_embeds = weighted_sum / total_weight 

#         if not isinstance(combined_embeds, torch.Tensor):
#             # 이 시점에서는 이미 가중 평균 연산이 끝났으므로, Tensor가 아닌 것은 심각한 오류입니다.
#             # 하지만 튜플이나 리스트로 남아있는 경우를 대비하여 다시 Tensor로 변환을 시도합니다.
#             if isinstance(combined_embeds, (list, tuple)) and isinstance(combined_embeds[0], torch.Tensor):
#                  combined_embeds = combined_embeds[0] # 튜플/리스트의 첫 번째 요소를 사용 (가장 유력한 문제 해결)
#             else:
#                 # 마지막 비상 가드: 최종 타입 불일치 시 오류 발생
#                 raise TypeError(f"Final combined_embeds is still not a torch.Tensor. Type: {type(combined_embeds)}")

#         # =======================================================
#         # ⭐️ 6️⃣ 생성 루프 적용 ⭐️
#         # =======================================================
#         self.pipe.set_ip_adapter_scale(adapter_scale)
#         images: List[Image.Image] = []
        
#         # 참고: num_images가 1보다 클 경우, gen을 루프 내에서 seed를 변경하며 사용해야
#         # 결과 이미지가 달라집니다. 현재는 하나의 gen을 사용하므로 같은 이미지를 num_images만큼 생성할 수 있습니다.
#         # 시드 변경 로직은 복잡해지므로, 현재 코드 구조를 유지합니다.

#         for _ in range(num_images):
#             out = self.pipe(
#                 prompt=prompt,
#                 guidance_scale=guidance_scale,
#                 num_inference_steps=steps,
#                 generator=gen,
                
#                 # ✅ 수정: 통합된 임베딩 텐서를 리스트에 넣어 전달
#                 ip_adapter_image_embeds=[combined_embeds], 
#                 # ✅ 원본 코드의 에러 인자 제거: iip_adapter_image_embeds -> ip_adapter_image_embeds로 수정되었을 것입니다.
#                 # ip_adapter_scale 인자는 pipe.set_ip_adapter_scale로 이미 설정되었으므로 중복이지만, 
#                 # 파이프라인 구현에 따라 안전을 위해 유지합니다.
#                 ip_adapter_scale=adapter_scale,
                
#                 image=control_image,
#                 controlnet_conditioning_scale=controlnet_conditioning_scale,
#                 control_guidance_start=control_guidance_start,
#                 control_guidance_end=control_guidance_end,
#             ).images[0]
#             images.append(out)

#         if self.device == "cuda":
#             torch.cuda.empty_cache()

#         job_id = uuid.uuid4().hex[:12]
#         return images, job_id


# # ============================================================
# # Global Singleton Loader
# # ============================================================
# _global_pipe: Optional[GlassLabSDXL] = None


# def get_pipe() -> GlassLabSDXL:
#     """
#     Lazy singleton: only load pipeline once at startup
#     """
#     global _global_pipe
#     if _global_pipe is None:
#         print("[INIT] Loading GlassLabSDXL pipeline (first time)...")
#         _global_pipe = GlassLabSDXL()
#     return _global_pipe





# from __future__ import annotations
# from typing import List, Tuple, Optional
# from PIL import Image
# import numpy as np
# import torch
# import cv2
# import uuid
# import gc

# from diffusers import (
#     StableDiffusionXLControlNetPipeline,
#     ControlNetModel,
#     AutoencoderKL,
# )
# from diffusers.models.attention_processor import AttnProcessor2_0

# # --- Model IDs ---
# SDXL_BASE_ID = "stabilityai/stable-diffusion-xl-base-1.0"
# VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
# CONTROLNET_UNION_ID = "xinsir/controlnet-union-sdxl-1.0"
# IP_ADAPTER_REPO = "h94/IP-Adapter"
# IP_ADAPTER_SUBFOLDER = "sdxl_models"
# IP_ADAPTER_WEIGHT = "ip-adapter_sdxl.bin"

# IMG_SIZE = 1024


# # --- Utilities ---
# def _resize_1024(im: Image.Image) -> Image.Image:
#     """Resize to 1024x1024 RGB"""
#     im = im.convert("RGB")
#     if im.size != (IMG_SIZE, IMG_SIZE):
#         im = im.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
#     return im


# def _make_control_from_init(init_img: Image.Image) -> Image.Image:
#     """Generate ControlNet conditioning from init image"""
#     np_img = np.array(init_img)
#     edges = cv2.Canny(np_img, 100, 200)
#     edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
#     control = np.repeat(edges[:, :, None], 3, axis=2)
#     return Image.fromarray(control)


# # ============================================================
# # 🧠 GlassLabSDXL Model Wrapper
# # ============================================================
# class GlassLabSDXL:
#     def __init__(self, device: Optional[str] = None, cache_dir: Optional[str] = None):
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         self.dtype = torch.float16
#         self.cache_dir = cache_dir

#         print("[GlassLabSDXL] 🚀 Loading pipeline...")

#         # --- Load base components ---
#         vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=self.dtype)
#         controlnet = ControlNetModel.from_pretrained(CONTROLNET_UNION_ID, torch_dtype=self.dtype)
#         self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
#             SDXL_BASE_ID,
#             controlnet=controlnet,
#             vae=vae,
#             torch_dtype=self.dtype,
#             use_safetensors=True,
#         ).to(self.device)

#         # --- Load IP Adapter ---
#         self.pipe.load_ip_adapter(
#             IP_ADAPTER_REPO,
#             subfolder=IP_ADAPTER_SUBFOLDER,
#             weight_name=IP_ADAPTER_WEIGHT,
#         )

#         # --- Attention setup ---
#         try:
#             self.pipe.disable_xformers_memory_efficient_attention()
#         except Exception:
#             pass
#         self.pipe.unet.set_attn_processor(AttnProcessor2_0())
#         if hasattr(self.pipe, "controlnet"):
#             self.pipe.controlnet.set_attn_processor(AttnProcessor2_0())

#         # --- Memory optimizations ---
#         self.pipe.enable_attention_slicing("max")
#         self.pipe.enable_vae_tiling()
#         self.pipe.set_progress_bar_config(disable=True)

#         print("[GlassLabSDXL] ✅ Model loaded successfully.")

#     # --------------------------------------------------------
#     # 🔹 Combine IP Adapter embeddings (weighted average)
#     # --------------------------------------------------------
#     def _combine_ip_adapter_embeds(self, ref_images: List[Image.Image], weights: List[float]):
#         embeds_list = []
#         for img in ref_images:
#             emb = self.pipe.prepare_ip_adapter_image_embeds(
#                 ip_adapter_image=img,
#                 ip_adapter_image_embeds=None,
#                 device=self.device,
#                 num_images_per_prompt=1,
#                 do_classifier_free_guidance=True,
#             )
#             embeds_list.append(emb[0])  # (2, N, C)

#         pos = torch.stack([e[0] for e in embeds_list], dim=0)
#         neg = torch.stack([e[1] for e in embeds_list], dim=0)
#         w = torch.tensor(weights, device=pos.device, dtype=pos.dtype).view(-1, 1, 1)

#         weighted_pos = (pos * w).sum(dim=0, keepdim=True)
#         weighted_neg = (neg * w).sum(dim=0, keepdim=True)
#         return torch.cat([weighted_pos, weighted_neg], dim=0)  # (2, N, C)

#     # --------------------------------------------------------
#     # 🔹 Main Generation Function
#     # --------------------------------------------------------
#     @torch.inference_mode()
#     def generate_batch(
#         self,
#         init_image: Image.Image,
#         concept_images: List[Image.Image],
#         condition_images: List[Image.Image],
#         num_images: int = 4,
#         prompt: str = "glasses",
#         guidance_scale: float = 7.5,
#         steps: int = 20,
#         adapter_scale: float = 1.2,
#         controlnet_conditioning_scale: float = 0.4,
#         control_guidance_start: float = 0.0,
#         control_guidance_end: float = 0.4,
#         seed: Optional[int] = None,
#     ) -> Tuple[List[Image.Image], str]:
#         """Generate images with weighted IP-Adapter reference"""
#         init_1024 = _resize_1024(init_image)
#         concepts_1024 = [_resize_1024(im) for im in concept_images]
#         conds_1024 = [_resize_1024(im) for im in condition_images]

#         control_image = _make_control_from_init(init_1024)

#         # Reference set
#         ref_images = [init_1024] + concepts_1024 + conds_1024
#         weights = [0.8] + [0.8] * len(concepts_1024) + [0.4] * len(conds_1024)

#         # IP Adapter embed combine
#         combined_embeds = self._combine_ip_adapter_embeds(ref_images, weights)
#         self.pipe.set_ip_adapter_scale(adapter_scale)

#         gen = torch.Generator(device=self.device)
#         if seed is not None:
#             gen.manual_seed(int(seed))

#         outputs = []
#         for _ in range(num_images):
#             out = self.pipe(
#                 prompt=prompt,
#                 ip_adapter_image_embeds=[combined_embeds],
#                 image=control_image,
#                 controlnet_conditioning_scale=controlnet_conditioning_scale,
#                 control_guidance_start=control_guidance_start,
#                 control_guidance_end=control_guidance_end,
#                 guidance_scale=guidance_scale,
#                 num_inference_steps=steps,
#                 generator=gen,
#             ).images[0]
#             outputs.append(out)

#         # Memory cleanup
#         self.pipe._clear_ip_adapter_image_embeds()
#         torch.cuda.empty_cache()
#         gc.collect()

#         job_id = uuid.uuid4().hex[:12]
#         return outputs, job_id


# # ============================================================
# # 🔹 Global Singleton Loader
# # ============================================================
# _global_pipe: Optional[GlassLabSDXL] = None

# def get_pipe() -> GlassLabSDXL:
#     global _global_pipe
#     if _global_pipe is None:
#         print("[INIT] Loading global GlassLabSDXL pipeline...")
#         _global_pipe = GlassLabSDXL()
#     return _global_pipe



# # app/models/controlnet_sdxl_model.py
# from __future__ import annotations
# from typing import List, Optional, Tuple
# from PIL import Image
# import numpy as np
# import torch, cv2, gc, uuid
# from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL


# class ControlNetSDXL:
#     def __init__(self, device: str = "cuda"):
#         print("🔹 Loading SDXL ControlNet + IP-Adapter pipeline...")
#         self.device = device

#         # --- Load core models ---
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

#         # --- Load IP Adapter ---
#         self.pipe.load_ip_adapter(
#             "h94/IP-Adapter",
#             subfolder="sdxl_models",
#             weight_name="ip-adapter_sdxl.bin"
#         )

#         try:
#             self.pipe.disable_xformers_memory_efficient_attention()
#         except Exception:
#             pass

#         self.pipe.enable_attention_slicing("max")
#         self.pipe.enable_vae_tiling()
#         self.pipe.enable_vae_slicing()
#         self.pipe.set_progress_bar_config(disable=True)

#         print("✅ Model load complete.")

#     # ============================================================
#     # Utility: Control Image (Canny + Dilation)
#     # ============================================================
#     def get_control_image(self, init_image: Image.Image) -> Image.Image:
#         img_np = np.array(init_image.convert("RGB"))
#         edges = cv2.Canny(img_np, 100, 200)
#         kernel = np.ones((3, 3), np.uint8)
#         edges_thick = cv2.dilate(edges, kernel, iterations=1)
#         control_img = np.concatenate([edges_thick[:, :, None]] * 3, axis=2)
#         return Image.fromarray(control_img)

#     # ============================================================
#     # Utility: Weighted Embedding (IP Adapter)
#     # ============================================================
#     def set_adapter(self, image_list: List[Image.Image], weights: List[float]) -> torch.Tensor:
#         embeds_list = []

#         for img in image_list:
#             embeds = self.pipe.prepare_ip_adapter_image_embeds(
#                 ip_adapter_image=img,
#                 ip_adapter_image_embeds=None,
#                 device=self.device,
#                 num_images_per_prompt=1,
#                 do_classifier_free_guidance=True
#             )

#             # ✅ 최신 diffusers: [(cond, uncond)] 구조 flatten
#             if isinstance(embeds, (list, tuple)) and isinstance(embeds[0], (list, tuple)):
#                 embeds = embeds[0]

#             embeds_list.append(embeds)

#         # cond/uncond 각각 스택
#         pos = torch.stack([e[0] for e in embeds_list], dim=0)  # (n, N, C)
#         neg = torch.stack([e[1] for e in embeds_list], dim=0)  # (n, N, C)

#         w = torch.tensor(weights, device=pos.device, dtype=pos.dtype).view(-1, 1, 1)
#         weighted_pos = (pos * w).sum(dim=0, keepdim=True)
#         weighted_neg = (neg * w).sum(dim=0, keepdim=True)

#         combined = torch.cat([weighted_pos, weighted_neg], dim=0)  # (2, N, C)
#         return combined

#     # ============================================================
#     # Main Generate Function
#     # ============================================================
#     def generate(
#         self,
#         prompt: str,
#         base_img: Image.Image,
#         ref_imgs: List[Image.Image],
#         weights: List[float],
#         steps: int = 20,
#         guidance: float = 7.5,
#         seed: int = 1234
#     ) -> Image.Image:
#         control_image = self.get_control_image(base_img)
#         embeds = self.set_adapter(ref_imgs, weights)

#         self.pipe.set_ip_adapter_scale(1.1)
#         generator = torch.Generator(device=self.device).manual_seed(seed)

#         out = self.pipe(
#             prompt=prompt,
#             guidance_scale=guidance,
#             num_inference_steps=steps,
#             ip_adapter_image_embeds=[embeds],
#             image=control_image,
#             controlnet_conditioning_scale=0.4,
#             control_guidance_start=0.0,
#             control_guidance_end=0.4,
#             generator=generator
#         ).images[0]

#         gc.collect()
#         if self.device == "cuda":
#             torch.cuda.empty_cache()

#         return out


# # ============================================================
# # Global Singleton Loader
# # ============================================================
# _global_model: Optional[ControlNetSDXL] = None


# def get_pipe() -> ControlNetSDXL:
#     global _global_model
#     if _global_model is None:
#         print("[INIT] Loading ControlNetSDXL pipeline (once)...")
#         _global_model = ControlNetSDXL()
#     return _global_model


from __future__ import annotations
from typing import List, Optional, Tuple
from PIL import Image
import numpy as np
import torch, cv2, gc, uuid
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL


class ControlNetSDXL:
    def __init__(self, device: str = "cuda"):
        print("🔹 Loading SDXL ControlNet + IP-Adapter pipeline...")
        self.device = device

        # --- Load core models ---
        self.vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16
        )

        self.controlnet = ControlNetModel.from_pretrained(
            "xinsir/controlnet-union-sdxl-1.0",
            torch_dtype=torch.float16
        )

        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=self.controlnet,
            vae=self.vae,
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to(self.device)

        # --- Load IP Adapter ---
        self.pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter_sdxl.bin"
        )

        try:
            # ⭐️ Xformers 비활성화 로직은 남겨둡니다 ⭐️
            self.pipe.disable_xformers_memory_efficient_attention()
        except Exception:
            pass
        
        # Attention Slicing은 메모리 절약에 중요
        self.pipe.enable_attention_slicing("max")
        self.pipe.enable_vae_tiling()
        self.pipe.enable_vae_slicing()
        self.pipe.set_progress_bar_config(disable=True)

        print("✅ Model load complete.")

    # ============================================================
    # Utility: Control Image (Canny + Dilation)
    # ============================================================
    def get_control_image(self, init_image: Image.Image) -> Image.Image:
        img_np = np.array(init_image.convert("RGB"))
        edges = cv2.Canny(img_np, 100, 200)
        kernel = np.ones((3, 3), np.uint8)
        edges_thick = cv2.dilate(edges, kernel, iterations=1)
        control_img = np.concatenate([edges_thick[:, :, None]] * 3, axis=2)
        return Image.fromarray(control_img)

    # ============================================================
    # Utility: Weighted Embedding (IP Adapter)
    # ============================================================
    def set_adapter(self, image_list: List[Image.Image], weights: List[float]) -> torch.Tensor:
        embeds_list = []

        for img in image_list:
            embeds = self.pipe.prepare_ip_adapter_image_embeds(
                ip_adapter_image=[img], # ⭐️ List로 감싸는 것이 가장 안정적인 형태이므로 유지 ⭐️
                ip_adapter_image_embeds=None,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True
            )

            # ⭐️ Tensor 추출 로직: 튜플/리스트 언래핑 강제 ⭐️
            while not isinstance(embeds, torch.Tensor):
                 if isinstance(embeds, (list, tuple)) and embeds:
                    embeds = embeds[0] 
                 else:
                    raise TypeError("IP Adapter returned an empty or invalid format.")

            # 이제 embeds는 (2, N, C) 형태의 Tensor여야 합니다.
            embeds_list.append(embeds)

        # cond/uncond 각각 스택
        pos = torch.stack([e[0] for e in embeds_list], dim=0)  # (n, N, C)
        neg = torch.stack([e[1] for e in embeds_list], dim=0)  # (n, N, C)

        w = torch.tensor(weights, device=pos.device, dtype=pos.dtype).view(-1, 1, 1)
        weighted_pos = (pos * w).sum(dim=0, keepdim=True)
        weighted_neg = (neg * w).sum(dim=0, keepdim=True)

        combined = torch.cat([weighted_pos, weighted_neg], dim=0)  # (2, N, C)
        print("combined type: ", type(combined), combine.shape)
        return combined

    # ============================================================
    # Main Generate Function
    # ============================================================
    def generate(
        self,
        prompt: str,
        base_img: Image.Image,
        ref_imgs: List[Image.Image],
        weights: List[float],
        steps: int = 20,
        guidance: float = 7.5,
        seed: int = 1234
    ) -> Image.Image:
        control_image = self.get_control_image(base_img)
        embeds = self.set_adapter(ref_imgs, weights)

        self.pipe.set_ip_adapter_scale(1.1)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        out = self.pipe(
            prompt=prompt,
            guidance_scale=guidance,
            num_inference_steps=steps,
            ip_adapter_image_embeds=[embeds],
            image=control_image,
            controlnet_conditioning_scale=0.4,
            control_guidance_start=0.0,
            control_guidance_end=0.4,
            generator=generator
        ).images[0]

        # 생성 후 메모리 정리
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return out


# ============================================================
# Global Singleton Loader
# ============================================================
_global_model: Optional[ControlNetSDXL] = None


def get_pipe() -> ControlNetSDXL:
    global _global_model
    if _global_model is None:
        print("[INIT] Loading ControlNetSDXL pipeline (once)...")
        _global_model = ControlNetSDXL()
    return _global_model
