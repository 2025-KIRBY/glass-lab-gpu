# app/models/controlnet_sdxl_model.py

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

        self.size = (1024, 1024)

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
        # self.pipe.enable_attention_slicing("max")
        # self.pipe.enable_vae_tiling()
        # self.pipe.enable_vae_slicing()
        # self.pipe.set_progress_bar_config(disable=True)

        print("✅ Model load complete.")

    def _resize_1024(self, img: Image.Image) -> Image.Image:
        return img.convert("RGB").resize(self.size, Image.LANCZOS)

    # ============================================================
    # Utility: Control Image (Canny + Dilation)
    # ============================================================
    def get_control_image(self, init_image: Image.Image) -> Image.Image:
        init_image = self._resize_1024(init_image)

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
                    # print("combined type: ", type(embeds))
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
        print("combined type: ", type(combined), combined.shape)
        return [combined]

    # ============================================================
    # Main Generate Function
    # ============================================================
    def generate(
        self,
        prompt: str,
        base_img: Image.Image,
        ref_imgs: List[Image.Image],
        weights: List[float],
        steps: int = 40,
        guidance: float = 7.5,
        seed: int = 1234,
        controlnet_condition_scale: float,
        control_guidance_end: float
    ) -> Image.Image:
        base_img = self._resize_1024(base_img)
        ref_imgs = [self._resize_1024(img) for img in ref_imgs]
        
        control_image = self.get_control_image(base_img)
        embeds = self.set_adapter(ref_imgs, weights)

        self.pipe.set_ip_adapter_scale(1.2)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        out = self.pipe(
            prompt=prompt,
            guidance_scale=guidance,
            num_inference_steps=steps,
            ip_adapter_image_embeds= embeds,
            image=control_image,
            controlnet_conditioning_scale=controlnet_condition_scale,
            control_guidance_start=0.0,
            control_guidance_end=control_guidance_end,
            #generator=generator
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
