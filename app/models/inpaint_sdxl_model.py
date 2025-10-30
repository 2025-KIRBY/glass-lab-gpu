from __future__ import annotations
from typing import List, Optional, Tuple
from PIL import Image
import numpy as np, torch, cv2, gc, uuid, time
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel, AutoencoderKL

class InpaintSDXL:
    def __init__(self, device: str = "cuda"):
        print("🔹 Loading SDXL Inpainting + ControlNet + IP-Adapter pipeline...")
        self.device = device

        # 모델들 불러오기 
        self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

        self.controlnet = ControlNetModel.from_pretrained("xinsir/controlnet-union-sdxl-1.0", torch_dtype=torch.float16)

        self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-inpainting-1.0",
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
            self.pipe.disable_xformers_memory_efficient_attention()
        except Exception:
            pass

        print("✅ Inpainting pipeline loaded successfully.")

    # ============================================================
    # Utility: Control Image (Canny + Dilation)
    # ============================================================
    # control image
    def get_control_image(self, base_image: Image.Image) -> Image.Image:
        img_np = np.array(base_image.convert("RGB"))
        edges = cv2.Canny(img_np, 100, 200)
        kernel = np.ones((3, 3), np.uint8)
        edges_thick = cv2.dilate(edges, kernel, iterations=1)
        control_img = np.concatenate([edges_thick[:, :, None]] * 3, axis=2)
        return Image.fromarray(control_img)

    # ============================================================
    # Utility: Weighted Embedding (IP Adapter)
    # ============================================================
    # ip-adapter weighted embed
    def set_adapter(self, image_list: List[Image.Image], weights: List[float]) -> List[torch.Tensor]:
        embeds_list = []

        for img in image_list:
            embeds = self.pipe.prepare_ip_adapter_image_embeds(
                ip_adapter_image=[img],
                ip_adapter_image_embeds=None,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True
            )
            while not isinstance(embeds, torch.Tensor):
                if isinstance(embeds, (list, tuple)) and embeds:
                    embeds = embeds[0]
                else:
                    raise TypeError("Invalid IP-Adapter output format.")
            embeds_list.append(embeds)

        pos = torch.stack([e[0] for e in embeds_list], dim=0)
        neg = torch.stack([e[1] for e in embeds_list], dim=0)
        w = torch.tensor(weights, device=pos.device, dtype=pos.dtype).view(-1, 1, 1)
        weighted_pos = (pos * w).sum(dim=0, keepdim=True)
        weighted_neg = (neg * w).sum(dim=0, keepdim=True)
        combined = torch.cat([weighted_pos, weighted_neg], dim=0)
        return [combined]

    # ============================================================
    # Main Generate Function
    # ============================================================
    def inpaint_generate(
        self,
        prompt: str,
        base_img: Image.Image,
        mask_img: Image.Image,
        ref_imgs: List[Image.Image],
        weights: List[float],
        steps: int = 20,
        guidance: float = 7.5,
        seed: int = 1234,
        # num_outputs: int = 5 -> 반복은 서비스 계층에서 시키는 걸로!
    ) -> Image.Image:
        """호출 당 이미지 1장 인페인팅해서 새로운 이미지 생성!"""
        control_img = self.get_control_image(base_img)
        embeds = self.set_adapter(ref_imgs, weights)

        self.pipe.set_ip_adapter_scale(1.2)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # results = []
        for i in range(num_outputs):
            out = self.pipe(
                prompt=prompt,
                guidance_scale=guidance,
                num_inference_steps=steps,
                ip_adapter_image_embeds=embeds,
                control_image=control_img,
                controlnet_conditioning_scale=0.2,
                control_guidance_start=0.0,
                control_guidance_end=0.15,
                image=base_img,
                mask_image=mask_img,
                generator=generator
            ).images[0]
            # results.append(out)

        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            
        return out

# ==============================================
_global_model: Optional[InpaintSDXL] = None

def get_pipe() -> InpaintSDXL:
    global _global_model
    if _global_model is None:
        print("[INIT] Loading InpaintSDXL pipeline (once)...")
        _global_model = InpaintSDXL()
    return _global_model