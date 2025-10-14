from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image
from PIL import Image
import numpy as np
import torch, cv2, time, gc, os

class ControlNetSDXL:
    def __init__(self, device="cuda"):
        print("🔹 Loading SDXL ControlNet + IP Adapter pipeline...")
        self.device = device
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

        # IP Adapter 로드
        self.pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter_sdxl.bin"
        )

        print("✅ Model load complete.")

    # --- IP Adapter embedding 가중합 ---
    def set_adapter(self, image_list, weights):
        embeds_list = []
        for img in image_list:
            embed = self.pipe.prepare_ip_adapter_image_embeds(
                ip_adapter_image=img,
                ip_adapter_image_embeds=None,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True
            )
            embeds_list.append(embed[0])

        pos = torch.stack([e[0] for e in embeds_list], dim=0)
        neg = torch.stack([e[1] for e in embeds_list], dim=0)
        w = torch.tensor(weights, device=pos.device, dtype=pos.dtype).view(-1, 1, 1)

        weighted_pos = (pos * w).sum(dim=0, keepdim=True)
        weighted_neg = (neg * w).sum(dim=0, keepdim=True)
        combined = torch.cat([weighted_pos, weighted_neg], dim=0)
        return combined

    # --- Canny 엣지 생성 ---
    def get_control_image(self, init_image):
        img_np = np.array(init_image)
        edges = cv2.Canny(img_np, 100, 200)
        kernel = np.ones((3, 3), np.uint8)
        edges_thick = cv2.dilate(edges, kernel, iterations=1)
        control_img = np.concatenate([edges_thick[:, :, None]] * 3, axis=2)
        control_img = Image.fromarray(control_img)
        return control_img

    # --- 메인 생성 함수 ---
    def generate(self, prompt, base_img, ref_imgs, weights):
        control_image = self.get_control_image(base_img)
        embeds = self.set_adapter(ref_imgs, weights)
        self.pipe.set_ip_adapter_scale(1.1)

        generator = torch.Generator(device=self.device).manual_seed(1234)

        out = self.pipe(
            prompt=prompt,
            guidance_scale=7.5,
            num_inference_steps=40,
            ip_adapter_image_embeds=[embeds],
            image=control_image,
            controlnet_conditioning_scale=0.4,
            control_guidance_start=0.0,
            control_guidance_end=0.4,
            generator=generator
        ).images[0]

        gc.collect()
        torch.cuda.empty_cache()

        return out
