from fastapi import FastAPI, UploadFile, Form
from PIL import Image
import os
from models.controlnet_sdxl_model import ControlNetSDXL
import torch

app = FastAPI()
# device = "cuda"
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = ControlNetSDXL(device)
os.makedirs("outputs", exist_ok=True)

@app.get("/")
def root():
    return {"message": "SDXL ControlNet Server is running 🚀"}

@app.post("/generate")
async def generate_image(
    prompt: str = Form(...),
    base: UploadFile = None,
    refs: list[UploadFile] = None
):
    base_img = Image.open(base.file).convert("RGB")
    ref_imgs = [Image.open(r.file).convert("RGB") for r in refs] if refs else []

    # 코랩 weights와 동일
    weights = [0.75, 0.6, 0.6, 0.6][:len(ref_imgs)]

    result = model.generate(prompt, base_img, ref_imgs, weights)
    path = f"outputs/result.png"
    result.save(path)

    return {"status": "ok", "file": path}
