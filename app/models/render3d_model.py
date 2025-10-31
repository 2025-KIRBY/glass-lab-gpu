import torch
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.rembg import BackgroundRemover

def load_hunyuan_pipeline(device: str = "cuda"):
    """
    Hunyuan3D-2 파이프라인을 로드하여 반환
    """
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    return pipe

def remove_background(image):
    """
    rembg를 이용해 배경 제거
    """
    rembg = BackgroundRemover()
    return rembg(image)
