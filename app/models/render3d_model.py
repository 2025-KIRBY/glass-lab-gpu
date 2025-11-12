import torch
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.rembg import BackgroundRemover

import traceback

# def load_hunyuan_pipeline(device: str = "cuda"):
#     """
#     Hunyuan3D-2 파이프라인을 로드하여 반환
#     """
#     pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
#         "tencent/Hunyuan3D-2",
#         torch_dtype=torch.float16 if device == "cuda" else torch.float32
#     ).to(device)
#     return pipe

_hunyuan_pipeline = None

def load_hunyuan_pipeline(device="cuda"):
    global _hunyuan_pipeline
    try:
        # 이미 로드된 경우 재사용
        if _hunyuan_pipeline is not None:
            print("🔹 Using cached Hunyuan3D pipeline")
            return _hunyuan_pipeline

        print(f"🔹 Loading Hunyuan3D model to {device} ...")
        dtype = torch.float16 if device == "cuda" else torch.float32

        pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2",
            torch_dtype=dtype
        ).to(device)

        _hunyuan_pipeline = pipe
        print("✅ Hunyuan3D pipeline loaded successfully")
        return _hunyuan_pipeline

    except Exception as e:
        print("❌ [load_hunyuan_pipeline] Failed:", e)
        traceback.print_exc()
        _hunyuan_pipeline = None
        return None  # 실패 시 명시적으로 None 반환

def remove_background(image):
    """
    rembg를 이용해 배경 제거
    """
    rembg = BackgroundRemover()
    return rembg(image)
