import torch, os
from PIL import Image
from app.models._3d_modeling import load_hunyuan_pipeline, remove_background

async def render_3d_model(image_path: str) -> str:
    """
    입력 이미지 → 배경 제거 → Hunyuan3D 모델링 → GLB 파일로 저장
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1️⃣ 이미지 로드 및 배경 제거
    image = Image.open(image_path).convert("RGBA")
    image = remove_background(image)

    # 2️⃣ 모델 로드
    pipeline = load_hunyuan_pipeline(device=device)

    # 3️⃣ 3D 변환 수행
    mesh = pipeline(image=image)[0]

    # 4️⃣ 결과 저장
    output_dir = "/workspace/outputs/3d"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "demo.glb")
    mesh.export(output_path)

    return output_path
