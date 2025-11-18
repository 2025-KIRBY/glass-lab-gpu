# # import torch, os
# # from PIL import Image
# # from app.models.render3d_model import load_hunyuan_pipeline, remove_background

# # async def render_3d_model(image_path: str) -> str:
# #     """
# #     입력 이미지 → 배경 제거 → Hunyuan3D 모델링 → GLB 파일로 저장
# #     """
# #     device = "cuda" if torch.cuda.is_available() else "cpu"

# #     # 1️⃣ 이미지 로드 및 배경 제거
# #     image = Image.open(image_path).convert("RGBA")
# #     image = remove_background(image)

# #     # 2️⃣ 모델 로드
# #     pipeline = load_hunyuan_pipeline(device=device)

# #     # 3️⃣ 3D 변환 수행
# #     mesh = pipeline(image=image)[0]

# #     # 4️⃣ 결과 저장
# #     output_dir = "/workspace/outputs/3d"
# #     os.makedirs(output_dir, exist_ok=True)
# #     output_path = os.path.join(output_dir, "demo.glb")
# #     mesh.export(output_path)

# #     return output_path
# import torch, os, traceback
# from PIL import Image
# from app.models.render3d_model import load_hunyuan_pipeline, remove_background

# async def render_3d_model(image_path: str) -> str:
#     """
#     입력 이미지 → 배경 제거 → Hunyuan3D 모델링 → GLB 파일로 저장
#     """
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     try:
#         # 1️⃣ 이미지 로드 및 배경 제거
#         print(f"🔹 Loading input image: {image_path}")
#         image = Image.open(image_path).convert("RGBA")
#         image = remove_background(image)

#         # 2️⃣ 모델 로드
#         print(f"🔹 Loading Hunyuan3D pipeline on {device} ...")
#         pipeline = load_hunyuan_pipeline(device=device)

#         # ✅ pipeline이 None이거나 함수형이 아닌 경우 대비
#         if pipeline is None:
#             raise RuntimeError("load_hunyuan_pipeline() returned None")
#         if not callable(pipeline):
#             raise TypeError(f"Invalid pipeline object: {type(pipeline)} (not callable)")

#         # 3️⃣ 3D 변환 수행
#         print("🔹 Running 3D generation...")
#         mesh = pipeline(image=image)[0]

#         # ✅ mesh 타입 검사
#         if mesh is None:
#             raise RuntimeError("pipeline(image=image) returned None mesh")

#         # 4️⃣ 결과 저장
#         output_dir = "/workspace/outputs/3d"
#         os.makedirs(output_dir, exist_ok=True)
#         output_path = os.path.join(output_dir, "demo.glb")
#         mesh.export(output_path)

#         print(f"✅ 3D model exported successfully → {output_path}")
#         return output_path

#     except Exception as e:
#         print("❌ [Render3D Error]", e)
#         traceback.print_exc()
#         raise e
