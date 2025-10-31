# app/api/endpoints/render3d.pyß
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import time, os, uuid
from app.services.render3d_service import render_3d_model

router = APIRouter(prefix="/api/v1", tags=["3D Render"])

@router.post("/render3d")
async def render_3d(final_image: UploadFile = File(...)):
    try:
        start_time = time.time()
        if not final_image:
            raise HTTPException(status_code=400, detail="final_image가 누락되었습니다.")
        
        # 1️⃣ 파일 저장
        upload_dir = "/workspace/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        filename = f"{uuid.uuid4().hex}_{final_image.filename}"
        file_path = os.path.join(upload_dir, filename)
        with open(file_path, "wb") as f:
            f.write(await final_image.read())
        
        # 2️⃣ 3D 변환 서비스 실행
        output_path = await render_3d_model(file_path)
        elapsed = round(time.time() - start_time, 2)

        # 3️⃣ 성공 응답
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "code": 200,
                "message": "3D 변환 성공",
                "data": {
                    "input_image": filename,
                    "output_mesh_url": f"/files/{os.path.basename(output_path)}",
                    "model": "tencent/Hunyuan3D-2",
                    "inference_time": elapsed
                }
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "code": 500,
                "message": f"3D 변환 중 오류 발생: {str(e)}"
            }
        )
