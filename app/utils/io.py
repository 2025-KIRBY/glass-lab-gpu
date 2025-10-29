# app/utils/io.py
import io, zipfile, re, uuid, shutil
from pathlib import Path
from fastapi import UploadFile
from PIL import Image

_SAFE = re.compile(r"[^A-Za-z0-9._-]")

def new_job_id() -> str:
    # 짧고 충돌 적은 고유 ID
    return uuid.uuid4().hex[:12]

def save_file(dst_dir: Path, f: UploadFile) -> Path:
    """업로드 파일을 안전한 이름으로 저장하고 경로를 반환"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    raw = (f.filename or "upload.bin")
    name = _SAFE.sub("_", Path(raw).name)  # 위험문자 제거
    out = dst_dir / f"{uuid.uuid4().hex[:8]}_{name}"  # 충돌 방지 prefix
    with out.open("wb") as w:
        shutil.copyfileobj(f.file, w)
    return out

def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()

def pil_list_to_zip_bytes(images: list[Image.Image], names: list[str] | None = None) -> bytes:
    """여러 장 이미지를 ZIP 바이트로 패킹 (PNG로 저장)"""
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_STORED) as z:
        for i, im in enumerate(images):
            fn = names[i] if names and i < len(names) else f"result_{i+1:03d}.png"
            z.writestr(fn, pil_to_png_bytes(im))
    bio.seek(0)
    return bio.getvalue()
