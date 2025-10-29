# app/core/security.py
"""
최소 보안/안정성 의존성:
- require_auth: API_KEY가 있으면 Bearer 토큰 검사, 없으면 통과
- concurrency_guard: 동시에 돌 수 있는 요청 수 제한 (GPU 보호)
- guard_uploads: 업로드 파일의 MIME/확장자/크기(헤더 기반) 가드
"""

import os
import asyncio
from typing import Iterable
from fastapi import Header, HTTPException, UploadFile

# ===== 1) 인증 (선택) ======================================
# 환경변수에 API_KEY가 없으면 인증을 건너뜀
_API_KEY = os.getenv("API_KEY")  # .env 없어도 환경변수로 줄 수 있음(없으면 None)

def require_auth(authorization: str | None = Header(default=None)):
    """API_KEY가 설정되어 있으면 Bearer 토큰을 검증. 없으면 통과."""
    if not _API_KEY:
        return True  # 인증 비활성화 모드
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="사용자 인증이 필요한 서비스입니다")
    token = authorization.split(" ", 1)[1]
    if token != _API_KEY:
        raise HTTPException(status_code=401, detail="사용자 인증이 필요한 서비스입니다")
    return True


# ===== 2) 동시성 제한 (필수 아님, 권장) =====================
# 한 프로세스에서 동시에 처리할 최대 요청 수 (모델 추론 보호용)
_MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "2"))
_SEMAPHORE = asyncio.Semaphore(_MAX_CONCURRENCY)

async def concurrency_guard():
    """동시에 처리할 요청 수를 제한(GPU/메모리 보호)."""
    await _SEMAPHORE.acquire()
    try:
        yield
    finally:
        _SEMAPHORE.release()


# ===== 3) 업로드 가드 (간단형) ===============================
# 허용 MIME/확장자 (필요시 추가)
_ALLOWED_MIME = {
    "image/png",
    "image/jpeg",
}
_ALLOWED_EXT = {".png", ".jpg", ".jpeg"}

# 최대 업로드 크기 (HTTP 헤더 Content-Length 기준, 바디 전체 상한)
# uvicorn 실행 시 --limit-max-request 로 전체 요청 크기 상한을 걸면 더 안전.
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "15"))  # 파일 한 장 당 권고 최대 (헤더/힌트 기반)
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024

def _has_allowed_ext(filename: str) -> bool:
    import os
    _, ext = os.path.splitext(filename or "")
    return ext.lower() in _ALLOWED_EXT

def guard_uploads(files: Iterable[UploadFile]):
    """
    아주 가벼운 업로드 검사:
    - content_type이 허용 목록인지
    - 파일명 확장자가 허용 목록인지
    - (선택) file.spool_max_size 힌트로 과도한 용량 감지 시 차단
      *정확한 크기 검증은 서버 레벨에서 --limit-max-request 권장*
    """
    for f in files:
        # MIME 체크 (브라우저/클라이언트가 보내는 값이므로 100% 신뢰는 금물)
        if f.content_type not in _ALLOWED_MIME:
            raise HTTPException(status_code=400, detail=f"허용되지 않은 파일 타입: {f.content_type}")

        # 확장자 체크
        if not _has_allowed_ext(f.filename or ""):
            raise HTTPException(status_code=400, detail=f"허용되지 않은 확장자: {f.filename}")

        # 크기 간단 가드 (정밀하지 않음! 요청 전체 상한은 uvicorn 옵션으로)
        # 업로드 스트림 특성상 정확한 파일 크기는 사전 알기 어렵다.
        # 너무 큰 가능성이 보이면 차단하는 '힌트' 정도로 사용.
        if hasattr(f.file, "max_size"):
            try:
                if f.file.max_size and f.file.max_size > MAX_FILE_BYTES:
                    raise HTTPException(status_code=413, detail=f"파일이 너무 큼(>{MAX_FILE_MB}MB): {f.filename}")
            except Exception:
                pass  # 드물게 속성이 없거나 접근 불가한 경우 스킵

    return True
