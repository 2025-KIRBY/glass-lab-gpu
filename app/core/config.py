# 환경변수 & 경로 보장
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    # 인증키: None이면 인증 비활성화(개발 모드)
    API_KEY: str | None = None

    # 백엔드 백업 저장 루트 (jobs/<job_id>/ 구조)
    STORAGE_ROOT: str = "./storage"

    # 기본 생성 개수
    STAGE1_NUM_OUTPUTS: int = 6
    STAGE2_NUM_OUTPUTS: int = 6

    # CORS 허용 도메인 (콤마 구분). 빈 문자열이면 미사용.
    CORS_ORIGINS: str = ""

settings = Settings()

# 필요한 디렉토리 보장
Path(settings.STORAGE_ROOT).mkdir(parents=True, exist_ok=True)
