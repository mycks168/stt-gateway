from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv(override=False)


@dataclass(frozen=True)
class Settings:
    host: str = os.getenv("STT_HOST", "0.0.0.0")
    port: int = int(os.getenv("STT_PORT", "23000"))
    model: str = os.getenv("STT_MODEL", "large-v3-turbo")
    device: str = os.getenv("STT_DEVICE", "auto")
    compute_type: str = os.getenv("STT_COMPUTE_TYPE", "int8_float16")
    language: str = os.getenv("STT_LANGUAGE", "ja")
    api_key: str = os.getenv("STT_API_KEY", "")


settings = Settings()
