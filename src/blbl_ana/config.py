from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "").strip()
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "").strip()
    qwen_model: str = os.getenv("QWEN_MODEL", "qwen2.5-7b-instruct").strip()

    serpapi_api_key: str = os.getenv("SERPAPI_API_KEY", "").strip()
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "").strip()

    bili_ua: str = os.getenv(
        "BILI_UA",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    ).strip()

