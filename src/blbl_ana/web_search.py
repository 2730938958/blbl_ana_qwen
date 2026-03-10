from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

from .config import AppConfig


def serpapi_search(query: str, *, cfg: Optional[AppConfig] = None, k: int = 5) -> List[Dict[str, Any]]:
    cfg = cfg or AppConfig()
    if not cfg.serpapi_api_key:
        return [
            {
                "title": "SERPAPI_API_KEY missing",
                "snippet": "未配置 SERPAPI_API_KEY，无法进行全网搜索。你可以在 .env 中配置它。",
                "link": "",
            }
        ]
    url = "https://serpapi.com/search.json"
    resp = requests.get(
        url,
        params={
            "engine": "google",
            "q": query,
            "api_key": cfg.serpapi_api_key,
            "num": k,
        },
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    results = []
    for r in (data.get("organic_results") or [])[:k]:
        results.append(
            {
                "title": r.get("title", ""),
                "snippet": r.get("snippet", ""),
                "link": r.get("link", ""),
            }
        )
    return results or [{"title": "No results", "snippet": "没有搜索到结果。", "link": ""}]

