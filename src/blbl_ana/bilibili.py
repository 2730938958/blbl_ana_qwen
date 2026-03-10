from __future__ import annotations

import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

from .config import AppConfig


BV_RE = re.compile(r"(BV[a-zA-Z0-9]{10})")


class BiliClient:
    """
    Minimal B站评论抓取客户端（公开接口，适合作为 Demo）。

    说明：
    - 真实环境可能遇到风控/限流/接口变动，需要更完善的重试与鉴权。
    """

    def __init__(self, cfg: Optional[AppConfig] = None, timeout_s: int = 15):
        self.cfg = cfg or AppConfig()
        self.sess = requests.Session()
        self.sess.headers.update(
            {
                "User-Agent": self.cfg.bili_ua,
                "Referer": "https://www.bilibili.com",
                "Origin": "https://www.bilibili.com",
            }
        )
        self.timeout_s = timeout_s

    def resolve_bvid(self, text_or_bvid: str) -> str:
        m = BV_RE.search(text_or_bvid)
        if not m:
            raise ValueError("未识别到 BV 号，请输入 BVxxxxxxxxxx 或包含 BV 的链接。")
        return m.group(1)

    def bvid_to_aid(self, bvid: str) -> int:
        url = "https://api.bilibili.com/x/web-interface/view"
        resp = self.sess.get(url, params={"bvid": bvid}, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"获取视频信息失败: {data}")
        return int(data["data"]["aid"])

    def fetch_comments(
        self,
        aid: int,
        *,
        page: int = 1,
        page_size: int = 20,
        sort: int = 2,
    ) -> Dict[str, Any]:
        """
        拉取一级评论（含分页）。

        sort:
        - 0: 按时间
        - 2: 按热度（常用）
        """
        url = "https://api.bilibili.com/x/v2/reply"
        resp = self.sess.get(
            url,
            params={"type": 1, "oid": aid, "pn": page, "ps": page_size, "sort": sort},
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"获取评论失败: {data}")
        return data["data"]

    def fetch_hot_comments(self, aid: int, *, pn: int = 1, ps: int = 20) -> Dict[str, Any]:
        """
        拉取热评（接口可能随时间变化；失败时可退化为 fetch_comments(sort=2)）。
        """
        url = "https://api.bilibili.com/x/v2/reply/hot"
        resp = self.sess.get(
            url, params={"type": 1, "oid": aid, "pn": pn, "ps": ps}, timeout=self.timeout_s
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"获取热评失败: {data}")
        return data["data"]

    def fetch_comments_main(
        self,
        aid: int,
        *,
        next_cursor: int = 0,
        page_size: int = 20,
        mode: int = 3,
    ) -> Dict[str, Any]:
        """
        拉取一级评论（新版 main 接口，支持 cursor 翻页）。

        mode:
        - 3: 常用展示（通常能拿到“推荐/热度相关”的列表）
        """
        url = "https://api.bilibili.com/x/v2/reply/main"
        resp = self.sess.get(
            url,
            params={"type": 1, "oid": aid, "mode": mode, "next": next_cursor, "ps": page_size},
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"获取评论失败(main): {data}")
        return data["data"]

    def search_videos(
        self,
        keyword: str,
        *,
        page: int = 1,
        page_size: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        通过关键词搜索视频，返回简单结构列表：
        [{"bvid": "...", "title": "...", "url": "..."}]
        """
        url = "https://api.bilibili.com/x/web-interface/search/type"
        resp = self.sess.get(
            url,
            params={
                "search_type": "video",
                "keyword": keyword,
                "page": page,
                "page_size": page_size,
            },
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"搜索视频失败: {data}")
        result = data.get("data") or {}
        items = result.get("result") or []
        out: List[Dict[str, Any]] = []
        for it in items:
            bvid = it.get("bvid")
            title = it.get("title") or it.get("name") or ""
            if not bvid:
                continue
            out.append(
                {
                    "bvid": bvid,
                    "title": title,
                    "url": f"https://www.bilibili.com/video/{bvid}",
                }
            )
        return out


def iter_comment_replies(data: Dict[str, Any], *, is_hot: bool) -> Iterable[Dict[str, Any]]:
    replies = data.get("replies") or []
    for r in replies:
        r["_is_hot"] = is_hot
        yield r


def pull_comments_by_bvid(
    bvid_or_url: str,
    *,
    max_pages: int = 5,
    page_size: int = 20,
    sleep_s: float = 0.3,
    include_hot: bool = True,
    cfg: Optional[AppConfig] = None,
) -> Tuple[str, int, List[Dict[str, Any]]]:
    """
    返回 (bvid, aid, raw_reply_objects)。
    """
    client = BiliClient(cfg=cfg)
    bvid = client.resolve_bvid(bvid_or_url)
    aid = client.bvid_to_aid(bvid)

    raw: List[Dict[str, Any]] = []

    if include_hot:
        try:
            hot = client.fetch_hot_comments(aid, pn=1, ps=20)
            raw.extend(list(iter_comment_replies(hot, is_hot=True)))
            time.sleep(sleep_s)
        except Exception:
            # hot 接口可能不稳定，允许退化
            pass

    # 优先使用 main(cursor) 接口（旧 pn/ps 接口在部分视频上 pn>1 会返回空）
    try:
        next_cursor = 0
        for _ in range(max_pages):
            data = client.fetch_comments_main(aid, next_cursor=next_cursor, page_size=page_size, mode=3)
            rs = list(iter_comment_replies(data, is_hot=False))
            if rs:
                raw.extend(rs)

            cursor = data.get("cursor") or {}
            is_end = bool(cursor.get("is_end"))
            next_cursor = int(cursor.get("next") or 0)

            if is_end or not rs:
                break
            time.sleep(sleep_s)
    except Exception:
        # fallback：旧接口
        for pn in range(1, max_pages + 1):
            data = client.fetch_comments(aid, page=pn, page_size=page_size, sort=2)
            rs = list(iter_comment_replies(data, is_hot=False))
            if not rs:
                break
            raw.extend(rs)
            time.sleep(sleep_s)

    return bvid, aid, raw

