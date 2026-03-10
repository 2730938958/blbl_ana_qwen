from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .preprocess import clean_comment_text, intent_simple, sentiment_simple
from .schema import CommentDoc, IntentLabel, SentimentLabel


def _get(d: Dict[str, Any], path: Sequence[str], default: Any = None) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def raw_reply_to_doc(
    raw: Dict[str, Any],
    *,
    video_bvid: str,
    video_aid: int,
    brand_or_topic: str,
) -> Optional[CommentDoc]:
    msg = _get(raw, ["content", "message"], "")
    if not msg or not isinstance(msg, str):
        return None

    rpid = int(raw.get("rpid") or 0)
    if not rpid:
        return None

    parent = raw.get("parent")
    parent_rpid = int(parent) if isinstance(parent, int) and parent != 0 else None

    like_count = int(raw.get("like") or 0)
    created_at = int(raw.get("ctime") or 0)
    is_hot = bool(raw.get("_is_hot") or False)

    comment_text = msg
    clean_text = clean_comment_text(comment_text)
    if not clean_text:
        return None

    sentiment, sentiment_score = sentiment_simple(clean_text)
    intent = intent_simple(clean_text)

    return CommentDoc(
        video_bvid=video_bvid,
        video_aid=video_aid,
        rpid=rpid,
        parent_rpid=parent_rpid,
        is_hot=is_hot,
        like_count=like_count,
        created_at=created_at,
        brand_or_topic=brand_or_topic,
        comment_text=comment_text,
        clean_text=clean_text,
        sentiment=sentiment,
        sentiment_score=sentiment_score,
        intent=intent,
    )


def build_comment_docs(
    raw_replies: List[Dict[str, Any]],
    *,
    video_bvid: str,
    video_aid: int,
    brand_or_topic: str,
) -> List[CommentDoc]:
    out: List[CommentDoc] = []
    seen = set()
    for r in raw_replies:
        d = raw_reply_to_doc(
            r, video_bvid=video_bvid, video_aid=video_aid, brand_or_topic=brand_or_topic
        )
        if not d:
            continue
        if d.id in seen:
            continue
        seen.add(d.id)
        out.append(d)
    return out


def basic_stats(docs: Sequence[CommentDoc]) -> Dict[str, Any]:
    total = len(docs)
    hot = sum(1 for d in docs if d.is_hot)
    senti = {"positive": 0, "neutral": 0, "negative": 0}
    intents: Dict[str, int] = {}
    for d in docs:
        senti[d.sentiment] += 1
        intents[d.intent] = intents.get(d.intent, 0) + 1
    return {
        "total_comments": total,
        "hot_ratio": (hot / total) if total else 0.0,
        "sentiment": senti,
        "intent": dict(sorted(intents.items(), key=lambda kv: kv[1], reverse=True)),
    }

