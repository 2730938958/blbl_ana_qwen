from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .local_qwen import Qwen25LLM
from .schema import CommentDoc


INTENT_SET = [
    "want_buy",  # 想买/种草
    "already_bought",  # 已入手/使用体验
    "ready_to_buy",  # 准备冲/准备下单
    "wait_discount",  # 蹲降价/等活动
    "ask_link",  # 求链接/哪里买
    "compare",  # 纠结/对比/选哪个
    "question",  # 询问术语/梗/不懂
    "other",
]

PERSONA_SET = [
    "student",
    "worker",
    "gamer",
    "none",
]


_JSON_RE = re.compile(r"\{[\s\S]*\}")


def _safe_json_extract(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def classify_comment_intent(llm: Qwen25LLM, comment: str) -> Dict[str, Any]:
    """
    用 LLM 给单条评论打标：购买意图 + 人群画像（可为空）。
    返回尽量稳定的 JSON；解析失败时返回兜底结果。
    """
    prompt = f"""
你是电商/口碑分析助手。请对“单条评论”做标签分类，输出严格 JSON（不要输出多余文字）。

可选意图 intent（只能选 1 个）：
{INTENT_SET}

可选人群 persona（可以选 0-2 个；如果没有就 []）：
{PERSONA_SET}（其中 none 表示没有明确人群）

输出 JSON schema：
{{
  "intent": "<one of intent>",
  "persona": ["<persona>", "..."],
  "confidence": 0.0,
  "evidence": "<very short>"
}}

评论：{comment}
""".strip()

    raw = llm(prompt)
    obj = _safe_json_extract(raw)
    if not obj:
        return {"intent": "other", "persona": [], "confidence": 0.0, "evidence": ""}

    intent = obj.get("intent", "other")
    if intent not in INTENT_SET:
        intent = "other"

    persona = obj.get("persona", [])
    if persona is None:
        persona = []
    if isinstance(persona, str):
        persona = [persona]
    if not isinstance(persona, list):
        persona = []
    persona = [p for p in persona if p in PERSONA_SET and p != "none"]

    conf = obj.get("confidence", 0.0)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    evidence = obj.get("evidence", "")
    if not isinstance(evidence, str):
        evidence = ""

    return {"intent": intent, "persona": persona, "confidence": conf, "evidence": evidence}


def analyze_comments_intent(
    llm: Qwen25LLM,
    docs: List[CommentDoc],
    *,
    limit: Optional[int] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    对评论列表逐条打标，返回：
    - stats: intent/persona 分布
    - rows: 每条评论的打标结果（用于 UI 展示）
    """
    use_docs = docs[:limit] if (limit and limit > 0) else docs

    intent_counter: Counter[str] = Counter()
    persona_counter: Counter[str] = Counter()
    rows: List[Dict[str, Any]] = []

    for d in use_docs:
        r = classify_comment_intent(llm, d.clean_text or d.comment_text)
        intent_counter[r["intent"]] += 1
        for p in r["persona"]:
            persona_counter[p] += 1
        rows.append(
            {
                "楼层ID": d.rpid,
                "点赞数": d.like_count,
                "意图(LLM)": r["intent"],
                "人群(LLM)": ",".join(r["persona"]) if r["persona"] else "",
                "置信度": r["confidence"],
                "证据": r["evidence"],
                "评论内容": d.comment_text,
            }
        )

    stats = {
        "n": len(use_docs),
        "intent": dict(intent_counter),
        "persona": dict(persona_counter),
    }
    return stats, rows

