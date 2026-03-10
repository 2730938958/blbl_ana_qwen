from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .local_qwen import Qwen25LLM
from .schema import CommentDoc


_JSON_RE = re.compile(r"\{[\s\S]*\}")


def _safe_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _format_comments(comments: Sequence[CommentDoc], *, max_chars: int = 12000) -> str:
    """
    将评论压缩成一个可给 LLM 的语料片段（带少量权重信息）。
    """
    lines: List[str] = []
    used = 0
    for d in comments:
        # 简单权重：点赞数高的更重要（让模型知道）
        line = f"- (like={d.like_count}) {d.clean_text or d.comment_text}"
        if used + len(line) + 1 > max_chars:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


def summarize_batch(llm: Qwen25LLM, comments: Sequence[CommentDoc]) -> Dict[str, Any]:
    """
    对一个批次评论做“局部总结”，输出结构化 JSON，便于最终合并。
    """
    corpus = _format_comments(comments)
    prompt = f"""
你是产品口碑与购买意图分析师。下面是一批 B 站评论（同一主题/同一视频），请基于语料做总结，并只输出严格 JSON（不要输出其它文字）。

输出 JSON schema:
{{
  "summary": "...",
  "seed_keywords": ["..."]
}}

要求：
- summary：用 5-8 句总结这批评论的总体口碑与购买意图、主要关注点。
- seed_keywords：高频种草/吐槽关键词（8-20 个，中文短词）。

评论语料：
{corpus}
""".strip()

    raw = llm(prompt)
    obj = _safe_json(raw)
    if obj:
        return obj
    # 兜底：返回空结构，避免整体流程中断
    return {"summary": "", "seed_keywords": []}


def _merge_weighted_items(batches: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    counter: Counter[str] = Counter()
    evidence_map: Dict[str, List[str]] = {}
    for b in batches:
        for it in b.get(key, []) or []:
            label = it.get("label")
            if not isinstance(label, str) or not label.strip():
                continue
            w = it.get("weight", 0)
            try:
                w = int(w)
            except Exception:
                w = 0
            counter[label] += max(0, w)
            ev = it.get("evidence", []) or []
            if isinstance(ev, str):
                ev = [ev]
            if label not in evidence_map:
                evidence_map[label] = []
            for e in ev:
                if isinstance(e, str) and e.strip() and len(evidence_map[label]) < 5:
                    evidence_map[label].append(e.strip())

    out = []
    for label, w in counter.most_common(12):
        out.append({"label": label, "weight": w, "evidence": evidence_map.get(label, [])[:3]})
    return out


def summarize_corpus_map_reduce(
    llm: Qwen25LLM,
    docs: List[CommentDoc],
    *,
    batch_size: int = 50,
    max_batches: int = 12,
) -> Dict[str, Any]:
    """
    Map-Reduce：
    - map：每 batch 总结一次
    - reduce：合并权重与关键词，再让 LLM 生成最终报告（更稳更全）
    """
    # 按点赞数排序，让信息密度更高
    docs_sorted = sorted(docs, key=lambda d: d.like_count, reverse=True)
    batches: List[List[CommentDoc]] = []
    for i in range(0, min(len(docs_sorted), batch_size * max_batches), batch_size):
        batches.append(docs_sorted[i : i + batch_size])

    partials: List[Dict[str, Any]] = []
    for b in batches:
        partials.append(summarize_batch(llm, b))

    merged_seed_keywords = sorted(
        {
            kw
            for p in partials
            for kw in (p.get("seed_keywords") or [])
            if isinstance(kw, str) and kw.strip()
        }
    )

    merged = {
        "batch_summaries": [
            p.get("summary", "")
            for p in partials
            if isinstance(p.get("summary", ""), str) and p.get("summary", "").strip()
        ],
        "seed_keywords": merged_seed_keywords[:40],
        "n_comments_used": sum(len(b) for b in batches),
        "n_total_comments": len(docs),
    }

    # 最后让 LLM 把 merged 变成最终 summary + seed_keywords
    prompt = f"""
你是产品口碑与购买意图分析师。下面是对评论语料的“分批总结结果”（summary + keywords）。
请把它合并成最终结果，只输出严格 JSON（不要输出其它文字）。

输入要点（JSON）：
{json.dumps(merged, ensure_ascii=False)}

输出 JSON schema：
{{
  "summary": "...",
  "seed_keywords": ["..."]
}}
""".strip()

    raw = llm(prompt)
    obj = _safe_json(raw)
    if obj:
        if not isinstance(obj.get("summary", ""), str):
            obj["summary"] = ""
        kws = obj.get("seed_keywords", [])
        if kws is None:
            kws = []
        if isinstance(kws, str):
            kws = [kws]
        if not isinstance(kws, list):
            kws = []
        obj["seed_keywords"] = [k for k in kws if isinstance(k, str) and k.strip()][:30]
        return {"summary": obj["summary"].strip(), "seed_keywords": obj["seed_keywords"]}

    # fallback：直接返回 merged
    return {
        "summary": "\n".join(merged["batch_summaries"][:6]).strip(),
        "seed_keywords": merged["seed_keywords"][:30],
    }

