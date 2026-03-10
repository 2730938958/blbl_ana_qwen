from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


SentimentLabel = Literal["positive", "neutral", "negative"]
IntentLabel = Literal[
    # 购买/交易相关
    "want_buy",
    "already_bought",
    "ready_to_buy",
    "wait_discount",
    "ask_link",
    "compare",

    # 售前/售后/价格/参数
    "ask_price",
    "ask_release",
    "ask_param",
    "ask_recommendation",
    "after_sales",

    # 体验反馈（非交易）
    "praise",
    "complaint",
    "bug_report",
    "feature_request",

    # 互动与内容相关
    "question",
    "answer",
    "request_explain",
    "request_review",
    "thanks",
    "joke_meme",
    "argument",
    "off_topic",

    # 人群线索（保留）
    "persona_student",
    "persona_worker",
    "persona_gamer",

    "other",
]


@dataclass(frozen=True)
class CommentDoc:
    """A single Bilibili comment document for analysis + retrieval."""

    video_bvid: str
    video_aid: int
    rpid: int
    parent_rpid: Optional[int]
    is_hot: bool
    like_count: int
    created_at: int  # unix seconds

    brand_or_topic: str
    comment_text: str
    clean_text: str

    sentiment: SentimentLabel
    sentiment_score: float  # [-1, 1]
    intent: IntentLabel

    @property
    def id(self) -> str:
        return f"{self.video_aid}:{self.rpid}"

