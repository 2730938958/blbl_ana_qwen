from __future__ import annotations

import re
from typing import Tuple

from .schema import IntentLabel, SentimentLabel


_WS_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+")
_TAG_RE = re.compile(r"#\S+")
_EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)


# 情感词表（Demo 规则版）：尽量覆盖常见中文口语/互联网表达
# 注：这是弱监督启发式规则，不保证领域泛化；后续可替换为分类模型/LLM。
POS_WORDS = {
    # 强正向/赞美
    "真香",
    "香",
    "神",
    "封神",
    "逆天",
    "无敌",
    "爆杀",
    "天花板",
    "王炸",
    "稳",
    "稳定",
    "强",
    "很强",
    "巨强",
    "顶",
    "顶中顶",
    "牛",
    "牛逼",
    "nb",
    "yyds",
    "绝了",
    "太强了",
    "优秀",
    "出色",
    "惊艳",
    "满意",
    "很满意",
    "超预期",
    "舒服",
    "很舒服",
    "顺手",
    "顺滑",
    "丝滑",
    "流畅",
    "不卡",
    "给力",
    "靠谱",
    "可靠",
    "放心",
    # 喜好/推荐/购买动机
    "喜欢",
    "很喜欢",
    "爱了",
    "狠狠爱了",
    "入了",
    "入手",
    "下单",
    "冲了",
    "准备冲",
    "想买",
    "想入",
    "推荐",
    "强烈推荐",
    "建议买",
    "值得",
    "很值",
    "值这个价",
    "不亏",
    "性价比",
    "性价比高",
    "真划算",
    "划算",
    "便宜",
    "实惠",
    "良心",
    "真良心",
    # 产品体验（数码常见）
    "续航强",
    "续航不错",
    "电池给力",
    "不发热",
    "不烫",
    "散热好",
    "温度低",
    "信号好",
    "拍照好",
    "成像好",
    "画质好",
    "屏幕好",
    "屏幕舒服",
    "亮度高",
    "做工好",
    "质感好",
    "手感好",
    "音质好",
    "系统好用",
    "体验好",
    # 中轻度正向/认可
    "还行",
    "不错",
    "挺好",
    "可以",
    "能用",
    "够用",
    "OK",
    "ok",
    "有点东西",
    "真不错",
    "确实可以",
    "确实不错",
}

NEG_WORDS = {
    # 强负向/辱骂
    "垃圾",
    "辣鸡",
    "废物",
    "一坨",
    "离谱",
    "逆天",
    "抽象",
    "糟糕",
    "烂",
    "很烂",
    "太烂了",
    "不行",
    "不太行",
    "不咋地",
    "拉",
    "拉胯",
    "垮",
    "差",
    "很差",
    "太差了",
    "失望",
    "大失望",
    "翻车",
    "又翻车",
    "踩雷",
    "雷",
    "坑",
    "大坑",
    "智商税",
    "割韭菜",
    "恰烂钱",
    "骗",
    "骗钱",
    # 价格/价值感
    "贵",
    "太贵",
    "溢价",
    "不值",
    "不值这个价",
    "不划算",
    "不香",
    "性价比低",
    # 体验问题（数码常见）
    "发热",
    "热",
    "烫",
    "巨烫",
    "烫手",
    "散热差",
    "温度高",
    "卡",
    "卡顿",
    "很卡",
    "掉帧",
    "帧率低",
    "掉电",
    "耗电",
    "续航差",
    "电池拉胯",
    "信号差",
    "断流",
    "网差",
    "拍照差",
    "糊",
    "过曝",
    "偏色",
    "屏幕差",
    "刺眼",
    "闪屏",
    "漏光",
    "绿边",
    "做工差",
    "品控",
    "品控差",
    "瑕疵",
    "系统烂",
    "bug",
    "BUG",
    "崩溃",
    "闪退",
    "死机",
    # 服务/售后/体验
    "售后差",
    "服务差",
    "退货",
    "维权",
    "不推荐",
    "劝退",
    "别买",
}


def clean_comment_text(text: str) -> str:
    t = text or ""
    t = _URL_RE.sub(" ", t)
    t = _TAG_RE.sub(" ", t)
    t = _EMOJI_RE.sub(" ", t)
    t = t.replace("\u200b", " ")
    t = _WS_RE.sub(" ", t).strip()
    return t


def sentiment_simple(text: str) -> Tuple[SentimentLabel, float]:
    """
    轻量情感（Demo 级）：词表打分，输出 label + score[-1,1]。
    """
    t = text
    pos = sum(1 for w in POS_WORDS if w in t)
    neg = sum(1 for w in NEG_WORDS if w in t)
    score = 0.0
    if pos or neg:
        score = (pos - neg) / float(pos + neg)
    if score > 0.2:
        return "positive", float(score)
    if score < -0.2:
        return "negative", float(score)
    return "neutral", float(score)


def intent_simple(text: str) -> IntentLabel:
    t = text

    # 售后/服务
    if any(k in t for k in ["售后", "保修", "退货", "换货", "维修", "维权", "客服"]):
        return "after_sales"

    # 价格/发售/参数/推荐
    if any(k in t for k in ["多少钱", "价格", "降价", "涨价", "预算", "值不值", "溢价"]):
        return "ask_price"
    if any(k in t for k in ["什么时候发售", "开售", "发售", "发布", "上市", "什么时候出"]):
        return "ask_release"
    if any(k in t for k in ["参数", "配置", "跑分", "规格", "多少瓦", "多少赫兹", "电池", "摄像头"]):
        return "ask_param"
    if any(k in t for k in ["推荐", "建议买", "怎么选", "值不值得买", "买哪个", "求推荐"]):
        return "ask_recommendation"

    # 内容相关请求
    if any(k in t for k in ["测评", "评测", "对比评测", "横评", "深度", "拆机", "续集", "下一期", "出个"]):
        return "request_review"
    if any(k in t for k in ["科普", "解释一下", "讲讲", "啥意思", "什么意思", "这是什么", "看不懂"]):
        return "request_explain"

    # 互动（感谢/回答）
    if any(k in t for k in ["谢谢", "感谢", "多谢", "谢了", "辛苦了", "respect", "瑞思拜"]):
        return "thanks"
    if any(k in t for k in ["我来回答", "答案是", "解释下", "简单说", "不是这样的"]):
        return "answer"

    # 玩梗/搞笑/表情党（粗略）
    if any(k in t for k in ["doge", "哈哈", "笑死", "绷不住", "草", "梗", "xswl", "233", "乐"]):
        return "joke_meme"

    # 引战/对线/攻击（粗略）
    if any(k in t for k in ["孝子", "粉黑大战", "对线", "急了", "破防", "带节奏", "喷"]):
        return "argument"

    if any(k in t for k in ["求链接", "链接", "哪里买", "怎么买", "求地址"]):
        return "ask_link"
    if any(k in t for k in ["想买", "想入", "种草了", "准备买", "准备冲", "冲了"]):
        return "want_buy"
    if any(k in t for k in ["已入手", "已买", "到手了", "买了", "用了一周", "用了几天"]):
        return "already_bought"
    if any(k in t for k in ["蹲降价", "等降价", "等活动", "等618", "等双11"]):
        return "wait_discount"
    if any(k in t for k in ["纠结", "对比", "还是", "选哪", "哪个好", "X和Y"]):
        return "compare"

    # 体验反馈（正/负向）
    if any(k in t for k in ["喜欢", "爱了", "真香", "满意", "推荐", "不错", "挺好", "优秀", "惊艳"]):
        return "praise"
    if any(k in t for k in ["垃圾", "拉胯", "翻车", "失望", "踩雷", "坑", "别买", "劝退", "不推荐"]):
        return "complaint"

    # BUG/问题反馈 & 功能诉求
    if any(k in t for k in ["bug", "BUG", "闪退", "崩溃", "死机", "卡顿", "断流", "掉帧"]):
        return "bug_report"
    if any(k in t for k in ["希望", "建议增加", "能不能", "能否", "加个", "改进", "优化一下"]):
        return "feature_request"

    if any(k in t for k in ["学生党", "学生", "宿舍"]):
        return "persona_student"
    if any(k in t for k in ["上班", "打工人", "通勤"]):
        return "persona_worker"
    if any(k in t for k in ["游戏", "帧率", "发热", "散热", "王者", "原神"]):
        return "persona_gamer"
    if "?" in t or "？" in t or any(k in t for k in ["什么意思", "谁是", "怎么回事", "为啥", "为什么"]):
        return "question"
    return "other"

