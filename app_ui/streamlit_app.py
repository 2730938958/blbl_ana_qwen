from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import streamlit as st

from blbl_ana.agent import build_llm, make_agent_executor
from blbl_ana.bilibili import BiliClient, pull_comments_by_bvid
from blbl_ana.corpus_summary_llm import summarize_corpus_map_reduce
from blbl_ana.pipeline import basic_stats, build_comment_docs
from blbl_ana.vector_store import build_faiss, load_faiss, read_docs_jsonl, save_faiss, write_docs_jsonl


st.set_page_config(page_title="B站评论分析 + Agent", layout="wide")

st.title("PLP B站评论分析（Demo）")


def _data_dir() -> Path:
    d = Path("data")
    d.mkdir(parents=True, exist_ok=True)
    return d


with st.sidebar:
    st.header("数据源")
    bvid_or_url = st.text_input("BV 号或视频链接", value="")
    brand_or_topic = st.text_input("品牌/关键词（用于分组）", value="demo_topic")
    max_pages = st.number_input("抓取次数（每次最多 20 条）", min_value=1, max_value=50, value=3, step=1)
    include_hot = st.checkbox("尝试抓取热评", value=True)
    build_btn = st.button("抓取并构建索引", type="primary")


if "index_dir" not in st.session_state:
    st.session_state.index_dir = None
if "docs_path" not in st.session_state:
    st.session_state.docs_path = None
if "stats" not in st.session_state:
    st.session_state.stats = None
if "preview_rows" not in st.session_state:
    st.session_state.preview_rows = None
if "last_bvid" not in st.session_state:
    st.session_state.last_bvid = None
if "llm_intent_stats" not in st.session_state:
    st.session_state.llm_intent_stats = None
if "llm_intent_rows" not in st.session_state:
    st.session_state.llm_intent_rows = None
if "llm_corpus_summary" not in st.session_state:
    st.session_state.llm_corpus_summary = None

store = None
docs = None
stats = st.session_state.stats
index_dir: Optional[Path] = st.session_state.index_dir
docs_path: Optional[Path] = st.session_state.docs_path


if build_btn:
    if not bvid_or_url.strip():
        st.error("请先输入 BV 号或视频链接。")
    else:
        with st.spinner("抓取评论中…"):
            bvid, aid, raw = pull_comments_by_bvid(
                bvid_or_url, max_pages=int(max_pages), include_hot=include_hot
            )
        docs = build_comment_docs(raw, video_bvid=bvid, video_aid=aid, brand_or_topic=brand_or_topic)
        stats = basic_stats(docs)

        ddir = _data_dir()
        index_dir = ddir / f"faiss_{bvid}"
        docs_path = ddir / f"comments_{bvid}.jsonl"

        with st.spinner("向量化并写入索引…（首次会下载 embedding 模型）"):
            store = build_faiss(docs)
            save_faiss(store, str(index_dir))
            write_docs_jsonl(docs, str(docs_path))

        st.success(f"完成：抓取 {len(docs)} 条评论。索引已保存到 {index_dir}。")

        # 持久化到 session_state，避免点“问一下”后丢失
        st.session_state.index_dir = index_dir
        st.session_state.docs_path = docs_path
        st.session_state.stats = stats
        st.session_state.last_bvid = bvid
        st.session_state.llm_intent_stats = None
        st.session_state.llm_intent_rows = None
        st.session_state.llm_corpus_summary = None
        st.session_state.preview_rows = [
            {
                "楼层ID": d.rpid,
                "是否热评": d.is_hot,
                "点赞数": d.like_count,
                "情感": d.sentiment,
                "意图": d.intent,
                "评论内容": d.comment_text,
            }
            for d in docs[:10]
        ]


st.divider()

tab_data, tab_analysis, tab_qa = st.tabs(["评论与视频", "统计与总体洞察", "咨询交互"])

with tab_data:
    if st.session_state.preview_rows:
        st.subheader("示例评论")
        st.caption("展示当前视频抓取到的前若干条评论，用于快速感受语料。")
        st.dataframe(st.session_state.preview_rows, use_container_width=True)

    st.markdown("---")
    st.subheader("视频检索（根据关键词找相关视频）")

    kw = st.text_input("输入关键词（例如：机型名/品牌/梗）", value="", key="video_search_kw")
    max_videos = st.number_input(
        "返回视频数量", min_value=1, max_value=50, value=10, step=1, key="video_search_n"
    )
    search_btn = st.button("搜索相关视频", key="video_search_btn")

    if search_btn and kw.strip():
        with st.spinner("正在调用 B 站搜索…"):
            cli = BiliClient()
            try:
                items = cli.search_videos(kw.strip(), page=1, page_size=int(max_videos))
            except Exception as e:
                st.error(f"搜索失败：{e}")
                items = []

        if items:
            st.markdown("**搜索结果：**")
            for it in items:
                st.markdown(f"- [{it['title']}]({it['url']})")
        else:
            st.info("没有搜到相关视频，换个关键词试试。")

with tab_analysis:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("基础统计（模块①）")
        st.caption("显示当前视频评论的数量、热评占比、情感分布等。")

        if stats:
            st.json(stats, expanded=False)

    with col2:
        st.subheader("总体购买意图 / 人群线索（模块②）")
        st.caption("把整批评论当作语料做总体洞察（map-reduce），输出 summary + seed_keywords。")

        batch_size = st.number_input(
            "每批评论数（越大越省调用、但越容易超上下文）",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            key="llm_batch_size",
        )
        max_batches = st.number_input(
            "最多批次数（用于控制总耗时）",
            min_value=1,
            max_value=30,
            value=12,
            step=1,
            key="llm_max_batches",
        )
        run_sum_btn = st.button("用 Qwen 总结整批评论（总体洞察）", type="secondary", key="llm_sum_btn")

        if run_sum_btn:
            if docs is None:
                if st.session_state.docs_path:
                    docs = read_docs_jsonl(str(st.session_state.docs_path))
                else:
                    st.error("未找到评论数据，请先抓取并构建索引。")
                    st.stop()

            llm = build_llm()
            with st.spinner("LLM 正在进行 map-reduce 总结（可能需要几分钟）…"):
                st.session_state.llm_corpus_summary = summarize_corpus_map_reduce(
                    llm, docs, batch_size=int(batch_size), max_batches=int(max_batches)
                )
            st.success("总体洞察生成完成。")

        if st.session_state.llm_corpus_summary:
            st.markdown("**总体洞察（LLM）**")
            st.json(st.session_state.llm_corpus_summary, expanded=False)
        elif stats:
            st.markdown("**参考：规则版意图分布**")
            st.json(stats.get("intent", {}), expanded=False)

with tab_qa:
    st.subheader("咨询交互（模块③）")

    q = st.text_input(
        "提问（例如：评论区的2399是什么意思 / 这手机发热吗 / 全网最低价多少）",
        value="",
        key="qa_input",
    )

    run_btn = st.button("问一下", disabled=not q.strip(), key="qa_btn")

    if run_btn:
        ddir = _data_dir()

        if docs is None or store is None:
            # try load last index by scanning data dir
            if st.session_state.index_dir and st.session_state.docs_path:
                index_dir = st.session_state.index_dir
                docs_path = st.session_state.docs_path
                store = load_faiss(str(index_dir))
                docs = read_docs_jsonl(str(docs_path))
            else:
                faiss_dirs = sorted(ddir.glob("faiss_BV*"), key=lambda p: p.stat().st_mtime, reverse=True)
                docs_files = sorted(ddir.glob("comments_BV*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
                if not faiss_dirs or not docs_files:
                    st.error("未找到已保存的索引，请先在左侧抓取并构建索引。")
                    st.stop()
                else:
                    index_dir = faiss_dirs[0]
                    docs_path = docs_files[0]
                    store = load_faiss(str(index_dir))
                    docs = read_docs_jsonl(str(docs_path))
                    st.session_state.index_dir = index_dir
                    st.session_state.docs_path = docs_path

        # 本地 Qwen 模型（首次加载会稍慢）
        llm = build_llm()
        agent = make_agent_executor(llm=llm, vector_store=store, docs_for_stats=docs)
        with st.spinner("Agent 思考中…"):
            out = agent.invoke({"input": q})
        st.markdown(out.get("output", ""))

