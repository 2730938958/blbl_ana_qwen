from __future__ import annotations

import argparse
from pathlib import Path

from blbl_ana.agent import build_llm, make_agent_executor
from blbl_ana.vector_store import load_faiss, read_docs_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bvid", default="BV1Y4e6ejEGk", help="BVxxxxxxxxxx")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--q", default="这产品风评如何", help="question")
    args = ap.parse_args()

    ddir = Path(args.data_dir)
    store = load_faiss(str(ddir / f"faiss_{args.bvid}"))
    docs = read_docs_jsonl(str(ddir / f"comments_{args.bvid}.jsonl"))

    # 使用本地 Qwen LLM
    llm = build_llm()
    agent = make_agent_executor(llm=llm, vector_store=store, docs_for_stats=docs)
    out = agent.invoke({"input": args.q})
    print(out.get("output", ""))


if __name__ == "__main__":
    main()

