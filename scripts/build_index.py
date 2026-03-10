from __future__ import annotations

import argparse
from pathlib import Path

from blbl_ana.bilibili import pull_comments_by_bvid
from blbl_ana.pipeline import basic_stats, build_comment_docs
from blbl_ana.vector_store import build_faiss, save_faiss, write_docs_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bvid", required=True, help="BVxxxxxxxxxx or url containing it")
    ap.add_argument("--topic", default="demo_topic", help="brand/topic tag")
    ap.add_argument("--max-pages", type=int, default=3)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bvid, aid, raw = pull_comments_by_bvid(args.bvid, max_pages=args.max_pages, include_hot=True)
    docs = build_comment_docs(raw, video_bvid=bvid, video_aid=aid, brand_or_topic=args.topic)
    print("stats:", basic_stats(docs))

    store = build_faiss(docs)
    save_faiss(store, str(out_dir / f"faiss_{bvid}"))
    write_docs_jsonl(docs, str(out_dir / f"comments_{bvid}.jsonl"))
    print("saved:", out_dir / f"faiss_{bvid}", out_dir / f"comments_{bvid}.jsonl")


if __name__ == "__main__":
    main()

