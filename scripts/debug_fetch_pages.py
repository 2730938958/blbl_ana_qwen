from __future__ import annotations

from blbl_ana.bilibili import BiliClient


def main() -> None:
    bvid = "BV1Y4e6ejEGk"
    cli = BiliClient()
    aid = cli.bvid_to_aid(bvid)
    for sort in (2, 0):
        print("=== sort", sort, "===")
        for pn in (1, 2, 3, 4, 5):
            data = cli.fetch_comments(aid, page=pn, page_size=20, sort=sort)
            page = data.get("page") or {}
            rs = data.get("replies") or []
            rpids = [r.get("rpid") for r in rs]
            print(
                "pn",
                pn,
                "n",
                len(rs),
                "count",
                page.get("count"),
                "num",
                page.get("num"),
                "size",
                page.get("size"),
                "rpids_first3",
                rpids[:3],
            )


if __name__ == "__main__":
    main()

