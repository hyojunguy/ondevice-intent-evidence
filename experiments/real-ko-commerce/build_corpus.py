#!/usr/bin/env python3
"""AI-Hub 71603 Validation 라벨 JSON 229개 → 하네스가 읽는 단일 코퍼스.

⛔ 산출물 `aihub_corpus.json` 은 **커밋되지 않는다**(`.gitignore` 의 `*corpus*.json`).
   AI-Hub 원본의 단순 재가공물이라 공개·재배포 대상이 아니다. 커밋되는 것은 이 코드뿐이다.
"""
from __future__ import annotations
import collections, json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent


def main() -> int:
    keys = set(json.loads((HERE / "aihub_map.json").read_text(encoding="utf-8"))["map"])
    rows, seen, skipped = [], set(), collections.Counter()
    for f in sorted((HERE / "aihub_raw").glob("*.json")):
        for r in json.loads(f.read_text(encoding="utf-8")):
            if r.get("Source") != "쇼핑몰":
                skipped["비쇼핑몰"] += 1; continue
            k = f'{r["Domain"]}/{r["MainCategory"]}'
            if k not in keys:
                skipped[f"미대응:{k}"] += 1; continue
            # ⛔ ProductName 을 strip 하지 마라. 뒤 공백만 다른 제품명이 75종 있고,
            #    strip 하면 평가쌍이 1,174 → 1,170 으로 줄어 앞선 모든 수치와 비교가 깨진다.
            #    공백 중복은 제거가 아니라 **보고**한다(§평가 단위).
            rows.append({"product": r.get("ProductName") or "",
                         "domain": r["Domain"], "main": r["MainCategory"],
                         "text": (r.get("RawText") or "").strip()})
    out = HERE / "aihub_corpus.json"
    out.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    prod = {(r["product"], f'{r["domain"]}/{r["main"]}') for r in rows}
    strip_pairs = {(p.strip(), k) for p, k in prod}
    print(f"행 {len(rows):,} · 평가쌍 {len(prod):,} · 고유 제품 문자열 {len({p for p,_ in prod}):,} "
          f"· 공백 정규화 시 쌍 {len(strip_pairs):,} · 라벨 {len({k for _,k in prod})}")
    for k, v in skipped.most_common(6):
        print(f"  건너뜀 {k}: {v:,}")
    print(f"▶ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
