#!/usr/bin/env python3
"""대응표를 **Training split 에서 결정론적으로** 도출한다 — 평가셋을 보고 만들지 않는다.

⛔ 왜: 1판 대응표는 평가에 쓸 Validation 제품명을 읽고 사람이 썼다. 그건 학습 누수가
   아니라 **평가 정의 누수**이고, 순열 기준선으로는 잡히지 않는다("애매한 대응을 유리한
   쪽으로 고를 수 있었나"를 묻지 않기 때문).

여기서는 사람도 모델도 개입하지 않는다:
  AI-Hub 카테고리 c 의 **Training 제품명**에 우리 리프 이름이 문자열로 나타나면 그 리프의
  중분류를 c 의 후보로 센다. 언급 비율이 MIN_SHARE 이상인 중분류만 정답 집합에 넣는다.
⛔ 모델 점수를 쓰지 않는다 — 평가할 모델로 정답을 만들면 순환이다.
⛔ Validation 은 이 파일 어디에서도 열지 않는다.
"""
from __future__ import annotations
import argparse, collections, json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--min-share", type=float, default=0.05,
                    help="그 카테고리 리프 언급의 이 비율 이상을 차지하는 중분류만 정답으로 인정")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "data"))
    from taxonomy_ko import TAXONOMY

    l2_of_leaf = {}
    for l1, mids in TAXONOMY.items():
        for l2, leaves in mids.items():
            for lf in leaves:
                l2_of_leaf.setdefault(lf.replace(" ", ""), f"{l1}>{l2}")
    leafn = sorted((x for x in l2_of_leaf if len(x) >= 2), key=len, reverse=True)

    rows = json.loads(pathlib.Path(a.train).read_text(encoding="utf-8"))
    per = collections.defaultdict(collections.Counter)
    seen_products = collections.defaultdict(set)
    for r in rows:
        key = f"{r['domain']}/{r['main']}"
        p = (r["product"] or "").replace(" ", "")
        if not p or p in seen_products[key]:
            continue                      # 제품 하나가 리뷰 수만큼 세지 않게
        seen_products[key].add(p)
        for ln in leafn:
            if ln in p:
                per[key][l2_of_leaf[ln]] += 1
                break                     # 가장 긴 일치 하나만

    out = {"_method": "Training split 제품명 ↔ 리프 이름 문자열 일치. 사람·모델 개입 없음.",
           "_min_share": a.min_share,
           "_provenance": "AI-Hub 71603 Training 라벨링데이터 · Validation 미열람",
           "map": {}}
    for key, cnt in sorted(per.items()):
        tot = sum(cnt.values())
        keep = [l2 for l2, n in cnt.most_common() if n / tot >= a.min_share]
        out["map"][key] = {"l2": keep, "support": {l2: cnt[l2] for l2 in keep},
                           "matched_products": tot}
    (HERE / "aihub_map_train.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                               encoding="utf-8")

    hand = json.loads((HERE / "aihub_map.json").read_text(encoding="utf-8"))["map"]
    print(f"{'카테고리':<24}{'수작업':>4}{'자동':>4}{'교집합':>5}  자동에만 있는 것")
    agree = 0
    for k in sorted(hand):
        h = set(hand[k]["l2"]); d = set(out["map"].get(k, {}).get("l2", []))
        agree += len(h & d)
        extra = ", ".join(sorted(d - h)[:3])
        print(f"  {k:<22}{len(h):>4}{len(d):>4}{len(h & d):>5}  {extra}")
    tot_h = sum(len(v["l2"]) for v in hand.values())
    print(f"\n수작업 대응 {tot_h}개 중 Training 자동 도출과 겹치는 것 {agree}개 ({agree/tot_h:.0%})")
    print(f"▶ {HERE / 'aihub_map_train.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
