#!/usr/bin/env python3
"""결론이 **대응표의 관대함**에 의존하는가 — 평가 정의 누수에 대한 방어.

⛔ 문제: 1판 대응표는 평가에 쓸 Validation 제품명을 읽고 사람이 썼다. 그건 평가 정의
   누수이고 순열 기준선으로는 안 잡힌다. 그리고 대응표는 소급해서 '안 본 것'으로 만들 수 없다.
⇒ 그래서 여기서는 **누수를 없앴다고 주장하지 않고 상한을 잰다**: 대응표를 좁힐수록
   결론이 무너지는지 본다. 좁은 대응표에서도 같은 결론이면, 관대함이 결론을 만든 것이 아니다.

  arm A  hand        1판 수작업 대응표 (Validation 을 보고 썼다 — 가장 관대)
  arm B  hand∩train  Training 제품명이 문자열로 지지하는 것만 남긴다
  arm C  minimal     카테고리당 Training 지지도 1위 중분류 **하나만**
  arm D  unseen      arm B 의 대응표로, Training 에 없는 Validation 제품만 평가

⛔ 자동 도출 대응표(aihub_map_train.json)를 정답으로 쓰지 않는다 — 짧은 리프 이름의
   우발 일치로 **더 넓어져서** 정확도를 부풀린다(자동차기기 → 뷰티>스킨케어).
"""
from __future__ import annotations
import argparse, json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from aihub_surface import run_deploy, ROOT, perm_baseline


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--train", required=True)
    a = ap.parse_args()
    import numpy as np
    sys.path.insert(0, str(ROOT / "data")); sys.path.insert(0, str(ROOT / "experiments/distill-ko"))
    from taxonomy_ko import TAXONOMY
    import leaf_repr
    names, _a, _b, _c, l2_of = leaf_repr.flatten(TAXONOMY)
    l2_of = np.asarray(l2_of)
    pair = {}
    for li, (x, y, _z) in enumerate(names):
        pair.setdefault(int(l2_of[li]), f"{x}>{y}")
    idp = {v: k for k, v in pair.items()}

    hand = json.loads((HERE / "aihub_map.json").read_text(encoding="utf-8"))["map"]
    auto = json.loads((HERE / "aihub_map_train.json").read_text(encoding="utf-8"))["map"]
    rows = json.loads(pathlib.Path(a.corpus).read_text(encoding="utf-8"))
    train = json.loads(pathlib.Path(a.train).read_text(encoding="utf-8"))
    tprod = {(r["product"] or "").strip() for r in train}

    seen = {}
    for r in rows:
        seen.setdefault((r["product"], f"{r['domain']}/{r['main']}"), None)
    qs = [(p, k) for (p, k) in seen]
    labels = [k for _, k in qs]
    tops = run_deploy([q for q, _ in qs])

    def l2r(t):
        s, o = [], {}
        for li in t:
            v = int(l2_of[li])
            if v not in s:
                s.append(v); o.setdefault(v, len(s) - 1)
        return o
    rl = [l2r(t) for t in tops]

    def build(kind):
        m = {}
        for k, v in hand.items():
            sup = auto.get(k, {}).get("support", {})
            if kind == "hand":
                keep = v["l2"]
            elif kind == "hand_train":
                keep = [x for x in v["l2"] if x in sup] or [max(v["l2"], key=lambda x: sup.get(x, 0))]
            else:  # minimal
                cand = [x for x in v["l2"] if x in sup] or v["l2"]
                keep = [max(cand, key=lambda x: sup.get(x, 0))]
            m[k] = {idp[x] for x in keep}
        return m

    out = {"_what": "대응표를 좁혀가며 같은 결론이 서는지 본다",
           "_train_overlap": {"validation_고유제품": len({p for p, _ in qs}),
                              "training에도_있음": len({p for p, _ in qs} & tprod)},
           "arms": {}}
    for tag, kind in (("A_hand", "hand"), ("B_hand∩train", "hand_train"), ("C_minimal", "minimal")):
        g = build(kind)
        def hit(labs, gm=g):
            return np.array([any(r.get(x, 99) < 5 for x in gm[lb]) for r, lb in zip(rl, labs)])
        h = hit(labels)
        out["arms"][tag] = {
            "평균_대응L2수": round(sum(len(v) for v in g.values()) / len(g), 2),
            "n": len(qs), "중분류_top5": round(float(h.mean()), 4),
            "순열기준선": perm_baseline(hit, labels),
            "리프트_pp": round(float((h.mean() - perm_baseline(hit, labels)["평균"]) * 100), 2)}
    # arm D — Training 에 없는 제품만 (B 대응표)
    g = build("hand_train")
    unseen = np.array([p not in tprod for p, _ in qs])
    def hitD(labs, gm=g):
        return np.array([any(r.get(x, 99) < 5 for x in gm[lb]) for r, lb in zip(rl, labs)])
    hD = hitD(labels)
    lab_u = [l for l, u in zip(labels, unseen) if u]
    rl_u = [r for r, u in zip(rl, unseen) if u]
    def hitU(labs, gm=g):
        return np.array([any(r.get(x, 99) < 5 for x in gm[lb]) for r, lb in zip(rl_u, labs)])
    out["arms"]["D_unseen제품_hand∩train"] = {
        "n": int(unseen.sum()), "중분류_top5": round(float(hD[unseen].mean()), 4),
        "순열기준선": perm_baseline(hitU, lab_u)}
    # macro (카테고리 평균) — micro 가 큰 카테고리에 끌려가는지
    ga = build("hand")
    def hitA(labs, gm=ga):
        return np.array([any(r.get(x, 99) < 5 for x in gm[lb]) for r, lb in zip(rl, labs)])
    hA = hitA(labels); lab = np.array(labels)
    per = {l: float(hA[lab == l].mean()) for l in sorted(set(labels))}
    out["micro_vs_macro"] = {"micro": round(float(hA.mean()), 4),
                             "macro_20카테고리": round(float(np.mean(list(per.values()))), 4),
                             "읽기": "micro 가 높으면 큰 카테고리가 끌어올린 것이다"}
    (HERE / "map_sensitivity.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
