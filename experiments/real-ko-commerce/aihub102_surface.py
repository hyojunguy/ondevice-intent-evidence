#!/usr/bin/env python3
"""실제 고객 주문 발화로 배포 경로를 잰다 (AI-Hub 102 Validation · 고객 발화 307,551건).

71603 이 **리뷰**였다면 이 데이터는 **주문·문의 발화**다 — 우리 제품이 실제로 받는 입력에
가장 가깝다. 다만 라벨 14종이 업태와 상품군을 섞어 놓아 **대분류(L1)까지만** 판정한다
(`aihub102_map.json` 의 _granularity).

⛔ 대응 폭이 넓으면 적중이 쉬워진다 — 같은 대응으로 라벨만 섞은 순열 기준선을 함께 낸다.
⛔ 데이터는 재배포하지 않는다. 코퍼스 경로는 인자로 받는다.
"""
from __future__ import annotations
import argparse, json, pathlib, random, sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from aihub_surface import run_deploy, ROOT, BOOT, perm_baseline, lexical_hook_flags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--sample", type=int, default=8000)
    a = ap.parse_args()

    import numpy as np
    sys.path.insert(0, str(ROOT / "data")); sys.path.insert(0, str(ROOT / "experiments/distill-ko"))
    from taxonomy_ko import TAXONOMY
    import leaf_repr

    names, l1s, _l2, l1_of, l2_of = leaf_repr.flatten(TAXONOMY)
    l1_of = np.asarray(l1_of)
    id_of_l1 = {}
    for li, (a_, _b, _c) in enumerate(names):
        id_of_l1.setdefault(a_, int(l1_of[li]))

    cfg = json.loads((HERE / "aihub102_map.json").read_text(encoding="utf-8"))["map"]
    gold_l1 = {k: {id_of_l1[x] for x in v["l1"]} for k, v in cfg.items()}

    rows = [r for r in json.loads(pathlib.Path(a.corpus).read_text(encoding="utf-8"))
            if r["cat"] in gold_l1]
    rnd = random.Random(20260914)

    sets = {}
    utt = [(r["text"], r["cat"]) for r in rows if r["text"] and len(r["text"]) >= 5]
    sets["실제 고객 발화"] = rnd.sample(utt, min(a.sample, len(utt)))
    ment = sorted({(p.strip(), r["cat"]) for r in rows for p in (r["product"] or "").split("|")
                   if len(p.strip()) >= 2})
    sets["상품명 엔티티"] = rnd.sample(ment, min(a.sample, len(ment)))

    rng = np.random.default_rng(0)
    out = {"source": "AI-Hub 102 Validation · 고객 발화",
           "surface": "대분류(L1) — 라벨 입도가 중분류를 지탱하지 못한다",
           "deployed_model": json.loads((ROOT / "artifacts/MANIFEST.json").read_text(
               encoding="utf-8"))["_provenance"]["source_model"],
           "n_pool": {"고객발화": len(utt), "상품명엔티티_고유": len(ment)},
           "map_avg_l1": round(sum(len(v) for v in gold_l1.values()) / len(gold_l1), 2),
           "sets": {}}

    for sname, qs in sets.items():
        tops = run_deploy([q for q, _ in qs])
        labels = [k for _, k in qs]

        def rank1(top):
            seen, o = [], {}
            for li in top:
                v = int(l1_of[li])
                if v not in seen:
                    seen.append(v); o.setdefault(v, len(seen) - 1)
            return o
        rl = [rank1(t) for t in tops]

        def hit(labs, k):
            return np.array([any(r.get(g, 99) < k for g in gold_l1[lb]) for r, lb in zip(rl, labs)])

        res = {"대분류_top1": hit(labels, 1), "대분류_top3": hit(labels, 3),
               "대분류_top5": hit(labels, 5)}
        n = len(qs); ix = rng.integers(0, n, (BOOT, n)); rec = {"n": n}
        for k2, v in res.items():
            b = v[ix].mean(1)
            rec[k2] = {"acc": round(float(v.mean()), 4),
                       "ci": [round(float(np.percentile(b, 2.5)), 4),
                              round(float(np.percentile(b, 97.5)), 4)],
                       # ⛔ 1회 추출은 잡음 — 반복해서 평균·CI 를 낸다
                       "순열기준선": perm_baseline(
                           lambda p, kk=int(k2[-1]): hit(p, kk), labels)}
        # 절단 검사 — top-32 가 대분류 5개를 못 만들면 top-5 가 아니다
        dist = np.array([len(r) for r in rl])
        rec["절단검사"] = {"대분류수_중앙값": int(np.median(dist)),
                       "5개미만_비율": round(float((dist < 5).mean()), 4)}
        # 기전 축 — 질의가 리프 이름과 형태소를 공유하는가
        hook = lexical_hook_flags([q for q, _ in qs], names)
        rec["어휘후크"] = {("있음" if hv else "없음"): {
            "n": int((hook == hv).sum()),
            **{k2: round(float(v[hook == hv].mean()), 4) for k2, v in res.items()}}
            for hv in (True, False) if (hook == hv).sum() >= 30}
        lab = np.array(labels)
        rec["라벨별_대분류_top5"] = {l: {"n": int((lab == l).sum()),
                                    "acc": round(float(res["대분류_top5"][lab == l].mean()), 4)}
                                 for l in sorted(set(labels))}
        out["sets"][sname] = rec
        print(f"\n── {sname} (n={n:,})", flush=True)
        for k2 in ("대분류_top1", "대분류_top3", "대분류_top5"):
            c = rec[k2]
            print(f"   {k2:<12} {c['acc']:.4f} CI[{c['ci'][0]:.4f},{c['ci'][1]:.4f}]"
                  f"  · 순열기준선 {c['순열기준선']['평균']:.4f}"
                  f"[{c['순열기준선']['CI'][0]:.4f},{c['순열기준선']['CI'][1]:.4f}]", flush=True)

    (HERE / "aihub102_surface.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    print(f"\n▶ {HERE / 'aihub102_surface.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
