#!/usr/bin/env python3
"""선택적 위험 곡선 — **얼마를 포기하면 얼마가 정확해지나**.

4판 리뷰 지적: 실패 구간(어휘앵커 없음)을 찾았으면 다음 질문은 "막을 것이냐"가 아니라
"거절했을 때 무엇을 얻느냐"다. 임계를 정하기 전에 곡선부터 그린다.

신호 3종(전부 기기 안에서 계산 가능한 것만):
  anchor  질의에 리프 이름이 부분문자열로 들어 있나 (어휘 항이 붙을 조건)
  margin  1위 중분류 점수 − 5위 중분류 점수 (배포 랭커가 낸 점수 그대로)
  code    제조사 모델코드형 문자열인가 (⛔ 고정효과 통제 후 효과 없음 — 대조용으로만 둔다)

⛔ 이 곡선은 게이트가 아니다. 임계값을 논문에 못박지 않는다 — 운영 비용이 정해져야 임계가
   정해지고, 그 비용은 아직 없다.
"""
from __future__ import annotations
import argparse, json, pathlib, re, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TOPK = 32


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--corpus", required=True)
    a = ap.parse_args()
    import numpy as np
    sys.path.insert(0, str(ROOT / "data")); sys.path.insert(0, str(ROOT / "experiments/distill-ko"))
    from taxonomy_ko import TAXONOMY
    import leaf_repr
    names, _x, _y, _z, l2_of = leaf_repr.flatten(TAXONOMY); l2_of = np.asarray(l2_of)
    pair = {}
    for li, (p, q, _r) in enumerate(names):
        pair.setdefault(int(l2_of[li]), f"{p}>{q}")
    idp = {v: k for k, v in pair.items()}
    cfg = json.loads((HERE / "aihub_map.json").read_text(encoding="utf-8"))["map"]
    gold = {k: {idp[x] for x in v["l2"]} for k, v in cfg.items()}

    rows = json.loads(pathlib.Path(a.corpus).read_text(encoding="utf-8"))
    seen = {}
    for r in rows:
        seen.setdefault((r["product"], f'{r["domain"]}/{r["main"]}'), None)
    qs = [q for q, _ in seen]; labels = [k for _, k in seen]

    payload = "".join(f"{q.replace(chr(9),' ')}\t0\n" for q in qs)
    r = subprocess.run(["cargo", "run", "--quiet", "--release", "-p", "oicr-sdk-core",
                        "--example", "sweep_lexical", "--", "--topk", str(TOPK)],
                       cwd=ROOT, input=payload, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(r.stderr[-800:])
    recs = [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")]

    hit, margin = [], []
    for rec, lb in zip(recs, labels):
        order, best = [], {}
        for li, sc in zip(rec["top"], rec["score"]):
            v = int(l2_of[li])
            if v not in best:
                best[v] = sc; order.append(v)
        hit.append(any(best and (order.index(g) < 5) for g in gold[lb] if g in order))
        # ⛔ 5위가 없으면(중분류가 5개 미만) 마진은 정의되지 않는다 — 마지막 점수를 쓴다.
        margin.append(float(best[order[0]] - best[order[min(4, len(order) - 1)]]))
    hit = np.array(hit, dtype=float); margin = np.array(margin)
    leafn = sorted({c.replace(" ", "") for _x, _y, c in names if len(c) >= 2},
                   key=len, reverse=True)
    anchor = np.array([any(ln in q.replace(" ", "") for ln in leafn) for q in qs])
    code = np.array([bool(re.search(r"[A-Za-z]{2,}[-\s]?\d{2,}|\d{2,}[A-Za-z]{2,}", q))
                     for q in qs])

    def curve(score):
        o = np.argsort(-score); h = hit[o]; n = len(h)
        out = []
        for cov in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4):
            k = max(1, int(round(cov * n)))
            out.append({"coverage": cov, "n": k, "accuracy": round(float(h[:k].mean()), 4)})
        return out

    res = {"_what": "선택적 위험 — 신호 내림차순으로 상위 X% 만 답했을 때의 정확도",
           "n": len(qs), "전체정확도": round(float(hit.mean()), 4),
           "곡선": {
               "margin 단독": curve(margin),
               "anchor 우선, 동률은 margin": curve(anchor.astype(float) * 1e3 + margin),
               "anchor + margin − code": curve(anchor.astype(float) * 1e3 + margin
                                               - code.astype(float) * 1e2),
           },
           "_읽는법": "커버리지 100% 행이 무조건 답할 때의 값이다. 신호가 쓸모 있으면 "
                   "커버리지를 낮출수록 정확도가 오른다."}
    (HERE / "selective_risk.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    for k, v in res["곡선"].items():
        print(f"── {k}")
        print("   " + "  ".join(f"{x['coverage']:.0%}:{x['accuracy']:.4f}" for x in v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
