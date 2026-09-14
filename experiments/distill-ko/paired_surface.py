#!/usr/bin/env python3
"""두 아티팩트의 제품 표면을 **짝 부트스트랩**으로 비교한다.

# 왜 필요한가 (2026-09-06)

재증류가 미견 표면을 .7802 → .7987 로 올렸는데, `product_surface.py` 가 내는 것은
**팔마다 독립인 CI** 라 [.7685,.7918] 과 [.7870,.8099] 처럼 겹친다. 독립 CI 가 겹치는
것은 "차이가 없다"가 아니다 — 두 팔이 **같은 4,639 질의**를 봤으므로 질의 난이도의
분산이 양쪽에 공통으로 들어 있고, 그걸 빼지 않으면 검정력이 통째로 낭비된다.

⛔ 그래서 판정은 이 스크립트가 한다. 같은 질의 인덱스를 함께 재표집해 **차이의 CI** 를
   낸다. CI 가 0 을 포함하면 유의하지 않다 — 점추정 부호로 판정하지 마라.

# 쓰는 법

    # 배포본 아티팩트를 올려둔 상태에서
    python3 experiments/distill-ko/product_surface.py --only "처음 보는" --dump A.json
    # 후보 아티팩트로 갈아끼운 뒤
    python3 experiments/distill-ko/product_surface.py --only "처음 보는" --dump B.json
    python3 experiments/distill-ko/paired_surface.py --base A.json --cand B.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

BOOT = 4000
SEED = 20260906


def main() -> int:
    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="기준선 덤프(현 배포본)")
    ap.add_argument("--cand", required=True, help="후보 덤프")
    ap.add_argument("--out", default="")
    ap.add_argument("--same-model-axis", default="",
                    help="⛔ 모델이 아니라 **다른 축**(택소노미 크기 등)을 비교할 때만 쓴다. "
                         "그 축이 무엇인지 한 줄로 적어야 하고, 그 문자열이 결과 JSON 에 남는다. "
                         "빈 값이면 같은 모델 두 덤프는 거부한다(아티팩트를 안 갈아끼우고 두 번 잰 사고 방지).")
    a = ap.parse_args()

    A = json.loads(pathlib.Path(a.base).read_text(encoding="utf-8"))
    B = json.loads(pathlib.Path(a.cand).read_text(encoding="utf-8"))
    if A["deployed_model"] == B["deployed_model"] and not a.same_model_axis:
        sys.exit("⛔ 두 덤프가 같은 모델이다 — 아티팩트를 갈아끼우지 않고 두 번 쟀다 "
                 "(모델이 아닌 축을 비교하는 것이면 --same-model-axis 로 그 축을 명시해라)")

    rng = np.random.default_rng(SEED)
    out = {"base_model": A["deployed_model"], "cand_model": B["deployed_model"],
           "boot": BOOT, "distributions": {}}
    if a.same_model_axis:
        out["same_model_axis"] = a.same_model_axis
    for dname, da in A["dists"].items():
        db = B["dists"].get(dname)
        if db is None:
            sys.exit(f"⛔ 후보 덤프에 분포가 없다: {dname}")
        # ⛔ 짝이 성립하는지 코드가 확인한다. 질의 집합이 다르면 이 비교는 무의미하다.
        #    택소노미 크기 축에서는 리프 인덱스가 밀려 idx 기반 sha 가 정당하게 달라지므로,
        #    그 경우에만 **이름 기반** sha 로 확인한다(재색인에 불변, 정답 변경은 여전히 잡는다).
        key = "queries_key_sha" if a.same_model_axis else "queries_sha"
        if key not in da or key not in db:
            sys.exit(f"⛔ 덤프에 {key} 가 없다({dname}) — product_surface.py 로 덤프를 다시 떠라")
        if da[key] != db[key]:
            sys.exit(f"⛔ 질의 집합이 다르다({dname}): {da[key]} vs {db[key]}")
        n = da["n"]
        ix = rng.integers(0, n, (BOOT, n))
        rows = {}
        for k in da:
            if k in ("n", "queries_sha", "queries_key_sha"):
                continue
            va = np.asarray(da[k], dtype=bool)
            vb = np.asarray(db[k], dtype=bool)
            d = vb.astype(np.int8) - va.astype(np.int8)
            boot = d[ix].mean(1)
            lo, hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
            rows[k] = {
                "base": round(float(va.mean()), 4),
                "cand": round(float(vb.mean()), 4),
                "delta_pp": round(float(d.mean()) * 100, 2),
                "ci_pp": [round(float(lo) * 100, 2), round(float(hi) * 100, 2)],
                "significant": bool(lo > 0 or hi < 0),
                "flip_win": int((vb & ~va).sum()), "flip_lose": int((va & ~vb).sum()),
            }
        out["distributions"][dname] = {"n": n, **rows}
        print(f"\n── {dname} (n={n:,}) · 짝 부트스트랩 {BOOT:,}")
        for k in ("대분류_top1", "중분류_top1", "중분류_top5", "소분류_top1", "소분류_top5"):
            r = rows[k]
            mark = " ★제품 표면" if k == "중분류_top5" else ""
            sig = "유의" if r["significant"] else "무의미"
            print(f"   {k:<14} {r['base']:.4f} → {r['cand']:.4f}  "
                  f"Δ{r['delta_pp']:+.2f}pp CI[{r['ci_pp'][0]:+.2f},{r['ci_pp'][1]:+.2f}] "
                  f"{sig} (+{r['flip_win']}/-{r['flip_lose']}){mark}")

    dest = pathlib.Path(a.out) if a.out else pathlib.Path(__file__).with_name("paired_surface.json")
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n▶ {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
