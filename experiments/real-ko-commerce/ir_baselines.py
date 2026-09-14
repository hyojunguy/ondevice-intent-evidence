#!/usr/bin/env python3
"""실데이터 위에서 **IR 기준선**과 비교한다 — 순열 기준선은 chance 이지 IR 기준선이 아니다.

cs.IR 심사자가 순열보다 먼저 묻는 것은 "BM25 보다 나은가"다. 제품명에 '선크림'·'세탁세제'
같은 말이 들어 있으니 **카테고리 이름 키워드 매칭만으로 맞히는 것 아니냐**는 의심이 정당하다.
순열 기준선은 그 질문에 답하지 못한다.

같은 대응표·같은 질의 위에서 배포 경로의 두 항을 분해해 잰다(모두 **배포 코드**가 낸 순위):
  lexical-primary  w=50   어휘 겹침이 지배, 밀집은 동점 처리용
  dense-only       w=0    정적 임베딩 코사인만
  deployed         w=0.2  실제 배포 조합
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from aihub_surface import ROOT, perm_baseline
TOPK = 32


def run_w(queries, w):
    payload = "".join(f"{q.replace(chr(9),' ').replace(chr(10),' ')}\t0\n" for q in queries)
    r = subprocess.run(["cargo", "run", "--quiet", "--release", "-p", "oicr-sdk-core",
                        "--example", "sweep_lexical", "--",
                        "--topk", str(TOPK), "--weights", str(w)],
                       cwd=ROOT, input=payload, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"실패(w={w}):\n{r.stderr[-600:]}")
    recs = [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")]
    assert len(recs) == len(queries), f"{len(recs)} != {len(queries)}"
    return [x["top"] for x in recs]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    a = ap.parse_args()
    import numpy as np
    sys.path.insert(0, str(ROOT / "data")); sys.path.insert(0, str(ROOT / "experiments/distill-ko"))
    from taxonomy_ko import TAXONOMY
    import leaf_repr
    names, _x, _y, _z, l2_of = leaf_repr.flatten(TAXONOMY)
    l2_of = np.asarray(l2_of)
    pair = {}
    for li, (p, q, _r) in enumerate(names):
        pair.setdefault(int(l2_of[li]), f"{p}>{q}")
    idp = {v: k for k, v in pair.items()}
    cfg = json.loads((HERE / "aihub_map.json").read_text(encoding="utf-8"))["map"]
    gold = {k: {idp[x] for x in v["l2"]} for k, v in cfg.items()}

    rows = json.loads(pathlib.Path(a.corpus).read_text(encoding="utf-8"))
    seen = {}
    for r in rows:
        seen.setdefault((r["product"], f"{r['domain']}/{r['main']}"), None)
    qs = [(p, k) for (p, k) in seen]
    labels = [k for _, k in qs]
    hook_src = sorted({c.replace(" ", "") for _p, _q, c in names if len(c) >= 2},
                      key=len, reverse=True)
    hook = np.array([any(h in q.replace(" ", "") for h in hook_src) for q, _ in qs])

    def rk(t):
        s, o = [], {}
        for li in t:
            v = int(l2_of[li])
            if v not in s:
                s.append(v); o.setdefault(v, len(s) - 1)
        return o

    out = {"_what": "같은 대응표·같은 질의 위 IR 기준선 비교 (모두 배포 코드가 낸 순위)",
           "n": len(qs), "arms": {}}
    rng = np.random.default_rng(0); n = len(qs); ix = rng.integers(0, n, (2000, n))
    for tag, w in (("lexical-primary (w=50)", 50), ("dense-only (w=0)", 0),
                   ("deployed hybrid (w=0.2)", 0.2)):
        rl = [rk(t) for t in run_w([q for q, _ in qs], w)]
        def hit(labs, r_=rl):
            return np.array([any(r.get(g, 99) < 5 for g in gold[lb]) for r, lb in zip(r_, labs)])
        h = hit(labels); b = h[ix].mean(1)
        out["arms"][tag] = {
            "중분류_top5": round(float(h.mean()), 4),
            "ci": [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)],
            "순열기준선": perm_baseline(hit, labels)["평균"],
            "어휘후크_있음": round(float(h[hook].mean()), 4),
            "어휘후크_없음": round(float(h[~hook].mean()), 4)}
        print(f"  {tag:<26} {h.mean():.4f} "
              f"[{np.percentile(b,2.5):.4f},{np.percentile(b,97.5):.4f}]  "
              f"후크O {h[hook].mean():.4f} · 후크X {h[~hook].mean():.4f}", flush=True)
    # ⛔ 독립 CI 겹침으로 "차이 없음"을 말하지 마라 — 같은 질의를 보는 두 팔은 **짝**이다.
    #    질의 단위로 짝지어 차이의 CI 를 낸다(paired_surface.py 와 같은 논리).
    hits = {}
    for tag, w in (("lexical-primary (w=50)", 50), ("dense-only (w=0)", 0),
                   ("deployed hybrid (w=0.2)", 0.2)):
        rl = [rk(t) for t in run_w([q for q, _ in qs], w)]
        hits[tag] = np.array([any(r.get(g, 99) < 5 for g in gold[lb])
                              for r, lb in zip(rl, labels)], dtype=float)
    dep = hits["deployed hybrid (w=0.2)"]
    out["짝비교_vs_deployed"] = {}
    for tag in ("lexical-primary (w=50)", "dense-only (w=0)"):
        d = dep - hits[tag]
        bs = d[ix].mean(1)
        out["짝비교_vs_deployed"][tag] = {
            "delta_pp": round(float(d.mean() * 100), 2),
            "ci_pp": [round(float(np.percentile(bs, 2.5) * 100), 2),
                      round(float(np.percentile(bs, 97.5) * 100), 2)],
            "유의": bool(np.percentile(bs, 2.5) > 0 or np.percentile(bs, 97.5) < 0)}
        c = out["짝비교_vs_deployed"][tag]
        print(f"  배포 − {tag:<24} {c['delta_pp']:+.2f}pp "
              f"CI[{c['ci_pp'][0]:+.2f},{c['ci_pp'][1]:+.2f}]  "
              f"{'유의' if c['유의'] else '유의하지 않음'}", flush=True)

    (HERE / "ir_baselines.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    print(f"▶ {HERE / 'ir_baselines.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
