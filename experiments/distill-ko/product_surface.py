#!/usr/bin/env python3
"""제품 표면 판정 — **중분류 · top-5** 로 정했을 때 이 모델은 쓸 만한가.

2026-08-29 사용자 결정: 화면은 소분류 top-1 이 아니라 **중분류까지, top-5** 로 보여준다.
같은 모델이 표면에 따라 "쓸 만함"과 "못 씀"을 오가므로, 표면을 정했으면 **그 표면의 숫자**를
재고 그것으로 판정해야 한다. 소분류 top-1(26.4%)은 더 이상 제품 지표가 아니다.

⛔ 점수 조합을 파이썬이 다시 짜지 않는다 — 배포 경로(`sweep_lexical --topk`)가 낸 리프
   순위를 받아 **롤업만** 한다. 중분류 순위는 그 중분류에 속한 리프의 최고 순위로 정한다
   (화면이 중분류를 보여줄 때 하는 일과 같다).
"""
from __future__ import annotations
import argparse, json, pathlib, random, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TOPK = 32
BOOT = 2000


def _rows_sha(rows) -> str:
    """짝 부트스트랩의 전제 — 두 덤프가 **같은 질의를 같은 순서로** 봤는가."""
    import hashlib
    h = hashlib.sha256()
    for q, g in rows:
        h.update(f"{q}\t{g}\n".encode("utf-8"))
    return h.hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="",
                    help="이 분포 하나만 잰다(부분일치). 두 모델을 **짝지어** 비교할 때 "
                         "미견 분포만 돌리려고 쓴다.")
    ap.add_argument("--dump", default="",
                    help="질의별 적중 불리언을 이 경로에 쓴다. 두 아티팩트에서 각각 뜬 뒤 "
                         "`paired_surface.py` 로 **짝 부트스트랩**을 돌리면 독립 CI 겹침이 "
                         "아니라 차이 자체의 CI 를 볼 수 있다. ⛔ 질의 순서가 두 덤프에서 "
                         "같아야 짝이 성립한다 — 같은 코드가 만들므로 결정론이다.")
    a = ap.parse_args()
    # ⛔ 부분 측정이 정본 파일을 덮으면, 다음에 그 파일을 읽는 쪽은 분포 4개 중 1개짜리를
    #    전체 결과로 읽는다. --only 는 반드시 --dump 와 함께 쓴다.
    if a.only and not a.dump:
        sys.exit("⛔ --only 는 --dump 와 함께 써라 — 부분 측정으로 product_surface.json 을 덮지 않는다")

    import numpy as np
    sys.path.insert(0, str(ROOT / "data")); sys.path.insert(0, str(ROOT / "experiments/user-sim"))
    sys.path.insert(0, str(ROOT / "experiments/dim-vs-vocab")); sys.path.insert(0, str(HERE))
    from taxonomy_ko import TAXONOMY
    from aux_scaling import parse_catalog
    import leaf_repr

    names, l1s, l2_list, l1_of, l2_of = leaf_repr.flatten(TAXONOMY)
    idx = {f"{a}|{b}|{c}": i for i, (a, b, c) in enumerate(names)}
    l2_of = np.asarray(l2_of); l1_of = np.asarray(l1_of)

    llm = json.loads((ROOT / "experiments/event-intent/utterances_llm.json").read_text(encoding="utf-8"))
    hold = set(json.loads((HERE / "stage3_holdout_leaves.json").read_text(encoding="utf-8")))
    dists = {}
    dists["처음 보는 카테고리 · 구어체"] = [(t, idx[k]) for k, ts in llm.items() if k in idx and k in hold
                                    for t in ts if t and k.split("|")[2] not in t]
    dists["전체 · 구어체"] = [(t, idx[k]) for k, ts in llm.items() if k in idx
                          for t in ts if t and k.split("|")[2] not in t]
    from gen_users import build_pool
    pool, _ = build_pool(random.Random(20260828))
    per, tmpl = {}, []
    for r in pool:
        k = f"{r['l1']}|{r['l2']}|{r['leaf']}"
        if r["band"] != "clear" or k not in idx or per.get(k, 0) >= 2:
            continue
        per[k] = per.get(k, 0) + 1
        tmpl.append((r["text"], idx[k]))
    dists["이름을 그대로 친 질의"] = tmpl
    _, _, items = parse_catalog((ROOT / "artifacts/catalog.bin").read_bytes())
    dists["광고 소재 → 카테고리"] = [(t, lf) for _, t, lf in items]

    rng = np.random.default_rng(0)
    # ⛔ 어느 모델에서 잰 값인지 박는다. 없으면 화면이 **이전 배포본의 숫자**를 현재
    #    값으로 보여줄 수 있고, 그건 lexical_w_sweep 에서 이미 한 번 고친 결함이다.
    deployed_model = json.loads(
        (ROOT / "artifacts/MANIFEST.json").read_text(encoding="utf-8"))["_provenance"]["source_model"]
    out = {"surface": "중분류(L2) · top-5 (2026-08-29 결정)",
           "deployed_model": deployed_model,
           "levels": {"대분류": len(l1s), "중분류": len(l2_list), "소분류": len(names)},
           "note": "리프 순위는 배포 경로(밀집+어휘 겹침)가 낸 것. 중분류 순위는 그 중분류 리프의 최고 순위.",
           "distributions": {}}
    dump: dict[str, dict] = {}
    for dname, rows in dists.items():
        if a.only and a.only not in dname:
            continue
        payload = "".join(f"{q.replace(chr(9),' ')}\t{g}\n" for q, g in rows)
        r = subprocess.run(["cargo", "run", "--quiet", "--release", "-p", "plataid-sdk-core",
                            "--example", "sweep_lexical", "--", "--topk", str(TOPK)],
                           cwd=ROOT, input=payload, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"실패:\n{r.stderr[-700:]}")
        recs = [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")]
        assert len(recs) == len(rows), f"행 수 불일치 {len(recs)} != {len(rows)}"

        def hits(level_of, k):
            """정답의 상위 카테고리가 top-k 안에 있나 — 같은 상위끼리는 **한 칸**으로 센다."""
            out_ = np.zeros(len(recs), dtype=bool)
            for i, rec in enumerate(recs):
                g = level_of[rec["gold"]]
                seen, rank = [], None
                for li in rec["top"]:
                    v = int(level_of[li])
                    if v not in seen:
                        seen.append(v)
                    if v == g:
                        rank = seen.index(v); break
                out_[i] = rank is not None and rank < k
            return out_

        leaf1 = np.array([rec["top"][:1] == [rec["gold"]] for rec in recs])
        leaf5 = np.array([rec["gold"] in rec["top"][:5] for rec in recs])
        res = {"n": len(recs),
               "소분류_top1": leaf1, "소분류_top5": leaf5,
               "중분류_top1": hits(l2_of, 1), "중분류_top5": hits(l2_of, 5),
               "대분류_top1": hits(l1_of, 1), "대분류_top5": hits(l1_of, 5)}
        ci = {}
        n = len(recs); ix = rng.integers(0, n, (BOOT, n))
        for k, v in res.items():
            if k == "n":
                continue
            b = v[ix].mean(1)
            ci[k] = {"acc": round(float(v.mean()), 4),
                     "ci": [round(float(np.percentile(b, 2.5)), 4),
                            round(float(np.percentile(b, 97.5)), 4)]}
        out["distributions"][dname] = {"n": n, **ci}
        if a.dump:
            # ⛔ 리프 인덱스 기반 sha 는 **택소노미가 커지면 질의가 같아도 바뀐다**(중간
            #    삽입이 뒤의 인덱스를 민다). 그래서 이름 기반 sha 를 함께 낸다 — 재색인에는
            #    불변이고, 정답 리프가 진짜로 바뀌면 여전히 잡는다.
            dump[dname] = {"n": n, "queries_sha": _rows_sha(rows),
                           "queries_key_sha": _rows_sha([(q, "|".join(names[g])) for q, g in rows]),
                           **{k: v.astype(int).tolist() for k, v in res.items() if k != "n"}}
        print(f"\n── {dname} (n={n:,})", flush=True)
        for k in ("대분류_top1", "중분류_top1", "중분류_top5", "소분류_top1", "소분류_top5"):
            mark = "  ★제품 표면" if k == "중분류_top5" else ""
            print(f"   {k:<14} {ci[k]['acc']:.4f}  CI[{ci[k]['ci'][0]:.4f},{ci[k]['ci'][1]:.4f}]{mark}", flush=True)

    if a.dump:
        pathlib.Path(a.dump).write_text(json.dumps(
            {"deployed_model": deployed_model, "dists": dump}, ensure_ascii=False),
            encoding="utf-8")
        print(f"\n▶ 덤프 {a.dump}  (모델 {deployed_model})")
        return 0  # ⛔ 부분 측정으로 정본 파일을 덮지 않는다
    (HERE / "product_surface.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    print(f"\n▶ {HERE / 'product_surface.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
