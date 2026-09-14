#!/usr/bin/env python3
"""실제 한국어 커머스 텍스트로 배포 경로를 잰다 (AI-Hub 71603 Validation).

이 저장소의 정확도는 그동안 **전부 합성 발화**에서 나왔다. 여기서는 사람이 쓴 실제
쇼핑몰 리뷰와 실제 판매 제품명을 그대로 배포 경로(`sweep_lexical`)에 넣고, AI-Hub 가
붙여 둔 카테고리 라벨을 정답으로 본다.

⛔ 점수 조합을 파이썬이 다시 짜지 않는다 — `product_surface.py` 와 같은 계약으로
   리프 순위를 받아 중분류로 롤업만 한다.
⛔ AI-Hub 의 20개 카테고리와 우리 256개 중분류는 1:1 이 아니다. 대응은 사람이 만든
   `aihub_map.json` 이고, **대응 폭이 넓으면 적중이 쉬워진다.** 그래서 같은 대응으로
   라벨만 섞은 **순열 기준선**을 함께 낸다 — 그 차이가 모델이 실제로 한 일이다.
⛔ 데이터는 재배포하지 않는다(AI-Hub 표준 약관). 코퍼스 경로는 인자로 받는다.
"""
from __future__ import annotations
import argparse, collections, json, pathlib, random, re, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TOPK = 32
BOOT = 2000


def run_deploy(queries: list[str]) -> list[list[int]]:
    payload = "".join(f"{q.replace(chr(9),' ').replace(chr(10),' ')}\t0\n" for q in queries)
    r = subprocess.run(["cargo", "run", "--quiet", "--release", "-p", "oicr-sdk-core",
                        "--example", "sweep_lexical", "--", "--topk", str(TOPK)],
                       cwd=ROOT, input=payload, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"배포 경로 실패:\n{r.stderr[-800:]}")
    recs = [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")]
    assert len(recs) == len(queries), f"행 수 불일치 {len(recs)} != {len(queries)}"
    return [rec["top"] for rec in recs]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="aihub_corpus.json (재배포 금지 — 로컬 경로)")
    ap.add_argument("--review-sample", type=int, default=6000)
    a = ap.parse_args()

    import numpy as np
    sys.path.insert(0, str(ROOT / "data")); sys.path.insert(0, str(ROOT / "experiments/distill-ko"))
    from taxonomy_ko import TAXONOMY
    import leaf_repr

    names, l1s, l2_list, l1_of, l2_of = leaf_repr.flatten(TAXONOMY)
    l2_of = np.asarray(l2_of); l1_of = np.asarray(l1_of)
    l2_id = {f"{a_}>{b_}": i for i, (a_, b_) in enumerate(
        (x.split(">") if ">" in x else (None, x)) for x in [])}
    # l2_list 는 중분류 이름만 담는다 — 같은 이름이 여러 대분류에 없는지 확인하고 인덱스를 잡는다.
    pair_of_l2 = {}
    for li, (a_, b_, _c) in enumerate(names):
        pair_of_l2.setdefault(int(l2_of[li]), f"{a_}>{b_}")
    assert len(set(pair_of_l2.values())) == len(pair_of_l2), "중분류 인덱스가 대분류를 섞는다"
    id_of_pair = {v: k for k, v in pair_of_l2.items()}
    l1_of_l2 = {k: int(l1_of[[i for i in range(len(names)) if int(l2_of[i]) == k][0]])
                for k in pair_of_l2}

    cfg = json.loads((HERE / "aihub_map.json").read_text(encoding="utf-8"))["map"]
    gold_l2 = {k: {id_of_pair[x] for x in v["l2"]} for k, v in cfg.items()}
    gold_l1 = {k: {l1_of_l2[i] for i in v} for k, v in gold_l2.items()}

    rows = json.loads(pathlib.Path(a.corpus).read_text(encoding="utf-8"))
    key = lambda r: f"{r['domain']}/{r['main']}"
    assert set(map(key, rows)) <= set(gold_l2), set(map(key, rows)) - set(gold_l2)

    rnd = random.Random(20260914)
    sets: dict[str, list[tuple[str, str]]] = {}
    seen = {}
    for r in rows:
        seen.setdefault((r["product"], key(r)), None)
    sets["실제 제품명"] = [(p, k) for (p, k) in seen]
    revs = [(r["text"], key(r)) for r in rows if r["text"] and len(r["text"]) >= 6]
    sets["실제 리뷰 문장"] = rnd.sample(revs, min(a.review_sample, len(revs)))

    rng = np.random.default_rng(0)
    out = {"source": "AI-Hub 71603 Validation · 라벨링데이터 20묶음",
           "surface": "중분류(L2) · top-5 — product_surface.py 와 같은 표면",
           "deployed_model": json.loads((ROOT / "artifacts/MANIFEST.json").read_text(
               encoding="utf-8"))["_provenance"]["source_model"],
           "map_avg_l2": round(sum(len(v) for v in gold_l2.values()) / len(gold_l2), 2),
           "sets": {}}

    for sname, qs in sets.items():
        tops = run_deploy([q for q, _ in qs])
        labels = [k for _, k in qs]

        def ranks(level_of, top):
            seen_, out_ = [], {}
            for li in top:
                v = int(level_of[li])
                if v not in seen_:
                    seen_.append(v)
                    out_.setdefault(v, len(seen_) - 1)
            return out_

        l2r = [ranks(l2_of, t) for t in tops]
        l1r = [ranks(l1_of, t) for t in tops]

        def hit(rlist, goldmap, labs, k):
            return np.array([any(rl.get(g, 99) < k for g in goldmap[lb])
                             for rl, lb in zip(rlist, labs)])

        res = {"중분류_top1": hit(l2r, gold_l2, labels, 1),
               "중분류_top5": hit(l2r, gold_l2, labels, 5),
               "대분류_top1": hit(l1r, gold_l1, labels, 1),
               "대분류_top5": hit(l1r, gold_l1, labels, 5)}
        # 순열 기준선 — 같은 대응, 같은 라벨 분포, 질의와의 연결만 끊는다.
        perm = list(labels); rnd.shuffle(perm)
        base = {k2: hit(l2r if "중분류" in k2 else l1r,
                        gold_l2 if "중분류" in k2 else gold_l1, perm,
                        1 if "top1" in k2 else 5) for k2 in res}

        n = len(qs); ix = rng.integers(0, n, (BOOT, n)); rec = {"n": n}
        for k2, v in res.items():
            b = v[ix].mean(1)
            rec[k2] = {"acc": round(float(v.mean()), 4),
                       "ci": [round(float(np.percentile(b, 2.5)), 4),
                              round(float(np.percentile(b, 97.5)), 4)],
                       "순열기준선": round(float(base[k2].mean()), 4)}
        # ⛔ 모델코드 분해가 진짜 축이다. 브랜드 마스킹(OO/**)은 **도메인과 교란**돼 있다 —
        #    AI-Hub 는 패션·화장품·생활만 가리고 가전·IT 는 그대로 둔다. 그래서 "가려진 쪽이
        #    더 잘 맞는다"는 관측은 마스킹의 효과가 아니라 가전·IT 제품명이 모델코드라는
        #    사실이다("신일전자 SIF-DCP25NK"). 두 분해를 같이 내고, 인용은 모델코드 쪽으로 한다.
        modelcode = np.array([bool(re.search(r"[A-Za-z]{2,}[-\s]?\d{2,}|\d{2,}[A-Za-z]{2,}", q))
                              for q, _ in qs])
        rec["모델코드형_제품명"] = {}
        for tag, sel in (("모델코드 포함", modelcode), ("한국어 서술", ~modelcode)):
            if sel.sum() < 30:
                continue
            rec["모델코드형_제품명"][tag] = {"n": int(sel.sum()), **{
                k2: round(float(v[sel].mean()), 4) for k2, v in res.items()}}
        # 브랜드 마스킹 분해 — AI-Hub 는 패션·화장품·생활 도메인의 브랜드를 OO/** 로 가린다.
        #   가려진 질의는 어휘 겹침 신호가 통째로 빠지므로, 섞어서 하나의 수로 내면 안 된다.
        masked = np.array([bool(re.search(r"(^|\s)OO(\s|$)|\*\*", q)) for q, _ in qs])
        rec["브랜드마스킹"] = {}
        for tag, sel in (("가려짐", masked), ("그대로", ~masked)):
            if sel.sum() < 30:
                continue
            rec["브랜드마스킹"][tag] = {"n": int(sel.sum()), **{
                k2: round(float(v[sel].mean()), 4) for k2, v in res.items()}}
        # 라벨별 중분류 top-5 — 어디서 무너지나
        lab = np.array(labels)
        rec["라벨별_중분류_top5"] = {l: {"n": int((lab == l).sum()),
                                    "acc": round(float(res["중분류_top5"][lab == l].mean()), 4)}
                                 for l in sorted(set(labels))}
        out["sets"][sname] = rec
        print(f"\n── {sname} (n={n:,})", flush=True)
        for k2 in ("대분류_top1", "대분류_top5", "중분류_top1", "중분류_top5"):
            c = rec[k2]; mark = "  ★제품 표면" if k2 == "중분류_top5" else ""
            print(f"   {k2:<12} {c['acc']:.4f} CI[{c['ci'][0]:.4f},{c['ci'][1]:.4f}]"
                  f"  · 순열기준선 {c['순열기준선']:.4f}{mark}", flush=True)

    # 라벨별 분해 — 어디서 무너지나
    (HERE / "aihub_surface.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print(f"\n▶ {HERE / 'aihub_surface.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
