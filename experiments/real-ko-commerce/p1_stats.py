#!/usr/bin/env python3
"""4판 리뷰의 통계 요구 4건을 한 번에 잰다 (실제 제품명 1,174쌍).

  P1-4  셀별 **맞춤 순열 귀무** — 2×2 네 셀 각각에서 그 셀의 라벨만 섞는다.
        0.2847 이 "우연 수준으로 내려간다"고 말하려면 그 셀의 우연이 얼마인지 알아야 한다.
  P1-5  **BM25** 리프 문자열 기준선 — lexical-primary(w=50)는 어휘 '전용'이 아니다.
        밀집 점수가 여전히 동점을 가른다. 진짜 어휘 전용 기준선을 따로 세운다.
  P1-2  **교사 vs 밀집전용** 짝 부트스트랩 — 교사 대 하이브리드보다 깨끗한 대조다.
        하이브리드에는 어휘 항이 섞여 있어 "정적 임베딩의 손실"을 가리기 때문이다.
        교사+어휘(w=0.2) 팔도 같이 낸다.
  P1-9  **출처 카테고리 고정효과** — 후크·모델코드 효과가 도메인 구성의 그림자인지 본다.

⛔ AI-Hub 텍스트는 이 머신을 떠나지 않는다(약관상 제3자 제공 소지). 교사도 로컬에서 돈다.
⛔ 어휘 항은 파이썬이 재구현하지 않는다 — Rust `--dump-lexical` 이 배포 경로의 값을 그대로 준다.
"""
from __future__ import annotations
import argparse, json, os, pathlib, random, re, subprocess, sys

os.environ.setdefault("USE_TF", "0")                 # ⛔ transformers 의 TF 임포트가 맥에서 데드락
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TOPK, BOOT, PERM = 32, 2000, 400
TEACHER = "jhgan/ko-sroberta-multitask"


def sweep(queries, extra):
    payload = "".join(f"{q.replace(chr(9),' ').replace(chr(10),' ')}\t0\n" for q in queries)
    r = subprocess.run(["cargo", "run", "--quiet", "--release", "-p", "oicr-sdk-core",
                        "--example", "sweep_lexical", "--", *extra],
                       cwd=ROOT, input=payload, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"sweep 실패 {extra}:\n{r.stderr[-800:]}")
    recs = [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")]
    assert len(recs) == len(queries), f"{len(recs)} != {len(queries)}"
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()

    import numpy as np, torch
    from transformers import AutoModel, AutoTokenizer
    sys.path.insert(0, str(ROOT / "data")); sys.path.insert(0, str(ROOT / "experiments/distill-ko"))
    from taxonomy_ko import TAXONOMY
    import leaf_repr

    names, _l1s, _l2l, _l1of, l2_of = leaf_repr.flatten(TAXONOMY)
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
        seen.setdefault((r["product"], f'{r["domain"]}/{r["main"]}'), None)
    qs = [q for q, _ in seen]; labels = [k for _, k in seen]
    n = len(qs); print(f"평가쌍 {n:,}", flush=True)

    def l2rank(top):
        s, o = [], {}
        for li in top:
            v = int(l2_of[li])
            if v not in s:
                s.append(v); o.setdefault(v, len(s) - 1)
        return o

    def hits(tops, labs=None):
        labs = labels if labs is None else labs
        return np.array([any(l2rank(t).get(g, 99) < 5 for g in gold[lb])
                         for t, lb in zip(tops, labs)], dtype=float)

    # ── 팔 1·2: 배포(w=0.2) · 밀집전용(w=0) ──────────────────────────────
    arms = {}
    arms["deployed (w=0.2)"] = [r["top"] for r in sweep(qs, ["--topk", str(TOPK)])]
    arms["dense-only (w=0)"] = [r["top"] for r in sweep(qs, ["--topk", str(TOPK),
                                                             "--weights", "0"])]
    print("  배포·밀집전용 완료", flush=True)

    # ── 팔 3: BM25 (리프 문자열, 문자 bigram) ────────────────────────────
    # ⛔ 한국어 제품명은 공백 토큰화가 신뢰되지 않는다(붙여쓴 복합어). 문자 bigram 을 쓴다.
    import collections, math
    def toks(s):
        s = re.sub(r"\s+", "", s)
        return [s[i:i + 2] for i in range(len(s) - 1)] or ([s] if s else [])
    docs = [toks(f"{p} {q} {r}") for p, q, r in names]
    df = collections.Counter(t for d in docs for t in set(d))
    N, avgdl = len(docs), sum(map(len, docs)) / len(docs)
    idf = {t: math.log(1 + (N - v + 0.5) / (v + 0.5)) for t, v in df.items()}
    tf = [collections.Counter(d) for d in docs]
    k1, b = 1.5, 0.75
    post = collections.defaultdict(list)
    for di, c in enumerate(tf):
        dl = len(docs[di])
        for t, f in c.items():
            post[t].append((di, f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))))
    bm_tops = []
    for q in qs:
        sc = np.zeros(N)
        for t in set(toks(q)):
            if t in post:
                w = idf[t]
                for di, v in post[t]:
                    sc[di] += w * v
        bm_tops.append(list(np.argsort(-sc)[:TOPK]))
    arms["BM25 (leaf strings, char bigram)"] = bm_tops
    print("  BM25 완료", flush=True)

    # ── 팔 4·5: 교사 · 교사+어휘(0.2) ───────────────────────────────────
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(TEACHER)
    mdl = AutoModel.from_pretrained(TEACHER).to(dev).eval()

    def enc(texts):
        out = []
        for i in range(0, len(texts), a.batch):
            bt = tok(texts[i:i + a.batch], padding=True, truncation=True,
                     max_length=128, return_tensors="pt").to(dev)
            with torch.no_grad():
                h = mdl(**bt).last_hidden_state
            m = bt["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)      # mean pooling — 교사 카드의 계약
            out.append(torch.nn.functional.normalize(v, dim=-1).float().cpu().numpy())
        return np.concatenate(out)

    # ⛔ 2026-09-14 교정. 처음에는 교사에게 평평한 문자열 `"{대} {중} {리프}"` 를 줬고,
    #    주석에 "교사에게만 좋은 문장을 주면 비교가 아니다" 라고 적었다. **방향이 반대였다** —
    #    배포 경로는 그 문자열을 쓰지 않는다. `scripts/build_taxonomy.py` 는 가중 센트로이드
    #        unit(0.80·unit(emb(리프)) + 0.14·unit(emb(중분류)) + 0.06·unit(emb(대분류)))
    #    를 쓰고, `leaf_repr.py` 헤더의 실측은 평평한 문자열이 그보다 **16.66pp 나쁘다**고
    #    말한다. 즉 교사는 우리보다 나쁜 표현으로 재지고 있었다.
    #    ⇒ 두 팔을 다 낸다: 같은 문자열(하한)과 같은 **구성**(공정).
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_taxonomy import LEAF_W, MID_W, TOP_W

    def unit(a):
        n = np.linalg.norm(a, axis=-1, keepdims=True)
        return a / np.clip(n, 1e-9, None)

    L_flat = enc([f"{p} {q} {r}" for p, q, r in names])
    uniq_leaf = enc([r for _p, _q, r in names])
    mids = sorted({f"{p} {q}".split(" ")[1] for p, q, _r in names})
    mid_ix = {m: i for i, m in enumerate(mids)}
    M = enc(mids)
    tops = sorted({p for p, _q, _r in names}); top_ix = {a: i for i, a in enumerate(tops)}
    T_ = enc(tops)
    L_cent = unit(LEAF_W * unit(uniq_leaf)
                  + MID_W * unit(M[[mid_ix[q] for _p, q, _r in names]])
                  + TOP_W * unit(T_[[top_ix[p] for p, _q, _r in names]]))
    Q = enc(qs)
    S = Q @ L_cent.T                     # 공정 팔 — 배포와 **같은 구성**
    S_flat = Q @ L_flat.T                # 하한 팔 — 배포와 같은 문자열
    # ⛔ 어느 쪽을 정본으로 삼나 — **교사에게 더 좋은 쪽**이다. 이 비교의 목적은
    #    "421MB 가 이 과제에서 무엇을 사 주나"이고, 교사를 불리하게 재면 우리 설계의
    #    대가를 과소평가하게 된다. 실측: 평평한 문자열이 센트로이드보다 낫다
    #    (0.7751 vs 0.7564) — `leaf_repr.py` 의 −16.66pp 는 **우리 정적 인코더** 얘기지
    #    트랜스포머엔 반대로 작용한다(세 벡터를 평균내면 문장 문맥이 사라진다).
    arms["teacher (768d, 421MB)"] = [list(x) for x in np.argsort(-S_flat, axis=1)[:, :TOPK]]
    arms["teacher, deployment-style centroid"] = [list(x) for x in
                                                  np.argsort(-S, axis=1)[:, :TOPK]]
    lex = np.zeros_like(S)
    for i, rec in enumerate(sweep(qs, ["--dump-lexical"])):
        for j, v in rec["nz"]:
            lex[i, j] = v
    arms["teacher + lexical (w=0.2)"] = [list(x) for x in
                                         np.argsort(-(S_flat + 0.2 * lex), axis=1)[:, :TOPK]]
    print("  교사 2팔 완료", flush=True)

    H = {k: hits(v) for k, v in arms.items()}

    # ── 축 정의 (P1-1: 이름은 '형태소 공유'가 아니라 '리프 이름 어휘 앵커') ─────
    leafn = sorted({c.replace(" ", "") for _x, _y, c in names if len(c) >= 2},
                   key=len, reverse=True)
    anchor = np.array([any(ln in q.replace(" ", "") for ln in leafn) for q in qs])
    code = np.array([bool(re.search(r"[A-Za-z]{2,}[-\s]?\d{2,}|\d{2,}[A-Za-z]{2,}", q))
                     for q in qs])
    lab = np.array(labels)

    rng = np.random.default_rng(0); ix = rng.integers(0, n, (BOOT, n))
    out = {"_source": "AI-Hub 71603 Validation · 라벨링데이터 20묶음 (출처: AI 허브)",
           "_surface": "중분류(L2) top-5", "n": n,
           "_axis_note": ("어휘앵커 = 질의 문자열이 어떤 리프 이름(공백 제거, 2자 이상)을 "
                          "부분문자열로 포함하는가. 형태소 분석이 아니라 **리프 이름 전체의 "
                          "부분문자열 검사**다."),
           "arms": {k: {"acc": round(float(v.mean()), 4),
                        "앵커O": round(float(v[anchor].mean()), 4),
                        "앵커X": round(float(v[~anchor].mean()), 4)} for k, v in H.items()}}

    # ── P1-2 짝 부트스트랩 ─────────────────────────────────────────────
    def paired(x, y):
        d = x - y; bs = d[ix].mean(1)
        f = lambda s: {"delta_pp": round(float(d[s].mean() * 100), 2),
                       "ci_pp": [round(float(np.percentile((d[s])[np.random.default_rng(1)
                                 .integers(0, s.sum(), (BOOT, int(s.sum())))].mean(1), 2.5) * 100), 2),
                                 round(float(np.percentile((d[s])[np.random.default_rng(1)
                                 .integers(0, s.sum(), (BOOT, int(s.sum())))].mean(1), 97.5) * 100), 2)],
                       "n": int(s.sum())}
        return {"전체": {"delta_pp": round(float(d.mean() * 100), 2),
                       "ci_pp": [round(float(np.percentile(bs, 2.5) * 100), 2),
                                 round(float(np.percentile(bs, 97.5) * 100), 2)], "n": n},
                "앵커O": f(anchor), "앵커X": f(~anchor)}
    out["짝비교"] = {
        "교사 − 밀집전용": paired(H["teacher (768d, 421MB)"], H["dense-only (w=0)"]),
        "교사 − 배포": paired(H["teacher (768d, 421MB)"], H["deployed (w=0.2)"]),
        "교사+어휘 − 배포": paired(H["teacher + lexical (w=0.2)"], H["deployed (w=0.2)"]),
        "배포 − BM25": paired(H["deployed (w=0.2)"], H["BM25 (leaf strings, char bigram)"]),
        "배포 − 밀집전용": paired(H["deployed (w=0.2)"], H["dense-only (w=0)"]),
        # 민감도: 교사에게 배포와 같은 **구성**(가중 센트로이드)을 줬을 때
        "교사(센트로이드) − 배포": paired(H["teacher, deployment-style centroid"],
                                H["deployed (w=0.2)"])}

    # ── P1-4 셀별 맞춤 순열 귀무 ────────────────────────────────────────
    dep_tops = arms["deployed (w=0.2)"]
    cells = {}
    rnd = random.Random(7)
    for cv in (False, True):
        for hv in (True, False):
            m = (code == cv) & (anchor == hv)
            if m.sum() < 30:
                continue
            sub_tops = [t for t, k in zip(dep_tops, m) if k]
            sub_lab = [l for l, k in zip(labels, m) if k]
            accs = []
            for _ in range(PERM):
                p = list(sub_lab); rnd.shuffle(p)
                accs.append(float(hits(sub_tops, p).mean()))
            accs = np.array(accs); obs = float(H["deployed (w=0.2)"][m].mean())
            cells[f"모델코드{'O' if cv else 'X'}·어휘앵커{'O' if hv else 'X'}"] = {
                "n": int(m.sum()), "관측": round(obs, 4),
                "셀내_순열귀무": round(float(accs.mean()), 4),
                "귀무CI": [round(float(np.percentile(accs, 2.5)), 4),
                          round(float(np.percentile(accs, 97.5)), 4)],
                "초과_pp": round((obs - float(accs.mean())) * 100, 2),
                "귀무초과_p": round(float((accs >= obs).mean()), 4)}
    out["P1_4_셀별_맞춤귀무"] = cells

    # ── 라벨별 **정확한** 맞춤 귀무 ─────────────────────────────────────
    #   라벨 L 의 귀무 = 아무 질의의 top-5 에 L 의 정답 중분류가 들어갈 확률.
    #   순열 추출이 아니라 전 질의에 대한 평균이라 정확값이다(표집 오차 0).
    dep_l2r = [l2rank(t) for t in dep_tops]
    per_label = {}
    for s in sorted(set(lab)):
        g = gold[s]
        null = float(np.mean([any(r.get(x, 99) < 5 for x in g) for r in dep_l2r]))
        m = lab == s
        per_label[s] = {"n": int(m.sum()),
                        "관측": round(float(H["deployed (w=0.2)"][m].mean()), 4),
                        "맞춤귀무": round(null, 4),
                        "초과_pp": round((float(H["deployed (w=0.2)"][m].mean()) - null) * 100, 2),
                        "허용_중분류수": len(g),
                        "어휘앵커_비율": round(float(anchor[m].mean()), 3)}
    out["라벨별_맞춤귀무"] = dict(sorted(per_label.items(), key=lambda kv: kv[1]["초과_pp"]))

    # ── P1-9 출처 카테고리 고정효과 ─────────────────────────────────────
    import pandas as pd, statsmodels.formula.api as smf
    df = pd.DataFrame({"y": H["deployed (w=0.2)"], "anchor": anchor.astype(int),
                       "code": code.astype(int), "src": lab})
    # ⛔ 로지스틱은 여기서 **특이행렬로 죽는다** — 20개 출처 중 일부가 완전분리이거나
    #    그 안에서 code 가 상수라 설계행렬의 랭크가 모자란다. 규제를 걸어 억지로 수를
    #    만드는 대신 **선형확률모형(OLS) + HC1 로버스트 표준오차**로 간다. 분리에 면역이고,
    #    계수가 그대로 pp 라 본문 수치와 같은 단위로 읽힌다.
    sep = {s: round(float(df.y[df.src == s].mean()), 3) for s in sorted(set(lab))}
    fits = {"_추정량": "선형확률모형(OLS) · HC1 로버스트 SE · 계수 단위 = 확률 pp",
            "_왜_로짓이_아닌가": "출처 고정효과를 넣으면 설계행렬이 특이해진다(완전분리 출처 존재)",
            "_출처별_정답률": sep}
    for tag, f in (("고정효과 없음", "y ~ anchor * code"),
                   ("출처 카테고리 고정효과", "y ~ anchor * code + C(src)")):
        m = smf.ols(f, data=df).fit(cov_type="HC1")
        fits[tag] = {k: {"coef_pp": round(float(m.params[k]) * 100, 2),
                         "p": float(f"{m.pvalues[k]:.3g}"),
                         "ci_pp": [round(float(m.conf_int().loc[k, 0]) * 100, 2),
                                   round(float(m.conf_int().loc[k, 1]) * 100, 2)]}
                     for k in ("anchor", "code", "anchor:code") if k in m.params}
        fits[tag]["r2"] = round(float(m.rsquared), 4)

    # ⛔ 계수가 줄어든 것이 **교란 해소**인지 **다중공선성으로 SE 만 커진 것**인지 구분한다.
    #    층화(카테고리 내) 가중평균 차이를 직접 계산해 본다 — 모형 가정이 없는 대조군이다.
    def strat(flag):
        num = den = 0.0; usable = 0
        for s in sorted(set(lab)):
            m_ = lab == s
            a1, a0 = (m_ & flag), (m_ & ~flag)
            if a1.sum() < 5 or a0.sum() < 5:
                continue
            w = float(m_.sum()); usable += 1
            num += w * (H["deployed (w=0.2)"][a1].mean() - H["deployed (w=0.2)"][a0].mean())
            den += w
        return {"층화차이_pp": round(num / den * 100, 2) if den else None,
                "쓸수있는_출처수": usable, "전체_출처수": len(set(lab))}
    fits["층화_대조"] = {"어휘앵커": strat(anchor), "모델코드": strat(code)}
    fits["출처별_구성"] = {s: {"n": int((lab == s).sum()),
                          "모델코드_비율": round(float(code[lab == s].mean()), 3),
                          "어휘앵커_비율": round(float(anchor[lab == s].mean()), 3)}
                      for s in sorted(set(lab))}
    out["P1_9_고정효과"] = fits

    (HERE / "p1_stats.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print(json.dumps(out["arms"], ensure_ascii=False, indent=2))
    print(json.dumps(out["P1_4_셀별_맞춤귀무"], ensure_ascii=False, indent=2))
    print(f"\n▶ {HERE / 'p1_stats.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
