#!/usr/bin/env python3
"""stage3cov 를 배포 아티팩트로 승격할지 **코드가 판정한다**.

# 왜 별도 게이트인가

`stage3_eval.py` 는 이득 하나(홀드아웃 리프 top-1)만 잰다. 그건 승격 근거가 아니라
**승격 후보 자격**이다. 배포는 그 이득이 다른 축을 깎지 않았을 때만 정당하다 —
2026-08-29 에 "대분류가 좋아지는데 리프가 나빠지는" 설정을 최적이라 부를 뻔했다.

# 규율

1. ⛔ **배포되는 것으로 잰다** — 128d 절단 + 행별 4bit(v2 포맷과 같은 양자화기).
   fp32 이득이 양자화 후에 안 남으면 그 이득은 배포되지 않는다.
2. ⛔ **목적함수는 리프다.** 대분류가 좋아져도 리프가 유의하게 나빠지면 승격 금지.
3. ⛔ **두 경로를 분리해서 본다**(experiments/mrl-vs-pca/evaluate.py 와 같은 구분):
   학습 없는 내적(리프 검색·STS·소재매칭)과 학습된 선형(YNAT·NSMC)은 서로 반대로 움직인다.
4. 판정은 짝지은 부트스트랩 CI 가 한다. "좋아 보인다"는 측정이 아니다.
5. ⛔ **제품 화면(중분류·L2 top-5)도 목적함수다** (2026-09-07 round 45 사고).
   리프 top-1 만 보던 시절, e6 재증류에서 게이트가 PROMOTE 를 냈는데 실제 배포 경로
   (`sweep_lexical`, 밀집+어휘)로 잰 L2 top-5 는 **−0.34pp** 였다 — 게이트가 제품을
   나쁘게 만드는 후보를 승인할 수 있었다는 뜻이다. 이 파일은 그 L2 top-5 를
   **아티팩트를 갈아끼우지 않고** 순수-파이썬 `leaf_repr` 밀집 경로로 근사한다 —
   `l2_approx_check.py` 로 4개 회차(e6/e7/e8/e10)에서 검증: 부호·유의성 4/4 일치,
   컨트롤(같은 아티팩트끼리)은 Δ=0. 자동 PROMOTE 는 이제 리프 top-1 **과** L2 top-5
   둘 다 유의하게 좋아졌을 때만 나간다 — 화면이 평평하거나(유의 안 함) 나빠지면
   PROMOTE-IF-JUSTIFIED 로 강등해 사람이 화면 숫자를 보고 결정하게 한다.

exit 0 = 승격 가능(PROMOTE 또는 근거 명시 요구) · 1 = 승격 금지(BLOCK) · 3 = 미측정
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# ⛔ `evaluate.py` 가 두 실험 디렉터리에 있다(event-intent · mrl-vs-pca). sys.path 로 섞으면
#    자기 자신을 임포트해 순환한다 — 파일 경로로 직접 로드한다(캡슐 gotcha).
for p in ("data", "experiments/dim-vs-vocab", "experiments/user-sim"):
    sys.path.insert(0, str(ROOT / p))


def _load(name: str, path: pathlib.Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

BOOT = 4000
DIM, BITS = 128, 4

# 기본은 "현 배포본 vs 후보". ⛔ 기준선은 **지금 배포된 것**이어야 한다 — 이미 승격된
# 모델을 건너뛰고 두 세대 전과 비교하면 후보가 실제로 무엇을 개선했는지 알 수 없다.
DEFAULT_BASE = ("stage2-ship", "out/potion-ko-128d-ship")
DEFAULT_CAND = ("stage3cov", "out/potion-ko-128d-stage3cov")

# 축 계약: (표시명, 방향) — direction="primary" 는 유의하게 좋아져야 승격,
# "regression" 은 유의하게 나빠지면 승격 금지, "context" 는 보고만 한다.
AXES = {
    "leaf_holdout_top1": ("홀드아웃 리프 top-1 (제품 목적함수)", "primary"),
    "leaf_holdout_l2_top5": ("홀드아웃 중분류(L2) top-5 (제품 화면)", "primary_surface"),
    "leaf_holdout_top5": ("홀드아웃 리프 top-5", "regression"),
    "leaf_holdout_l1": ("홀드아웃 대분류 top-1", "context"),
    "leaf_train_top1": ("학습 리프 top-1 (암기)", "context"),
    "taxo_material_top1": ("카탈로그 소재 → 리프 top-1 (제품 경로)", "regression"),
    "template_leaf_top1": ("템플릿 발화 → 리프 top-1 (리프명 포함)", "regression"),
    "klue_sts_rho": ("KLUE-STS 스피어만", "regression"),
    "ynat_acc": ("KLUE-YNAT 정확도 (학습된 선형)", "regression"),
    "nsmc_acc": ("NSMC 정확도 (학습된 선형)", "regression"),
}

# ⛔ 2026-09-08 결정 (a) — 사용자가 "10,000 리프까지 계속 민다, 제품 표면(L2 top-5) 우선,
#   일반언어(KLUE-YNAT/KLUE-STS) 회귀는 대가로 수용한다"고 명시적으로 결정했다
#   (redistill_curve.json 의 e7↔e8 경계 트레이드오프에 대한 응답). 조용히 축을 지우지
#   않고 **이름 붙인 상한**으로 대체한다 — "회귀가 있어도 통과"가 아니라 "이 폭까지만".
#
#   근거 수치 (지금까지 실측된 전체 재증류 회귀 폭):
#     YNAT: round45(5,628리프) e8 -0.74pp · e10 -0.78pp / round-cov985(6,020리프) e8 -1.00pp
#           CI[-1.50,-0.51] · e10 -1.21pp CI[-1.76,-0.69]  → 관측 구간 [-0.74, -1.21]pp
#     STS : round45 e10 -0.02(rho) / round-cov985 e8 -0.01 · e10 -0.02        → 관측 구간 [-0.01,-0.02]
#   상한은 **관측된 최악값의 약 1.7~2배**로 잡는다 — 지금 관측된 정상 회귀는 통과시키되,
#   진짜 파국적 회귀(예: 다음 커버리지 확장에서 -3pp 이상)는 여전히 막는다. 표면
#   (leaf_holdout_l2_top5, "primary_surface")과 제품 경로 축(taxo_material_top1·
#   template_leaf_top1)은 대상이 아니다 — 사용자 결정문이 명시한 것은 "일반언어"뿐이고,
#   나머지 회귀축까지 풀면 이 게이트가 실질적으로 무력화된다. NSMC 도 결정문에 이름이
#   없어 엄격 유지(strict) — YNAT·STS 만 완화 대상.
#   ⛔ 값을 손으로 다시 낮추지 마라 — 다음 라운드에서 이 상한 자체가 막히면, 그건
#   게이트가 일하고 있다는 신호지 버그가 아니다. 재조정하려면 새 실측을 이 표에 추가하고
#   날짜·근거를 함께 남긴다.
REGRESSION_TOLERANCE = {
    # axis: 허용 회귀 폭(양수, delta*unit 의 절대값 상한). 초과해야만 차단.
    "ynat_acc": 2.0,       # pp 단위. 관측 최악 -1.21pp 의 ~1.65배
    "klue_sts_rho": 0.04,  # rho 단위. 관측 최악 -0.02 의 2배
}


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> int:
    import numpy as np
    from model2vec import StaticModel
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression
    from datasets import load_dataset
    from taxonomy_ko import TAXONOMY
    quant_rowwise = _load("_mrl_evaluate", ROOT / "experiments/mrl-vs-pca/evaluate.py").quant_rowwise
    from aux_scaling import parse_taxonomy, parse_catalog

    ap = argparse.ArgumentParser()
    ap.add_argument("--utterances",
                    default=str(ROOT / "experiments/event-intent/utterances_llm.json"))
    # ⛔ 2026-08-29: 기본이 8,000/4,000 이었고 그 표본으로 "회귀 0" 이라 판정해 승격했다.
    #    같은 두 모델을 20,000/10,000 으로 다시 재니 YNAT 이 **−0.98pp CI[−1.58,−0.37]**
    #    로 뒤집혔다 — 없던 회귀가 생긴 게 아니라 **검정력이 없었다**. 회귀 게이트를
    #    과소표본으로 돌리면 "무의미" 는 "괜찮다" 가 아니라 "못 봤다" 다.
    ap.add_argument("--train-n", type=int, default=20000)
    ap.add_argument("--test-n", type=int, default=10000)
    ap.add_argument("--out", default=str(HERE / "promote_gate_results.json"))
    ap.add_argument("--base", nargs=2, metavar=("LABEL", "PATH"), default=DEFAULT_BASE,
                    help="기준선 (기본: 현 배포본)")
    ap.add_argument("--cand", nargs=2, metavar=("LABEL", "PATH"), default=DEFAULT_CAND,
                    help="승격 후보")
    a = ap.parse_args()

    base_tag, cand_tag = a.base[0], a.cand[0]
    ARMS = {base_tag: (HERE / a.base[1]), cand_tag: (HERE / a.cand[1])}
    for tag, p in ARMS.items():
        if not (p / "config.json").exists():
            print(f"⬜ UNMEASURED — {tag}: {p} 없음"); return 3

    # ── 재료 ────────────────────────────────────────────────────────────────
    names = [(l1, l2, lf) for l1, m in TAXONOMY.items() for l2, fs in m.items() for lf in fs]
    # ⛔ 리프 표현은 **배포되는 것**이어야 한다 — 가중 센트로이드(0.80/0.14/0.06),
    #    `"{중분류} {리프}"` 문자열이 아니다. 2026-08-29 실측: 그 문자열 표현은 템플릿
    #    질의에서 −16.66pp, 카탈로그 소재에서 −17.23pp 다(`leaf_repr_sweep.json`).
    #    즉 예전 게이트는 제품이 안 쓰는 표현 위에서 판정하고 있었다.
    sys.path.insert(0, str(HERE))
    import leaf_repr
    idx = {n: i for i, n in enumerate(names)}
    l1s = list(TAXONOMY)
    l1_of = np.array([l1s.index(n[0]) for n in names])
    # 제품 화면(L2 top-5)용 중분류 매핑 — 정본(leaf_repr.flatten, build_taxonomy.py 와 동순)에서
    # 가져온다. 여기서 다시 짜면 갈라진다(leaf_repr.py 자신의 경고와 같은 계열).
    _, _, _, _l1of_chk, l2_of = leaf_repr.flatten(TAXONOMY)
    l2_of = np.asarray(l2_of)
    assert list(_l1of_chk) == list(l1_of), "leaf_repr.flatten 순서가 이 파일의 names 순서와 어긋난다"

    up = pathlib.Path(a.utterances)
    if not up.exists():
        print(f"⬜ UNMEASURED — {up} 없음"); return 3
    llm = json.loads(up.read_text(encoding="utf-8"))
    hold = set(json.loads((HERE / "stage3_holdout_leaves.json").read_text(encoding="utf-8")))

    # ⛔ verbatim 을 여기서 **다시** 거른다 — 생성 시 걸렀다는 것을 믿지 않는다.
    q_hold, y_hold, q_train, y_train = [], [], [], []
    for key, texts in llm.items():
        parts = key.split("|")
        if len(parts) != 3 or tuple(parts) not in idx:
            continue
        tgt = (q_hold, y_hold) if key in hold else (q_train, y_train)
        for t in texts:
            if t and parts[2] not in t:
                tgt[0].append(t); tgt[1].append(idx[tuple(parts)])
    if len(q_hold) < 200:
        print(f"⬜ UNMEASURED — 홀드아웃 발화 {len(q_hold)}건"); return 3
    y_hold, y_train = np.asarray(y_hold), np.asarray(y_train)

    # 템플릿 팔 (리프명이 문장에 들어 있는 쉬운 질의) — 이 축이 깎이면 회귀다.
    import random
    from gen_users import build_pool
    tr_pool, _ = build_pool(random.Random(20260828))
    q_tmpl, y_tmpl, per = [], [], {}
    for r in tr_pool:
        if r["band"] != "clear":
            continue
        k = (r["l1"], r["l2"], r["leaf"])
        if k not in idx or per.get(k, 0) >= 2:
            continue
        per[k] = per.get(k, 0) + 1
        q_tmpl.append(r["text"]); y_tmpl.append(idx[k])
    y_tmpl = np.asarray(y_tmpl)

    # 제품 경로: 카탈로그 소재 → 리프
    _, _, _l1, _l2, _leaves = parse_taxonomy((ROOT / "artifacts/taxonomy.bin").read_bytes())
    _, _, items = parse_catalog((ROOT / "artifacts/catalog.bin").read_bytes())
    mat_txt = [t for _, t, _ in items]
    mat_gold = np.array([lf for _, _, lf in items])

    # 실데이터 축
    sts = load_dataset("klue/klue", "sts")["validation"]
    s1t = [r["sentence1"] for r in sts]; s2t = [r["sentence2"] for r in sts]
    sts_gold = np.array([r["labels"]["real-label"] for r in sts])
    yn = load_dataset("klue/klue", "ynat")
    ytr = ([r["title"] for r in yn["train"]][:a.train_n], [r["label"] for r in yn["train"]][:a.train_n])
    yte = ([r["title"] for r in yn["validation"]][:a.test_n], [r["label"] for r in yn["validation"]][:a.test_n])
    ns = load_dataset("e9t/nsmc", revision="refs/convert/parquet")
    ntr = ([r["document"] for r in ns["train"]][:a.train_n], [r["label"] for r in ns["train"]][:a.train_n])
    nte = ([r["document"] for r in ns["test"]][:a.test_n], [r["label"] for r in ns["test"]][:a.test_n])

    print(f"홀드아웃 발화 {len(q_hold):,} · 학습리프 발화 {len(q_train):,} · 템플릿 {len(q_tmpl):,}\n"
          f"소재 {len(mat_txt):,} · STS {len(sts_gold):,} · YNAT {len(yte[0]):,} · NSMC {len(nte[0]):,}",
          flush=True)

    # ── 팔별 측정 ───────────────────────────────────────────────────────────
    def norm(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

    per_item: dict[str, dict] = {}
    scalar: dict[str, dict] = {}
    for tag, path in ARMS.items():
        m0 = StaticModel.from_pretrained(str(path))
        # ⛔ 배포되는 양자화. fp32 로 재면 배포본이 아닌 것을 재게 된다.
        deq, _, _ = quant_rowwise(
            np.ascontiguousarray(np.asarray(m0.embedding)[:, :DIM]).astype(np.float32), BITS)
        enc = StaticModel(deq, m0.tokenizer, config=m0.config).encode
        C = norm(leaf_repr.leaf_vectors(enc, TAXONOMY))
        if tag == cand_tag:
            # 후보가 곧 배포될 것이므로, 헬퍼가 배포 아티팩트를 실제로 재현하는지 여기서 본다.
            try:
                print(f"    leaf_repr ↔ artifacts 코사인 {leaf_repr.verify_against_artifacts(C):.4f}",
                      flush=True)
            except AssertionError as e:
                print(f"    ⚠️ {e}", flush=True)
        d: dict[str, np.ndarray] = {}

        def leaf_hits(q, y, prefix, want_l2=False):
            if len(q) == 0:
                return
            sims = norm(np.asarray(enc(q), dtype=np.float32)) @ C.T
            t1 = sims.argmax(1)
            t5 = np.argpartition(-sims, 5, axis=1)[:, :5]
            d[f"{prefix}_top1"] = (t1 == y)
            d[f"{prefix}_top5"] = np.array([y[i] in t5[i] for i in range(len(y))])
            d[f"{prefix}_l1"] = (l1_of[t1] == l1_of[y])
            if want_l2:
                # 제품 화면(중분류·top-5) — 배포 경로 sweep_lexical 의 hits() 와 같은 규칙:
                # 랭크순 top-32 를 훑어 "같은 중분류=한 칸"으로 세고, 서로 다른 5칸 안에
                # 정답 중분류가 있는지 본다. TOPK=32 는 product_surface.py 와 동일 상수.
                TOPK = 32
                top = np.argpartition(-sims, TOPK, axis=1)[:, :TOPK]
                order = np.argsort(-np.take_along_axis(sims, top, axis=1), axis=1)
                top = np.take_along_axis(top, order, axis=1)
                l2top, gold_l2 = l2_of[top], l2_of[y]
                hit = np.zeros(len(q), dtype=bool)
                for i in range(len(q)):
                    seen: list[int] = []
                    for v in l2top[i]:
                        if v not in seen:
                            seen.append(v)
                        if len(seen) >= 5:
                            break
                    hit[i] = bool(gold_l2[i] in seen)
                d[f"{prefix}_l2_top5"] = hit

        leaf_hits(q_hold, y_hold, "leaf_holdout", want_l2=True)
        leaf_hits(q_train, y_train, "leaf_train")
        d["template_leaf_top1"] = (norm(np.asarray(enc(q_tmpl), dtype=np.float32)) @ C.T).argmax(1) == y_tmpl

        # 소재 → 리프도 **같은 배포 표현**으로 잰다(별도 텍스트를 쓰면 다른 것을 재게 된다).
        d["taxo_material_top1"] = ((norm(np.asarray(enc(mat_txt), dtype=np.float32)) @ C.T).argmax(1) == mat_gold)

        d["_sts_cos"] = (norm(np.asarray(enc(s1t))) * norm(np.asarray(enc(s2t)))).sum(1)
        scalar.setdefault(tag, {})["klue_sts_rho"] = float(spearmanr(d["_sts_cos"], sts_gold).statistic)

        for name, (tr, te) in {"ynat": (ytr, yte), "nsmc": (ntr, nte)}.items():
            clf = LogisticRegression(max_iter=2000).fit(enc(tr[0]), tr[1])
            d[f"{name}_acc"] = (clf.predict(enc(te[0])) == np.asarray(te[1]))
        per_item[tag] = d
        print(f"  {tag}: " + " · ".join(
            f"{k} {v.mean():.4f}" for k, v in d.items() if not k.startswith("_")), flush=True)
        print(f"    klue_sts_rho {scalar[tag]['klue_sts_rho']:.4f}", flush=True)

    # ── 짝지은 부트스트랩 ───────────────────────────────────────────────────
    rng = np.random.default_rng(0)
    A, B = per_item[base_tag], per_item[cand_tag]
    findings, blockers = {}, []
    for axis, (label, kind) in AXES.items():
        if axis == "klue_sts_rho":
            n = len(sts_gold); ix = rng.integers(0, n, (BOOT, n))
            deltas = np.array([
                spearmanr(B["_sts_cos"][i], sts_gold[i]).statistic
                - spearmanr(A["_sts_cos"][i], sts_gold[i]).statistic for i in ix])
            base, new = scalar[base_tag][axis], scalar[cand_tag][axis]
            delta = new - base
            unit = 1.0
        else:
            if axis not in A or axis not in B:
                continue
            n = len(A[axis]); ix = rng.integers(0, n, (BOOT, n))
            deltas = B[axis][ix].mean(1) - A[axis][ix].mean(1)
            base, new = float(A[axis].mean()), float(B[axis].mean())
            delta = new - base
            unit = 100.0
        lo, hi = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
        sig = lo > 0 or hi < 0
        verdict = "무의미"
        if sig:
            verdict = "유의 개선" if delta > 0 else "⛔ 유의 악화"
        findings[axis] = {"label": label, "kind": kind, "n": int(n),
                          base_tag: round(base, 4), cand_tag: round(new, 4),
                          "delta": round(delta * unit, 2), "ci": [round(lo * unit, 2), round(hi * unit, 2)],
                          "significant": bool(sig), "verdict": verdict}
        if kind in ("regression", "primary_surface") and sig and delta < 0:
            tol = REGRESSION_TOLERANCE.get(axis, 0.0)  # 이름 없는 축은 0 = 기존과 동일(무관용)
            if abs(delta * unit) > tol:
                blockers.append(axis)
            else:
                findings[axis]["verdict"] += f" (허용 상한 {tol} 이내 — 2026-09-08 결정(a) 관용)"
        u = "pp" if unit == 100.0 else ""
        print(f"  {label:<38} {base:.4f} → {new:.4f}  "
              f"{delta*unit:+.2f}{u} CI[{lo*unit:+.2f},{hi*unit:+.2f}]  {verdict}", flush=True)

    # ⛔ 게이트는 "나빠졌나"만 판정한다. "좋아졌나"를 승격 조건으로 박아 두면 **커버리지
    #    변경을 영원히 막는다** — 홀드아웃(처음 보는 리프)은 구조적으로 커버리지 이득을
    #    못 본다. 2026-08-29 실측: 학습 리프를 197개 늘려도 홀드아웃은 +0.11pp(무의미)인데,
    #    그 197개는 각자 untrained→trained 로 옮겨 가며 **+10.45pp** 를 번다
    #    (`seen_leaf_results.json`). 지표가 못 보는 이득을 지표로 차단하지 않는다.
    #    ⇒ 회귀가 없고 목적함수가 평평하면 판정은 사람에게 넘기고, 근거를 적게 한다.
    primary = findings.get("leaf_holdout_top1", {})
    improved = bool(primary.get("significant") and primary.get("delta", 0) > 0)
    # ⛔ round 45 사고 수정: 리프가 좋아진다고 자동 PROMOTE 하지 않는다 — 제품 화면
    #    (L2 top-5)도 **같이** 유의하게 좋아져야 자동 승격이다. e6 이 정확히 이 구멍이었다:
    #    리프는 유의 개선인데 화면은 점추정 −0.50pp(유의는 아님)라 예전 게이트는 PROMOTE 를
    #    냈다. 화면이 평평/역행이면(유의 개선 미달) PROMOTE-IF-JUSTIFIED 로 강등해 사람이
    #    화면 숫자를 보고 근거를 대게 한다 — 화면도 리프와 같은 구조적 한계(홀드아웃은 커버리지
    #    이득을 원천적으로 못 본다)를 가지므로, 여기서도 차단(BLOCK)이 아니라 강등만 한다.
    surface = findings.get("leaf_holdout_l2_top5", {})
    surface_improved = bool(surface.get("significant") and surface.get("delta", 0) > 0)
    auto_promote = improved and surface_improved
    ok = not blockers
    if blockers:
        note = "회귀 축이 유의하게 나빠졌다."
    elif auto_promote:
        note = "목적함수(리프)와 제품 화면(L2 top-5) 모두 유의하게 좋아졌다."
    else:
        gaps = []
        if not improved:
            gaps.append("목적함수(리프)가 평평하다")
        if not surface_improved:
            d = surface.get("delta")
            gaps.append(f"제품 화면(L2 top-5)이 유의하게 개선되지 않았다"
                        + (f"(점추정 {d:+.2f}pp)" if d is not None else "(미측정)"))
        note = " · ".join(gaps) + " — 승격하려면 홀드아웃 밖의 근거(예: 커버리지 이득)를 명시해야 한다."
    res = {
        "decision": ("BLOCK" if blockers else
                     "PROMOTE" if auto_promote else "PROMOTE-IF-JUSTIFIED"),
        "primary_improved": improved,
        "surface_improved": surface_improved,
        "note": note,
        "base": {"label": base_tag, "model": a.base[1]},
        "candidate": {"label": cand_tag, "model": a.cand[1]},
        "quantization": f"{DIM}d @ {BITS}bit 행별 스케일 (배포 정본)",
        "leaf_representation": f"배포 센트로이드 {leaf_repr.WEIGHTS} (scripts/build_taxonomy.py)",
        "provenance": {
            "utterances": str(up.relative_to(ROOT)), "utterances_sha256_16": sha(up),
            "leaves_with_utterances": len(llm),
            "holdout_leaves": len(hold),
            "train_n": a.train_n, "test_n": a.test_n, "bootstrap": BOOT,
            "power_note": "⛔ '무의미'는 '괜찮다'가 아니라 '이 표본으로는 못 봤다'다. "
                          "회귀 축의 n 을 줄이면 회귀가 사라진 것처럼 보인다.",
        },
        "blockers": blockers, "axes": findings,
    }
    pathlib.Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n▶ 판정: {res['decision']}" + (f"  차단축: {blockers}" if blockers else ""))
    print(f"   {res['note']}")
    print(f"   {a.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
