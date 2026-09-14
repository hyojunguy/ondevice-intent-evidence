#!/usr/bin/env python3
"""λ_p 스윕 — 구매확률 부스트(`add_purchase_boost`, 배포 코드 그대로)를 켜면 REES46 실
구매에서 정답 L2 의 텍스트 랭킹 순위가 실제로 좋아지는가.

행 구성: STRONG(다수결≥60%) 브리지 카테고리만, k≥2 인 (유저,카테고리)에서 앞선 구매들을
purchase history 로, 마지막 구매 시점을 "지금 이 카테고리를 검색하는 순간"으로 삼아
Qwen 생성 문구 하나를 질의로 쓴다 — 텍스트만으론 그 문구가 정답 L2 를 몇 등으로 보는지,
자기 구매이력을 부스트로 얹으면 몇 등으로 바뀌는지를 실 배포 코드(`purchase_rank_sweep`
Rust 예제)로 잰다.

⛔ 질의 문구는 실사용자가 그 순간 친 검색어가 아니라 **Qwen 이 그 카테고리를 보고 생성한
대리 문구**다 — "질의×구매 조인트가 실재하는지"를 재는 게 아니라 "부스트 메커니즘이
이런 형태의 애매한 질의에서 도움이 되는 방향인지"를 재는 것이다. 인용 시 이 구분을 유지.

`--cross` 플래그: 이력에 타겟 카테고리 자신뿐 아니라 같은 유저가 (now_s 이전에) 산 다른
STRONG 카테고리도 함께 넣는다 — `add_purchase_boost` 는 프로필에 기록된 leaf 를 전부
부스트하므로, 경쟁 카테고리도 같이 부스트돼 타겟의 랭킹을 깎을 수 있다. 기본(플래그 없음)은
2026-09-05 원래 실행과 동일한 낙관적 세팅(타겟 카테고리 이력만).

`--lambdas`/`--out-tag`: 2026-09-05(22) 가 남긴 미결 — 크로스카테고리 세팅의 최적점이
0.5~1.5 사이 어딘가라는 것까지만 좁혔다(1.0 은 유의하게 좋고 2.0 은 유의하게 나쁘다).
더 촘촘한 격자를 돌릴 때 **기존 결과 파일을 덮지 않도록** 출력 태그를 분리한다.
⛔ 부트스트랩(`lambda_p_bootstrap.py`)은 raw 파일을 재사용하므로 태그를 바꿔 돌린
스윕은 부트스트랩에도 같은 raw 경로를 넘겨야 한다.
"""
from __future__ import annotations
import argparse, json, random, subprocess, sys, pathlib

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LAMBDAS = [0.0, 0.2, 0.5, 1.0, 2.0]


def strong_categories() -> dict:
    bridge = json.loads((HERE / "rees46_category_bridge.json").read_text())
    out = {}
    for cat, v in bridge.items():
        m2 = v.get("mapped_l2")
        m1 = v.get("mapped")
        if not m2 or not m1 or m2["votes"] / m2["of"] < 0.6:
            continue
        if m1["l2"] != m2["l2"]:  # 리프-레벨과 L2-레벨 투표가 서로 다른 카테고리를 가리키면 제외
            continue
        if not v.get("phrases"):
            continue
        out[cat] = {"l1": m1["l1"], "l2": m1["l2"], "leaf": m1["leaf"], "phrases": v["phrases"]}
    return out


def build_rows(cats: dict, cross: bool = False, lambdas: list[float] | None = None) -> list[dict]:
    """cross=False: 2026-09-05 원래 세팅(타겟 카테고리 자기 이력만, 낙관적 상한).
    cross=True: 같은 유저의 다른 STRONG 카테고리 구매(now_s 이전분)도 이력에 얹어
    경쟁 부스트를 재현한다 — build_rows() 가 유일한 교체 지점이라는 원장 기록대로
    이 함수만 바꾸고 run_sweep()/main() 은 그대로 둔다."""
    import pandas as pd

    lambdas = LAMBDAS if lambdas is None else lambdas
    df = pd.read_parquet("/tmp/rees46/purchases_all.parquet")
    df = df[df["category_code"].isin(cats.keys())].copy()
    # ⛔ `.astype("int64") // 10**9` 는 pandas 버전에 따라 datetime64 단위(ns/us)가 달라
    # 1000배 틀린 값을 준다(pandas 3.0.3 에서 실측: us 단위라 결과가 1/1000 로 축소돼
    # 부스트 편차가 전부 0에 뭉개짐 — 2026-09-05 재실행에서 처음 발견). 단위에 무관하게
    # 실제 epoch-초를 구하는 방식으로 고정한다.
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    df["ts"] = (pd.to_datetime(df["event_time"]) - epoch).dt.total_seconds().astype("int64")
    rng = random.Random(20260905)
    rows = []

    if not cross:
        for (uid, cat), g in df.groupby(["user_id", "category_code"]):
            ts = sorted(g["ts"].tolist())
            if len(ts) < 2:
                continue
            c = cats[cat]
            train_ts, now_s = ts[:-1], ts[-1]
            history = [[c["l1"], c["l2"], c["leaf"], len(train_ts), train_ts[0], train_ts[-1]]]
            phrase = rng.choice(c["phrases"])
            rows.append({"phrase": phrase, "target_l2": c["l2"], "history": history,
                         "now_s": now_s, "lambda_p": lambdas})
        return rows

    # cross=True: 유저별로 카테고리→타임스탬프 리스트를 먼저 모은 뒤, 각 타겟 카테고리에
    # 대해 "그 순간(now_s) 이전"의 다른 카테고리 구매만 경쟁 이력으로 얹는다(미래 누수 방지).
    ts_by_user_cat = df.groupby(["user_id", "category_code"])["ts"].apply(
        lambda s: sorted(s.tolist())
    )
    by_user: dict = {}
    for (uid, cat), ts in ts_by_user_cat.items():
        by_user.setdefault(uid, {})[cat] = ts
    for uid, by_cat in by_user.items():
        for cat, ts in by_cat.items():
            if len(ts) < 2:
                continue
            c = cats[cat]
            train_ts, now_s = ts[:-1], ts[-1]
            history = [[c["l1"], c["l2"], c["leaf"], len(train_ts), train_ts[0], train_ts[-1]]]
            for cat2, ts2 in by_cat.items():
                if cat2 == cat:
                    continue
                prior = [t for t in ts2 if t < now_s]
                if not prior:
                    continue
                c2 = cats[cat2]
                history.append([c2["l1"], c2["l2"], c2["leaf"], len(prior), prior[0], prior[-1]])
            phrase = rng.choice(c["phrases"])
            rows.append({"phrase": phrase, "target_l2": c["l2"], "history": history,
                         "now_s": now_s, "lambda_p": lambdas})
    return rows


def run_sweep(rows: list[dict]) -> list[dict]:
    payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    proc = subprocess.run(
        ["cargo", "run", "--quiet", "--release", "-p", "oicr-sdk-core", "--example", "purchase_rank_sweep"],
        cwd=ROOT, input=payload, capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        print(proc.stderr[-2000:], file=sys.stderr)
        raise SystemExit("purchase_rank_sweep failed")
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cross", action="store_true",
                     help="타겟 카테고리 이력 + 다른 STRONG 카테고리 경쟁 이력도 포함")
    ap.add_argument("--lambdas", default=None,
                     help="쉼표 구분 λ_p 격자 (기본: %s)" % ",".join(str(x) for x in LAMBDAS))
    ap.add_argument("--out-tag", default="",
                     help="출력 파일명 접미 태그 — 기존 결과를 덮지 않으려면 지정 (예: fine)")
    args = ap.parse_args()
    lambdas = LAMBDAS if not args.lambdas else [float(x) for x in args.lambdas.split(",")]
    tag = ("-cross" if args.cross else "") + (f"-{args.out_tag}" if args.out_tag else "")

    cats = strong_categories()
    print(f"{len(cats)} strong (leaf-consistent) bridged categories")
    print(f"lambda grid: {lambdas}")
    rows = build_rows(cats, cross=args.cross, lambdas=lambdas)
    mode = "k>=2 same-category repeat purchases + cross-category competing history" if args.cross \
        else "k>=2 same-category repeat purchases (own-category history only)"
    print(f"{len(rows)} eval rows ({mode})")
    if not rows:
        print("no rows -- nothing to sweep")
        return 1
    results = run_sweep(rows)

    # 원본 per-row ranks 를 남긴다 — 부트스트랩 CI 는 이 파일을 재사용해 Rust 재실행 없이 돈다.
    raw_path = HERE / f"lambda_p_sweep-raw{tag}-2026-09-05.json"
    raw_path.write_text(json.dumps(results, ensure_ascii=False))
    print(f"-> {raw_path} (raw per-row ranks, for bootstrap)")

    out = {"n_categories": len(cats), "n_rows": len(results), "cross": args.cross,
           "grid": lambdas, "lambdas": {}}
    for lam in lambdas:
        key = f"{lam:.2f}"  # Rust 쪽 `format!("{lambda_p:.2}")` 와 정확히 맞춘다
        ranks = [r["ranks"].get(key) for r in results]
        ranks = [r for r in ranks if r is not None]
        n = len(ranks)
        top1 = sum(1 for r in ranks if r == 1) / n
        top5 = sum(1 for r in ranks if r <= 5) / n
        mrr = sum(1.0 / r for r in ranks) / n
        print(f"lambda_p={lam:<4} n={n:5} top1={top1:.4f} top5={top5:.4f} mrr={mrr:.4f}")
        out["lambdas"][key] = {"n": n, "top1": top1, "top5": top5, "mrr": mrr}

    out_path = HERE / f"lambda_p_sweep{tag}-2026-09-05.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
