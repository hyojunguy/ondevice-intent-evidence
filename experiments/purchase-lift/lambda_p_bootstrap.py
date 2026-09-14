#!/usr/bin/env python3
"""λ_p 스윕(`lambda_p_sweep.py`)의 raw per-row ranks 를 재실행 없이 재사용해, top-1 리프트의
부트스트랩 신뢰구간(paired, row-level resample)을 낸다.

2026-09-05(21) 원장이 남긴 미결 2개 중 (1)번 — "그 +2.0pp가 통계적으로 유의한가" 를
숫자로 답한다. (2)번(크로스카테고리 경쟁 부스트)은 `lambda_p_sweep.py --cross` 가 만든
raw-cross 파일을 같은 방식으로 함께 검정한다.

인자 없이 돌리면 위 두 세팅(coarse 격자)을 그대로 재현한다. `--raw`/`--pairs` 를 주면
임의의 raw 파일·임의의 쌍을 검정한다 — 2026-09-05(23) 의 fine 격자(0.5~1.5, 0.1 간격)가
이 경로로 돈다. ⛔ 쌍은 raw 파일에 실제로 있는 키(`{:.2}` 포맷, 예 `0.90`)여야 한다.
"""
from __future__ import annotations
import argparse, json, pathlib, sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
N_BOOT = 5000
SEED = 20260905
PAIRS = [("0.00", "0.20"), ("0.00", "0.50"), ("0.00", "1.00"), ("0.00", "2.00")]


def bootstrap_ci(is_top1: dict[str, np.ndarray], lam_lo: str, lam_hi: str, rng: np.random.Generator) -> dict:
    n = len(is_top1[lam_lo])
    lo, hi = is_top1[lam_lo], is_top1[lam_hi]
    point = hi.mean() - lo.mean()
    idx = rng.integers(0, n, size=(N_BOOT, n), dtype=np.int32)  # row-level resample, paired across lambdas
    diffs = hi[idx].mean(axis=1) - lo[idx].mean(axis=1)
    diffs.sort()
    lo_ci = diffs[int(0.025 * N_BOOT)]
    hi_ci = diffs[int(0.975 * N_BOOT) - 1]
    p_le_zero = float((diffs <= 0).mean())
    return {
        "lambda_lo": lam_lo, "lambda_hi": lam_hi, "n": n,
        "point_diff_pp": round(float(point) * 100, 3),
        "ci95_pp": [round(float(lo_ci) * 100, 3), round(float(hi_ci) * 100, 3)],
        "excludes_zero": bool(lo_ci > 0 or hi_ci < 0),
        "p_diff_le_zero": round(p_le_zero, 4),
    }


def run_setting(tag: str, path: pathlib.Path, pairs: list[tuple[str, str]] | None = None) -> dict:
    raw = json.loads(path.read_text())
    lambdas = sorted({k for r in raw for k in r["ranks"].keys()})
    is_top1 = {lam: np.array([r["ranks"][lam] == 1 for r in raw], dtype=np.float64) for lam in lambdas}
    rng = np.random.default_rng(SEED)
    pairs = PAIRS if pairs is None else pairs
    results = [bootstrap_ci(is_top1, lo, hi, rng) for lo, hi in pairs if lo in is_top1 and hi in is_top1]
    print(f"\n=== {tag} (n={len(raw)}, {N_BOOT} bootstrap resamples, paired row-level) ===")
    for r in results:
        sig = "significant (CI excludes 0)" if r["excludes_zero"] else "NOT significant"
        print(f"  λ={r['lambda_lo']}->{r['lambda_hi']}: dtop1={r['point_diff_pp']:+.2f}pp "
              f"95%CI[{r['ci95_pp'][0]:+.2f}, {r['ci95_pp'][1]:+.2f}]pp "
              f"P(d<=0)={r['p_diff_le_zero']:.4f}  -> {sig}")
    return {"setting": tag, "n": len(raw), "n_boot": N_BOOT, "pairs": results}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=None, help="raw per-row ranks JSON (lambda_p_sweep.py 산출)")
    ap.add_argument("--pairs", default=None,
                     help="쉼표 구분 'lo:hi' 쌍 (예: 0.00:1.00,0.90:1.00). 생략 시 기본 4쌍")
    ap.add_argument("--tag", default="", help="출력 파일명 접미 태그")
    args = ap.parse_args()

    if args.raw:
        pairs = None
        if args.pairs:
            pairs = [tuple(p.split(":")) for p in args.pairs.split(",")]
        path = pathlib.Path(args.raw)
        if not path.is_absolute():
            path = HERE / path
        out = [run_setting(path.name, path, pairs)]
        suffix = f"-{args.tag}" if args.tag else ""
        out_path = HERE / f"lambda_p_bootstrap{suffix}-2026-09-05.json"
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"\n-> {out_path}")
        return 0

    settings = [
        ("same-category-only (2026-09-05 원 세팅, 낙관적 상한)",
         HERE / "lambda_p_sweep-raw-2026-09-05.json"),
        ("cross-category-competing (경쟁 부스트 포함, 재측정)",
         HERE / "lambda_p_sweep-raw-cross-2026-09-05.json"),
    ]
    out = []
    for tag, path in settings:
        if not path.exists():
            print(f"skip {tag}: {path} not found (run lambda_p_sweep.py first)", file=sys.stderr)
            continue
        out.append(run_setting(tag, path))

    out_path = HERE / "lambda_p_bootstrap-2026-09-05.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
