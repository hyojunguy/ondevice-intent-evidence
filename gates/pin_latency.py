#!/usr/bin/env python3
"""온디바이스 지연을 **원장에 못박는다** — 논문이 인용할 수 있는 하나의 값.

⛔ 2026-09-15 발견: 논문이 p95 2.404 ms 를 다섯 곳에서 인용하는데 그 값이 어느 아티팩트에도
   없었다. 게이트가 매 실행 다시 재기 때문이고, 실측 분포는 2.37~2.61 ms 로 약 10% 흔들린다.
   12.0% 사용률을 주장하면서 분모가 아니라 분자가 그만큼 흔들리면 인용이 성립하지 않는다.
⇒ 여기서 N 회 돌려 **중앙값과 폭**을 함께 남기고, 논문·비교표는 이 파일만 읽는다.
"""
from __future__ import annotations
import argparse, datetime as dt, json, os, pathlib, statistics, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=7)
    ap.add_argument("--max-load", type=float, default=0.35,
                    help="load1/코어수 상한 — 넘으면 측정을 거부한다")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = json.loads((ROOT / "gates" / "budgets.json").read_text(encoding="utf-8"))["latency"]

    # ⛔ 조용한 머신에서만 잰다. 2026-09-15 실측: 다른 세션 잡이 CPU 306% 를 먹는 동안
    #    p95 가 2.4 → 3.0 ms 로 25% 올랐다. 부하를 안 보고 잰 값은 기기 성능이 아니라
    #    그날의 경합을 재는 것이다([[power-measurement-baseline]] 의 베이스라인 규율과 동형).
    load1 = os.getloadavg()[0]
    ncpu = os.cpu_count() or 1
    if load1 / ncpu > a.max_load and not a.force:
        sys.exit(f"⛔ 머신이 바쁘다: load1 {load1:.2f} / {ncpu} cores = "
                 f"{load1/ncpu:.2f} > {a.max_load}. 조용해지면 다시 돌려라 (--force 로 무시).")
    runs = []
    for i in range(a.runs):
        r = subprocess.run(["cargo", "run", "--quiet", "--release", "-p", "oicr-sdk-core",
                            "--example", "latency", "--",
                            "--chars", str(cfg["input_chars"]),
                            "--catalog", str(cfg["catalog_size"]),
                            "--iters", str(cfg["iterations"])],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"측정 실패:\n{r.stderr[-500:]}")
        runs.append(json.loads(r.stdout.strip().splitlines()[-1]))
        print(f"  run {i+1}/{a.runs}  p95 {runs[-1]['p95_ms']:.4f}", flush=True)

    def agg(k):
        xs = [x[k] for x in runs]
        return {"median": round(statistics.median(xs), 4),
                "min": round(min(xs), 4), "max": round(max(xs), 4)}

    out = {"_measured_at": dt.datetime.now().isoformat(timespec="seconds"),
           "_what": "배포 바이너리 Tier-0 지연. 논문이 인용하는 단일 정본.",
           "_why_pinned": "게이트는 매 실행 재측정하므로 값이 흔들린다. 인용은 이 중앙값으로 한다.",
           "device": "development Mac (Apple silicon)",
           "load1_per_core_at_measure": round(load1 / ncpu, 3),
           # ⛔ 2026-09-14: 예전에는 여기에 cfg["catalog_size"](=1000)를 그대로 넣었다.
           #    그런데 `--catalog` 는 **무력한 인자**다 — catalog.bin 은 요청 크기와 무관하게
           #    통째로 로드된다(실측 6,020). 그래서 원장이 "1000개로 쟀다"고 말하면서
           #    같은 파일 아래쪽에 6,020을 적는 자기모순이 나왔다. 프로토콜에는 **실측만** 적는다.
           "protocol": {"input_chars": cfg["input_chars"],
                        "iterations": cfg["iterations"],
                        "catalog_entries": runs[0]["catalog_size"],
                        "_catalog_arg_is_inert": (
                            f"--catalog {cfg['catalog_size']} 는 전달되지만 무시된다 — "
                            f"catalog.bin 전체({runs[0]['catalog_size']}개)가 로드된다"),
                        },
           "runs": a.runs, "budget_p95_ms": cfg["p95_ms"],
           "p50_ms": agg("p50_ms"), "p95_ms": agg("p95_ms"), "p99_ms": agg("p99_ms"),
           "taxonomy_leaves": runs[0]["taxonomy_leaves"]}
    out["used_pct_of_budget"] = round(out["p95_ms"]["median"] / cfg["p95_ms"] * 100, 1)
    (ROOT / "artifacts" / "latency.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                     encoding="utf-8")
    print(f"\np95 중앙값 {out['p95_ms']['median']} ms "
          f"(폭 {out['p95_ms']['min']}~{out['p95_ms']['max']}) · 예산의 {out['used_pct_of_budget']}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
