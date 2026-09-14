#!/usr/bin/env python3
"""서버 CPU 측정을 stdout 에서 건져 원장으로 만든다.

⛔ 파드 안에서 JSON 을 못 쓴 이유는 `rust:slim` 에 python3 가 없어서다. 측정 자체는
   성공했고 stdout 에 7줄이 다 있다 — 로그를 그 자리에서 건지는 규율이 없었으면
   측정을 통째로 날릴 뻔했다([[gpu-job-cleanup]]).
⛔ 여기서 수치를 다시 계산하지 않는다. 바이너리가 찍은 줄을 파싱해 중앙값만 낸다.
"""
from __future__ import annotations
import argparse, datetime as dt, json, pathlib, re, statistics, sys

HERE = pathlib.Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", nargs="+", required=True, help="gpu_experiment 결과 JSON 들")
    a = ap.parse_args()
    rows = []
    for f in a.out:
        d = json.loads(pathlib.Path(f).read_text(encoding="utf-8"))
        tail = d.get("stdout_tail") or ""
        runs = [json.loads(l) for l in tail.splitlines() if l.startswith('{"p50_ms"')]
        cpu = (re.search(r"model name\s*:\s*(.+)", tail) or [None, "unknown"])[1].strip()
        cores = (re.search(r"model name.*\n(\d+)", tail) or [None, "?"])[1]
        tag = pathlib.Path(f).stem.replace("out_cpu_", "")
        if not runs:
            rows.append({"env": tag, "status": "unmeasured",
                         "reason": (d.get("reason") or "")[:140]})
            continue
        rows.append({"env": tag, "status": "measured", "node": d.get("node"),
                     "cpu": cpu, "cores": cores, "runs": len(runs),
                     "p50_ms": round(statistics.median(r["p50_ms"] for r in runs), 4),
                     "p95_ms": round(statistics.median(r["p95_ms"] for r in runs), 4),
                     "p95_min": round(min(r["p95_ms"] for r in runs), 4),
                     "p95_max": round(max(r["p95_ms"] for r in runs), 4),
                     "catalog_size": runs[0]["catalog_size"],
                     "taxonomy_leaves": runs[0]["taxonomy_leaves"]})
    out = {"_measured_at": dt.datetime.now().isoformat(timespec="seconds"),
           "_what": "배포 바이너리 Tier-0 지연을 서버 CPU 에서. 맥과 같은 코드·같은 프로토콜.",
           "_protocol": "512자 · 200회 · 워밍업 20 · release 빌드 · latency_probe::run",
           "_caveat": "공유 클러스터 노드다. 다른 테넌트 부하가 섞일 수 있어 하한이 아니라 관측값이다.",
           "budget_p95_ms": 20.0, "environments": rows}
    (HERE / "cpu_latency.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
