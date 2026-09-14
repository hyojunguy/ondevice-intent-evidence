#!/usr/bin/env python3
"""C1 — 배포 아티팩트의 전송 바이트가 5MiB 이하인가.

⛔ "모델은 별도 다운로드"로 예산을 빼돌리지 못하게, 첫 실행에 실제로 내려오는 것을 전부 센다.
"""
from __future__ import annotations
import sys
from _common import BUDGETS, ROOT, PASS, FAIL, UNMEASURED, gzip_bytes, walk_files, report, main_guard


def run_gate() -> int:
    cfg = BUDGETS["size"]
    limit = cfg["limit_bytes"]
    rows, total, missing = [], 0, []

    for rel in cfg["artifacts"]:
        target = ROOT / rel
        if not target.exists():
            missing.append(rel)
            continue
        sub = sum(gzip_bytes(f) for f in walk_files(target))
        total += sub
        rows.append(f"{sub:>10,}  {rel}")

    if missing:
        return report("C1 size", UNMEASURED,
                      rows + [f"없는 아티팩트: {', '.join(missing)}",
                              "→ scripts/build_artifacts.py 와 web 빌드를 먼저 돌려라"])

    pct = total / limit * 100
    rows += [f"{'─' * 10}",
             f"{total:>10,}  합계 (gzip)",
             f"{limit:>10,}  상한",
             f"{'':>10}  사용률 {pct:.1f}% · 여유 {limit - total:,} 바이트"]
    return report("C1 size", PASS if total <= limit else FAIL, rows)


# ⛔ 최상위에서 부르면 이 모듈을 **import 할 수 없다** — import 하는 순간 게이트가
#    돌고 sys.exit 로 호출자를 죽인다. 게이트를 재사용하려던 코드가 여기서 잘렸다.
if __name__ == "__main__":
    main_guard(run_gate)
