#!/usr/bin/env python3
"""모든 게이트를 돌리고 하나라도 실패하면 exit 1.

⛔ UNMEASURED 는 통과가 아니다. 배포 게이트로 쓸 때는 --strict 로 UNMEASURED 도 막는다.
"""
from __future__ import annotations
import subprocess, sys, pathlib

HERE = pathlib.Path(__file__).resolve().parent
GATES = ["size_gate.py", "latency_gate.py", "egress_gate.py", "parity_gate.py",
         "fl_round_gate.py", "dp_gate.py", "attribution_gate.py",
         "catalog_gate.py", "code_size_gate.py", "ram_gate.py",
         "behavior_gate.py", "tier1_optional_gate.py", "taxonomy_hygiene_gate.py"]


def main() -> int:
    strict = "--strict" in sys.argv
    results = {}
    for g in GATES:
        p = HERE / g
        if not p.exists():
            print(f"⬜ UNMEASURED  {g} (게이트 미구현)")
            results[g] = 3
            continue
        results[g] = subprocess.run([sys.executable, str(p)], cwd=HERE).returncode

    print("\n── 요약 ──")
    for g, rc in results.items():
        print(f"  {'PASS' if rc==0 else 'FAIL' if rc==1 else 'UNMEASURED'}\t{g}")
    failed = [g for g, rc in results.items() if rc == 1]
    unmeasured = [g for g, rc in results.items() if rc not in (0, 1)]
    if failed:
        print(f"\n❌ 실패 {len(failed)}건 — 배포 금지: {failed}")
        return 1
    if unmeasured:
        msg = "배포 금지" if strict else "인용 금지"
        print(f"\n⬜ 미측정 {len(unmeasured)}건 — {msg}: {unmeasured}")
        return 1 if strict else 0
    print("\n✅ 전부 통과 — 이 수치는 인용할 수 있다")
    return 0


sys.exit(main())
