#!/usr/bin/env python3
"""I1 — wasm 빌드와 네이티브 빌드가 같은 벡터를 내는가.

한 코어를 두 타깃으로 컴파일하므로 원칙적으로 같아야 하지만, 부동소수 연산 순서·
LLVM 최적화 차이로 갈릴 수 있다. 갈리면 기기마다 다른 광고가 매칭되고 FL 이 집계할
공통 파라미터가 사라진다 — 조용히 깨지는 종류의 실패라 코드가 지킨다.

⛔ 여기서 두 번째 구현(TS 재작성)을 비교 대상으로 삼지 않는다. 그건 발산을 막는 게
   아니라 만드는 것이다(CONTRACT.md — 코어는 하나).
"""
from __future__ import annotations
import json, sys
from _common import BUDGETS, ROOT, PASS, FAIL, UNMEASURED, run, report, main_guard

TOL = 1e-5


def load(lines: str, who: str):
    out = []
    for ln in lines.splitlines():
        ln = ln.strip()
        if ln.startswith("{"):
            out.append(json.loads(ln))
    if not out:
        raise ValueError(f"{who} 가 아무 벡터도 안 냈다")
    return out


def run_gate() -> int:
    native = run(["cargo", "run", "--quiet", "--release", "-p", "plataid-sdk-core",
                  "--example", "vectors"])
    if native.returncode != 0:
        return report("I1 parity", UNMEASURED,
                      ["네이티브 덤프 실패"] + (native.stderr or "").strip().splitlines()[-3:])

    js = ROOT / "gates" / "fixtures" / "dump_wasm_vectors.cjs"
    if not (ROOT / "gates" / "fixtures" / "wasm-node" / "plataid_sdk_wasm.js").exists():
        return report("I1 parity", UNMEASURED,
                      ["wasm(node) 바인딩이 없다",
                       "wasm-bindgen --target nodejs --out-dir gates/fixtures/wasm-node <wasm> 로 생성"])
    # ⛔ 낡은 node 바인딩은 "덤프 실패"라는 같은 증상으로 나타나므로 먼저 갈라 준다.
    #    바인딩이 wasm 바이너리보다 오래됐으면 그건 코어 버그가 아니라 빌드 누락이다.
    binary = ROOT / "target/wasm32-unknown-unknown/release/plataid_sdk_wasm.wasm"
    binding = ROOT / "gates/fixtures/wasm-node/plataid_sdk_wasm_bg.wasm"
    if binary.exists() and binding.exists() and binding.stat().st_mtime < binary.stat().st_mtime:
        return report("I1 parity", UNMEASURED,
                      ["node 바인딩이 wasm 바이너리보다 낡았다",
                       "wasm-bindgen --target nodejs --out-dir gates/fixtures/wasm-node "
                       "target/wasm32-unknown-unknown/release/plataid_sdk_wasm.wasm"])
    wasm = run(["node", str(js)])
    if wasm.returncode != 0:
        return report("I1 parity", UNMEASURED,
                      ["wasm 덤프 실패"] + (wasm.stderr or "").strip().splitlines()[-3:])

    a, b = load(native.stdout, "native"), load(wasm.stdout, "wasm")
    if len(a) != len(b):
        return report("I1 parity", FAIL, [f"행 수 불일치: native {len(a)} vs wasm {len(b)}"])

    worst, worst_at, dim = 0.0, None, None
    for ra, rb in zip(a, b):
        if ra["text"] != rb["text"]:
            return report("I1 parity", FAIL, [f"입력 정렬 어긋남: {ra['text']!r} vs {rb['text']!r}"])
        if len(ra["vec"]) != len(rb["vec"]):
            return report("I1 parity", FAIL, [f"차원 불일치 @ {ra['text']!r}"])
        dim = len(ra["vec"])
        for i, (x, y) in enumerate(zip(ra["vec"], rb["vec"])):
            d = abs(float(x) - float(y))
            if d > worst:
                worst, worst_at = d, (ra["text"], i)

    lines = [f"입력 {len(a)}행 × {dim}차원 = {len(a) * (dim or 0):,} 요소 비교",
             f"최대 절대차 {worst:.3e} (허용 {TOL:.0e})"]
    if worst_at:
        lines.append(f"최대차 위치: {worst_at[0][:36]!r} dim[{worst_at[1]}]")
    return report("I1 parity", PASS if worst <= TOL else FAIL, lines)


# ⛔ 최상위에서 부르면 이 모듈을 **import 할 수 없다** — import 하는 순간 게이트가
#    돌고 sys.exit 로 호출자를 죽인다. 게이트를 재사용하려던 코드가 여기서 잘렸다.
if __name__ == "__main__":
    main_guard(run_gate)
