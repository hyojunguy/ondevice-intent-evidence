#!/usr/bin/env python3
"""C5 — 이기종·탈락 환경에서 FL 라운드가 수렴하는가.

판정은 fl-server 의 수렴 시뮬레이션 테스트가 진다. 이 게이트는 그 테스트를 실제로
돌리고 결과를 읽을 뿐이다 — 통과 여부를 여기서 다시 주장하지 않는다.

⛔ `running 0 tests` 를 단순 부분문자열로 보면 안 된다. lib/bin 타깃은 테스트가 없어
   항상 그 줄을 찍는다 — 그걸 미측정으로 읽으면 게이트가 자기 자신에게 속는다.
"""
from __future__ import annotations
import re, sys
from _common import PASS, FAIL, UNMEASURED, run, report, main_guard

FILTER = "converge"


def run_gate() -> int:
    r = run(["cargo", "test", "--release", "-p", "oicr-fl-server", "--", "--nocapture", FILTER])
    out = (r.stdout or "") + (r.stderr or "")
    if "no matching package" in out or "error: package ID specification" in out:
        return report("C5 fl-round", UNMEASURED, ["oicr-fl-server 크레이트가 아직 없다"])

    # 전 타깃 합계로 본다 — 한 타깃이 0이어도 다른 타깃이 돌았으면 측정된 것이다.
    ran = sum(int(n) for n in re.findall(r"^running (\d+) tests?$", out, re.M))
    if ran == 0:
        return report("C5 fl-round", UNMEASURED,
                      [f"필터 {FILTER!r} 에 걸리는 테스트가 0건 — 시뮬레이션 미구현"])

    names = [ln.strip() for ln in out.splitlines()
             if ln.strip().startswith("test ") and " ... ok" in ln]
    # 수렴 근거를 로그에서 그대로 인용한다(주장 대신 출력).
    evidence = [ln.strip() for ln in out.splitlines()
                if re.search(r"loss|round|particip|dropout", ln, re.I) and len(ln.strip()) < 160][:6]
    lines = [f"필터 {FILTER!r} · 실행 {ran}건 · 통과 {len(names)}건"] + names + evidence
    if r.returncode != 0:
        lines += out.strip().splitlines()[-6:]
    # 수렴 근거 줄에서 BCE 시작·끝 값을 뽑아 원장에 박는다 — 논문이 "0.69 → 0.20" 을
    # 인용하는데 그 값이 어느 파일에도 없었다(2026-09-14 감사).
    # ⛔ `evidence` 는 필터에 걸린 **아무 테스트**의 줄을 앞에서 6개 자른다 — 그래서
    #    C5 의 BCE 대신 랭킹헤드 AUC 가 담기고 있었다. BCE 는 따로 뽑는다.
    bce = [(int(m.group(1)), float(m.group(2)))
           for ln in out.splitlines()
           if (m := re.search(r"round\s+(\d+): held-out BCE = ([\d.]+)", ln))]
    losses = [float(x) for ln in evidence for x in re.findall(r"\d+\.\d+", ln)]
    return report("C5 fl-round", PASS if r.returncode == 0 else FAIL, lines,
                  facts={"tests_run": ran, "tests_passed": len(names),
                         "evidence_lines": evidence,
                         "heldout_bce_checkpoints": bce,
                         "heldout_bce_initial": bce[0][1] if bce else None,
                         "heldout_bce_final": bce[-1][1] if bce else None,
                         "loss_values_seen": losses[:12]})


# ⛔ 최상위에서 부르면 이 모듈을 **import 할 수 없다** — import 하는 순간 게이트가
#    돌고 sys.exit 로 호출자를 죽인다. 게이트를 재사용하려던 코드가 여기서 잘렸다.
if __name__ == "__main__":
    main_guard(run_gate)
