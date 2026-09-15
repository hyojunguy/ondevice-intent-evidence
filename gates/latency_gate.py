#!/usr/bin/env python3
"""C2 — Tier 0 경로의 p95 벡터화 지연이 예산 안인가.

⛔ Tier 0 만 잰다. OS LLM(Tier 1)은 수백 ms 라 상시 경로가 아니고, 같이 재면 게이트가
   거짓말을 하게 된다(docs/spec/02-budgets.md).
⛔ 여기서 재는 기기는 CI 기준선(개발 맥)이다. 실기기 수치는 scripts/ios/run_all.sh 가
   artifacts/ios-suite.json 에 따로 남긴다. 저사양 모바일은 평가 대상이 아니고,
   그 전까지 "모바일에서 50ms"라고 인용하지 않는다.

# (dim × bits) 참고 매트릭스 — PASS/FAIL 과 분리 (2026-08-28)

`--matrix` 는 **아직 배포되지 않은** 조합(128d, 4bit 니블 패킹)이 예산 안에 드는지 미리
재는 참고 수치를 `artifacts/latency-matrix.json` 에 남긴다. `DIM` 은
`crates/sdk-core/src/lib.rs` 의 컴파일 상수라 한 바이너리가 여러 dim 을 못 잰다 — 그래서
`_dim_override()` 가 그 파일을 **임시로** 고쳐 dim 마다 다시 빌드하고(느리지만 정직하다),
끝나면 반드시 원문으로 되돌린다. bits(4/8)는 같은 빌드 안에서 런타임 인자로 스윕한다.

⛔ 이 매트릭스는 **PASS/FAIL 을 절대 좌우하지 않는다** — 아래 `run_gate()` 의 판정은 여전히
   `crates/sdk-core/src/lib.rs` 에 실제로 컴파일된(=배포될) dim 하나만 본다
   (`gates/code_size_gate.py` 의 `measuredBytes` vs `unmeasuredProxy` 분리와 같은 규율).
⛔ 매트릭스 셀의 가중치는 합성(synthetic)이다 — 크기/지연은 바이트 모양이 정하므로 실측으로
   인용 가능하지만, 이 산출물로 검색 품질을 말하지 마라(`scripts/build_artifacts.py` 와 동일 계약).
"""
from __future__ import annotations
import contextlib, datetime as dt, json, os, platform, re, sys
from _common import BUDGETS, ROOT, PASS, FAIL, UNMEASURED, run, report, main_guard

LIB_RS = ROOT / "crates" / "sdk-core" / "src" / "lib.rs"
DIM_RE = re.compile(r"(pub\s+const\s+DIM\s*:\s*usize\s*=\s*)(\d+)(\s*;)")

# ⛔ 사용자가 명시한 전환: 지금 배포된 64d·int8 옆에 곧 배포될 128d·4bit 을 나란히 잰다.
#    "곧 배포될" 조합이 늘면 여기만 넓힌다 — budgets.json 은 이 팀이 동시 편집 중이라
#    건드리지 않는다(⛔ 소유 파일 목록).
MATRIX_DIMS = (64, 128)
MATRIX_BITS = (8, 4)
MATRIX_CACHE = ROOT / "artifacts" / "latency-matrix.json"


def run_gate() -> int:
    cfg = BUDGETS["latency"]
    r = run(["cargo", "run", "--quiet", "--release", "-p", "oicr-sdk-core",
             "--example", "latency", "--",
             "--chars", str(cfg["input_chars"]),
             "--catalog", str(cfg["catalog_size"]),
             "--iters", str(cfg["iterations"])])
    if r.returncode != 0:
        tail = (r.stderr or "").strip().splitlines()
        return report("C2 latency", UNMEASURED,
                      ["sdk-core example `latency` 를 못 돌렸다"] + tail[-3:])
    try:
        m = json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return report("C2 latency", UNMEASURED,
                      ["example 이 마지막 줄에 JSON 을 안 찍었다", r.stdout[-200:]])

    budget, p95 = cfg["p95_ms"], m["p95_ms"]
    # ⛔ 2026-08-28: 여기 있던 `cfg['catalog_size']`(요청값 1000)는 **실측이 아니었다** —
    #    `catalog.bin` 은 요청 크기와 무관하게 통째로 로드되므로(`--catalog` 인자는
    #    example 이 그냥 되찍는 참고값), 실제로 top_k 가 훑는 건 카탈로그 파일의
    #    전체 항목 수(`m['catalog_size']`, 지금 3000)다. 설정값을 실측값인 척 인용하고
    #    있었다 — 둘이 다르면 둘 다 보여준다.
    catalog_actual = m.get("catalog_size")
    catalog_note = (f"{catalog_actual}"
                     if catalog_actual in (None, cfg["catalog_size"])
                     else f"{catalog_actual} (요청 {cfg['catalog_size']} — catalog.bin 은 전체 로드됨)")
    lines = [
        f"기기: {platform.platform()} ({platform.processor() or platform.machine()})",
        f"입력 {cfg['input_chars']}자 · 카탈로그 {catalog_note}"
        f" · 택소노미 리프 {m.get('taxonomy_leaves', '?')} · {m['iterations']}회",
        f"p50 {m['p50_ms']:.3f}ms · p95 {p95:.3f}ms · max {m.get('max_ms', float('nan')):.3f}ms",
        f"예산 {budget}ms · 사용률 {p95 / budget * 100:.1f}%",
        "⚠️ 개발 맥 기준선이다. 실기기는 artifacts/ios-suite.json (iPhone 12 Pro · 14 Pro Max).",
        "⚠️ 저사양 모바일은 평가하지 않았다 — 이 PASS 를 그 하드웨어 계층으로 외삽하지 마라.",
        "⚠️ 판정은 이 한 조합(현재 컴파일된 DIM · int8)에만 적용된다 — 아래 참고 매트릭스는",
        "   PASS/FAIL 과 무관하다.",
    ]
    lines += _matrix_reference_lines()
    return report("C2 latency", PASS if p95 <= budget else FAIL, lines)


def _matrix_reference_lines() -> list[str]:
    """캐시된 `--matrix` 결과를 참고용으로 붙인다. 없으면 만드는 법만 안내하고 끝낸다 —
    기본 실행에서 매트릭스를 재생성하지 않는다(느리고, 공유 소스 파일을 임시로 건드린다)."""
    if not MATRIX_CACHE.exists():
        return ["", "⬜ (dim × bits) 참고 매트릭스 없음 — `python3 latency_gate.py --matrix` 로 생성 가능"]
    try:
        doc = json.loads(MATRIX_CACHE.read_text(encoding="utf-8"))
    except (ValueError, OSError) as e:
        return ["", f"⬜ 참고 매트릭스 캐시를 못 읽었다 ({e}) — --matrix 로 재생성"]

    budget = doc.get("budget_p95_ms", BUDGETS["latency"]["p95_ms"])
    out = ["", f"── 참고 매트릭스 (미배포 조합 포함, PASS/FAIL 아님 · {doc.get('generated_at', '?')}) ──"]
    for c in doc.get("cells", []):
        if "error" in c:
            out.append(f"  ⬜ dim={c.get('dim', '?')} bits={c.get('bits', '?')}: {c['error'][:120]}")
            continue
        p95c = c["p95_ms"]
        out.append(
            f"  dim={c['dim']:>3} bits={c['bits']} synthetic  "
            f"p50 {c['p50_ms']:.3f}ms · p95 {p95c:.3f}ms · "
            f"예산 대비 {p95c / budget * 100:.1f}%  {'✅' if p95c <= budget else '❌'}"
        )
    if doc.get("load_warning"):
        out.append(f"  ⚠️ {doc['load_warning']}")
    return out


@contextlib.contextmanager
def _dim_override(new_dim: int):
    """⛔ `crates/sdk-core/src/lib.rs` 의 `DIM` 상수를 임시로 바꾼다. 이 파일은 다른
    세션도 읽고 편집할 수 있는 공유 소스다 — 컨텍스트를 벗어나면(예외가 나도) 항상
    진입 시점의 원문으로 정확히 되돌리고, 되돌아갔는지 바이트 단위로 검증한다."""
    original = LIB_RS.read_text(encoding="utf-8")
    m = DIM_RE.search(original)
    if not m:
        raise RuntimeError(f"{LIB_RS} 에서 `pub const DIM: usize = N;` 을 못 찾았다 — 손대지 않는다")
    patched = original[: m.start()] + f"{m.group(1)}{new_dim}{m.group(3)}" + original[m.end():]
    LIB_RS.write_text(patched, encoding="utf-8")
    try:
        yield
    finally:
        LIB_RS.write_text(original, encoding="utf-8")
        if LIB_RS.read_text(encoding="utf-8") != original:
            raise RuntimeError(
                f"⛔ {LIB_RS} 복구 검증 실패 — 파일이 원문과 다르다. "
                f"수동으로 `git checkout -- {LIB_RS}` 를 즉시 실행하라."
            )


def _load_warning() -> str | None:
    """다른 세션이 이 머신에서 무거운 작업을 돌리고 있으면 지연 수치가 오염된다
    (demo-env-preflight / world-model 규율과 동형 — 오염을 조용히 감추지 않는다)."""
    try:
        load1, _, _ = os.getloadavg()
    except (OSError, AttributeError):
        return None
    ncpu = os.cpu_count() or 1
    if load1 > 0.7 * ncpu:
        return (f"load1={load1:.2f} (ncpu={ncpu}) — 70% 문턱을 넘었다. "
                f"이 매트릭스 수치는 동시 부하로 오염됐을 수 있다, 인용 전 재측정 권장.")
    return None


def build_matrix(dims: tuple[int, ...] = MATRIX_DIMS,
                  bits_list: tuple[int, ...] = MATRIX_BITS) -> dict:
    """⛔ 이 함수 하나가 `lib.rs` 를 dim 개수만큼 다시 쓰고 빌드한다 — 수 분 단위다.
    `run_gate()` 의 기본 실행 경로에서는 절대 호출되지 않는다(opt-in `--matrix` 전용)."""
    load_warning = _load_warning()
    cells: list[dict] = []
    for dim in dims:
        with _dim_override(dim):
            b = run(["cargo", "build", "--quiet", "--release", "-p", "oicr-sdk-core",
                     "--example", "latency_matrix"])
            if b.returncode != 0:
                cells.append({"dim": dim, "error": "빌드 실패: " + (b.stderr or "")[-800:]})
                continue
            for bits in bits_list:
                r = run(["cargo", "run", "--quiet", "--release", "-p", "oicr-sdk-core",
                         "--example", "latency_matrix", "--", "--bits", str(bits)])
                if r.returncode != 0:
                    cells.append({"dim": dim, "bits": bits,
                                  "error": "실행 실패: " + (r.stderr or "")[-800:]})
                    continue
                try:
                    cells.append(json.loads(r.stdout.strip().splitlines()[-1]))
                except (ValueError, IndexError):
                    cells.append({"dim": dim, "bits": bits, "error": "JSON 파싱 실패",
                                  "stdout_tail": r.stdout[-300:]})

    doc = {
        "_comment": ("⛔ 참고 수치 — 미배포 조합 포함. PASS/FAIL 판정에 쓰지 마라, 그건 "
                     "`gates/latency_gate.py`(인자 없이 실행)가 배포 구성(현재 컴파일된 "
                     "DIM)에만 낸다. 가중치는 synthetic=true — 크기/지연은 바이트 모양이 "
                     "정하므로 실측이지만 검색 품질과는 무관하다."),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "budget_p95_ms": BUDGETS["latency"]["p95_ms"],
        "device": f"{platform.platform()} ({platform.processor() or platform.machine()})",
        "load_warning": load_warning,
        "cells": cells,
    }
    MATRIX_CACHE.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return doc


def main() -> int:
    if "--matrix" in sys.argv:
        try:
            build_matrix()
        except Exception as e:  # noqa: BLE001 — 매트릭스 실패가 배포 게이트(run_gate)를 막으면 안 된다
            print(f"⬜ UNMEASURED  C2 latency matrix 빌드 실패: {e}")
    return run_gate()


# ⛔ 최상위에서 부르면 이 모듈을 **import 할 수 없다** — import 하는 순간 게이트가
#    돌고 sys.exit 로 호출자를 죽인다. 게이트를 재사용하려던 코드가 여기서 잘렸다.
if __name__ == "__main__":
    main_guard(main)
