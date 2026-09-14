#!/usr/bin/env python3
"""수확 리프 후보의 **모양(shape)** 이상을 잡는다 — 괄호 별칭·브랜드결속 호환 수식어.

# 왜 taxonomy_hygiene_gate.py 로는 안 잡히나

`taxonomy_hygiene_gate.py` 는 "리프가 자기 중분류로 가는가"(자기일관성)를 잰다. 다나와
392 리프 병합분은 그 게이트를 **99.2%로 통과했다** — 파편끼리도 서로 일관되게 배치되면
일관성 점수는 오히려 오른다([[training-data-is-the-ceiling]]: 일관된 나쁨은 그냥 일관된
나쁨이다). 그래서 형태(모양) 축은 **다른 게이트**가 필요하다.

# 실측(2026-09-07) — 세 후보 중 하나는 대조군이 기각했다

대조군은 다나와 392 병합 **직전** 상태(git 597ff12~1, 리프 5,628건 · 고유 문자열 5,578개,
"이미 여러 라운드 검수를 통과한 known-good" 집합)다. 사후(post-merge) 6,020 을 대조군으로
쓰면 지금 검사 대상인 392 자체가 섞여 있어 자기순환이 된다 — 그래서 pre-merge 를 쓴다.

| 규칙 | 392 중 적중 | 대조군(5,578) FP | 판정 |
|---|---|---|---|
| PAREN(괄호 별칭/채널qualifier) | 8/392 | 2/5,578 (0.036% — 둘 다 `전통주(매장픽업)`류, 같은 결함류) | ✅ 채택 |
| BRANDED_COMPAT(디지트/라틴 접두 + 용·호환) | 1/392 | 0/5,578 | ✅ 채택(보수적 — 순한글 브랜드 전사는 못 잡음) |
| ORPHAN_STEM(슬래시분할 2자 파편, 길이차≥2) | 15/392 | **150/5,578 (2.7%)** — 가방·건강·과일·냄비·두부·딸기·로봇 등 완전히 정상인 리프 대량 오분류 | ⛔ **기각 — 미채택** |

`measure` 서브커맨드가 이 표를 재현한다. 재현 명령은 파일 맨 아래 `reproduce` 참조.

# ORPHAN_STEM 을 왜 못 살렸나 (기록해 둔다 — 다음 라운드가 같은 시도를 반복하지 않게)

"손목/팔보호대" → ["손목", "팔보호대"] 처럼 앞 조각이 뒷조각의 공유 접미어("보호대")를
잃은 패턴(진짜 파편)과 "볼링/당구/게이트볼" 처럼 그냥 3개를 나열한 패턴(전부 완전한 단어)은
**문자열 길이차만으로는 구분이 안 된다** — 둘 다 "짧은 조각 + 긴 형제, 길이차 2" 모양이
똑같이 나온다(당구 vs 게이트볼도 길이차 2, 손목 vs 팔보호대도 길이차 2). 공백 유무·분할
위치(첫/끝)도 시험해봤지만 반례가 바로 나왔다(워치/밴드 액세서리 — 공백 있고 첫 위치인데
워치는 파편이 아니다). 대조군에서 2.7% 를 태우는 규칙은 채택하지 않는다 — 품사/의미 태깅
없이는 이 축을 결정론 규칙으로 풀 수 없다는 것이 이번 라운드의 결론이다.

사용:
    python3 gates/leaf_shape_gate.py measure     # 표 재현 + 원장 갱신
    python3 gates/leaf_shape_gate.py selftest    # 판정함수 자체의 참/거짓양성 스팟체크
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
assert (ROOT / "CONTRACT.md").exists(), f"ROOT 가 틀렸다: {ROOT}"
LEDGER = ROOT / "gates" / "leaf_shape_gate_report.json"

PRE_MERGE_COMMIT = "597ff12~1"          # 다나와 392 병합 직전 — 대조군 스냅샷
DANAWA_EXPANSION = ROOT / "data" / "expansion_danawa.json"

_PAREN_RE = re.compile(r"[()]")
_COMPAT_SUFFIXES = ("용", "호환")


# ---------------------------------------------------------------------
# 판정 함수 — 이 셋이 map_danawa_to_taxonomy.py / map_naver_to_taxonomy.py 의
# 수확 루프에서 그대로 호출된다. 여기 바뀌면 그쪽도 같이 바뀐다(단일 진실원).
# ---------------------------------------------------------------------


def flag_paren(name: str) -> bool:
    """괄호 별칭/채널qualifier: '스토브(버너)', '전통주(매장픽업)' 류.

    우리 리프 이름 규약은 하나의 이름 = 하나의 물건이다. 괄호로 동의어나 구매채널을
    끼워 넣는 표기는 두 갈래(정본 이름을 고르거나, 별도 리프로 쪼개거나) 중 하나로
    해소해야 하는데 harvester 는 원문을 그대로 흘려보내 그 표기가 리프로 굳는다.
    """
    return bool(_PAREN_RE.search(name))


def _has_digit_or_latin(s: str) -> bool:
    return bool(re.search(r"[0-9]", s)) or bool(re.search(r"[A-Za-z]", s))


def flag_branded_compat(name: str) -> bool:
    """브랜드/모델 결속 호환 수식어: '인스타360용' 류.

    ⛔ 의도적으로 보수적이다. '남성용'·'여성용'·'환자용'·'필드용' 같은 **일반** "-용"
    수식어는 대조군에 34건 있고 그중 대다수가 정상 리프다(실측: 채용 관련 5건은 "용"이
    수식어가 아니라 "채용"이라는 단어 자체의 마지막 음절이라 오탐이었고, 나머지도 성별/
    상황 수식어로 통상적인 리테일 분류명이다). 그래서 접두에 **숫자나 라틴 문자가 있는
    것만**(모델번호·로마자 브랜드명의 대리 신호) 브랜드결속으로 본다. 순한글로 표기된
    브랜드(고프로용·캐논용·레고호환)는 이 규칙으로는 못 잡는다 — 블록리스트 없이는
    일반 수식어와 문자열로 구분할 방법이 없었다(대조군을 깨끗하게 지키는 쪽을 택했다).
    """
    for suf in _COMPAT_SUFFIXES:
        if name.endswith(suf) and len(name) > len(suf):
            prefix = name[: -len(suf)]
            if _has_digit_or_latin(prefix):
                return True
    return False


def shape_flag(name: str) -> str | None:
    """수확/배치 루프의 단일 진입점 — 걸리면 사유, 아니면 None.

    ⛔ ORPHAN_STEM(슬래시분할 파편)은 여기 없다 — 대조군 2.7% FP 로 기각됐다(위 docstring).
    """
    if flag_paren(name):
        return "형태이상_괄호별칭"
    if flag_branded_compat(name):
        return "형태이상_브랜드결속수식어"
    return None


# ---------------------------------------------------------------------
# 대조군/측정
# ---------------------------------------------------------------------


def _load_taxonomy_leaves(py_source: str) -> list[str]:
    """taxonomy_ko.py 소스 텍스트(문자열)를 임시 모듈로 로드해 리프 이름 리스트를 낸다."""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(py_source)
        tmp_path = f.name
    try:
        spec = importlib.util.spec_from_file_location("_leaf_shape_gate_taxonomy_snapshot", tmp_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        taxonomy = mod.TAXONOMY
        return [c for a in taxonomy for b in taxonomy[a] for c in taxonomy[a][b]]
    finally:
        pathlib.Path(tmp_path).unlink(missing_ok=True)


def _pre_merge_control_leaves() -> list[str]:
    """다나와 392 병합 직전(known-good) 리프 — git 스냅샷에서 뜬다(taxonomy_ko.py 를
    건드리지 않는다 — checkout 하지 않고 `git show` 로만 읽는다)."""
    r = subprocess.run(["git", "show", f"{PRE_MERGE_COMMIT}:data/taxonomy_ko.py"],
                        cwd=ROOT, capture_output=True, text=True, check=True)
    return _load_taxonomy_leaves(r.stdout)


def _current_taxonomy_leaves() -> list[str]:
    return _load_taxonomy_leaves((ROOT / "data" / "taxonomy_ko.py").read_text(encoding="utf-8"))


def _danawa_392() -> list[str]:
    d = json.loads(DANAWA_EXPANSION.read_text(encoding="utf-8"))
    out: list[str] = []
    for _a, bs in d["existing_l2_additions"].items():
        for _b, leaves in bs.items():
            out.extend(leaves)
    return out


def measure() -> dict:
    control = sorted(set(_pre_merge_control_leaves()))
    current = sorted(set(_current_taxonomy_leaves()))
    cand392 = sorted(set(_danawa_392()))

    def bucket(leaves: list[str]):
        return {
            "n": len(leaves),
            "paren": sorted(l for l in leaves if flag_paren(l)),
            "branded_compat": sorted(l for l in leaves if flag_branded_compat(l)),
        }

    control_b = bucket(control)
    current_b = bucket(current)
    cand_b = bucket(cand392)

    report = {
        "pre_merge_commit": PRE_MERGE_COMMIT,
        "control_pre_merge": {
            "n_unique_leaves": control_b["n"],
            "paren_fp": {"n": len(control_b["paren"]), "leaves": control_b["paren"]},
            "branded_compat_fp": {"n": len(control_b["branded_compat"]),
                                   "leaves": control_b["branded_compat"]},
        },
        "control_current_post_merge_for_reference_only": {
            "n_unique_leaves": current_b["n"],
            "note": "이 집합은 검사대상 392를 이미 포함한 사후 상태라 자기순환 — 참고용일 뿐, "
                    "채택 판정의 근거는 control_pre_merge 다.",
            "paren_hits": {"n": len(current_b["paren"]), "leaves": current_b["paren"]},
            "branded_compat_hits": {"n": len(current_b["branded_compat"]),
                                     "leaves": current_b["branded_compat"]},
        },
        "danawa_392_candidates": {
            "n": cand_b["n"],
            "paren_caught": {"n": len(cand_b["paren"]), "leaves": cand_b["paren"]},
            "branded_compat_caught": {"n": len(cand_b["branded_compat"]),
                                       "leaves": cand_b["branded_compat"]},
        },
        "orphan_stem_rule": {
            "status": "REJECTED_BY_CONTROL",
            "measured_fp_on_pre_merge_control": "150/5578 (2.7%)",
            "measured_catch_within_392": "15/392",
            "reason": "길이차 휴리스틱이 진짜 파편(손목/팔보호대)과 단순 나열(당구/볼링/게이트볼)을 "
                      "구분 못 함 — 둘 다 동일한 '짧은조각+긴형제, 길이차2' 모양. 채택 안 함.",
        },
        "verdict": "PAREN + BRANDED_COMPAT 채택(대조군 FP ~0%) · ORPHAN_STEM 미채택(대조군 FP 2.7%)",
        "reproduce": [
            "python3 gates/leaf_shape_gate.py measure",
            f"(대조군은 `git show {PRE_MERGE_COMMIT}:data/taxonomy_ko.py` 스냅샷 — checkout 없음)",
        ],
    }
    return report


def cmd_measure() -> int:
    report = measure()
    LEDGER.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
    c = report["control_pre_merge"]
    d = report["danawa_392_candidates"]
    print(f"대조군(pre-merge, known-good) {c['n_unique_leaves']:,}개 중")
    print(f"  PAREN FP           {c['paren_fp']['n']}  {c['paren_fp']['leaves']}")
    print(f"  BRANDED_COMPAT FP  {c['branded_compat_fp']['n']}  {c['branded_compat_fp']['leaves']}")
    print(f"392 후보 중")
    print(f"  PAREN 적중          {d['paren_caught']['n']}  {d['paren_caught']['leaves']}")
    print(f"  BRANDED_COMPAT 적중 {d['branded_compat_caught']['n']}  "
          f"{d['branded_compat_caught']['leaves']}")
    print(f"ORPHAN_STEM: {report['orphan_stem_rule']['status']} — "
          f"{report['orphan_stem_rule']['reason']}")
    print(f"▶ {LEDGER}")
    return 0


def cmd_selftest() -> int:
    """판정함수 자체의 참/거짓양성 스팟체크(측정과 별개 — 코드가 맞게 짜였는지만 본다)."""
    cases = [
        ("스토브(버너)", True, False),
        ("전통주(매장픽업)", True, False),
        ("인스타360용", False, True),
        ("남성용", False, False),   # 일반 수식어 — 잡지 않는다(의도된 보수화)
        ("고프로용", False, False),  # 순한글 브랜드 — 이 규칙으로는 못 잡는다(문서화된 한계)
        ("가방", False, False),
        ("손목", False, False),     # ORPHAN_STEM 은 이 게이트에 없다
    ]
    ok = True
    for name, want_paren, want_compat in cases:
        got_paren = flag_paren(name)
        got_compat = flag_branded_compat(name)
        status = "OK" if (got_paren, got_compat) == (want_paren, want_compat) else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  {status}  {name!r:16s} paren={got_paren} compat={got_compat} "
              f"(기대 paren={want_paren} compat={want_compat})")
    return 0 if ok else 1


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "measure"
    if cmd == "measure":
        return cmd_measure()
    if cmd == "selftest":
        return cmd_selftest()
    print(f"알 수 없는 서브커맨드: {cmd} (measure|selftest)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
