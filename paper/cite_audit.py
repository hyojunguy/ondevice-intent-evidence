#!/usr/bin/env python3
"""논문이 인용한 **특징적인 수치**가 저장소의 원장에 실제로 있는지 코드가 판정한다.

왜 필요한가 (2026-09-14, 한 세션에 같은 결함 4건):
  1. p95 2.404 ms      — 다섯 곳에서 인용했는데 어느 아티팩트에도 없었다(게이트가 매번 재측정).
  2. 2.404 / 2.746 ms  — 유저임베딩 표가 원장(1.421/1.763)이 아니라 위 미검증 값에 델타를 얹고 있었다.
  3. ESCI 0.2369       — 2,960 리프 시절 측정인데 6,020 리프 수치 옆에 그 사실 없이 놓여 있었다.
  4. +17.09 / +5.94pp  — 산문 README 에서 왔고, 그것도 2,960 리프 판이었다. 원장은 +18.23/+6.51.

넷 다 "사람이 눈으로 대조"로는 못 잡았다. 그래서 코드가 잡는다([[evaluator-must-act]]).

판정 대상은 **특징적인 수치만**이다 — 소수 3자리 이상이거나 천단위 구분이 있는 정수.
연도·절 번호·한 자리 백분율은 산문이라 제외한다. 오탐을 줄이는 쪽이 게이트를 살린다.
"""
from __future__ import annotations
import argparse, json, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEX = ROOT / "paper" / "main_en.tex"
# 원장이 사는 곳. 텍스트 산출물(README·md)은 **원장이 아니다** — 일부러 뺀다.
LEDGER_DIRS = ("experiments", "artifacts", "gates")
# 계산으로 유도되는 값들(백분율·차이)은 원장에 문자열로 없을 수 있다 → 허용 목록
ALLOW = re.compile(r"^(19|20)\d\d$")
# 분류는 셋이다. **미분류가 남으면 실패**한다 — 모르는 채로 통과시키지 않는 것이 요점이다.
#
#   DERIVED    원장 값의 합·차·비율. 유도식을 적는다(적지 않으면 검산이 불가능하다).
#   NO_LEDGER  약관상 못 담거나, 은퇴시키려고 인용하는 값.
#   PROSE_ONLY ⛔ **결함이다.** 산문 문서에만 있고 원장이 없다. 숨기지 않고 매번 띄운다 —
#              2026-09-14 에 이 계열로 네 건이 틀렸다(+17.09/+5.94 는 산문 README 의
#              2,960 리프 시절 값이었고 실측은 +18.23/+6.51 이었다).
DERIVED = {
    "1332138": "quant4_surface.json: taxonomy.i8 623,807 + catalog.i8 708,331",
    "906659":  "quant4_surface.json: taxonomy.q4 414,737 + catalog.q4 491,922",
    "425479":  "위 둘의 차 (1,332,138 − 906,659)",
    "206555":  "quant4_surface.json: L1_plus_L2.i8 3,352,283 − limit 3,145,728",
    "2352740": "ram_gate 의 L3 상주 바이트 — 언팩 후 크기, budgets.ram 에서 유도",
    "268903":  "code_size 여유 = 제품 약속 3,145,728 − (L1 2,876,825 + L2 49,979) 의 잔여 계산",
    "4096":    "federated 레이아웃: PROJ_DIM 64 × 64",
    "4198":    "4,096 + combiner_len 5 + mlp 89 + θu 8 (gates/budgets.json)",
    "4262":    "4,198 + purchase_stats·rank_head 등 (gates/budgets.json)",
}
NO_LEDGER = {
    "307551": "AI-Hub 102 코퍼스 행수 — 약관상 재가공 데이터를 커밋할 수 없다",
    "22900":  "AI-Hub 71603 코퍼스 행수 — 위와 같다",
    "3362619": "은퇴시킨 수치. 본문이 '더 이상 쓰지 않는다'고 말하려고 인용한다",
    "216891":  "위와 같다", "2404": "위와 같다 (지연 p95, 원장 없음이 확인되어 철회)",
    "69482": "위와 같다 (FedPer 축소 주장, 철회)", "74663": "위와 같다",
    "0.2316": "철회한 미견-리프 수치. 본문이 '이 셋을 철회한다'고 말하려고 인용한다 — "
              "대체 근거는 coverage_curve.json (미견 818리프에서 0.2037→0.2537 단조 상승)",
    "0.3341": "위와 같다. ⚠️ 인접 원장 둘이 0.3312 를 말한다(표본이 달라 반증은 아니다)",
    "1.0006": "철회한 이미지 노름. featureprint_stats.json 이 어떤 통계로도 재현되지 않음을 보인다",
    "4002327": "차원 결정 당시의 **사전 추정치**. 실측(6,115,038)이 이를 반증한 것이 본문 논지다",
    "4117221": "위와 같다", "2276098": "위와 같다",
    # 빌드 당시의 값. 그 시점 코퍼스 상태가 남아 있지 않아 재계수가 불가능하고, 재계수가
    # 주는 것은 **드리프트 도착점**(corpus_counts.json 에 있다). 드리프트 자체가 논지다.
    "26294": "빌드 당시 합성 발화 수. 현재 트리 재계수 26,322 은 corpus_counts.json 에 있다",
    "2933":  "위와 같다 (발화가 있는 리프 수, 재계수 2,935)",
    "10748": "위와 같다 (캡 없는 en 팔, 재계수 10,740)",
}
DERIVED_EXTRA = {
    "0.0108": "real-v2-ablation: ⑤ 0.023632 − ③ 0.012796 (w3d)",
    "0.0087": "위와 같다 (w7d): ⑤ 0.033578 − ③ 0.024882",
    "0.0009": "real-v2-temporal: ⑤ − ③ (w3d) = 0.000881",
    "0.0032": "위와 같다 (w7d) = 0.003186",
    # 2026-09-15 집 맥: 역사 기록이라 재계수 불가로 분류돼 있었으나, 원측정이 트리에 있다.
    #   재는 대상은 커버리지가 아니라 **하드블록 리프 수**였다.
    "2931":   "verbatim_blocked.json: taxonomy_leaves 2960 − hard_blocked.tally.all_verbatim 29",
}
# ⛔ 여기 남은 것은 **결함**이다. 원장이 이 머신에 없다 — 대부분 집 맥에서 측정됐다.
#    숨기지 않고 매 실행 출력한다. 원장을 받아오거나 재측정하면 위 분류로 올린다.
PROSE_ONLY = {
    "0.5429": "행동축 '표본만 줄임' 팔. evaluate.py 가 results.json 한 경로에만 쓰는데 그 파일은 "
              "n=5,250 판(교사 0.6789)으로 덮였다 — n=1,187 판은 소실. 스크립트에 표본 크기 인자가 "
              "없어 그때의 즉석 수정본도 없다. ⛔ 재실행은 회수가 아니라 새 측정이다"
              "([[experiment-ledger-first]] #2). 표에 † 로 표시하고 그 행이 지탱하던 가설 판정을 "
              "'supported' 에서 낮췄다",
}
EXEMPT = {**DERIVED, **DERIVED_EXTRA, **NO_LEDGER, **PROSE_ONLY}

# ⛔ 2026-09-14: 처음에는 **모든** JSON 을 건초더미에 넣었고, 그래서 이 도구가
#    거짓 통과를 냈다. 합성 행동 로그(behavior*.json, 29 MB)에는 `"t_ms": 69482` 같은
#    타임스탬프가 수만 개 있어서, 논문의 페이로드 69,482 바이트가 "원장에 있다"로 잡혔다.
#    실제로는 아무 원장에도 없었다(그 값은 64d 시절 값이고 현재 실측은 79,968 이다).
#    ⇒ **원장은 작다.** 대용량 표본 덤프는 원장이 아니므로 건초더미에서 뺀다.
LEDGER_MAX_BYTES = 400_000


def ledger_values() -> set[str]:
    """원장의 **스칼라 필드 값**만 모은다 — 원문 문자열 매칭이 아니다.

    ⛔ 2026-09-14, 이 도구가 거짓 통과를 두 번 냈다. 처음에는 모든 JSON 을 한 덩어리
       텍스트로 이어 붙여 `in` 으로 찾았는데, 합성 행동 로그의 `"t_ms": 69482` 와
       ESCI 제품 ID 목록이 아무 5~7자리 수나 공급했다. 그래서 **어느 원장에도 없는**
       페이로드 69,482 바이트가 "있다"로 통과했다(실측은 79,968).
    ⇒ 파싱해서 값을 모으고, **50개가 넘는 배열은 데이터로 보고 건너뛴다.** 원장 필드는
       스칼라이거나 짧은 배열(CI 2개, 곡선 7개)이다.
    """
    vals: set[str] = set()

    def walk(o) -> None:
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            if len(o) > 50:            # 표본 덤프·ID 목록 — 원장이 아니다
                return
            for v in o:
                walk(v)
        elif isinstance(o, bool):
            return
        elif isinstance(o, int):
            vals.add(str(o)); vals.add(str(abs(o)))
        elif isinstance(o, float):
            # ⛔ 부호를 떼고도 넣는다. 논문은 $-0.1572$ 로 쓰고 정규식은 `0.1572` 를
            #    잡으므로, 원장의 -0.15722… 를 절댓값으로도 색인하지 않으면 음수 계수가
            #    전부 "미뒷받침"으로 잡힌다(2026-09-14 실측: 그래서 8건이 오탐이었다).
            for x in (o, abs(o)):
                vals.add(repr(x))
                for r in range(1, 7):
                    vals.add(f"{x:.{r}f}")
        elif isinstance(o, str) and len(o) <= 400:
            # "3,352,283 B" 처럼 설명문 안에 적힌 값도 원장으로 인정한다
            for m in re.finditer(r"\d[\d,]*\.?\d*", o):
                vals.add(m.group(0).replace(",", ""))

    for d in LEDGER_DIRS:
        for f in (ROOT / d).rglob("*.json"):
            if any(s in f.parts for s in ("node_modules", "tokenlearn_features")):
                continue
            try:
                if f.stat().st_size > LEDGER_MAX_BYTES:
                    continue
                walk(json.loads(f.read_text(encoding="utf-8", errors="ignore")))
            except (OSError, json.JSONDecodeError):
                pass
    return vals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="미뒷받침이 있으면 exit 1")
    a = ap.parse_args()
    tex = TEX.read_text(encoding="utf-8")
    # 주석 제거 — 우리가 은퇴시킨 수치를 설명하는 문장은 본문이지만, % 주석은 아니다.
    tex = re.sub(r"(?<!\\)%.*", "", tex)
    # ⛔ 참고문헌은 제외한다 — arXiv 번호가 전부 "미뒷받침 수치"로 잡혀 게이트를 죽인다.
    tex = tex.split(r"\begin{thebibliography}")[0]
    hay = ledger_values()

    cand: dict[str, int] = {}
    # ⛔ 면제된 값은 cand 에 안 들어간다. 생존 판정을 cand 로 하면 **항상** "인용 안 함" 이 된다
    #    (2026-09-15 실측: 표에 버젓이 있는 0.5429 가 stale 로 잡혔다). 면제 **전** 집합을 따로 둔다.
    seen_all: set[str] = set()
    for m in re.finditer(r"\d{1,3}(?:\{,\}\d{3})+|\d+\.\d+|\d+", tex):
        raw = m.group(0)
        plain = raw.replace("{,}", "")
        # ⛔ 소수점을 지운 형태만 비교하면 "0.0109" 같은 키가 영영 안 맞는다.
        frac0 = plain.split(".")[1] if "." in plain else ""
        if len(frac0) >= 3 or ("{,}" in raw and len(plain) >= 4):
            seen_all.add(plain)
        if ALLOW.match(plain) or plain in EXEMPT or plain.replace(".", "") in EXEMPT:
            continue
        frac = plain.split(".")[1] if "." in plain else ""
        big = "{,}" in raw and len(plain) >= 4
        if len(frac) >= 3 or big:
            cand[plain] = cand.get(plain, 0) + 1

    missing = []
    for v, n in sorted(cand.items()):
        if v in hay:
            continue
        # 0.7504 ↔ 0.75040 / .7504 같은 표기 차이만 흡수한다(부분문자열 매칭 금지)
        alts = {v.rstrip("0").rstrip("."), v.lstrip("0"), "0" + v if v.startswith(".") else v}
        if any(x and x in hay for x in alts):
            continue
        missing.append((v, n))

    print(f"검사한 특징 수치 {len(cand)}종 · 원장 JSON {sum(1 for d in LEDGER_DIRS for _ in (ROOT/d).rglob('*.json'))}개")
    # ⛔ 본문이 더 이상 인용하지 않는 항목까지 계속 띄우면 목록이 낡는다 — 살아 있는 인용만 센다.
    live = {k: v for k, v in PROSE_ONLY.items() if k in seen_all}
    stale = sorted(set(PROSE_ONLY) - set(live))
    if live:
        print(f"\n⚠️ 산문 문서에만 근거가 있는 수치 {len(live)}종 — **원장 없음, 결함으로 추적한다**")
        for k, why in sorted(live.items()):
            print(f"   {k:>9}  {why}")
    if stale:
        print(f"\n🧹 더 이상 인용되지 않아 결함 목록에서 내릴 항목: {', '.join(stale)}")
    if not missing:
        print("\n✅ 미분류 수치 없음 — 나머지는 전부 원장에 있거나 사유가 적혀 있다")
        return 0
    print(f"\n⚠️ 원장에서 못 찾은 수치 {len(missing)}종 (인용 {sum(n for _, n in missing)}회)")
    print("   ⛔ 전부가 결함은 아니다 — 백분율·차이처럼 유도된 값은 여기 걸린다.")
    print("   확인할 것은 하나다: **이 값을 만든 측정이 파일로 남아 있는가.**\n")
    for v, n in missing:
        print(f"   {v:>14}  ×{n}")
    return 1 if a.strict else 0


if __name__ == "__main__":
    sys.exit(main())
