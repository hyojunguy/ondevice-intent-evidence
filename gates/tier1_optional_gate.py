#!/usr/bin/env python3
"""C13 — Tier 1(기기 OS LLM) **없이도 제품이 완전히 선다.**

# 왜 이 게이트가 생겼나 (2026-08-30)

사용자 지시: *"폰 안의 로컬 llm을 사용할 수 있을때와 없을때 모두 동작 할 수 있는
모델이어야 해."*

이건 새 요구가 아니다. `docs/spec/01-architecture.md:44` 가 처음부터 적고 있었다 —
"⛔ Tier 1 을 전제로 설계하지 마라. 없는 기기가 다수다." **그런데 게이트 11개 중 이걸
검사하는 것은 0개였다.** 이 레포에서 산문으로만 적힌 규칙은 반복해서 건너뛰어졌다
(오늘만도 "라운드 스냅샷 코드는 이미 있다"가 없었고, 화면의 크기 배율이 220 으로 굳어
있었다 — 실제 197).

# 이 축이 조용히 깨지는 방식

학습 데이터에 Tier 1 출력이 **항상** 들어 있으면 모델은 그 칸을 믿게 학습된다.
Tier 1 없는 기기에서는 그 칸이 0 으로 들어간다. 그러면:

  - 에러가 **안 난다** — 숫자가 0 일 뿐이다
  - 순위는 **나온다** — 다만 아무도 검증한 적 없는 순위다
  - 게이트는 **통과한다** — Tier 1 있는 세계에서만 재니까

이 레포가 반복해 당한 모양 그대로다: **틀렸는데 시끄럽지 않은 실패.**

# 무엇을 판정하나

1. **채널 선언 트립와이어.** `plataid_sdk_core::TIER1_RANKING_CHANNEL` 을 소스에서 읽는다.
   `false` 면 랭킹 입력에 Tier 1 이 없다는 뜻이라 "없이도 도나"는 자명하게 참이다 —
   구조 검사(2·3)만 하고 통과한다. `true` 인데 두 팔 측정 결과가 없으면 **FAIL** 이다.
   ⛔ UNMEASURED 가 아니라 FAIL 인 이유: 채널이 있는데 검증이 없는 상태는 채널이 없는
   상태보다 **나쁘다**. 못 잰 것이 아니라 안 잰 것이다.

2. **카탈로그는 Tier 0 임베더로만 만든다.** 광고 카탈로그 벡터를 Tier 1 으로 만들면
   Tier 1 없는 기기에서는 **비교 대상 자체가 없어** 매칭이 아예 불가능해진다. 설계가
   미리 막아둔 자리이고(`01-architecture.md`), 이 게이트가 그게 유지되는지 본다.
   판정: `artifacts/MANIFEST.json::_provenance.source_model` 이 배포 모델과 같은가.

3. **투영 P 없이도 배포 경로가 돈다.** 스펙은 P 를 "Tier 1 있을 때만 로드 — 지연 로드"로
   둔다(`02-budgets.md:13`). 지금 `projection.bin` 은 **미학습 항등**이고 나머지 게이트가
   전부 통과한다는 사실이 곧 이 성질의 실증이다. 이 게이트는 그 상태가 유지되는지 —
   즉 배포 경로가 학습된 P 를 **요구하지 않는지** 확인한다.

# 두 팔 측정 계약 (채널이 생기면)

`experiments/user-embedding/tier1_arms.json` 에 아래 모양으로 나와야 한다.
판정 표면은 **중분류 top-5**(제품 표면)다 — 유리한 입도가 기본값이 되면 아무도 안 본다
(2026-08-29 사고).

    {
      "surface": "l2_top5",
      "n": <홀드아웃 표본>,
      "off": {"acc": ..., "ci": [lo, hi]},     # Tier 1 없음 — 기준선
      "on":  {"acc": ..., "ci": [lo, hi]},     # Tier 1 있음
      "off_vs_baseline": {"delta_pp": ..., "ci": [lo, hi]},  # 이전 배포본 대비
      "on_minus_off":    {"delta_pp": ..., "ci": [lo, hi]},
      "off_vs_baseline_leaf": {"delta_pp": ..., "ci": [lo, hi]},   # ⛔ 필수
      "on_minus_off_leaf":    {"delta_pp": ..., "ci": [lo, hi]}    # ⛔ 필수
    }

  - `off_vs_baseline` 의 CI 상단이 0 미만 → **BLOCK**. Tier 1 없는 기기가 나빠졌다.
  - `on_minus_off` 가 유의하지 않음 → 경고. Tier 1 이 값을 못 하고 있다(차단은 아니다 —
    Tier 1 이 무의미한 것은 프라이버시·비용상 오히려 안전한 실패다).

# ⛔ 리프 축도 본다 (2026-08-31 — 이 게이트가 스스로 눈이 멀어 있었다)

첫 판은 **중분류 top-5 하나만** 봤다. 그 상태로 실제 측정을 돌리니 이렇게 나왔다:

    on − off   표면 +0.17pp CI[-0.13,+0.47]  무의미
               리프 **-1.83pp** CI[-2.45,-1.20]  **유의 악화**

즉 **게이트는 통과하는데 채널을 켜면 손해**였다. 메커니즘도 이미 알려진 것이다 — 범주형
지각은 L1/L2 **블록 단위**라 블록 전체를 같은 값만큼 올린다. 블록은 맞히고(표면 +) 블록
안 리프 순서를 흩뜨린다(리프 −). 2026-08-29 에 행동 사전확률에서 같은 것을 관측했고,
그때 배운 것이 "유리한 입도가 기본값이 되면 아무도 안 본다"였는데 이 게이트가 그 함정을
그대로 재현했다.

⇒ 두 입도를 **둘 다** 요구하고, 어느 쪽이든 유의 악화면 차단한다. 표면만 좋아지는 채널은
열지 않는다.
"""
from __future__ import annotations

import json
import pathlib
import re

from _common import BUDGETS, FAIL, PASS, UNMEASURED, main_guard, report

ROOT = pathlib.Path(__file__).resolve().parents[1]
NAME = "C13 tier1-optional"

LIB_RS = ROOT / "crates" / "sdk-core" / "src" / "lib.rs"
ARMS = ROOT / "experiments" / "user-embedding" / "tier1_arms.json"
MANIFEST = ROOT / "artifacts" / "MANIFEST.json"
DEPLOYED_RUN = ROOT / "experiments" / "distill-ko" / "deployed_run.json"
PROJECTION = ROOT / "artifacts" / "projection.bin"
# 3층(OS LLM) 가용성 프로브. 관측 전용이어야 한다 — §3.5 참조.
PROBE = ROOT / "web" / "demo" / "src" / "features" / "content" / "osModel.ts"
# 제품 랭킹을 실제로 만드는 파일 — §3.6 이 여기에 지각이 안 들어왔는지 본다.
WASM_LIB = ROOT / "crates" / "sdk-wasm" / "src" / "lib.rs"

_CHANNEL_RE = re.compile(
    r"pub\s+const\s+TIER1_RANKING_CHANNEL\s*:\s*bool\s*=\s*(true|false)\s*;"
)


def channel_declared() -> bool | None:
    """소스에서 선언을 읽는다. 못 읽으면 None — 게이트가 추측하지 않는다."""
    if not LIB_RS.exists():
        return None
    m = _CHANNEL_RE.search(LIB_RS.read_text(encoding="utf-8"))
    return None if m is None else (m.group(1) == "true")


def sig(ci: list[float]) -> bool:
    return ci[0] > 0 or ci[1] < 0


def run_gate() -> int:
    lines: list[str] = []

    # ── 1. 채널 선언 ────────────────────────────────────────────────
    ch = channel_declared()
    if ch is None:
        return report(
            NAME,
            UNMEASURED,
            [
                f"{LIB_RS.relative_to(ROOT)} 에서 TIER1_RANKING_CHANNEL 선언을 못 읽었다.",
                "⛔ 이 게이트는 선언을 읽어 판정한다 — 상수가 사라졌으면 트립와이어가 없는 것이다.",
            ],
        )

    # ── 2. 카탈로그가 Tier 0 임베더에서 나왔나 ──────────────────────
    if not MANIFEST.exists() or not DEPLOYED_RUN.exists():
        return report(NAME, UNMEASURED, ["MANIFEST.json 또는 deployed_run.json 없음 — 빌드 먼저"])
    prov = json.loads(MANIFEST.read_text(encoding="utf-8")).get("_provenance", {})
    deployed = json.loads(DEPLOYED_RUN.read_text(encoding="utf-8")).get("deployed", "")
    src = str(prov.get("source_model", ""))
    if not deployed or not src.endswith(deployed):
        return report(
            NAME,
            FAIL,
            [
                "⛔ 카탈로그·택소노미가 배포 Tier 0 임베더에서 나오지 않았다.",
                f"   MANIFEST source_model = {src or '(없음)'}",
                f"   deployed_run deployed  = {deployed or '(없음)'}",
                "   Tier 1 로 만든 카탈로그는 Tier 1 없는 기기에서 비교 대상 자체가 사라진다.",
            ],
        )
    lines.append(f"카탈로그·택소노미 임베더 = 배포 Tier 0 ({deployed})")

    # ── 3. 학습된 투영 P 를 요구하지 않나 ───────────────────────────
    proj_note = str(prov.get("projection", ""))
    if PROJECTION.exists() and proj_note.startswith("IDENTITY"):
        lines.append(
            "투영 P = 미학습 항등 — 배포 경로가 학습된 P 를 요구하지 않는다는 실증"
            " (나머지 게이트가 이 상태로 전부 통과한다)"
        )
    elif PROJECTION.exists():
        lines.append(
            "투영 P = 학습본 — ⚠️ Tier 0 단독 경로가 여전히 도는지는 두 팔 측정이 답한다"
        )
    else:
        return report(NAME, UNMEASURED, ["artifacts/projection.bin 없음 — 빌드 먼저"])

    # ── 3.5 런타임 프로브가 랭킹에 닿을 경로가 없나 (2026-08-31) ────
    #
    # 3층(OS LLM) 가용성 프로브를 붙였다. 그 출력이 `content.fold()` 에 닿는 순간
    # 유저항을 통해 순위에 들어가고, 위에서 읽은 `TIER1_RANKING_CHANNEL = false`
    # 선언이 **거짓이 된다** — 그러면 이 게이트의 1번 검사가 통과하면서 실제로는
    # 검증 안 된 채널이 사는 상태가 된다. 산문으로 부탁하면 건너뛰어지므로(캡슐 §5)
    # 파일에 물어서 확인한다.
    #
    # ⛔ 검출은 좁게 한다 — import 목록과 코드 줄만 본다. 주석에 "fold" 가 있다고
    #    막으면 오탐이고, 오탐 한 번이 사람을 게이트에서 떼어놓는다
    #    ([[substring-detector-allowlist]]).
    if PROBE.exists():
        src_txt = PROBE.read_text(encoding="utf-8")
        imports = re.findall(r"^\s*import\s[^;]*?from\s+[\"']([^\"']+)[\"']", src_txt, re.M)
        code = "\n".join(
            ln for ln in src_txt.splitlines() if not re.match(r"\s*(//|\*|/\*)", ln)
        )
        leaks = [i for i in imports if "sdk" in i]
        calls = [c for c in ("fold(", "classify", "combiner", "events_push") if c in code]
        if leaks or calls:
            return report(
                NAME,
                FAIL,
                lines
                + [
                    "",
                    f"⛔ {PROBE.relative_to(ROOT)} 가 랭킹 경로에 닿는다.",
                    f"   import: {leaks or '(없음)'} · 호출: {calls or '(없음)'}",
                    "   OS 모델 출력이 콘텐츠 스케치로 들어가면 유저항을 통해 순위에",
                    "   반영되고, TIER1_RANKING_CHANNEL = false 선언이 거짓이 된다.",
                    "   채널을 정말 열려면 상수를 true 로 바꾸고 두 팔 측정을 같이 낸다.",
                ],
            )
        lines.append(f"3층 프로브 = 관측 전용 ({PROBE.name}: SDK import 0, 랭킹 호출 0)")
    else:
        lines.append("3층 프로브 없음 — 검사할 런타임 경로가 없다")

    # ── 3.6 상수가 false 인 동안 랭킹 경로가 지각을 안 먹나 (2026-08-31) ──
    #
    # `sdk-core` 는 지각 채널을 **읽을 수 있게** 되었다(`Percept`/`PerceptLog`,
    # `user_embedding.bin` v2). 선언이 `false` 라는 말은 "그걸 제품 랭킹에 안 먹인다"이지
    # "코드에 없다"가 아니다. 그래서 선언이 참인지는 **제품 표면**(wasm)에 물어야 한다 —
    # §3.5 가 프로브 파일 하나만 보는 것과 같은 이유로, 여기서는 랭킹을 실제로 만드는
    # 파일을 본다.
    if not ch and WASM_LIB.exists():
        wasm = WASM_LIB.read_text(encoding="utf-8")
        code = "\n".join(
            ln for ln in wasm.splitlines() if not re.match(r"\s*(//|\*|/\*)", ln)
        )
        used = [t for t in ("PerceptLog", "Percept::", "percepts_push") if t in code]
        if used:
            return report(
                NAME,
                FAIL,
                lines
                + [
                    "",
                    f"⛔ {WASM_LIB.relative_to(ROOT)} 가 지각을 다루는데 선언은 false 다.",
                    f"   발견: {used}",
                    "   선언이 거짓이면 이 게이트의 1번 검사가 통과하면서 실제로는",
                    "   검증 안 된 채널이 사는 상태가 된다 — 채널이 없는 것보다 나쁘다.",
                ],
            )
        lines.append(f"제품 랭킹 경로 = 지각 미사용 ({WASM_LIB.name}: 지각 심볼 0)")

    # ── 4. 채널이 없으면 여기까지 ───────────────────────────────────
    if not ch:
        lines += [
            "",
            "랭킹 입력에 Tier 1 채널 없음 (TIER1_RANKING_CHANNEL = false)",
            "  → 'Tier 1 없이 도나'는 자명하게 참이다. 뺄 것이 없다.",
            "  ⛔ 이 상수를 true 로 바꾸는 커밋은 두 팔 측정을 같이 만들어야 한다 —"
            " 없으면 이 게이트가 FAIL 한다.",
        ]
        # 측정이 이미 있으면 **판정을 보여 준다.** 닫아 둔 이유가 파일 안에만 있으면
        # 다음 사람이 "왜 안 켰지?"를 다시 재게 된다.
        if ARMS.exists():
            try:
                r = json.loads(ARMS.read_text(encoding="utf-8"))
                blockers = [
                    f"{lab} {r[k]['delta_pp']:+.2f}pp"
                    f" CI[{r[k]['ci'][0]:+.2f}, {r[k]['ci'][1]:+.2f}]"
                    for lab, k in (("표면", "on_minus_off"), ("리프", "on_minus_off_leaf"))
                    if k in r and r[k]["ci"][1] < 0
                ]
                lines.append(
                    f"  측정은 있다 ({ARMS.relative_to(ROOT)}) — "
                    + (
                        "켜면 해로워서 닫아 둔 것이다: " + " · ".join(blockers)
                        if blockers
                        else "켰을 때의 이득이 유의하지 않아 닫아 둔 것이다"
                    )
                )
            except (OSError, json.JSONDecodeError, KeyError, IndexError):
                lines.append(f"  ⚠️ {ARMS.relative_to(ROOT)} 를 못 읽었다")
        return report(NAME, PASS, lines)

    # ── 5. 채널이 있으면 두 팔 측정을 **요구**한다 ──────────────────
    if not ARMS.exists():
        return report(
            NAME,
            FAIL,
            lines
            + [
                "",
                "⛔ TIER1_RANKING_CHANNEL = true 인데 두 팔 측정이 없다.",
                f"   기대 경로: {ARMS.relative_to(ROOT)}",
                "   채널이 있는데 검증이 없는 상태는 채널이 없는 상태보다 나쁘다 —",
                "   Tier 1 없는 기기가 아무도 재보지 않은 순위를 받게 된다.",
            ],
        )
    try:
        r = json.loads(ARMS.read_text(encoding="utf-8"))
        off, on = r["off"], r["on"]
        ovb, omo = r["off_vs_baseline"], r["on_minus_off"]
        # ⛔ 두 입도를 **둘 다** 요구한다. 없으면 못 잰 것이고, 못 잰 축은 통과가 아니다.
        ovb_l, omo_l = r["off_vs_baseline_leaf"], r["on_minus_off_leaf"]
    except (OSError, json.JSONDecodeError, KeyError) as e:
        return report(
            NAME,
            FAIL,
            lines
            + [
                f"⛔ 두 팔 측정 파싱 실패: {e}",
                "   리프 축(off_vs_baseline_leaf · on_minus_off_leaf)도 **필수**다 —",
                "   표면만 재면 블록 단위 신호가 리프를 흩뜨리는 것을 못 본다(2026-08-31).",
            ],
        )

    surface = r.get("surface", "?")
    lines += [
        "",
        f"두 팔 · 표면 {surface} · 홀드아웃 {int(r.get('n', 0)):,}",
        f"  Tier1-off (기준선) {off['acc']:.4f}  CI[{off['ci'][0]:.4f}, {off['ci'][1]:.4f}]",
        f"  Tier1-on           {on['acc']:.4f}  CI[{on['ci'][0]:.4f}, {on['ci'][1]:.4f}]",
        f"  off vs 이전 배포본  {ovb['delta_pp']:+.2f}pp  "
        f"CI[{ovb['ci'][0]:+.2f}, {ovb['ci'][1]:+.2f}]",
        f"  on − off           {omo['delta_pp']:+.2f}pp  "
        f"CI[{omo['ci'][0]:+.2f}, {omo['ci'][1]:+.2f}]",
        f"  [리프 top-1] off vs 배포본 {ovb_l['delta_pp']:+.2f}pp "
        f"CI[{ovb_l['ci'][0]:+.2f}, {ovb_l['ci'][1]:+.2f}] · "
        f"on − off {omo_l['delta_pp']:+.2f}pp "
        f"CI[{omo_l['ci'][0]:+.2f}, {omo_l['ci'][1]:+.2f}]",
    ]

    if surface != "l2_top5":
        lines.append(
            f"  ⚠️ 판정 표면이 제품 표면(l2_top5)이 아니다 — {surface}."
            " 유리한 입도가 기본값이 되면 아무도 안 본다(2026-08-29)."
        )

    # ⛔ 핵심 규칙 ①: Tier 1 **없는** 기기가 나빠지면 차단. 두 입도 다 본다.
    for label, d in (("표면", ovb), ("리프", ovb_l)):
        if d["ci"][1] < 0:
            return report(
                NAME,
                FAIL,
                lines
                + [
                    "",
                    f"⛔ Tier 1 없는 기기가 {label} 축에서 유의하게 나빠졌다 — 배포 금지.",
                    "   모델이 Tier 1 에 기대고 있다는 뜻이다. 학습에 채널 드롭아웃을 넣어라.",
                ],
            )

    # ⛔ 핵심 규칙 ②: 채널을 **켜는 것 자체가** 어느 입도에서든 해로우면 차단.
    #
    # 첫 판에는 이 검사가 없었다. 표면만 보고 "무의미 = 경고"로 흘려보내는 동안 리프가
    # 유의하게 무너지고 있었다(2026-08-31 실측 −1.83pp). 켜서 손해인 채널을 여는 것은
    # 안 켜는 것보다 나쁘다 — 경고가 아니라 차단이다.
    for label, d in (("표면", omo), ("리프", omo_l)):
        if d["ci"][1] < 0:
            return report(
                NAME,
                FAIL,
                lines
                + [
                    "",
                    f"⛔ 채널을 켜면 {label} 축이 유의하게 나빠진다 — 채널 개방 금지.",
                    f"   on − off = {d['delta_pp']:+.2f}pp"
                    f" CI[{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}]",
                    "   범주형 지각은 L1/L2 **블록 단위**라 블록을 통째로 올린다 —",
                    "   블록은 맞히고 그 안 리프 순서를 흩뜨린다. 리프 해상도 신호",
                    "   (OS 조밀 이미지 임베딩 등)가 필요하다는 신호다.",
                ],
            )

    if not sig(omo["ci"]) and not sig(omo_l["ci"]):
        lines.append(
            "  ⚠️ Tier 1 이 값을 못 하고 있다(on − off 두 입도 다 무의미) — 차단은 아니지만"
            " 3층을 켜 둘 이유가 없다."
        )
    return report(NAME, PASS, lines)


main_guard(run_gate)
