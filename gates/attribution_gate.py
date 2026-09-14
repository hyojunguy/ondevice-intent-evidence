#!/usr/bin/env python3
"""C9 — 어트리뷰션 채널이 프라이버시 예산을 **실제로** 쓰고 있는가.

⛔ 이 게이트가 생긴 이유(2026-08-26): `impression_buckets` 는 길이 무제한 raw 광고 ID
   목록이었고, `clip_norm`·`noise_sigma`·ε 회계는 전부 `delta` 채널에만 걸려 있었다.
   즉 우리가 인용해 온 캠페인 ε 1.54 는 **두 채널 중 하나만의 값**이었다. `04-threat-model.md`
   는 "집계 버킷으로 완화"한다고 이미 적고 있었는데 코드가 그렇게 하지 않고 있었다.

이 게이트가 판정하는 것:

  1. 민감도가 유계인가 — ℓ0·ℓ∞ 상한이 선언돼 있고 축이 의도 헤드와 맞는가.
  2. 두 채널을 **합성한** 캠페인 ε 이 목표 이하인가.

⛔ 2번이 핵심이다. ε 은 채널마다 따로 자랑하는 숫자가 아니다. 사람이 겪는 손실은 합성이고,
   합성을 안 하면 채널을 늘릴수록 각 채널의 ε 은 그대로인데 실제 손실만 늘어난다.
   그래서 어트리뷰션을 추가하는 대가는 **학습에 쓸 예산이 준다**는 것이고, 이 게이트가
   그 대가를 숨기지 못하게 한다.

회계는 `dp_gate.py` 와 같은 Google `dp-accounting` PLD 를 쓴다. 없으면 UNMEASURED —
지어내지 않는다.
"""
from __future__ import annotations

import math
import sys

from _common import BUDGETS, FAIL, PASS, UNMEASURED, main_guard, report

try:
    from dp_accounting.pld import privacy_loss_distribution as _pld
except ImportError:  # pragma: no cover - 환경 차이
    _pld = None


def l2_sensitivity(max_active: int, max_count: int) -> float:
    """ℓ0 ≤ k, ℓ∞ ≤ m 이면 ℓ2 ≤ m·√k. `delta` 채널의 clip_norm 과 같은 자리의 값."""
    return max_count * math.sqrt(max_active)


def _channel_pld(z: float, q: float, rounds: int):
    return _pld.from_gaussian_mechanism(
        standard_deviation=z, sampling_prob=q, value_discretization_interval=1e-3
    ).self_compose(rounds)


def composed_campaign_epsilon(
    z_delta: float, z_attr: float | None, q: float, rounds: int, delta: float
) -> float | None:
    """두 가우시안 메커니즘을 라운드마다 함께 돌렸을 때의 캠페인 총 ε.

    각 채널의 PLD 를 만들어 라운드 수만큼 self-compose 한 뒤 서로 compose 한다.
    ⛔ ε 을 그냥 더하지 않는다 — 기본 합성은 느슨해서 실제보다 나쁜 숫자를 내고,
       그 숫자로 설계를 정하면 필요 없이 노이즈를 키우게 된다.
    """
    if _pld is None:
        return None
    a = _channel_pld(z_delta, q, rounds)
    # ⛔ z_attr=None 은 "어트리뷰션 채널이 없다"는 뜻이다. inf 를 넣어 무노이즈를 흉내내면
    #    PLD 가 NaN 으로 죽는다(실측) — 채널을 빼는 것은 큰 수를 넣는 것과 다르다.
    total = a if z_attr is None else a.compose(_channel_pld(z_attr, q, rounds))
    return float(total.get_epsilon_for_delta(delta))


def run_gate() -> int:
    attr = BUDGETS.get("attribution", {})
    fl = BUDGETS.get("fl", {})

    need_attr = ("buckets", "max_active_buckets", "max_count_per_bucket", "noise_sigma")
    missing = [k for k in need_attr if k not in attr]
    if missing:
        return report(
            "C9 attribution-budget",
            UNMEASURED,
            [f"budgets.json:attribution 에 없는 키: {missing}"],
        )

    buckets = int(attr["buckets"])
    max_active = int(attr["max_active_buckets"])
    max_count = int(attr["max_count_per_bucket"])
    sigma_attr = float(attr["noise_sigma"])
    sens = l2_sensitivity(max_active, max_count)

    lines = [
        f"축 {buckets} 버킷 · ℓ0 ≤ {max_active} · ℓ∞ ≤ {max_count} → ℓ2 민감도 Δ = {sens:.4f}",
    ]

    # ── 1. 민감도가 유계인가 ────────────────────────────────────────────────
    if max_active < 1 or max_count < 1 or max_active > buckets:
        return report(
            "C9 attribution-budget",
            FAIL,
            lines + [f"상한이 말이 안 된다: ℓ0={max_active}, ℓ∞={max_count}, 축={buckets}"],
        )

    # 축이 의도 헤드와 어긋나면 노출이 엉뚱한 버킷에 기록된다.
    head_classes = _head_class_count()
    if head_classes is None:
        lines.append("⚠️  head.bin 을 못 읽어 축 일치는 확인 못 했다")
    elif head_classes != buckets:
        return report(
            "C9 attribution-budget",
            FAIL,
            lines + [f"축 불일치: budgets.attribution.buckets={buckets} != head.bin 클래스 {head_classes}"],
        )
    else:
        lines.append(f"축 일치 확인: head.bin 클래스 {head_classes} == buckets")

    # ── 2. 합성 캠페인 ε ────────────────────────────────────────────────────
    need_fl = ("clip_norm", "noise_sigma", "delta", "max_rounds", "device_population")
    missing_fl = [k for k in need_fl if k not in fl]
    if missing_fl:
        return report("C9 attribution-budget", UNMEASURED, lines + [f"budgets.json:fl 에 없는 키: {missing_fl}"])

    clip = float(fl["clip_norm"])
    sigma_delta = float(fl["noise_sigma"])
    target_delta = float(fl["delta"])
    rounds = int(fl["max_rounds"])
    n = int(fl.get("assumed_participants", fl["min_participants"]))
    population = int(fl["device_population"])
    q = n / population if population else 1.0

    # Secure Aggregation: 서버는 합만 본다 → 집계 관점 노이즈는 σ√n.
    # 두 채널 모두 같은 참여자 집합이 같은 라운드에 기여하므로 같은 n·q 를 쓴다.
    z_delta = (sigma_delta * math.sqrt(n)) / clip
    z_attr = (sigma_attr * math.sqrt(n)) / sens
    lines.append(f"집계 관점 z: delta {z_delta:.3f} · attribution {z_attr:.3f} (n={n}, q={q:.3f})")

    eps = composed_campaign_epsilon(z_delta, z_attr, q, rounds, target_delta)
    if eps is None:
        return report(
            "C9 attribution-budget",
            UNMEASURED,
            lines
            + [
                "dp-accounting 미설치 — 합성 ε 을 지어내지 않는다.",
                "  복구: pip install dp-accounting",
            ],
        )

    target = float(fl.get("target_epsilon_campaign", 2.0))
    lines.append(f"캠페인 총 ε(두 채널 합성, {rounds}라운드) = {eps:.3f}  (목표 ≤ {target})")

    # 대가를 숨기지 않는다: 어트리뷰션이 얼마를 먹었나.
    delta_only = composed_campaign_epsilon(z_delta, None, q, rounds, target_delta)
    if delta_only is not None:
        lines.append(
            f"  이 중 delta 채널만이면 {delta_only:.3f} → 어트리뷰션이 쓴 몫 +{eps - delta_only:.3f}"
        )

    if eps > target:
        lines.append("⛔ 예산 초과 — noise_sigma 를 올리거나 max_active_buckets 를 줄여라.")
        lines.append("   ⚠️ 목표치를 올려서 통과시키지 마라. 그건 보장을 낮추는 것이지 고치는 게 아니다.")
        return report("C9 attribution-budget", FAIL, lines)

    k_anon = attr.get("min_bucket_count")
    if k_anon:
        lines.append(f"서버측 k-익명 억제 임계 {k_anon} — DP 노이즈와 보완재(A2)")
    else:
        lines.append("⚠️  min_bucket_count 미선언 — 희소 버킷이 노이즈를 뚫을 수 있다(A2)")

    return report("C9 attribution-budget", PASS, lines)


def _head_class_count() -> int | None:
    """head.bin 헤더에서 클래스 수를 읽는다: magic(8) | version(4) | n(4) | dim(4) | scale(4)."""
    import pathlib
    import struct

    p = pathlib.Path(__file__).resolve().parents[1] / "artifacts" / "head.bin"
    try:
        b = p.read_bytes()[:16]
        if len(b) < 16 or b[:8] != b"PLAIDHED":
            return None
        return struct.unpack("<I", b[12:16])[0]
    except OSError:
        return None


if __name__ == "__main__":
    sys.exit(main_guard(run_gate))
