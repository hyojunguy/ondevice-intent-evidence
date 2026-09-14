#!/usr/bin/env python3
"""C8 — 우리가 주장하는 프라이버시에 실제로 대가를 치르고 있는가. (zCDP 회계)

노이즈를 "넣었다"는 것은 보장이 아니다. 보장은 (clip, sigma, 참여자 수, delta) 로부터
계산되는 숫자다. 이 게이트가 없으면 sigma 를 조용히 낮춰도 문서는 계속 "차등
프라이버시"라고 쓴다.

회계는 **zCDP**(Bun–Steinke)를 쓴다. 고전 (ε,δ) 경계보다 합성이 정확해서 이 분야의
정론이고, NIST SP 800-226(2025) 이 정부·정책 심사 프레임워크로 자리잡은 뒤로는
심사자가 이 틀로 물어온다.

  가우시안:  ρ = Δ² / (2σ²)
  (ε,δ) 변환: ε = ρ + 2√(ρ·ln(1/δ))

⛔ Secure Aggregation 하에서 서버는 **합만** 본다. 참여자 n 명이 독립 노이즈를 더하면
   합의 노이즈 표준편차가 σ√n 이므로 **집계 관점 ρ 는 n 에 반비례한다.**
   즉 프라이버시는 노이즈 노브가 아니라 **채택 규모**에서 나온다. 이 게이트는 그
   사실을 숫자로 강제한다 — 목표 ε 을 만족하는 최소 참여자 수를 계산해 보고한다.
"""
from __future__ import annotations
import math, sys
from _common import BUDGETS, PASS, FAIL, UNMEASURED, report, main_guard

# ⛔ 프라이버시 회계를 손으로 짜지 않는다. 미묘하게 틀린 회계는 자신 있게 틀린 숫자를 내고
#    아무도 못 잡는다. Google `dp-accounting` 의 PLD 를 쓴다(현재 최선의 회계).
#    없으면 캠페인 수치를 UNMEASURED 로 두고 절대 지어내지 않는다.
try:
    from dp_accounting.pld import privacy_loss_distribution as _pld
except ImportError:  # pragma: no cover - 환경 차이
    _pld = None


def campaign_epsilon(z_eff: float, q: float, rounds: int, delta: float) -> float | None:
    """R 라운드 캠페인 전체 ε. `q` 는 라운드당 참여 확률(= 코호트/기기인구).

    ⛔ 서브샘플링 증폭은 공짜가 아니다. 성립 조건이 있다:
       (1) 참여가 실제로 무작위여야 한다 — 서버가 누가 참여했는지 알면 증폭이 무너진다,
       (2) Secure Aggregation 이 성립해야 한다(개별 업데이트 비관측),
       (3) Poisson 샘플링 가정이다 — 실제 FL 은 기기 가용성에 따르는 비복원 추출이라
           정확히 같지 않다.
       이 조건을 못 지키면 q=1(증폭 없음) 값을 써야 한다.
    """
    if _pld is None:
        return None
    d = _pld.from_gaussian_mechanism(
        standard_deviation=z_eff, sampling_prob=q, value_discretization_interval=1e-3
    )
    return float(d.self_compose(rounds).get_epsilon_for_delta(delta))


def rho_gaussian(clip: float, sigma: float) -> float:
    return clip * clip / (2.0 * sigma * sigma)


def eps_from_rho(rho: float, delta: float) -> float:
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))


def run_gate() -> int:
    fl = BUDGETS.get("fl", {})
    need = ("clip_norm", "noise_sigma", "delta", "target_epsilon_per_round", "max_rounds")
    missing = [k for k in need if k not in fl]
    if missing:
        return report("C8 dp-budget", UNMEASURED, [f"budgets.json:fl 에 없는 키: {missing}"])

    clip, sigma = float(fl["clip_norm"]), float(fl["noise_sigma"])
    delta, target = float(fl["delta"]), float(fl["target_epsilon_per_round"])
    n_assumed = int(fl.get("assumed_participants", fl["min_participants"]))
    z = sigma / clip

    # 집계 관점: n 명의 독립 노이즈 합 → sigma_agg = sigma * sqrt(n)
    def eps_at(n: int) -> float:
        return eps_from_rho(rho_gaussian(clip, sigma * math.sqrt(n)), delta)

    # 목표 ε 을 만족하는 최소 n (전수 탐색, 상한 100만)
    required_n = next((n for n in range(1, 1_000_001) if eps_at(n) <= target), None)

    eps_now = eps_at(n_assumed)

    # ── 캠페인 전체 (사람의 프라이버시에 실제로 해당하는 값) ──────────────────
    population = int(fl.get("device_population", n_assumed))
    target_campaign = float(fl.get("target_epsilon_campaign", 0.0))
    rounds = int(fl["max_rounds"])
    z_eff = sigma * math.sqrt(n_assumed) / clip
    q = min(1.0, n_assumed / population) if population > 0 else 1.0

    # ⛔ 길이는 ε 에 안 들어가지만, "안 들어간다"를 **말로 하지 말고 값으로** 찍는다.
    #    2026-08-30 에 θ 5개를 연합분에 넣으면서 이 게이트를 다시 돌린 이유가 그것이다.
    dim = int(BUDGETS["model"]["dim"])
    combiner_len = int(BUDGETS["fl"].get("combiner_len", 0))
    mlp_len = int(BUDGETS["fl"].get("combiner_mlp_len", 0))
    # ⛔ 팔이 늘 때마다 이 줄을 손으로 고치게 두면 게이트가 조용히 옛 길이를 보고한다 —
    #    2026-08-31 θu 를 붙였을 때 실제로 그랬다. 길이는 budgets.json 에서 읽는다.
    user_len = int(BUDGETS["fl"].get("combiner_user_len", 0))
    pstats_len = int(BUDGETS["fl"].get("purchase_stats_len", 0))
    rank_len = int(BUDGETS["fl"].get("rank_head_len", 0))
    federated_len = dim * dim + combiner_len + mlp_len + user_len + pstats_len + rank_len
    param_len = federated_len + dim

    eps_naive = eps_from_rho(rho_gaussian(clip, sigma * math.sqrt(n_assumed)) * rounds, delta)
    eps_camp = campaign_epsilon(z_eff, q, rounds, delta)

    lines = [
        f"clip={clip} · sigma={sigma} · z={z:.3f} · delta={delta:g}",
        f"단일 클라이언트 ρ = {rho_gaussian(clip, sigma):.2f}  (참고 — 서버는 이걸 보지 않는다)",
        f"집계 관점 @ n={n_assumed:,}: ε ≈ {eps_now:.2f}  (목표 ≤ {target:.2f})",
        f"목표를 만족하는 최소 참여자 수: {required_n:,}명" if required_n else "목표를 만족하는 n 이 100만 이하에 없다",
        "",
        f"── 캠페인 전체 ({rounds} 라운드) — 사람의 프라이버시에 해당하는 값 ──",
        f"증폭 없음(q=1, zCDP): ε ≈ {eps_naive:.2f}   ⛔ 모든 기기가 매 라운드 참여한다는 비현실적 가정",
        (f"서브샘플링(PLD, 인구 {population:,} → q={q:.4f}): ε ≈ {eps_camp:.2f}"
         f"  (목표 ≤ {target_campaign:.2f})" if eps_camp is not None
         else "⬜ 서브샘플링 회계 UNMEASURED — `uv pip install dp-accounting` 후 다시"),
        "⚠️ 증폭 성립 조건: 참여가 무작위 · SecAgg 성립 · Poisson 샘플링 가정."
        " 못 지키면 위의 q=1 값을 써야 한다.",
        "⚠️ 참고: Gboard 실배포는 캠페인 **전체** 누적 ε 0.99~13.7(δ=1e-10)을 보고한다.",
        "",
        f"── 결합기 θ({combiner_len}) + MLP({mlp_len}) + θu({user_len}) + 구매통계({pstats_len})"
        f" + 랭킹헤드({rank_len})를 연합 벡터에 넣은 뒤 ──",
        f"연합 벡터 길이 {federated_len:,} (= {dim}×{dim} + θ {combiner_len} + mlp {mlp_len}"
        f" + θu {user_len} + pstats {pstats_len} + rank {rank_len}) "
        f"· 전체 {param_len:,}",
        f"ε 변화: **없음**. ρ = clip²/(2σ²) 는 좌표 수와 무관하고, θ·mlp·θu 는 새 채널이 아니라 "
        f"**같은 clip 예산 안**에 들어간다(privatize 가 벡터 전체에 clip_l2 를 건다).",
        f"⚠️ 대신 좌표당 신호가 "
        f"{(combiner_len + mlp_len + user_len + pstats_len + rank_len) / federated_len * 100:.2f}% 묽어진다 — "
        f"공짜가 아니라 **아주 싼** 것이다. 어트리뷰션처럼 별도 채널로 뺐다면 ε 이 합성돼 "
        f"올랐을 것이다.",
    ]
    if z < 0.05:
        lines.append("⛔ z<0.05 는 사실상 노이즈 없음 — 프라이버시 주장 금지.")
        return report("C8 dp-budget", FAIL, lines)
    if eps_camp is None:
        lines.append("⬜ 캠페인 회계 없이 통과시키지 않는다 — 그게 이 게이트의 요점이다.")
        return report("C8 dp-budget", UNMEASURED, lines)
    if eps_camp > target_campaign:
        lines.append(f"⛔ 캠페인 ε {eps_camp:.2f} > 목표 {target_campaign:.2f}"
                     f" — 인구를 늘리거나 라운드를 줄이거나 sigma 를 올려야 한다.")
        return report("C8 dp-budget", FAIL, lines)
    return report("C8 dp-budget", PASS if eps_now <= target else FAIL, lines)


# ⛔ 최상위에서 부르면 이 모듈을 **import 할 수 없다** — import 하는 순간 게이트가
#    돌고 sys.exit 로 호출자를 죽인다. 게이트를 재사용하려던 코드가 여기서 잘렸다.
if __name__ == "__main__":
    main_guard(run_gate)
