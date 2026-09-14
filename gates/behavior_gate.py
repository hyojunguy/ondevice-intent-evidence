#!/usr/bin/env python3
"""C12 — 행동 축이 **제품 경로에** 배선돼 있고, 동의가 그것을 실제로 막나.

# 왜 이 게이트가 생겼나 (2026-08-29)

`EventLog`/`ContentSketch`/`DeviceContext` 는 2026-08-28 부터 코어에 있었지만
**그것을 쓰는 프로덕션 코드가 0개**였고, 게이트 10개 중 행동 축을 재는 것도 0개였다.
그래서 오프라인 실험이 낸 융합 이득을 제품이 재현하는지 아무도 확인하지 않았고,
확인하지 않는 축은 조용히 갈라진다 — 이 레포가 이미 여러 번 겪은 실패 모양이다
(FL 라운드가 돌면서 1차 랭커에 아무 영향이 없던 2026-08-27 이전 상태가 같은 계열).

# 무엇을 판정하나

1. ⛔ **동의 거부 = 비트 동일.** `BehavioralConsent::Denied` 로 만든 팔의 예측이
   텍스트만 팔과 **한 건도 다르지 않아야** 한다. 이건 정확도 문제가 아니라 법 문제다
   (2026-07-22 개인정보위 틱톡 처분의 1번 사유가 번들 동의였다). ν 는 그대로 주고
   측정한다 — ν=0 으로 돌려서 통과시키면 "동의가 막았다"가 아니라 "내가 껐다"를 잰다.
2. **행동이 실제로 순위를 움직인다.** 배선만 하고 상수가 0 이면 게이트는 초록인데
   축은 죽어 있다.
3. **홀드아웃 이득이 하한 이상.** 하한은 실측 CI 하단보다 낮게 잡혀 있다(budgets.json).
4. **누수 프로브.** 행동 argmax 가 정답과 거의 일치하면 그건 신호가 아니라 생성기가
   정답을 흘린 것이다(2026-08-28 사고: `events-only` 0.9933). 높으면 **FAIL** 이다 —
   좋아 보이는 수치일수록 경보다.
5. **ν 드리프트.** 코드에 박힌 `TUNED_NU` 가 train 스윕 최적점 근처인지. 멀어지면
   누군가 상수만 바꿨거나 데이터가 바뀐 것이다.
6. **결합기 축(2026-08-30).** 스칼라 ν 둘 대신 학습된 θ 다섯(`RankCombiner`)을 쓰는
   팔이 코어에 있다. 이 게이트는 두 가지를 진다 — (a) 그 팔의 A/B 결과를 **표면에
   드러내고**, (b) ⛔ **기본 팔이 스칼라인지**를 소스에서 확인한다. 측정되지 않은 θ 가
   조용히 기본이 되는 것이 이 축의 유일한 위험이고, 그건 정확도가 아니라 **거버넌스**
   문제라 코드가 지켜야 한다.

⛔ ν 는 **train split 에서** 고르고 판정은 heldout 에서 한다. 하네스가 그 분리를
   소유한다 — 게이트가 평가셋에서 튜닝하면 자기 자신을 채점하게 된다.
"""
from __future__ import annotations

import json
import pathlib
import sys

from _common import BUDGETS, FAIL, PASS, UNMEASURED, main_guard, report, run

ROOT = pathlib.Path(__file__).resolve().parents[1]
NAME = "C12 behavior-axis"


def run_gate() -> int:
    cfg = BUDGETS.get("behavior")
    if not cfg:
        return report(NAME, UNMEASURED, ["budgets.json:behavior 없음"])

    data = ROOT / cfg["dataset"]
    if not data.exists():
        return report(NAME, UNMEASURED, [f"{data} 없음 — experiments/event-intent/gen_behavior.py 먼저"])
    for a in ("vocab.txt", "embedding.bin", "taxonomy.bin"):
        if not (ROOT / "artifacts" / a).exists():
            return report(NAME, UNMEASURED, [f"artifacts/{a} 없음 — 빌드 먼저"])

    out = ROOT / "experiments" / "event-intent" / "product_path_results.json"
    p = run(
        ["cargo", "run", "--release", "-q", "-p", "oicr-sdk-core",
         "--example", "behavior_eval", "--", str(data), str(out)],
        timeout=1800,
    )
    if p.returncode != 0:
        tail = (p.stderr or p.stdout).strip().splitlines()[-6:]
        return report(NAME, UNMEASURED, ["behavior_eval 실패:"] + tail)

    try:
        r = json.loads(out.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return report(NAME, UNMEASURED, [f"결과 파싱 실패: {e}"])

    h = r["heldout"]
    delta_pp = h["delta"] * 100.0
    moved_ratio = r["predictions_moved"] / max(r["actionable_windows"], 1)
    leak = r["leak_probe"]["behavior_argmax_acc"]
    best = r["train_best"]
    nu, nu_l2 = r["nu_used"], r.get("nu_l2_used", 0.0)
    # 스윕에서 (ν, ν_l2) 자리의 **리프** 정확도. 격자에 없으면 드리프트 검사를 건너뛴다.
    at_nu = next(
        (
            x["leaf_acc"]
            for x in r["train_sweep"]
            if abs(x["nu"] - nu) < 1e-4 and abs(x.get("nu_l2", 0.0) - nu_l2) < 1e-4
        ),
        None,
    )

    leaf = r.get("leaf")
    lines = [
        f"홀드아웃 {r['n_heldout']:,}명 · train {r['n_train']:,}명 · "
        f"ν={nu:.2f} ν_l2={nu_l2:.2f} · 힌트={r.get('hint_space','?')} "
        f"(중분류 항 {'ON' if r.get('fine_grained') else 'OFF'}) · 배포 Rust 경로",
        f"대분류 28-way : {h['text']:.4f} → {h['text_plus_behavior']:.4f} "
        f"= {delta_pp:+.2f}pp  CI[{h['ci'][0]*100:+.2f}, {h['ci'][1]*100:+.2f}]",
        (
            f"⚠️ 리프 {int(r.get('total_leaves', 0)) or 2960}-way(**화면이 보여 주는 입도**): "
            f"{leaf['text']:.4f} → {leaf['text_plus_behavior']:.4f} = {leaf['delta']*100:+.2f}pp  "
            f"CI[{leaf['ci'][0]*100:+.2f}, {leaf['ci'][1]*100:+.2f}]"
            if leaf
            else "⬜ 리프 입도 미측정 — behavior_eval 이 낡았다"
        ),
        f"행태정보 동의 거부 {h['denied']:.4f} — 텍스트만과 예측 동일: {r['consent_identical']}",
        f"행동이 순위를 바꾼 사용자 {r['predictions_moved']:,} / 유효 창 {r['actionable_windows']:,} "
        f"({moved_ratio:.1%})",
        f"누수 프로브(대분류): 행동 argmax {leak:.4f} (우연 {r['leak_probe']['chance']:.4f})",
        (
            f"누수 프로브(중분류): {r['leak_probe']['l2_argmax_acc']:.4f} "
            f"(우연 {r['leak_probe']['l2_chance']:.4f}) — 힌트를 잘게 만들면 새 누수 축이 생긴다"
            if r["leak_probe"].get("l2_argmax_acc") is not None
            else "누수 프로브(중분류): 해당 없음 (중분류 항 OFF)"
        ),
    ]
    for band, v in sorted(r.get("bands", {}).items()):
        sig = "유의" if (v["ci"][0] > 0 or v["ci"][1] < 0) else "무의미"
        lines.append(
            f"  밴드 {band:<9} n={v['n']:<5} {v['text']:.4f} → {v['fused']:.4f} "
            f"({v['delta']*100:+.2f}pp, {sig})"
        )

    fails: list[str] = []

    # 1. 동의 — 법적 불변식. 다른 어떤 수치가 좋아도 이게 깨지면 FAIL 이다.
    if not r["consent_identical"]:
        fails.append(
            "⛔ 행태정보 동의를 거부했는데 예측이 달라졌다 — 동의 없이 행동이 쓰이고 있다. "
            "EventLog::new 의 타입 게이트가 우회된 것이다."
        )

    # 2. 축이 살아 있나
    if moved_ratio < cfg["min_moved_ratio"]:
        fails.append(
            f"⛔ 행동이 순위를 거의 안 바꾼다({moved_ratio:.1%} < {cfg['min_moved_ratio']:.0%}) — "
            "배선은 됐는데 축이 죽어 있다. ν 또는 MIN_EVENTS 를 확인하라."
        )

    # 3. 이득 하한
    if delta_pp < cfg["min_delta_pp"]:
        fails.append(
            f"⛔ 행동 이득 {delta_pp:+.2f}pp 가 하한 {cfg['min_delta_pp']:+.2f}pp 미달. "
            "⚠️ 하한을 낮춰서 통과시키지 마라 — 그건 회귀를 기록에서 지우는 것이다."
        )
    if not h["significant"]:
        fails.append("⛔ 이득의 신뢰구간이 0 을 걸친다 — 이 수치로 무엇을 주장하지 마라.")

    # 4. 누수 — 좋아 보이는 수치일수록 경보다
    leak_l2 = r["leak_probe"].get("l2_argmax_acc")
    if leak_l2 is not None and leak_l2 > cfg["max_leak_probe"]:
        fails.append(
            f"⛔ 중분류만으로 {leak_l2:.4f} 를 맞힌다 — 생성기가 정답 중분류를 베끼고 있다. "
            "리프 해상도를 올리려다 누수를 만든 것이다."
        )
    if leak > cfg["max_leak_probe"]:
        fails.append(
            f"⛔ 행동만으로 {leak:.4f} 를 맞힌다 — 신호가 아니라 생성기가 정답을 흘렸을 "
            "가능성이 높다(2026-08-28 events-only 0.9933 사고)."
        )

    # 5. ν 드리프트
    if at_nu is not None and best["leaf_acc"] > 0:
        ratio = at_nu / best["leaf_acc"]
        lines.append(
            f"ν 드리프트: train 리프 acc@(ν,ν_l2) {at_nu:.4f} / 최적 {best['leaf_acc']:.4f} "
            f"(ν*={best['nu']:.2f}, ν_l2*={best.get('nu_l2', 0.0):.2f}) = {ratio:.3f}"
        )
        if ratio < cfg["nu_train_optimality"]:
            fails.append(
                f"⛔ (TUNED_NU, TUNED_NU_L2) 가 train 최적점에서 멀어졌다({ratio:.3f} < "
                f"{cfg['nu_train_optimality']}). 상수만 바뀌었거나 데이터가 바뀌었다 — "
                f"(ν, ν_l2) = ({best['nu']:.2f}, {best.get('nu_l2', 0.0):.2f}) 로 "
                f"재조정하고 근거를 behavior.rs 에 적어라."
            )

    if leaf:
        ratio = (leaf["delta"] / h["delta"]) if h["delta"] else 0.0
        lines.append(
            f"⛔ 리프 이득은 대분류 이득의 **{ratio:.0%}** 다. 대분류 사전확률은 리프를 "
            f"**블록째** 밀기 때문에 블록을 맞게 골라도 그 안에서는 못 고른다 — 그리고 "
            f"화면·소재 매칭이 쓰는 것은 리프다. 헤드라인으로 대분류 수치를 쓰지 마라."
        )
        if not leaf["significant"]:
            fails.append("⛔ 리프 입도에서 이득의 CI 가 0 을 걸친다 — 화면이 보는 축에서 무의미하다.")
        leaf_pp = leaf["delta"] * 100.0
        if leaf_pp < cfg.get("min_leaf_delta_pp", 0.0):
            fails.append(
                f"⛔ 리프 이득 {leaf_pp:+.2f}pp 가 하한 {cfg['min_leaf_delta_pp']:+.2f}pp 미달. "
                "화면이 보는 축에서 회귀했다 — 대분류 수치가 좋아도 통과시키지 마라."
            )

    # ── 보수적 세계(행동이 '영역'만 말하는 판)도 함께 잰다 ────────────
    #
    # ⛔ 유리한 세계 하나만 재면 그게 곧 체리피킹이다. 리프 힌트는 제품 SDK 계약과 같지만
    #    현실의 호스트 앱이 항상 리프를 알려 준다는 보장은 없다 — 두 세계를 다 낸다.
    ref = ROOT / cfg.get("reference_dataset", "")
    if ref.exists():
        ref_out = ROOT / "experiments" / "event-intent" / "product_path_results.json"
        rp = run(
            ["cargo", "run", "--release", "-q", "-p", "oicr-sdk-core",
             "--example", "behavior_eval", "--", str(ref), str(ref_out)],
            timeout=1800,
        )
        if rp.returncode == 0:
            try:
                rr = json.loads(ref_out.read_text(encoding="utf-8"))
                lines.append(
                    f"[보수 세계 {ref.name} · 힌트=l1 · 중분류 항 OFF] "
                    f"대분류 {rr['heldout']['delta']*100:+.2f}pp · "
                    f"리프 {rr['leaf']['delta']*100:+.2f}pp"
                )
                if rr["leaf"]["delta"] <= 0:
                    fails.append(
                        "⛔ 보수 세계에서 리프 이득이 사라졌다 — 이득이 리프 힌트 유무에 "
                        "전적으로 의존한다면 그 전제를 스펙에 적어야 한다."
                    )
            except (OSError, json.JSONDecodeError):
                lines.append("⬜ 보수 세계 결과 파싱 실패")
        else:
            lines.append("⬜ 보수 세계 측정 실패")

    # ── 6. 결합기 축 ───────────────────────────────────────────────────
    #
    # ⛔ 기본 팔은 소스가 진다. A/B 파일이 없어도 이건 검사한다 — 파일이 없는데
    #    기본이 결합기면 "측정 없이 배포"이고, 그게 정확히 막아야 할 상태다.
    wasm = (ROOT / "crates/sdk-wasm/src/lib.rs").read_text(encoding="utf-8")
    default_scalar = "combiner_arm: CombinerArm::Scalar," in wasm
    lines.append("")
    lines.append(
        f"── 결합기 축 — 기본 팔: {'스칼라 ν(현행 유지)' if default_scalar else '⛔ 학습 팔'}"
    )
    if not default_scalar:
        fails.append(
            "⛔ wasm 세션의 `combiner_arm` 기본값이 CombinerArm::Scalar 가 아니다 — "
            "학습 팔(θ5/MLP)이 기본 배포본이 되려면 실라벨 A/B 근거와 함께 이 게이트를 "
            "고쳐야 한다(2026-08-30 사다리 수치는 합성 정답-상한 라벨이라 근거가 아니다)."
        )
    # ⛔ MLP 팔(89p)의 워름스타트 순위중립은 sdk-core 테스트가 지킨다
    #    (`warm_start_ranks_identically_to_the_deployed_combiner`). 여기서는 그 테스트가
    #    실제로 존재하는지만 대조한다 — 테스트를 지우면 이 게이트가 알아챈다.
    combiner_rs = (ROOT / "crates/sdk-core/src/combiner.rs").read_text(encoding="utf-8")
    if "fn warm_start_ranks_identically_to_the_deployed_combiner" not in combiner_rs:
        fails.append(
            "⛔ MLP 워름스타트 순위중립 테스트가 사라졌다 — sdk-core combiner.rs 를 확인하라."
        )
    else:
        lines.append("MLP(89p) 워름스타트 순위중립: sdk-core 테스트가 지킨다 (존재 확인 OK)")

    # ── 6b. 유저 임베딩 축 (2026-08-31 (나) 구조) ─────────────────────────
    #
    # ⛔ W 는 배포되지만 **기본은 꺼짐**이다. 근거가 합성 페르소나 패널뿐이라
    #    (ship_arch.json `_caveat`) 조용히 기본이 되면 배포본이 무엇인지 아무도 모른다.
    #    같은 이유로 θu 팔도 명시 호출로만 켜진다.
    ue_off = "user_embedding_on: false," in wasm
    lines.append(
        f"유저 임베딩 기본: {'꺼짐(현행 유지)' if ue_off else '⛔ 켜짐'}"
    )
    if not ue_off:
        fails.append(
            "⛔ wasm 세션의 `user_embedding_on` 기본값이 false 가 아니다 — 배포된 W 를 "
            "기본으로 켜려면 실패널 근거가 필요하다(지금 근거는 합성 페르소나 대역이다)."
        )
    # θu 팔이 배포 순위와 같은 워름스타트에서 출발하는지는 sdk-core 테스트가 진다.
    if "fn deployed_user_theta_ranks_like_the_deployed_combiner" not in combiner_rs:
        fails.append(
            "⛔ θu 워름스타트 순위중립 테스트가 사라졌다 — sdk-core combiner.rs 를 확인하라."
        )
    else:
        lines.append("θu(8p) 워름스타트 순위중립: sdk-core 테스트가 지킨다 (존재 확인 OK)")
    # ⛔ φ8(리프질량)과 φ8-user(유저항)는 폭이 같아 섞여도 길이 검사를 통과한다.
    #    표본 리스트가 실제로 분리돼 있는지 소스로 확인한다 — 합치는 리팩터가 오면 잡는다.
    if "ranking_samples_user" not in wasm:
        fails.append(
            "⛔ φ8-user 표본 리스트(`ranking_samples_user`)가 없다 — φ8 과 같은 리스트를 "
            "쓰면 θu 가 리프질량 축 위에서 유저항 가중치를 배운다(조용히 틀린다)."
        )
    else:
        lines.append("φ8-user 표본은 별도 리스트(`ranking_samples_user`) — 축 혼동 구조적 차단")

    # ── 6c. 지각 1층 동의 등급 (2026-08-31) ───────────────────────────────
    #
    # 화면 내용 읽기는 행태정보보다 **한 칸 위** 등급이다. 아래 셋이 그 결정을 코드로
    # 붙들고 있고, 하나라도 빠지면 등급이 조용히 합쳐진다:
    #
    #   (1) 코어가 두 동의를 따로 받고 둘 다 있을 때만 스케치를 만든다
    #       — 하나로 합친 bool 이면 번들이 표현 가능해진다.
    #   (2) wasm 세션의 스케치가 `Option` 이다
    #       — 항상 존재하면 "동의 없이도 담을 곳은 있다"가 되고, 그건 이 등급의 부정이다.
    #   (3) 행태정보 철회가 콘텐츠까지 흘러내린다
    #       — 2026-08-31 이전에는 이 줄이 없어서 **철회 뒤에도 스케치가 살아 있었다**.
    core_event = (ROOT / "crates/sdk-core/src/event.rs").read_text(encoding="utf-8")
    lines.append("")
    lines.append("── 지각 1층 — 화면 내용 읽기 동의 등급")
    tier_checks = [
        (
            "pub fn new(behavioral: BehavioralConsent, content: ContentConsent) -> Option<Self>"
            in core_event,
            "⛔ `ContentSketch::new` 가 두 동의를 따로 받고 Option 을 주지 않는다 — "
            "행태정보 없이 화면 읽기만 켜진 상태가 표현 가능해졌다.",
            "코어: 두 동의를 따로 받고 둘 다 있을 때만 스케치 생성",
        ),
        (
            "content: Option<ContentSketch>," in wasm,
            "⛔ wasm 세션의 콘텐츠 스케치가 `Option` 이 아니다 — 동의 없이도 담을 곳이 "
            "있으면 '동의가 없으면 접을 곳이 없다'는 계약이 거짓이 된다.",
            "wasm: 스케치가 Option — 동의 없으면 담을 곳 자체가 없다",
        ),
        (
            "sess.events = None;\n            sess.content = None;" in wasm,
            "⛔ `events_disable` 이 콘텐츠 스케치를 안 버린다 — 행태정보를 철회했는데 "
            "읽은 글이 계속 접혀 있으면 '철회했다'가 거짓말이다(2026-08-31 이전 상태).",
            "철회 cascade: 행태정보를 거두면 콘텐츠 스케치도 버린다",
        ),
    ]
    for ok, fail_msg, ok_msg in tier_checks:
        if ok:
            lines.append(f"  {ok_msg}")
        else:
            fails.append(fail_msg)

    ab_path = ROOT / "experiments/user-vector-gap/combiner_ab.json"
    if not ab_path.exists():
        lines.append(
            "⬜ A/B 미측정 — `cargo run --release -p oicr-fl-client --example "
            "user_vector_gap` 후 `experiments/user-vector-gap/combiner_ab.py`"
        )
    else:
        try:
            ab = json.loads(ab_path.read_text(encoding="utf-8"))
            byname = {(g["a"], g["b"]): g for g in ab["gaps"]}
            impl = byname.get(("comb_dep", "prior"))
            learn = byname.get(("comb_learn", "comb_dep"))
            lines.append(
                f"θ 배포 {ab['theta_deployed']} → 학습 {[round(v, 3) for v in ab['theta_learned']]} "
                f"(표본 {ab['theta_train_samples']:,}, 정답 라벨 = **상한**)"
            )
            for tag, g in (("구현 차이", impl), ("★ θ 학습", learn)):
                if g:
                    lines.append(
                        f"  {tag:<9} {g['b_acc']:.4f} → {g['a_acc']:.4f}  {g['delta_pp']:+.2f}pp "
                        f"CI[{g['ci'][0]:+.2f},{g['ci'][1]:+.2f}] "
                        f"{'유의' if g['significant'] else '무의미'}"
                    )
            for g in ab.get("leaf_top1", []):
                if (g["a"], g["b"]) == ("comb_learn", "comb_dep"):
                    lines.append(
                        f"  ⚠️ 같은 θ, 리프 top-1: {g['delta_pp']:+.2f}pp "
                        f"CI[{g['ci'][0]:+.2f},{g['ci'][1]:+.2f}] — **표면에 따라 부호가 갈린다**"
                    )
        except (OSError, json.JSONDecodeError, KeyError) as e:
            lines.append(f"⬜ A/B 결과 파싱 실패: {e}")

    # ── 6d. 이미지 접기 경로 — 실행 실증 (2026-09-01) ────────────────────
    #
    # §6c 는 소스 시그니처를 본다. 이 절은 **실제 wasm 을 실행**해 동의 사다리·투영
    # 부재·리비전 불일치·철회 cascade 가 런타임에서 지켜지는지 본다 — 정적 게이트가
    # 못 보는 구멍이 실재함을 사보타주가 증명했기 때문이다(2026-08-31 perception E2E).
    # 프로브 자신의 검사 목록은 gates/fixtures/image_fold_probe.cjs 헤더가 정본.
    probe = ROOT / "gates" / "fixtures" / "image_fold_probe.cjs"
    node_binding = ROOT / "gates" / "fixtures" / "wasm-node" / "oicr_sdk_wasm.js"
    if not node_binding.exists():
        lines.append("⬜ 이미지 접기 프로브: wasm(node) 바인딩 없음 — dev.sh 가 만든다")
    else:
        r = run(["node", str(probe)])
        ok = sum(1 for ln in r.stdout.splitlines() if ln.startswith("OK"))
        if r.returncode == 0:
            lines.append(f"이미지 접기 경로: 프로브 {ok}건 전부 통과 (동의 사다리·리비전·철회 cascade)")
        else:
            tail = [ln for ln in r.stdout.splitlines() if ln.startswith("FAIL")][:4]
            fails.append("⛔ 이미지 접기 프로브 실패 — " + (" · ".join(tail) or "출력 없음"))

    lines.append(
        "⚠️ 이 수치의 근거는 **합성 데이터 하나**다. 실사용 로그로 재보기 전까지 외부에 "
        "인용하지 마라 — 이 게이트의 용도는 회귀 감지다."
    )

    if fails:
        return report(NAME, FAIL, lines + fails)
    return report(NAME, PASS, lines)


if __name__ == "__main__":
    sys.exit(main_guard(run_gate))
