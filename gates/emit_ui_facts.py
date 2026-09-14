#!/usr/bin/env python3
"""게이트가 잰 값을 화면이 읽을 수 있는 한 파일로 내보낸다 — `artifacts/ui-facts.json`.

⛔ 이 파일이 있는 이유: 근거 패널이 숫자를 하드코딩하면 게이트와 **조용히 갈라진다.**
   갈라진 순간 데모는 자기가 통과했다고 주장하는 것과 다른 숫자를 보여주게 되고,
   그 상태는 아무도 눈치채지 못한다. 그래서 숫자는 코드가 소유하고 화면은 읽기만 한다
   ([[sonnet-format-determinism]] 동형).

⛔ 여기서 계산을 새로 짜지 않는다. DP 는 `dp_gate` 의 함수를, 크기는 `size_gate` 와
   같은 파일 목록을 그대로 쓴다. 두 번째 구현은 두 번째 진실을 만든다.
"""
from __future__ import annotations
import json, math, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
assert (ROOT / "CONTRACT.md").exists(), f"ROOT 가 틀렸다: {ROOT}"
sys.path.insert(0, str(HERE))

from _common import BUDGETS, ROOT as _GATE_ROOT, gzip_bytes, walk_files  # noqa: E402
from dp_gate import campaign_epsilon, eps_from_rho, rho_gaussian  # noqa: E402

assert _GATE_ROOT == ROOT, "게이트와 다른 ROOT 를 보고 있다"


def size_facts() -> dict:
    """크기 게이트와 **같은 목록·같은 측정 함수**를 쓴다. 갈리면 합계가 갈린다.

    ⛔ `web/demo/dist` 는 디렉터리다 — 파일만 가정하면 조용히 0 이 된다. 그래서
       게이트의 `walk_files`/`gzip_bytes` 를 그대로 부른다.
    """
    cfg = BUDGETS["size"]
    budget = int(cfg["limit_bytes"])

    # ⛔ 학습되지 않은 자리채움 아티팩트를 **학습본과 같은 모양으로** 보여주지 않는다.
    #    projection.bin 은 지금 항등 행렬이라 raw 는 학습본과 같은 4,116B 인데 gzip 은
    #    0 이 많아 76B 로 눌린다 — 합계가 그만큼 낙관적이다(5.24MB 중 ~4KB, 0.08%).
    #    작은 오차지만 **예산이 막으려는 방향의 오차**이므로 화면이 표기하게 한다.
    prov = {}
    mf = ROOT / "artifacts" / "MANIFEST.json"
    if mf.exists():
        try:
            prov = json.loads(mf.read_text(encoding="utf-8")).get("_provenance", {})
        except Exception:
            prov = {}
    placeholder_note = str(prov.get("projection", ""))

    items, total = [], 0
    for rel in cfg["artifacts"]:
        target = ROOT / rel
        if not target.exists():
            items.append({"path": rel, "gzipBytes": None, "missing": True, "placeholder": None})
            continue
        gz = sum(gzip_bytes(f) for f in walk_files(target))
        total += gz
        note = placeholder_note if rel.endswith("projection.bin") and placeholder_note.startswith("IDENTITY") else None
        items.append({"path": rel, "gzipBytes": gz, "missing": False, "placeholder": note})
    # ⛔ 화면이 말해야 하는 것은 **제품 약속**(L1 데이터 + L2 코드 <= 5 MiB)이지
    #    L1 혼자의 사용률이 아니다. 예산을 라인으로 쪼갠 뒤 화면이 L1 만 보여주면
    #    5MB 약속보다 **약한 것**을 주장하게 된다(2026-08-28).
    code_path = ROOT / "artifacts" / "code-size.json"
    code_bytes = None
    if code_path.exists():
        try:
            code_bytes = int(json.loads(code_path.read_text(encoding="utf-8"))["worstBytes"])
        except Exception:
            code_bytes = None
    promise_limit = int(BUDGETS["total"]["limit_bytes"])
    promise_total = (total + code_bytes) if code_bytes is not None else None

    return {
        "items": items,
        "codeBytes": code_bytes,
        "codeMeasured": code_bytes is not None,
        "promiseLimitBytes": promise_limit,
        "promiseTotalBytes": promise_total,
        "promiseUsedPct": round(promise_total / promise_limit * 100, 1) if promise_total else None,
        "totalGzipBytes": total,
        "budgetBytes": budget,
        "usedPct": round(100.0 * total / budget, 1) if budget else None,
    }


def dp_facts() -> dict:
    fl = BUDGETS["fl"]
    clip, sigma = float(fl["clip_norm"]), float(fl["noise_sigma"])
    delta, target = float(fl["delta"]), float(fl["target_epsilon_per_round"])
    n_assumed = int(fl.get("assumed_participants", fl["min_participants"]))

    def eps_at(n: int) -> float:
        return eps_from_rho(rho_gaussian(clip, sigma * math.sqrt(n)), delta)

    required = next((n for n in range(1, 1_000_001) if eps_at(n) <= target), None)
    # 화면이 곡선을 그릴 수 있게 몇 점만 준다(전 구간을 보내면 그냥 무겁다).
    curve = [{"n": n, "epsilon": round(eps_at(n), 4)}
             for n in (1, 10, 50, 100, 204, 500, 1_000, 5_000, 10_000)]
    # ── 캠페인 전체: 사람의 프라이버시에 실제로 해당하는 값 ──
    rounds = int(fl["max_rounds"])
    population = int(fl.get("device_population", n_assumed))
    target_campaign = float(fl.get("target_epsilon_campaign", 0.0))
    z_eff = sigma * math.sqrt(n_assumed) / clip
    q = min(1.0, n_assumed / population) if population > 0 else 1.0
    eps_naive = eps_from_rho(rho_gaussian(clip, sigma * math.sqrt(n_assumed)) * rounds, delta)
    eps_camp = campaign_epsilon(z_eff, q, rounds, delta)

    # 목표를 만족하는 최소 기기 인구 — "204명"의 캠페인판 제품 요구사항.
    required_pop = None
    if eps_camp is not None:
        for pop in (n_assumed, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000):
            if pop < n_assumed:
                continue
            e = campaign_epsilon(z_eff, min(1.0, n_assumed / pop), rounds, delta)
            if e is not None and e <= target_campaign:
                required_pop = pop
                break

    return {
        "clipNorm": clip,
        "noiseSigma": sigma,
        "delta": delta,
        "targetEpsilonPerRound": target,
        "assumedParticipants": n_assumed,
        "epsilonAtAssumed": round(eps_at(n_assumed), 4),
        "requiredParticipants": required,
        "maxRounds": int(fl["max_rounds"]),
        "curve": curve,
        "campaign": {
            "rounds": rounds,
            "devicePopulation": population,
            "samplingRate": round(q, 4),
            "epsilonNaive": round(eps_naive, 2),
            "epsilonSubsampled": round(eps_camp, 2) if eps_camp is not None else None,
            "targetEpsilon": target_campaign,
            "requiredPopulation": required_pop,
            "conditions": "증폭 성립 조건: 참여 무작위 · SecAgg 성립 · Poisson 샘플링 가정."
                          " 못 지키면 증폭 없음 값을 씁니다.",
            "reference": "Gboard 실배포 캠페인 전체 ε 0.99~13.7 (δ=1e-10)",
        },
        # ⛔ 화면이 이 캐비엇을 지우지 못하게 데이터에 실어 보낸다.
        "caveat": "위 ε 는 라운드당입니다. 사람의 프라이버시에 해당하는 값은 캠페인 전체이고,"
                  " 아래 캠페인 항목이 그 값입니다.",
    }


def fl_facts() -> dict:
    fl = BUDGETS["fl"]
    return {
        "minParticipants": int(fl["min_participants"]),
        "deadlineMs": int(fl["deadline_ms"]),
        "maxRounds": int(fl["max_rounds"]),
    }


def escalation_facts() -> dict | None:
    """S5 승격 임계 — `experiments/intent-bench/calibrate_escalation.py` 가 홀드아웃에서 잰 값.

    ⛔ 없으면 None 을 돌려준다. 화면은 그때 승격을 **근거로 쓰지 않는다** — 임계를 지어내
       보여주는 것보다 "아직 안 쟀다"가 낫다.
    """
    f = ROOT / "experiments" / "intent-bench" / "escalation_calibration.json"
    if not f.exists():
        return None
    c = json.loads(f.read_text(encoding="utf-8"))
    chosen = c.get("chosen")
    if not chosen:
        return None
    return {
        "signal": c["signal"],
        "marginThreshold": chosen["threshold"],
        "escalatedPct": chosen["escalatedPct"],
        "errorRecallPct": chosen["errorRecallPct"],
        "liftVsRandom": c.get("lift_vs_random"),
        # ⛔ 이 값은 **옛 60-way 합성 의도헤드** 수치다. 지금 배포본의 리프 정확도가
        #    아니다 — 화면이 둘을 같은 것으로 보여주면 안 된다(`model` 블록을 쓴다).
        "tier0Accuracy": c.get("tier0_accuracy"),
        "tier0AccuracyScope": "옛 60-way 합성 의도헤드 — 현재 배포본의 리프 정확도가 아니다",
        "verdict": c.get("verdict"),
        "caveat": c.get("caveat"),
        "synthetic": bool(c.get("SYNTHETIC")),
    }


def size_vs_teacher() -> dict | None:
    """"우리 모델 N MB · 교사보다 K배 작다" 를 **코드가 센다**.

    ⛔ 예전에는 QualityPanel 이 "이 2.1MB 모델" 과 "220배 작습니다" 를 각각 하드코딩했고,
       생성기도 같은 사실을 `teacherNote` 산문으로 **따로** 들고 있었다 — 같은 사실의 사본
       셋. 실제로 갈라져 있었다: 2.1MB 는 맞았지만 배율은 220 이 아니라 **197** 이었다
       (테이블이 커지는 동안 아무도 배율을 다시 안 쟀다). 화면이 배율을 말하려면 그 배율은
       파일에서 나와야 한다.

    분모는 `embedding.bin` 의 **raw 바이트**다 — 교사 421MB 가 on-disk 체크포인트 크기라
    같은 축으로 비교해야 한다. gzip(전송)이나 L1 합계(택소노미·카탈로그 포함)를 쓰면
    교사가 갖지 않은 것까지 세게 되어 배율이 흐려진다.
    """
    emb = ROOT / "artifacts" / "embedding.bin"
    if not emb.exists():
        return None
    ours = emb.stat().st_size
    # ko-sroberta-multitask(jhgan) 체크포인트 — 외부 고정 사실이라 파일에서 못 읽는다.
    teacher = 421_000_000
    return {
        "oursBytes": ours,
        "teacherBytes": teacher,
        "teacherLabel": "ko-sroberta-multitask",
        "ratio": round(teacher / ours, 1),
        "basis": "embedding.bin raw (on-disk) vs 교사 체크포인트 — 같은 축",
    }


def model_facts() -> dict | None:
    """Tier 0 모델이 **얼마나 맞히는가** — 화면이 한 번도 답하지 않던 질문.

    ⛔ 여기서 숫자를 지어내지 않는다. 전부 커밋된 결과 JSON 에서 읽는다:
       `promote_gate_e10.json`(승격 게이트) · `teacher_holdout_baseline.json`(교사 기준선) ·
       `lexical_w_sweep.json`(배포 경로 = 밀집 + 어휘 겹침) · `deployed_run.json`(무엇이
       배포됐나) · `utterances_llm.json`(커버리지).
    ⛔ 하나라도 없으면 그 항목은 None 이다. 못 잰 것을 채우지 않는다.
    """
    D = ROOT / "experiments" / "distill-ko"

    def load(rel):
        f = D / rel
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

    teacher, run = load("teacher_holdout_baseline.json"), load("deployed_run.json")
    # ⛔ 게이트 파일명을 박지 마라. 2026-08-29 에 `promote_gate_e10.json` 이 박혀 있었고,
    #    그 다음 배포(stage3deep)가 오자 화면이 **이전 배포본의 승격 델타**를 현재 값으로
    #    보여줄 뻔했다 — `lexical_w_sweep` 에서 이미 한 번 고친 결함과 같은 계열이다.
    #    정본은 `deployed_run.json::gate` 이고, 아래에서 후보 모델이 MANIFEST 와 같은지
    #    한 번 더 대조한다.
    # ⛔ gate 경로는 distill-ko 밖일 수 있다(stage4esci 는 experiments/esci-ko 에 산다).
    #    파일명만 잘라 D/ 에서 찾으면 gate=None → model_facts 전체가 None → 화면이
    #    UNMEASURED 로 떨어진다 — 승격 직후 실제로 그 상태였다(2026-08-30).
    _gate_rel = (run or {}).get("gate", "")
    if _gate_rel and "/" in _gate_rel:
        _gf = ROOT / _gate_rel
        gate = json.loads(_gf.read_text(encoding="utf-8")) if _gf.exists() else None
    else:
        gate = load(_gate_rel or "promote_gate_e10.json")
    lex = load("lexical_w_sweep.json")
    surf = load("product_surface.json")
    if not gate or not run:
        return None
    ax = gate.get("axes", {})
    cand = gate.get("candidate", {}).get("label")

    def pair(axis):
        a = ax.get(axis)
        if not a or cand not in a:
            return None
        # ⛔ 필드 이름을 `stage2` 로 두지 마라. 기준선은 게이트마다 다르다 —
        #    stage3deep 의 기준선은 stage2 가 아니라 **이전 배포본**이다.
        return {"deployed": a[cand], "base": a.get(gate.get("base", {}).get("label")),
                "deltaPp": a.get("delta"), "ci": a.get("ci"), "significant": a.get("significant")}

    utt = ROOT / "experiments/event-intent/utterances_llm.json"
    covered = len(json.loads(utt.read_text(encoding="utf-8"))) if utt.exists() else None
    # 배포 경로(밀집 + 어휘 겹침 w=0.20)의 실측 — 밀집만 잰 게이트 숫자와 **다른 값**이다.
    deploy_path = None
    # ⛔ 스윕이 **현재 배포본**에서 잰 것인지 확인한다. 아니면 통째로 버린다 —
    #    이전 배포본의 숫자를 화면이 현재 값으로 보여주는 것이 정확히 우리가 고친 결함이다.
    _src = json.loads((ROOT / "artifacts/MANIFEST.json").read_text(encoding="utf-8"))["_provenance"]["source_model"]
    if lex and lex.get("deployed_model") != _src:
        lex = None
    # 제품 표면(중분류 · top-5)도 같은 규율을 받는다 — 이전 배포본에서 잰 표면 숫자를
    # 현재 값으로 보여주면 화면이 거짓말한다.
    if surf and surf.get("deployed_model") != _src:
        surf = None
    # 게이트의 **후보**가 곧 배포본이어야 한다. 아니면 델타 블록을 통째로 버린다.
    if (gate.get("candidate") or {}).get("model", "").rsplit("/", 1)[-1] \
            != _src.rsplit("/", 1)[-1]:
        gate = {}
    if not gate:
        return None
    if lex:
        d = lex.get("distributions", {})
        cur = lex.get("current")
        def at(name):
            rows = (d.get(name) or {}).get("rows") or []
            r = next((x for x in rows if x.get("w") == cur), None)
            return {"top1": r["top1"], "top5": r["top5"], "n": (d.get(name) or {}).get("n")} if r else None
        deploy_path = {"lexicalWeight": cur, "llmHoldout": at("llm-holdout"),
                       "template": at("template(리프명 포함)"), "material": at("catalog-material")}

    # ── 제품 표면 ──────────────────────────────────────────────────────
    #
    # ⛔ 2026-08-29 사용자 결정: 화면은 **중분류(L2)까지, top-5** 로 보여준다. 같은
    #    모델이 표면에 따라 26.4% ↔ 78.8% 를 오가므로, 표면을 정했으면 **그 표면의
    #    숫자**를 화면이 말해야 한다. 소분류 top-1 은 지우지 않고 **하한**으로 남긴다 —
    #    지우면 "우리가 더 잘하는 축만 보여준다"가 된다.
    product_surface = None
    if surf:
        lv = surf.get("levels") or {}
        rows = []
        for name, dd in (surf.get("distributions") or {}).items():
            def cell(k):
                v = dd.get(k)
                return {"acc": v["acc"], "ci": v["ci"]} if v else None
            rows.append({"name": name, "n": dd.get("n"),
                         "surfaceTop1": cell("중분류_top1"), "surfaceTop5": cell("중분류_top5"),
                         "leafTop1": cell("소분류_top1"), "leafTop5": cell("소분류_top5"),
                         "l1Top1": cell("대분류_top1")})
        product_surface = {
            "surface": surf.get("surface"), "level": "l2", "k": 5,
            "levels": lv, "note": surf.get("note"), "distributions": rows,
            "leafNote": "소분류 top-1 은 제품 지표가 아니다 — 표면 결정 전의 값이고, "
                        "여기서는 하한으로만 남긴다.",
        }

    # ── 실질의 검증 (esci_ko) ──────────────────────────────────────────
    #
    # 데모가 지금까지 보여준 정확도는 전부 합성 발화 위였다. 이 블록이 처음으로
    # **실사용 질의**(아마존 jp → 한국어 번역, tasksource/esci)의 값을 화면에 올린다.
    # ⛔ 캐비엇을 숫자와 분리하지 마라 — 화면은 이 블록의 caveats 를 함께 렌더해야 한다.
    E = ROOT / "experiments" / "esci-ko"

    def loadE(name):
        f = E / name
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

    real_query = None
    c1 = loadE("esci_eval_results.json")
    if c1 and c1.get("deployed_model") != _src:
        c1 = None  # 드리프트 가드 — 이전 배포본에서 잰 실질의 숫자를 현재 값으로 보여주지 않는다
    # ⛔ 오염 가드: 배포본은 esci train 80% 를 **학습했다**. 전체셋 평가는 학습 질의가
    #    섞여 부풀고(실측: 전체 0.2905 vs 홀드아웃 0.2348), 그 값을 화면에 올리면
    #    암기를 성능으로 파는 것이다. 홀드아웃 pairs 로 잰 결과만 통과한다.
    if c1 and "holdout" not in str(c1.get("pairs_file", "")):
        c1 = None
    if c1:
        rows = {r["model"]: r for r in c1.get("rows", [])}
        ours = rows.get("우리 128d@4bit")
        teacher_r = rows.get("교사 ko-sroberta 421MB")
        # 승격 근거 델타 — 정본은 deployed_run.json::holdout_judge 가 가리키는 판정 파일.
        # ⛔ 파일명을 여기 박지 않는다(stage4 때 그렇게 했다가 다음 승격에서 조용히 죽었다).
        #    판정 파일이 cand 모델 경로를 기록하므로 배포본과 대조해 드리프트를 막는다.
        delta = None
        judge_rel = run.get("holdout_judge")
        if judge_rel:
            jf = ROOT / judge_rel
            hold = json.loads(jf.read_text(encoding="utf-8")) if jf.exists() else None
            cand_path = str(((hold or {}).get("cand") or ["", ""])[1])
            if hold and cand_path.rsplit("/", 1)[-1] == _src.rsplit("/", 1)[-1]:
                boots = hold.get("bootstrap") or {}
                # c5 형식은 {세트: {top1,...}} — 근거 세트는 판정 파일이 명시한다.
                ev = hold.get("evidence_set")
                b = boots.get(ev, {}) if ev else boots
                n_ev = next((r.get("n") for r in hold.get("rows", [])
                             if not ev or r.get("set") == ev), None)
                if b.get("top1"):
                    delta = {"top1": b.get("top1"), "top5": b.get("top5"),
                             "surface": b.get("surf"), "holdoutN": n_ev,
                             "vs": hold.get("vs") or "이전 배포본"}
        if ours:
            real_query = {
                "source": "아마존 실사용 검색질의 (tasksource/esci · jp → 한국어 번역)",
                "n": ours.get("n"),
                "ours": {"top1": ours.get("top1"), "top5": ours.get("top5"),
                         "surfaceTop5": ours.get("surface_l2_top5")},
                "teacher": ({"top1": teacher_r.get("top1"),
                             "surfaceTop5": teacher_r.get("surface_l2_top5")}
                            if teacher_r else None),
                "top1DiffVsTeacher": c1.get("bootstrap"),
                "promotionDelta": delta,
                "caveats": [
                    "이 질의들은 한국 사용자 분포가 아니라 아마존 사용자 분포의 한국어판입니다.",
                    "라벨은 Qwen 27B 만장일치 매핑입니다 — 인간 라벨이 아니고, 잡음 상한은 검증 일치율 ~89% 입니다.",
                ],
            }

    return {
        "deployed": run.get("deployed"),
        "productSurface": product_surface,
        "realQuery": real_query,
        "gateBaseLabel": (gate.get("base") or {}).get("label"),
        "gateCandidateLabel": cand,
        "stage": "distill → tokenlearn → 지도 대조학습 (3단계)",
        "leaves": 2960,
        "leavesWithUtterances": covered,
        "quantization": gate.get("quantization"),
        "leafRepresentation": gate.get("leafRepresentation"),
        "holdoutLeaves": (gate.get("provenance") or {}).get("holdout_leaves"),
        "denseOnly": {"leafTop1": pair("leaf_holdout_top1"), "leafTop5": pair("leaf_holdout_top5"),
                      "l1Top1": pair("leaf_holdout_l1"), "material": pair("taxo_material_top1"),
                      "template": pair("template_leaf_top1")},
        "deploymentPath": deploy_path,
        "teacherHoldoutTop1": (teacher or {}).get("teacher_holdout_top1"),
        "sizeVsTeacher": size_vs_teacher(),
        "realDataAxes": {k: pair(k) for k in ("klue_sts_rho", "ynat_acc", "nsmc_acc")},
        "gateDecision": gate.get("decision"),
        "caveats": [
            "학습·평가가 같은 생성기의 합성 발화다. 실사용 질의로 재검증 전에는 제품 성능이 아니다.",
            "밀집 코사인만 잰 값과 배포 경로(밀집+어휘 겹침) 값은 다르다 — 섞어 쓰지 마라.",
            "리프명이 질의에 든 발화는 평가에서 설계상 배제된다. 그래서 이 top-1 은 하한이다.",
        ],
    }


def egress_facts() -> dict:
    """C3 을 화면이 **잰 값**으로 말할 수 있게 한다.

    ⛔ 히어로는 오래 "원문 전송 0 바이트"를 실측처럼 보여 줬는데 그건 설계 불변식이지
       측정이 아니었다. 게이트가 실제로 재는 것은 다른 것이다 — 발화에 심어 둔 카나리아
       문자열이 나가는 바이트에서 **몇 건 검출되는가**. 지어낸 0 과 잰 0 은 화면에서
       똑같이 보이지만 같은 것이 아니다.

    표본이 없으면 `None` — 못 잰 것을 0 으로 채우지 않는다.
    """
    dump = ROOT / "artifacts" / "egress_sample.json"
    if not dump.exists():
        return {"measured": False}
    raw = dump.read_text(encoding="utf-8")
    low = raw.lower()
    from egress_gate import CANARIES

    leaked = [c for c in CANARIES if c.lower() in low]
    return {
        "measured": True,
        "probes": len(CANARIES),
        "detected": len(leaked),
        "payloadBytes": len(raw.encode()),
        "allowedFields": len(BUDGETS["egress"]["allowed_fields"]),
    }


def _assert_plain_text(obj) -> None:
    """화면(QualityPanel 등)은 ui-facts 문자열을 **평문으로** 렌더한다 — 마크다운 강조를
    넣으면 별표가 사람 눈에 그대로 노출된다(2026-08-30 화면 순회 실측: "**제품 지표가
    아니다**" 가 원문으로 보였다). 방출 시점에 코드로 막는다."""
    if isinstance(obj, str):
        assert "**" not in obj, f"ui-facts 문자열에 마크다운 별표: {obj[:60]!r}"
    elif isinstance(obj, dict):
        for v in obj.values():
            _assert_plain_text(v)
    elif isinstance(obj, list):
        for v in obj:
            _assert_plain_text(v)


def main() -> int:
    facts = {
        "generatedBy": "gates/emit_ui_facts.py",
        "size": size_facts(),
        "dp": dp_facts(),
        "fl": fl_facts(),
        "escalation": escalation_facts(),
        "model": model_facts(),
        "egress": egress_facts(),
        "latencyBudgetMs": BUDGETS.get("latency", {}).get("p95_ms"),
    }
    out = ROOT / "artifacts" / "ui-facts.json"
    out.write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{out} — 크기 {facts['size']['totalGzipBytes']:,}B "
          f"({facts['size']['usedPct']}%) · ε@{facts['dp']['assumedParticipants']:,} = "
          f"{facts['dp']['epsilonAtAssumed']} · 최소 {facts['dp']['requiredParticipants']}명")
    return 0


# ── 데모가 쓰는 **프리셋 리프**를 이름으로 풀어 준다 (2026-09-06(41)).
# ⛔ `ReadingPanel.tsx` 가 리프 인덱스를 **손으로 박고 있었다**(hintLeaf: 132 …).
#    순서가 곧 id 라 택소노미가 커지면 그 숫자가 조용히 남을 가리킨다 — 실측: 리프
#    2,960→4,949 확장에서 132(러닝화)가 `패션의류›패션소품›가발` 이 됐고, 이력 기반
#    광고 E2E 2건이 빈 슬롯으로 깨졌다. `behavior_leaf.json` 의 cat_hint 와 같은 계열이다.
def _emit_preset_leaves() -> None:
    import pathlib as _p, sys as _s
    root = _p.Path(__file__).resolve().parents[1]
    _s.path.insert(0, str(root / "data"))
    from taxonomy_ko import TAXONOMY  # noqa: PLC0415
    _paths = [f"{a}|{b}|{c}" for a, m in TAXONOMY.items()
              for b, fs in m.items() for c in fs]
    flat = {p_: i for i, p_ in enumerate(_paths)}
    presets = {"run": "신발가방|스포츠화|러닝화",
               "baby": "출산육아|유모차카시트|신생아카시트",
               "home": "가전|청소가전|로봇청소기"}
    out = {}
    for k, path in presets.items():
        if path not in flat:
            raise SystemExit(f"⛔ 프리셋 리프가 택소노미에 없다: {path}")
        out[k] = flat[path]
    dst = root / "web/demo/src/generated/presetLeaves.ts"
    dst.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f'  {k}: {v},  // {presets[k]}\n' for k, v in out.items())
    dst.write_text(
        "// ⛔ 생성물이다 — gates/emit_ui_facts.py 가 쓴다. 손으로 고치지 마라.\n"
        "// 리프 인덱스는 택소노미 순서에서 나오므로 리프가 늘면 바뀐다.\n"
        "export const PRESET_LEAVES: Record<string, number> = {\n" + body + "};\n",
        encoding="utf-8")
    print(f"✔ web/demo/src/generated/presetLeaves.ts  {out}")


if __name__ == "__main__":
    _emit_preset_leaves()
    raise SystemExit(main())
