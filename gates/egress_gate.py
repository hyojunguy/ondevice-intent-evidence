#!/usr/bin/env python3
"""C3 — 기기를 떠나는 바이트에 원본이 없는가.

판정 방법: fl-client 가 만든 실제 업로드 페이로드를 받아
(1) 최상위 키가 허용목록과 정확히 일치하는지
(2) 페이로드 어디에도 시드 원문 토큰이 나타나지 않는지
(3) 라운드 상한 바이트를 넘지 않는지
를 본다. 모델이나 문서의 주장이 아니라 **직렬화된 바이트**를 본다.
"""
from __future__ import annotations
import json, sys
from _common import BUDGETS, ROOT, PASS, FAIL, UNMEASURED, run, report, main_guard

# 페이로드에 이 문자열이 새어나오면 즉시 실패. 데모/테스트가 쓰는 시드 문장에서 따온다.
CANARIES = ["아이폰", "케이스", "iphone", "divorce", "임신", "salary", "비밀번호"]


def run_gate() -> int:
    dump = ROOT / "artifacts" / "egress_sample.json"
    if not dump.exists():
        r = run(["cargo", "run", "--quiet", "--release", "-p", "plataid-fl-client",
                 "--example", "dump_upload"])
        if r.returncode != 0 or not dump.exists():
            return report("C3 egress", UNMEASURED, [
                "업로드 표본을 못 얻었다 (fl-client example `dump_upload` 필요)",
                (r.stderr or "").strip().splitlines()[-1] if r.stderr.strip() else "",
            ])

    raw = dump.read_text(encoding="utf-8")
    payload = json.loads(raw)
    allowed = set(BUDGETS["egress"]["allowed_fields"])
    got = set(payload.keys())
    limit = BUDGETS["egress"]["max_round_bytes"]
    size = len(raw.encode())

    lines, code = [], PASS
    extra, absent = got - allowed, allowed - got
    lines.append(f"최상위 키 {len(got)}개")
    if extra:
        code = FAIL
        lines.append(f"⛔ 허용목록 밖 키: {sorted(extra)}")
    if absent:
        lines.append(f"(허용목록 중 미사용: {sorted(absent)})")

    low = raw.lower()
    leaked = [c for c in CANARIES if c.lower() in low]
    if leaked:
        code = FAIL
        lines.append(f"⛔ 원문 카나리아 유출: {leaked}")
    else:
        lines.append(f"카나리아 {len(CANARIES)}개 전부 미검출")

    lines.append(f"페이로드 {size:,} / 상한 {limit:,} 바이트")
    if size > limit:
        code = FAIL
        lines.append("⛔ 라운드 전송 상한 초과")

    rtb_code, rtb_lines = check_bid_request()
    code = max(code, rtb_code) if rtb_code != UNMEASURED else (rtb_code if code == PASS else code)
    lines.extend(rtb_lines)

    sup_code, sup_lines = check_purchase_suppression()
    code = max(code, sup_code) if sup_code != UNMEASURED else (sup_code if code == PASS else code)
    lines.extend(sup_lines)
    return report("C3 egress", code, lines)


def check_purchase_suppression() -> tuple[int, list[str]]:
    """C-P3(v2 P2) — 최근구매 억제 실행 실증. 실제 wasm 을 태워 (양성대조 → 억제 →
    노트 → 시계전진 복귀 → 동의 게이트 → 철회 cascade) 사다리를 판정한다.
    검사 목록의 정본은 gates/fixtures/purchase_suppress_probe.cjs 헤더."""
    probe = ROOT / "gates" / "fixtures" / "purchase_suppress_probe.cjs"
    binding = ROOT / "gates" / "fixtures" / "wasm-node" / "plataid_sdk_wasm.js"
    if not binding.exists():
        return UNMEASURED, ["⬜ 구매 억제 프로브: wasm(node) 바인딩 없음 — dev.sh 가 만든다"]
    r = run(["node", str(probe)])
    ok = sum(1 for ln in r.stdout.splitlines() if ln.startswith("OK"))
    if r.returncode == 0:
        return PASS, [f"구매 억제(C-P3): 프로브 {ok}건 전부 통과 (억제→노트→복귀→동의→철회)"]
    tail = [ln for ln in r.stdout.splitlines() if ln.startswith("FAIL")][:4]
    return FAIL, ["⛔ 구매 억제 프로브 실패 — " + (" · ".join(tail) or (r.stderr or "").strip()[:200])]


def _taxonomy_labels() -> list[str]:
    """`data/taxonomy_ko.py` 의 L1·L2 라벨 — 아티팩트가 없는 클론에서도 항상 있다."""
    sys.path.insert(0, str(ROOT / "data"))
    from taxonomy_ko import TAXONOMY  # noqa: E402
    return [d for d in TAXONOMY] + [m for d in TAXONOMY.values() for m in d]


def check_bid_request() -> tuple[int, list[str]]:
    """C14 — OpenRTB `BidRequest` 바이트(I4). 표본은 `ad-rtb` 예제가 **실제 sdk-core 분류**로
    만든다. 여기서 보는 것: (1) 최상위·user·device 키가 허용목록과 정확히 같은가 (2) RTB
    카나리아가 바이트에 없는가 (3) 크기 (4) intent 길이 (5) 카나리아가 라벨의 부분 문자열이
    아닌가 — (5)가 없으면 정상 요청이 (2)에서 FAIL 한다(2026-09-01: C3 의 `임신` ⊂ `임신출산`)."""
    rtb = BUDGETS.get("rtb")
    if not rtb:
        return UNMEASURED, ["OpenRTB: budgets.json 에 rtb 블록 없음"]
    dump = ROOT / "artifacts" / "egress_bid_request.json"
    if not dump.exists():
        r = run(["cargo", "run", "--quiet", "--release", "-p", "plataid-ad-rtb",
                 "--example", "dump_bid_request"])
        if r.returncode != 0 or not dump.exists():
            tail = (r.stderr or "").strip().splitlines()
            return UNMEASURED, ["OpenRTB: 요청 표본을 못 얻었다 (ad-rtb example `dump_bid_request`)",
                                tail[-1] if tail else ""]
    raw = dump.read_text(encoding="utf-8")
    req = json.loads(raw)
    lines, code = [], PASS

    labels = [l.lower() for l in _taxonomy_labels()]
    collide = {c: [l for l in labels if c.lower() in l] for c in rtb["canaries"]}
    collide = {c: ls for c, ls in collide.items() if ls}
    if collide:
        code = FAIL
        lines.append(f"⛔ OpenRTB 카나리아가 라벨과 겹친다(정상 요청이 FAIL 한다): {collide}")

    def keyset(name: str, obj: dict, allowed_key: str) -> None:
        nonlocal code
        got, allowed = set(obj.keys()), set(rtb[allowed_key])
        extra = got - allowed
        if extra:
            code = FAIL
            lines.append(f"⛔ OpenRTB {name} 허용목록 밖 키: {sorted(extra)}")
    keyset("최상위", req, "request_allowed_top")
    keyset("user", req.get("user", {}), "user_allowed")
    keyset("device", req.get("device", {}), "device_allowed")

    low = raw.lower()
    leaked = [c for c in rtb["canaries"] if c.lower() in low]
    if leaked:
        code = FAIL
        lines.append(f"⛔ OpenRTB 요청에 원문 카나리아: {leaked}")
    intent = req.get("user", {}).get("ext", {}).get("plataid", {}).get("intent", [])
    if len(intent) > rtb["max_intent_k"]:
        code = FAIL
        lines.append(f"⛔ OpenRTB intent {len(intent)}개 > max_intent_k {rtb['max_intent_k']}")
    if not intent and not req.get("user", {}).get("ext", {}).get("plataid", {}).get("floor_hit"):
        code = FAIL
        lines.append("⛔ OpenRTB 표본에 intent 가 없다 — 검사할 라벨이 없는 표본은 표본이 아니다")
    if req.get("device", {}).get("lmt") != 1:
        code = FAIL
        lines.append("⛔ OpenRTB device.lmt ≠ 1")
    size = len(raw.encode())
    if size > rtb["max_request_bytes"]:
        code = FAIL
        lines.append(f"⛔ OpenRTB 요청 {size:,} > 상한 {rtb['max_request_bytes']:,} 바이트")
    lines.append(f"OpenRTB {rtb['version']} 요청 {size:,} 바이트 · intent {len(intent)}개(≤{rtb['max_intent_k']}) · "
                 f"카나리아 {len(rtb['canaries'])}개 미검출 · user 키 {sorted(req.get('user', {}).keys())} · "
                 f"device 키 {sorted(req.get('device', {}).keys())}")
    return code, lines


# ⛔ 최상위에서 부르면 이 모듈을 **import 할 수 없다** — import 하는 순간 게이트가
#    돌고 sys.exit 로 호출자를 죽인다. 게이트를 재사용하려던 코드가 여기서 잘렸다.
if __name__ == "__main__":
    main_guard(run_gate)
