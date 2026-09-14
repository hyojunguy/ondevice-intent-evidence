#!/usr/bin/env python3
"""온디바이스 배포 경로 vs 서버 교사 — 같은 입력·같은 검색 크기로 나란히 놓는다.

⛔ 이 비교의 동기는 품질에서 나온다: 교사는 어휘후크 없는 구간에서 +17.37pp 앞선다
   (`real-ko-commerce/teacher_arm.json`). **그 17점의 값이 얼마인가**가 여기서 답할 것이다.
⛔ 단가는 쓰지 않는다 — GPU 시간당 단가 정본(llm-selfhost-calculator)이 이 머신에
   체크아웃돼 있지 않다. 근거 없는 달러를 지어내느니 지연·처리량·프라이버시 속성만 낸다.
⛔ 서버 지연에는 **네트워크 왕복이 빠져 있다.** 그러니 이것은 서버 경로의 하한이다.
"""
from __future__ import annotations
import json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main() -> int:
    dev = json.loads((ROOT / "artifacts" / "ui-facts.json").read_text(encoding="utf-8"))
    # ⛔ 여기서 다시 재지 않는다. 재측정하면 실행마다 값이 달라져(실측 2.37~3.05 ms) 논문이
    #    같은 수치를 두 개 갖게 된다 — 2026-09-15 에 실제로 그렇게 됐다. 못박은 원장만 읽는다.
    pin = ROOT / "artifacts" / "latency.json"
    if not pin.exists():
        sys.exit("⛔ artifacts/latency.json 이 없다 — `python3 gates/pin_latency.py` 를 먼저 돌려라")
    L = json.loads(pin.read_text(encoding="utf-8"))
    if L.get("_status", "").startswith("CONTAMINATED"):
        print(f"⚠️  온디바이스 원장이 오염 표시다: {L.get('_why','')[:80]}", file=sys.stderr)
    on = {"p50_ms": L["p50_ms"]["median"], "p95_ms": L["p95_ms"]["median"],
          "_pin_status": L.get("_status", "ok")}

    rows = []
    for f in sorted(HERE.glob("serve_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("status") != "measured":
            rows.append({"target": f.stem.replace("serve_", ""), "status": d.get("status"),
                         "reason": (d.get("reason") or "")[:120]})
            continue
        res = d["results"]
        rows.append({"target": f.stem.replace("serve_", ""), "status": "measured",
                     "node": d.get("node"), "gpu": res["_config"]["gpu"],
                     "dtype": res["_config"]["dtype"],
                     "single_p50_ms": res["single_stream"]["p50_ms"],
                     "single_p95_ms": res["single_stream"]["p95_ms"],
                     "peak_qps": res["peak_queries_per_sec"],
                     "reps": res["_config"]["reps"]})

    out = {"_what": "온디바이스 Tier-0 vs 서버 교사(768d/421MB) — 같은 512자 입력, 같은 6,020 리프 검색",
           "_caveat": ["서버 지연에 네트워크 왕복 미포함 — 서버 경로의 하한이다",
                       "서버 경로는 질의를 기기 밖으로 보낸다 — C3(원문 무유출)를 구조적으로 깬다",
                       "교사는 배포 경로가 아니다. 품질 이득은 teacher_arm.json 참조"],
           "on_device": {"path": "Tier-0 정적 임베딩 + 어휘 겹침 (배포 바이너리)",
                         "hardware": "개발 맥 (M-시리즈)", "payload_bytes": 2926804,
                         "p50_ms": on["p50_ms"], "p95_ms": on["p95_ms"],
                         "pin_status": on.get("_pin_status", "ok"),
                         "network_round_trips": 0},
           "server_teacher": rows}
    for r_ in rows:
        if r_.get("status") == "measured":
            r_["vs_on_device_p50"] = round(r_["single_p50_ms"] / on["p50_ms"], 2)
    (HERE / "compare.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
