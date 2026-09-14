#!/usr/bin/env python3
"""증류 코퍼스 수를 **읽기 전용으로 재계산해 원장에 못박는다**.

⛔ 왜 생겼나 (2026-09-15): 논문이 인용하는 26,294 · 2,933 · 4,101 · 1,396 · 10,748 이
   어느 JSON 원장에도 없었다. 전부 산문(README·스펙)에서 논문으로 건너온 값이고,
   그 계열에서 이미 네 건이 틀렸다(+17.09/+5.94pp 등). 계수 로직은 `esci_c5_prep.py`
   안에 있었지만 그 스크립트는 **학습 입력을 쓰는** 스크립트라 수치 확인 목적으로 돌릴 수 없다.
⇒ 같은 계수 로직을 복제하되 **아무것도 쓰지 않고** 수를 세어 `corpus_counts.json` 만 남긴다.

⚠️ 여기 나오는 수는 **현재 트리 기준**이고, 논문 §4 가 기록한 드리프트의 도착점이다
   (jp 4,101 → 4,093 · en 1,396 → 1,398). 상류 파이프라인이 시드를 안 받아서 생긴 것으로,
   드리프트 자체가 논문의 논지다 — 값이 다르다고 한쪽을 지우지 마라.
"""
from __future__ import annotations
import json, pathlib, random, sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DIST = ROOT / "experiments/distill-ko"
DEEP = ROOT / "experiments/event-intent/utterances_llm_deep.json"
SEED = 20260830
EN_CAP = 2000


def load_pass(path: pathlib.Path) -> list[dict]:
    """`esci_c5_prep.load_pass` 와 같은 게이트 — 매핑·번역 둘 다 pass 인 행만."""
    out = []
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("map_gate") == "pass" and r.get("tr_gate") == "pass" and r.get("query_ko"):
            out.append(r)
    return out


def add_count(rows, utts, hold, l1_of) -> tuple[int, int, int]:
    add = skip = dup = 0
    for r in rows:
        key = f"{l1_of.get((r['l2'], r['leaf']), '?')}|{r['l2']}|{r['leaf']}"
        if key in hold:
            skip += 1
            continue
        cur = utts.setdefault(key, [])
        if r["query_ko"] in cur:
            dup += 1
            continue
        cur.append(r["query_ko"])
        add += 1
    return add, skip, dup


def main() -> int:
    sys.path.insert(0, str(ROOT / "data"))
    from taxonomy_ko import TAXONOMY

    l1_of = {}
    for l1, mids in TAXONOMY.items():
        for l2, leaves in mids.items():
            for lf in leaves:
                l1_of[(l2, lf)] = l1
    hold = set(json.loads((DIST / "stage3_holdout_leaves.json").read_text(encoding="utf-8")))
    deep = json.loads(DEEP.read_text(encoding="utf-8"))

    jp_split = json.loads((HERE / "esci_split.json").read_text(encoding="utf-8"))
    jp = {r["example_id"]: r for r in load_pass(HERE / "esci_ko.jsonl")}
    en_all = load_pass(HERE / "esci_ko_us.jsonl")
    rng = random.Random(SEED)
    rng.shuffle(en_all)
    en_train_full = en_all[int(len(en_all) * 0.2):]

    jp_rows = [jp[i] for i in jp_split["train"] if i in jp]
    out = {
        "_what": "논문 §4 가 인용하는 증류 코퍼스 수 — 읽기 전용 재계산",
        "_caveat": "현재 트리 기준. 논문이 기록한 드리프트의 도착점이다(jp 4,101→4,093 · en 1,396→1,398).",
        "_inputs": {"deep": str(DEEP.relative_to(ROOT)),
                    "jp": "experiments/esci-ko/esci_ko.jsonl",
                    "en": "experiments/esci-ko/esci_ko_us.jsonl",
                    "holdout_leaves": "experiments/distill-ko/stage3_holdout_leaves.json"},
        "synthetic": {"leaves": len(deep), "utterances": sum(len(v) for v in deep.values())},
        "esci_gate_passed": {"jp": len(jp), "en_us": len(en_all)},
        "en_train_pool_after_holdout": len(en_train_full),
    }
    # ⛔ 팔마다 새 사본을 주면 안 된다. 실제 파이프라인(`esci_c5_prep.py`)은 jp 를 먼저
    #    같은 사전에 넣고 **그 뒤에** en 을 넣으므로, jp 가 이미 넣은 한국어 질의와 겹치는
    #    en 질의가 중복으로 걸러진다. 독립 실행하면 en 이 1,405 로 7건 많게 나온다 —
    #    파이프라인이 실제로 쓴 수는 1,398 이다. 순서가 곧 계약이다.
    for cap_tag, en_rows in (("en_capped", en_train_full[:EN_CAP]),
                             ("en_full", en_train_full)):
        u = {k: list(v) for k, v in deep.items()}
        a, s, d = add_count(jp_rows, u, hold, l1_of)
        out["added_jp"] = {"added": a, "skipped_holdout_leaf": s, "duplicate": d,
                           "n_rows": len(jp_rows)}
        a, s, d = add_count(en_rows, u, hold, l1_of)   # ⛔ jp 를 넣은 **같은** 사전에
        out[f"added_{cap_tag}"] = {"added": a, "skipped_holdout_leaf": s, "duplicate": d,
                                   "n_rows": len(en_rows),
                                   "_order": "jp 를 먼저 넣은 사전에 이어서 넣는다"}
        out[f"corpus_total_{cap_tag}"] = sum(len(v) for v in u.values())
        # 순서 효과를 수치로 남긴다 — 독립 계수는 파이프라인 값이 아니지만, 재계수가
        # 왜 어긋나는지 설명하는 유일한 숫자다.
        u2 = {k: list(v) for k, v in deep.items()}
        a2, _s2, _d2 = add_count(en_rows, u2, hold, l1_of)
        out[f"added_{cap_tag}_counted_independently"] = a2
    out["_en_cap"] = EN_CAP
    (HERE / "corpus_counts.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
