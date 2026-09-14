#!/usr/bin/env python3
"""C14 taxonomy-hygiene — **리프가 자기 중분류로 가는가**를 상시 감시한다.

# 왜

리프 이름은 계속 늘어난다(2,960 → 5,628, 2026-09-06). 늘릴 때마다 **남의 중분류의 질의를
빨아들이는 이름**이 섞여 들어올 수 있고, 그건 (36)이 rename 7건으로 고친 결함을 규모만
키워서 다시 만드는 일이다.

이 게이트가 재는 것: **기존 리프 전부**를 질의처럼 배포 경로에 태워, 지금 택소노미가 그
이름을 **자기 중분류로** 보내는 비율. 2026-09-06 실측 기준선은 **98.3%**(2,960 리프)였고,
떨어지는 50개는 그때 센 **동명 리프 49건과 같은 집합**이라 잡음이 아니라 알려진 목록이다.

⛔ 하한(`FLOOR`)을 낮춰서 통과시키지 마라 — 그건 회귀를 기록에서 지우는 것이다.
   떨어졌으면 새로 들어온 이름 중 남의 자리에 앉은 것을 찾아 고친다.

⛔ 이 게이트는 `scripts/expansion_l2_gate.py --control` 과 **같은 계산**이다. 거기는 확장
   후보를 거를 때 쓰는 대조군이고, 여기는 그 대조군 자체를 상시 감시선으로 쓴다.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
FLOOR = 0.97          # 2026-09-06 기준선 98.3% 에서 1.3pp 여유. ⛔ 낮추지 마라.


def main() -> int:
    import numpy as np
    sys.path.insert(0, str(ROOT / "data"))
    sys.path.insert(0, str(ROOT / "experiments/distill-ko"))
    from taxonomy_ko import TAXONOMY
    import leaf_repr

    names, l1s, l2_list, l1_of, l2_of = leaf_repr.flatten(TAXONOMY)
    l2_of = np.asarray(l2_of)
    payload = "".join(f"{c}\t0\n" for _, _, c in names)
    r = subprocess.run(["cargo", "run", "--quiet", "--release", "-p", "plataid-sdk-core",
                        "--example", "sweep_lexical", "--", "--topk", "4"],
                       cwd=ROOT, input=payload, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"❌ FAIL  C14 taxonomy-hygiene — 채점 실패\n{r.stderr[-400:]}")
        return 1
    recs = [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")]
    if len(recs) != len(names):
        print(f"❌ FAIL  C14 taxonomy-hygiene — 행 수 불일치 {len(recs)} != {len(names)}")
        return 1

    stray = [(i, int(l2_of[recs[i]["top"][0]])) for i in range(len(names))
             if recs[i]["top"] and int(l2_of[recs[i]["top"][0]]) != int(l2_of[i])]
    ok = len(names) - len(stray)
    rate = ok / len(names)
    verdict = "✅ PASS" if rate >= FLOOR else "❌ FAIL"
    print(f"{verdict}  C14 taxonomy-hygiene")
    print(f"    리프 {len(names):,}개 중 **자기 중분류로 가는 것 {ok:,} ({rate:.1%})** "
          f"· 하한 {FLOOR:.0%}")
    print(f"    남의 중분류로 가는 이름 {len(stray):,}개 — 대부분 동명 리프다(경로가 가른다).")
    for i, got in stray[:6]:
        a, b, c = names[i]
        x, y = l2_list[got]
        print(f"      {c:14s} {a}›{b}  → 모델은 {x}›{y} 로 보낸다")
    if rate < FLOOR:
        print("    ⛔ 하한을 낮추지 마라 — 새로 들어온 이름 중 남의 자리에 앉은 것을 찾아 고친다.")
        print("    ⛔ 확장 후보를 거를 때는 scripts/expansion_l2_gate.py 를, "
              "이 값과 **비교해서** 읽어라.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
