#!/usr/bin/env python3
"""C11 — 카탈로그가 아직 **전량 번들 가능**한가 (docs/spec/07-catalog-delivery.md).

지금 우리는 광고 요청 유출이 0 이다. 카탈로그가 SDK 에 통째로 실려 나가서 매칭이 기기
안에서 끝나기 때문이다 — 즉 "브로드캐스트 + 로컬 필터"를 공짜로 얻고 있다.

⛔ 그 성질은 **카탈로그가 작아서** 성립한다. 인벤토리가 늘어 예산을 넘기는 순간 "필요한
   것만 가져오자"는 압력이 생기고, 그게 곧 사용자별 요청이고, 그게 곧 프로필 전송이다.

   그리고 이 전환은 **조용히** 일어난다. 광고가 늘어나는 것은 영업의 성공이지 설계
   결정이 아니기 때문이다. 그래서 이 게이트가 있다 — 유출 경로는 사고로 열리면 안 되고
   결정으로만 열려야 한다. FAIL 이면 07 의 A/B/C 중 하나를 골라 위협 모델에 적어라.
   ⛔ 게이트를 끄거나 예산을 올려서 통과시키는 것은 선택지가 아니다.
"""
from __future__ import annotations

import gzip
import pathlib
import struct
import sys

from _common import BUDGETS, FAIL, PASS, UNMEASURED, main_guard, report

ROOT = pathlib.Path(__file__).resolve().parents[1]
CATALOG = ROOT / "artifacts" / "catalog.bin"
CAT_MAGIC = b"PLAIDCAT"

# 카탈로그에 배정하는 SDK 예산의 몫. 잔여 전량을 카탈로그가 먹으면 vocab 확장·차원 확장·
# 도메인 헤드 같은 다른 용도가 사라진다 — 그래서 명시적으로 나눠 둔다.
CATALOG_BUDGET_SHARE = 0.25


def run_gate() -> int:
    if not CATALOG.exists():
        return report("C11 catalog-bundleable", UNMEASURED, [f"{CATALOG} 없음 — 빌드 먼저"])

    raw = CATALOG.read_bytes()
    if len(raw) < 20 or raw[:8] != CAT_MAGIC:
        return report("C11 catalog-bundleable", FAIL, ["catalog.bin 헤더가 PLAIDCAT 이 아니다"])

    n = struct.unpack("<I", raw[12:16])[0]
    gz = len(gzip.compress(raw, 9))
    per_entry = gz / n if n else 0.0

    limit = int(BUDGETS.get("size", {}).get("limit_bytes", 0))
    if not limit:
        return report("C11 catalog-bundleable", UNMEASURED, ["budgets.json:size.limit_bytes 없음"])

    share = int(limit * CATALOG_BUDGET_SHARE)
    lines = [
        f"카탈로그 {n:,} 항목 · raw {len(raw):,}B · gzip {gz:,}B (항목당 {per_entry:.1f}B)",
        f"카탈로그 배정 예산 {share:,}B = SDK 예산 {limit:,}B 의 {CATALOG_BUDGET_SHARE:.0%}",
    ]

    # 초과 상태에서 "약 -520개 더 담을 수 있다"는 읽는 사람을 헷갈리게 한다.
    if per_entry > 0 and gz <= share:
        headroom = int((share - gz) / per_entry)
        lines.append(f"이 배정 안에서 더 담을 수 있는 항목 약 {headroom:,}개")

    if gz > share:
        lines += [
            "⛔ 카탈로그가 배정 예산을 넘었다 — 전량 번들이 더 이상 성립하지 않는다.",
            "   이제 광고 요청 유출이 **설계 결정**이 된다. docs/spec/07-catalog-delivery.md 의",
            "   A(버킷 샤딩 · 버킷 누설 + k-익명 임계 필수) / B(PIR · 유출 0, 비쌈) /",
            "   C(하이브리드) 중 하나를 고르고 04-threat-model.md 에 그 유출을 적어라.",
            "   ⚠️ CATALOG_BUDGET_SHARE 를 올려서 통과시키지 마라 — 그건 다른 용도의 예산을",
            "      말없이 먹는 것이고, 어차피 SDK 예산 자체에서 다시 막힌다.",
        ]
        return report("C11 catalog-bundleable", FAIL, lines)

    lines.append("전량 번들 가능 → 사용자별 광고 요청이 필요 없다(유출 0)")
    return report("C11 catalog-bundleable", PASS, lines)


if __name__ == "__main__":
    sys.exit(main_guard(run_gate))
