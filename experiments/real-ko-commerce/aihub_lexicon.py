#!/usr/bin/env python3
"""실제 소비자가 부르는 상품 이름이 우리 6천 리프 안에 있나 — 대응 판단이 필요 없는 측정.

⛔ 이 스크립트는 **집계 수치만** 낸다. 원본에서 뽑은 상품명 문자열은 산출물에 쓰지 않는다.
   AI-Hub 공식 안내(2026): 자유 공개가 되는 2차 저작물은 **실제 학습으로 나온 모델·가중치**이고,
   원본을 추출·정리·편집·변환한 **재가공 데이터는 출처를 표기해도 공개·재배포할 수 없다.**
   GitHub 같은 해외 플랫폼 저장은 국외 반출에도 해당한다. 미포함 어휘의 성격은 문자열 목록이
   아니라 **상위어 비율** 같은 수로만 보고한다.

정확도(=랭킹)와 달리 이 수치는 **사람이 만든 대응표를 거치지 않는다.** 실제 발화에서
주석자가 상품으로 표시한 문자열이 리프 이름과 그대로 일치하는지만 본다. 그래서
"택소노미가 실제 소비자 언어를 얼마나 담고 있나"의 하한으로 쓸 수 있다.
⛔ 데이터는 재배포하지 않는다 — 코퍼스/어휘 경로는 인자로 받는다.
"""
from __future__ import annotations
import argparse, collections, json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RETAIL = {"가구인테리어", "디지털가전", "생활잡화", "의류", "출산육아", "패션", "뷰티", "식품", "슈퍼"}



def hypernym_stats(missing: list[str], leaf: set[str]) -> dict:
    """미포함 어휘가 **상위어**인가 — 문자열을 내보내지 않고 성격만 수로 낸다.

    어떤 미포함 항목 h 가 우리 리프 2개 이상의 **접미어**이면(청바지·남성청바지 ⊃ 바지)
    그것은 택소노미에 없는 상위어다. 이 판정은 원본 문자열을 산출물에 남기지 않는다.
    """
    hyper = sum(1 for h in missing
                if len(h) >= 2 and sum(1 for l in leaf if l.endswith(h) and l != h) >= 2)
    return {"검사한_미포함_상위항목": len(missing),
            "그중_우리리프의_상위어": hyper,
            "비율": round(hyper / len(missing), 4) if missing else 0.0,
            "판정": "리프 2개 이상의 접미어이면 상위어로 센다(청바지·남성청바지 ⊃ 바지)",
            "note": "⛔ 문자열 자체는 내보내지 않는다 — AI-Hub 재가공 데이터 공개 금지"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus102", required=True)
    ap.add_argument("--lexicon98", required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "data"))
    from taxonomy_ko import TAXONOMY
    leaf = {l.replace(" ", "") for m in TAXONOMY.values() for v in m.values() for l in v}

    out = {"taxonomy": {"L1": len(TAXONOMY),
                        "L2": sum(len(m) for m in TAXONOMY.values()),
                        "leaf": len(leaf)},
           "sources": {}}

    rows = json.loads(pathlib.Path(a.corpus102).read_text(encoding="utf-8"))
    per = collections.defaultdict(lambda: [0, 0])
    ment = collections.Counter()
    for r in rows:
        for p in (r["product"] or "").split("|"):
            p = p.strip().replace(" ", "")
            if not p:
                continue
            ment[p] += 1
            per[r["cat"]][1] += 1
            per[r["cat"]][0] += int(p in leaf)
    hit = sum(v[0] for v in per.values()); tot = sum(v[1] for v in per.values())
    re_ = sum(per[c][0] for c in RETAIL if c in per); rt = sum(per[c][1] for c in RETAIL if c in per)
    out["sources"]["AI-Hub 102 고객 발화"] = {
        "언급": tot, "고유": len(ment),
        "리프_정확일치_언급비율": round(hit / tot, 4),
        "리프_정확일치_고유비율": round(sum(1 for p in ment if p in leaf) / len(ment), 4),
        "소매9종": {"n": rt, "일치율": round(re_ / rt, 4)},
        "비소매4종": {"n": tot - rt, "일치율": round((hit - re_) / (tot - rt), 4),
                  "구성": "음식점·카페·병원·건강 — 외식 메뉴와 일반약이 많다"},
        "카테고리별": {c: {"n": v[1], "일치율": round(v[0] / v[1], 4)} for c, v in sorted(per.items())},
        "미포함_성격": hypernym_stats(
            [p for p, c in ment.most_common(4000) if p not in leaf][:200], leaf)}

    lex = json.loads(pathlib.Path(a.lexicon98).read_text(encoding="utf-8"))["mentions"]
    tot98 = sum(lex.values()); h98 = sum(c for p, c in lex.items() if p.replace(" ", "") in leaf)
    out["sources"]["AI-Hub 98 콜센터 상품 엔티티"] = {
        "언급": tot98, "고유": len(lex),
        "리프_정확일치_언급비율": round(h98 / tot98, 4),
        "리프_정확일치_고유비율": round(sum(1 for p in lex if p.replace(" ", "") in leaf) / len(lex), 4),
        "note": "⛔ 이 데이터는 발화별 상품 카테고리 라벨이 없어 **정확도 평가에는 못 쓴다** — 어휘 커버리지 전용.",
        "미포함_성격": hypernym_stats(
            [p.replace(" ", "") for p, _ in sorted(lex.items(), key=lambda x: -x[1])
             if p.replace(" ", "") not in leaf][:200], leaf)}

    (HERE / "aihub_lexicon.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "카테고리별"}
                      for k, v in out["sources"].items()}, ensure_ascii=False, indent=2))
    print(f"\n▶ {HERE / 'aihub_lexicon.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
