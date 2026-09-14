#!/usr/bin/env python3
"""AI-Hub 측정이 **구현이 맞아서** 나온 수치인지 검사한다 (결론을 뒤집을 수 있는 것부터).

⛔ 이 저장소는 같은 계열 사고를 이미 두 번 겪었다 — 16축 A/B 가 전부 정확히 0.0000 이던
   것(파이썬 파서의 `assert ver == 2`), 그리고 대조군이 부서져 승리 폭이 부풀었던 것.
   그래서 "수치가 그럴듯하다"는 통과 근거가 아니다. 아래 네 검사를 통과해야 인용한다.

  A 절단   top-32 리프가 중분류 5개를 만들지 못하면 top-5 는 사실상 top-k(<5) 다.
  B 음성대조 질의 글자를 섞으면(어휘 후크 파괴 + 밀집 무의미) 정확도가 기준선으로 내려가야
           한다. 안 내려가면 질의를 안 읽고 맞히고 있다는 뜻이다.
  C 기전   '모델코드' 를 정규식으로 가르는 것은 표면 판정이다. 실제 기전은 **질의가 리프
           이름과 형태소를 공유하는가**이므로 그 축으로 다시 가른다.
  D 기준선 순열 기준선을 1회만 뽑으면 잡음이다. 반복해서 CI 를 낸다.
"""
from __future__ import annotations
import argparse, json, pathlib, random, sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from aihub_surface import run_deploy, ROOT


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--perm-reps", type=int, default=400)
    a = ap.parse_args()

    import numpy as np
    sys.path.insert(0, str(ROOT / "data")); sys.path.insert(0, str(ROOT / "experiments/distill-ko"))
    from taxonomy_ko import TAXONOMY
    import leaf_repr

    names, _l1s, _l2, l1_of, l2_of = leaf_repr.flatten(TAXONOMY)
    l2_of = np.asarray(l2_of)
    pair_of_l2 = {}
    for li, (x, y, _z) in enumerate(names):
        pair_of_l2.setdefault(int(l2_of[li]), f"{x}>{y}")
    id_of_pair = {v: k for k, v in pair_of_l2.items()}
    leafnames = sorted({c.replace(" ", "") for _x, _y, c in names if len(c) >= 2}, key=len, reverse=True)

    cfg = json.loads((HERE / "aihub_map.json").read_text(encoding="utf-8"))["map"]
    gold_l2 = {k: {id_of_pair[x] for x in v["l2"]} for k, v in cfg.items()}

    rows = json.loads(pathlib.Path(a.corpus).read_text(encoding="utf-8"))
    seen = {}
    for r in rows:
        seen.setdefault((r["product"], f"{r['domain']}/{r['main']}"), None)
    qs = [(p, k) for (p, k) in seen][: a.n]
    rnd = random.Random(7)

    def l2ranks(top):
        s, o = [], {}
        for li in top:
            v = int(l2_of[li])
            if v not in s:
                s.append(v); o.setdefault(v, len(s) - 1)
        return o

    def hit(rl, labs, k=5):
        return np.array([any(r.get(g, 99) < k for g in gold_l2[lb]) for r, lb in zip(rl, labs)])

    labels = [k for _, k in qs]
    tops = run_deploy([q for q, _ in qs])
    rl = [l2ranks(t) for t in tops]
    acc = hit(rl, labels).mean()
    out = {"n": len(qs), "중분류_top5": round(float(acc), 4)}

    # ── A. 절단
    distinct = np.array([len(r) for r in rl])
    out["A_절단"] = {"top32가_만든_중분류수_중앙값": int(np.median(distinct)),
                   "5개_미만인_질의": int((distinct < 5).sum()),
                   "비율": round(float((distinct < 5).mean()), 4),
                   "판정": "5개 미만이 있으면 그 질의의 top-5 는 절단된 것"}

    # ── B. 음성대조 (글자 섞기)
    def scramble(s):
        c = list(s); rnd.shuffle(c); return "".join(c)
    tops_s = run_deploy([scramble(q) for q, _ in qs])
    acc_s = hit([l2ranks(t) for t in tops_s], labels).mean()
    out["B_음성대조"] = {"글자섞은_질의_top5": round(float(acc_s), 4),
                    "원본_top5": round(float(acc), 4),
                    "낙폭_pp": round(float((acc - acc_s) * 100), 2)}

    # ── C. 기전 축 — 질의가 리프 이름과 형태소를 공유하나
    def has_hook(q):
        qn = q.replace(" ", "")
        return any(ln in qn for ln in leafnames)
    hook = np.array([has_hook(q) for q, _ in qs])
    h = hit(rl, labels)
    out["C_기전축"] = {}
    for tag, sel in (("어휘후크 있음", hook), ("어휘후크 없음", ~hook)):
        if sel.sum():
            out["C_기전축"][tag] = {"n": int(sel.sum()), "top5": round(float(h[sel].mean()), 4)}

    # ── D. 순열 기준선 반복
    accs = []
    for _ in range(a.perm_reps):
        perm = list(labels); rnd.shuffle(perm)
        accs.append(float(hit(rl, perm).mean()))
    accs = np.array(accs)
    out["D_순열기준선"] = {"평균": round(float(accs.mean()), 4),
                     "CI": [round(float(np.percentile(accs, 2.5)), 4),
                            round(float(np.percentile(accs, 97.5)), 4)],
                     "반복": a.perm_reps}

    (HERE / "verify_harness.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
