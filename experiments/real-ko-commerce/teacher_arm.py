#!/usr/bin/env python3
"""교사 인코더(768d, 421MB)를 **같은 실데이터·같은 대응표**로 돌린다 — IR 비교의 네 번째 팔.

리뷰 지적: 순열 기준선은 chance 이고, lexical/dense/deployed 만으로는 "3 MiB 제약이
품질을 얼마나 깎았나"에 답하지 못한다. 교사가 같은 판에서 몇 점인지가 그 답이다.

읽는 법(중요):
  교사 ≫ 배포  →  3 MiB 압축이 비용을 물렸다 (제약의 대가)
  교사 ≈ 배포  →  입력에 카테고리 정보가 없다 (적용 경계)
어느 쪽이 나와도 논문이 좋아진다 — 다만 **둘은 전혀 다른 결론**이라 반드시 재야 한다.

⛔ 이 스크립트는 **로컬에서 돈다.** AI-Hub 텍스트를 회사 클러스터로 올리지 않는다 —
   약관상 제3자 제공 소지가 있다. GPU 는 서빙 지연·처리량(합성 입력)에만 쓴다.
⛔ 교사는 배포 경로가 아니다. 여기서 나온 지연을 제품 수치로 인용하지 마라.
"""
from __future__ import annotations
import argparse, json, os, pathlib, sys, time

os.environ.setdefault("USE_TF", "0")        # ⛔ transformers 의 TF 임포트가 맥에서 데드락
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL = "jhgan/ko-sroberta-multitask"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="")
    a = ap.parse_args()

    import numpy as np, torch
    from transformers import AutoModel, AutoTokenizer
    sys.path.insert(0, str(ROOT / "data")); sys.path.insert(0, str(ROOT / "experiments/distill-ko"))
    from taxonomy_ko import TAXONOMY
    import leaf_repr

    dev = a.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    names, _x, _y, _z, l2_of = leaf_repr.flatten(TAXONOMY)
    l2_of = np.asarray(l2_of)
    pair = {}
    for li, (p, q, _r) in enumerate(names):
        pair.setdefault(int(l2_of[li]), f"{p}>{q}")
    idp = {v: k for k, v in pair.items()}
    cfg = json.loads((HERE / "aihub_map.json").read_text(encoding="utf-8"))["map"]
    gold = {k: {idp[x] for x in v["l2"]} for k, v in cfg.items()}

    rows = json.loads(pathlib.Path(a.corpus).read_text(encoding="utf-8"))
    seen = {}
    for r in rows:
        seen.setdefault((r["product"], f"{r['domain']}/{r['main']}"), None)
    qs = [(p, k) for (p, k) in seen]
    labels = [k for _, k in qs]

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(dev).eval()

    def encode(texts: list[str]) -> "np.ndarray":
        out = []
        for i in range(0, len(texts), a.batch):
            b = tok(texts[i:i + a.batch], padding=True, truncation=True,
                    max_length=128, return_tensors="pt").to(dev)
            with torch.no_grad():
                h = model(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)      # mean pooling — 교사 카드의 계약
            v = torch.nn.functional.normalize(v, dim=-1)
            out.append(v.float().cpu().numpy())
        return np.concatenate(out)

    # ⛔ 리프 표현은 배포 경로와 같은 문자열을 쓴다 — 교사에게만 더 좋은 문장을 주면
    #    비교가 아니라 교사 유리한 실험이 된다. leaf_repr 의 표현을 그대로 쓴다.
    leaf_texts = [f"{p} {q} {r}" for p, q, r in names]
    t0 = time.time(); L = encode(leaf_texts); t_leaf = time.time() - t0
    t0 = time.time(); Q = encode([q for q, _ in qs]); t_q = time.time() - t0

    S = Q @ L.T
    top = np.argsort(-S, axis=1)[:, :32]

    def l2rank(row):
        s, o = [], {}
        for li in row:
            v = int(l2_of[li])
            if v not in s:
                s.append(v); o.setdefault(v, len(s) - 1)
        return o
    rl = [l2rank(r) for r in top]
    hit = np.array([any(r.get(g, 99) < 5 for g in gold[lb]) for r, lb in zip(rl, labels)])

    hookset = sorted({c.replace(" ", "") for _p, _q, c in names if len(c) >= 2},
                     key=len, reverse=True)
    hook = np.array([any(h in q.replace(" ", "") for h in hookset) for q, _ in qs])
    rng = np.random.default_rng(0); n = len(qs); ix = rng.integers(0, n, (2000, n))
    b = hit[ix].mean(1)
    out = {"_model": MODEL, "_device": dev, "_params_note": "768d transformer, 421MB — 배포 경로가 아니다",
           "n": n, "중분류_top5": round(float(hit.mean()), 4),
           "ci": [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)],
           "어휘후크_있음": round(float(hit[hook].mean()), 4),
           "어휘후크_없음": round(float(hit[~hook].mean()), 4),
           "_encode_sec": {"leaves": round(t_leaf, 1), "queries": round(t_q, 1),
                           "note": "⛔ 맥 MPS 인코딩 시간이다. 서빙 지연으로 인용하지 마라."}}
    # ⛔ 독립 CI 로 "교사가 낫다/같다"를 말하지 마라 — 같은 질의를 보는 두 팔은 짝이다.
    sys.path.insert(0, str(HERE))
    from aihub_surface import run_deploy
    dep_top = run_deploy([q for q, _ in qs])
    dep_rl = [l2rank(t) for t in dep_top]
    dep = np.array([any(r.get(g, 99) < 5 for g in gold[lb])
                    for r, lb in zip(dep_rl, labels)], dtype=float)
    d = hit.astype(float) - dep
    bs = d[ix].mean(1)
    out["짝비교_교사_minus_배포"] = {
        "전체": {"delta_pp": round(float(d.mean() * 100), 2),
               "ci_pp": [round(float(np.percentile(bs, 2.5) * 100), 2),
                         round(float(np.percentile(bs, 97.5) * 100), 2)]},
    }
    for tag, sel in (("어휘후크 있음", hook), ("어휘후크 없음", ~hook)):
        ds = d[sel]
        jx = rng.integers(0, len(ds), (2000, len(ds)))
        bb = ds[jx].mean(1)
        out["짝비교_교사_minus_배포"][tag] = {
            "n": int(sel.sum()), "delta_pp": round(float(ds.mean() * 100), 2),
            "ci_pp": [round(float(np.percentile(bb, 2.5) * 100), 2),
                      round(float(np.percentile(bb, 97.5) * 100), 2)]}
    out["_reading"] = ("어휘후크 없는 구간에서 교사가 크게 앞서면 그 구간의 실패는 "
                       "**입력 정보 부족이 아니라 용량**이다 — 3 MiB 제약이 실제로 비용을 물린 자리다.")

    (HERE / "teacher_arm.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
