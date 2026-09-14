#!/usr/bin/env python3
"""차원 · 어휘 · 비트폭 — 같은 바이트를 어디에 쓸지 코드가 계산한다.

⛔ 왜 있나: README(2026-08-28)가 64->128 비용을 "임베딩 테이블 2배"로만 잡아
   70.7% 라고 적었다. 그런데 d 로 스케일하는 아티팩트는 embedding 말고도
   taxonomy·catalog·head·projection 넷이 더 있다. 실제는 76.2% 다.
   손으로 곱하면 또 빠뜨린다 -- 그래서 코드가 소유한다.

압축률은 가정하지 않고 실제 배포 아티팩트에서 잰다(embedding.bin raw vs gzip).
"""
from __future__ import annotations
import gzip, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 5_242_880  # gates/budgets.json size.limit_bytes

def measured_ratio() -> float:
    """int8 양자화 벡터의 실측 gzip 압축률. 가정하지 않는다."""
    b = (ROOT / "artifacts/embedding.bin").read_bytes()
    return len(gzip.compress(b, 9)) / len(b)

def gz(path: str) -> int:
    return len(gzip.compress((ROOT / path).read_bytes(), 9))

# 실측 gzip 비율 (experiments/dim-vs-vocab/pack_results.json, 2026-08-28).
# ⛔ 가정하지 마라: 4bit 패킹은 고차 PCA 성분이 중복돼 **int8 보다 잘 눌린다**
#    (d=512 에서 0.380 vs 0.628). 손으로 0.73 을 곱하면 예산을 크게 틀린다.
PACK_RATIO = {
    (64, 8): 0.794, (64, 4): 0.6721,
    (128, 8): 0.7792, (128, 4): 0.6436,
    (256, 8): 0.7304, (256, 4): 0.5569,
    (512, 8): 0.628, (512, 4): 0.380,
}


def pack_ratio(dim: int, bits: int) -> float:
    """가장 가까운 실측 차원의 비율을 쓴다. 외삽하지 않는다."""
    cand = [d for (d, b) in PACK_RATIO if b == bits]
    return PACK_RATIO[(min(cand, key=lambda d: abs(d - dim)), bits)]


def main() -> int:
    r = measured_ratio()
    man = json.loads((ROOT / "artifacts/MANIFEST.json").read_text())
    V0 = man["_provenance"]["vocab_size"]        # 31,497
    D0 = man["_provenance"]["dim"]               # 64
    LEAF = man["_provenance"]["taxonomy_leaves"] # 2,960
    CAT = man["_provenance"]["catalog_entries"]  # 3,000
    HEAD = 28                                    # budgets.json attribution.buckets

    # d 로 스케일하지 않는 잔여분: 실측 gzip 에서 벡터 몫을 빼서 얻는다.
    tax_text = gz("artifacts/taxonomy.bin") - int(LEAF * D0 * r)
    cat_text = gz("artifacts/catalog.bin") - int(CAT * D0 * r)
    aff = gz("artifacts/affinity.bin")
    vocab_gz_per_tok = gz("artifacts/vocab.txt") / V0

    def l1(vocab: int, d: int, bits: int = 8) -> dict[str, int]:
        """bits=4 면 요소당 0.5 바이트 + 행당 f32 스케일."""
        pr = pack_ratio(d, bits)
        bpe = bits / 8.0

        def tbl(rows: int) -> int:
            return int((rows * d * bpe + rows * 4) * pr)

        return {
            "embedding": tbl(vocab),
            "vocab.txt": int(vocab * vocab_gz_per_tok),
            "taxonomy": tbl(LEAF) + tax_text,
            "catalog": tbl(CAT) + cat_text,
            "head+proj": int((HEAD * d + d * d) * bpe * pr),
            "affinity": aff,
        }

    print(f"실측 압축률(int8 벡터) = {r:.4f}   상한 = {LIMIT:,} B")
    print(f"기준선: vocab {V0:,} · d {D0} · leaf {LEAF:,} · catalog {CAT:,}")
    print("⛔ L1(다운로드 데이터)만 센다. web/demo/dist 는 데모지 SDK 가 아니다.\n")
    L1_LIMIT = int(json.loads((ROOT / "gates/budgets.json").read_text(encoding="utf-8"))["size"]["limit_bytes"])
    print(f"{'vocab':>8} {'d':>4} {'bit':>4} {'L1 gzip':>11} {'L1선%':>7} {'약속%':>7} {'남는 여유':>11}")
    for vocab in (24_000, 31_497, 45_000):
        for d in (64, 128, 256, 512):
            for bits in (8, 4):
                tot = sum(l1(vocab, d, bits).values())
                flag = "" if tot <= L1_LIMIT else ("  L1초과" if tot <= LIMIT else "  OVER")
                print(f"{vocab:>8,} {d:>4} {bits:>4} {tot:>11,} {tot/L1_LIMIT*100:>6.1f}% "
                      f"{tot/LIMIT*100:>6.1f}% {LIMIT-tot:>11,}{flag}")
    print("\n[증분] 현재(31,497 x 64 int8) -> 31,497 x 512 @4bit")
    a, b = l1(V0, D0, 8), l1(V0, 512, 4)
    for k in a:
        if b[k] != a[k]:
            print(f"  {b[k]-a[k]:>+10,}  {k}")
    print(f"  ={sum(b.values())-sum(a.values()):>10,}  합계")
    return 0

if __name__ == "__main__":
    sys.exit(main())
