#!/usr/bin/env python3
"""이미지 임베딩(OS FeaturePrint)의 성질을 **원본 산출물에서 읽기 전용으로 되살려 원장에 못박는다**.

⛔ 왜 생겼나 (2026-09-15, 집 맥): 논문이 인용하는 `norm 1.0006` 이 어느 JSON 원장에도 없었다.
   산문(docs/spec/19)에서 논문(main.tex:644 · main_en.tex:1010)으로 건너온 값이다.
   ABO 이미지 세트와 `featureprints.bin` 은 이 머신에만 있어서 회사 맥에서는 확인이 불가능했다.

⇒ 재측정이 아니다. 2026-08-31 추출이 남긴 **그 산출물 자체**를 다시 읽어 수를 센다
   ([[experiment-ledger-first]] #2 — 없는 원장을 재측정으로 메우지 않는다. 노름은
   산출물의 성질이므로 원본 파일에서 복원할 수 있고, 그건 재측정이 아니라 회수다).

⚠️ 결과: 노름은 **1.0002** 다. 1.0006 은 네 세트 어느 것에서도, 어떤 통계로도 나오지 않는다
   (평균·제곱평균·p99·최대·float32 누산 전부 확인). "L2 정규화" 라는 주장 자체는 옳다.
⛔ 장당 3.55ms · 12,000장 42.6초는 **여기서 복원되지 않는다** — featureprint.swift 가 그 수를
   stderr 로만 찍고 아무 데도 저장하지 않았다. 재측정하면 그날이 아닌 오늘의 수다.
"""
from __future__ import annotations
import hashlib, json, pathlib, struct, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MAGIC = b"PLAIDFPB"
# featureprint.swift 헤더: <magic 8B><u32 version><u32 n><u32 dim><u32 revision><f32 n*dim>
HEADER = 24


def read_set(path: pathlib.Path) -> dict:
    raw = path.open("rb")
    head = raw.read(HEADER)
    if head[:8] != MAGIC:
        raise SystemExit(f"⛔ {path}: magic 불일치 {head[:8]!r} — featureprint.swift 산출물이 아니다")
    version, n, dim, revision = struct.unpack("<4I", head[8:HEADER])
    raw.close()
    expect = HEADER + n * dim * 4
    actual = path.stat().st_size
    if actual != expect:
        raise SystemExit(f"⛔ {path}: 크기 {actual} != 헤더가 약속한 {expect}")
    v = np.fromfile(path, dtype="<f4", offset=HEADER).reshape(n, dim)
    norms = np.linalg.norm(v, axis=1)
    # swift 는 실패한 장을 0 벡터로 채운다 → 0 행 수가 곧 failed 수다.
    zero = int((norms == 0).sum())
    nz = norms[norms > 0]
    return {
        "file": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": actual,
        "version": int(version),
        "n": int(n),
        "dim": int(dim),
        "revision": int(revision),
        "failed_rows": zero,
        "l2_mean": round(float(nz.mean()), 6),
        "l2_median": round(float(np.median(nz)), 6),
        "l2_min": round(float(nz.min()), 6),
        "l2_max": round(float(nz.max()), 6),
        "l2_p99": round(float(np.percentile(nz, 99)), 6),
        "sq_norm_mean": round(float((nz ** 2).mean()), 6),
    }


def main() -> int:
    sets = sorted(ROOT.glob("experiments/percept-vision*/featureprints.bin"))
    if not sets:
        raise SystemExit("⛔ featureprints.bin 이 없다 — 이 머신에 ABO 세트가 없다")
    rows = [read_set(p) for p in sets]
    abo = next((r for r in rows if r["file"] == "experiments/percept-vision/featureprints.bin"), None)
    out = {
        "question": "OS FeaturePrint 임베딩의 차원·revision·L2 노름 (원본 산출물에서 회수)",
        "recovered_from": "2026-08-31 추출본 featureprints.bin (재측정 아님)",
        "paper_claim": {"norm": 1.0006, "source": "docs/spec/19 → main.tex:644 / main_en.tex:1010"},
        "measured": {"norm": abo["l2_mean"] if abo else None},
        "verdict": "1.0006 은 재현되지 않는다. ABO 세트의 노름 평균은 1.0002 다 (L2 정규화 주장 자체는 성립).",
        "unrecoverable_here": {
            "ms_per_image": 3.55,
            "elapsed_s": 42.6,
            "why": "featureprint.swift 가 elapsed 를 stderr 로만 출력하고 저장하지 않았다. "
                   "재측정하면 2026-08-31 이 아니라 오늘의 수가 된다 — 원장에 그 자리로 넣지 마라.",
        },
        "sets": rows,
        # ⛔ 경로가 넷이라고 독립 추출이 넷인 게 아니다 — 2026-09-15 확인: 둘이 같은 sha256 이다
        #    (percept-vision 과 percept-vision-fashion-ko 가 같은 파일). 판정은 안 바뀌지만
        #    "4개 세트에서 재현 안 됨" 이라고 읽히면 독립성을 과장하게 된다.
        "distinct_files": len({r["sha256"] for r in rows}),
        "duplicate_paths": [
            [r["file"] for r in rows if r["sha256"] == h]
            for h in sorted({r["sha256"] for r in rows})
            if sum(1 for r in rows if r["sha256"] == h) > 1
        ],
        "repro": "python3 experiments/percept-vision/featureprint_stats.py",
    }
    dest = HERE / "featureprint_stats.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for r in rows:
        print(f"{r['file']:<52} n={r['n']} dim={r['dim']} rev={r['revision']} "
              f"failed={r['failed_rows']} l2_mean={r['l2_mean']} max={r['l2_max']}")
    print(f"▶ {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
