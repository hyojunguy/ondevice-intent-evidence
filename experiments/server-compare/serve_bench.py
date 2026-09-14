#!/usr/bin/env python3
"""교사 인코더를 **서버로 서빙하면** 얼마나 걸리고 얼마나 처리하나 — 온디바이스와의 대조군.

동기(측정 전에 정해 둔다): 교사는 어휘후크 없는 구간에서 배포 경로보다 +17.37pp 앞선다
(`teacher_arm.json`). 그 17점을 서버로 사려면 무엇을 내야 하는지가 이 실험이다.

⛔ AI-Hub 텍스트를 쓰지 않는다 — 합성 입력(우리 지연 게이트와 같은 씨앗 문자열)만 쓴다.
   약관상 원본·재가공 데이터를 다른 주체의 인프라에 올릴 수 없다.
⛔ 배포 경로와 같은 입력 길이(512자)·같은 워밍업 규율을 쓴다. 다르면 비교가 성립하지 않는다.
⛔ 단일 스트림 지연과 포화 처리량을 **같은 이름으로 부르지 않는다**([[serving-measurement-provenance]]).
   전자는 대화형 지연, 후자는 서빙 용량이다.
"""
from __future__ import annotations
import json, os, platform, statistics, sys, time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
MODEL = os.environ.get("OICR_MODEL_DIR", "jhgan/ko-sroberta-multitask")
SEED = "튼튼한 아이폰 케이스 배송 빠른 곳 추천 저렴한 가격 비교 후기 "
CHARS = 512
LEAVES = 6020          # 우리 택소노미 리프 수 — 검색 단계를 실제 크기로 재현한다
WARMUP = 20
ITERS = 200
BATCHES = (1, 8, 32, 128)
REPS = 3               # ⛔ 인용하려면 반복이 필요하다. 1회는 게이트 등급이다.


def main() -> int:
    import numpy as np, torch
    from transformers import AutoModel, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if dev == "cuda" else platform.processor()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(dev).eval()
    if dev == "cuda":
        model = model.half()                      # 서빙 기본 — fp16
    text = "".join([SEED] * (CHARS // len(SEED) + 1))[:CHARS]

    # 리프 행렬은 검색 단계를 실제 크기로 재현하기 위한 것이다(내용은 무관하므로 난수).
    rng = np.random.default_rng(0)
    L = torch.tensor(rng.normal(size=(LEAVES, 768)).astype("float32"), device=dev)
    L = torch.nn.functional.normalize(L, dim=-1)
    if dev == "cuda":
        L = L.half()

    def encode_and_rank(texts):
        b = tok(texts, padding=True, truncation=True, max_length=128,
                return_tensors="pt").to(dev)
        with torch.no_grad():
            h = model(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
            v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            v = torch.nn.functional.normalize(v, dim=-1)
            s = v @ L.T
            return torch.topk(s, 5, dim=-1)

    def sync():
        if dev == "cuda":
            torch.cuda.synchronize()

    out = {"_config": {"model": MODEL, "device": dev, "gpu": gpu_name,
                       "dtype": "fp16" if dev == "cuda" else "fp32",
                       "torch": torch.__version__, "input_chars": CHARS,
                       "leaves": LEAVES, "warmup": WARMUP, "iters": ITERS,
                       "reps": REPS, "max_length": 128},
           "_note": "⛔ 이것은 교사(421MB 트랜스포머) 서빙 수치다. 배포 경로가 아니다.",
           "single_stream": {}, "throughput": {}}
    print("SERVE_CONFIG " + json.dumps(out["_config"], ensure_ascii=False), flush=True)

    # ── 단일 스트림 지연 (대화형 1건)
    reps = []
    for _ in range(REPS):
        for _ in range(WARMUP):
            encode_and_rank([text])
        sync()
        xs = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            encode_and_rank([text])
            sync()
            xs.append((time.perf_counter() - t0) * 1000)
        xs.sort()
        reps.append({"p50_ms": xs[len(xs) // 2],
                     "p95_ms": xs[int(len(xs) * 0.95)],
                     "p99_ms": xs[int(len(xs) * 0.99)]})
    out["single_stream"] = {k: round(statistics.median(r[k] for r in reps), 3)
                            for k in ("p50_ms", "p95_ms", "p99_ms")}
    out["single_stream"]["reps"] = reps

    # ── 포화 처리량 (배치를 올려 천장을 본다)
    for bs in BATCHES:
        batch = [text] * bs
        try:
            for _ in range(5):
                encode_and_rank(batch)
            sync()
            t0 = time.perf_counter()
            n = max(10, ITERS // bs)
            for _ in range(n):
                encode_and_rank(batch)
            sync()
            el = time.perf_counter() - t0
            out["throughput"][f"batch_{bs}"] = {
                "queries_per_sec": round(n * bs / el, 1),
                "ms_per_query": round(el / (n * bs) * 1000, 4)}
        except RuntimeError as e:                       # OOM 등은 숨기지 않는다
            out["throughput"][f"batch_{bs}"] = {"error": str(e)[:160]}
    peak = max((v["queries_per_sec"] for v in out["throughput"].values()
                if "queries_per_sec" in v), default=0.0)
    out["peak_queries_per_sec"] = peak
    out_path = os.environ.get("OICR_OUT", "/work/results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
