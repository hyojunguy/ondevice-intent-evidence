#!/usr/bin/env python3
"""실 REES46 에서 p_buy AUC 가 0.53~0.59 로 무너지는 이유를 **가설이 아니라 분포로** 본다.

2026-09-05(24). (19)이 남긴 "실 AUC 0.53~0.59" 를 고치기 전에, 무엇이 신호를 죽이는지
먼저 센다. 문헌(Walmart RecSys'26 `arXiv:2608.28393`)이 지목하는 세 가지를 그대로 검사한다:

  H1. **검열(censoring) 제거** — 현행 평가행은 `k>=2` 인 (유저,카테고리)만 쓰고 마지막
      구매를 홀드아웃으로 뗀다. 즉 **재구매한 사람만 평가에 들어온다.** 논문: "dropping
      censored items destroys generalization to long-cadence buyers" (치명적이라고 명시).
  H2. **최근성 미사용** — 배포 `p_buy` 는 span=now−t_first 와 k 만 쓰고 t_last 를 안 본다.
      논문: survival 목적함수에서 recency 는 오르고 aggregate frequency 는 내린다.
      게다가 현행 평가는 now=t_last 라 **최근성이 구조적으로 항상 0** — 잴 수조차 없다.
  H3. **k 축퇴** — 21일 창에서 k 가 사실상 2에 몰려 있으면 (α+k) 항이 상수나 다름없어
      모델이 span 만으로 줄세우게 된다.

출력은 판정이 아니라 **숫자**다. 처방은 이 숫자를 보고 고른다.
"""
from __future__ import annotations
import collections
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_purchase_lift import p_buy, auc, DAY  # noqa: E402

EXCLUDED_CATEGORY = "construction.tools.light"


def load(cache_dir: str = "/tmp/rees46"):
    import pandas as pd

    df = pd.read_parquet(pathlib.Path(cache_dir) / "purchases_all.parquet")
    df["event_time"] = pd.to_datetime(df["event_time"])
    epoch = pd.Timestamp("1970-01-01", tz="UTC" if df["event_time"].dt.tz is not None else None)
    df["ts"] = (df["event_time"] - epoch).dt.total_seconds().astype("int64")
    return df[df["category_code"] != EXCLUDED_CATEGORY].copy()


def pct(x: float) -> str:
    return f"{x:.1%}"


def quantiles(vals: list[float], qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> str:
    v = sorted(vals)
    if not v:
        return "(비어 있음)"
    return " ".join(f"p{int(q*100)}={v[min(int(q * len(v)), len(v)-1)]:.1f}" for q in qs)


def main() -> int:
    df = load()
    t_min, t_max = df["ts"].min(), df["ts"].max()
    print(f"구매 이벤트 {len(df):,} · 유저 {df['user_id'].nunique():,} · "
          f"카테고리 {df['category_code'].nunique()} · 창 {(t_max - t_min)/DAY:.1f}일\n")

    # ── H1: 검열이 얼마나 잘려나가나 ─────────────────────────────────────────
    pair_counts = df.groupby(["user_id", "category_code"]).size()
    n_pairs = len(pair_counts)
    n_k1 = int((pair_counts == 1).sum())
    n_k2p = n_pairs - n_k1
    print("H1 검열 제거 —")
    print(f"  (유저,카테고리) 쌍 총 {n_pairs:,}")
    print(f"    k=1 (재구매 없음 → **평가에서 통째로 제외됨**): {n_k1:,} ({pct(n_k1/n_pairs)})")
    print(f"    k>=2 (평가에 들어옴)                          : {n_k2p:,} ({pct(n_k2p/n_pairs)})")
    print("  ⇒ 평가 모집단이 '재구매한 사람'만이라, 모델이 실제로 답해야 하는 질문"
          "('이 사람이 다시 살까')의 다수 케이스가 사라진다.\n")

    # ── H3: k 축퇴 ──────────────────────────────────────────────────────────
    rows = []
    for (uid, cat), g in df.groupby(["user_id", "category_code"]):
        ts = sorted(g["ts"].tolist())
        if len(ts) < 2:
            continue
        train, holdout = ts[:-1], ts[-1]
        rows.append({"cat": cat, "k": len(train), "span": train[-1] - train[0],
                     "gap": holdout - train[-1], "t_last": train[-1]})
    print(f"H3 k 축퇴 — 현행 평가행 {len(rows):,}")
    kd = collections.Counter(r["k"] for r in rows)
    for k in sorted(kd)[:6]:
        print(f"    k={k}: {kd[k]:,} ({pct(kd[k]/len(rows))})")
    tail = sum(v for k, v in kd.items() if k > 6)
    print(f"    k>6 : {tail:,} ({pct(tail/len(rows))})")
    print(f"  span(일): {quantiles([r['span']/DAY for r in rows])}")
    print(f"  gap(일) : {quantiles([r['gap']/DAY for r in rows])}")
    n_span0 = sum(1 for r in rows if r["span"] == 0)
    print(f"  ⇒ span=0 (k=2 이고 두 구매가 같은 초): {n_span0:,} ({pct(n_span0/len(rows))})")
    print("  ⇒ (α+k) 가 사실상 상수면 p_buy 의 줄세우기는 span 단독이 결정한다.\n")

    # ── 관측창 절단이 라벨을 얼마나 직접 결정하나 ────────────────────────────
    print("H1b 관측창 절단 —")
    room = [(t_max - r["t_last"]) / DAY for r in rows]
    print(f"  마지막 학습구매 이후 남은 관측 여유(일): {quantiles(room)}")
    for wd in (3, 7):
        w = wd * DAY
        y = [int(r["gap"] <= w) for r in rows]
        forced = sum(1 for r in rows if (t_max - r["t_last"]) <= w)
        print(f"  w={wd}d: 양성률 {pct(sum(y)/len(y))} · "
              f"남은 여유<=w 라 라벨이 강제로 1인 행 {forced:,} ({pct(forced/len(rows))})")
    print("  ⇒ 여유가 창보다 짧으면 '창 안 재구매'가 데이터 수집 방식 때문에 참이 된다"
          " — 모델 실력이 아니라 절단을 재는 부분.\n")

    # ── H2: 최근성이 구조적으로 0 ───────────────────────────────────────────
    print("H2 최근성 —")
    print("  현행 평가: 판정 시점 now = t_last(마지막 학습구매) → **모든 행에서 recency=0**.")
    print("  배포 현실: 광고 기회는 임의 시점에 온다 → recency>0 이 정상.")
    print("  ⇒ 지금 하네스로는 최근성 모델의 이득을 **잴 수 없다**(0으로 상수라서).\n")

    # ── 현행 스코어 3종의 실제 AUC 재확인(같은 행) ──────────────────────────
    print("현행 스코어 재확인 (같은 행, 배포 수식 그대로) —")
    for wd in (3, 7):
        w = wd * DAY
        y = [int(r["gap"] <= w) for r in rows]
        a_rfm = auc([(r["k"] / max(r["span"], DAY), yy) for r, yy in zip(rows, y)])
        a_mpg = auc([(p_buy(r["k"], r["span"], w), yy) for r, yy in zip(rows, y)])
        a_span = auc([(-r["span"], yy) for r, yy in zip(rows, y)])
        a_k = auc([(r["k"], yy) for r, yy in zip(rows, y)])
        print(f"  w={wd}d  RFM={a_rfm:.4f}  mPG={a_mpg:.4f}  "
              f"span단독(-span)={a_span:.4f}  k단독={a_k:.4f}")
    print("  ⇒ span 단독이 mPG 와 비슷하면, 모델이 파는 것은 사실상 span 하나다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
