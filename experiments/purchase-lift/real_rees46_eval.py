#!/usr/bin/env python3
"""mPG vs LightGBM on REAL purchase events — REES46 e-commerce clickstream (HF mirror
`jin-ying-so-cute/ecommerce-user-behavior-data`, 34.8M rows, Dec 2019, no gate/login).

# 왜

`README.md`·`lgbm_benchmark.py`는 전부 합성 데이터였다("실 구매 로그 확보 전 인용
금지"라는 경고가 스스로 그걸 인정한다). 이 스크립트는 **진짜** (user, category,
timestamp) 구매 이벤트로 같은 질문(C-P1: mPG AUC, 카테고리 학습 사전의 값어치, LightGBM
대비)을 다시 잰다 — 합성 생성기가 아니라 실사용자 행동이다.

⛔ 여전히 인용에 조건이 붙는다(README §하단 참조): (1) 라이선스 미확인 미러
(2) 커버되는 창이 21일뿐(12/01~12/22) (3) 카테고리 하나(`construction.tools.light`,
전체 구매의 절반 이상)가 명백한 매핑 오류로 보여 제외했다 — 안 지웠으면 "모두가 매일
공구를 산다"는 유령 신호가 전체를 지배했을 것이다 (4) 우리 제품의 실제 카테고리
체계(2,960리프)가 아니라 이 데이터의 134개 카테고리다.

# 방법 (재현 가능)

1. 10개 parquet 샤드를 내려받아 purchase 행만 남긴다(`fetch()` — 매번 34.8M행을
   스캔하지만 필터 후엔 62만행이라 메모리는 안전).
2. `construction.tools.light` 제외(명백한 매핑 오류, docstring 위 참조).
3. **사용자를 절반으로 쪼갠다** — `prior_fit`(카테고리별 평균 재구매 주기를 추정하는
   데만 쓴다) / `eval`(k/span/gap 평가 행을 만드는 데만 쓴다). 이렇게 안 하면 "내가 배운
   사전으로 나를 맞힌다"는 자기참조 누수가 된다 — 실배포에서 FL 사전은 **다른 기기들**의
   집계이지 내 기록의 반영이 아니다. 이 분리가 그걸 흉내낸다.
4. `eval_purchase_lift.py`의 `p_buy`/`auc`를 그대로 가져다 쓴다(공식을 다시 안 짠다 —
   배포 코드와 같은 수식이어야 이 결과가 배포에 대해 의미가 있다).
"""
from __future__ import annotations
import argparse, json, pathlib, random, sys, urllib.request

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
assert (ROOT / "CONTRACT.md").exists(), f"ROOT 가 틀렸다: {ROOT}"
sys.path.insert(0, str(ROOT / "scripts"))
from eval_purchase_lift import p_buy, auc, DAY  # noqa: E402

REPO = "jin-ying-so-cute/ecommerce-user-behavior-data"
SHARD_URL = "https://huggingface.co/api/datasets/{repo}/parquet/default/train/{i}.parquet"
EXCLUDED_CATEGORY = "construction.tools.light"  # 매핑 오류로 보임 — docstring 참조


def fetch(cache_dir: pathlib.Path):
    import pandas as pd

    cache_dir.mkdir(parents=True, exist_ok=True)
    combined_path = cache_dir / "purchases_all.parquet"
    if combined_path.exists():
        return pd.read_parquet(combined_path)
    frames = []
    for i in range(10):
        local = cache_dir / f"shard_{i}.parquet"
        if not local.exists():
            urllib.request.urlretrieve(SHARD_URL.format(repo=REPO, i=i), local)
        df = pd.read_parquet(local, columns=["event_time", "event_type", "category_code", "user_id"])
        frames.append(df[df["event_type"] == "purchase"].drop(columns=["event_type"]))
        print(f"  shard {i}: {len(df):,} rows scanned", file=sys.stderr)
    all_p = pd.concat(frames, ignore_index=True)
    all_p.to_parquet(combined_path)
    return all_p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/tmp/rees46")
    ap.add_argument("--windows", default="3,7")
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()

    import pandas as pd

    print("fetching/loading REES46 purchase events...", file=sys.stderr)
    df = fetch(pathlib.Path(a.cache_dir))
    df["event_time"] = pd.to_datetime(df["event_time"])
    # ⛔ `.astype("int64") // 10**9` 는 pandas 버전에 따라 datetime64 단위(ns/us)가 달라
    # 1000배 틀린 초를 준다(실측: pandas 3.0.3 은 us 단위라 결과가 1/1000 로 축소돼
    # span/gap 이 전부 뭉개진다 — `lambda_p_sweep.py` 에서 처음 발견, 2026-09-05(22)).
    # 단위에 무관하게 실제 epoch-초를 구한다.
    _epoch = pd.Timestamp("1970-01-01", tz="UTC" if df["event_time"].dt.tz is not None else None)
    df["ts"] = (df["event_time"] - _epoch).dt.total_seconds().astype("int64")
    n_before = len(df)
    df = df[df["category_code"] != EXCLUDED_CATEGORY].copy()
    print(f"rows: {n_before:,} -> {len(df):,} after excluding {EXCLUDED_CATEGORY!r} "
          f"({(n_before - len(df)) / n_before:.1%} dropped)")

    t_min, t_max = df["event_time"].min(), df["event_time"].max()
    print(f"time span: {t_min} -> {t_max} ({(t_max - t_min).days} days)")

    rng = random.Random(a.seed)
    users = df["user_id"].unique().tolist()
    rng.shuffle(users)
    half = len(users) // 2
    prior_fit_users = set(users[:half])
    eval_users = set(users[half:])

    # 사전 적합 — prior_fit_users 만으로 카테고리별 평균 재구매 주기(초)
    fit_df = df[df["user_id"].isin(prior_fit_users)].sort_values("event_time")
    cat_cycle_s: dict[str, float] = {}
    for cat, g in fit_df.groupby("category_code"):
        gaps = []
        for _, ug in g.groupby("user_id"):
            ts = sorted(ug["ts"].tolist())
            if len(ts) >= 2:
                gaps.extend([b - a for a, b in zip(ts, ts[1:])])
        if len(gaps) >= 5:  # 표본이 너무 적은 카테고리는 사전을 못 배운 것으로 남긴다
            cat_cycle_s[cat] = sum(gaps) / len(gaps)
    print(f"학습된 카테고리 사전: {len(cat_cycle_s)}/{df['category_code'].nunique()} "
          f"(prior_fit 인구, eval 인구와 서로소)")

    # 평가 행 — eval_users 만, k>=2(마지막 1건을 홀드아웃으로 뗀다)
    eval_df = df[df["user_id"].isin(eval_users)].sort_values("event_time")
    rows = []
    for (uid, cat), g in eval_df.groupby(["user_id", "category_code"]):
        ts = sorted(g["ts"].tolist())
        if len(ts) < 2:
            continue
        train_ts, holdout_ts = ts[:-1], ts[-1]
        rows.append({
            "cat": cat,
            "k": len(train_ts),
            "span": train_ts[-1] - train_ts[0],
            "gap": holdout_ts - train_ts[-1],
        })
    print(f"평가 행(k>=2, eval 인구): {len(rows):,}")

    kinds = sorted({r["cat"] for r in rows})
    kind_ix = {k: i for i, k in enumerate(kinds)}
    global_cycle_s = sum(cat_cycle_s.values()) / max(len(cat_cycle_s), 1)

    out = {"n_rows": len(rows), "n_prior_fit_users": len(prior_fit_users),
           "n_eval_users": len(eval_users), "excluded_category": EXCLUDED_CATEGORY,
           "time_span_days": (t_max - t_min).days, "learned_categories": len(cat_cycle_s),
           "windows": {}}

    for wd in (int(x) for x in a.windows.split(",")):
        w_s = wd * DAY
        y = [int(r["gap"] <= w_s) for r in rows]
        pos_rate = sum(y) / len(y)

        auc_uninformed = 0.5
        auc_rfm = auc([(r["k"] / max(r["span"], DAY), yy) for r, yy in zip(rows, y)])
        auc_mpg_boot = auc([(p_buy(r["k"], r["span"], w_s), yy) for r, yy in zip(rows, y)])
        auc_mpg_learned = auc([
            (p_buy(r["k"], r["span"], w_s, alpha=2.0,
                   beta_s=2.0 * cat_cycle_s.get(r["cat"], global_cycle_s)), yy)
            for r, yy in zip(rows, y)
        ])

        try:
            import lightgbm as lgb
            import numpy as np

            n = len(rows)
            idx = list(range(n))
            random.Random(a.seed + wd).shuffle(idx)
            n_test = int(0.3 * n)
            test_idx_list, train_i = idx[:n_test], idx[n_test:]
            y_arr = np.array(y)

            def feats(idxs, with_cat):
                return np.array(
                    [[rows[i]["k"], rows[i]["span"] / DAY]
                     + ([kind_ix[rows[i]["cat"]]] if with_cat else []) for i in idxs],
                    dtype=float,
                )

            lgbm_auc = {}
            for with_cat, label in ((False, "no_category"), (True, "with_category")):
                xtr = feats(train_i, with_cat)
                xte = feats(test_idx_list, with_cat)
                cat_idx = [2] if with_cat else []
                dtr = lgb.Dataset(xtr, label=y_arr[train_i], categorical_feature=cat_idx,
                                   free_raw_data=False)
                params = dict(objective="binary", metric="auc", verbosity=-1, seed=7,
                              num_leaves=15, min_data_in_leaf=20, learning_rate=0.05)
                bst = lgb.train(params, dtr, num_boost_round=200)
                pred = bst.predict(xte)
                lgbm_auc[label] = auc(list(zip(pred, y_arr[test_idx_list])))
        except ImportError:
            lgbm_auc = {"no_category": None, "with_category": None}

        print(f"\n[w={wd}d] n={len(rows)} pos_rate={pos_rate:.1%}")
        print(f"  무정보                              : {auc_uninformed:.4f}")
        print(f"  RFM(빈도)                            : {auc_rfm:.4f}")
        print(f"  mPG boot(글로벌 기본, 카테고리 무시)   : {auc_mpg_boot:.4f}")
        print(f"  mPG 학습사전(실데이터에서 학습, 카테고리별) : {auc_mpg_learned:.4f}")
        print(f"  LightGBM(k,span — 카테고리 모름)      : {lgbm_auc['no_category']}")
        print(f"  LightGBM(k,span,cat — 카테고리 앎)    : {lgbm_auc['with_category']}")

        out["windows"][f"w{wd}d"] = {
            "pos_rate": pos_rate, "uninformed": auc_uninformed, "rfm": auc_rfm,
            "mpg_boot": auc_mpg_boot, "mpg_learned_real_prior": auc_mpg_learned,
            "lgbm_no_category": lgbm_auc["no_category"],
            "lgbm_with_category": lgbm_auc["with_category"],
        }

    if a.json_out:
        pathlib.Path(a.json_out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"\n-> {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
