#!/usr/bin/env python3
"""실 REES46 구매확률 — **배포에 충실한** 평가 하네스 v2 + 최근성 모델 + 브라우징 신호.

2026-09-05(24). (19)의 "실 AUC 0.53~0.59" 를 `diagnose_real.py` 로 뜯어보니 절반이
모델이 아니라 **평가 설계**였다. v1 이 한 것과 이 하네스가 바꾼 것:

| 축 | v1 (`real_rees46_eval.py`) | v2 (여기) | 근거 |
|---|---|---|---|
| 모집단 | 재구매한 쌍만(k>=2) — 79.8% 제외 | **1건이라도 산 쌍 전부** | 논문: 검열 제거는 "catastrophic" |
| 판정 시점 | now = 마지막 학습구매 → **recency 항상 0** | **스냅샷 t0 고정**, recency = t0−t_last > 0 | 광고 기회는 임의 시점에 온다 |
| 라벨 지평 | gap <= w, 창 끝에 걸리면 강제로 1(46.5%) | t0+w <= t_max 라 **전 행이 완전 관측** | 절단을 실력으로 오독하지 않기 |
| 구매 단위 | 상품 이벤트 그대로 (장바구니 분할이 재구매로) | **주문(user_session) 단위 dedup** | gap 중앙값 0.5일 = 같은 주문 |

⛔ 이 넷을 바꾸면 v1 숫자와 **직접 비교할 수 없다**(과제가 다르다). v1 은 "재구매한
사람 중 누가 더 빨리 오나", v2 는 "산 적 있는 사람이 앞으로 w일 안에 다시 사나"다.
후자가 배포에서 실제로 묻는 질문이다.

# 모델

`--with-browse` 없이는 전부 **기기에 이미 있는 상태만** 쓴다 — 리프당 (k, t_first, t_last)
+ L2 사전. 새 상태 0.

- `mpg`      : 배포 현행. p_buy = 1−((β+T)/(β+T+w))^(α+k), T = t0−t_first. **recency 미사용**
- `mpg_cat`  : 위 + L2 별 학습 β (FL 사전 채널이 배울 값)
- `weibull_m`: η = (β+T)/(α+k) 로 개인 주기를 추정하고, **경과 dt 를 조건으로** 건다
               P = 1 − exp((dt/η)^m − ((dt+w)/η)^m). m=1 이면 지수분포(무기억) = recency 무시.
               m<1 = 감소 hazard(논문 실측 k̂≈0.911) → 최근 산 사람이 더 곧 산다.
- `--with-browse` 팔은 **조회/장바구니**를 쓴다. 기기엔 이미 관심 프로필이 있으므로 상태를
  새로 만드는 게 아니라 **구매 확률에 그 신호를 연결**하는 문제다. 여기서 그 값어치를 먼저 잰다.
- 기준선     : 무정보 · RFM(k/T) · recency 단독 · LightGBM(같은 피처 = 실현 가능 상한)

인용 규율: AUC 차이는 **paired 부트스트랩 CI** 로만 판정한다([[experiment-ledger-first]]).
"""
from __future__ import annotations
import argparse, json, pathlib, sys, urllib.request

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
assert (ROOT / "CONTRACT.md").exists(), f"ROOT 가 틀렸다: {ROOT}"

DAY = 86400.0
REPO = "jin-ying-so-cute/ecommerce-user-behavior-data"
SHARD_URL = "https://huggingface.co/api/datasets/{repo}/parquet/default/train/{i}.parquet"
EXCLUDED_CATEGORY = "construction.tools.light"  # brand=apple·price 1302 — 스마트폰 오분류
DEFAULT_ALPHA = 1.0
DEFAULT_BETA_S = 30.0 * DAY


# ── 데이터 ────────────────────────────────────────────────────────────────────
def shard_path(cache_dir: pathlib.Path, i: int) -> pathlib.Path:
    local = cache_dir / f"shard_{i}.parquet"
    if not local.exists():
        urllib.request.urlretrieve(SHARD_URL.format(repo=REPO, i=i), local)
    return local


def to_epoch_s(series):
    """⛔ `.astype('int64')//10**9` 금지 — pandas 버전마다 단위(ns/us)가 달라 1000배 틀린다."""
    import pandas as pd

    s = pd.to_datetime(series)
    epoch = pd.Timestamp("1970-01-01", tz="UTC" if s.dt.tz is not None else None)
    return (s - epoch).dt.total_seconds().astype("int64")


def purchases_with_session(cache_dir: pathlib.Path):
    """구매 이벤트 + user_session. 기존 purchases_all.parquet 은 세션 컬럼이 없어 새로 만든다."""
    import pandas as pd

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "purchases_sess.parquet"
    if path.exists():
        return pd.read_parquet(path)
    frames = []
    for i in range(10):
        df = pd.read_parquet(
            shard_path(cache_dir, i),
            columns=["event_time", "event_type", "category_code", "user_id", "user_session"],
        )
        frames.append(df[df["event_type"] == "purchase"].drop(columns=["event_type"]))
        print(f"  shard {i}: {len(df):,} rows scanned", file=sys.stderr)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(path)
    return out


def to_our_taxonomy(df, bridge_path: pathlib.Path):
    """REES46 `category_code` 를 **우리 중분류(L2) id** 로 갈아끼운다.

    왜: REES46 의 132 카테고리는 우리 분류 체계가 아니다. 어블레이션이 그 축에서만
    성립하면 "우리 제품에서도 되나"에 답하지 못한다. 브리지(`rees46_category_bridge.json`,
    배포 분류기 5-문구 투표)로 L2 에 올려 **같은 어블레이션**을 다시 돌린다.

    ⛔ 브리지는 노이즈가 있다(약한 합의 다수). 그래서 이 경로의 수치는 "우리 축에서도
       방향이 남는가"를 보는 것이지 카테고리 단위 정밀 주장이 아니다.
    ⛔ 252 밖 L2 는 버린다 — 기기의 `l2_stats` 가 같은 자리에서 버리므로 여기서도 같게 한다.
    """
    import json as _json
    import sys as _sys

    # experiments/purchase-lift/<file>.py → parents[2] 가 레포 루트다([[repo-root-path-depth]]).
    root = pathlib.Path(__file__).resolve().parents[2]
    assert (root / "CONTRACT.md").exists(), f"ROOT 가 틀렸다: {root}"
    _sys.path.insert(0, str(root))
    from data.taxonomy_ko import TAXONOMY  # noqa: PLC0415

    idx, n = {}, 0
    for l1, mids in TAXONOMY.items():
        for m in mids:
            idx[f"{l1}|{m}"] = n
            n += 1
    bridge = _json.loads(bridge_path.read_text())
    m = {}
    for cat, rec in bridge.items():
        key = (rec.get("mapped_l2") or {}).get("leaf_key")
        c = idx.get(key) if key else None
        if c is not None and c < 252:
            m[cat] = f"l2-{c:04d}"
    before_rows, before_cats = len(df), df["category_code"].nunique()
    df = df[df["category_code"].isin(m)].copy()
    df["category_code"] = df["category_code"].map(m)
    print(
        f"우리 택소노미로 사상: 행 {before_rows:,}→{len(df):,} · "
        f"카테고리 {before_cats}→{df['category_code'].nunique()} (REES46 132 → 우리 L2)"
    )
    return df


def load_orders(cache_dir: str, dedup: bool):
    """(user, category, order) 단위 구매. dedup=True 면 같은 세션의 같은 카테고리를 1건으로."""
    df = purchases_with_session(pathlib.Path(cache_dir))
    df = df[df["category_code"] != EXCLUDED_CATEGORY].copy()
    df["ts"] = to_epoch_s(df["event_time"])
    n0 = len(df)
    if dedup:
        # 같은 세션 안의 같은 카테고리 = 한 주문 → 최초 시각 1건으로 접는다.
        df = df.groupby(["user_id", "category_code", "user_session"], as_index=False)["ts"].min()
    else:
        df = df[["user_id", "category_code", "ts"]]
    print(f"구매 이벤트 {n0:,} -> {len(df):,} ({'세션 dedup' if dedup else 'dedup 없음'})")
    return df.sort_values("ts")


def browse_events(cache_dir: str, pairs_df):
    """평가 대상 (user,category) 쌍의 view/cart 이벤트만 골라 캐시한다.
    ⛔ 전체 조회이벤트는 32M행이라 통째로 올리지 않는다 — 쌍으로 먼저 좁힌 뒤 모은다."""
    import pandas as pd

    cache = pathlib.Path(cache_dir) / "browse_evalpairs.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    keep = pairs_df.drop_duplicates()
    frames = []
    for i in range(10):
        df = pd.read_parquet(
            shard_path(pathlib.Path(cache_dir), i),
            columns=["event_time", "event_type", "category_code", "user_id"],
        )
        df = df[df["event_type"].isin(("view", "cart"))]
        df = df.merge(keep, on=["user_id", "category_code"], how="inner")
        df["ts"] = to_epoch_s(df["event_time"])
        frames.append(df[["user_id", "category_code", "event_type", "ts"]])
        print(f"  browse shard {i}: kept {len(df):,}", file=sys.stderr)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(cache)
    return out


# ── 평가행 (스냅샷) ───────────────────────────────────────────────────────────
def build_rows(df, w_s: float, prior_frac: float, seed: int, t0=None, pop: str = "eval",
               user_universe=None, cat_ix=None):
    """스냅샷 t0(기본 t_max − w). t0 이전 이력이 있는 모든 (user,cat) 쌍이 후보.
    라벨 = (t0, t0+w] 에 그 카테고리 구매가 있는가. 지평이 전부 관측 안이라 절단 없음.

    `t0`/`pop`/`user_universe`/`cat_ix` 는 **시간 일반화 검사**(--temporal)용이다:
    이른 스냅샷에서 계수를 적합하고 늦은 스냅샷에서 평가하려면 (a) t0 를 직접 주고
    (b) 유저 분할이 두 스냅샷에서 **같아야** 하며(안 그러면 같은 사람이 양쪽에 든다)
    (c) 카테고리 인덱스가 공유돼야 한다. 인자를 안 주면 기존 동작 그대로다."""
    t_max = int(df["ts"].max())
    if t0 is None:
        t0 = t_max - int(w_s)
    t0 = int(t0)
    hist = df[df["ts"] <= t0]
    # ⛔ 지평을 w 로 **닫는다** — 기본 t0 에서는 t_max 가 곧 t0+w 라 차이가 없지만,
    #    이른 스냅샷에서는 닫지 않으면 미래를 통째로 보게 된다(라벨이 부풀어 오른다).
    fut = df[(df["ts"] > t0) & (df["ts"] <= t0 + int(w_s))]

    rng = np.random.default_rng(seed)
    users = np.array(hist["user_id"].unique() if user_universe is None else user_universe)
    rng.shuffle(users)
    fit_users = set(users[: int(len(users) * prior_frac)].tolist())

    # L2(=카테고리) 사전: fit 인구의 연속 구매 간격 평균. eval 인구와 서로소(자기참조 누수 차단).
    fit = hist[hist["user_id"].isin(fit_users)]
    cat_cycle: dict[str, float] = {}
    for cat, g in fit.groupby("category_code"):
        gaps = []
        for _, ug in g.groupby("user_id"):
            ts = ug["ts"].to_numpy()
            if len(ts) >= 2:
                gaps.extend(np.diff(np.sort(ts)).tolist())
        if len(gaps) >= 5:
            cat_cycle[cat] = float(np.mean(gaps))
    global_cycle = float(np.mean(list(cat_cycle.values()))) if cat_cycle else 30 * DAY

    if pop == "fit":
        ev = hist[hist["user_id"].isin(fit_users)]
    elif pop == "all":
        ev = hist
    else:
        ev = hist[~hist["user_id"].isin(fit_users)]
    agg = ev.groupby(["user_id", "category_code"])["ts"].agg(["count", "min", "max"])
    fut_pairs = set(map(tuple, fut[["user_id", "category_code"]].drop_duplicates().to_numpy()))

    uids = agg.index.get_level_values(0)
    cnames = agg.index.get_level_values(1)
    cats = sorted(cnames.unique())
    if cat_ix is None:
        cat_ix = {c: i for i, c in enumerate(cats)}
    r = {
        "k": agg["count"].to_numpy().astype(np.float64),
        "t_first": agg["min"].to_numpy().astype(np.float64),
        "t_last": agg["max"].to_numpy().astype(np.float64),
        "cat": np.array([cat_ix[c] for c in cnames], dtype=np.int32),
        "cycle": np.array([cat_cycle.get(c, global_cycle) for c in cnames], dtype=np.float64),
        "uid": uids.to_numpy(),
        "cname": cnames.to_numpy(),
    }
    r["T"] = t0 - r["t_first"]          # 관측기간 (배포 span)
    r["dt"] = t0 - r["t_last"]          # 최근성 — v1 에서는 구조적으로 0이던 축
    r["y"] = np.array([1 if p in fut_pairs else 0 for p in zip(uids, cnames)], dtype=np.int8)
    return r, t0, len(cat_cycle), len(cats)


def attach_browse(r, browse, t0: float):
    """평가행에 t0 이전 조회/장바구니 요약을 붙인다(미래 누수 없음)."""
    import pandas as pd

    b = browse[browse["ts"] <= t0]
    key = pd.DataFrame({"user_id": r["uid"], "category_code": r["cname"]})
    out = {}
    for etype, tag in (("view", "v"), ("cart", "c")):
        sub = b[b["event_type"] == etype]
        g = sub.groupby(["user_id", "category_code"])["ts"]
        stats = pd.DataFrame({f"n_{tag}": g.size(), f"last_{tag}": g.max()}).reset_index()
        for win_d in (1, 7):
            lim = t0 - win_d * DAY
            gw = sub[sub["ts"] > lim].groupby(["user_id", "category_code"])["ts"].size()
            stats = stats.merge(gw.rename(f"n_{tag}{win_d}d").reset_index(),
                                on=["user_id", "category_code"], how="left")
        key = key.merge(stats, on=["user_id", "category_code"], how="left")
    for tag in ("v", "c"):
        out[f"n_{tag}"] = key[f"n_{tag}"].fillna(0.0).to_numpy(dtype=np.float64)
        out[f"n_{tag}1d"] = key[f"n_{tag}1d"].fillna(0.0).to_numpy(dtype=np.float64)
        out[f"n_{tag}7d"] = key[f"n_{tag}7d"].fillna(0.0).to_numpy(dtype=np.float64)
        # 조회 이력이 없으면 "아주 오래됨"으로 둔다(0 으로 두면 방금 본 것과 구분이 안 된다).
        last = key[f"last_{tag}"].to_numpy(dtype=np.float64)
        out[f"dt_{tag}"] = np.where(np.isnan(last), 90 * DAY, t0 - np.nan_to_num(last))
    r.update(out)
    return r


# ── 모델 (기본 팔은 기기 상태 (k, t_first, t_last) + 카테고리 사전만 사용) ─────
def m_mpg(r, w_s, beta_s=None):
    beta = np.full_like(r["T"], DEFAULT_BETA_S) if beta_s is None else beta_s
    bt = beta + r["T"]
    return 1.0 - (bt / (bt + w_s)) ** (DEFAULT_ALPHA + r["k"])


def m_weibull(r, w_s, m, beta_s=None, alpha=DEFAULT_ALPHA):
    """개인 주기 η 를 사후평균으로 잡고 **경과 dt 를 조건**으로 건 Weibull 잔여수명.
    m=1 이면 지수(무기억) — dt 가 사라져 mPG 와 같은 줄세우기가 된다."""
    beta = np.full_like(r["T"], DEFAULT_BETA_S) if beta_s is None else beta_s
    eta = (beta + r["T"]) / (alpha + r["k"])
    dt = np.maximum(r["dt"], 1.0)
    return 1.0 - np.exp(np.clip((dt / eta) ** m - ((dt + w_s) / eta) ** m, -50.0, 0.0))


# ── 지표 ──────────────────────────────────────────────────────────────────────
def auc_np(score: np.ndarray, y: np.ndarray) -> float:
    """동점을 0.5 로 처리하는 rank 기반 AUC(= Mann-Whitney U)."""
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i : j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    rk = np.empty(len(s), dtype=np.float64)
    rk[order] = ranks
    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (rk[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def boot_ci(sa, sb, y, n_boot=2000, seed=20260905):
    """paired row-level 부트스트랩으로 AUC(b) − AUC(a) 의 95%CI."""
    rng = np.random.default_rng(seed)
    n = len(y)
    d = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        d[i] = 0.0 if yy.sum() in (0, n) else auc_np(sb[idx], yy) - auc_np(sa[idx], yy)
    d.sort()
    return float(d[int(0.025 * n_boot)]), float(d[int(0.975 * n_boot) - 1])


def fusion_features(r, w_s, cat_rate, m=0.9):
    """기기에서 실제로 계산 가능한 피처만. 각각이 기기 상태 하나에 대응한다:
    구매 (k,t_first,t_last) · 조회/장바구니 마지막 시각과 최근 카운트 · L2 기저율(FL 채널).
    ⛔ 여기 없는 피처는 기기가 못 만든다 — 상한 비교를 정직하게 하려면 늘리지 말 것."""
    p = np.clip(m_weibull(r, w_s, m), 1e-6, 1 - 1e-6)
    feats = [np.log(p / (1 - p)),                 # 구매 모델 로짓
             np.exp(-r["dt"] / (7 * DAY)),        # 구매 최근성
             np.log1p(r["k"])]                    # 빈도
    names = ["logit_weibull", "recency_buy", "log1p_k"]
    if "dt_v" in r:
        feats += [np.exp(-r["dt_v"] / (3 * DAY)), np.exp(-r["dt_c"] / (3 * DAY)),
                  np.log1p(r["n_v7d"]), np.log1p(r["n_c7d"])]
        names += ["recency_view", "recency_cart", "log1p_view7d", "log1p_cart7d"]
    feats.append(np.log(np.maximum(cat_rate, 1e-4)))  # L2 기저율 — FL 사전 채널이 나르는 값
    names.append("log_cat_rate")
    return np.column_stack(feats), names


def fit_fusion(X, names, y, tr, keep: list[str]):
    """로지스틱 융합 헤드 — 파라미터 1~8개. 이 계수가 곧 FL 로 학습·배포할 페이로드다.
    `keep` 로 피처 부분집합을 골라 **이득이 어느 피처에서 나오는지** 분해한다."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    cols = [names.index(k) for k in keep]
    Xs = X[:, cols]
    sc = StandardScaler().fit(Xs[tr])
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(Xs[tr]), y[tr])
    coef = dict(zip(keep, (clf.coef_[0] / sc.scale_).round(4).tolist()))
    return clf.decision_function(sc.transform(Xs)), coef


def cat_base_rate(r, y, tr):
    """L2 기저율 — **train 행에서만** 추정(누수 차단). 표본 적은 L2 는 전역률로 수축.
    배포에서는 FL 구매통계 채널이 나르는 값이다(새 채널이 아니라 기존 채널의 재해석)."""
    n_cat = int(r["cat"].max()) + 1
    num, den = np.zeros(n_cat), np.zeros(n_cat)
    np.add.at(num, r["cat"][tr], y[tr])
    np.add.at(den, r["cat"][tr], 1.0)
    prior = y[tr].mean()
    return ((num + 20.0 * prior) / (den + 20.0))[r["cat"]]


def user_split(r, seed: int, test_frac: float = 0.3):
    """⛔ 행 단위로 쪼개면 같은 유저가 train·test 양쪽에 들어가 누수가 된다
    (한 유저가 여러 카테고리 행을 갖는다). **유저 단위**로 자른다."""
    uids = np.unique(r["uid"])
    rng = np.random.default_rng(seed)
    rng.shuffle(uids)
    test_users = set(uids[: int(len(uids) * test_frac)].tolist())
    is_te = np.array([u in test_users for u in r["uid"]])
    return np.where(~is_te)[0], np.where(is_te)[0]


def temporal_check(df, browse, w_s, prior_frac, seed, gap_s, boot, shapes_best=0.9):
    """⛔ **시간 일반화** — (24)가 미결로 남긴 것. 융합 계수를 **이른 스냅샷의 fit 인구**에서
    적합하고 **늦은 스냅샷의 eval 인구**에서 평가한다. 유저도 시간도 분리된다.

    이게 배포의 실제 모양이다: FL 은 과거 라운드에서 계수를 배워 미래에 쓴다. (24)의
    +0.013~0.025 는 같은 시점 스냅샷 안에서 잰 값이라, 그 숫자가 시간을 건너 남는지는
    **아직 답이 없었다**. 여기서 답한다 — 안 남으면 FL 배선을 지어도 소용없다."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    t_max = int(df["ts"].max())
    t0_eval = t_max - int(w_s)
    t0_fit = t0_eval - int(gap_s)
    universe = df["user_id"].unique()
    all_cats = sorted(df["category_code"].unique())
    cat_ix = {c: i for i, c in enumerate(all_cats)}

    fit_r, _, _, _ = build_rows(df, w_s, prior_frac, seed, t0=t0_fit, pop="fit",
                                user_universe=universe, cat_ix=cat_ix)
    ev_r, _, _, _ = build_rows(df, w_s, prior_frac, seed, t0=t0_eval, pop="eval",
                               user_universe=universe, cat_ix=cat_ix)
    if browse is not None:
        fit_r = attach_browse(fit_r, browse, t0_fit)
        ev_r = attach_browse(ev_r, browse, t0_eval)
    yf, ye = fit_r["y"].astype(np.int64), ev_r["y"].astype(np.int64)
    print(f"\n=== 시간 일반화 (w={w_s/DAY:.0f}d, 스냅샷 간격 {gap_s/DAY:.0f}일) ===")
    print(f"  적합: t0={t0_fit} · fit 인구 {len(yf):,}행 · 양성 {yf.mean():.2%}")
    print(f"  평가: t0={t0_eval} · eval 인구 {len(ye):,}행 · 양성 {ye.mean():.2%}")
    print(f"  유저 교집합 {len(set(fit_r['uid']) & set(ev_r['uid'])):,} (0 이어야 한다)")

    # L2 기저율은 **적합 스냅샷에서** 만든다 — 카테고리 '이름'으로 옮긴다(인덱스 아님).
    rate_by_name: dict = {}
    prior = float(yf.mean())
    import collections

    num = collections.Counter()
    den = collections.Counter()
    for c, yy in zip(fit_r["cname"], yf):
        num[c] += int(yy)
        den[c] += 1
    for c in set(fit_r["cname"]) | set(ev_r["cname"]):
        rate_by_name[c] = (num[c] + 20.0 * prior) / (den[c] + 20.0)
    fit_rate = np.array([rate_by_name[c] for c in fit_r["cname"]])
    ev_rate = np.array([rate_by_name[c] for c in ev_r["cname"]])

    Xf, names = fusion_features(fit_r, w_s, fit_rate, m=shapes_best)
    Xe, _ = fusion_features(ev_r, w_s, ev_rate, m=shapes_best)
    base_ev = m_mpg(ev_r, w_s)
    ladder = [("③ 기저율+k+최근성", ["log_cat_rate", "log1p_k", "recency_buy"])]
    if browse is not None:
        ladder.append(("⑤ + 조회/장바구니",
                       ["log_cat_rate", "log1p_k", "recency_buy", "logit_weibull",
                        "recency_view", "recency_cart", "log1p_view7d", "log1p_cart7d"]))
    out = {"t0_fit": t0_fit, "t0_eval": t0_eval, "gap_days": gap_s / DAY,
           "n_fit": int(len(yf)), "n_eval": int(len(ye)),
           "auc_deployed": float(auc_np(base_ev, ye)), "arms": {}}
    print(f"  {'mpg(배포 현행)':<26} {out['auc_deployed']:.4f}  (기준)")
    print(f"  {'weibull m=0.9':<26} "
          f"{auc_np(m_weibull(ev_r, w_s, shapes_best), ye):.4f}")
    for label, keep in ladder:
        cols = [names.index(k) for k in keep]
        sc = StandardScaler().fit(Xf[:, cols])
        clf = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(Xf[:, cols]), yf)
        s = clf.decision_function(sc.transform(Xe[:, cols]))
        a = auc_np(s, ye)
        lo, hi = boot_ci(base_ev, s, ye, n_boot=boot, seed=seed)
        d = a - out["auc_deployed"]
        print(f"  {label:<26} {a:.4f}  Δ={d:+.4f} 95%CI[{lo:+.4f},{hi:+.4f}] "
              f"{'유의' if lo > 0 else '비유의'}")
        out["arms"][label] = {"auc": float(a), "delta": float(d), "ci95": [lo, hi],
                              "coef": dict(zip(keep, (clf.coef_[0] / sc.scale_).round(4).tolist()))}
    return out


def lgbm_auc(X, y, cat_cols, seed):
    import lightgbm as lgb

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    te, tr = idx[: int(0.3 * len(y))], idx[int(0.3 * len(y)) :]
    ds = lgb.Dataset(X[tr], label=y[tr], categorical_feature=cat_cols, free_raw_data=False)
    bst = lgb.train(dict(objective="binary", metric="auc", verbosity=-1, seed=7, num_leaves=31,
                         min_data_in_leaf=50, learning_rate=0.05), ds, num_boost_round=300)
    return auc_np(bst.predict(X[te]), y[te]), bst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/tmp/rees46")
    ap.add_argument("--windows", default="3,7")
    ap.add_argument("--shapes", default="0.6,0.8,0.9,1.0")
    ap.add_argument("--prior-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--no-dedup", action="store_true", help="세션 dedup 끄기(v1 동작 비교용)")
    ap.add_argument("--with-browse", action="store_true", help="조회/장바구니 신호 팔 추가")
    ap.add_argument("--fusion", action="store_true", help="로지스틱 융합 헤드(기기 실행 가능) 평가")
    ap.add_argument("--temporal", action="store_true",
                     help="시간 일반화: 이른 스냅샷에서 계수 적합 → 늦은 스냅샷에서 평가")
    ap.add_argument("--gap-days", type=float, default=7.0, help="두 스냅샷 간격(일)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--taxonomy-l2", action="store_true",
                    help="REES46 카테고리를 우리 중분류(L2)로 사상해 같은 어블레이션을 돌린다")
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()

    import pandas as pd

    df = load_orders(a.cache_dir, dedup=not a.no_dedup)
    if a.taxonomy_l2:
        df = to_our_taxonomy(df, pathlib.Path(__file__).resolve().parent / "rees46_category_bridge.json")
    windows = [int(x) for x in a.windows.split(",")]
    shapes = [float(x) for x in a.shapes.split(",")]

    built = {}
    for wd in windows:
        built[wd] = build_rows(df, wd * DAY, a.prior_frac, a.seed)

    browse = None
    if a.with_browse:
        pairs = pd.concat(
            [pd.DataFrame({"user_id": built[wd][0]["uid"],
                           "category_code": built[wd][0]["cname"]}) for wd in windows],
            ignore_index=True,
        )
        print(f"브라우징 스캔 대상 쌍 {pairs.drop_duplicates().shape[0]:,}")
        browse = browse_events(a.cache_dir, pairs)
        print(f"조회/장바구니 이벤트 {len(browse):,}")

    out = {"dedup": not a.no_dedup, "prior_frac": a.prior_frac, "seed": a.seed,
           "n_boot": a.boot, "shapes": shapes, "with_browse": a.with_browse,
           "taxonomy": "our-l2" if a.taxonomy_l2 else "rees46", "windows": {}}

    for wd in windows:
        w_s = wd * DAY
        r, t0, n_learned, n_cats = built[wd]
        if browse is not None:
            r = attach_browse(r, browse, t0)
        y = r["y"].astype(np.int64)
        n, pos = len(y), int(y.sum())
        print(f"\n[w={wd}d] 스냅샷 t0={t0} · 평가쌍 {n:,} · 양성 {pos:,} ({pos/n:.2%}) · "
              f"학습된 카테고리 사전 {n_learned}/{n_cats}")
        print(f"  recency dt(일): p25={np.percentile(r['dt'],25)/DAY:.1f} "
              f"p50={np.percentile(r['dt'],50)/DAY:.1f} p95={np.percentile(r['dt'],95)/DAY:.1f}"
              f"   · k=1 비율 {np.mean(r['k']==1):.1%}")

        scores = {
            "rfm_k_over_T": r["k"] / np.maximum(r["T"], DAY),
            "recency_only": -r["dt"],
            "mpg(배포 현행)": m_mpg(r, w_s),
            "mpg_cat(L2 사전)": m_mpg(r, w_s, beta_s=DEFAULT_ALPHA * r["cycle"]),
        }
        for m in shapes:
            scores[f"weibull m={m}"] = m_weibull(r, w_s, m)
        if browse is not None:
            scores["view_recency_only"] = -r["dt_v"]
            scores["cart_recency_only"] = -r["dt_c"]
            scores["n_view_7d"] = r["n_v7d"]
            # 구매 모델 × 조회 최근성 — 기기에서 곱 하나로 되는 최소 결합
            best_m = 0.9
            scores["weibull0.9 × view_recency"] = m_weibull(r, w_s, best_m) * np.exp(
                -r["dt_v"] / (7 * DAY)
            )

        aucs = {k: auc_np(v, y) for k, v in scores.items()}
        base = scores["mpg(배포 현행)"]
        print(f"  {'무정보':<30} 0.5000")
        for k in sorted(aucs, key=lambda x: -aucs[x]):
            mark = "  <- 배포 현행" if k.startswith("mpg(") else ""
            print(f"  {k:<30} {aucs[k]:.4f}{mark}")

        best = max(aucs, key=lambda k: aucs[k])
        lo, hi = boot_ci(base, scores[best], y, n_boot=a.boot, seed=a.seed)
        print(f"  ⇒ 최고={best} Δ={aucs[best]-aucs['mpg(배포 현행)']:+.4f} "
              f"95%CI[{lo:+.4f},{hi:+.4f}] {'유의' if lo > 0 else '비유의'}")

        # LightGBM 상한 — (a) 기기 구매상태만 (b) + 브라우징
        ceil = {}
        feats_buy = [r["k"], r["T"] / DAY, r["dt"] / DAY,
                     r["k"] / np.maximum(r["T"], DAY) * DAY, r["cat"]]
        ceil["purchase_only"], _ = lgbm_auc(np.column_stack(feats_buy), y, [4], a.seed + wd)
        print(f"  {'LightGBM(구매상태만)':<30} {ceil['purchase_only']:.4f}  <- 같은 정보 상한")
        if browse is not None:
            feats_all = feats_buy[:-1] + [
                r["n_v"], r["n_v1d"], r["n_v7d"], r["dt_v"] / DAY,
                r["n_c"], r["n_c1d"], r["n_c7d"], r["dt_c"] / DAY, r["cat"],
            ]
            ceil["with_browse"], bst = lgbm_auc(np.column_stack(feats_all), y,
                                                 [len(feats_all) - 1], a.seed + wd)
            names = ["k", "T", "dt", "rate", "n_view", "n_view_1d", "n_view_7d", "dt_view",
                     "n_cart", "n_cart_1d", "n_cart_7d", "dt_cart", "cat"]
            imp = sorted(zip(names, bst.feature_importance("gain")), key=lambda x: -x[1])
            print(f"  {'LightGBM(+조회/장바구니)':<30} {ceil['with_browse']:.4f}  <- 정보 추가 상한")
            print("    gain 상위: " + ", ".join(f"{k}={v/sum(i[1] for i in imp):.1%}"
                                                 for k, v in imp[:6]))

        # ── 융합 헤드: 같은 test 분할에서 전 팔을 다시 잰다(적합 팔과 공정 비교) ──
        fusion = {}
        if a.fusion:
            tr, te = user_split(r, a.seed + 100 + wd)
            yt = y[te]
            X, names = fusion_features(r, w_s, cat_base_rate(r, y, tr))
            print(f"  ── 어블레이션 (유저 단위 분할: train {len(tr):,} / test {len(te):,}) ──")
            ladder = [
                ("① L2 기저율만", ["log_cat_rate"]),
                ("② + 빈도 k", ["log_cat_rate", "log1p_k"]),
                ("③ + 구매 최근성", ["log_cat_rate", "log1p_k", "recency_buy"]),
                ("④ + weibull 로짓", ["log_cat_rate", "log1p_k", "recency_buy", "logit_weibull"]),
            ]
            if browse is not None:
                ladder.append(("⑤ + 조회/장바구니",
                               ["log_cat_rate", "log1p_k", "recency_buy", "logit_weibull",
                                "recency_view", "recency_cart", "log1p_view7d", "log1p_cart7d"]))
            arms = {"mpg(배포 현행)": base, "weibull m=0.9": scores["weibull m=0.9"]}
            coefs = {}
            for label, keep in ladder:
                s, c = fit_fusion(X, names, y, tr, keep)
                arms[label] = s
                coefs[label] = c
            test_auc = {k: auc_np(v[te], yt) for k, v in arms.items()}
            for k, v in arms.items():
                if k == "mpg(배포 현행)":
                    print(f"  {k:<26} {test_auc[k]:.4f}  (기준)")
                    continue
                lo2, hi2 = boot_ci(base[te], v[te], yt, n_boot=a.boot, seed=a.seed)
                d = test_auc[k] - test_auc["mpg(배포 현행)"]
                print(f"  {k:<26} {test_auc[k]:.4f}  Δ={d:+.4f} "
                      f"95%CI[{lo2:+.4f},{hi2:+.4f}] {'유의' if lo2 > 0 else '비유의'}")
                fusion[k] = {"auc": test_auc[k], "delta": d, "ci95": [lo2, hi2]}
            for label, c in coefs.items():
                print(f"    계수 {label}: {json.dumps(c, ensure_ascii=False)}")
            fusion["coefs"] = coefs
            fusion["test_auc"] = test_auc
            fusion["n_train"], fusion["n_test"] = int(len(tr)), int(len(te))

        out["windows"][f"w{wd}d"] = {
            "fusion": fusion,
            "t0": int(t0), "n": n, "pos": pos, "pos_rate": pos / n,
            "learned_categories": n_learned, "k1_frac": float(np.mean(r["k"] == 1)),
            "auc": {k: float(v) for k, v in aucs.items()},
            "best": best, "delta_vs_deployed": float(aucs[best] - aucs["mpg(배포 현행)"]),
            "ci95": [lo, hi], "lgbm_ceiling": {k: float(v) for k, v in ceil.items()},
        }

    if a.temporal:
        out["temporal"] = {
            f"w{wd}d": temporal_check(df, browse, wd * DAY, a.prior_frac, a.seed,
                                      a.gap_days * DAY, a.boot)
            for wd in windows
        }

    if a.json_out:
        pathlib.Path(a.json_out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"\n-> {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
