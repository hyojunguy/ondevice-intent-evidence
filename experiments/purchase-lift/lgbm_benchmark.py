#!/usr/bin/env python3
"""mPG(closed-form, on-device) vs LightGBM(trained, off-device) — same synthetic
holdout, same features available, same AUC metric.

# 왜

mPG(수정 포아송-감마)를 고른 이유는 계획서(§2)에 적힌 대로 **모델 용량 우위가 아니라
온디바이스 제약**(폐쇄형 O(1) 갱신·리프당 ~10바이트·런타임 추론 불요)이다. 그런데 "그래서
raw 예측력은 얼마나 손해보나"는 여태 재본 적이 없었다 — 이 스크립트가 그 질문에 답한다.

같은 합성 데이터(`gen_purchase_synthetic.py`) · 같은 홀드아웃 라벨(`gap<=window`) ·
같은 AUC 구현(`eval_purchase_lift.auc`)을 재사용해 mPG 와 LightGBM 을 **공정하게** 겨룬다.
카테고리 정보 유무를 축으로 둘씩 짝짓는다(카테고리 모름 vs 모름, 앎 vs 앎) — 안 그러면
LightGBM 에게 mPG 보다 더 많은 정보를 주고 이겼다고 말하는 것이 된다.

⛔ 합성 데이터다(누수프로브 19.9%, 기준선 20% 근처 = 정상). 방향성 참고용 — 실 구매
로그 확보 전 "성능 우위/열위 주장"으로 인용 금지(README.md 와 같은 규율).
⛔ 여기서 재는 것은 **raw AUC** 뿐이다. 모델 크기·런타임 추론 비용(LightGBM 은 트리
앙상블을 어떤 형태로든 온디바이스에 실어야 하고, mPG 는 그 자체가 산술식이라 실을 것이
없다)은 별도 축이고 여기서 안 잰다 — mPG 를 고른 진짜 이유는 그 축이다.
"""
from __future__ import annotations
import argparse, json, pathlib, random, sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
assert (ROOT / "CONTRACT.md").exists(), f"ROOT 가 틀렸다: {ROOT}"
sys.path.insert(0, str(ROOT / "scripts"))
from eval_purchase_lift import load, p_buy, auc, DAY, KIND_CYCLES_D  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/purchase_synth_p1.jsonl")
    ap.add_argument("--windows", default="7,30")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()

    import lightgbm as lgb
    import numpy as np

    rows = load(a.data)
    print(f"rows={len(rows)}")

    rng = random.Random(a.seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    n_test = int(a.test_frac * len(idx))
    test_idx = set(idx[:n_test])
    train = [rows[i] for i in range(len(rows)) if i not in test_idx]
    test = [rows[i] for i in range(len(rows)) if i in test_idx]
    print(f"train={len(train)} test={len(test)} "
          f"(mPG needs no training fold — split exists only so LightGBM gets one)")

    kinds = sorted({r["kind"] for r in rows})
    kind_ix = {k: i for i, k in enumerate(kinds)}

    def feats(rs: list[dict], with_kind: bool):
        return np.array(
            [[r["k"], r["span"] / DAY] + ([kind_ix[r["kind"]]] if with_kind else []) for r in rs],
            dtype=float,
        )

    out = {"n": len(rows), "n_train": len(train), "n_test": len(test), "windows": {}}
    for wd in (int(x) for x in a.windows.split(",")):
        w_s = wd * DAY
        y_train = np.array([int(r["gap"] <= w_s) for r in train])
        y_test = np.array([int(r["gap"] <= w_s) for r in test])
        pos_rate = float(y_test.mean())

        lgbm_auc = {}
        for with_kind, label in ((False, "no_category"), (True, "with_category")):
            xtr, xte = feats(train, with_kind), feats(test, with_kind)
            cat_idx = [2] if with_kind else []
            dtr = lgb.Dataset(xtr, label=y_train, categorical_feature=cat_idx, free_raw_data=False)
            params = dict(objective="binary", metric="auc", verbosity=-1, seed=7,
                          num_leaves=15, min_data_in_leaf=20, learning_rate=0.05)
            bst = lgb.train(params, dtr, num_boost_round=200)
            pred = bst.predict(xte)
            lgbm_auc[label] = auc(list(zip(pred, y_test)))

        mpg_boot = auc([(p_buy(r["k"], r["span"], w_s), int(r["gap"] <= w_s)) for r in test])
        mpg_kind = auc([(p_buy(r["k"], r["span"], w_s, alpha=2.0,
                                beta_s=2.0 * KIND_CYCLES_D[r["kind"]] * DAY),
                          int(r["gap"] <= w_s)) for r in test])
        rfm = auc([(r["k"] / max(r["span"], DAY), int(r["gap"] <= w_s)) for r in test])

        print(f"\n[w={wd}d] test={len(test)} pos_rate={pos_rate:.1%}")
        print(f"  무정보(0.5)                       : 0.5000")
        print(f"  RFM(빈도)                          : {rfm:.4f}")
        print(f"  mPG boot(카테고리 모름)             : {mpg_boot:.4f}")
        print(f"  LightGBM(k,span — 카테고리 모름)    : {lgbm_auc['no_category']:.4f}")
        print(f"  mPG kind사전 오라클(카테고리 앎)     : {mpg_kind:.4f}")
        print(f"  LightGBM(k,span,kind — 카테고리 앎)  : {lgbm_auc['with_category']:.4f}")

        out["windows"][f"w{wd}d"] = {
            "pos_rate": pos_rate, "uninformed": 0.5, "rfm": rfm,
            "mpg_boot": mpg_boot, "lgbm_no_category": lgbm_auc["no_category"],
            "mpg_kind_oracle": mpg_kind, "lgbm_with_category": lgbm_auc["with_category"],
        }

    if a.json_out:
        pathlib.Path(a.json_out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"→ {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
