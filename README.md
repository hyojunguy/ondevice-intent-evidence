# Evidence for *Privacy-Preserving On-Device Commercial Intent Inference*

This repository holds the measurement artifacts and verification gates cited by the paper, so
that every number in it can be traced to the file that produced it. It is an evidence bundle,
not the product and not the data: the SDK crates, the category taxonomy, the model weights and
**every dataset** are absent.

**No dataset here is redistributable, including our own.** The AI-Hub corpora are built under
projects of Korea's National Information Society Agency, whose terms permit releasing trained
models and weights but not data extracted or re-processed from the originals. Our own material is
no freer: the Korean query set is a machine translation of Amazon ESCI queries, the taxonomy was
harvested from commercial category trees, and the synthetic utterances were generated conditioned
on both. So there are no corpus rows, no translated queries, no utterance lists, no leaf
inventories and no quoted text anywhere in this repository. Where an analysis would have printed
a list of terms, it reports a count.

**What you can and cannot do with this.** You can audit every figure in the paper against its
ledger entry and the sha256 in `artifacts/MANIFEST.json`, and read the exact code that produced
it. You cannot regenerate those figures from this bundle alone — `run_all.py` reports
`UNMEASURED` wherever an artifact is absent rather than inventing a verdict. Regenerating them
means obtaining the sources independently, which for the AI-Hub figures means your own approved
access.

**Identifiers are neutralized.** Crate and namespace names in this published copy are rewritten
to a vendor-neutral `oicr-*` by `sanitize_evidence_export.py`, and local absolute paths are
replaced. The working tree uses internal names; nothing about the measurements changes.

## What is here

| Path | What it is |
|---|---|
| `gates/` | The verification gates. Each one adjudicates one claim and prints PASS/FAIL; `run_all.py` runs the suite. These decide the numbers — the paper reports their output, it does not assert alongside them. |
| `gates/budgets.json` | The declared budgets (size, latency, privacy) the gates check against. |
| `artifacts/MANIFEST.json` | Provenance of the shipped artifacts: source model, vocabulary size, byte counts, sha256. |
| `artifacts/ui-facts.json` | The generated fact sheet the demo UI reads, so screen and paper cannot drift apart. |
| `experiments/distill-ko/` | Product-surface accuracy and the promotion-gate decision for the shipped build. |
| `experiments/purchase-lift/` | The purchase-probability track: the degenerate-evaluation diagnosis, the second harness, the ablation, temporal generalization, the LightGBM comparison, and the λ_p sweep. |
| `experiments/real-ko-commerce/` | The product surface measured against real Korean commerce text (AI-Hub datasets 71603, 102, 98): the category mapping we wrote, the harnesses, and aggregate results. Each accuracy sits beside a permutation baseline under the identical mapping, because a broad mapping makes a top-5 hit easy. Dataset 98 carries no per-utterance category, so it appears as lexical coverage only. |
| `experiments/percept-vision*/`, `experiments/user-*/`, `experiments/real-ko-bench/`, `experiments/esci-ko/` | The remaining measured axes cited in the paper, including the negative results. |

## Reading the purchase-probability files in order

The paper reports a measurement mistake of ours, and these files are that sequence:

1. `real-rees46-results-2026-09-04.json` — first pass on a real log. AUC 0.53–0.59.
2. `diagnose_real.py` — why. 62.6% of evaluation rows had a single prior purchase; 90.4% positive
   rate with 46.5% forced by window truncation; 79.8% of pairs dropped for being censored.
3. `eval_real_v2.py` + `real-v2-ablation-2026-09-05.json` — the second harness. The same formula,
   unchanged, scores 0.7425 / 0.6915.
4. `real-v2-temporal-2026-09-05.json` — does it survive being fit in the past and used in the
   future. Most of it does; browse/cart coefficients do not.
5. `lgbm_benchmark-2026-09-04.json` — the ceiling. A gradient-boosted model on the same features
   ties, which is the argument that information rather than capacity is the constraint.

⛔ The two harnesses ask different questions. Do not read 0.53→0.74 as an improvement; it is a
different task. That is the point of including both.

## Caveats that travel with the data

- The REES46 mirror and the original Kaggle dataset carry no stated license. These files are
  measurements *over* that data, not the data itself, and nothing here redistributes it.
- The observation window is 21 days, which is why the cold-start cycle prior is reported as
  unbuilt rather than estimated.
- Synthetic-sequence numbers are statements about mechanism direction only. Where a figure comes
  from synthetic data the file says so, and the paper repeats it.
- Some gates report UNMEASURED on a machine lacking the relevant toolchain (Android NDK, browser
  runtime). That is the honest state of that machine, not a silent pass.
- `attribution_gate.py` is FAIL at the time of writing: the declared 28-bucket attribution axis
  and the 60-class shipped intent head have drifted apart. The paper reports this rather than
  waiting for green.

## Envelope this bundle corresponds to

The paper's constraints are a 3 MiB deployment payload, 20 ms p95 Tier-0 latency, and zero raw
egress. `gates/budgets.json` here is the file the gates read, so those limits are checkable
rather than asserted: `total.limit_bytes = 3145728`, `latency.p95_ms = 20`.

Shipping artifacts are 4-bit (taxonomy v2, catalog v3). `experiments/distill-ko/quant4_surface.json`
is the A/B that measures what that cost — including the axis that pays most (ad creative to leaf
top-1, −1.00 pp), not only the product surface (−0.02 pp).

One gate is red: `attribution_gate.py` (a 28-bucket budget axis against a 60-class intent head).
It is reported in the paper rather than waited out.

## License

The measurement artifacts and gate scripts here are released under the MIT License.
