# Completed TP4/MTP4 staging comparison — 2026-09-24

Packed verification staging activated on all four ranks and completed all 64 throughput batches. All rates below are aggregate tokens/s, medians of three measured repeats after warmup. C2/C4, budget 2048, MTP4, identical prompt hashes; prefix reads skipped while candidate cache/offload writes remain enabled. The repeated control fell back to native QSA and supplied no staging speedup.

## Prefill

| Workload | Old | Corrected control | Repeated control | Staging | vs old |
|---|---:|---:|---:|---:|---:|
| prefill-2048-c2 | 3842.7 | 3156.1 | 2713.8 | 3072.8 | -20.0% |
| prefill-8192-c2 | 3730.4 | 3520.6 | 3659.0 | 3624.6 | -2.8% |
| prefill-16384-c2 | 3793.5 | 3390.9 | 3480.6 | 3485.1 | -8.1% |
| prefill-31740-c2 | 3339.9 | 3472.5 | 3452.3 | 3450.6 | +3.3% |
| prefill-2048-c4 | 3976.6 | 3546.1 | 3160.5 | 3161.7 | -20.5% |
| prefill-8192-c4 | 3904.9 | 3517.3 | 3473.2 | 3619.5 | -7.3% |
| prefill-16384-c4 | 3788.7 | 3112.5 | 3536.9 | 3575.1 | -5.6% |
| prefill-31740-c4 | 3318.7 | 3234.7 | 3514.1 | 3502.9 | +5.6% |

## Generation after all first outputs

| Workload | Old | Corrected control | Repeated control | Staging | vs old |
|---|---:|---:|---:|---:|---:|
| tg-short-math-c2 | 282.3 | 287.3 | 277.7 | 306.5 | +8.6% |
| tg-short-code-c2 | 315.1 | 322.2 | 322.4 | 337.1 | +7.0% |
| tg-short-prose-c2 | 188.6 | 190.1 | 194.6 | 207.4 | +9.9% |
| tg-long-31740-c2 | 125.8 | 106.3 | 129.4 | 164.0 | +30.4% |
| tg-short-math-c4 | 414.6 | 421.0 | 423.7 | 504.6 | +21.7% |
| tg-short-code-c4 | 426.2 | 467.5 | 477.0 | 547.1 | +28.4% |
| tg-short-prose-c4 | 266.9 | 281.7 | 282.9 | 329.0 | +23.3% |
| tg-long-31740-c4 | 146.4 | 138.7 | 148.2 | 223.8 | +52.9% |

## Generation including prefill

| Workload | Old | Corrected control | Repeated control | Staging | vs old |
|---|---:|---:|---:|---:|---:|
| tg-short-math-c2 | 249.8 | 253.2 | 261.7 | 257.9 | +3.3% |
| tg-short-code-c2 | 294.6 | 300.2 | 290.8 | 284.9 | -3.3% |
| tg-short-prose-c2 | 169.2 | 182.7 | 186.9 | 198.4 | +17.2% |
| tg-long-31740-c2 | 56.9 | 54.5 | 59.1 | 64.5 | +13.4% |
| tg-short-math-c4 | 396.7 | 315.7 | 328.3 | 449.8 | +13.4% |
| tg-short-code-c4 | 387.2 | 421.2 | 453.1 | 482.1 | +24.5% |
| tg-short-prose-c4 | 235.1 | 256.6 | 258.3 | 299.6 | +27.4% |
| tg-long-31740-c4 | 62.8 | 61.2 | 62.3 | 69.9 | +11.3% |

## Findings and limits

All eight decode-only TG cases improve versus the old baseline in this run. Long-context TG reaches 164.0 at C2 and 223.8 at C4, versus 125.8 and 146.4. Prefill remains about 20% slower at 2k, with smaller deficits at 8k/16k; 31,740-token prefill improves. C2 short-code E2E throughput remains 3.3% lower. This does not establish overall production nonregression.

The separate eager-selection-order control completed eight C4/31,740/256 batches after throughput measurements. Native ordering produced 1/12 exact within-mode repeat pairs; canonical ordering produced 0/12. Median per-request acceptance lengths were 3.427 and 3.369. Eager ordering alone did not fix numerical variability. Captured decode ordering and tie membership were unchanged; these results do not isolate the remaining cause. Exact mismatches remain failures.

Staging preserves selected index order and passes exact component tests across six graph replay mutations, production cache strides/alignment, and real target/draft reduced probes with native cache lifecycle checks. It remains default-off: low-overlap 32k component cases are about 2% slower, C6 full runtime is unqualified, and full numerical/model-quality qualification remains open.

Old/new capacity, cache architecture, graph and prefill settings differ as recorded in the comparison JSON; this is a working-runtime comparison, not a single-variable kernel benchmark. Generated continuations and MTP acceptance vary. Three-repeat medians are not confidence bounds.

## Evidence

- Exact container: `fd5961f2a30bb521743821a572647294e310653b5e56bbe973ef67ddb3e5d325`, exit 0 at `2026-09-24T01:07:50.855099595Z`; 76 source/helper hashes verified, serving unchanged.
- `runs/runtime-ab-qsa-packed-v3-{launch,screen,collection,comparison}.json` and `runs/runtime-ab-qsa-packed-v3-selection-analysis.json`.
- Raw screen SHA256: `f3de167a7dddf3039226759f1b473d34cecd1a1bdf4f58e515a6d0238074ed12`.
- Production-layout component: `runs/qsa-verify-production-layout-v1-collection.json`; real target/draft lifecycle: `runs/reduced-qsa-staging-v2-collection.json`.

Raw `runs/` evidence references in this record refer to the development archive;
that archive and site orchestration are not bundled with this public source tree.
