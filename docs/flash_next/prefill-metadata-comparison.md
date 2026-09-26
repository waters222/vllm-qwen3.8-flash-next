# Prefill metadata reuse — completed full comparison

TP4/MTP4, C2/C4, budget 2048. Three measured repeats after warmup; aggregate tokens/s. Staging and metadata-reuse arms have identical config and prompt hashes. Prefix reads are skipped; candidate cache writes/offload stay enabled. Metadata reuse is default-off and not established as a consistent improvement.

## Prefill
| Case | Old baseline | Staging | Staging + metadata reuse | vs staging |
|---|---:|---:|---:|---:|
| prefill-2048-c2 | 3842.7 | 3072.8 | 2731.0 | -11.1% |
| prefill-8192-c2 | 3730.4 | 3624.6 | 3677.9 | +1.5% |
| prefill-16384-c2 | 3793.5 | 3485.1 | 3439.0 | -1.3% |
| prefill-31740-c2 | 3339.9 | 3450.6 | 3492.9 | +1.2% |
| prefill-2048-c4 | 3976.6 | 3161.7 | 3697.1 | +16.9% |
| prefill-8192-c4 | 3904.9 | 3619.5 | 3553.6 | -1.8% |
| prefill-16384-c4 | 3788.7 | 3575.1 | 3339.9 | -6.6% |
| prefill-31740-c4 | 3318.7 | 3502.9 | 3211.2 | -8.3% |

## Decode after all first outputs
| Case | Old baseline | Staging | Staging + metadata reuse | vs staging |
|---|---:|---:|---:|---:|
| tg-short-math-c2 | 282.3 | 306.5 | 309.1 | +0.9% |
| tg-short-code-c2 | 315.1 | 337.1 | 344.8 | +2.3% |
| tg-short-prose-c2 | 188.6 | 207.4 | 198.8 | -4.2% |
| tg-long-31740-c2 | 125.8 | 164.0 | 173.7 | +5.9% |
| tg-short-math-c4 | 414.6 | 504.6 | 474.6 | -5.9% |
| tg-short-code-c4 | 426.2 | 547.1 | 533.9 | -2.4% |
| tg-short-prose-c4 | 266.9 | 329.0 | 329.5 | +0.1% |
| tg-long-31740-c4 | 146.4 | 223.8 | 224.9 | +0.5% |

## Generation including prefill
| Case | Old baseline | Staging | Staging + metadata reuse | vs staging |
|---|---:|---:|---:|---:|
| tg-short-math-c2 | 249.8 | 257.9 | 287.8 | +11.6% |
| tg-short-code-c2 | 294.6 | 284.9 | 287.6 | +1.0% |
| tg-short-prose-c2 | 169.2 | 198.4 | 175.5 | -11.5% |
| tg-long-31740-c2 | 56.9 | 64.5 | 68.5 | +6.1% |
| tg-short-math-c4 | 396.7 | 449.8 | 439.7 | -2.2% |
| tg-short-code-c4 | 387.2 | 482.1 | 472.1 | -2.1% |
| tg-short-prose-c4 | 235.1 | 299.6 | 306.2 | +2.2% |
| tg-long-31740-c4 | 62.8 | 69.9 | 71.6 | +2.4% |

## Interpretation

Reuse activated on all ranks: 2,388 exact metadata reads and 13,134 reuses each. It did not consistently close prefill gaps: C4/2k improved about17%, while C2/2k worsened about11% versus staging. Native admission/first-output patterns vary between repeats. These separate runs cannot isolate a small timing effect; a paired 2k control with exact scheduler chunk logs is recorded separately. No failed numerical gate is reclassified.

The preceding profile found substantial waiting at metadata readback APIs, but this overlaps GPU work. Rank0 C4/2k had about186ms total GPU idle gaps versus about699ms in twelve-byte copy API calls; removing those calls cannot be assumed to save699ms. Profile timings include instrumentation and are not throughput qualification.

## Evidence

- Container `0934a13ca4b05138032f2463a4cf8c7e2b0a9901119278a47d9f7fc8de39ab08`, exit0 at `2026-09-24T02:04:08.823887501Z`;76 source/helper hashes and serving identity verified.
- `runs/runtime-ab-prefill-metadata-v1-{launch,screen,collection,comparison,source-audit}.json`.
- Raw screen SHA256 `cd0fe869eb6f8a89edb881f6082901de4cbfcc0939768748f79ead3b1ed2bd67`.
- Component31 tests plus exact graph cases: `runs/qsa-prefill-metadata-v2-collection.json`; native lifecycle: `runs/reduced-prefill-metadata-v1-collection.json`.
- Profile/clock analysis: `runs/runtime-ab-prefill-profile-v1-trace-analysis.json` and `runs/runtime-ab-prefill-profile-v1-gpu-timeline.json`.

Raw `runs/` evidence references in this record refer to the development archive;
that archive and site orchestration are not bundled with this public source tree.
