# Text-only TP4/MTP4 recovery comparison — completed

All three arms completed64/64 batches, exit0, no container OOM or scheduler preemptions. Rates are aggregate tokens/s, medians of three measured repeats after one warmup. Native cache retention remains enabled in both candidate arms.

## Prefill

| Case | Old | Regressed | Fixed 2048 | Fixed vs old |
|---|---:|---:|---:|---:|
| prefill-2048-c2 | 3842.7 | 1727.4 | 3156.1 | -17.9% |
| prefill-8192-c2 | 3730.4 | 2226.3 | 3520.6 | -5.6% |
| prefill-16384-c2 | 3793.5 | 2083.3 | 3390.9 | -10.6% |
| prefill-31740-c2 | 3339.9 | 1997.2 | 3472.5 | +4.0% |
| prefill-2048-c4 | 3976.6 | 2940.3 | 3546.1 | -10.8% |
| prefill-8192-c4 | 3904.9 | 3156.0 | 3517.3 | -9.9% |
| prefill-16384-c4 | 3788.7 | 3286.4 | 3112.5 | -17.8% |
| prefill-31740-c4 | 3318.7 | 3129.4 | 3234.7 | -2.5% |

## Generation after all first outputs

| Case | Old | Regressed | Fixed 2048 | Fixed vs old |
|---|---:|---:|---:|---:|
| tg-short-math-c2 | 282.3 | 217.5 | 287.3 | +1.7% |
| tg-short-code-c2 | 315.1 | 216.8 | 322.2 | +2.3% |
| tg-short-prose-c2 | 188.6 | 156.7 | 190.1 | +0.8% |
| tg-long-31740-c2 | 125.8 | 119.2 | 106.3 | -15.5% |
| tg-short-math-c4 | 414.6 | 401.6 | 421.0 | +1.5% |
| tg-short-code-c4 | 426.2 | 439.5 | 467.5 | +9.7% |
| tg-short-prose-c4 | 266.9 | 245.9 | 281.7 | +5.6% |
| tg-long-31740-c4 | 146.4 | 134.7 | 138.7 | -5.2% |

## Generation including prefill

| Case | Old | Regressed | Fixed 2048 | Fixed vs old |
|---|---:|---:|---:|---:|
| tg-short-math-c2 | 249.8 | 182.3 | 253.2 | +1.4% |
| tg-short-code-c2 | 294.6 | 205.8 | 300.2 | +1.9% |
| tg-short-prose-c2 | 169.2 | 151.2 | 182.7 | +8.0% |
| tg-long-31740-c2 | 56.9 | 42.5 | 54.5 | -4.2% |
| tg-short-math-c4 | 396.7 | 368.4 | 315.7 | -20.4% |
| tg-short-code-c4 | 387.2 | 418.5 | 421.2 | +8.8% |
| tg-short-prose-c4 | 235.1 | 232.5 | 256.6 | +9.1% |
| tg-long-31740-c4 | 62.8 | 58.7 | 61.2 | -2.6% |

## Interpretation and limits

Short generation recovered in C2 and C4. Prefill still has deficits up to17.9%; long generation is down15.5% at C2 and5.2% at C4. Recovery is incomplete. The fixes remove the472 per-request prefill threshold, fuse uniform speculative GDN metadata staging, and prevent local KV offload from accidentally enabling embedding-only multimodal inputs.

Long-context MTP acceptance and generated outputs differ between repeats and arms. Stream intervals while all requests are active improved, but these are not GPU kernel timings. Lower acceptance requires more verification steps and remains an investigation target; no numerical/exact-text failure has been reclassified.

Historical baseline capacity/context/graph settings differ from the cache candidate; see the original comparison. This is a comparison of working configurations, not an isolated source-only experiment. Prefix reads are skipped; native candidate writes remain enabled. Decode includes lower-concurrency finishing tails. No image-support qualification is implied.

Evidence: runs/runtime-ab-fixed-v2-{launch,screen,collection,comparison}.json; exact container8c8d496e83cece6ca54c283142539befc3a8d79945d90db079cb21510014dd4b exited2026-09-23T21:25:20.980911216Z. Collection verified74 source/helper pins and unchanged serving identity. Original arms: runs/runtime-ab-baseline-v3-screen.json and runs/runtime-ab-candidate-v2-screen.json.


Raw `runs/` evidence references in this record refer to the development archive;
that archive and site orchestration are not bundled with this public source tree.
