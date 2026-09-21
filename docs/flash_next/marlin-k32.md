# Unpadded TP4 Marlin investigation

2026-09-20 experimental result; default-off integration, not serving-qualified.

Equal TP4 divides the 640-channel routed experts into160-channel shards.
Existing Marlin preparation pads each to192 channels, adding20% to the
converted expert tensor storage. The original checkpoint stays unchanged.

An isolated private operator instantiated the unchanged native Marlin
arithmetic template for BF16/U4 asymmetric group32, K32/N256,128 threads,
four pipeline stages, M-block8/16/32/48. It uses FP32 reduction, no atomics, and
explicit SM86/shape/precision guards. The installed native library was not
replaced. Gate/up uses native Marlin without padding; expert-down uses the
private specialization.

Rejected alternative: K32/N128 with128 threads gives only one register-pipeline
iteration. The existing loop assumes an even count of at least two, so this
configuration produced incorrect output. It is excluded from the revised
builder and guarded by a CPU generator test. N256 gives two iterations.

The revised one-GPU screen loaded all512 real layer0 experts in each of four
logical TP4 shard layouts. M1/2/3/4/6/8/12/16 all passed an exploratory
relative-L2<.02 bound against padded Marlin; maximum observed difference was
.0003995. Original input tensors were unchanged. These were seeded synthetic
activations/routes, not distributed TP4 or full-model output validation.

Converted expert storage decreased436,469,760 to363,724,800bytes per rank/layer.
Projected across48 such layers this saves3.25195GiB/rank, before runtime
allocation effects. Warmed local complete-expert-block timings were broadly
similar or better: rank0 M1 33.28→32.77us, M2 47.36→44.13us, M4 65.41→61.44us.
Some higher-row cases regressed slightly; no end-to-end speedup is established.

Both the broad private-library screen and real-engine integration screen
passed168 comparisons: four logical shards, fourteen row counts through2048,
and three changed-input states (valid, partly masked, fully masked). Graph
replays used changed activations/routes; candidate/native and graph/eager
comparisons passed, masked outputs were zero, and raw weights were unchanged.
These remain one-GPU seeded-input screens, not distributed qualification.

The default-off `VLLM_FLASH_TP4_MARLIN_K32=1` gate requires exact TP4/PP1/noEP/
noMTP, SM86, BF16/U4 group32 shapes, at most2048 batched tokens, and disabled
padded chunked repacking. It pins the private library and native extension.
The source-preservation export includes its builder and compilation recipe;
the complete updated Dockerfile has not yet been clean-build GPU-qualified.

Full-model no-MTP control reached Ready on the development quartet at about
21.18GiB/device (19.96GiB reported model allocations). All twelve bounded
outputs match the retained fixture. Three512-token repeats across math/code/
prose and C1--4 completed. Math C1 decode88.86tok/s; C2--4 aggregate
138.04/193.45/244.29tok/s. This is not broad model-accuracy qualification or a
promotion. Eight C1 context tests through32K also completed.

The separate `VLLM_FLASH_TP4_MTP_Q8=1` gate permits MTP1--4 with both draft
INT8 flags required. Four distributed ranks passed real512-expert conversion
checks: lower-peak and native weights/scales are bit-exact, selected scales
match an independent whole-checkpoint reduction, and transient peak drops
212.81MiB/rank. Only draft experts are quantized; target checkpoint AWQ is
unchanged. Draft activations remain BF16 (W8A16), not W8A8.

Full TP4 Q8 MTP3 reached Ready at about22.66GiB/device (20.81GiB reported
model allocations). All twelve fixture outputs match and all36 batches pass
token/timing/acceptance audits, with zero recorded preemptions. The fixture
retains its known11/12 strict score due to a case-sensitive capitalization
mismatch. These are short-prompt, seed17, thinking-off,512-output-token
medians; C1 decode and concurrent aggregate are different metrics.

| Prompt | C1 decode | C2 aggregate | C3 aggregate | C4 aggregate |
| --- | ---: | ---: | ---: | ---: |
| Math |164.77|177.80|240.93|297.59|
| Code |172.10|195.94|266.95|323.62|
| Prose |122.19|130.52|189.42|215.32|

Math acceptance remains73--75% across C1--4, code83--85%, prose43--46%.
Prose concurrent throughput regresses against the no-MTP control. The MTP
trial allocates750MB GPU cache/rank versus275MB in the control; both retain
56GiB host-QSA allowance and240K configured context. Four requests were
observed active; this does not prove every graph forms an efficient batch.

The MTP4 depth screen is ongoing. C>=2 profiling at fixed TP4/PP1 follows;
balanced-depth selection and general concurrency scaling remain unresolved.
Unequal128/192 sharding remains an unintegrated fallback prototype. The
complete source-built image remains unqualified; these results use pinned
overlays on the retained evaluation image, not a clean rebuild.
