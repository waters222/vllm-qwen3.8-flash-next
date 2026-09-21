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

Full-model no-MTP control is loading on the development quartet. Model output,
memory feasibility and end-to-end speed remain unproven. MTP follows only
after that control qualifies. Unequal128/192 sharding remains an unintegrated
fallback prototype.
