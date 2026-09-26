# Experimental TP4 GDN projections

`VLLM_FLASH_GDN_TP4_PROJECTION=1` opts into the SM86 BF16 TP4 input and B/A
projections. It defaults to zero. Existing TP2 flags and implementations are
unchanged. This is local development code, not a serving recommendation.

The installer requires the known GDN module names, an unquantized merged
column-parallel layer, TP4, expected global output sizes and contiguous BF16
weights. LoRA, batch invariance, bias, other devices/layouts and unsupported
row counts retain the original method. Repeated installation is idempotent.
The registered custom op has a fake implementation for compilation and
rejects calls outside its guarded runtime scope.

| Projection | Local weight shape | Optimized rows |
| --- | --- | --- |
| GDN input Q/K/V/Z | 4096 × 2560 | 2, 4, 6, 8, 10, 16, 20 |
| GDN B/A | 24 × 2560 | 2, 4, 6, 8, 10, 16 |

Both use BN32/BK128 tensor-core tiles with FP32 accumulation and BF16 output.
All other rows use native projection, including M1/M24/M30/M32. Installed
paired timing repeats showed rank-specific regressions at M24/M30, so those
shapes were removed from dispatch despite favorable standalone medians.
An initial M1
candidate changed whole-layer output and state hashes relative to native on
all four ranks; it was removed. Its failed exactness comparison remains in
`runs/tp4-projection-cache-native-comparison.json` in the handoff project.
Passing BF16 operator tolerances never replaces model-quality evaluation.

With MTP depth four, four and six requests can supply 20 and 30 unpadded
target-verification rows. Concurrency alone does not determine the executed
shape: graph padding, scheduling and verification lengths matter. A graph
padded to 32 rows takes the native path. More rows can improve weight reuse,
but this projection change does not establish long-context KV bandwidth,
full verification throughput, MTP acceptance or C4/C6 capacity.

## Verification

The guarded reduced launcher is
`context/infra/inference/reduced/launch_local.py` in the handoff project.
See that directory's README and the project's STATUS for commands and the
exact active handle. Only GPUs 0–3 are authorized. Do not overlap runs.

`--row-benchmark --tp4-projections` exercises actual layer weights, installer
idempotence, explicit native fallback, disabled/batch-invariant/bias guards,
fullgraph compilation with the eager backend, and CUDA graph replay. Timings
pair shared-pool graphs in alternating order; allocation and wrapper dispatch
are included. Inputs are seeded diagnostic BF16 tensors, not a distribution
of production activations. The operator screen uses rtol/atol 0.02 and records
bitwise exactness separately.

Adding `--projection-cache-check` follows this with native and installed
whole-layer repeatability and the recovered one/two-target-layer plus MTP
cache diagnostic. Independently compare the native/installed validation
hashes as well as the matched hot/cold copy, acceptance/rejection,
cancellation and expiry evidence. Selected layers are loaded; full-model
gates remain necessary later. No serving container or checkpoint is changed.

The other TP2 attention-output and shared-epilogue optimizations still need
their own TP4 shape benchmarks. These two TP4 projection paths do not imply
that all seven historical TP2-specific hooks have been qualified for TP4.
