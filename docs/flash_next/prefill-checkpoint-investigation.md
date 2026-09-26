# Native prefill checkpoint investigation

The default-off `VLLM_FLASH_PREFILL_CHECKPOINT` experiment exports the native
reusable GDN/PLE state inside a larger prefill forward. It leaves native cache
hashes, block ownership, refcounts, offload completion and idle TTL unchanged.
Its purpose is to avoid an extra whole-backbone pass at the944-token boundary.
The timing below is experimental; no numerical or production qualification is claimed.

## Completed component evidence

The first component gate passed6 tests on each of4 GPUs: independent native
GDN prefix-state equality, history copies in both physical layouts, native
strides, null-state masking, source/neighbor preservation and speculative slack.
The expanded gate passed10 tests/rank, adding actual PLE owner/kernel execution
in prefill-only and mixed MTP/prefill batches. Exporting PLE history leaves the
complete forward output and other cache state exactly unchanged.

Evidence: `runs/prefill-checkpoint-component-v1-collection.json` and
`runs/prefill-checkpoint-ple-v1-collection.json`, with frozen-source hashes and
all rank reports. The PLE gate does not exercise the full model metadata path.

## Real-weight lifecycle findings

The checkpoint-enabled reduced v2 run exported66 checkpoints/rank and passed32
exact real-input QSA staging comparisons/rank. Native RAM prefix bytes restored
exactly; ownership and cancellation fences passed. Fresh2048-token requests ran
in one forward. Hot reuse ran944→1888→2048; cold reuse ran944→2048. Exact output
comparisons failed, including expiry/shared-cancellation comparisons. The raw
screen remains false.

The checkpoint-disabled control used the same2048-token prompt and2048 budget.
Fresh requests split0→944→2048. Both one-target-plus-MTP and two-target-plus-MTP
suites, on all ranks, showed:

- Repeated fresh execution and cold reuse matched exactly.
- Hot reuse differed from fresh execution with a different chunk schedule.
- Independent fresh0→944→1888→2048 executions matched the hot suffix and final
  outputs exactly, proving a pre-existing dependence on chunk shape.
- Prefix bytes matched exactly. Expiry and the original combined screen failed.

This control explains a source of numerical differences; it does not turn any
failed screen into a pass or establish that checkpointing adds no differences.

The enabled v3 run repeated those split controls. Its independent fresh split
repeats matched each other, but checkpoint-derived hot reuse did not match the
split control, on any rank in either target suite. The candidate therefore has
an additional unresolved prefix-computation difference and remains unqualified.
Evidence: `runs/reduced-prefill-checkpoint-v3-*`.

The subsequent real-weight trace locates differences before checkpoint arithmetic:
first944 GDN projection inputs differ by up to0.0625 BF16 between the2048-row
and944-row forward, on all ranks in both suites. Initial recurrent state is
identical; projected q/k/v/g/beta and exported state then differ. This implicates
the upstream input/hyperconnection computation, not a demonstrated cache-copy
error. The exact originating operator is not yet isolated. Raw numerical screens
remain failed. See `runs/reduced-prefill-checkpoint-trace-v1-attribution.json`.

Evidence: `runs/reduced-prefill-checkpoint-v2-*` and
`runs/reduced-prefill-checkpoint-control-v1-*`. The attribution JSON files hash
their input rank reports and retain both raw verdicts and matched-schedule
comparisons. Neither fixture is model-quality evaluation or full C4 execution.

## Current limits

The full-model2k timing completed16 batches with81 frozen sources and serving
identity verified (`runs/runtime-ab-prefill-checkpoint-v1-*`). Each request used
one2048-token forward. With metadata reuse enabled, medians were:

| Concurrency | Earlier paired test, checkpoint OFF | Checkpoint ON | Historical baseline |
|---|---:|---:|---:|
| C2 | 2625.0 | 3335.2 | 3842.7 |
| C4 | 3834.4 | 3839.0 | 3976.6 |

Units are prompt tokens/s; three measured repeats follow warmup. These are
cross-run observations with CPU frequency variation, not isolated causal gains.
Checkpoint-ON repeat ranges were3335–3600 at C2 and3607–3841 at C4. All ranks
reported checkpoint exports; no OOM occurred. Only16/36 measured first tokens
matched the earlier paired run; this one-token check does not qualify generation
quality, and the reduced exact-output failures remain unchanged.

Read-only clock samples show the slow batches coincide with one or more worker
cores reporting800MHz, while the fastest C4 batches keep all four near1.9GHz.
This offers a separate way to address launch lag without changing prefill
arithmetic. Host clock settings were not modified.

Checkpointing remains disabled by default. Existing TG staging gains do not
depend on it. Broader prefill performance, full-model cache reuse and numerical
qualification remain separate requirements. The full benchmark launcher now
supports the experiment and requires positive activation receipts on all ranks;
only the completed2k timing above has full-model evidence.

Raw `runs/` evidence references in this record refer to the development archive;
that archive and site orchestration are not bundled with this public source tree.
