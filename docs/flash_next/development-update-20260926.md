# Accumulated Flash-Next development update — September 26, 2026

This update preserves the engine, tests and benchmark work accumulated after
`abb876649c06f070c9067e6994fbdf29557212a1` on the existing experimental branch.
It is not a production promotion. Native cache-state integrity, numerical
repeatability and performance are separate gates; failed exact-output checks
remain failures.

## Engine and test changes

* Native hot/cold prefix retention reuses vLLM hashes, refcounts, LRU and CPU
  offload completion. Main target/MTP QSA KV retains its independent host-RAM
  pool. The tested idle TTL is 3,600 seconds. The old streaming-session swap
  prototype remains disabled with zero swap bytes in the measured profile.
  Cache allocation, ownership, cancellation, expiry, load barriers and failure
  diagnostics are recorded in [the cache log](session-swap.md).
* [Guarded TP4 GDN projections](tp4-projections.md) retain native fallbacks for
  unsupported rows/layouts. They do not imply that every TP2 specialization
  has been qualified on TP4.
* Uniform speculative GDN metadata staging and the local-offload multimodal
  policy fix address measured text regressions. Local KV retention no longer
  implicitly opts into embedding-only multimodal input handling. See the
  [initial recovery comparison](runtime-recovery-comparison.md).
* Packed QSA verification staging is opt-in through
  `VLLM_FLASH_QSA_VERIFY_STAGING=1`. Selected index order and host KV ownership
  remain unchanged. Component tests include changing-input graph replay;
  [full-model comparisons](qsa-staging-comparison.md) show workload-dependent
  TG gains while retaining performance and numerical limitations.
* Prefill metadata reuse is opt-in through
  `VLLM_FLASH_QSA_PREFILL_METADATA_CACHE=1`. Reuse stays within a forward context.
  [Measured results](prefill-metadata-comparison.md) are mixed, not a uniform
  prefill speedup.
* Native prefill checkpoint export remains default-off through
  `VLLM_FLASH_PREFILL_CHECKPOINT`. Component checks passed, but real-weight
  numerical checks failed. [Checkpoint attribution](prefill-checkpoint-investigation.md)
  records both pre-existing chunk-shape sensitivity and an additional unresolved
  candidate difference. It must not be treated as a qualified optimization.
* [Cache-block experiments](cache-block-experiment.md) and the
  [bounded QSA prefill sweep](prefill-bounded-tuning.md) did not justify replacing
  the retained native 944-token blocks and 2,048-token prefill budget. The latter
  found small component gains accompanied by failed exact-output checks.

## Current runtime evidence

The [image/MTP matrix](image-mtp-benchmark.md) completed all 66 batches with the
existing engine: C2/C4, two/four images per request, and 1,024 output tokens.
Image support required configuration changes, not another image-specific engine
patch. Image prefill includes processor and encoder costs. Its prompts differ
from earlier text fixtures, so its rates are not a historical regression A/B.

The latest reduced reference-order cache control moved the second fresh run
after hot/cold runs. Both fixtures on all four ranks then matched hot/cold
prefix bytes and GDN entry-state digests, including the previously mismatching
MTP compressed index. Native copy/rejection/replay/staging checks passed.
Raw numerical checks still failed: hot/cold suffix schedules differ, and the
long fixture also exhibits fresh-run variability. This is evidence about one
control, not a runtime fix or proof of general race freedom.

A separately provisioned test API passed model listing, four concurrent text
completions and streaming completion/usage checks. It remains text-only. API
smoke does not qualify numerical/cache correctness, long-context capacity or
image answer quality. Cluster manifests, credentials, model weights, compiled
binaries and deployment operations are outside this source update.

## Source preservation and review

The source manifest now covers 73 engine files. Existing historical pins are
retained; current revisions are recorded separately. Eight additional source
files include five pre-existing modules with baseline hashes and three new
experimental modules. The allowlisted source build context has 86 files total.
The consolidated image recipe has not been newly built or qualified here.

The [measured image source pins](image-mtp-runtime-sources.json) cover 70 Python
sources and the native Marlin library. Of those Python files, 67 match this
worktree byte for byte. Three contain additional opt-in experiments in the
worktree: larger prefill chunking in `flash_marlin_k32_sm86.py` and `model.py`,
and cache-block overrides in `platforms/interface.py`. Their guards are absent
from the measured image. They are preserved for review, not qualified by the
image/MTP benchmark. The full worktree also contains disabled experimental
modules beyond the measured image's allowlist.

Preserve the exact engine bytes. Do not apply automatic formatting across the
experimental engine sources merely to satisfy unrelated historical style debt.
The retained diagnostic launcher has explicit site identity/model guards; it
is not a portable production launcher. Do not run it against another deployment
by weakening those guards.

AI assistance was used. The existing fork PR is the review vehicle; no new
upstream PR or deployment promotion is proposed. Human contribution review is
required before publication. CPU validation and the exact review snapshot are
recorded in the accompanying review evidence; CPU success does not replace the
remaining model and numerical/cache gates.

## Validation of this source update

* `tools/flash_next/run_cpu_tests.py`: 244 tests run, 3 skipped, successful.
  This includes the 73-file source hash contract and 86-file build export.
* Scoped native cache/offload/config suite: 500 passed, 3 skipped. Run with
  `VLLM_TARGET_DEVICE=cpu`, an offline cache of the public OPT-125m configuration,
  and `--confcutdir=tests/v1` to avoid unrelated root integration fixtures.
* Python AST parsing and Ruff `E9,F63,F7,F82`: all 83 changed Python files pass.
  This is not a clean full upstream lint/pre-commit claim.
* `git diff --check`: passes. Historical GPU evidence is documented separately;
  no GPU test or serving restart was performed for this packaging operation.

The native pytest selection is:

```sh
VLLM_TARGET_DEVICE=cpu HF_HUB_OFFLINE=1 PYTHONPATH=. \
  .venv/bin/python -m pytest -q -p no:cacheprovider --confcutdir=tests/v1 \
  tests/v1/core/test_contiguous_kv_packing.py \
  tests/v1/core/test_single_type_kv_cache_manager.py \
  tests/v1/kv_connector/unit/offloading_connector/test_metrics.py \
  tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py \
  tests/v1/kv_connector/unit/offloading_connector/test_worker.py \
  tests/v1/kv_offload/cpu/test_manager.py \
  tests/v1/kv_offload/test_factory.py \
  tests/config/test_multimodal_config.py::test_local_offload_preserves_embedding_input_policy
```

The pinned runtime image supplied dependencies for both CPU runs, with no GPU
devices, no model weights, no network during tests and a read-only source mount.
The OPT configuration revision was `27dcfa74d334bc871f3234de431e71c6eeba5dd6`.
Earlier failed setup attempts are retained in local evidence: missing host
Torch/NumPy, missing root-test `tblib`, absent offline OPT configuration, and an
unspecified platform that rejects hybrid KV configuration. No test assertion was
weakened to get the final passing run. Skips do not count as qualification.
