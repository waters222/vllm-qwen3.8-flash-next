# Flash-Next: preserved RTX 3090 engine modifications

This branch preserves the engine source used by a qualified four-GPU
Qwen3.8-Flash-Next evaluation on 2026-09-20. It is **not** an upstream vLLM
release, a generic deployment preset, or a newly rebuilt/validated image.

Post-snapshot change: [MTP padding correctness fix](mtp-padding-fix.md).
Also under validation: [opt-in QSA page cap](qsa-page-cap.md).
The experimental MTP branch also preserves [draft-only INT8 conversion and
MTP-aware scheduling diagnostics](mtp-int8-experts.md). These are not a serving
promotion; the original snapshot branch/tag remain recovery points.
Further experimental work: [MTP1 phase-pair admission](mtp-pairs.md), whose
broader concurrency screen remains mixed, and [lower-peak TP4 repacking](tp4-chunked-repack.md),
which passed a real-layer bitwise/memory probe but is not yet full-model qualified.
The subsequent full Marlin load still failed VRAM allocation. A
[native unpadded K32 specialization](marlin-k32.md) now has passing real-weight
small-row and prefill/graph screens, full-model TP4 no-MTP measurements, and
a completed TP4 Q8 MTP3 short-prompt concurrency screen. MTP concurrent gains
remain workload-dependent; profiling and depth selection are still in progress.
It is default-off and is not part of the historical serving image.
Revised source files are tracked separately from the original image hashes;
the byte-for-byte preservation statements below describe the snapshot.

## Provenance

- Upstream base: `dc36fcce902a63eab06c1b93a5c4a5ee178a0c56`.
- Installed version: `0.29.1rc1.dev47+gdc36fcce9`.
- Reference model: `cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4`.
- Publisher prerequisites and licenses: [publisher](publisher/README.md).
- Exact source hashes: [source manifest](source_manifest.json).
- Selected engine flags and arguments: [serving profile](serving-profile.json).

The source audit compared all 2,700 installed Python files with the pinned
checkout. It identified **40 modified or added engine files**, with 2,458
unchanged upstream files and 202 generated/bundled dependency files excluded.
The 40 engine files match the installed reference byte for byte. Generated
`_version.py`, wheel-bundled dependencies, model weights, compiled binaries,
credentials, private hostnames, infrastructure paths and raw traces are not
part of this source migration.

The fork is based on the deployed commit, not current upstream `main`.
Keep upstream updates separate until compatibility and numerical/performance
validation have passed. The imported engine and CUDA builder files retain
their original bytes rather than receiving automatic formatting changes.
This matters because source and binary pins are enforced by the development
and runtime checks. Historical formatting/lint debt is not an upstream-ready
contribution; human review and model evaluations are required before a PR.

## Preserved changes

| Area | Preserved implementation |
| --- | --- |
| Publisher model support | Quantization compatibility, host PLE/Engram, QSA host KV, model-state hooks; dormant MTP support code |
| TP2 RAM KV | Rank-local QSA shape/index handling for TP2; host-backed main KV with GPU side state |
| PP scheduling and placement | Guarded phase-balanced admission, phase observer, and stage-local vocabulary ownership |
| Hyperconnections | SM86 combine/norm, up/gate, down/SiLU, injection, paired down/injection, and M3 dispatch |
| GDN | Input projection specializations, metadata reuse, multirow and M3 paths, B/A fusion, tensor-core multirow adapter |
| Attention output | SM86 M1 and multirow projection paths |
| Shared expert | BF16 shared-expert epilogue specialization |
| Routed experts | Output sum, token alignment/layout and guarded private Marlin scheduling extension |

All existing opt-in guards and native fallbacks are retained. Some installed
code is diagnostic or inactive in the reference profile. The evaluated
configuration is **TP2/PP2, no EP, no MTP, text only**, with a 25/23 layer
partition, host-RAM QSA KV and Engram offload. It uses graph sizes 1/2/3/4,
four request slots, a 2,048-token batch budget and 240,000 advertised context.
This does not assert every four-request/full-context combination fits memory.
TP pairs were PCIe-local; hardware topology and available host RAM matter.

## Preserved performance reference

The existing image measured the following normal-generation rates. This
migration did not run another GPU benchmark or change the deployment.

| Measurement | Tokens/s |
| --- | ---: |
| C1 decode, original comparison | 92.222 |
| C1 decode, fresh pre-profile control | 92.194 |
| C1 decode, exact server restored after profiling | 92.219 |
| C2 aggregate | 158.504 |
| C3 aggregate | 193.555 |
| C4 aggregate | 253.516 |
| C1 nominal prefill 2K / 8K / 16K / 32K | 2857 / 4952 / 5632 / 6046 |

Decode used 31 prompt tokens, 512 output tokens, seed 17, temperature zero,
thinking disabled, two warmups and five measured rounds. C1 is inverse median
request TPOT; concurrent figures are end-to-end aggregate batch rates.
Prefill used a separate nominal-input harness including one output token/API
overhead. These are not directly interchangeable metrics.

The decode client is preserved in
`benchmarks/flash_next/benchmark_controlled.py`. Run it only against an
intentionally isolated loopback server:

```bash
.venv/bin/python benchmarks/flash_next/benchmark_controlled.py \
  --label verification --output /tmp/flash-verification.json \
  --concurrency 1 --runs 5 --warmups 2 --output-tokens 512
```

Bounded reference validation passed 14 scheduler/context canaries and matched
all 12 saved behavior answers. The strict expected-answer checker scored 11/12
because of an inherited `No` versus `no` capitalization mismatch. Free
generation can vary even without a code change; these checks do not establish
general quality equivalence. C2/C4 variability remains unresolved. Multirow
kernel gains must not be equated with the same gain at request concurrency C.

## CPU preservation checks

Use a dedicated environment. These checks require no torch, CUDA or model
weights and do not claim GPU numerical or performance qualification:

```bash
uv venv --python 3.12
uv pip install -r requirements/lint.txt
.venv/bin/pre-commit install
.venv/bin/python tools/flash_next/run_cpu_tests.py
```

The suite preserves existing PP admission/vocabulary and Marlin work-partition
tests, checks all 40 installed source hashes and Python syntax, regenerates
both qualified CUDA sources from pinned upstream input, and checks the
allowlisted Docker build context. GPU integration and broader trajectory
tests remain in the deployment/development repository; they were not copied
wholesale because they contain site-specific orchestration and fixtures.

Migration checks on 2026-09-20 passed all 21 CPU tests and Ruff checks for
the new exporter, test runner and CPU tests. The local credential scan found
only reviewed hexadecimal commit/checksum pins. The byte-preserved engine,
builders and benchmark retain their historical formatting; the complete
upstream pre-commit suite is not claimed to pass. Preservation commits bypass
automatic commit hooks to avoid rewriting these pinned source bytes.

## Rebuild path and its limits

Do not rely on an untagged chain of local Docker layers. The source-only build
context uses the original registry-addressable base:

`docker.io/vllm/vllm-openai@sha256:43f13b4c624ab9e9e6753d0eeb5953268bff334f2a826239e6f4a2197d47bb96`

```bash
task_build_dir=$(mktemp -d)
.venv/bin/python tools/flash_next/export_build_context.py --output "$task_build_dir"
docker build --network=none --pull=false \
  --tag flash-next-preserved:20260920 "$task_build_dir"
```

Fetch the pinned base separately if absent. Compilation requires no GPUs,
model mounts or network. The exporter includes only 40 engine files, nine
upstream CUDA/header inputs, two pinned builders and the Dockerfile. It does
not send `.git`, `.venv`, credentials or the complete repository to Docker.

The custom extension sources are also retained at
`csrc/flash_next/marlin_dp4/`. The pinned builder generates identical source
bytes and uses SM86, two compiler jobs and the inherited CUDA/PyTorch toolchain.
The historical build used torch `2.13.0+cu130` and CUDA toolkit `13.0.88`.
Its reference library hash is
`78541fcf5b9f9fa3ce3bc5f7fcb8de7f558c0f58be91def817a864bf3d040912`.

**This new consolidated Docker recipe has not yet been executed or GPU
qualified.** It intentionally fails if the rebuilt private library differs
from the qualified hash. Toolchain/build-path differences can change binary
bytes; do not silently weaken the runtime pin or label a rebuild equivalent.
Record and review any new artifact, then run the original numerical, graph,
trajectory and serving gates before adopting it. The runtime also verifies
the native Marlin binary and operator schema.

The reference image ID in the manifest identifies the tested local image;
it is not a promise that an image with that ID is available from a registry.
A separate durable image/binary archive and off-node backup are still needed.

## Deliberately not promoted

Rejected HC tensor/split-K probes, experimental QSA GPU read-cache work and
the unlaunched direct-read cache variant are not enabled or presented as
production improvements here. Their records remain in the development
repository. Raw Nsight traces and infrastructure runbooks also remain outside
this public fork. This branch preserves the deployed engine, not every
historical experiment or the entire cluster configuration.

No upstream PR, serving restart, weight change or model deployment is part of
this preservation operation. AI assistance was used for the original work
and this migration.
