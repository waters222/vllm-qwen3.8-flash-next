# Hot/cold request state with RAM-backed QSA

Status: experimental, not fully qualified. See the
[September 26 update](development-update-20260926.md) for the current runtime,
performance and remaining numerical/cache failures. Older active-job and
deployment statements below are historical records, not current launch authority.

### Final controller handoff after recovery (2026-09-23 UTC)

The injected native-load failure check and fresh-engine recovery check passed.
The recovery container exited 0 without OOM at 00:51:47 UTC.
Both 128-token requests completed. Cached-token counts were 0 and 5664.
Ten result checks and nine provenance checks passed.
The checks do not prove physical DMA recovery, durable RAM storage or universal text equality.
Production qualification remains incomplete.

The user requested that the original controller stop after this check.
Continue in `/home/water/coding/flash-next-codex` on water-server when the user requests it.
Read AGENTS.md, STATUS.md and REPRODUCE.md in that project first.
The project contains the raw reports, analysis, provenance and remaining work.
No new experiment will start from the original controller thread.

### Full-engine failure/recovery fixture prepared (2026-09-23 UTC)

The next disposable TP4/MTP4 test will inject an exception on rank 3 after a
real native CPU-to-GPU load completes but before connector acknowledgement.
It must observe EngineDeadError, zero emitted tokens, unhealthy-engine rejection,
and rejection of a subsequent request. A separate new engine must recompute
the same prompt with zero cached tokens, then demonstrate hot reuse. This is
not a physical DMA-fault test or recovery of volatile RAM contents.

The diagnostic requires an explicit gate and cannot run together with other
fixtures. CPU tests reject missing/wrong request evidence, incomplete rank
acknowledgements, emitted output, healthy/reusable failed engines, and invalid
fresh/hot controls. Engine helper suite: 188 tests, 184 pass and four skips;
targeted lint passes. Only diagnostic helpers/tests changed. Full-engine
injection/recovery has not run yet; no qualification claim is made.

The existing performance container `3c75c8910439` remains live, most recently
273/380 requests, no reported error/OOM. Wait for authoritative completion
before starting the failure fixture; do not overlap development GPU trials.

### Event-poll failures preserve pending state and report worker failure (2026-09-23 UTC)

Added a CPU-injected event.query exception to the existing native connector
worker tests. The call runs through SingleDirectionOffloadingHandler,
CPUOffloadingWorker, OffloadingConnectorWorker and WorkerProc. Both load/store
and synchronous/asynchronous output branches publish FAILURE, not a successful
RPC result; no connector completion acknowledgement is published. The pending
transfer/event and load-request entry remain, and transfer pools are not reused.
The asynchronous queue is drained deterministically: this is not a thread-race
or physical CUDA-fault test.

Selected suite: 15 pass, 229 deselected, no skips, 0.87s.
Container `0a49a76c2ad8343f05bcd230cf5f84884801c3c58de7e26b6973806119029037`
ran 00:04:54--00:05:26 UTC, exit 0/no OOM; network none, no GPUs/model,
4 GiB/equal swap and two CPUs. All 65 engine plus 11 test/helper hashes match;
the image's native multiproc_executor.py hash also matches the local source
(`3ed0149a8a8e1550c4f062af40e645d1b734e70f99981bc5bf15f0360507648a`).
Evidence: `native-event-failure-cpu-v2.json/.log/.xml`; XML local=remote SHA256
`78b43c2c28b5b24a3ede4486a78fbd9a75c5ae89e72abb900e454f2f9545b392`.
The earlier synchronous-only run is retained (13 pass). Source preservation
(2 tests), targeted lint and diff checks pass. No engine changes.

Source inspection shows the executor raises on FAILURE, fatal EngineCore errors
send ENGINE_CORE_DEAD, and the client rejects later operations with EngineDeadError.
That downstream chain is source-backed, not a full-model failure/recovery test.
Recovery must use a newly initialized engine and re-prefill the conversation;
do not acknowledge failed transfers, resume partial restores, or treat volatile
RAM cache as durable recovery state. No production restart is authorized by
this record. A full-engine failure/restart validation remains outstanding.

The warmed performance trial `3c75c8910439` is now processing requests:
3/380 recorded, running/no OOM and no reported exception. Poll that exact
container; no restart or overlapping development GPU job.

### Deferred shared readers across TTL validated on native CPU path (2026-09-22)

The native scheduler defers duplicate complete-prefix loads while a leader's
source chunks are loading. New 2/4/8-reader cases verify that deferral, preserve
the exact source chunk objects and positive references through four staggered
worker acknowledgements despite repeated >3600-second private-clock advances,
and require a miss if other required unpinned prefix state expired. A retained
partial checkpoint does not resurrect that incomplete prefix. After final
release and another idle interval, all 32 CPU slots are free.

Selected native tests: 11 pass, 229 deselected, 0.81s, zero skips. Container
`94c1df93a430f8d607a71bcedd2d0574acc8f2f4d5c59070d10f77349fe13a81`
ran 23:56:55--23:57:26 UTC, exit 0/no OOM. Network none, no GPUs/model,
4 GiB/equal swap, two CPUs; 65 engine and 11 test/helper hashes verified.
Evidence: `native-shared-readers-cpu-v4.json/.log/.xml`; XML SHA256
(local=remote): `249ece95725c8a7c89aced9a3862f7ea02efa3495b47a450c9560686dd451f86`.
Source preservation tests (2), lint and diff checks pass. Engine unchanged.

Retained setup/fixture failures: v1 rejected a source path containing /models
as if it were a weight mount (no container started); v2 incorrectly expected
simultaneous native duplicate loads; v3 incorrectly expected a complete hit
after a different required checkpoint expired. v4 checks the native protocol
without bypassing these guards. This is scheduler/manager metadata evidence,
not concurrent GPU transfer, C8 full-model, unpaused race or recovery qualification.

### Same-engine staging comparison completed (2026-09-22)

Trial `95f8f5216cd328fe567744cb02d26024a2a8aa4c3db6046fe448731c3c7e5453`
finished 23:54:45 UTC, exit 0/no OOM. All 33 requests and 22 phases completed;
all eight independent measurement checks pass, including alternating order,
matched hot prefixes, valid clocks, four-rank acknowledgements and restoration
of original staging flags. No profiler or iteration timing was enabled.

Four measured pairs per concurrency (seconds to first token):

| C | Off median (range) | On median (range) |
| --- | --- | --- |
| 1 | 0.860 (0.513--1.045) | 0.763 (0.750--0.846) |
| 2 | 3.772 (3.748--3.787) | 1.035 (0.872--1.040) |

C2 improves in all four pairs, 3.62--4.34x lower TTFT. C1 has two pairs in each
direction; the earlier separate-run regression does not reproduce consistently,
and these variable samples do not establish a C1 gain. This is synthetic hot
prefix latency, not decode throughput or cold restoration performance. Native
shutdown force-killed the remaining engine process after request completion
and successful flag restoration; the container exited 0. Do not describe that
as a fully graceful shutdown validation.

Raw screen SHA256 (local=remote):
`e528b09fc20444f2675072799c8c0e8603af52ee1b61e9ba72b0cff017488b0d`.
Evidence: `screen-full-tp4-prefill-staging-ab.json`,
`prefill-staging-ab-analysis.json`, launch manifest and terminal log.

The existing 380-request warmed hot/cold/fresh fixture is next, using three
measured repeats, paired cold loads and all-request staging. Trial
`3c75c8910439d4012e6195e68bab3b67d832cb6d59a885b5d9343cc6e619a94b`
started 23:58:53 UTC, running/no OOM; root
`/tmp/flash-session-gpu-20260921-42ybzd`. Evidence:
`launch-full-tp4-warmed-prefix-all-staged.json`. Serving remains running at its
original start time. No serving changes or publication.

### All-staged 32k integrity audit completed (2026-09-22)

Trial `0d7db247c13a7b6df9394ca5ab12bdadd98e208677db4a110c3dece1dc795f5c`
completed all 200 requests at 23:42:35 UTC, exit 1/no OOM. All eight growing
continuation lifecycle checks pass. Native ownership passes all 48 rank/turn/
branch joins; physical-version continuity passes 19344 exact page comparisons,
zero different, zero unobserved, and zero conflicting producer observations.
This revalidates resident target/MTP RAM pages with all-request staging enabled.

The combined/numerical result remains false and is not waived. Strict equality
to the immediately preceding branch holds in only 8/48 comparisons: native
lookup can choose an earlier duplicate physical cache version. Each actual
restored version matches its observed producer exactly. This is scoped RAM-page
integrity/ownership evidence, not GPU recurrent-state race freedom, deterministic
text, broader capacity, or production qualification.

Artifacts: `screen-full-tp4-long-resident-all-staged.json`,
`long-resident-all-staged-analysis.json`, matching ownership/barrier JSON and
terminal log. Raw screen SHA256:
`49f7b787f5bebdce69e5ed343e8dec01c945edf80b4803527e39472e1a64b8a8`.
All three copied JSON hashes match the completed remote artifacts. The same
engine staging A/B preflight then passed with development GPUs free and the
serving peer still running at its original start time.

The A/B trial is now running/no OOM, started 23:46:05 UTC:
`95f8f5216cd328fe567744cb02d26024a2a8aa4c3db6046fe448731c3c7e5453`,
root `/tmp/flash-session-gpu-20260921-kDLoMr`. Evidence:
`launch-full-tp4-prefill-staging-ab.json`. Its 65 engine source hashes and all
19 optimization flags match the completed integrity audit; only diagnostic
helpers/configuration differ. Production remains unchanged. Re-ran the
independent state/performance analyzer tests: 41 pass (the first invocation
omitted the engine helper PYTHONPATH and was corrected without a code change).
No serving settings, model files, or source pins changed.

### Same-engine staging A/B fixture prepared (2026-09-22)

The next performance diagnostic uses 33 first-token requests in one loaded
TP4/PP1 Q8-MTP4 engine: C1 and C2 each get prime/warmup, then four alternating
off/on pairs with identical prompts and native hot-prefix hits. Native
pause(wait, clear_cache=False), GPU synchronization and the flag RPC occur
outside request timing. Four matching worker acknowledgements are required;
the RPC rejects resident requests or a model other than twelve target plus
one draft RAM-QSA modules. Original module flags are restored in finally.
A failed acknowledgement cannot resume generation.

This is explicit `--prefill-staging-ab`, gated by
`FLASH_PREFILL_STAGING_AB=1`; the launcher resets inherited choices. No
profiler, iteration tracing, kernel override, or lifecycle fixture may be
combined. The helper measures first-token latency only and leaves integrity
and production-qualified fields false. CPU suite: 185 tests, 181 pass and
four skips; standalone screen suite: 88 tests, 86 pass and two skips.
Independent performance analyzer: five tests pass, including altered clocks,
cache-hit evidence, worker acknowledgements and restoration negative controls.
Lint and diff checks pass. These are diagnostic tests, not a GPU A/B result.

The existing 32k all-staged resident-page audit remains the only development
GPU job. Latest inspection: running/no OOM, 174/200 requests recorded. Do not
launch A/B until that exact container is terminal and its evidence is retained.
Serving is unchanged; no commit, push, or promotion occurred.

### C1 trace separates tiny-copy waits from transfer cost (2026-09-22)

Read-only reanalysis of the completed baseline/candidate traces found the same
435 `cudaMemcpyAsync` API calls across the two C1 target-prefill steps per rank.
Exactly 24 calls/rank are 8-byte device-to-pageable-host transfers: twelve per
step. Their actual GPU transfer duration totals only about 25--27 microseconds
per rank in both runs.

Host API time for those same 24 calls differs substantially:

| TP rank | Baseline API wall time | Candidate API wall time |
| --- | --- | --- |
| 0 | 23.70 ms | 473.61 ms |
| 1 | 27.48 ms | 1.18 ms |
| 2 | 24.66 ms | 14.72 ms |
| 3 | 21.26 ms | 690.36 ms |

For example, the candidate rank-3 longest call spends 78.407 ms in the host
API while its correlated GPU memcpy takes 1.6 microseconds. This distinguishes
waiting/synchronization from moving eight bytes; it does not establish why the
candidate run waits longer or prove that removing the API removes all that time.
CUDA work queued before the readback can account for waiting, and rank times
overlap. Do not add API, collective and nested operator durations as independent
costs.

The count/size pattern is consistent with the QSA staging helper reading
`query_start_loc.tolist()` (two INT32 offsets at C1), followed by the four-byte
sequence-length read, once per target QSA layer. Without stack traces this is
a source-backed inference, not an exact call-site attribution. Native common
attention metadata already carries CPU query offsets and CPU sequence-length
upper bounds; reusing those could avoid readback synchronization, but upper
bounds are not automatically exact MTP lengths. No metadata or engine change
was made on this evidence. First repeat unprofiled measurements; any later
metadata reuse needs separate correctness checks.

The existing trace summarizer now correlates memcpy APIs and GPU transfers,
reports unjoined/ambiguous IDs explicitly, and separates their clocks. Four
unit tests pass, including a long API/tiny-transfer negative-control fixture.
Evidence: `hot-prefill-baseline-copy-waits-v2.json` and
`hot-prefill-candidate-copy-waits-v2.json`; original trace artifacts and earlier
analyses remain unchanged. The 32k resident-page audit continues on development
GPUs 0--3; serving remains untouched.

### Multi-request staging removes the traced C2 direct-RAM bottleneck (2026-09-22)

Candidate `a46164434cf36f2da5900dfdaa52b9839ab63259d049183626d248e406a17c4a`
completed nine requests, 23:07:57--23:16:33 UTC, exit 0/no OOM. All four native
traces contain 14515 kernels and the expected four nonempty target-prefill
geometries. The C2 target steps now have 24 staged calls each and zero direct
QSA calls; C1 retains 12 staged calls per step. C2 target staged kernels total
52.8--55.2 ms/rank, replacing the baseline's approximately 2848 ms/rank of
direct-RAM work plus about 27 ms of staged work. These totals are kernel time,
not an end-to-end speedup or a sum across TP ranks.

Matched prompts, engine arguments (apart from the trace output directory),
TP4/Q8-MTP4, cache hits, clocks and native token accounting pass independent
comparison checks. One warm and one profiled sample per concurrency:

| C | Phase | Baseline TTFT | Candidate TTFT |
| --- | --- | --- | --- |
| 1 | Unprofiled warm | 0.524 s | 0.760 s |
| 2 | Unprofiled warm | 3.750 s | 0.840 s |
| 1 | Profiled capture | 0.632 s | 1.328 s |
| 2 | Profiled capture | 3.773 s | 1.346 s |

C2 warm TTFT is 4.47x lower in this bounded comparison. C1 is slower, so this
is not an overall serving-gain claim. Its staged QSA time remains similar;
the candidate trace shows strongly uneven increased collective residence
across ranks (whole-trace NCCL BF16 reduction totals about 419--1702 ms/rank).
Collective residence includes waiting and does not identify CPU scheduling,
communication, or profiler overhead as the cause. Repeated unprofiled tests
are still needed. The profile contains no decode-throughput measurement.

All local trace hashes match the worker manifests and the final raw screen
matches remote SHA256:
`b8831c6cb69c61db20b2cb7aac1e668e28f4f7d3148a8ae85bff30a65c14edb2`.
1538 kernels/rank remain outside target execution attribution; they are not
silently assigned. The baseline had the same unassigned count. The short
screen's integration/production-qualified fields remain false by design.

Evidence: `hot-prefill-staging-comparison.json`,
`hot-prefill-all-staged-profile-analysis.json`,
`hot-prefill-all-staged-validation.json`,
`screen-full-tp4-hot-prefill-all-staged.json`, traces, launch/barrier and logs.
The enhanced performance analyzer passes four tests including negative controls
for topology, cache tier, incomplete output, clocks and scheduled-token counts.
The unchanged trace analyzer passes three tests; lint/diff checks pass.

The existing 200-request 32k C2 resident-page/ownership audit started at
23:20:42 UTC with `--stage-all-prefill`:
`0d7db247c13a7b6df9394ca5ab12bdadd98e208677db4a110c3dece1dc795f5c`,
root `/tmp/flash-session-gpu-20260921-BmugH8`. Authoritative inspection confirms
running/no OOM. All engine/helper hashes and optimization flags match the
completed candidate profile. Evidence:
`launch-full-tp4-long-resident-all-staged.json`. This is the next correctness
gate, not a serving promotion. All serving settings remain unchanged.

### Opt-in multi-request prefill staging passes reduced GPU checks (2026-09-22)

The candidate adds default-off `VLLM_QSA_STAGE_ALL_PREFILL=1`. It stages every
eligible request (at least 64 query rows) sequentially through the existing
per-rank arena and keeps direct fallback for short/decode rows. It reuses the
existing staged QSA kernel and each request's own block table. Cache ownership,
TTL, retention, weights and kernel implementations are unchanged.

Reduced GPU test `af36a9962be2def893439430ee8ec47aad4f6e252acfc543956a5b9333c94b86`
ran 23:03:16--23:04:21 UTC, exit 0/no OOM: **24 tests pass**, no skips,
60.620s. It used development GPU 0, no model weights/network, 8 GiB/equal swap,
four CPUs and read-only engine mounts. Twelve new cases exercise the real
dispatch plus CUDA kernels with one to four requests, shared first pages,
private suffixes, mixed short/long query rows and one/eight-page arenas.
Baseline and candidate match the independent attention reference at the
existing BF16 tolerance; raw RAM-cache bytes remain exactly unchanged.
The other twelve cases retain existing poisoned-page/causal-tail coverage.

These are synthetic operator inputs at TP4-local attention-head dimensions,
not checkpoint-derived activations, full-model cache restoration or a
throughput benchmark. Full-model validation is still required.

Engine QSA source pin changes from
`f5d637775c48e3c6489d750cff7d2ecbc373d1ae4810da7ea1b93a005450e0e3` to
`0f41defc21377213c1f58657c6624fe17e61126ef30670209763ee4cd5e6650c`.
The revision is recorded in `post_snapshot_sources`; historical image pins
remain unchanged. The initial preservation check caught updating the historical
entry instead of the revision; that bookkeeping mistake was corrected, and
both preservation tests now pass. The complete local CPU runner reports
183 tests: 179 pass/four dependency skips. Ruff/diff checks pass.

The isolated full-model launcher now has explicit `--stage-all-prefill`;
controls force the flag off rather than inheriting it, and both launch and
screen evidence record the choice. The same nine-request TP4/Q8-MTP4 profile
started with the candidate enabled at 23:07:57 UTC:
`a46164434cf36f2da5900dfdaa52b9839ab63259d049183626d248e406a17c4a`,
root `/tmp/flash-session-gpu-20260921-pSqhvu`. Authoritative inspection confirms
running/no OOM. Launch comparison finds exactly one changed engine source,
`nvidia/qsa.py`; all 19 optimization flags match the baseline. Evidence:
`launch-full-tp4-hot-prefill-all-staged.json`. Serving GPUs 4--7 retain their
original start time. No promotion/commit/push occurred.

Evidence: `qsa-multi-stage-gpu-launch.json`, `qsa-multi-stage-gpu.log`,
`qsa-multi-stage-gpu.xml`. XML SHA256:
`c3f94add8ca5d8a001aeb2fb23a89ab6e7045f5d08eda36b3dc43a5addb5688e`.

### Profile identifies single-request staging fallback as the C2 bottleneck (2026-09-22)

Diagnostic `ca52ec0049d1c0de4178efd39125cb2333eb79250df2d6aea35e60c02c8e899f`
completed nine requests and exited 0/no OOM at 22:56:58 UTC. Four native traces
contain 14450 kernels each. Local trace hashes match the worker manifests;
the final screen SHA256 matches the remote file:
`2a491af956d6f8b1c26bd80bc6dacf67f703f744471c458c06cb675a639526b1`.
The screen intentionally keeps production/integration qualification false:
this is a short profiler diagnostic, not a full integrity or throughput gate.

CUDA launch correlations identify the four expected nonempty target executions
on every rank: C1 944/492 tokens, C2 1888/984 tokens. Four additional zero-token
connector iterations have no kernels and remain in the report. Version 1 of
the analysis did not distinguish those idle iterations; version 2 does and
retains both raw results. Three analyzer tests cover asynchronous attribution,
ambiguous launches, idle-versus-unexpected geometry and overlapping intervals.

Per-rank target-step QSA times (range across four ranks):

| Step | Staged QSA kernels | Direct-RAM QSA kernels |
| --- | --- | --- |
| C1, 944 tokens | 18.14--19.02 ms, 12 calls | none |
| C1, 492 tokens | 8.65--8.88 ms, 12 calls | none |
| C2, 1888 tokens | 17.59--18.37 ms, 12 calls | 1871.42--1872.55 ms, 12 calls |
| C2, 984 tokens | 8.62--8.74 ms, 12 calls | 975.38--975.92 ms, 12 calls |

The native C2 worker-result waits total approximately 3.747 seconds; the
direct-RAM target kernels alone total approximately 2.848 seconds per rank.
Do not sum these across TP ranks or equate kernel occupancy with compute
utilization. Another 1538 kernels/rank (311--315 ms) are outside the attributed
target scopes and are explicitly unassigned, not silently charged to prefill.

The source explains the path difference:
`vllm/models/qwen4_exp/nvidia/qsa.py::_qsa_staged_prefill` picks only the
largest request; `forward_qsa` sends remaining rows through direct UVA reads.
C1 therefore stages all target prefill rows; C2 stages one request and leaves
the other on the expensive direct path. This is an attention execution-path
bottleneck, not evidence of native cold-cache load latency or cache corruption.

Next: test staging every eligible prefill request sequentially through the
same per-rank arena, retaining direct fallback for short/decode rows. Reuse
existing staged-attention kernels and one-layer fixtures; validate divergent
block tables, shared prefixes/private tails, mixed prefill/decode and arena
reuse before a full-model performance and state-integrity rerun. This must
not change cache ownership, TTL or retention. No engine change or speedup
claim has been made yet.

Evidence: `screen-full-tp4-hot-prefill-profile.json`,
`hot-prefill-profile-analysis-v2.json`, `hot-prefill-profile-validation.json`,
`hot-prefill-profile/`, and the launch/barrier records. Serving GPUs 4--7
remain at the original start time; the completed diagnostic is retained.
No promotion, commit or push.

### Short native C1/C2 hot-prefill profile started (2026-09-22)

Diagnostic `ca52ec0049d1c0de4178efd39125cb2333eb79250df2d6aea35e60c02c8e899f`
started at 22:47:42 UTC in `/tmp/flash-session-gpu-20260921-8wFMGb`.
It uses development GPUs 0--3 only. Serving container `7d408c2bc15d`
remains running at its original 2026-09-20T19:45:53 start.

The bounded fixture primes and warms C1/C2 prefixes (six requests), then uses
native vLLM start/stop profiling for three hot first-token requests. It requires
7100 prompt tokens, 5664 cached tokens, zero cold-load bytes and four rank trace
artifacts. Stack, shape and memory retention are disabled; collection is capped
at 32 engine iterations. It skips the 380-request benchmark. Trace existence
alone is not GPU-event coverage or an integrity pass.

Only diagnostic helpers changed; the 65 engine source pins and optimization
flags remain unchanged. Helper tests report 135 pass/four skips; two source
preservation tests pass. The existing raw-trace summarizer now attributes kernels
through CUDA launch correlations rather than GPU timestamp overlap; three local
tests cover asynchronous launches, ambiguous IDs and overlapping durations.
Raw CUDA coverage and phase attribution still need inspection after completion.
Evidence: `launch-full-tp4-hot-prefill-profile.json`. No promotion/commit/push.

### Native timing locates the C2 anomaly in batched prefill (2026-09-22)

Diagnostic `068fe15cf08cc647f5381f3e93b8f5cb57be2c63414ef72ec3164f6af7f5f719`
completed all **380 requests**, 22:05:19--22:34:51 UTC, no OOM. All seven
independent measurement checks pass, and all **32 phase captures** account for
the exact expected uncached prompt tokens with valid contiguous iteration
records. Exit 1 preserves the separate numerical-output screen failure;
there is no engine-error or blanket quality-pass claim.

Three measured repetitions after one discarded warmup, 7100 input/128 output,
5664 tokens reused for hot/cold, TP4/PP1/Q8-MTP4:

| C | Tier | Median TTFT (s) | Aggregate post-first tok/s | Draft acceptance |
| --- | --- | --- | --- | --- |
| 1 | Fresh | 3.155 | 102.23 | 72.70% |
| 1 | Hot | 0.538 | 111.59 | 71.50% |
| 1 | Cold | 0.772 | 103.55 | 72.98% |
| 2 | Fresh | 16.390 | 148.21 | 72.73% |
| 2 | Hot | 3.762 | 145.51 | 72.85% |
| 2 | Cold | 1.963 | 107.62 | 73.72% |

C2 hot batches schedule 1888 then 984 context tokens. Their summed native
prefill wait windows are 3.722/3.732/3.764 seconds, close to the measured TTFT.
C1 hot schedules 944 then 492 tokens, totaling 0.516--0.521 seconds in those
windows. C2 fresh spends 16.331--16.361 seconds in prefill windows across seven
1888-token batches and one 984-token batch.

C2 cold takes a different path: 944 context tokens for one request, 1436 for
two requests, then 492 context tokens mixed with five scheduled generation
tokens. Its total uncached prompt work is still 2872 tokens, but load readiness
staggers the batches and first-token times. This links the lower cold TTFT to
different admission/execution geometry, not a faster cold cache transfer.
It does not identify the responsible kernel or separate compute, communication,
offload stores, CPU/UVA memory accesses and other worker-side waiting.

Native elapsed time measures host worker-result waiting, not full GPU service
time; matching context totals does not prove complete trailing-decode capture
or rule out late frontend statistics. This instrumented run is not a replacement
for the prior uninstrumented benchmark. In particular, cold decode is lower
than the earlier run; different output histories and logging overhead prevent
calling that a cache-engine regression.

Evidence: `screen-full-tp4-prefix-native-iterations.json`,
`prefix-native-iterations-analysis.json`,
`prefix-native-iterations-prefill-diagnosis.json`, and matching launch,
barrier and tail-log records. Raw SHA256, identical locally/remotely:
`2ae28797daeed260239318c0f02ad0d59f72dd4e840ff69d0dcac43e8f667537`.
The completed diagnostic is retained. No GPU trial remains running on the
development quartet at this point; serving GPUs 4--7 remain unchanged.
Next performance work should isolate the paired-prefill cost, not change
cache restore policy based only on these timings. No promotion/commit/push.

### Source export and review entry point checked (2026-09-22)

The fork's `docs/flash_next/README.md` now links the native-prefix qualification
record, distinguishes it from the disabled streaming prototype and historical
serving snapshot, and describes current export/test prerequisites. The exporter
verifies 65 engine files and emits 78 allowlisted build inputs. Both lightweight
commands pass (two source-preservation tests and one build-context test),
including with `python -S` disabling third-party site packages. The broader
`tools/flash_next/run_cpu_tests.py` reports 181 tests: 177 pass, four skips.
Its docstring no longer incorrectly promises not to import engine dependencies.
Ruff/diff checks pass. No engine or frozen runtime source changed.

These checks establish source preservation and export boundaries, not a rebuilt
image or GPU qualification. The pending implementation remains local and requires
human review before same-branch/PR publication. No commit, push or deployment was
performed. Performance diagnostic `068fe15cf08c` remains running/no OOM,
last observed at 125/380 requests with no reported error.

### Bounded 240k-sized retention planning validated on CPU (2026-09-22)

The completed full-model page audit shows two resident host groups per rank:
12 target layers and one draft layer, each with 944-token, 966656-byte pages.
Native shared-pool packing uses the largest group's stride
(`12 * 966656 = 11599872` bytes), including for blocks owned by the smaller
draft group. Four TP workers allocate independent backings. The planner rounds
each backing to a conservative power-of-two pinned reservation; its host
ceiling applies to the sum across workers, not separately to each worker.

| Host blocks including null | Planned backing across TP4 | Conservative reservation |
| --- | --- | --- |
| 1024 (completed long audit) | 44.25 GiB | 64 GiB |
| 2048 | 88.5 GiB | 128 GiB |
| 2560 (CPU-qualified candidate) | 110.625 GiB | 128 GiB |
| 4096 | 177 GiB | 256 GiB |

The native planner regression uses this page geometry, four worker plans,
240000 max context and a 128 GiB total host ceiling. It accepts 2560 blocks,
checks the packed stride/backing and 32 GiB reservation per worker, and rejects
4096 under the same ceiling. It allocates no large tensors. The existing native
LRU pressure fixture also passes at 2560 blocks: four independent prefixes of
254 complete pages per target/draft group (239776 aligned tokens), private
tails, and 32 intervening short requests retain all four prefixes.
Earlier 2048-block pressure failure remains unchanged. This does not establish
a minimum size, active C4 GPU capacity or 240k full-model generation correctness.

Four native CPU suites pass **227 tests**, one CUDA-only DMA test skipped,
34.15s test time. Container
`be53918f8c892e47704c8a175e9c55688b0dded6a85b975d2a4b844bd31f6570`
ran 22:15:26--22:16:05 UTC, exit 0/no OOM, network none, no GPUs/model,
4 GiB/equal swap and two CPUs. All 65 engine and nine test hashes were verified.
Evidence: `native-capacity-fit-cpu.json/.log/.xml` and
`resident-host-budget-analysis.json`. XML SHA256:
`d0e9149ecba851eebf7c261bc7f4bd46c20ab43fe11d433d05e08f1eda42e170`.

The 8 GiB native CPU-offload tier, model weights and other process memory are
additional to these resident-QSA numbers. Reservations are not measured RSS;
a larger ceiling alone does not grow the configured block pool. No live
budgets, engine code or serving settings changed. The ongoing performance run
keeps its original frozen 512-block fixture; the 2560-block candidate is not
deployed or GPU-qualified.

### Native-iteration performance diagnostic running (2026-09-22)

After the resident-page integrity gap closed, started
`068fe15cf08cc647f5381f3e93b8f5cb57be2c63414ef72ec3164f6af7f5f719`
at 22:05:19 UTC, root `/tmp/flash-session-gpu-20260921-XsmXTb`.
Authoritative inspection confirms running/no OOM. This is the existing warmed
C1/C2 fresh/hot/cold fixture: three measured repeats after one discarded warmup,
7100 input/128 output tokens, TP4/PP1/Q8-MTP4 full graphs, 2048/944 balanced
prefill, private native cold buffers, and native iteration-statistics capture.
Expected total: 380 requests. No page hashing or worker timing hooks.

All 65 engine hashes and 19 optimization flags match the completed v3 audit.
The launcher verified the read-only model, isolated network, free development
quartet and unchanged serving peer before execution. Only GPUs 0--3 are used;
the original serving container remains running at its September 20 start.
Evidence: `launch-full-tp4-prefix-native-iterations.json`.

The purpose is to separate scheduled prefill/mixed/decode work and admission
geometry behind the C2 TTFT anomaly. Frontend statistics can arrive late and
CPU iteration time is not GPU kernel time; validate context accounting before
attributing latency. This is instrumented diagnostic evidence, not a new
uninstrumented performance baseline. No result yet, no promotion/commit/push.

Source review of `EngineCore.capture_iteration_details`, `step` and
`step_with_batch_queue` confirms that native elapsed time surrounds worker
result waiting (and inline sampling where applicable). Scheduling and initial
dispatch are outside the window; queued work may already be executing.
It is neither CPU compute time nor complete per-iteration GPU service time.
The offline analyzer now reports summed/median observed wall time per scheduled
context/generation geometry and explicitly documents this limitation.
Its existing three tests pass, including repeated mixed-shape aggregation and
invalid/missing context-accounting cases. Only offline analysis/tests changed;
the running source snapshot remains unchanged.

### Full-model resident RAM version integrity passes (2026-09-22)

Expanded audit `7cd2ddfdf10395f89cf57c3aae0b0959d495df1bdf841c459e5ff80a4bb76d8a`
completed all **200 requests**, 21:31:03--22:01:22 UTC, without OOM.
All 32 scheduler ownership checkpoints and all four worker-rank captures are
present. The TP4/PP1/Q8-MTP4 full-graph run reused hot and pressure-evicted cold
prefixes at 28320, 29264 and 30208 tokens, with inputs growing to 32588.

Native ownership passes all 48 rank/turn/branch comparisons. Comparing each
consumer page to the observed earlier physical producer version selected by
native lookup gives **19344 exact page comparisons, zero different bytes,
zero unobserved versions and zero conflicting producer observations**.
These are comparisons, not unique physical pages. The earlier 408-observation
coverage gap is closed. Native prefix keys, physical IDs, group/layer and
logical page indices are matched; consumer/future snapshots cannot serve as
their own producer.

The stricter immediately-previous-own-branch comparison remains 8/48 exact:
native lookup can select a different earlier cached physical version for the
same prefix key. Its failure and the original numerical-consistency failure
are preserved. The combined numerical screen exits 1, not an engine crash.
Do not describe this as identical fresh/hot/cold generated text or a blanket
feature pass. The byte/ownership result is scoped to complete resident
target/MTP RAM pages in this run; earlier native-copy evidence covers the
offloaded GPU state separately. Broader race/capacity and performance
qualification remain outstanding.

Evidence: `screen-full-tp4-long-resident-pages-v3.json`,
`long-resident-pages-v3-analysis.json`,
`full-tp4-long-resident-pages-v3-ownership.json`,
`full-tp4-long-resident-pages-v3-barrier.json`, and the matching tail log.
Raw report SHA256 (local equals remote):
`91b13eeee8568879de17af2e9d09b2bb3d3a13b2d89e77d618a09a8a54fde591`.
Engine code and serving GPUs 4--7 are unchanged. No promotion, commit or push.

### Native admission across the idle deadline validated (2026-09-22)

The real native KVCacheManager admission path was exercised with one GPU-like
group and two direct-host groups (target/draft-like), using CPU metadata only.
A consumer finds a shared 32-token prefix immediately before the 3600-second
deadline, then allocates after the deadline. Native two-phase admission pins
all groups' offered hits before allocating the private suffix. The test verifies
unchanged hashes and block identities, positive references, distinct writable
suffix blocks, no expiry while owned, and a miss after release plus another
idle interval. No replacement allocator or engine change was needed.

The full native allocator suite passes **15 tests and 15 subtests**, 26.28s.
Container `3f3ce250d701360f7dc749cfc4ffac0e4cd0013f710aaae45b93c89d2db507f5`
ran 21:40:00--21:40:29 UTC, exit 0/no OOM. It had no network, GPU or model,
4 GiB/equal swap and two CPUs. All 65 engine and two helper/test hashes were
verified. Evidence: `native-admission-ttl-cpu.json/.log/.xml`; local and remote
XML SHA256:
`a382dda50817a5f31c5d89d0140901f356fef967b62c80e1a8653b302b5ae940`.

This validates allocator ownership across the deadline, not GPU tensor bytes
or concurrent physical transfers. Expanded full-model audit `7cd2ddfdf103`
continues from its original start; last observed 84/200 requests, no error/OOM.
Serving GPUs 4--7 are unchanged. No promotion, commit or push.

The offline resident-page analyzer now also requires its existing
TP4/PP1/MTP4/32k configuration check before either native-ownership or
physical-version qualification can pass. Previously the check was reported
but omitted from those two verdicts. Four negative configuration cases now
exercise that requirement in the existing fixture; all three selected analyzer
tests pass. Reanalysis of the previous raw run is unchanged: ownership passes,
version coverage fails with 18936 exact comparisons, zero differences and 408
unobserved versions. Evidence: `resident-version-topology-gate-check.json`.
This changes only offline validation, not the engine or running GPU capture.

### Native expiry metric export validated (2026-09-22)

Reviewed native observability while the expanded GPU audit initialized.
Local/external prefix-hit and query counters count tokens, not requests;
offload load/store-byte counters measure transfers. These aggregate counters
do not provide per-request cold-load attribution under concurrency.
The CPU usage/read/write gauges measure chunks pinned by active transfers,
not all idle retained entries or allocated pinned RAM. A zero usage gauge
therefore does not mean an empty RAM cache. This distinction is now documented
in `docs/features/kv_offloading_usage.md`.

Extended the existing native connector metrics suite with an actual
`CollectorRegistry` and native `OffloadPromMetrics` using real Prometheus
counter/gauge/histogram classes. A one-chunk native CPU manager with a private
clock exports zero expirations at 3599 seconds, one at 3600 and still one at
7200; the expired key is a miss. No fake metric implementation or replacement
offload backend is used. This verifies manager deltas, native registration and
Prometheus exposition, not a deployed HTTP endpoint or GPU transfer.

The complete metrics suite passes: **23 tests**, 0.25s test time.
Container `2a26c5901607755143a02d20dccf97a120c35e55cd2b3e39d034603c713c2327`
ran 21:35:22--21:35:58 UTC, exited 0/no OOM; root
`/tmp/flash-native-metrics-20260922-7VYEPJ`.
Network none, no GPUs/model, 4 GiB/equal swap, two CPUs and read-only sources.
The 65 engine and 12 test/helper hashes were verified.
Evidence: `native-expiry-metrics-cpu.json/.log/.xml`.
XML SHA256:
`e7ddd2db565d300395d669e4b84faa7c043c6132685500d01f5251498d786e56`.

Only the test and documentation changed; engine code and the active GPU snapshot
are unchanged. Targeted Ruff and diff checks pass. Expanded GPU audit
`7cd2ddfdf103` remains running/no OOM at its original 21:31:03 start, last
observed loading shard 34/38. Serving `7d408c2bc15d` retains its original
September 20 start. No GPU integrity verdict, restart, promotion, commit or push.

### Expanded producer-version capture running (2026-09-22)

The bounded page plan now captures seed boundaries 28320/29264, first-turn
boundaries 28320/29264/30208, second-turn boundaries 29264/30208 and third-turn
boundary 30208, for both hot and cold C2 chains. This adds observations of the
earlier complete physical versions that native lookup may select later.
The page hook allows at most three ordered unique boundaries per phase;
scheduler ownership capture is capped at 32 records. The 4 GiB long-snapshot
hash-work limit, 8 MiB chunk size and 32-phase cap are unchanged.

The analyzer accepts only the recorded legacy or expanded plan and requires
24 or 32 ownership checkpoints respectively. Its version comparison searches
completed earlier phases only, using native prefix key, physical page ID,
group/layer and logical index. It never uses the consumer or future snapshots
as a source. Conflicting observations, missing versions and byte differences
remain failures. The original own-producer and numerical verdicts are preserved.

Reanalysis of the previous raw report still produces 18936 exact version
comparisons, zero observed-version differences and **408 unobserved versions**;
the expanded analyzer does not turn that incomplete capture into a pass.
Synthetic tests exercise self-comparison rejection and require the complete
expanded ownership plan. Three selected state-analyzer tests pass. The engine
session suite passes 138 tests with four dependency/opt-in skips; two
source-preservation tests and targeted Ruff/diff checks pass.

Native CPU container
`e4badbcb261536562925fe68376f124dda620b995ae9c9b7b36695e3073ef6f5`
ran 21:28:18--21:29:15 UTC, exit 0/no OOM: **14 tests and 15 subtests pass**,
53.09s test time. It accepts the 32nd ownership record, refuses an excess record
and preserves native ownership. No network, GPUs or model were available;
the 65 engine and two helper/test hashes were verified. Evidence:
`native-resident-ownership-cpu-v3.json/.log/.xml`.
XML SHA256:
`b3fc713438dab45552747f52a1b0e602a06b866075469a0f4df31b8702d36b8f`.

Full GPU retry
`7cd2ddfdf10395f89cf57c3aae0b0959d495df1bdf841c459e5ff80a4bb76d8a`
is running/no OOM, started **21:31:03.142331547 UTC**, root
`/tmp/flash-session-gpu-20260921-DPWteb`.
Launch evidence: `launch-full-tp4-long-resident-pages-v3.json`.
All 65 engine hashes and 19 kernel flags match v2; only
`session_prefill_trace.py`, `prefix_cache_screen.py` and
`prefix_load_barrier.py` changed. TP4/PP1, Q8 MTP4, FULL decode graphs, C2,
32768 context, 1024 resident host blocks, private native cold buffers and all
memory/GPU/network boundaries remain unchanged.

The last authoritative check confirmed the container running during startup.
Poll this exact handle; do not overlap another GPU trial or restart because an
observation times out. Serving `7d408c2bc15d` retains its original September 20
start. No engine change, serving promotion, commit or push occurred.
Performance qualification remains deferred pending complete version evidence.

### Long resident audit: ownership passes; producer-version coverage incomplete (2026-09-22)

Trial `bd68551a1bef79cef1f8ec59853ec891788b3bbd2409a1b6c84369c36f5b4045`
completed all 200 requests and all four rank captures, then exited 1/no OOM at
21:19:08 UTC (start 20:49:31 UTC). The original combined and numerical verdicts
remain false. Raw report SHA256:
`8a4a076f31e5e6307729efe9004529c3369eac95f2074a92661a0df9997380db`.
Local and remote raw hashes match. Evidence is
`screen-full-tp4-long-resident-pages-v2.json`, ownership/barrier sidecars and
the final 85 log lines. Development GPUs are free; serving retains its original
September 20 start and GPU allocation.

All 24 native scheduler ownership checkpoints and all 48 worker-to-owner joins
pass: correct prefix keys, group/page mappings, positive references and cache-map
membership. Request completion, four-rank coverage and real growing hot/cold
reuse through 30208 cached tokens also pass. However, only **8/48** comparisons
to each branch's immediately preceding producer are byte-exact. The original
strict failure is retained in `long-resident-pages-v2-analysis.json`.

Investigation distinguishes physical versions from logical prefixes. Native
`BlockHashToBlockMap` explicitly retains multiple physical blocks for one hash;
lookup returns the first cached version, not necessarily the version computed
by the immediately preceding request. At the first continuation, both consumer
branches exactly match the captured version from the first producer branch.
Different producer versions may differ numerically; that does not itself show
a transfer mutation.

A separate version-aware analysis binds payloads to rank, native prefix key,
group/layer, logical index and physical page ID. It checks against both captured
producer branches without altering the original verdict:
**18936 page comparisons exact, zero observed-version mismatches, and
408 comparisons without a captured producer version**. These are comparisons,
not unique pages. There are no conflicting captured producer observations.
All same-physical-page own-producer comparisons are exact; every original
mismatch also changes physical page ID.

The 408 missing observations occur on newly included boundary pages in later
turns. They are an explicit coverage gap, not a pass and not established cache
corruption. Both resident continuity verdicts remain false. Artifact:
`long-resident-pages-v2-version-analysis.json`. CPU tests reject changed bytes
on an observed version, mismatched native keys, missing versions and conflicting
producer observations; the existing synthetic ownership/continuity test passes.
No engine implementation was changed.

Next action: extend the bounded diagnostic to capture earlier published physical
versions at the extra producer boundaries, and reconcile consumers against
earlier snapshots of the selected version. Preserve the strict original
comparison and fail on missing versions. Do not relax byte checks or change
native cache selection. The planned performance rerun is deferred until this
integrity-coverage issue is resolved; no serving promotion, commit or push.

### Native iteration analysis now checks context accounting (2026-09-22)

The offline performance analyzer now reports native iteration geometry separately
from frontend timing verification. Before interpreting a phase, it requires
the expected phase name, bounded complete records, contiguous iteration indices,
valid nonnegative counts/times, no unexpected encoder/dummy work, and scheduled
context tokens equal to the sum of prompt tokens minus cache hits. It reports
context-only, mixed and generation CPU elapsed time plus exact batch geometry.
Scheduled generation tokens include speculative verification; they are not
accepted output tokens.

These checks do not establish complete trailing decode capture or eliminate
late frontend attribution. Native iteration collection adds overhead; its
results must not be labeled an uninstrumented benchmark or GPU kernel timing.
The existing 380-request timing evidence still verifies independently and keeps
its original numerical failure. Missing iteration evidence fails the separate
context-accounting check instead of invalidating unrelated stream timestamps.

Three analyzer unit tests pass, including wrong phase, missing records,
duplicate indices, mismatched context totals, excessive request counts and
nonfinite elapsed time. The 138-test engine session suite also passes with
four dependency/opt-in skips; targeted Ruff and diff checks pass. No engine or
GPU-running snapshot changed.

At the latest live observation, long-page audit `bd68551a1bef` had completed
188 requests, captured all 24 scheduler ownership records with passing metadata
checks, and was still running fresh-reference generations with all four dev
GPUs active. Final per-rank resident-byte analysis remains pending; this is not
a final integrity pass. Serving GPUs 4--7 remain unchanged.

### Failed final-rank acknowledgement keeps the source pinned (2026-09-22)

Extended the existing native late-follower/expiry regression with a failed final
worker result, for one and four workers. The four-worker case delivers three
successful acknowledgements, then exercises native
`OffloadingConnectorWorker.get_finished` with a failed transfer result.
The worker raises and publishes no completion metadata; its load entry remains.
The scheduler still has one acknowledgement pending and the exact RAM source
chunks remain referenced after a further 3601-second private-clock advance.
No successful final acknowledgement is invented.

This records vLLM's current **fail-fast** behavior: worker transfer failures
terminate normal engine processing. It does not demonstrate in-process recovery
or successful reuse after a partial transfer. The test uses the actual native
worker/scheduler/CPU-manager logic, with mocked requests and transfer results;
it does not induce a physical CUDA/DMA failure.

Selected native worker completion/submission tests and all four follower
variants pass: **8 passed, 229 deselected**, 0.57s test time.
Container `9d9108d33bae` exited 0/no OOM, 20:53:58--20:54:33 UTC.
Root: `/tmp/flash-native-failed-rank-20260922-unz7zS`.
Evidence: `native-failed-rank-cpu.json/.log/.xml`.
XML SHA256:
`9d9d3e19570ba3979dcdabff463f3a14f156868b98b79c14116e9cbc6cb8c412`.
The 65 frozen engine hashes and 11 test/helper hashes were verified. CPU
container limits remain network none, no GPUs/model, 4 GiB/equal swap and two
CPUs. Only the existing test was extended; engine behavior is unchanged.
Source-preservation tests and targeted Ruff/diff checks pass.

The first full-suite attempt `ea52fcd60f6c` exited 1/no OOM at
21:02:39 UTC: 144 passed, two skipped, and all 91 failures came from
`ModelConfig` validation because the network-isolated fixture could not resolve
`facebook/opt-125m`. This was not a full-suite pass. The terminal runtime,
common failure message and XML remain in
`native-offloading-scheduler-worker-cpu.json/.xml`.

A fresh CPU-only container used the same frozen engine/tests, with a read-only
synthetic OPT configuration at `/native/facebook/opt-125m/config.json` and
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`. The configuration supplies only
scheduler-test model metadata; no model weights or network access were added.
No engine behavior, test assertion or test selection was changed.

That full-suite retry **passed: 235 passed, two skipped**, 37.61s test time.
The two skips are native cache-registration cases whose GPU-backend parameter
sets are empty in this CPU-only environment; they are not GPU-copy coverage.
Container `3efab747b3ffeb54d269a28e60553bce726d1d9288ce961e1e48e8cf531bcadb`
ran 21:04:40--21:05:49 UTC, exited 0/no OOM, with the same 4 GiB/two-CPU
limits and no GPU/model access. All 65 engine and 11 test/helper hashes were
rechecked against the frozen files after completion.

Evidence: `native-offloading-scheduler-worker-offline-cpu.json/.log/.xml`
and `offline-opt125m-config.json`. The JSON records the full reproduction
command, fixture hash, source hashes, runtime and skip reasons. XML SHA256:
`4828b13ad70a01898a2757b3b8f506a11d35d0dd12529849ce423eb9a64ef3f6`.

The long-page GPU retry `bd68551a1bef` remains running/no OOM, last observed
at 122 completed requests and 18 scheduler ownership checkpoints. Its exact
start remains 20:49:31 UTC. This is progress, not a final GPU integrity verdict.
Serving remains at its original start; no restart, commit, push or promotion
occurred.

### Long-page limit failure retained; ownership observer passes CPU tests (2026-09-22)

The first long-page trial `d153ad508dcb` is terminal: exit 1/no OOM,
20:28:56--20:42:25 UTC. It completed 76 base requests and stopped during the
first long hot-seed prefill, before producing a long-context snapshot.
The diagnostic counted **2996949376 bytes** against its **2147483648-byte**
work limit and raised. This is an incomplete audit, not a cache-corruption
finding and not a long-context pass. The independent verifier rejects it.

Evidence: `screen-full-tp4-long-resident-pages.json`, barrier, final 160 log
lines and `long-resident-pages-bound-failure.json`. Raw SHA256:
`e3fb110d2fab7e9c3e59401989bfec124ccbba3791d8fca82ea574530396a94b`.
The failed container and frozen inputs are retained.

The corrected observer keeps **all allocated state** in the snapshot. Its work
limit is 4 GiB only for boundaries >=8192; shorter boundaries keep 2 GiB.
Hashing chunks remain 8 MiB, table bounds remain 4096, and no model/cache
capacity changed. A metadata-only tensor test crosses the old limit without
allocating GiBs, verifies acceptance below 4 GiB, and verifies rejection above
4 GiB before hashing. Existing real small-tensor hash/no-mutation tests remain.
This tests the diagnostic limit, not actual multi-GiB data integrity.

A new read-only observer uses native `resolve_block_hashes`,
`make_block_hash_with_group_id`, request block objects and cache-map
`contain` membership. It checks expected prefix/group hashes, hash-token
boundaries, pool/object identity and positive references. Multiple native
versions of the same prefix are allowed; comparing against only the first
lookup result would incorrectly reject valid versions. Observation does not
call cache lookup, expire pages or alter references.

Two isolated CPU trials passed the complete allocator suite: 14 tests and
15 subtests each, no GPU/model access, network none, 4 GiB/equal-swap and two
CPUs. The second also tests collection at the selected native scheduled
boundary, ignored unrelated phases and duplicate rejection:

- `f6c774ed4117`, root `/tmp/flash-native-ownership-20260922-pntlUe`,
  exit 0/no OOM, 26.84s test time; `native-resident-ownership-cpu.json/.log/.xml`.
- `e52505ed5ce4`, root `/tmp/flash-native-ownership-v2-20260922-wij6qv`,
  exit 0/no OOM, 25.56s; `native-resident-ownership-cpu-v2.json/.log/.xml`.

The existing diagnostic scheduler now records these ownership witnesses when
paired long-page auditing is selected, at the same producer/consumer prefix
boundaries as the worker snapshots. It retains up to 24 scheduler checkpoints
in `prefix-resident-ownership.json`; no scheduling policy is changed.
The independent continuation analyzer joins those records to every worker
rank's physical page IDs and compares native prefix keys across each chain.
Ownership and byte-continuity verdicts remain separate. CPU analyzer tests
reject a wrong physical mapping and duplicate scheduler record even when
worker bytes still match.

Local checks: 138 session tests (134 pass/four dependency or opt-in skips),
two source-preservation tests, three selected state-analyzer tests and targeted
Ruff pass. The 65 engine pins are unchanged.

The corrected retry is now running as
`bd68551a1bef79cef1f8ec59853ec891788b3bbd2409a1b6c84369c36f5b4045`,
started 20:49:31.963228254 UTC, root
`/tmp/flash-session-gpu-20260921-YFPILg`. Its launch record is
`launch-full-tp4-long-resident-pages-v2.json`. All 65 engine hashes and 19
kernel flags match the failed trial; only diagnostic helpers changed.
The ownership sidecar is `results/prefix-resident-ownership.json`.
The last check confirmed running/no OOM during initialization. Serving
`7d408c2bc15d` remains at its original September 20 start time.
Poll this exact retry; no additional GPU trial should overlap it.
No new GPU integrity result or production promotion is claimed.

### Native iteration diagnostics prepared for the C2 latency anomaly (2026-09-22)

Read-only inspection narrows the next measurement. `PairLoadGate.branch` only
recognizes `batch-cold-[01]` (plus the native random suffix). It does not hold
the warmed benchmark's `perf-*` requests. Thus the earlier controlled cold
pair and the warmed performance phases do not have identical promotion rules.

The completed warmed report shows these first-token gaps between its two
requests, calculated on their common absolute frontend clock:

| Measured repeat | Hot gap (ms) | Cold gap (ms) |
| --- | --- | --- |
| 1 | 0.01110 | 319.828 |
| 2 | 0.01101 | 322.304 |
| 3 | 0.01077 | 598.537 |

Evidence: `warmed-prefix-c2-admission-analysis.json`, using raw report SHA256
`92006b2c1ecd30ab0baa83fd0e5f22bf912eeeb0cf855685a020089f23629798`.
Staggered frontend output supports inspecting scheduling geometry; it does
not prove batch shapes or establish the cause of the TTFT difference.

The diagnostic launcher and screen now accept the optional
`--performance-iteration-details` flag only with a warmed performance fixture.
This sets vLLM's existing `enable_logging_iteration_details` EngineArgs option.
The existing frontend StatLogger captures native `SchedulerIterationDetails`
for each performance phase: context/generation request and token counts,
iteration index, encoder counts and elapsed engine-core CPU wall time.
No engine implementation, scheduler policy, worker hook or CUDA synchronization
was added.

Capture is bounded to 1024 records per phase. Missing records, duplicate or
out-of-order indices, invalid counts/times and overflow fail its coverage gate.
Raw iteration records are retained with the phase rather than interpreted as
GPU kernel time. Late frontend statistics can cross phase boundaries; compare
context-token totals with the actual uncached suffix before attributing timing.
Collection/logging overhead means this is a diagnostic follow-up, not the
uninstrumented benchmark itself.

Local validation: 138 session tests (134 pass, four dependency/opt-in skips),
two source-preservation tests and targeted Ruff pass. Both command parsers
advertise the new option. Tests cover mixed prefill/decode geometry, default
off behavior, invalid/duplicate/overflow records and missing phase coverage.
All changes are diagnostic/test-only and are not in the frozen running audit.

The long-page audit `d153ad508dcb` remains running/no OOM at its original
20:28:56 UTC start time. The engine has started and recorded five requests,
last `pressure-2`, with no reported error. No integrity verdict is available
yet. Do not restart it or launch another GPU trial over it.
Serving and the 65 engine source pins remain unchanged.

### Long-prefix resident-RAM audit running (2026-09-22)

Container `d153ad508dcbe9bfa9e199c147f0e0c166c788b0c1f624a926622cde64868d6b`
started at 20:28:56.117030592 UTC and was confirmed running/no OOM during
engine initialization. Root: `/tmp/flash-session-gpu-20260921-vkKDHL`.
Launch record: `launch-full-tp4-long-resident-pages.json`.

This is the prepared 200-request TP4/PP1/Q8 MTP4 32k C2 continuation audit,
using FULL decode graphs, all 19 optimized flags, 2048/944 balanced prefill,
1024 resident host blocks, native private cold buffers and the existing paired
admission diagnostic. It hashes retained RAM pages before the next forward;
synchronization changes timing, so this is not a benchmark or race-freedom test.
No new long-context state result is available yet.

All 65 engine hashes and 19 flags match the completed warmed benchmark.
Only diagnostic helpers differ. Network remains none, weights read-only,
240 GiB/equal swap container ceiling, and only development GPUs 0--3 are used.
Serving `7d408c2bc15d` was verified running at its original September 20
19:45:53.401210251 UTC start time. No serving restart, route/context change,
commit, push or promotion occurred.

Poll this exact audit container; do not restart it on observation timeout.
If rollback is needed, stop only this new audit and retain its artifacts.
After completion, retrieve `results/screen.json` and run the k3s analyzer:

```sh
/home/water/Documents/work/vllm-qwen3.8-flash-next/.venv/bin/python \
  infra/inference/analyze_flash_prefix_state.py REPORT.json \
  --engine-root /home/water/Documents/work/vllm-qwen3.8-flash-next \
  --continuation-pages
```

Preserve the original numerical/combined verdict separately from the
resident-RAM continuity result. Broader logical ownership, unpaused lifecycle
coverage and final human review remain open.

### Warmed benchmark complete; long-prefix byte audit prepared (2026-09-22)

The warmed trial `51e18e41019b` completed at 20:16:30 UTC, exit 1/no OOM.
All 380 requests are complete and finite. Independent raw-row analysis
recomputes all six timing/acceptance summaries and passes seven measurement
checks. There are three measured cycles after one discarded warm-up per C.
The original combined numerical screen remains false/AssertionError.

TP4/PP1/Q8 MTP4, 7100 input tokens, 128 output tokens, 5664 cached tokens:

| C | Mode | Median TTFT (s) | Median aggregate post-first tok/s | Draft acceptance |
| --- | --- | --- | --- | --- |
| 1 | Fresh | 3.178 | 94.06 | 76.58% |
| 1 | Hot | 0.523 | 118.18 | 75.52% |
| 1 | Cold | 0.524 | 102.77 | 75.26% |
| 2 | Fresh | 16.343 | 139.66 | 70.34% |
| 2 | Hot | 3.785 | 144.89 | 72.11% |
| 2 | Cold | 1.938 | 137.88 | 75.52% |

These are frontend synthetic-fixture measurements with diagnostic batch
admission, not isolated prefill or production latency. C2 cold TTFT is lower
than hot; the scheduling/geometry cause is unresolved. Positive phase transfer
bytes do not prove both C2 requests individually loaded. Different output
histories and acceptance confound causal decode comparisons. Three repeats
do not establish confidence bounds. Each measured cold phase loaded
141430784 bytes; hot/fresh phases loaded zero. No cache-integrity or quality
claim follows from the timing result.

Evidence: `screen-full-tp4-warmed-prefix-performance.json`,
`full-tp4-warmed-prefix-performance-barrier.json`, retained final 80 log lines,
and `warmed-prefix-performance-analysis.json` with runtime, source/report
hashes and reproduction command. Raw SHA256:
`92006b2c1ecd30ab0baa83fd0e5f22bf912eeeb0cf855685a020089f23629798`.
The independent analyzer is `infra/inference/analyze_flash_prefix_performance.py`
in the k3s repository. Two regression tests pass, including missing transfer,
wrong namespace, invalid timing and altered-summary negative controls.

The bounded page reader is now connected to the C2 continuation fixture.
Each hot/cold chain captures its own producer and next consumer at
28320/29264/30208 tokens. Intermediate turns capture two ordered boundaries;
fresh references remain untraced. Prefix-input SHA256 values and namespaces
bind each comparison to the corresponding earlier input. Raw snapshots are
saved before coverage indexing so a coverage failure does not erase evidence.
Worker rank IDs are explicitly recorded.

The separate `analyze_flash_prefix_state.py --continuation-pages` mode requires
all four ranks and 48 rank/mode/turn/branch comparisons. It validates both RAM
groups, every one of the 12 target and one draft layer, complete logical page
ranges, physical table bindings, BF16 geometry and hashes. It compares each
chain with its own producer, not an independently recomputed hot/cold state.
Full native logical-key/refcount ownership, GPU recurrent state and race freedom
remain separate gates.

Local validation: 137 session tests (133 pass, four dependency/opt-in skips),
two source-preservation tests, and three selected state-analyzer tests pass.
The new state test rejects a corrupted consumer page, mismatched prefix hash,
missing group/layer and invalid page geometry/table. Targeted Ruff passes.
Only diagnostics/tests/docs changed; the 65 engine pins are unchanged.
GPU qualification of the new long-page protocol is still pending.

### Bounded 32k prefix-page reader prepared (2026-09-22)

The existing worker diagnostic now accepts integer snapshot boundaries below
32768 tokens and an explicit per-phase `page_boundary`. Installation accepts
a prompt bound up to 32768; default page phases still use 5664. Each explicit
boundary must be positive and below the installed prompt bound. Shape-only
traces reject page-boundary overrides. The phase cap is 32 (previously 16) to
fit the existing base screen plus bounded continuation phases.

The original 2 GiB per-snapshot payload cap, 4096-entry block-table cap,
8 MiB hashing chunks, native slot/pool bounds, pre-read synchronization and
method restoration remain. No cache bytes, scheduling rules or engine kernels
are changed. This reader distinguishes complete prefix pages from the private
writable tail; it does not itself assert that independently recomputed states
must match or establish scheduler-key ownership.

CPU tests exercise 28320/29264/30208 phase boundaries and reset to the old
5664 default. A small host-tensor fixture at 30208 checks 32 complete
944-token pages plus one writable tail, exact page hashes and no mutation.
Invalid/type-mismatched bounds, excess phases and existing byte-budget guards
fail closed. Local suite: 136 tests, 132 pass and four dependency/opt-in skips;
two source-preservation tests pass; targeted Ruff and diff checks pass.

This is a diagnostic primitive only. It is not yet connected to the long
continuation screen or independently analyzed at those boundaries, and no new
GPU byte result is claimed. The active warmed benchmark uses its earlier
frozen helper; all 65 engine source files remain unchanged.

At 242 requests, all three C1 measured cycles were complete and C2 warm-up
was underway. C1 hot/cold TTFT observations were approximately 0.52 seconds,
with variable decode rates and MTP acceptance. These are preliminary raw
observations, not the final benchmark verdict.

### Native RAM-pool pressure at 240k-sized prefix counts (2026-09-22)

Extended the existing native host-pool pressure test to one/two/four/eight
independently retained prefixes. Each seed has 254 complete 944-token pages
(239776 tokens), two host groups for target/MTP, and a private writable tail
per group. Then 32 distinct short requests apply allocation pressure.

| Retained seeds | Tested host block capacity | All seed prefixes retained after pressure |
| --- | --- | --- |
| 1 | 1024 | Yes |
| 2 | 1024 / 2048 | No / Yes |
| 4 | 2048 / 4096 | No / Yes |
| 8 | 8192 | Yes |

These are native allocator/LRU metadata tests, not allocated tensor-byte
budgets or 240k model-generation qualification. They do not establish minimum
capacities, GPU fit, active concurrency, or correct full-model restoration.
The pool also needs room for retained history, private tails and intervening
requests; aggregate context alone is not a retention guarantee.

The four-suite CPU regression finished in `4a7847c2b214`,
`/tmp/flash-native-capacity-20260922-70xCcl`, 20:02:03--20:02:43 UTC:
225 pass, one CUDA-only skip, zero failures, exit 0/no OOM.
All ten pressure cases pass, including six new large-prefix cases.
All 65 engine and nine test/helper files were hash-verified. Evidence:
`native-capacity-240k-cpu.json/.log/.xml`; XML SHA256
`024dc63836280868b091d1b7470d1a6045478d5761a33fbc7d6f5aafd66b38ce`.
No engine code, serving budget, or live placement changed.

The reduced GPU harness still has an 8192-context limit, 32-entry request
block tables and deliberately small pools; increasing only prompt length
would not qualify long context. Its sizing and bounds must be updated together,
or the existing full-model 32k continuation audit must gain resident-page
coverage. This remains future GPU validation, not a claimed pass.

The warmed benchmark remains live at 149 requests (C1 measured repeat two
starting). Its discarded C1 warm-up proves the intended tiers: prime/fresh
miss with zero loads; hot/cold both reuse 5664 tokens; only cold loads
141430784 bytes. No final performance summary is available yet.

### Native late-follower expiry regression passes (2026-09-22)

Added a case to the existing native offload scheduler test suite, not a new
cache implementation. It uses real `OffloadingConnectorScheduler` and
`CPUOffloadingManager`, the existing synthetic hybrid attention/recurrent
request fixture, and one/four worker acknowledgements.

At 3601 seconds, unacquired follower entries expire and its lookup returns a
miss without creating a load job. The leader's already prepared load keeps
every source chunk pinned until the final worker acknowledgement. Completion
starts a new idle TTL; the retained keys remain hits and expire at 7201
seconds, returning all 32 CPU slots. Both cases pass, no skips, in container
`bb1491a230d8` (19:53:12--19:53:46 UTC, exit 0/no OOM).
Artifacts: `native-follower-expiry-cpu-v2.json/.log/.xml`; XML SHA256
`569dd2c5506ed0a1e8c4315eaa86e90ed9617557b5c9e146d6b8467afb961176`.
All 65 engine and ten test/helper files were hash-verified.

The initial trial `ca6d013ca381` failed an additional test expectation that a
new 28-token partial-tail lookup would survive expiry of its earlier complete
recurrent checkpoint. Native lookup does not promise that. The corrected test
checks the protected source entries and renewed TTL directly; preceding
follower-miss and all-worker pin assertions passed in both runs. The failed
trial is retained as `native-follower-expiry-cpu.json/.log/.xml`.
No engine source changed.

These are CPU metadata tests with synthetic requests/destination IDs, not
model execution, DMA, resident RAM-QSA byte checks or a full-model late
follower qualification. The warmed GPU benchmark keeps its frozen snapshot
and remains active. Adjacent native scheduler regressions also passed:
13 tests, 210 deselected, no failures/skips, in `9b463d4ab8ff` at
19:55:13--19:55:44 UTC (exit 0/no OOM). Evidence is
`native-follower-expiry-regression.json/.log/.xml`. The GPU benchmark had
completed 23 setup requests without a reported error at this check; measured
performance cycles were still pending. Serving identity/start is unchanged.

### Warmed C1/C2 benchmark started (2026-09-22)

Container `51e18e41019b`, root `/tmp/flash-session-gpu-20260921-PoF0kI`,
started at 19:46:20 UTC and is running/initializing. Poll this exact handle;
an observation timeout is not permission to restart it. The launch record is
`launch-full-tp4-warmed-prefix-performance.json`.

This isolated development-quartet run selects TP4/Q8 MTP4 FULL decode graphs,
three measured repetitions plus one discarded warm-up per C1/C2, proven
fresh/hot/cold tiers, no byte audit, and no expiry/cancellation fixture.
All 65 engine hashes and 19 optimization flags match the active-expiry audit;
the launcher verified all 73 uploaded engine/helper files. Maximum context
remains 8192; the fixture is not a production-context capacity qualification.
There is no performance result yet. Serving GPUs 4--7 remain untouched.

### Full-model active expiry and cancellation state audit passes (2026-09-22)

Container `82a271938dee` finished at 19:41:11 UTC, exit 1/no OOM.
The 114-request run contains 113 complete finite normal requests and one
intentional partial native abort. Independent reanalysis passes all ten
lifecycle/state checks and reproduces the recorded four-rank byte assessment.

With both requests active, private diagnostic clocks advance 3601 seconds
against the configured 3600-second idle TTL. Native expiry removes 491 idle
host blocks and 2611 idle CPU chunks while retaining 20 referenced host blocks.
Both requests retain their 210-entry/41-group ownership tables and reference
counts across expiry. After the peer abort, survivor references drop from 228
to 210 as expected; its mappings and bytes remain unchanged on every rank:
732 page views and 321079488 hashed bytes per rank. Both later requests reuse
5664 prefix tokens; the cold pair actually loads 141430784 bytes.

No CPU transfer chunks were pending at the pause (`protected_cpu_chunks=0`).
This run does not establish pending-transfer expiry protection; the separate
native CPU/prepared-load tests cover that case. The paused audit also cannot
prove unpaused race freedom or performance, and remains an 8192-context C2 test.

The combined screen remains false/AssertionError: survivor/reuse generated
tokens differ at index 14 and logprobs are not exact. Numerical variability
remains separate, not declared harmless. Raw screen, abort ownership, barrier
and logs are retained; `active-expiry-audit-analysis.json` contains the
independent reanalysis, reproduction code and source SHA256
`a7f3993b33fe37d3763ef26407ef57a2c321f7333f54ee749d7901f319643c84`.
Serving `7d408c2bc15d` remains running at its original September 20 start.

Local verification rerun: 135 tests (131 pass, four skips), plus two source
preservation tests pass. No engine/kernel change was needed for this result.
The next isolated run measures warmed C1/C2 fresh/hot/cold timing and MTP
acceptance without byte-audit instrumentation.

### Warmed C1/C2 prefix benchmark prepared (2026-09-22)

The existing untraced 32k diagnostic is not a matched performance verdict:
for example its initial C1 hot/cold decode observations are 85.1/103.0 tok/s
with different MTP acceptance rates (44.0%/59.9%), and first-use JIT affected
some long-turn timings. These values are diagnostic observations, not cache
speedup/regression claims.

The native `benchmarks/benchmark_prefix_caching.py` was reviewed. It measures
bulk repeated-prefix elapsed time but does not distinguish proven GPU-hot versus
native CPU-cold hits or collect this fixture's per-request MTP/stream evidence.
The new `tools/flash_next/prefix_cache_benchmark.py` therefore reuses the
existing native AsyncLLM request/statistics callbacks and adds only bounded
trial orchestration; it does not replace native caching or transfers.

Opt-in `--performance-repeats {3,5}` requires untraced TP4/Q8 MTP4 graph mode,
balanced C2 prefixes and no overlapping expiry/cancellation/byte audit.
Each C1/C2 series discards one complete warm-up cycle, then repeats identical
prompts/settings with separate cache namespaces. Every cycle primes, measures
a hot hit, applies 32 pressure requests and measures a proven cold hit.
Cache-bypassed fresh references alternate before/after this sequence.
The fixture requires complete finite outputs, actual concurrency, real cold
load bytes, matched hot/cold hit lengths and fresh misses.

Results retain raw frontend timestamps, TTFT, phase e2e output rate, aggregate
post-first-output decode rate, and draft acceptance/mean acceptance length.
Aggregate decode uses one common clock interval and excludes each initial
speculative burst; it does not sum overlapping per-request rates. Summaries
exclude warm-up, use medians and ranges, and state that few repeats do not
establish confidence bounds. Output/numerical differences are not hidden.
This mode is C1/C2 only and does not satisfy broader-concurrency qualification.

CPU tests cover warm-up exclusion, tier proof, alternate reference order,
shared-clock throughput and invalid MTP evidence. 135 local tests run:
131 pass and four dependency/opt-in skips; source preservation two tests
pass; targeted Ruff/diff checks pass. No benchmark GPU result exists yet.
The active-expiry trial keeps its earlier frozen sources and is unchanged.
After it finishes, the three-repeat option implies 380 requests including
the existing 76-request prefix screen and gets a bounded 3600-second timeout.

### Full-model active-expiry audit running (2026-09-22)

Container `82a271938dee`, root `/tmp/flash-session-gpu-20260921-WTTxZM`,
started 19:23:48 UTC and is running/initializing. Poll this exact handle after
observation timeouts; do not restart it. The launch record
`launch-full-tp4-active-expiry-audit.json` selects eager TP4/Q8 MTP4,
C2 divergent prefixes, paired native cold loads and the paused
`--active-cancel --audit-active-cancel --expire-active-cache` fixture.
All 65 engine hashes and 19 kernel flags match the successful copy audit.
The active-expiry helper has native CPU integration evidence but no full-model
GPU verdict yet. Serving remains unchanged.

### Full-model native copy integrity and counters pass (2026-09-22)

Retry `464f7a4b69b2`, root `/tmp/flash-session-gpu-20260921-wfsB1Z`,
completed 18:51:08--19:20:24 UTC, exit 1/no OOM. All 200 requests are
complete/finite. Every rank returned 8245 consecutive audit records:
8238 stores (6773876736 bytes) and seven loads (330553344 bytes).
Every observed source/destination copy is exact, source bytes stay unchanged,
and each CPU load matches its recorded stored version. Each rank retained
2611 version entries at collection.

Independent analysis passes all nine copy checks. Across four ranks, the
32952 stores / 27095506944 bytes and 28 loads / 1322213376 bytes match native
completed-job counters exactly. Thus the successful run crosses the old
4096-record cap and closes the observed payload/counter reconciliation.
This synchronous audit changes timing; it is not a performance benchmark or
proof of race freedom, logical scheduler-key ownership, or all resident RAM-KV
bytes at long context.

Independent continuation analysis also passes all 13 checks: 200 unique
complete requests, 14 recorded C2 pairs, raw-row agreement, actual concurrency
two, unpadded FULL graph dispatch, input/settings replay, fresh misses, and real
cold loads through input lengths 30700/31644/32588 with
28320/29264/30208 cached tokens per branch. Cold pair loads remain
220368896/223657984/453894144 bytes.

The original combined screen stays false / AssertionError / exit 1 because
numerical consistency is false. Across hot/cold/two-fresh comparisons, one of
six turn/branch token comparisons differs; all six logprob comparisons differ.
These are not declared harmless or quality-equivalent. Keep them separate
from the passing observed copy/lifecycle gates, per the user's decision.

Evidence: `native-copy-audit-v2-analysis.json`,
`native-copy-audit-v2-lifecycle-analysis.json`, raw
`screen-full-tp4-graph-c2-copy-audit-v2.json`, its log/barrier and launch record.
Raw report SHA256:
`bd573469bf9bcaf17ea0b3bad1b0d86a80d5a14608cc6d8a553cac646ff66586`.

The next isolated trial selects the already CPU-tested paused active-expiry
audit: `--active-cancel --audit-active-cancel --expire-active-cache`.
It uses eager TP4/Q8 MTP4 at 8k context, not the graph/copy-audit timing mode.
Its launch record is `launch-full-tp4-active-expiry-audit.json`.
The development quartet was verified free after the retry exited; serving
`7d408c2bc15d` remains at its original September 20 start time. No serving,
weight, kernel or engine source changes, commits or pushes occurred.

### Independent native-copy reconciliation prepared (2026-09-22)

Local `infra/inference/analyze_flash_native_copy.py` independently checks
all four rank records, contiguous copy indices, both directions, exact
source/destination hashes, retained CPU-version verification, payload metadata,
rank totals and complete unique finite responses. It separately reconciles
audited bytes/copy counts with native completed-job counters. A counter gap
requires review; it is neither a corruption diagnosis nor a complete-coverage
pass. Original combined/numerical verdicts are retained unchanged.

Three CPU analyzer tests pass, including missing-rank/corrupt-state/incomplete
output and counter-gap cases; Ruff passes. Running it against the saved failed
102-request audit correctly fails integrity and reconciliation. Evidence:
`native-copy-audit-incomplete-analysis.json`. This is verifier evidence, not
a new successful GPU test. The corrected trial `464f7a4b69b2` is still live
and had recorded 36 requests at the latest check; it must finish before its
final records can be assessed. No running sources or serving state changed.

### Native CPU active-expiry integration passes (2026-09-22)

Container `085e0e8e9adb`, root
`/tmp/flash-native-active-expiry-20260922-E6j928`, ran 18:54:51--18:55:26 UTC,
exit 0/no OOM. One native-manager integration test passed in 31.374 seconds.
It exercises actual `IdleExpiringHostBlockPool`, native device block pool and
`CPUOffloadingManager` with a synthetic scheduler request table: one shared
prefix, two private suffixes, one idle entry in each tier, a pinned CPU load
and an unfinished CPU store. Private-clock expiry removes exactly the two
idle entries while preserving all owned mappings and pending source/destination
slots. Releasing one branch preserves the peer; final release restores the
native free-block counts. Clocks are restored and expired lookups miss.

The CPU-only container uses the same 65 frozen engine modules as the running
copy-audit retry. All 73 engine/helper/test hashes were verified. No GPU,
model weights or production resources were mounted; network none, 4 GiB
memory/equal swap and two CPUs. Evidence: `native-active-expiry-cpu.json`
and its log. Local tests now discover 133 cases: 129 pass and four
dependency/opt-in skips; the new opt-in case passed in this native CPU run.
Source-preservation tests (two), targeted Ruff and diff checks pass.

This proves the observed manager-level ownership/expiry integration, not
model-state bytes during a live GPU request or arbitrary concurrent execution.
The prepared full-model active-expiry fixture remains pending behind the
currently running native-copy audit.

### Corrected native copy audit running (2026-09-22)

Retry `464f7a4b69b2`, root `/tmp/flash-session-gpu-20260921-wfsB1Z`,
started 18:51:08 UTC and is running/initializing on development GPUs 0--3.
Its launch record is `launch-full-tp4-graph-c2-copy-audit-v2.json`.
All 65 engine hashes and 19 kernel flags match the failed first audit exactly.
Only diagnostic helpers changed; `expire_active_cache=false` keeps the newly
prepared active-expiry fixture out of this retry. No final result yet.
Poll this exact handle after observation timeouts; do not restart it.
Serving remains unchanged.

### Copy audit stopped at a diagnostic bound; corrected retry (2026-09-22)

`ce13bc9d26e87` exited 1/no OOM at 18:44:52 UTC, after 102 complete/finite
recorded requests. All four workers raised `Native transfer exceeds audit
bounds` in the diagnostic wrapper during a native store. No final copy records
or copy verdict returned; the subsequent collection RPC also found the engine
dead. This is an incomplete failed test, not an integrity pass or an observed
copy mismatch. Earlier GPU allocation warnings did not terminate the run.

The saved counters show 16332 completed stores and 12 loads across four
workers (4083+3 per rank), consistent with reaching the wrapper's 4096-record
cap soon afterward. The old combined error did not identify its specific
bound, so that attribution remains an inference. The retry raises only the
bounded metadata record cap to 32768 and identifies record versus
descriptor/payload bounds explicitly. The 2 GiB per-copy, 65536-descriptor
and 65536-CPU-version bounds and every copy/version comparison remain.
A CPU test exercises 4097 exact copies and rejects at the new cap before
calling native copy. Failed final collection now records its error without
replacing the original exception. No engine/kernel/model/cache-budget changes.

Evidence: `native-copy-audit-bound-failure.json`, full raw screen/log/barrier,
and `launch-full-tp4-graph-c2-copy-audit-v2.json` for the retry.
Development GPUs 0--3 were verified free after the terminal failure before
retry launch; serving retains its original start time.

### Active-request TTL/cancellation audit prepared (2026-09-22)

The existing paused abort fixture accepts `--expire-active-cache` only with
the active-cancel byte audit. Before native victim cancellation, private
diagnostic clocks advance 3601 seconds; native host/CPU expiry runs while two
live requests retain their pages. The observer checks both request mappings,
hashes and unchanged references, all referenced host pages and any pending
transfer chunk identities/slots. The existing all-rank survivor byte audit
spans expiry plus cancellation. The fixture requires real idle host/CPU
expiry, restored clocks, preserved ownership, survivor completion and reuse.

This is diagnostic code only, not a new eviction mechanism or serving
setting. CPU tests cover protected-reference/hash/CPU-slot mutation, missing
owners, clock restoration, expiry-before-abort ordering and missing evidence.
132 local tests run: 129 pass, three dependency/GPU skips; source-preservation
two tests pass and targeted Ruff/diff checks pass. No active-request TTL
GPU result is available yet; the copy audit remains the next GPU trial.

### Current frozen-source native CPU regression passes (2026-09-22)

Container `3d8be01dcb54`, root
`/tmp/flash-native-regression-20260922-qlho2a`, ran 18:41:00--18:41:44 UTC,
exit 0/no OOM. The four native manager/factory/packing/pool suites report
219 passed, one CUDA-only skip and 15 warnings in 37.67 seconds.
All 65 engine modules came from the running full-model audit's frozen
`GW6RZG` source; all 74 engine/test hashes were verified before launch.
The container had no GPU access or model weights, network none, 4 GiB memory
with equal swap limit, two CPUs and read-only source/test mounts.

Evidence: `native-regression-graph-cpu.json`, its log and JUnit XML.
This refreshes CPU configuration/ownership/TTL regression evidence, not
full-model state integrity. Serving `7d408c2bc15d` remains at its original
September 20 start time. Full-model audit `ce13bc9d26e87` is still running:
76 initial requests were recorded and long continuations have not yet produced
a final verdict. Allocation warnings were followed by continued inference;
they are not classified as a terminal OOM or silently removed from evidence.

### Full-model native copy-version audit running (2026-09-22)

Container `ce13bc9d26e87`, root `/tmp/flash-session-gpu-20260921-GW6RZG`,
started at 18:28:06 UTC on development GPUs 0--3 after both weight-free CUDA
probes passed and the quartet was confirmed free. Docker reports it running,
still initializing; no copy-audit verdict is available. Poll this exact handle,
not a replacement, after observation timeouts. Serving `7d408c2bc15d` remains
running at its original September 20 start time.

Launch/source record: `launch-full-tp4-graph-c2-copy-audit.json`.
All 65 engine source hashes and 19 kernel flags match the completed untraced
C2 continuation run. The same 200-request C1/C2/32k fixture now selects
`--audit-native-copies` instead of untraced observation. The worker audit
checks actual native stores/loads and the retained CPU source version; it is
explicitly timing-perturbing and not a performance comparison. Graph decode,
TP4/PP1/Q8 MTP4, cache budgets and native LRU/3600-second TTL are unchanged.
Lifecycle, copy integrity and numerical verdicts remain distinct. No production
promotion, weights, routes, commits or pushes changed.

### Graph-mode concurrent 32k lifecycle passes (2026-09-22)

Container `6713c2ac7027` completed at 18:20:55 UTC, exit 1/no OOM, after all
200 requests. All eight fixture lifecycle checks pass. Independent analysis
also verifies 200 unique complete/finite responses, all 14 C2 phases matching
their raw request records, actual concurrency two, unpadded FULL ten-token
graph dispatch in every pair, identical replayed prompts/settings, no loads
for hot/fresh phases, and real loads for every cold continuation.

| Turn | Input tokens per branch | Hot/cold cached tokens per branch | Cold pair load bytes |
| --- | ---: | ---: | ---: |
| 1 | 30700 | 28320 | 220368896 |
| 2 | 31644 | 29264 | 223657984 |
| 3 | 32588 | 30208 | 453894144 |

Fresh references report zero cached tokens. Pair load totals are recorded
once; overlapping per-request counters are not summed. The last turn's larger
load total is preserved as measured, without assigning a cause from totals alone.
Both hot and cold chains retain their actual prescribed conversation inputs
and changed sampling penalties through all three turns.

Original numerical consistency remains false, as does the original combined
screen; its AssertionError and exit 1 are preserved. Log probabilities differ
in every branch comparison, and tokens differ in four of the six turn/branch
comparisons across hot/cold/two-fresh paths. These are not asserted harmless.
This run proves the observed long-context lifecycle, not long-context cache
byte equality or quality equivalence.

Evidence: `screen-full-tp4-graph-c2-continuations.json`,
`full-tp4-graph-c2-continuations-barrier.json`,
`graph-c2-continuations-analysis.json` (includes reproduction code), and the
launch/source record. Raw report SHA256:
`e00e25068fa660924a246f57a5cee2593520f6538a72d78a494ff25f17cf41ab`.

The copy-audit CUDA smoke `4bbf30a622ce` then ran on development GPU 0 only,
18:22:09--18:22:26 UTC, exit 0/no OOM. The real native small/large copy paths
passed with two GPU pages per CPU chunk, relocated destinations, exact audited
bytes and unchanged other pages. Evidence: `native-copy-audit-cuda.json` and
its log. This validates the diagnostic helper's native CUDA roundtrip, not
full-model transfer provenance.

The separate native pending-zero/load regression `417efa2c8451` ran
18:24:04--18:24:33 UTC, exit 0/no OOM, after the copy test exited and the
quartet was confirmed free. All three CUDA repetitions passed without the
synchronous copy-audit wrapper. This covers the known delayed-zero/load
ordering hazard, not every asynchronous race. Evidence:
`native-load-ordering-cuda.json` and its log. Serving remains unchanged.

The full-model native-copy-audited C2/32k follow-up is being uploaded/launched
with the same engine sources, budgets, all 19 flags and graph decode. It uses
`--audit-native-copies` instead of `--untraced-balanced-prefix`; it is not a
performance benchmark. No full-model copy-audit outcome exists yet.

### Native CUDA probes prepared, stopped (2026-09-22)

Two weight-free probes are created but MUST NOT start while
`6713c2ac7027` is running:

- Copy-audit roundtrip: `4bbf30a622ce`, root
  `/tmp/flash-native-copy-audit-20260922-YLFTIy`; record
  `native-copy-audit-cuda.json`. All 73 engine/helper/test hashes verified.
- Native pending-zero/load ordering: `417efa2c8451`, root
  `/tmp/flash-native-load-ordering-20260922-L7xR0v`; record
  `native-load-ordering-cuda.json`. Calls the existing
  `test_load_waits_for_pending_compute_stream_writes` with its normal default
  `VllmConfig` context: three CUDA repetitions, no copy-audit wrapper.

Both use only development GPU 0 when started, the existing pinned image and
65 unchanged engine overlays, network none, 8 GiB memory/equal swap limit,
four CPUs, unprivileged UID, and read-only source mounts. No model weights,
credentials or serving caches are mounted. Recheck the full trial's terminal
state, free development quartet, exact GPU UUID and unchanged serving peer
before starting either probe. Run them sequentially. No GPU verdict exists yet.

The full-model trial has now completed 188 requests, including both final
cold branches at 32588 input tokens. Fresh long references are still pending.
This does not establish its final lifecycle or numerical verdict.

### Native copy-version audit prepared; not GPU-qualified (2026-09-22)

While `6713c2ac7027` continues unchanged, the next diagnostic can opt in with
`--audit-native-copies`. This replaces the untraced observation flag; it is
explicitly timing-perturbing. The running snapshot does not contain this code.

`tools/flash_next/session_transfer_audit.py` wraps the existing worker-private
CPU offload copy function, without replacing descriptors, the copy backend,
scheduler, ownership or eviction policy. It resolves descriptors only into
known tensor rows; it never dereferences arbitrary addresses. It hashes source
bytes, invokes the native copy, and checks destination equality and unchanged
sources. Each later load must match the last observed store version for those
CPU fragments; slot reuse updates that version. This avoids comparing
independently recomputed hot/cold states as though they were the same version.
Only hashes and bounded metadata are retained, not tensor contents.

The named worker RPC installs after startup and removes the wrapper at
collection. Guards require TP4, private direct-layout CPU buffers and drained
handlers, reject overlapping/ambiguous descriptors, and cap readback at
2 GiB/copy, descriptors and retained versions at 65536, and records at 4096.
Both copy directions must be observed on every rank before the independent
copy verdict can pass; empty/missing-rank evidence fails. A requested audit
cannot be omitted from the combined diagnostic verdict.

Local session suite: 130 discovered, 127 pass and three skips (two prior
dependency skips plus the explicitly gated CUDA roundtrip). CPU tests cover
store-version replacement, modified CPU sources, altered destinations, tensor
bounds, non-vacuous four-rank verdicts and configuration guards. Targeted Ruff,
diff checks and source preservation pass. A tiny native CUDA roundtrip test
covers CPU chunks with two GPU pages, relocated destinations and both small
and large copy payloads, but has NOT run yet. Run it on the development quartet
only after the current full-model trial terminates; then decide whether the
full-model audited follow-up is ready.

Scope remains limited: synchronous copy auditing cannot prove race freedom,
scheduler hash-to-request ownership, untouched non-destination pages or
resident main-RAM-KV integrity. Those remain separate checks. No engine source,
serving configuration, weights, commit or push changed for this audit helper.

### Graph-mode C2 long-conversation lifecycle run (2026-09-22)

The live report has reached 84 completed requests: both seed requests at
29756 input tokens and all three hot turns for both branches have complete,
finite 128-token outputs. Hot reuse grows through 28320, 29264 and 30208
cached tokens at 30700, 31644 and 32588 input tokens respectively. Docker
still reports this exact container running/no OOM; cold and fresh-reference
continuations remain pending. This is progress evidence, not a final gate.

Container `6713c2ac7027`, root `/tmp/flash-session-gpu-20260921-oCIeuS`,
started at 17:51:54 UTC on development GPUs 0--3. Docker confirms it is running;
the serving peer remains unchanged at its original September 20 start time.
Launch/source record: `launch-full-tp4-graph-c2-continuations.json`.
Poll this exact container; an observation timeout is not a reason to restart it.

The diagnostic adds `--continuation-concurrency 2` to the existing bounded
continuation fixture. Two equal-length seeds share a real prefix and have
private tails. Each hot conversation appends its own actual answer; cold and
two fresh-reference namespaces replay those identical prompts. Both branches
use the same salt within a phase, and hot/cold/reference namespaces are isolated.
Three growing turns have 30700, 31644 and 32588 input tokens plus 128 outputs.
Native pause/admission starts each pair together; native statistics must
confirm two running requests. The existing cold-load barrier applies only to
the initial short-prefix test, not the new continuation phases.

Each turn has one shared 32-request GPU-eviction sequence. Phase load totals
are recorded once; overlapping per-request deltas must not be summed. Gates
check real cold loads, hot hits, increasing reuse, sampling changes, prompt
identity, concurrent execution and complete finite outputs. Lifecycle and
numerical verdicts are separate; the original combined verdict is retained.
This untraced fixture does not itself audit cache bytes or establish quality
equivalence. No runtime outcome is available yet.

TP4/PP1/Q8 MTP4, graph decode and all 19 optimization flags remain enabled.
The 32k configuration reuses the tested 1024-block resident host pool,
128 GiB host ceiling, 1 GiB/rank GPU KV budget, 8 GiB native CPU tier and
LRU/3600-second idle TTL. All 65 engine source hashes match the passing graph
page audit; only diagnostic code changed. There are no arithmetic-ordering
controls, model-weight changes, serving changes or production promotion.

Local validation: 127 session tests discovered, 125 pass and two dependency
skips; both source-preservation tests, targeted Ruff and diff checks pass.
New unit coverage rejects serial execution, missing phase load bytes,
non-finite/truncated output and invalid concurrency configuration, and proves
numerical differences do not automatically fail lifecycle checks.

### Full-model graph prefix-page integrity passes (2026-09-22)

Container `b54b98191a82`, root `/tmp/flash-session-gpu-20260921-Q5WGuF`,
ran from 17:27:56 to 17:40:13 UTC on development GPUs 0--3 and exited 1
without OOM. All 76 requests completed with finite outputs. The independent
state analyzer passes all eight checks: real C1/C2 cold restores, paired native
loads, complete traces, and exact observed checkpoint and working-state bytes
on all four ranks at the 5664-token prefix boundary. C1 cold traffic was
141430784 bytes. Target/draft RAM KV, compressed keys, target convolution
history and temporal state all match in the audited scope.

No arithmetic-ordering controls were enabled. Native target-step metrics
confirm actual FULL graph dispatch for 430 five-token and 131 ten-token steps.
The original combined screen remains false/exit 1: fresh repeats differ, and
hot/cold generated tokens first differ at index 59 for C1 and 29/34 for C2.
These numerical failures remain separate from the passing observed-state
gate, per the user's decision; quality equivalence is not established.

Evidence: `launch-full-tp4-graph-page-audit.json`,
`screen-full-tp4-graph-page-audit.json`,
`full-tp4-graph-page-barrier.json`, and `graph-page-state-analysis.json`.
Raw report SHA256:
`27a11f87cce54ced25378907091d1f7181a44b49180d3079709980c19f904d55`.
Re-running the analyzer reproduces all eight passing checks. Serving
`7d408c2bc15d` remains running with its original September 20 start time.

The engine sources, TP4/PP1/Q8 MTP4 topology, host/device/CPU cache budgets,
graph captures, all 19 optimized flags and native prefix/TTL behavior match
the completed untraced graph trial. The diagnostic adds the existing shape/page
observer for C1 and divergent C2 at the 5664-token reuse boundary. It records
both retained checkpoints and working states on all four ranks. The observer
runs during eager prefill after startup graph capture; graph decode remains
enabled. There are no arithmetic-ordering controls and no cancellation phase
in this audit. Its timings are explicitly perturbed and not performance data.

This qualifies only the observed C1/C2 prefix restoration state, not all
lifecycle transitions, longer concurrent histories or production operation.
Concurrent multi-turn/long-context reuse and broader race/failure coverage
remain open. The feature is still experimental; no production promotion,
commit or push occurred.

Interpretation note from source review: native GPU `BlockPool` can retain
duplicate blocks for one prefix hash and selects the first cached block;
`CPUOffloadingManager.prepare_store` skips a key already present in its tier.
Therefore a hot/cold mismatch is an observed state-equality failure, not by
itself proof that a transfer corrupted bytes: separately computed versions
must be distinguished from the actual exported/restored version. Equal
comparisons still prove only the observed state scope. No ownership or
replacement policy was changed for this interpretation.

### Full-model graph lifecycle screen completes (2026-09-22)

Container `bc52c32ebde9` exited 1/no OOM at 17:24:50 UTC after all
114 requests. All seven cancellation/reuse lifecycle checks pass: one native
abort at 16 tokens, its peer surviving to 128 tokens, both later reuse requests
complete with 5664 cached tokens, real cold loads, concurrency two and drained
frontend requests. Independent analysis verifies all 113 normal responses are
complete and finite, plus the intentional partial abort. All 113 available
native MTP summaries reconcile their histograms, accepted/proposed totals and
rates.

Real cold traffic: C1 141430784 bytes, initial C2 pair 282861568 bytes,
cancellation pair 141430784 bytes. Native target-step metrics record FULL graph
dispatch for 726 five-token and 125 ten-token steps, with zero padding in those
observations; prefill runs in NONE mode. This confirms actual full-model graph
use, not just capture configuration.

The original combined screen remains false/exit 1 because fresh repeats and
cached-output equality differ. The survivor/replay difference is again at
position 14. These are retained numerical failures, not asserted harmless.
No full-model cache-byte observation was made in this untraced run.
A separate graph-mode page audit is being prepared/launched against the
unchanged engine sources.

Evidence: `screen-full-tp4-graph-manual-budget.json`,
`full-tp4-graph-load-barrier.json`, `full-graph-lifecycle-analysis.json`
and launch/source record. The analysis retains raw-report hash, timings,
native MTP rates and graph table. Client timings are fixture measurements,
not an isolated prefill benchmark, matched engine comparison or production
qualification; concurrent graph windows overlap. Serving is unchanged.

### Next-run graph prefill byte audit prepared (2026-09-22)

The local diagnostic now also permits graph decode with the existing complete
prefix shape/page audit (`--trace-prefix-shapes --trace-prefix-pages`), without
arithmetic-ordering controls or kernel fallbacks. Untraced mode remains
mutually exclusive with these traces. The observer is installed after startup
capture; it reads state before eager prefill at the reuse boundary and does
not replace graph decode. Its existing guard still rejects unexpected capture
while a trace phase is active. Timing is explicitly labeled perturbed by audit.

This prepares direct full-model graph-mode cache evidence because fresh
uncached requests in `bc52c32ebde9` already show numerical variability.
It does not modify that running frozen snapshot or its tests. No new audit
container has been launched yet. Local suite: 125 discovered, 123 pass,
two dependency skips; both source preservation tests and targeted Ruff pass.
No engine source, serving configuration or weights changed for this
diagnostic-only extension.

### Full-model manual-budget graph screen running (2026-09-22)

Container `bc52c32ebde9`, root `/tmp/flash-session-gpu-20260921-mM9dz6`,
started at 17:11:27 UTC on GPUs 0--3 after the reduced graph test passed.
Launch/source record: `launch-full-tp4-graph-manual-budget.json`.
Poll this exact handle; no restart on observation timeouts. Serving remains
`7d408c2bc15d` at its original September 20 start time.

This is TP4/PP1/Q8 MTP4, 8k/C2, 512 resident-RAM blocks, 8 GiB native CPU tier,
LRU/3600-second TTL, 750 MB/rank explicit GPU KV budget, balanced 2048-token
prefill and all 19 optimization flags. The explicit graph opt-in now passes
the startup configuration guard. Requested decode captures are
`[1,2,3,4,5,6,8,10]`. There are no arithmetic ordering controls, tensor
traces or paused byte audits. The existing divergent-prefix hot/cold and
active-abort/reuse fixtures are unchanged.

The report records native target-step graph dispatch and per-request MTP
summaries plus client stream timings. These are measurements only once
requests run, not a performance or cache-integrity pass in advance.
Concurrent graph-observation windows overlap. No production promotion,
commit or push occurred.

### Reduced native graph/cold relocation passes (2026-09-22)

Container `4c6940d23a74` exited 0/no OOM at 17:08:20 UTC. All four ranks
completed all M1/2/4 cases. Each case captured five native FULL graphs (one
target layer and four MTP steps) and replayed each three times without
re-entering its Python forward. Real CPU loads per rank were 1644544,
3289088 and 6578176 bytes for M1/2/4. Restored state bytes, resident RAM
and non-destination pages pass exact checks; relocated replay writes remain
isolated to destination/scratch pages. Model parameters and storage are
unchanged.

All recorded eager-versus-graph, repeated-graph and graph hot/cold outputs
are exact (maximum absolute difference zero), as are final persistent states.
These numerical results are separate from the integrity gate and apply only
to this bounded fixture. This is not full-model scheduling, long-context,
target-QSA, graph-padding, batched-verification or acceptance-rate qualification.

Evidence: `reduced-tp4-graph-rank0.json` through `rank3.json`,
`reduced-graph-replay-analysis.json`, and the launch/source record. Independent
post-run checks required all ranks/cases, actual graph/replay counts, real
positive transfers, unchanged parameters and every integrity field.
Full-model graph-mode qualification is the next gate. Serving is unchanged.

### Reduced native graph/cold-relocation screen started (2026-09-22)

Container `4c6940d23a74`, root `/tmp/flash-reduced-mtp-20260922-VZlxj1`,
is running on the authorized development quartet. Launch/source record:
`launch-reduced-tp4-graph-replay.json`. TP4/PP1, one real target GDN
layer plus Q8 MTP4, all 19 allowlisted optimized kernel flags enabled.
Serving `7d408c2bc15d` remains unchanged.

The test reuses native `CUDAGraphWrapper`, distributed capture contexts,
independent host/device allocation and native CPU transfers. At M1/2/4 it
captures five forwards, then replays each three times (hot, hot repeat,
cold with relocated GPU page IDs). Captured metadata tensor addresses stay
fixed while block-table contents change. Gates require exact restored bytes,
real nonzero transfers, unchanged resident RAM/non-destination pages, finite
outputs, and replay without re-entering the Python forward. Numerical output
and final arithmetic-state comparisons are recorded separately.

This is not full-model startup, scheduler, batched target verification, target
QSA or padding qualification. Local harness tests: 34 pass; syntax/error lint
passes. Existing harness files retain broader style findings. The new helper's
two long description strings were wrapped locally after this run froze;
no behavior changed and the frozen source hash remains in the launch record.
No GPU result or performance claim is available yet; poll this exact handle
without restarting on observation timeouts.

### Manual-budget graph gate: CPU validation only (2026-09-22)

The local engine now accepts optional boolean `allow_cudagraph` under
`flash_next_direct_host_kv`, default false. Non-eager execution additionally
requires a positive integer `kv_cache_memory_bytes`, compilation mode
`NONE`, and `FULL_DECODE_ONLY`; existing DP1/hybrid/budget guards remain.
Automatic profiling and the GPU-only host-group allocator remain rejected.
The diagnostic graph configuration supplies this explicit opt-in and rejects
missing/invalid GPU budgets before engine startup.

Pinned-image CPU container `91fc26e08732` exited 0/no OOM: 75 native packing
tests passed and one GPU-dependent test skipped. Tests include identical
independent-pool plans for eager/opted-in graphs and rejection of unsupported
budgets/compile modes. Evidence: `native-graph-config-cpu.json`.
Local diagnostics: 124 discovered, 122 pass, two dependency skips; both source
preservation tests, targeted Ruff and diff checks pass. Native pytest was
unavailable in the local environment, so it ran in the existing pinned image,
CPU-only with no network, model weights or production cache mounts.

This is configuration/planner validation, **not GPU capture/replay validation**.
No new GPU diagnostic was launched, and no serving configuration was changed.
Next: reduced TP4 target layer(s) plus MTP, real capture/replay and native cold
restoration, then a full-model graph screen only if the reduced check passes.
The source manifest records the sole engine-file revision; the reduced launcher
retains the old snapshot pin and overlays the reviewed local revision.
No commit, push or production promotion occurred.

### CUDA-graph diagnostic rejected before requests (2026-09-22)

Container `0c90012fa943` exited 1/no OOM at 16:47:33 UTC, before any
requests. All workers rejected configuration with
`Experimental direct host KV requires enforce_eager`. The saved raw report is
`screen-full-tp4-graph-rejected.json`; this is an initialization rejection,
not a cache-restoration failure or performance measurement. The launch entry
below is historical. Serving `7d408c2bc15d` remains running at its original
September 20 start time; no serving configuration or model weights changed.

The guard protects an unsupported temporary CUDA-graph profiling cache: its
allocator is GPU-only. Do not remove that protection unconditionally.
Inspection identifies a narrower experimental route: an explicit positive
`kv_cache_memory_bytes` skips the temporary memory profiler, while real
capture follows allocation of the independent host/device pools. This route
still needs configuration tests and reduced TP4/target-plus-MTP capture/replay
validation before another full-model graph trial. Eager-mode integrity
evidence is unaffected; numerical variability remains a separate assessment.

### CUDA-graph decode hot/cold and cancellation screen started (2026-09-22)

Container `0c90012fa943`, root `/tmp/flash-session-gpu-20260921-Px9gMf`,
is confirmed running on development GPUs 0--3. Launch/source record:
`launch-full-tp4-graph-active-cancel.json`. Poll this exact handle; do not
restart on an observation timeout. Serving `7d408c2bc15d` remains running
at its original September 20 start time.

The diagnostic retains the TP4/PP1/Q8 MTP4 native prefix/CPU-offload contract,
512 resident-RAM blocks, 8 GiB CPU tier, LRU/3600-second TTL, 750 MB/rank GPU
cache, balanced 2048-token prefill budget and 8k/C2 fixture. It requests
`FULL_DECODE_ONLY` CUDA graphs with capture sizes
`[1,2,3,4,5,6,8,10]`, bounded for C2/MTP4. All 19 recorded optimization
flags are enabled; expert/QSA ordering controls, native-kernel fallbacks,
tensor tracing and paused byte audits are rejected in this mode.
The ordinary hot/cold and unpaused cancellation fixtures are unchanged.

Each request now records client stream timing and vLLM's native per-request
MTP summary when available. This first graph-enabled run does not enable
per-step graph-replay metrics: capture logs/configuration must not be described
as proof of actual replay frequency. No latency/acceptance result is available
yet. This is not a full production-context/concurrency benchmark or cache-byte
audit. Local suite: 123 tests, 121 pass and two dependency skips; two source
preservation tests and Ruff/diff checks pass. No production engine source,
serving configuration, route or model weight changed.

For subsequent snapshots, graph-enabled diagnostics now also set native
`cudagraph_metrics=True` and reuse `CUDAGraphLogging`. Reports retain its
native dispatch table and grouped native `CUDAGraphStat` observations during
each request. These are engine target-step observations, not per-rank or
draft-head profiling; concurrent request windows overlap and must not be
summed as independent work. This is a local reporting addition only, absent
from frozen run `0c90012fa943`. The same 123-test suite still has 121 passes
and two dependency skips; no production engine source changed.

### Paused cancellation integrity audit passes (2026-09-22)

Container `19166ad10cb3` completed 114 requests and exited 1/no OOM at
16:32:13 UTC. All nine lifecycle/integrity checks pass. Before and after native
peer abort, the survivor has identical cache bytes and mappings on all four
ranks: 732 tensor-page views and 321079488 hashed bytes per rank, covering all
41 groups and expected target/MTP layer parts. Native scheduler ownership
preserves all 210 survivor block-table entries and their hashes. Summed
references decrease from 228 to 210 as the peer releases shared ownership;
all surviving entries remain positively referenced.

The pause was triggered at observed victim/survivor lengths 16/11. In-flight
work drained before native abort, so the final aborted response contains
26 tokens. The survivor and both post-abort reuse requests complete 128 tokens.
Both later requests reuse 5664 tokens; the cold pair loads 141430784 bytes.
Independent post-run reanalysis verifies complete finite evidence for all
113 normal responses plus the intentional partial abort, and rechecks both
byte and lifecycle assessments against the saved raw report.

The survivor/replay numerical difference remains at position 14, exactly as
in the untraced case; it is a separate result, not waived or used as evidence
of corruption. The historical combined screen remains false/exit 1 because
it still includes numerical equality. Production engine hashes and kernel
flags match the untraced run. This paused, timing-perturbed audit complements
that run; it does not prove unpaused race freedom, physical DMA failure
recovery, longer-context concurrent reuse or production performance.

Artifacts: `screen-full-tp4-active-cancel-audit.json`,
`full-tp4-active-cancel-ownership.json`,
`full-tp4-active-cancel-audit-barrier.json`,
`active-cancel-audit-analysis.json` and the launch/source record.
The analysis includes raw-report/analyzer hashes. No diagnostic GPU process
remains from this run. Serving `7d408c2bc15d` remains running at its original
September 20 start time. No production promotion, commit or push occurred.
The implementation goal remains open for the outstanding qualification and
same-PR preservation/review work.

### Paused active-cancellation byte/ownership audit started (2026-09-22)

Container `19166ad10cb3`, root `/tmp/flash-session-gpu-20260921-QXFqvM`,
is confirmed running on development GPUs 0--3. Launch/source record:
`launch-full-tp4-active-cancel-audit.json`. Poll this exact handle; do not
restart because an observation times out. Serving `7d408c2bc15d` remains
running at its original September 20 start time.

The new diagnostic reuses the full TP4/MTP4 active-abort fixture. Once both
cold requests emit tokens, native `pause_generation(mode="keep",
clear_cache=False)` freezes scheduling. Named worker RPCs hash every allocated
survivor cache page before and after native peer abort, then native scheduling
resumes. The comparator requires four ranks, all 41 cache groups, all expected
layer/tensor parts, nonempty page hashes, and identical bytes and mappings.
A read-only wrapper around native scheduler `finish_requests` checks that the
survivor retains its pool objects, hashes and positive references. A shared
reference count may decrease when the peer releases its ownership.
This is deliberately timing-perturbed and does not prove unpaused race freedom
or physical DMA fault recovery; the preceding untraced lifecycle result remains
separate evidence.

Local diagnostic suite: 119 tests, 117 pass and two dependency skips. Both
production-source preservation tests, Ruff and diff checks pass. No production
engine source changed. The frozen run's inherited scope strings still say
untraced/before-forward; the explicit `audit_active_cancel`, worker snapshots
and scheduler audit identify the added paused observation. Those two display
labels have been corrected locally for subsequent snapshots only, without
editing or restarting the live frozen run. No GPU audit result is available
yet, and there is no promotion, commit or push.

The local next-run diagnostic now enables vLLM's native
`per_request_spec_decode_metrics="summary"` and stores the final native
`RequestSpecDecodeMetrics.to_dict()` payload; it does not invent acceptance
counters. Read-only inspection verified both the option and completion-output
field in the pinned running image. Client stream timing records the first
nonempty output, last output and the size of the initial token burst.
Post-first-output throughput excludes that entire burst, not an assumed single
token; a single-output response has no inferred decode rate. Nonfinite or
nonmonotonic timings fail validation. This reports client-observed timing,
not isolated prefill time or qualified serving performance. These local changes
are not present in frozen run `19166ad10cb3`.
The current local suite has 121 tests: 119 pass and two dependency skips;
both preservation tests pass. A further local diagnostic guard delegates
multi-request/shutdown aborts directly to native cleanup, so only the intended
single-victim abort invokes the ownership observer. Its regression test passes.
This guard is also not part of the frozen live run. No production engine source
changed.

### Full-model active cancellation: lifecycle gate passes (2026-09-22)

Container `eff5c31567f9` completed all 114 requests and exited 1/no OOM
at 16:02:07 UTC. There are 113 complete finite responses and one intentionally
partial response: native `AsyncLLM.abort` stopped the victim at 16 tokens,
while its shared-prefix peer had emitted 11 tokens and remained active.
The peer completed all 128 requested tokens. Both cold requests reused 5664
tokens, the pair loaded 141430784 bytes, and maximum running concurrency was
two. Both subsequent ordinary requests reused 5664 tokens and completed
128 tokens. Frontend requests drained; all seven lifecycle checks pass.

The survivor versus subsequent C1 replay differs at output position 14,
including log-probabilities. This is recorded separately under the user's
decision; this untraced case does not establish whether its state bytes match.
Initial C1/C2 hot/cold comparisons remain exact, while C2 fresh-versus-cached
differences remain at positions 29/34. The original combined screen and exit 1
are preserved, not relabeled as a wholly passing numerical test.

Artifacts: `screen-full-tp4-active-cancel.json`,
`full-tp4-active-cancel-barrier.json`, `active-cancel-analysis.json`,
and the matching launch/source record. Independent post-run validation found
complete finite output evidence for all normal responses and a finite,
nonempty, native-aborted partial response for the sole victim.
This adds full-model active-cancellation lifecycle evidence, not a tensor-byte
or backend-refcount audit, physical transfer-failure recovery qualification,
or production-performance result. The frozen production engine sources and
kernel flags match the previous run. Serving `7d408c2bc15d` is still running
at its original September 20 start time. No test process remains from this run;
no serving change, promotion, commit or push occurred.

### Full-model active-cancellation diagnostic started (2026-09-22)

Container `eff5c31567f9`, root `/tmp/flash-session-gpu-20260921-Ky8xsC`,
is running on development GPUs 0--3. Launch/source evidence is
`launch-full-tp4-active-cancel.json`. This adds an untraced TP4/PP1/Q8-MTP4
lifecycle fixture after the existing 76-request C1/C2 control: warm two
shared-prefix/private-suffix requests, pressure-evict their GPU state, admit
both cold requests through native vLLM, then invoke `AsyncLLM.abort` only
after both emit at least eight tokens. The survivor must finish all 128 tokens;
the intentionally aborted response must have a finite, nonempty partial output
and native `abort` finish reason. Both prompts must remain reusable, with real
cold-load bytes and C2 execution observed. Numerical comparisons are reported
separately, not used to waive lifecycle failures. Frontend cleanup is checked;
this is not a tensor-byte audit or physical in-flight DMA failure test.

The local diagnostic suite has 116 tests: 114 pass, two dependency skips.
Both preservation tests, Ruff and diff checks pass; all 65 production engine
sources remain unchanged. Only the isolated diagnostic adds the new case.
Poll the exact live container; do not restart it on an observation timeout.
Serving `7d408c2bc15d` remains unchanged. No production qualification or promotion.

### 1024-block 32k continuations pass real cold reuse (2026-09-22)

Container `ae0df639192b` completed all 186 requests and exited 1/no OOM
at 15:31:58 UTC. All three continuation gates pass:

| Input tokens | Cached tokens | Native cold-load bytes |
| --- | --- | --- |
| 30700 | 28320 | 220368896 |
| 31644 | 29264 | 223657984 |
| 32588 | 30208 | 226947072 |

Each hot/cold/two-fresh comparison has identical 128 output tokens and recorded
log-probabilities. Seed equality, increasing reuse, separate prefix namespaces,
changed repetition penalties and complete finite output evidence pass.
The initial C1 and both C2 hot/cold comparisons also match exactly.
The frozen combined screen still reports failure because C2 fresh-versus-cached
text differs at positions 29/34. That numerical result is retained separately
under the user's revised criterion; it is not a cache-state failure by itself.

Compared with the retained 512-block run, all 186 request names, prompt hashes
and output token sequences are identical. All production engine source hashes
and kernel flags match. Only the diagnostic source differs (the larger host
pool and complete-output check). The former long misses become real restores.
This supports the capacity explanation without changing model arithmetic.

Artifacts: `screen-full-tp4-32k-host1024-continuations.json`,
`full-tp4-32k-host1024-continuations-barrier.json`,
`32k-host1024-continuations-analysis.json`, and the matching launch record.
This is an untraced C1 continuation result up to 32588 input tokens, not an
all-rank long-context byte audit, simultaneous long C2, 240k qualification,
or a production-performance result. Full lifecycle/capacity/serving validation
and same-PR preservation/review remain open. No test container is still running
from this pair. Serving `7d408c2bc15d` remains at its original September 20
start time; no promotion or service change occurred.

### User decision: separate cache integrity from numerical variability (2026-09-22)

The user explicitly selected: prioritize cache-state integrity and evaluate
engine numerical variability separately. This resolves the earlier pending
question. Identical text across different batching/cache paths is no longer
a release requirement by itself. Corrupt or incomplete restored state,
incorrect ownership, unsafe reuse, missing real transfers, expiry/cancellation
failures, or non-finite/incomplete output evidence still fail qualification.
Original token/logprob comparisons and their combined verdicts are retained;
the decision does not relabel them as exact or prove optimized/native accuracy.

The existing raw-trace analyzer now emits a separate, bounded state assessment:
TP4/MTP4, C1/C2, 5664-token cached boundaries, all four ranks, actual native
cold loads and checkpoint/working-state coverage. It requires the expected
target/draft RAM-KV, compressed-index, convolution-history and recurrent-state
counts; missing ranks, partial categories, unequal bytes and missing transfers
fail. Untraced matching text cannot satisfy this byte-integrity assessment.
Thirty-three local analyzer/harness tests and Ruff/diff checks pass.

Reanalysis of the saved fully optimized trace and the prepared-load-expiry
trace passes this observed-state gate. Artifacts:
`optimized-state-integrity-assessment.json` and
`pending-expiry-state-integrity-assessment.json`, with raw-report and analyzer
hashes. The optimized run's fresh/hot/cold outputs still vary, explicitly
reported in the separate numerical section. These results do not qualify
unobserved lifecycle cases, long contexts, performance, or production.

The current 1024-block run `ae0df639192b` remains live/no OOM. Its first
30700-token long cold request now genuinely reloads 220368896 bytes and
reuses 28320 tokens, unlike the 512-block run's miss. Remaining long turns
and fresh comparisons are still in progress. No serving change or promotion.

### 32k run completed: exact recomputation, no long cold hits (2026-09-22)

Container `4dbf55444dd5` exited 1/no OOM at 14:59:48 UTC after all 186
requests. For all three long turns (30700/31644/32588 input tokens), hot,
pressure-following, and two cache-bypassed fresh outputs match exactly:
128 tokens and their recorded log-probabilities. Seed equality and changed
request-local repetition penalties also pass. All 186 responses pass the
post-hoc output-completeness gate.

However, every long pressure-following request has zero cached tokens and
zero load bytes. Matched reuse, actual cold loading and growing reuse fail;
the original continuation and overall verdicts remain false. This is evidence
for the tested recompute fallback, not long-context restore qualification.
The 7100-token C1 hot/cold control still reuses 5664 tokens and restores
141430784 bytes exactly. The original C2 fresh-versus-cached differences
remain at output positions 29/34; repeated fresh C2 is exact.

Artifacts: `screen-full-tp4-32k-continuations.json`,
`full-tp4-32k-continuations-barrier.json`, and
`32k-continuations-analysis.json`, plus the original launch/source record.
The retained 512-block run is not overwritten. A follow-up is launching the
prepared 1024-block RAM pool on development GPUs 0--3 with the same prompts,
pressure, arithmetic controls and other budgets. It also includes the new
fail-closed output-completeness check; no production engine source changed.
Launch record: `launch-full-tp4-32k-host1024-continuations.json`.
Container `ae0df639192b` started at 15:02:30 UTC, root
`/tmp/flash-session-gpu-20260921-raRegh`. Initial inspection confirms
running/no OOM and the requested 1024 host blocks, 128 GiB ceiling and
1 GiB/rank GPU cache. Poll this exact handle; do not restart on observation
timeout. Serving remains running at its original September 20 start time.
The feature is still experimental and unqualified.

### Long-context pressure exposes the RAM retention limit (2026-09-22)

The running 32k fixture's first long cold request completed at 30700 input
tokens with zero cached tokens and zero load bytes. That is a recomputation,
not a qualified cold restore. The original 512-block RAM pool is shared by
target and MTP groups. The earlier all-rank trace identifies two host groups,
39 and 40, both with 944-token blocks.

A native CPU allocator reproduction now covers the capacity boundary. Each
request has two host groups, complete cached blocks and a private unhashed
tail. After 32 distinct seven-complete-block pressure requests, a 512-block
pool retains the short seed (five complete blocks per group), but loses part
of long seeds (31 or 33 blocks per group). A 1024-block pool retains the
33-block seed. All four cases pass, native references drain, and no TTL
expiry occurs. This supports capacity eviction as the explanation; it does
not by itself prove every live full-model lookup decision.

Evidence: `native-host-pressure-cpu.json/xml`. Container
`4cd460d06353` exited 0/no OOM at 14:55:30 UTC: 4 passed, 33 deselected,
15 warnings (14 existing Torch deprecations and the isolated unregistered CPU
test mark). The recorded 65 engine source hashes were checked before launch.
No GPU devices, model weights, or serving cache mounts were supplied.

The next 32768-token diagnostic is prepared with 1024 host blocks, retaining
the existing 128 GiB host-byte ceiling, 1 GiB/rank GPU budget, 8 GiB native CPU
offload budget, LRU, and 3600-second TTL. The 8192-token fixture stays at 512
host blocks. The allocator's byte-budget guard still applies. This changes
only a diagnostic configuration, not engine arithmetic or serving settings.
Local validation remains 111 passing session tests and 2 dependency skips;
Ruff/diff checks pass. Container `4dbf55444dd5` is still running its frozen
512-block configuration; it is not restarted or relabeled as a passing run.

### Complete-output gate and archived evidence audit (2026-09-22)

The ordinary-request diagnostic now rejects shortened output even when two
paths return the same tokens. Each request must contain its requested token
count, one log-probability map per token, a finite entry for the selected token,
and finite values for every other recorded candidate. Failure is saved before
the diagnostic raises. This closes a validation gap; it is not an inference
engine or cache-transfer change.

The new helper was also applied offline to all 527 requests in the two completed
8k continuation screens, prepared-load expiry screen, and idle-expiry screen.
All have the expected 128 tokens (8 for pressure requests) and complete finite
probability evidence. Their original failed combined verdicts remain failed;
this audit neither clears C2 fresh-versus-cached differences nor establishes
optimized/native accuracy equivalence. Report and validator hashes are in
`output-completeness-audit.json` in the benchmark evidence folder.

Local validation: 113 session tests discovered, 111 pass and 2 dependency skips;
Ruff and diff checks pass. The running 32k container `4dbf55444dd5` retains
its frozen earlier diagnostic source; it will need the same offline audit.
Latest inspection: running/no OOM, 80 completed requests through `turn-hot-3`.
All three long hot continuations completed at 30700/31644/32588 input tokens,
reusing 28320/29264/30208 cached tokens respectively, with zero cold-load bytes
and 128 output tokens/logprob positions each. Cold and fresh long-context
comparisons remain pending. The initial C2 combined gate remains failed.
Serving container `7d408c2bc15d` remains running with
its original 2026-09-20T19:45:53 start time. No restart or promotion.

### Bounded32k continuation fixture prepared (2026-09-22)

The existing full-model continuation diagnostic now accepts
`--continuation-context 32768` only with `--continuations`; default8192
behavior is unchanged. The32k seed has29756 tokens (same492-token remainder
modulo944); successive turns reach30700,31644 and32588 input tokens. Each
retains the preceding actual128 generated tokens, changes native repetition
penalties as before, and must match hot/cold/two fresh paths exactly.

The prior750,000,000-byte GPU-cache budget reported32,203-token aggregate
capacity, insufficient to declare a32768-token model limit. This isolated
fixture explicitly uses1 GiB GPU cache/rank; main resident-host blocks512,
host cap128 GiB, CPU offload8 GiB, native LRU and3600-second TTL are unchanged.
The model limit remains bounded32768; initial C1/C2 requests remain7100 tokens
and pressure prompts remain7100. Only continuation inputs grow. The test's
wall-time bound is3600 seconds instead of1800; container/GPU/network boundaries
are unchanged. This is not a240k capacity qualification or a serving change.

112 local session tests discovered:110 pass,2 dependency skips; Ruff/diff
checks pass. Added checks cover aligned geometry, context/lookahead limits,
source-config immutability, unchanged RAM budgets, required continuation mode,
and full driver generation at the new lengths. GPU validation is being
launched with the18 non-layout optimization flags and canonical expert/QSA
controls. Container `4dbf55444dd5` started at14:30:43 UTC, root
`/tmp/flash-session-gpu-20260921-icccyy`, on development GPUs0--3.
Launch/source record: `launch-full-tp4-32k-continuations.json`.
Initial inspection confirms running/no OOM; poll this exact handle.
Serving remains at its original2026-09-20T19:45:53 start time.

### Paired reduced expert-order control passes (2026-09-22)

Container `4747fa80e7fd` completed at14:22:38 UTC, exit0/no OOM. All four
ranks pass parameter integrity, the existing short/MTP rejection checks, and
every three-turn check for target0 and target0+3 with Q8 MTP4. Fresh/hot/cold
hidden chunks and final outputs are exact; retained prefix bytes remain
unchanged; actual cold transfers and native cleanup pass.

The18 optimization flags and engine source hashes match the failed QSA-only
run. The identical coordinator overlay has a different temporary source path,
not different code. Adding canonical expert ordering is the intended treatment;
the hook executed310 times/rank for target0 and620 for target0+3.
This paired result implicates native expert layout ordering in the reduced
repeatability failure. It does not prove full-model batch invariance, eliminate
cross-kernel numerical differences, or qualify production output accuracy.

Evidence: `reduced-tp4-optimized-canonical-continuations-rank0.json` through
rank3, `reduced-optimized-canonical-analysis.json`, and the matching launch
record. Both development diagnostics are terminal. Serving unchanged.
The question about production bit-identical output across batch/cache paths
was pending at this point. It is now resolved by the September 22 decision
recorded above: prioritize cache-state integrity and track numerical variability
separately. The original strict comparison results remain recorded.

### Reduced optimized kernels with QSA-only control fail repeatability (2026-09-22)

Container `9604a8e5f762` finished14:14:39 UTC, exit1/no OOM. All four ranks
pass parameter integrity. The completed one- and two-target-layer reports fail
exact output gates even for repeated fresh requests with matched schedules.
With target0, target hidden output matches but draft hidden/logits differ
(maximum absolute differences0.03125/0.125 in the short reference pair).
With target0+3, target/draft/logits differ (0.0009765625/0.0546875/0.2578125).
These are tensor differences, not model-quality scores.

Short-prefix checkpoint bytes match hot/cold. Every appended turn has genuine
hot/cold reuse, namespace isolation and native cleanup; retained prefix bytes
remain unchanged during each forward. However all three turns fail fresh
repeatability and exact final outputs. Their separately computed hot/cold
prefixes differ after earlier computations have already diverged. Thus QSA
order/tie control alone is insufficient for this reduced optimized fixture;
this does not identify a transfer corruption or yet isolate an expert kernel.

Evidence: `reduced-tp4-optimized-qsa-only-continuations-rank0.json` through
rank3, `reduced-optimized-qsa-only-analysis.json`, and the matching launch
record. The paired follow-up keeps the same18 non-layout optimization flags
and QSA controls, adding only canonical expert ordering. It uses the existing
reduced launcher and no production engine changes. Container `4747fa80e7fd`
started at14:16:40 UTC, root `/tmp/flash-reduced-mtp-20260922-Ot8d3b`.
Launch/source record:
`launch-reduced-tp4-optimized-canonical-continuations.json`.
The completed paired result is recorded above.
Serving unchanged.

### Mixed optimized/canonical continuations completed (2026-09-22)

Container `887ad1a1b647` completed186 requests, exit1/no OOM at14:06:42 UTC.
All three continuation gates pass: identical128-token outputs and recorded
logprobs across hot, cold and two fresh references, including changed
request-local repetition penalties. Cached boundaries3776/4720/5664 and
cold-load bytes134852608/138141696/141430784 match the native-kernel fixture.
C1 fresh/hot/cold and both C2 hot/cold comparisons are exact; repeated fresh
C2 also matches. Fresh versus cached C2 differs at29/34, so the original
combined verdict remains failed.

This retains18 optimization flags, with native expert layout and canonical
expert/QSA ordering/ties. It demonstrates that this configuration did not
reproduce the earlier within-run nondeterminism; it does not prove every
enabled optimized branch was exercised or clear all optimized implementations.

Cross-configuration comparison is NOT exact: identical C1 reference prompts
first differ at token6 (first recorded logprob difference at1; common-history
top-k-intersection maximum delta1.874999). The continuation seed first differs
at125 (logprob difference at1; maximum9.689965). These are not full-distribution
or quality metrics and are not dismissed as harmless rounding. Later
continuation prompts differ across configurations because they append actual
generated answers, so their outputs are not compared across configurations.
Within each configuration, the cold and fresh paths receive identical history.

Evidence: `screen-full-tp4-optimized-canonical-continuations.json`,
`full-tp4-optimized-canonical-continuations-barrier.json`,
`optimized-canonical-continuations-analysis.json` and the matching launch
record. No production promotion or serving changes.

### Reduced per-kernel isolation prepared (2026-09-22)

The existing one/two-target-layer TP4/Q8-MTP4 launcher now accepts repeated
`--enable-kernel VLLM_FLASH_FLAG` arguments. It uses the full launcher's
existing19-name optimization allowlist, keeps unselected kernels disabled,
and records every flag in launch evidence. Unknown names (including RAM-KV
or topology controls) fail locally before remote inspection. Canonical expert
ordering still refuses the optimized expert-layout flag. No serving or
model-weight settings change.

This prepares fast kernel-family isolation if the full mixed diagnostic fails;
it does not establish that any enabled branch executes for a given fixture.
That requires the weighted run and operator/shape evidence. Default reduced
behavior remains all-native. Existing reduced test suite:32 passed; Ruff and
diff checks pass. No reduced GPU run was launched while the full mixed run
occupied GPUs0--3. Reduced container `9604a8e5f762` started at14:08:50 UTC,
root `/tmp/flash-reduced-mtp-20260922-BNR0Nt`. It tests target0 and target0+3
with Q8 MTP4, three growing turns, fixed QSA ordering/ties, native expert
ordering, and the18 non-layout optimization flags enabled. No expert-order
hook is installed. This isolates whether that hook is necessary for reduced
repeatability; it does not compare optimized outputs to native arithmetic.
Launch/source evidence:
`launch-reduced-tp4-optimized-qsa-only-continuations.json`.
Initial inspection confirmed running/no OOM; this QSA-only run subsequently
failed repeatability, as recorded above.

### Mixed optimized/canonical continuation diagnostic prepared (2026-09-22)

The first mixed launch was rejected locally before remote work because the
existing canonical expert-order hook requires native expert layout. That guard
is intentional: it wraps native grouping and cannot validate the optimized
layout kernel. The launcher now provides an explicit `--optimized-canonical`
mode. It requires native-prefix plus all three expert/QSA controls, rejects the
all-native override, disables only VLLM_FLASH_MOE_LAYOUT_SM86, and preserves
every other inherited kernel/quantization/topology/RAM setting.

The next test repeats the untraced186-request continuation fixture with that
mode. Relative to the passing continuation control,18 inherited optimization
flags are retained rather than disabled; expert layout stays native and
canonical ordering/tie selection stays fixed. This isolates the optimized
arithmetic family from those ordering controls, not any single kernel, and is
not a performance or fully unmodified serving qualification.

111 local session tests discovered:109 pass,2 dependency skips; Ruff/diff
checks pass. The environment regression requires all non-layout flags and
source inputs to remain unchanged and rejects conflicting overrides.
No production engine implementation was changed. Container `887ad1a1b647`
started at13:40:23 UTC, root `/tmp/flash-session-gpu-20260921-aeSWet`, on
development GPUs0--3. Frozen evidence:
`launch-full-tp4-optimized-canonical-continuations.json`. All18 non-layout
optimization flags are confirmed1; layout is0. Initial runtime inspection
confirmed running/no OOM. The completed result and limitations are above.

### Full-model growing continuations pass under controlled arithmetic (2026-09-22)

Container `18f8c44b6d4d` completed186 requests and exited at13:35:28 UTC,
exit1/no OOM. The added continuation gate passes every check for all three
turns. Seed requests are fresh and identical; native cache salts isolate warm,
cold and the two cache-bypassed reference namespaces. Each new prompt retains
the previous warm answer's actual128 tokens. Repetition penalties change from
1.0 to1.05 to1.1 and back to1.0 through new native requests.

| Turn | Input tokens | Reused tokens, hot and cold | Actual cold load bytes | Hot/cold/both fresh outputs |
| --- | ---: | ---: | ---: | --- |
| 1 | 6156 | 3776 | 134852608 | All128 tokens and recorded logprobs exact |
| 2 | 7100 | 4720 | 138141696 | All128 tokens and recorded logprobs exact |
| 3 | 8044 | 5664 | 141430784 | All128 tokens and recorded logprobs exact |

Each cold turn follows32 pressure requests; warm turns have no cold loads,
and both references have zero cached tokens/load bytes. This is untraced
TP4/Q8-MTP4 with native-kernel fallbacks and canonical expert/QSA diagnostics.
It qualifies this bounded C1 multi-turn fixture, not production kernels,
240k contexts, concurrent continuations or arbitrary numerical equivalence.

Original C1 and C2 hot/cold checks also pass exactly. Original fresh C2 repeats
match, but fresh-versus-cached C2 still differs at output29/14, so the combined
screen deliberately remains failed. No gate was waived or serving changed.
Artifacts: `screen-full-tp4-continuations.json`,
`full-tp4-continuations-barrier.json`, and
`launch-full-tp4-continuations.json`.

### Current native CPU regression (2026-09-22)

CPU-only container `dc5cbb64f0c3` ran13:14:36--13:15:25 UTC, exit0/no OOM:
204 passed,1 skipped,15 warnings. It uses the same65 frozen engine source
overlays as the active continuation launch, with current manager, factory,
contiguous packing and single-type cache-manager tests. Nine copied test/package
files have retained SHA256 hashes. No GPU devices, model weights or serving
cache were mounted; network none, unprivileged UID1000,2 CPUs,4 GiB limit.

The sole skip is the CUDA variant of
`test_native_cold_capacity_counts_packed_gpu_blocks_not_aliased_layers`:
"Native DMA needs CUDA". Its CPU variant runs; this result makes no DMA or
weighted-output claim. Warnings are14 existing Torch deprecations and the
isolated harness's unregistered cpu_test mark. An initial container
`43390d443056` exited2 before collection because uv's image-default cache
directory was not writable. The successful run explicitly uses an isolated
temporary UV_CACHE_DIR; no tests or assertions were disabled.

Evidence: `native-regression-current-cpu.json` (exact command, source/test
hashes, isolation, terminal state and log), plus
`native-regression-current-cpu.xml`. The full-model continuation container
`18f8c44b6d4d` subsequently completed; see its result above. Serving unchanged.

### Full-model prepared-load expiry completed (2026-09-22)

Corrected container `6b42b02672f9` completed76 requests and exited at
13:07:32 UTC, exit1/no OOM. The additional pending-expiry gate passes every
check. Both cold requests had acquired their native state; one load remained
pending when the private cache clocks advanced3601 seconds. Native eviction
removed497 idle host pages and2568 CPU chunks, while12 referenced host blocks,
43 CPU source chunks and204 destination blocks retained their ownership.
Clocks were restored. Both native load acknowledgements preceded promotion;
the pair barrier recorded12 deferrals.

C1 and both C2 hot/cold outputs match all128 tokens and recorded logprobs.
Actual cold loads:141,430,784 bytes C1 and282,861,568 bytes C2. On every rank,
checkpoint and working-state traces match for37 convolution histories,
36 temporal checkpoints,78 RAM-KV pages (72 target/6 draft), and78 compressed-key
pages. All observed subsequent own/packed query sequences match between each
hot/cold pair. This is traced, controlled arithmetic, not production timing,
physical-DMA failure injection, or complete race-freedom evidence.

The original combined verdict remains false: fresh C2 repeats are exact, but
fresh versus cached output first differs at29/14, as before. Neither this gate
nor the expiry success waives that unresolved failure. The first-load expiry
plus an unacquired follower's recomputation remains separately unqualified.

Retained evidence: `screen-full-tp4-pending-expiry-paired.json`,
`full-tp4-pending-expiry-paired-event.json`,
`full-tp4-pending-expiry-paired-barrier.json`, and
`pending-expiry-paired-analysis.json`. Source/launch:
`launch-full-tp4-pending-expiry-paired.json`. Serving unchanged.

### Prepared-load expiry: diagnostic barrier failure and correction (2026-09-22)

Container `cd6b0994bb4a` exited at12:44:41 UTC, exit1/no OOM, after75
completed requests. Root cause was the diagnostic PairLoadGate timeout:
"Diagnostic cold pair did not both finish native loads". The expiry event ran
with only one prepared load. It expired497 idle host blocks and2568 CPU chunks
while preserving12 owned host blocks,43 pinned CPU chunks and204 destination
blocks. Both private clocks were restored.

The completed `batch-cold-1` request reports zero cached tokens. The barrier
nevertheless required both requests to acknowledge cold loads. Thus the test
combined expiry-before-follower-acquisition with an incompatible two-load
barrier. This is not a passing full-model expiry test or a cache-corruption
clearance. The follower's request-scoped transfer delta is a global-counter
difference and must not be interpreted as its own cold load.
Its generated tokens/logprobs differ from the earlier C2 fresh references;
the follower executed without its held peer, so those are not matched-shape
comparisons. No output-correctness claim is made for this fallback.

The diagnostic now waits until both cold requests are WAITING_FOR_REMOTE_KVS
before expiry, with at least one newly prepared load. It protects/checks all
outstanding cold-pair jobs, not only the new job, and records two acquired
requests. The result gate requires that evidence in addition to all previous
ownership, eviction, concurrency and output checks. The original first-load
expiry plus follower-recomputation lifecycle still needs its own full-model
qualification without a two-cold-load barrier.

Artifacts retained: `screen-full-tp4-pending-expiry.json`,
`full-tp4-pending-expiry-event.json`,
`full-tp4-pending-expiry-barrier.json` and the original launch record.
Local tests:110 discovered,108 passed,2 dependency skips. Only diagnostic
tooling changed; no production engine change or serving restart.

Corrected full-model run `6b42b02672f9` started at12:53:10 UTC on GPUs0--3,
root `/tmp/flash-session-gpu-20260921-46xniO`. Frozen launch/source evidence:
`launch-full-tp4-pending-expiry-paired.json`. It retains TP4/Q8-MTP4,
canonical arithmetic, C2 paired promotion and page/shape tracing. Initial
inspection confirmed it was running; the completed result is recorded above.
Serving `7d408c2bc15d` remains running with its2026-09-20T19:45:53 start time.
The user clarified the production interface on 2026-09-21: ordinary completed
requests reuse state through native vLLM prefix matching. The resumable-request
coordinator below is a diagnostic prototype, not the intended production cache.
Continue on
`flash-next/mtp-q8-experimental-20260920` and PR
<https://github.com/zumi87/vllm-qwen3.8-flash-next/pull/1>.

## Required outcome

### Full-model three-turn continuation control prepared (2026-09-22)

The native full-model screen now supports `--continuations` after its existing
C1/C2 checks. It starts at5212 input tokens and appends three944-token turns,
reaching6156,7100 and8044. Each appended turn includes the preceding warm
run's actual128 generated tokens plus816 deterministic prompt tokens; cold
and fresh runs receive exactly that same history. The8192-token limit retains
room for output and MTP lookahead. This is a synthetic token-prompt lifecycle
test, not a chat-template or model-quality evaluation.

The prepared continuation fixture also changes native request-local repetition
penalty across its seed/three turns:1.0,1.05,1.1, then back to1.0. Warm, cold
and fresh references for each turn receive identical settings, greedy decoding
and the same seed. Reports record the requested penalty, require matched
per-turn settings, and reject a fixture that failed to exercise the changes.
Each request constructs a new native SamplingParams; no previous sampler or
output counter is restored. GPU output equality remains the required evidence
that cache reuse respects those new settings; CPU fixtures alone do not prove
it. This local change does not alter the frozen running expiry container.

Warm/cold chains use distinct native cache salts; each cold turn follows32
distinct pressure requests. Two fresh-reference namespaces run only after both
chains. Required evidence includes real cold loads, zero reference hits/loads,
matching positive hot/cold boundaries that grow across turns, identical prompts,
and exact128-token outputs plus recorded logprobs across warm/cold/fresh paths.
The existing screen gates remain mandatory. This control is explicitly
untraced and cannot combine with expiry diagnostics.

110 local session tests discovered:108 pass,2 dependency skips; Ruff passes.
Tests cover generated-answer retention, context bounds, namespace isolation,
reference ordering, real-load requirements and output mismatches.
The corrected paired-expiry run is terminal. Full continuation container
`18f8c44b6d4d` started at13:10:47 UTC on GPUs0--3, root
`/tmp/flash-session-gpu-20260921-RyV5zm`. Frozen launch/source record:
`launch-full-tp4-continuations.json`. Initial runtime inspection confirms
running/no OOM. This is TP4/Q8-MTP4 with canonical arithmetic and native-kernel
fallbacks, untraced, with no expiry extension. The completed result is recorded
above. Serving remains unchanged.

### Prepared-load expiry control prepared (2026-09-22)

The full-model diagnostic now also accepts `--pending-expiry`. Its existing
native scheduler subclass first delegates scheduling unchanged. Once the cold
C2 phase has a real prepared load, it advances only the private host/CPU cache
clocks, invokes native idle eviction, and verifies that pinned host pages,
CPU source objects/refcounts and destination ownership are unchanged.
The original scheduler output and transfer metadata are returned unchanged.

This boundary is after native source acquisition but before worker submission,
not proof that physical DMA is active. The result must show actual unrelated
evictions, positive protected-source/destination counts, restored clocks,
real cold-load bytes, C2 execution, and exact hot/cold tokens/logprobs.
Its gate is additional and cannot waive existing reference failures.
Tests cover unpinned/acknowledged/store-job refusal, protected-source replacement,
native schedule delegation, one-shot execution, and false-positive output gates.

106 local session tests discovered:104 pass,2 dependency skips; Ruff passes.
After the idle-expiry run exited, container `cd6b0994bb4a` started at
12:29:51 UTC on development GPUs0--3, root
`/tmp/flash-session-gpu-20260921-9EmA71`. Launch/source evidence:
`launch-full-tp4-pending-expiry.json`. It enables this new option, canonical
arithmetic, C2 paired promotion and existing page/shape traces; it does not
repeat the idle-expiry extension. The traces will compare measured checkpoint
and working state after restoration; their readbacks perturb timing.
This original run later failed in the diagnostic pair barrier; see the
failure/correction entry above. Serving remains unchanged; no promotion or push.

### Full-model native idle-expiry control prepared (2026-09-22)

The existing isolated paired-load scheduler can now run `--idle-expiry`.
After the ordinary C1/C2 hot/cold screen, a dedicated ordinary request triggers
expiry only if native requests, connector jobs and connector request ownership
are drained, and every non-null host page is unowned. Both tiers must still
have their configured 3600-second TTL. Only the diagnostic scheduler process's
two cache clocks advance by3601 seconds; imports restore in `finally`.
Native host-pool and CPU-manager eviction methods do the actual work.

The next request must be a real cache miss without bypassing cache lookup and
without cold loads; its128 tokens and recorded logprobs must exactly match a
new cache-bypassed reference. A following request must regain a native hot hit
and match the earlier hot output. Both tiers must report positive evictions.
The new verdict is additional: it cannot waive earlier fresh/hot/cold or C2
failures. No production engine source, TTL, clock, cache or serving setting is
changed. This is not a wall-clock60-minute soak or active-reader expiry test.

CPU tests:103 discovered,101 passed,2 dependency-related skips; Ruff and
diff checks pass. Tests include busy-state refusal, pinned-page refusal,
clock restoration after exceptions, one-shot admission ordering, and negative
miss/recomputation/output gates. The full-model launch uses the existing
TP4/Q8-MTP4 canonical arithmetic control, paired C2 promotion and no tensor
traces. Container `ae94e56b4ab0` started at12:10:23 UTC on development
GPUs0--3, root `/tmp/flash-session-gpu-20260921-Z6fQgO`.
Launch/source evidence: `launch-full-tp4-idle-expiry.json`.
Completed at12:26:11 UTC, exit1/no OOM,79 requests. The idle-expiry gate PASSES:
507 host pages and2611 CPU chunks expire through native policies; private
clocks restore. The ordinary expiry-trigger request has zero cached tokens and
zero load bytes, and its128 tokens plus recorded logprobs exactly match a fresh
cache-bypassed reference. The next request reuses5664 tokens with zero cold
loads and matches the original hot output.

C1 fresh/hot/cold checks pass. Both C2 hot/cold branches match all128 tokens
and logprobs, with real cold loads141,430,784 bytes (C1) and282,861,568 (C2).
C2 repeated fresh outputs match, but fresh versus cached still differs at29/14,
so the original combined screen remains failed. This is an expiry/lifecycle
pass under controlled arithmetic, not production qualification.
Evidence: `screen-full-tp4-idle-expiry.json`, `idle-expiry-analysis.json`,
`full-tp4-idle-expiry-event.json` and `full-tp4-idle-expiry-barrier.json`.
Serving peer `7d408c2bc15d` remains unchanged. No promotion or push.

### Multi-turn continuation control prepared (2026-09-22)

The reduced harness now supports `--continuations`: a 1920-token seed plus
three successive 944-token appends (2864, 3808 and 4752 tokens), using native
request cache salts to isolate warm, cold and two fresh-reference namespaces.
Warm/cold chains run before references, so reference writes cannot warm them.
Ten independent pressure requests precede each cold continuation; actual load
bytes must be positive. This is teacher-forced state validation, not a full
conversation-quality or throughput benchmark.

Each turn requires matching positive hot/cold reused boundaries, identical
retained RAM/index prefix bytes, no writes to complete retained prefix blocks,
exact target/draft hidden rows for every matched suffix chunk, exact final
target/draft/logit hashes, fresh-reference repeatability, and drained native
request/transfer ownership. It does not claim sharing of partially filled
blocks. Optional host capacity is 64 blocks for the one-target fixture and
128 for two targets; GPU capacity remains 32 blocks. No serving or engine
implementation settings change. All 31 local harness tests and Ruff pass,
including negative controls for stale ownership and mismatched restored bytes.
After the full-model trace exited, container `85be3776539e` started at
11:46:42 UTC on GPUs0--3, root `/tmp/flash-reduced-mtp-20260922-EFdIZr`.
Launch/source evidence: `launch-reduced-tp4-continuations.json`. Target0 then
target0+3, Q8 MTP4 and stable expert layout.

Completed at11:53:04 UTC, exit1/no OOM. All four ranks retain parameter
integrity and the original short-seed checks pass. Every appended turn performs
real cold transfers with matched positive hot/cold hit lengths944,1888,2832,
isolated namespaces, zero reference hits, unchanged measured retained bytes
during forward, and fully drained native ownership. Load bytes/rank are
1,644,544 /2,466,816 /3,289,088 for one target, and
2,466,816 /4,111,360 /5,755,904 for two targets.

All continuation exact-output gates FAIL. Even repeated fresh references first
differ in chunk1888--2400: GDN state and the one-target hidden output remain
exact, but MTP hidden output differs. Adding target QSA makes its output differ
there too. The third two-target turn also begins with different hot/cold prefix
bytes, after earlier independently computed turns already diverged; this is
not sufficient to attribute a transfer error. Frozen failed reports:
`reduced-tp4-continuations-rank0.json` through rank3 and
`continuations-analysis.json`. No failure has been waived.

Next controlled run enables the existing canonical QSA selection and tie
diagnostics for this same short-seed continuation fixture. Native top-k
replays and score/set audits are recorded before canonicalization. This tests
whether selection arithmetic causes the divergence without changing cache
ownership, transfers, prompts or comparison gates. These are diagnostic
overrides, not a proposed serving implementation or performance result.
Controlled container `ea3fd7fbccc7` started at11:55:32 UTC, root
`/tmp/flash-reduced-mtp-20260922-Z6sA6j`; launch/source evidence is
`launch-reduced-tp4-continuations-stable-qsa.json`. It uses GPUs0--3 only.
All31 local harness tests and Ruff pass.

Completed at12:01:38 UTC, exit0/no OOM. All four ranks pass the original
short-seed checks, parameter integrity, all MTP restore/rejection cases, and
every gate for all three appended turns with both one and two target layers.
Warm, cold and repeated fresh reference hidden chunks/final logits are exact;
retained prefix hashes and native cleanup pass. Each turn has real cold loads
and the same944/1888/2832 reused boundaries and load sizes recorded above.
Reports: `reduced-tp4-continuations-stable-qsa-rank0.json` through rank3;
summary: `continuations-stable-qsa-analysis.json`.

Native top-k replay with identical scores changes ordering in348--349 of512
rows across eight repeats, with unchanged sets at that observation point.
Across110 selections /44,000 rows (one target) and220 /88,000 (two targets)
per rank, native versus canonical membership differs in0--2 and13--15 rows,
respectively, but selected-score multisets always match and there are zero
invalid/duplicate indices. This implicates QSA selection ordering and tied
membership in the reduced output divergence; it is not a full-model accuracy
claim or proof that all optimized numerical issues share this cause.
The passing control qualifies these reduced multi-turn cache paths, not the
native-selection exact-output gate or production serving. Serving unchanged.

### Optimized full-model cache-state trace (2026-09-22)

Container `2b8b19b453e3` started11:30:54 UTC on development GPUs0--3,
root `/tmp/flash-session-gpu-20260921-mTtefc`, after the successful reduced
cancellation/expiry run exited. Frozen launch/source evidence:
`launch-full-tp4-optimized-state-trace.json`.

It repeats the optimized TP4/Q8-MTP4 C1/C2 matched-load fixture, with all19
inherited optimized kernel flags enabled and no canonical expert/QSA controls.
The added existing page/shape trace hashes retained main RAM KV, target/draft
compressed-index state, convolution history and recurrent checkpoints at the
5664-token boundary, and records the subsequent query inputs/layout.
This separates measured restoration-state differences from later numerical
divergence. Tensor readback synchronizes workers and perturbs timing: it is
not a race-freedom or production-throughput proof. Original output gates remain.

Completed at11:45:35 UTC, exit1/no OOM,76 requests. Original exactness gate
remains failed: fresh C1 repeats differ at output6; fresh C2 repeats at21/71.
Hot/cold first accepted-token differences are65 for C1 and29/34 for C2;
recorded logprobs differ too. Real cold loads are141,430,784 bytes for C1 and
282,861,568 for C2. All C2 phases reach two running requests; paired promotion
waits for both native acknowledgements (12 deferrals).

On all four ranks, C1 and both C2 hot/cold pairs have exactly matching measured
checkpoint and working state:37 convolution histories,36 temporal checkpoints,
78 main RAM-KV pages (72 target,6 draft), and78 compressed-key pages.
Initial query inputs/geometry also match: first differences occur after7 C1
suffix steps and8 C2 packed suffix steps (5/6 decode steps respectively).
Subsequent speculative inputs and schedules diverge; these are not whole-run
matched-computation comparisons. The C2 fresh branches themselves differ in
retained state despite identical input prefixes, before any cold restoration.

Evidence: `screen-full-tp4-optimized-state-trace.json`,
`optimized-state-trace-analysis.json`, and
`full-tp4-optimized-state-trace-barrier.json`. This supports correct measured
restoration but does not prove every state is covered, identify the first
differing arithmetic operator, exclude a kernel/integration bug, or establish
race freedom. Tracing perturbs timing. Do not waive the failed output gate.
Serving remains unchanged; no engine implementation or deployment changes.

### Shared-load cancellation/expiry control prepared (2026-09-22)

The reduced TP4/MTP harness now supports `--shared-cancel`. It submits two
ordinary native requests with an aligned shared prefix and divergent final16
prompt tokens. The initial frozen test incorrectly requires two overlapping
pending loads. Source review found native `_chunks_being_loaded` deliberately
defers the second lookup until the first transfer completes.

The revised local test preserves that native guard: it requires one pinned
leader load and a deferred follower with matching prefix hashes, aborts the
leader through native `finish_requests`, and verifies that its destination
remains owned until acknowledgement. After cleanup the duplicate-load fence
must clear, and the follower must submit its own real load from the same
CPU source objects and acquire the retained main-RAM prefix pages.

Before acknowledgement it advances only the private cache clocks by3601 seconds
and expires unrelated idle entries. The surviving owner's shared bytes and
ownership must remain valid after the peer's cleanup and after every subsequent
real target/MTP forward. Every surviving target/draft hidden chunk and final
target/draft/logit output must match the existing reference suffix schedule.
All native jobs/request ownership must drain. No original output gate is removed.

30 local harness tests/Ruff pass, including negative controls for missing loads,
expiry, shared owners, cleanup, changed hidden rows and final outputs.
After the full-model run exited, reduced container `7b6c1bfb2f2f` started
11:06:29 UTC, root `/tmp/flash-reduced-mtp-20260922-g0Lofd`, with target0 then
target0+3, Q8 MTP4, stable expert layout, shared cancellation and idle TTL.
Launch/source evidence: `launch-reduced-tp4-shared-cancel.json`.
The initial run exited1/no OOM at11:11:32 UTC; all four ranks rejected the
incorrect two-pending-load assumption. Its failure log/terminal state are
`shared-cancel-initial-failure.json`, with four
`reduced-tp4-shared-cancel-initial-rank*.json` reports retained.

Corrected diagnostic `dfb4c9f559ee` started11:12:36 UTC, root
`/tmp/flash-reduced-mtp-20260922-zIDZQZ`, after the initial container exited.
Launch/source evidence: `launch-reduced-tp4-deferred-cancel.json`.
It keeps native duplicate-load suppression and requires successful follower
reloading after leader cancellation/expiry. Completed11:18:02 UTC, exit1/no OOM.
All four ranks, target0 and target0+3, preserve pinned bytes/ownership, defer
the duplicate load, retain the cancelled destination until acknowledgement,
clear the native fence and drain all ownership. However, the follower
recomputes from token0 instead of reloading; its entire target/draft hidden
sequence and final target/draft/logit outputs match the fresh reference.
The original reload gate remains false. Evidence:
`reduced-tp4-deferred-cancel-rank0.json` through `rank3.json` and
`deferred-cancel-analysis.json`.

The next revision separates cancellation-only (requires real follower reload
and positive cached prefix) from cancellation-plus-expiry (requires safe exact
fresh recomputation, real expiry and preserved pinned state). This distinguishes
a cancellation reuse problem from a valid cache-miss fallback after other
entries expire. Missing output, ownership, expiry or cleanup evidence still
fails.30 local tests/Ruff pass. Engine/serving source is unchanged.

Split-control run `16097e837fad` started11:22:00 UTC, root
`/tmp/flash-reduced-mtp-20260922-wGu00S`, after the prior container exited.
Frozen launch/source evidence: `launch-reduced-tp4-cancel-split-expiry.json`.
It runs target0 then target0+3 with Q8 MTP4 and both cancellation cases plus
the existing idle-TTL control. Completed11:27:12 UTC, exit0/no OOM.
All four ranks pass both cancellation cases, original prefix/MTP checks,
idle-TTL checks and parameter integrity.

Without expiry, the follower reloads the same CPU source objects and acquires
the retained main-RAM prefix, resuming at944 cached tokens. Combined leader
and follower load bytes/rank are3,289,088 for target0 and4,933,632 for target0+3.
Every resumed target/draft hidden chunk and final target/draft/logit output
matches the reference suffix. With expiry, unrelated29/123 host blocks and
92/153 CPU chunks expire; pinned1/2 host blocks and2/3 CPU chunks remain exact.
The follower safely recomputes from0 and matches all fresh-reference hidden
chunks/final outputs. Both paths preserve the shared prefix and drain native
jobs, fences and request ownership.

Evidence: `reduced-tp4-cancel-split-expiry-rank0.json` through `rank3.json`
and `cancel-split-expiry-analysis.json`. This validates reduced native
overlapping-load cancellation/reuse and expiry-driven recomputation, not
simultaneous full-model decoding or physical DMA fault recovery.
Serving `7d408c2bc15d` remains unchanged.
The initial frozen test and revised local source are distinct; launch hashes
identify which was executed. The revised test covers native overlapping-load
deferral and one surviving model execution, not simultaneous C2 execution or
full-model concurrent cancellation.
Only diagnostic harness/launcher code changed; serving is untouched.

### Optimized-kernel matched-load control (2026-09-22)

After the untraced canonical control completed, container `4499cbb79e17`
started10:50:19 UTC on development GPUs0--3, root
`/tmp/flash-session-gpu-20260921-FXTSyJ`. Frozen source/launch evidence:
`launch-full-tp4-optimized-paired-loads.json`. Preflight verified source hashes,
free development GPUs, read-only weights and the unchanged serving peer.

It retains TP4/PP1, Q8 MTP4,7100-token prompts,128 output tokens, C1/C2,
private native CPU offload, RAM QSA KV,3600-second TTL, native prefix matching,
and matched cold-load promotion. All19 inherited optimized kernel flags are
enabled; the canonical expert layout, canonical QSA selection/tie controls,
native fallback override and tensor-readback tracing are disabled. Eager
execution and controlled admission remain, so this is an optimized-kernel
correctness control, not a production CUDA-graph throughput benchmark.
The original fresh/hot/cold screen gates remain unchanged.

Completed76 requests at11:04:01 UTC, exit1/no OOM. Exact-output qualification
fails: C1 cache-bypassed references first differ at output33; C2 repeated fresh
pairs differ at29/34. These requests have identical prompt hashes and zero
cached tokens. C1 hot/cold first differs at43; C2 hot/cold differs at21/34,
with non-exact recorded logprobs. Real cold loads remain141,430,784 bytes
for C1 and282,861,568 bytes for C2; every C2 phase reaches two running requests.
Both native load acknowledgements precede cold-pair promotion.

Evidence: `screen-full-tp4-optimized-paired-loads.json`,
`full-tp4-optimized-paired-loads-barrier.json`,
`optimized-paired-loads-analysis.json`, and
`full-tp4-optimized-paired-loads.tail.log`.
This run changes both kernel flags and canonical arithmetic controls relative
to the passing hot/cold diagnostic. It cannot identify a particular optimized
kernel or demonstrate cache corruption; cache-bypassed output already differs.
Do not promote or waive the failed exactness gates. Serving container
`7d408c2bc15d` remains unchanged; no deployment promotion or push.

Offline same-history analysis is retained in
`optimized-common-history-logprobs.json`. C1 repeated cache-bypassed runs
already differ at output-position0 (maximum common recorded-token logprob
delta0.46054), before their first token mismatch at33. Across positions0--33,
the maximum recorded common-token delta is2.46593. The two C2 branches have
same-history maxima4.23950/1.78806. These are top-k intersection measurements,
not full-distribution errors or quality scores. They must not be dismissed
as established harmless rounding, or compared after token histories diverge.

### Native transfer-failure guard tests (2026-09-22)

The existing native connector worker suite now tests rejected submissions and
reported unsuccessful completions for both loads and stores. All four cases
pass in an isolated CPU-only container using the frozen development engine
overlay: `c55af9464b28`,10:41:51--10:42:21 UTC, exit0/no OOM.
No GPU devices, network, production mounts writable, or model execution were
used. The pytest backend is mocked; actual connector control flow is exercised.

Failed transfers do not publish completion metadata. Failed loads retain their
job ownership; rejected stores remain queued. This verifies native fail-fast
behavior in normal Python execution, not automatic in-process recovery,
physical partial-DMA behavior, or full-model client failure handling.
Production connector source and serving settings are unchanged.

Evidence: `native-transfer-failure-cpu.json` (command, source hash, isolation,
terminal state and log) and `native-transfer-failure-cpu.xml` (four passing
cases). The tests extend
`tests/v1/kv_connector/unit/offloading_connector/test_worker.py` in the engine
checkout; native submission/completion assertions remain unchanged.

### Pending-load expiry coverage (2026-09-22)

The reduced harness now accepts `--pending-load-expiry`. After the existing
controls it pressure-evicts a real prefix, submits a new cold load and advances
only the two private cache clocks by3601 seconds before worker acknowledgement.
It requires native main-RAM-KV pages and CPU transfer chunks to be pinned,
expires unrelated idle entries, verifies protected ownership and main-RAM-KV
bytes, restores clocks, and requires exact target/draft hidden chunks and final
outputs against the pre-existing reference with the same suffix schedule.
Missing loads, missing expiry, lost ownership or output differences cannot pass.
This is submitted-but-unacknowledged load coverage, not proof of physically
in-flight DMA or simultaneous multi-request execution. Existing two/six-owner
allocator tests cover shared-prefix refcounts, not real concurrent model expiry.

29 local harness tests and Ruff pass. Reduced diagnostic `a50e5767912c`,
root `/tmp/flash-reduced-mtp-20260922-Ku8Mht`, started10:23:26 UTC after
the full-model run exited. Launch/source evidence:
`launch-reduced-tp4-pending-expiry.json`. It exercises target layer0, then
layers0+3, each with Q8 MTP4, pending-load expiry, cancellation and idle TTL.
Completed10:28:55 UTC, exit0/no OOM. All four ranks pass parameter
integrity, native prefix/MTP output checks, pending-load expiry, cancellation
and the subsequent idle-TTL/recomputation control. At the submitted load's
944-token boundary, the one-target phase retains1 main-RAM-KV block and2
CPU source chunks/rank while29 idle host blocks and92 CPU chunks expire.
The two-target phase retains2 host blocks and3 CPU chunks while123/153 idle
entries expire. Load sizes are1,644,544 and2,466,816 bytes/rank respectively.
Protected ownership and main-RAM-KV bytes stay exact; all resumed target/draft
hidden chunks and final target/draft/logit outputs match the reference suffix.
No cache clock or inference-source change survives the diagnostic.

Archived evidence: `reduced-tp4-pending-expiry-rank0.json` through
`rank3.json`, `reduced-tp4-pending-expiry-analysis.json`, and terminal log
`reduced-tp4-pending-expiry.tail.log`. This does not cover actual simultaneous
multi-request execution, physically-in-flight DMA, full-model expiry or injected
transfer failures. Serving remains unchanged.

### Untraced matched-load follow-up (2026-09-22)

The controller now supports explicit `--untraced-balanced-prefix` alongside
balanced C2 admission and paired native cold-load promotion. It rejects
page/shape/prefill/prompt-chunk/host-registration tracing in this mode. It
retains native hit/load counters and strict token/logprob output comparisons;
no original screen gate is removed. Native kernel fallbacks and the same
canonical expert/QSA diagnostic settings are retained, so this is a correctness
control without model-tensor readback, not a production-performance benchmark.
100 session tests discovered:98 pass,2 dependency skips; Ruff passes.
Full-model container `dd8189097835` started10:32:03 UTC, root
`/tmp/flash-session-gpu-20260921-09eBlP`. Frozen launch/source evidence:
`launch-full-tp4-untraced-paired-loads.json`. Completed76 requests at
10:46:50 UTC, exit1/no OOM. The original combined screen remains false:
fresh-versus-cached C2 first differences remain at output indices29/14.
Fresh pairs repeat exactly. No output gate was waived.

Without tensor-readback tracing, hot/cold output tokens and recorded logprobs
match exactly for C1 and both C2 branches,128 output tokens each. Cold loads
are141,430,784 bytes for C1 and282,861,568 bytes for C2; every C2 phase reaches
two running requests. The native async paired-load barrier is exercised,
deferring12 promotions before both load acknowledgements. This independently
reproduces the prior traced hot/cold result, but still retains canonical
expert/QSA diagnostics and controlled scheduling.

Evidence: `screen-full-tp4-untraced-paired-loads.json`,
`full-tp4-untraced-paired-loads-barrier.json`,
`untraced-paired-loads-analysis.json`, and terminal log
`full-tp4-untraced-paired-loads.tail.log`. Longer contexts, broader lifecycle
coverage and production-kernel/performance qualification remain open.
No serving changes or promotion.

### Reduction-order control and paired-load diagnostic (2026-09-22)

Reduced run `dac4878067c3`, root
`/tmp/flash-reduced-mtp-20260922-bx106Z`, ran09:59:40--10:04:41 UTC
and exited0. Launch: `launch-reduced-tp4-reduction-orders.json`.
Four archived `reduced-tp4-reduction-orders-rank*.json` reports and the
matching `reduced-tp4-reduction-orders-analysis.json` preserve results.
All ranks pass reduced MTP/native-prefix checks, parameter integrity and
unchanged replay KV state. Native BF16 reduction replay exactly reproduces
the observed MLP reduction at10/984/1888 rows. FP32 reduction of the same
BF16 partials, rounded back to BF16, exactly matches the FP64-summed reference
at all sizes and repeats. At10 rows it eliminates the row-position difference;
all25,600 native BF16 output elements are compatible with at least one of
24 sequential or3 paired BF16 accumulation orders. This is per-element
compatibility, not identification of NCCL's exact algorithm. At984/1888 rows,
270/494 position differences remain after FP32 reduction, matching the reference:
some routed-expert partials already differ. No model dtype, cache logic or
serving arithmetic was changed.28 local harness tests/Ruff pass.

For the next full-model control, `tools/flash_next/prefix_load_barrier.py`
adds a diagnostic-only subclass of the actually resolved native scheduler
(AsyncScheduler when selected). It defers promotion only for
`batch-cold-0/1` until both native remote-load acknowledgements are present,
then delegates promotion and ownership changes to the native implementation.
Other requests/statuses are unchanged; missing peers time out after30 seconds
and failed or ambiguous loads fail closed. The controller saves release and
promotion evidence and retains all original output gates. No production
scheduler source is changed.99 engine session tests discovered:97 pass,
2 dependency skips; Ruff passes. This diagnostic still needs GPU validation.
Full-model diagnostic `7052d9030b8b` started10:07:41 UTC, root
`/tmp/flash-session-gpu-20260921-pCp00I`, launch/source evidence
`launch-full-tp4-paired-cold-loads.json`. It retains the previous
TP4/Q8-MTP4,7100-token/128-output-token, balanced944-token prefill control,
with private native CPU buffers, direct RAM KV and page/shape traces.
`--paired-cold-loads` is the added scheduling control.
Completed76 requests; exit1/no OOM at10:22:18 UTC. The original combined
fresh/hot/cold screen remains false because fresh-versus-cached C2 outputs
differ at output indices29/14. Fresh C2 pairs repeat exactly. No gate was
waived and no serving setting changed.

The new matched HOT-VERSUS-COLD control passes for C1 and both C2 branches:
all128 tokens and recorded logprobs are exact. C1 loads141,430,784 bytes;
C2 loads282,861,568 bytes and reaches two running requests. The native
AsyncScheduler subclass deferred12 promotion attempts, then acknowledged
both loads before delegating both promotions. All four ranks have complete
page/shape trace coverage. Both C2 branches have identical measured hot/cold
state at checkpoint and working columns:37 convolution histories,36 temporal
states,78 main-RAM-KV pages and78 compressed-index pages. Whole suffix query
inputs and packed geometry match through completion:35/36 observed steps,
including33/34 decode steps for branches0/1. No own-query or packed-query
difference is recorded. The earlier cold-only scheduling mismatch is removed
in this diagnostic, without changing inference arithmetic.

Evidence: `screen-full-tp4-paired-cold-loads.json`,
`full-tp4-paired-cold-loads-barrier.json`,
`paired-cold-loads-analysis.json`, and
`full-tp4-paired-cold-loads.tail.log`. The passing matched transfer control
does not establish global batch-invariant generation, uninstrumented race
safety, longer-context correctness, expiry/cancellation under full concurrency,
capacity or performance. Those remain separate qualification requirements.

### Duplicate-row MLP attribution (2026-09-22)

Reduced TP4 diagnostic `b0e477bb0bf5`, root
`/tmp/flash-reduced-mtp-20260922-pUQevz`, started09:38:48 UTC.
Launch/source hashes: `launch-reduced-tp4-duplicate-rows.json`.
It retains the prior real-input peer/router reference checks and adds identical
activation banks at different row positions:5/492/944 rows per branch,
10/984/1888 packed. Four repeats compare native/direct BF16 router scores,
ordered/sorted expert IDs, route weights, local expert output and final MLP
output across the two halves. All rows are checked, not only the first five.
Inputs come from real layer0 MLP captures repeated to the required lengths;
this is an MLP component test, not an actual two-request attention prefill.
Cache state and parameter integrity must remain unchanged. Position invariance
is reported separately from fixed-shape repeatability and cannot waive the
full-model output gate. Native prefix/MTP checks follow.26 local tests/Ruff
pass. The GPU run completed at09:43:46 UTC, exit0/no OOM. All four
rank reports are archived as `reduced-tp4-duplicate-rows-rank0.json`
through `rank3.json`. Fixed-layout repeats are exact, but row-position
invariance fails: router scores, expert IDs and route weights match at all
sizes; final MLP differences reach0.001953125 for10/984 rows and0.00048828125
for1888 rows. Local routed output is exact for10 rows, differs by at most
0.00006103515625 on ranks0/3 for984 rows and rank0 for1888 rows, and matches
on the other ranks. This implicates computation outside cache transfers but
does not distinguish expected arithmetic from a kernel bug. No serving or
inference-source changes.

The next bounded diagnostic observes shared gate/up, activation, down, gate
scaling and TP reduction input/output. It requires instrumentation to preserve
the unobserved MLP output exactly, records all four repeats, and compares native
reduction against an FP64 sum of the same BF16 rank partials. This reference
does not substitute for a full model reference.27 local tests/Ruff pass.
Launch evidence: `launch-reduced-tp4-shared-reduction.json`;
container `a3d0ff538e9f`, root
`/tmp/flash-reduced-mtp-20260922-eHAohu`, started09:51:11 UTC.
Completed09:56:12 UTC, exit0/no OOM. All four ranks pass fixed-layout
repeatability, instrumentation-vs-unobserved equality, parameter integrity,
unchanged replay KV state and the existing reduced MTP/native-prefix screen.
Evidence: `reduced-tp4-shared-reduction-rank0.json` through `rank3.json`,
`reduced-tp4-shared-reduction-analysis.json` and terminal/backend log
`reduced-tp4-shared-reduction.tail.log`.

All observed shared-expert stages are row-position exact at10/984/1888 rows
on every rank. Their three linear methods are native UnquantizedLinearMethod.
At10 rows, routed output and TP input are also row-position exact on EVERY
rank, while TP output differs in1136 elements per half, max0.001953125.
An FP64 sum of the same BF16 partials is row-position exact; the native
reduction result equals the final MLP output. Thus this fixture directly
isolates the position-dependent difference to TP reduction, not cache, router,
shared expert or local routed arithmetic. Runtime reports PYNCCL as the only
enabled TP all-reduce backend; native custom all-reduce is disabled on the
four PCIe-only GPUs. This is not yet proof that all differences are acceptable
rounding or that all cache paths are correct.

At984/1888 rows, some local routed/TP inputs already differ, so reduction is
not the sole source there. The FP64-summed, BF16-rounded reference still has
270/494 differing half-pair elements, max0.000244140625/0.00048828125.
Next bounded control should compare native reduction with BF16 accumulation
orders and a native FP32 reduction of the same partials, and separately
attribute routed-expert position differences. Do not require or silently
enable global batch invariance as a cache implementation change. Full-model
matched hot/cold scheduling and original output gates remain unresolved.

The engine session suite passes94 tests with2 dependency skips (96 discovered);
these are not full-model qualification. Serving identity `7d408c2bc15d`
and its2026-09-20T19:45:53 start remain unchanged. No inference-source changes
or serving promotion were made.

Source inspection identifies native
`Scheduler._try_promote_blocked_waiting_request` as the transition that makes
completed remote loads runnable. A future matched-load diagnostic could defer
that transition until both test requests have completed, while retaining native
load, acknowledgement and block ownership logic. No such barrier is implemented
or qualified yet; admission synchronization alone was insufficient.

### Completed reduced cold-load cancellation screen (2026-09-22)

The reduced native prefix harness now offers `--cancel-cold-load`. After its
existing exact fresh/hot/cold control, it pressure-evicts GPU state, submits
another real cold load and calls native `Scheduler.finish_requests` with
`FINISHED_ABORTED` before worker wait/completion acknowledgement. The probe
requires the request and all non-null block references to remain owned at that
boundary, then drains native worker/all-rank acknowledgements and requires
request/job/deferred-store bookkeeping cleanup. A subsequent normal request
must match the uncached reference's suffix schedule, every target/draft hidden
chunk and final target/draft/logit hashes. Merely aborting without a load or
matching only final output cannot pass.

This is a submitted-but-unacknowledged transfer test, not proof that PCIe DMA
is physically incomplete at the cancellation instant. It does not yet cover
mid-kernel cancellation, injected transfer failures or automatic recovery.
No inference-source, serving or cache-policy change was made.

Completed reduced cancellation/expiry run `26fd728d4b4e`,
root `/tmp/flash-reduced-mtp-20260922-tAxL58`,09:27:52--09:33:09 UTC,
exit0/no OOM. Launch/source hashes: `launch-reduced-tp4-cancel-expiry.json`.
Four final reports: `reduced-tp4-cancel-expiry-rank0.json` through
`rank3.json`; terminal/serving identity in `reduced-tp4-cancel-expiry.tail.log`.
Both the layer0+MTP and layer0+layer3+MTP phases pass on all four TP ranks,
using1920-token prompts/512-token budgets. Each abort occurs with one real
load job pending acknowledgement at944 cached tokens:1,644,544 bytes and8
non-null blocks/rank in the one-target phase;2,466,816 bytes and10 blocks/rank
with two targets. Native ownership remains until acknowledgement, all tracked
request/load/store/deferred-free bookkeeping drains, and the subsequent request's
suffix hidden chunks and target/draft/logit outputs are exact.

The configured3600-second idle TTL also passes after cancellation. A private
diagnostic clock shift of3601 seconds expires30 host blocks/64 CPU chunks in
the one-target phase and94/106 in the two-target phase on each rank. Expired
requests hit zero cached tokens, load zero bytes and exactly reproduce fresh
target/draft hidden chunks and outputs; the next request reuses944 tokens.
Clocks restore afterward; no OS clock change or wall-clock soak. Parameter
integrity and existing MTP restore/rejection checks pass. Full-model concurrent
cancellation, active-reader expiry, transfer failures/recovery and long-context
capacity remain unqualified.25 local harness/analysis tests pass.

### Completed balanced full-model control (2026-09-22)

`09c5ade19435` completed76 requests and exited1/no OOM at09:27:05 UTC.
Final evidence: `screen-full-tp4-balanced-native-threshold.json`,
`balanced-native-threshold-analysis.json` and
`full-tp4-balanced-native-threshold.tail.log`. The reproducible offline
analyzer is `infra/inference/analyze_flash_prefix_state.py` in k3s.
Page/shape coverage passes all four ranks. Original output gate remains false.
C1 fresh/hot/cold tokens and recorded hot/cold logprobs match. C2 fresh pairs
repeat exactly; hot differs from fresh at output29/14, cold at29/34.
Hot versus cold now differs at49/14. All C2 phases reach two running requests;
each cold phase loads141,430,784 bytes and reused prefixes hit5664 tokens.

The controlled admission works for fresh/hot prefill: both requests process
944 rows together (1888 packed), then492 each (984 packed) for their suffix.
But cold loads become runnable at different times: branch0 first resumes
alone (944 rows); branch1 first resumes beside branch0's492-row suffix
(1436 packed). Branch0 first decode shares497 rows with peer prefill instead
of10 decode rows. Consequently this is still NOT a matched hot/cold computation
control despite synchronized admission.

On every rank, measured hot/cold retained state matches for both requests:
37 consumed convolution histories,36 temporal checkpoints,78 main RAM-KV
pages and78 compressed-key pages. Counts match at checkpoint and native
working columns. Fresh branch0 also matches its hot state. Fresh branch1
differs from its hot state in34/37 histories,35/36 temporal checkpoints and
all78+78 KV/index pages. Importantly, the two uncached branches themselves
have this same state difference despite identical input-token hashes and
positions through5664 in six synchronous two-request chunks. Their divergence
therefore precedes cold transfer and user-specific suffix text. This does not
prove harmless arithmetic or rule out a kernel/model-integration bug.

Next numerical attribution should test identical real layer inputs at different
packed row positions (including native/direct BF16 router and expert outputs).
Do not run another identical full screen. Full-model exact cold comparison also
needs matched post-load scheduling, not admission alone. Neither original
verdict nor production settings were changed; serving remains unchanged.

### Router arithmetic attribution (2026-09-22)

Reduced TP4/Q8-MTP4 diagnostic `9c8d20ff50c4` started09:00:14 UTC,
root `/tmp/flash-reduced-mtp-20260922-c3e7nW`. Evidence/source hashes:
`launch-reduced-tp4-peer-router-reference.json`. It extends the prior
cache-free MLP replay with direct torch BF16 linear results, a five-row CPU
FP64 reference using the same loaded weights, runtime router implementation/
dtype/shape metadata, and sorted expert-ID comparisons. It preserves native
outputs and weights. The high-precision reference is not an unquantized
checkpoint or model-quality baseline. Normal native prefix/MTP checks follow.
22 local harness tests and Ruff pass. Only isolated diagnostic
tooling changed; serving remains unchanged.

Completed09:05:01 UTC, exit0/no OOM. Reports:
`reduced-tp4-peer-router-reference-rank0.json` through `rank3.json`.
All four ranks pass parameter integrity, fixed-shape repeatability, MTP
restoration/rejection and native prefix lifecycle checks. Runtime identifies
the router as BF16 `UnquantizedLinearMethod` using
`default_unquantized_gemm`, with contiguous512x2560 BF16 weights.
Native router scores exactly match direct `torch.nn.functional.linear`
at all five tested cases. The CPU FP64 reference hash matches across ranks;
maximum native FP64 errors are0.0399907/0.0399907/0.0311527/0.0363986/0.0363986
for5/10/497/949/949 rows. Different batch sizes thus already change the
ordinary PyTorch BF16 router computation, without our expert or RAM-KV path.
The sorted expert-ID sets are identical for every focal row/case/rank:
the earlier ordered-ID differences were reordering, not different experts.
This does not isolate the shared/expert contribution to final MLP differences
or qualify all full-model numerical behavior. No weights/kernel source changed.

Next full-model control uses native2048 input/scheduled budgets and944-token
long-prefill limit with the existing two-request capacity. A diagnostic admission
observer delegates to native `add_request` while native pause(mode=keep,
clear_cache=False) holds scheduling; it resumes after both admissions, restoring
the observer and resuming on failures/timeouts. Ordinary `generate` remains
responsible for output collection and cancellation. The C2 prompt uses its own
Document2000 prefix rather than reusing C1's computed prefix. Actual per-request
and packed shapes/state hashes will decide whether execution really matches;
settings alone are not proof. Original exact-output gates are unchanged.
42 controller tests pass, including native-admission failure/timeout cleanup;
the session suite96 tests has94 passes/two dependency skips.

The initial full launch `9f9b5ce4a0a6` (root
`/tmp/flash-session-gpu-20260921-86h2xV`) exited1/no OOM at09:09:58 UTC,
before engine construction: this checkout does not accept the older
`max_num_partial_prefills`/`max_long_partial_prefills` EngineArgs fields.
Evidence: `launch-full-tp4-balanced-prefix.json`,
`screen-full-tp4-balanced-prefix-argument-failure.json` and the matching
`full-tp4-balanced-prefix-argument-failure.tail.log`. No requests ran.
Those unsupported fields were removed after checking the current EngineArgs
and SchedulerConfig; the supported native threshold,2048 budgets and synchronized
admission remain.42 controller tests/Ruff pass. Replacement launch evidence:
`launch-full-tp4-balanced-native-threshold.json`; results pending.

Replacement `09c5ade19435` / `flash-session-gpu-w6m8w6` started
2026-09-22T09:12:49 UTC; root
`/tmp/flash-session-gpu-20260921-w6m8W6`. Docker confirms it running on
development GPUs0--3. Poll this exact handle; do not restart it on an observation
timeout. Results go to `results/screen.json`. Serving identity/start are
unchanged; no source commits, pushes or production promotion occurred.

### Reduced peer-batch attribution (2026-09-22)

The first MLP replay `baf2313b9b6f` (root
`/tmp/flash-reduced-mtp-20260922-ulo3OV`) exited1/no OOM at08:44:52 UTC.
The diagnostic incorrectly sliced the return from native `fused_marlin_moe`;
its supplied reduction callback writes the caller's output buffer and returns
None. This was an observation-hook failure, not a cache or inference result.
The hook now snapshots the supplied buffer when the return is None and
preserves the exact native return. A CPU regression checks both return styles,
snapshot independence and missing-output failure; all21 harness tests pass.

Follow-up `3a4ed5bf0e0c`, root
`/tmp/flash-reduced-mtp-20260922-dxBH5l`, uses the same TP4 one-target-layer
and Q8 MTP4 diagnostic. Launch/source hashes:
`launch-reduced-tp4-peer-batch-buffer.json`. It captures eight real four-row
MLP inputs, then replays five fixed focal rows alone or with5/492/944 peers,
including two peer-content variants at944. Four repeats compare router
logits, selected experts/weights, routed and final MLP outputs. Inputs and
KV state must remain unchanged during MLP-only replay. This does not execute
attention or MTP during replay and cannot alone explain full-model drift.
The normal reduced native offload/MTP checks follow afterward.
Only diagnostic code changed; serving GPUs4--7 remain untouched.

Completed08:54:04 UTC, exit0/no OOM. Four reports are archived as
`reduced-tp4-peer-batch-buffer-rank0.json` through `rank3.json`;
`reduced-tp4-peer-batch-buffer-analysis.json` summarizes them without changing
any verdict. On every rank, parameter integrity, all fixed-shape repeat checks,
three state-restoration cases,60 MTP rejection cases and the native prefix
lifecycle pass. Fresh/hot/cold target/draft/logit outputs are exact; hot/cold
reuse944 tokens and cold loads1,644,544 bytes/rank. This is the1920-token,
one-target reduced fixture, not full-model qualification.

The cache-free MLP replay is **not batch-invariant**. All focal input hashes
match. Against five rows alone, final MLP maximum absolute differences are
0.0009765625 at10 rows and0.001953125 at497/949 rows. Router scores are exact
at10 rows but differ by up to0.0625 at497/949 rows; respectively2/5 ordered
expert-ID entries differ. The report does not retain sorted expert IDs, so
this does not establish selection-set changes versus ordering. Routed local
expert outputs also differ (maximum0.0003662109375 across ranks/cases).
The two different peer-content banks at949 rows produce identical focal
final-output hashes, which also match across all four TP ranks. KV state
remains unchanged during replay. No attention, MTP or offload runs within
that replay, so these particular differences do not require cache corruption.

This establishes component-level batch-shape dependence, not its complete
numerical cause or harmlessness, and not a proof of the full C2 mismatch.
The gate is an unquantized native ReplicatedLinear inherited from Qwen3Next;
the current expert path still includes our TP4 K32 support. A bounded next
attribution is native gate versus direct linear/high-precision reference on
these exact inputs, plus ordered-versus-set route comparison. Then qualify
full-model restoration under matched computation and extend cancellation,
active-reader TTL, long-context/capacity and uninstrumented serving checks.
Serving ID/start remain unchanged.21 local harness tests pass; engine session
suite94 tests:92 pass/two dependency skips. No serving changes or promotion.

### Completed consumed-convolution/decode-shape probe (2026-09-22)

Development container `6f8ae03ea282`, root
`/tmp/flash-session-gpu-20260921-tJcNQE`, keeps the prior TP4/Q8-MTP4,
C1/C2 divergent prompts,1024-token budgets and deterministic-selection controls.
Launch and frozen source hashes:
`launch-full-tp4-conv-decode-shapes.json`. Inference sources/settings are
unchanged; only read-only diagnostic tools changed. Completed results below.

The page trace additionally hashes the prefill convolution history window:
GDN uses the first `conv_kernel_size - 1` temporal entries after native
layout canonicalization; PLE first selects its native trailing
`conv_state_len + num_spec_tokens` capacity, then hashes its history window.
These rules follow the native prefill kernels. Full-page hashes are retained.
Snapshots also record the native Mamba running column so checkpoint and
working-state comparisons do not rely solely on table position.

While page tracing is enabled, the first-target-layer hook now retains
decode-only steps as well as prefill steps (at most512 per named phase).
It records native internal request IDs, actual GPU query row boundaries,
token-position ranges and input-ID hashes, plus padding rows. It does not
change inputs or scheduling and returns no token/tensor payloads. Original
shape-only mode retains its prior64-step behavior and schema. Explicit
readbacks perturb timing; this is not a performance or race-free qualification.

The corrected public/internal request-ID coverage check is included in this
frozen run.39 local tooling tests pass; the session suite discovers93 tests,
91 pass/two dependency skips; Ruff/diff checks pass. Tests verify scratch-only
changes leave the extra history hash unchanged, consumed-history changes alter
it, both cache layouts and PLE capacity slicing, actual mixed request geometry,
decode-only capture and hook/native-method restoration. The offline comparator
also reports history equality/missingness separately from whole-page equality;
that reporting-only extension was added after this run was frozen.
Serving GPUs4--7 remain unchanged. No commits, pushes or promotion occurred.

The offline request-step comparator now separates a request's own position/
input hashes from its surrounding packed batch (including padding and peer
rows), ignores randomized ID spelling only after selecting the exact request,
and preserves whole steps crossing the comparison boundary. Missing evidence
does not count as an observed match. It is reporting only, not an equality
waiver, and was added after the active run was frozen.40 tooling tests pass;
the session suite discovers94 tests,92 pass/two skips; Ruff/diff pass.

#### Completed evidence and next action

The probe exited1/no OOM at08:31:34 UTC after76 requests. Both page and shape
coverage checks pass on all four ranks. Final reports:
`screen-full-tp4-conv-decode-shapes.json`,
`conv-decode-shapes-analysis.json`, and the honestly labeled
`full-tp4-conv-decode-shapes.tail.log`. The unchanged output gate still fails
fresh-versus-cached C2 at55/36; hot/cold tokens and recorded logprobs are exact
for C1 and both C2 branches. Real hits remain5664 tokens, each cold phase
loads141,430,784 bytes, and all C2 phases reach two running requests.

On every rank, all37 consumed convolution history windows match between fresh
C1 repeats, hot/cold C1, both hot/cold C2 branches, and first-branch fresh/hot
C2. This holds at checkpoint column5 and native running column6. Thus the
earlier full-convolution differences in those comparisons are outside the
prefill history window. The selected36 temporal checkpoints,78 RAM-KV pages
and78 compressed-key pages also match for those pairs. The second fresh C2
branch differs from cached in34/37 convolution histories,35/36 temporal
checkpoints and all78 KV/78 compressed-key pages, before suffix execution.

Per-request decode traces explain an important remaining mismatch in controls.
For C2 branch0, the final prefill query is identical (492 rows). Its first
decode query has identical own positions7100--7104 and input hash in fresh
and cached runs, but the peer is prefilling0--943 versus5664--6607. The next
packed step has949 versus497 total rows. Own queries remain identical for
the first14 decode steps; peer computation differs starting at step0.
First logprob differences occur at output1/0, before output-token differences
55/36. Step indices are not output-token indices. Hot/cold own AND complete
packed decode records match for all39/33 steps of the two requests on every
rank. This gives stronger evidence against a cold-copy error in this fixture;
it does not prove all kernel arithmetic is batch-invariant or race-free.

Do not repeat a full run only to collect the same state/shape evidence.
Next numerical attribution should use a one/two-layer mixed-batch control
with identical own inputs/state and deliberately different peer rows, or a
genuinely matched full-model execution control. Broader cancellation/inflight
failure, concurrent TTL/long-context capacity, uninstrumented optimized
performance and same-PR review/publication remain open. No equality gate was
waived and no production source/settings changed.

### Completed full-model page comparison (2026-09-22)

The chunked probe `68d5cb366349` completed all76 requests and exited1/no
OOM at08:03:07 UTC. Final evidence:
`screen-full-tp4-prefix-pages-chunked.json`; the earlier partial report is
retained separately. The original verdict remains failed: C2 fresh-versus-
cached token differences remain55/36, while both hot/cold pairs match all128
tokens and recorded logprobs. C1 fresh/hot/cold remain exact. Every reuse
hits5664 tokens; cold phases load141,430,784 bytes and C2 reaches two running
requests. No serving changes occurred.

Readback succeeded:435,390,656 bytes per fresh snapshot and286,050,496 per
cached snapshot, below the new2GiB total cap. The original coverage checker
mistook vLLM's native eight-hex-character request-ID suffix for missing requests.
All12 expected requests are present on all four ranks. A source-backed helper
now maps exact public IDs or their native suffix form, rejects wrong owners,
duplicates, malformed suffixes, missing requests and zero-byte snapshots, and
records the mapping. Native request randomization remains enabled.37 tooling
tests pass; session suite91 discovered/89 pass/two skips, Ruff/diff pass.
Offline coverage verification does not rewrite the historical report verdict.

Offline analysis: `prefix-pages-chunked-analysis.json`. On every rank and
for C1 and both C2 hot/cold pairs, selected retained state hashes are exact:

- 72 target RAM-KV pages and6 draft RAM-KV pages, with unchanged RAM page IDs.
- 72 target compressed-key pages and6 draft compressed-key pages.
- All36 GDN temporal checkpoints,36 complete GDN convolution checkpoints and
  one PLE convolution checkpoint at the5664 boundary.
- All151 selected GPU entries have different physical IDs after cold restore,
  while their logical content hashes are identical.

Both C2 cached prefixes also match the selected state from C1 reference0;
their78 RAM page IDs match that reference. In contrast, the second fresh C2
branch differs from its cached counterpart in35/36 temporal checkpoints and
all78 RAM-KV/78 compressed-key pages before resuming at5664. Repeated fresh
C2 runs reproduce the same temporal/KV/compressed-key hashes. Its historical
prefix was computed in949-row mixed steps rather than944-row isolated steps.
This is evidence of different prefix computation, not cold-transfer damage.

The first C2 fresh branch has matching temporal/KV/compressed-key state but
different full convolution hashes. Full convolution hashes include speculative
scratch; such differences also exist between exact-output fresh repeats.
They cannot be classified as corruption from these hashes alone. Ring pages,
unused historical recurrent pages and future writable slots are excluded from
the selected comparison, not declared equal. Full-model qualification remains
open. Next diagnostic should isolate the convolution bytes actually consumed
by prefill and record per-request decode batch geometry; do not rerun merely
to repair request-ID coverage.

### Chunked page-readback follow-up (2026-09-22)

The first page probe `76266834c0b6` is terminal: exited1/no OOM at
07:43:46 UTC before completing a request. Its256MiB aggregate read limit
raised `Prefix-page readback exceeds byte bound`; this is a diagnostic
capacity failure, not evidence of cache corruption. The incomplete report is
retained in k3s as `screen-full-tp4-prefix-pages-bound-failure.json`.

The successor `68d5cb366349`, root
`/tmp/flash-session-gpu-20260921-OepKz8`, uses the same engine/settings and
a corrected bounded reader. It checks the total planned page bytes before
any page hashing, caps aggregate reads at2GiB per snapshot, and hashes logical
row-major bytes in chunks of at most8MiB per temporary tensor buffer. It does
not omit pages to fit the limit or weaken equality gates. Only hashes and
descriptors leave the worker. Local chunk tests cover strided FP32/BF16 views,
identical full-byte hashes, no input mutation and rejection before hashing
when the total limit is exceeded.36 tooling tests pass; the session suite
discovers90 tests,88 pass/two skips; Ruff and diff checks pass.

An offline comparator now distinguishes logical content from physical page IDs
and keeps missing pages and RAM/GPU pool layouts separate. Table-position
categories are not semantic validity masks for recurrent or ring caches.
Source hashes/settings: k3s `launch-full-tp4-prefix-pages-chunked.json`.
Results pending. Serving GPUs4--7 remain unchanged. This supersedes the
historical "active" status of the first page probe below.


Active full-model page-provenance diagnostic `76266834c0b6`, started07:35:09
UTC, root `/tmp/flash-session-gpu-20260921-Ll0K5O`; k3s launch/source hashes
`launch-full-tp4-prefix-pages.json`. TP4/Q8-MTP4, C1/C2 divergent tails,
1024-token budgets and deterministic-selection controls. New
`--trace-prefix-pages` delegates native V2 input preparation and hashes each
request's live pages at5664 before target forward. It preserves separate
pool identities, validates kernel-split table geometry, labels allocated
working/suffix pages and caps readback at256MiB per snapshot. Positional
boundary labels are not semantic valid-byte masks for recurrent/ring data.
Only hashes/descriptors are returned. Explicit device synchronization makes
GPU-written pinned RAM readable but prevents race/performance qualification.
Required request coverage and method/hook restoration are checked.34 targeted
tooling tests pass;131 local tests discovered/129 pass/two skips; Ruff and
diff checks pass. Confirmed running, results pending. Serving unchanged.

C2 numerical follow-up of the completed1024 control: first logprob differences
at positions2/0, first token differences43/28; maximum common top-five logprob
deltas1.3125/4.0 through the first divergent output. K3s evidence:
`prefill1024-common-history-logprobs.json`. These are not uniformly last-bit
effects. New reporting excludes later positions whose generated histories
already differ, records top-k overlap and never invents missing probabilities.
31 tooling tests/Ruff pass; no tolerance/gate change. The active divergent
run is frozen and predates this reporting addition. Native duplicate-hash
lookup selects the first remaining block; actual provenance is not yet traced.

Completed k3s reduced control `ab6c93245ccb`, started07:16:16 UTC, root
`/tmp/flash-reduced-mtp-20260922-ReFTXo`; evidence/source hashes
`launch-reduced-tp4-expiry.json`. TP4/two target layers/Q8 MTP4,7100-token
full-prefill/matched-shape fixture. `--cache-expiry` adds model-backed native
TTL expiry, miss/recomputation and renewed-hot-reuse coverage. It advances
only private diagnostic cache clock imports by3601 seconds after drain,
verifies the configured3600-second policy and restores imports on failure.
Full target/MTP chunk hashes and final outputs are compared; no numerical
gate is relaxed. Exited1/no OOM at07:21:54 UTC. One-target expiry passes.
Two-target expiry removes200 host blocks/166 CPU chunks per rank, misses both
tiers and exactly reproduces both fresh split controls' full target/MTP
hidden and output hashes. Renewed hot reuse is exact with5664-token hits.
The original unsplit comparison fails and is retained. K3s evidence:
`reduced-tp4-expiry-rank0.json` through `rank3.json`, and offline
`reduced-tp4-expiry-analysis.json`.19 local reduced harness tests/Ruff pass;
matched controls are selected by schedule only. The run also audits79,616
QSA rows/rank:138--140 tied membership changes, no wrong scores, invalid
indices or duplicates. Full-model C2 and broader lifecycle remain unqualified.
That diagnostic is terminal; the full-model successor above is active.
Serving untouched.

Completed divergent-tail successor: `c4862568a09d`, started07:01:02 UTC,
root `/tmp/flash-session-gpu-20260921-MCy3lH`; k3s launch/source evidence
`launch-full-tp4-prefill1024-divergent.json`. Keeps TP4/Q8-MTP4, native prefix
offload,1024-token budgets and deterministic selection diagnostics. Adds C2
private prompt tails after at least6608 shared tokens, explicit direct hot/cold
token/logprob comparisons and packed-shape reporting. Exited1/no OOM at
07:15:05 UTC;76 requests. C1 exact; C2 fresh pairs repeat and branches remain
distinct after7087 shared prompt tokens. Each cold branch exactly matches
its hot counterpart for128 tokens and recorded logprobs, with5664-token hits
and141,430,784-byte cold phases. All C2 phases reach two running requests.
Fresh-versus-cached differences remain at55/36; suffix traces match across
all four ranks. K3s evidence: `screen-full-tp4-prefill1024-divergent.json`,
`full-tp4-prefill1024-divergent.tail.log`. Full gate remains failed; serving
GPUs4--7 remain unchanged. Latest CPU suite128 discovered/126 pass/two skips.

Completed chunk-shape control: `ecf884dacde3`, started06:44:11 UTC,
root `/tmp/flash-session-gpu-20260921-4tAcGO`, k3s evidence/source hashes
`launch-full-tp4-prefill1024-shapes.json`. C1/C2 TP4/Q8-MTP4, both prefill
budgets1024, deterministic selection diagnostics and first-layer packed
position-range tracing for every named phase. All-layer activation tracing
and divergent tails are disabled in this control. Check actual recorded
shapes, not just configured budgets. Exited1/no OOM at06:57:56 UTC;76 requests.
C1 fresh repeats, hot and cold match all128 tokens and recorded logprobs.
C2 fresh repeats match; hot/cold both differ from fresh at43/28, but directly
comparing hot and cold gives exact128-token/logprob matches for both requests.
Each cold phase loads141,430,784 bytes; each C2 phase reaches two running
requests. K3s evidence: `screen-full-tp4-prefill1024-shapes.json`,
`full-tp4-prefill1024-shapes.tail.log`. Full correctness gate remains failed.

All four rank traces match. Fresh/hot/cold suffix packed shapes match in both
C1 and C2. Fresh C2's second historical prefix is computed alongside five
decode rows (949-row steps), unlike its first (944-row steps); cache provenance
and decode schedules remain confounders. Position traces are not request IDs
or historical-state equality proofs. Direct hot/cold and suffix-shape reporting
are now explicit; no numerical gate is relaxed. Latest suite126 discovered,
124 pass,two dependency skips;29 targeted tooling tests pass. Marlin experts
do not advertise native batch-invariance support; no compatibility guard was
bypassed. Serving unchanged.

Completed full-model diagnostic: `1fa7c4cfd4e8`, started06:20:37 UTC,
root `/tmp/flash-session-gpu-20260921-CGIsZ0`, k3s evidence
`launch-full-tp4-stable-ties-c2.json`. TP4/PP1 Q8 MTP4, C1/C2 native-prefix
hot/cold screen, deterministic expert/QSA diagnostics and target chunk tracing.
Exited1/no OOM at06:39:14 UTC. C1 fresh/cold128-token outputs and logprobs
match, and all48 target layers/four prefill chunks match on all ranks.
C1 hot diverges at47. C2 fresh repeats match; hot/cold differ from their
references at4/94, with the first hot/cold pair also differing from each other
at6. Both cold phases load141,430,784 bytes; all C2 phases reach two running
requests. Full correctness remains failed. K3s evidence:
`screen-full-tp4-stable-ties-c2.json`, `full-tp4-stable-ties-c2.tail.log`.
Serving GPU4--7 identity/start remain unchanged.

Next control: `--prefix-prefill-budget 1024 --trace-prefix-shapes`. Both
scheduling/input budgets are bounded together; other cache/MTP settings stay
unchanged. The trace captures only first-target-layer packed position ranges
for each named phase, excludes pressure traffic and removes its hook afterward.
These ranges are not request-ID assignments. Local tooling tests27 pass.
This is diagnostic-only; no production budget or performance change.

Launched above: `--divergent-prefixes` extends the C2 model screen to distinct
private prompt tails after at least6608 shared tokens. Per-branch prompt hashes
are retained, and uncached/hot/cold comparisons keep the same branch order.
Thirty-one local full-runner tooling tests pass; branch result is recorded above.
The local contract suite previously passed119 tests with two dependency skips.

Deferred-free audit: existing native block ownership routing already returns
flattened RAM/GPU lists to their correct pools. No engine fix is needed for
that path. A new regression exercises the actual native scheduler fence,
colliding pool-local IDs, restored capacity and idle TTL starting only after
the fence. CPU-only trial `8877a85e3a71`, root
`/tmp/flash-host-deferred-20260922-KpR7LF`; k3s evidence
`launch-host-deferred-free.json`. Passed one test,64 deselected,28.79s;
exit0/no OOM at06:31:01 UTC. Log: `host-deferred-free.log`; no GPUs exposed.

Completed matched-shape diagnostic: `914b1bc34352`, root
`/tmp/flash-reduced-mtp-20260922-hwuIeG`, k3s evidence
`launch-reduced-tp4-matched-prefill.json`. Two new skip-prefix references use
the native scheduler with a6608 boundary matching the hot suffix split. The
original strict verdict remains unchanged. Sixteen reduced harness tests
pass. Full-runner tie-control support passes23 CPU tests; not yet launched.
Exited1/no OOM at06:16:54 UTC. Four k3s reports
`reduced-tp4-matched-prefill-rank0.json` through `rank3.json` show exact hot
suffix hidden rows/final logits against repeated uncached split references
(zero cached tokens). Cold matches the unsplit reference; measured prefix
bytes are exact. The cross-shape hot gate remains failed, not relaxed. This
isolates measured reduced divergence to selection and prefill geometry but
does not qualify full-model/concurrent or uninstrumented execution. Latest
tests:17 reduced harness and24 full-runner tooling, including short score-width
padding for the deterministic-selection diagnostic.

Completed tie diagnostic: `84f5d7fffda0`, root
`/tmp/flash-reduced-mtp-20260922-1oThyK`, k3s evidence
`launch-reduced-tp4-full-prefill-stable-ties.json`. Stable score ties choose
logical block indices before canonical ordering. Per-chunk score/visibility/
selection hashes distinguish membership changes from attention arithmetic.
This is diagnostic only; fifteen local harness tests pass.
Exited1/no OOM at06:09:08 UTC. Four k3s reports
`reduced-tp4-full-prefill-ties-rank0.json` through `rank3.json` establish exact
uncached target/draft chunks, exact measured hot/cold prefix bytes, and exact
cold suffix hidden rows/final logits. Cold hits5664 and loads10,689,536 bytes
per rank. Hot final output remains different with944+492-token chunks instead
of1436. This isolates a remaining shape/state comparison; full-model cache
correctness remains unqualified. Serving has not changed.

Completed canonical-order control: `5e83baeca7c5`, started2026-09-22T05:53:15 UTC,
root `/tmp/flash-reduced-mtp-20260922-CHQpPq`, k3s launch evidence
`launch-reduced-tp4-full-prefill-stable-qsa.json`. Only the native selected
block order changes; membership and invalid tails are preserved. This remains
a diagnostic, not a serving/kernel promotion. CPU cold tier1GiB and the
all-chunk equality gate are enabled. Exited1/no OOM at05:58:55 UTC. Four k3s
reports `reduced-tp4-full-prefill-stable-rank0.json` through `rank3.json` show
exact final uncached/cold rows but differing intermediate uncached chunks and
hot final rows. Hot uses944+492-token chunks; cold uses1436, both after a5664
hit. Cold loads10,689,536 bytes/rank. Target KV/index and draft main-KV prefix
bytes match; draft compressed-index bytes differ. Numerical drift and chunk
shape differences remain confounders, not evidence that restore is cleared.
The strict correctness gate fails; serving remains unchanged.

Completed native-audit run: `ffd7fdc71fe7`, root
`/tmp/flash-reduced-mtp-20260922-npklUm`, launch/source evidence in k3s:
`launch-reduced-tp4-full-prefill-native-audit.json`. The fixture bound is fixed;
native selection replay auditing is enabled, canonical QSA ordering disabled.
Exited1/no OOM at05:50:42 UTC. Four k3s reports:
`reduced-tp4-full-prefill-native-rank0.json` through `rank3.json` (same prefix).
All ranks have matching attention inputs and historical target KV/index bytes,
but different attention outputs from chunk2. Eight fixed-score top-k repeats
change ordering in1724--1725/1888 rows with zero changed selected sets. Thus
ordering variation is confirmed; its causal effect still needs the canonical
control. Uncached target/MTP/logit max differences:0.0003662109375/0.0625/0.25.
The small one-target control passes. Long hot hits5664, cold misses because
the512MiB global CPU tier cannot retain all pressure-test stores. Next long
test uses1GiB CPU cold storage and canonical block ordering; resident main-KV
budget stays unchanged. Success now also requires all uncached target/draft
chunk hashes to match;14 local tests/Ruff pass. Serving unchanged.

Active reduced follow-up: `4316c538b353`, started2026-09-22T05:37:18 UTC,
root `/tmp/flash-reduced-mtp-20260922-vNj8O1`, launch evidence in k3s
`launch-reduced-tp4-full-prefill-shape.json`. The model's indexer budget is2048
tokens/CR4. Prior1920-token reduced tests did not cross it; the full-model
first mismatch does. The7100-token reduced run tests that hypothesis, without
assuming a particular kernel is responsible. **Exited1/no OOM at05:42:17 UTC
before long-prefix execution**, because the reused metadata fixture still
bounded query count to512. Four reports retained in k3s as
`reduced-tp4-full-prefill-bound-rank0.json` through `rank3.json` (same prefix).
The fixture now permits explicit2048 within actual batch capacity, retaining
default512; thirteen tests/Ruff pass. The next native run audits eight repeated
fixed-input top-k calls, reporting order/set variation and restoring the
original output. Optional canonical selection remains disabled in that rerun.
No inference or serving change.

Full-runner validation is in progress: isolated TP4/PP1 Q8 MTP4 container
`5ef3a6d5e684` started 2026-09-22T05:03:56 UTC on development GPUs0--3.
The native AsyncLLM screen covers C1 and C2, 7100-token prompts / 128 greedy
output tokens under an 8192-token cap, native prefix matching and actual cold
offload with diagnostic stable Marlin ordering. C2 additionally requires
observed simultaneous running requests, exact matched-concurrency uncached/
hot/cold token comparisons and actual cold bytes; serialized submission is
not a pass. New TP4 launcher flags remove inherited PP placement while keeping
RAM KV and enabling the validated unpadded K32/Q8 gates. Nineteen CPU
controller/comparator tests passed before launch. **Completed at05:17:20 UTC,
exit1 / no OOM, all76 requests finish but exactness fails.** C1 uncached/hot/
cold first differ at output index6. C2 uncached pairs differ at6/33, hot94/65,
cold94/6. Uncached first-token logprobs already differ before token choices
diverge. All hot/cold hits are5664 tokens; each cold phase loads141,430,784
bytes; all C2 phases observe two running requests. Stable ordering executes
43,536 times/rank. This proves actual reuse/transfer/concurrency, not numerical
correctness, and the uncached runs still use our patched RAM-KV engine.
Evidence in k3s: `screen-full-tp4-native-prefix-c2.json`,
`full-tp4-native-prefix-c2.log`,
`launch-full-tp4-native-prefix-c2.json`, remote root
`/tmp/flash-session-gpu-20260921-GLIzH3`. Serving and its context limit remain
unchanged; this bounded diagnostic cannot qualify 240K or all workloads.

The next diagnostic traces all prefill chunks by explicit request and token
positions, including later chunks omitted by the previous trace. It stores
hashes only, marks different schedules unmatched, and excludes decode beyond
the prompt. CPU readback perturbs timing; no race-free claim follows from a
passing traced run. Two new regressions pass (all21 controller tests, Ruff).
No inference source or correctness threshold is changed.

Traced C1 follow-up `3b19fd20069a` started2026-09-22T05:20:14 UTC on the
development quartet; remote root `/tmp/flash-session-gpu-20260921-ipIQiu`.
Launch/source hashes: `launch-full-tp4-prompt-chunks.json` in k3s evidence.
The full TP4/Q8 MTP4 model is loading with the same native-offload settings.
Results pending; serving ID/start remain unchanged.

**Completed05:34:18 UTC, exit1/no OOM.** All four ranks cover48 layers and
identical1888/1888/1888/1436-token prefill chunks. First chunks match in all
layers; layers0--2 match in every chunk. Layer3 (first QSA block) first differs
in chunk2 despite identical inputs; later layer3 chunks also differ. This
localizes the block, not yet attention/HC/MLP or a specific implementation.
Uncached token mismatch is82, hot128/128 exact, cold mismatch123; cold hit5664
and141,430,784 load bytes. Tracing changes timing, not correctness thresholds.
Evidence: `screen-full-tp4-prompt-chunks.json`, `full-tp4-prompt-chunks.log`.
Reduced follow-up in k3s matches7100-token/maxchunk2048 geometry and96MiB arena
with attention/MLP input/output hashes; host pool256/max512MiB preserves RAM
control prefixes during pressure. Existing one-target controls are unchanged.
Eleven harness tests/Ruff pass; no inference source or serving changes.

Two-real-target-layer native prefix test `424f624c505e` completed successfully
on all four TP4 ranks (2026-09-22T04:51:43--04:56:56 UTC, exit 0 / no OOM).
Uncached-repeat/hot/cold target and MTP hidden/logits, including full-chunk
hidden hashes, are exact. Target/draft host prefix IDs and bytes match after
a real 944-token cold hit and 2,466,816-byte restore/rank (GDN plus separate
target/draft compressed-index groups). Native geometry is BLNHC, block944,
32 GPU / 128 host blocks, packed stride822272; transfer isolation checks pass.
Evidence in k3s: `reduced-tp4-two-target-alias-rank0.json` through `rank3.json`
(same prefix), `launch-reduced-tp4-two-target-alias.json`. Diagnostic stable
Marlin ordering remains enabled; prefix scheduling is serial. Real layer0
feeds real layer3 directly, omitting layers1/2; this is not a full-backbone
accuracy or concurrent native speculative-verification test. Serving stays
unchanged. Native multi-request model and full-runner/e2e gates remain open.

Expanded packing/native-prefix suites pass all 214 tests, including eight
new shared-suffix/branch-lifetime cases (`64d755ae366a`, exit 0, 59.99s).
Evidence in k3s: `host-complete-suite.log`, `launch-host-complete-suite.json`.
The first real two-target-layer harness run `f8be47c846cf` failed only its
layer3 weight-name verification: the reused helper normalized layer0 aliases
only. The check now recognizes the exact layer3 MoERunner reporting alias;
the corrected run is `424f624c505e`. These are harness edits, not engine or
weight changes, and two-target-layer model correctness is not yet established.

Native-manager branch lifetime cases pass for two and six concurrent owners
(`b8a3730f49db`, exit 0, 26.45s): shared prefix refcounts, distinct private
suffixes, refusal of an extra branch at capacity, expiry of idle suffixes
without retiring active shared pages, and recovery of all host capacity after
release. The controlled clock exercises the actual 3600-second host TTL.
Evidence in k3s: `host-branches-regression.log`, `launch-host-branches.json`.
These are ownership/lifecycle tests, not model-backed concurrent decoding.

The retained-hit shared-suffix audit passes without an engine change:
`KVCacheManager.allocate_slots` already trims retained groups to the actual
local/external resume boundary before admission and allocation. Six new cases
in `test_contiguous_kv_packing.py` pin the original three RAM pages, evict GPU
hits, and check 0/16/32-token resume boundaries with/without a free private
page. They verify private suffix ownership and capacity refusal; all pass
(`e0a7e611ba6a`, exit 0, 26.72s). Evidence in k3s:
`host-suffix-regression.log` and `launch-host-suffix-before.json`.

Reduced native scheduler run `d04029a5f1ee` passes on all four ranks with the
existing diagnostic-only stable Marlin layout hook (66 calls/rank). It ran
2026-09-22T04:33:00--04:37:42 UTC, exit 0 / no OOM. Uncached-repeat, hot and
cold final target hidden, MTP hidden and logits are bit-exact (zero maximum
difference). Retained host IDs and main-KV/compressed-index prefix bytes match
after an actual 944-token / 1,644,544-byte cold load per rank. Three aligned
cases and 60 controlled MTP rejection/replay cases/rank pass too. Evidence in
k3s: `reduced-tp4-native-prefix-stable-rank0.json` through `rank3.json` (same
prefix), plus `launch-reduced-tp4-native-prefix-stable.json`.
This separates the resident-host adoption defect from native expert-ordering
variation in the reduced test. Stable ordering is a correctness diagnostic,
not a promoted performance patch. Native concurrent speculative verification,
target-QSA plus GDN/MTP integration, long contexts and full-model serving
remain unqualified. Before expanding, audit that retained host hits beyond
the actual restored boundary are not reused as writable shared suffix pages.
Serving remains unchanged; all 65 current manifest source entries match.

Weighted coordinator retest `e54b068b39c6` verifies resident-host adoption in
the reduced TP4 scheduler probe: all four ranks reuse host block 1, hot/cold
944-token main-KV and compressed-index prefix hashes match, pre-forward GDN
states match, and each rank actually loads 1,644,544 bytes. Cold MTP
hidden/logit maximum differences drop from 1.16796875 / 8.453125 to
0.03125 / 0.125. Uncached repeats still differ by 0.03125 / 0.15625, so
strict output equality remains failed. Evidence in k3s:
`reduced-tp4-native-prefix-host-hit-rank0.json` through `rank3.json` (same
prefix), plus the matching launch JSON. The launch guard first rejected
manifest bookkeeping: the updated hash now belongs to `post_snapshot_sources`,
the historical upstream hash is restored and verified against the recorded
commit, and the frozen diagnostic snapshot retains its `1c6f1cae...` hash.
Follow-up `d04029a5f1ee` uses the existing diagnostic-only stable Marlin layout
hook; this is not a kernel promotion or a relaxed correctness gate.

Confirmed resident-host adoption regression: a device-first cache miss reduced
the subsequent retained-host lookup to zero, although the connector's
independent host-bound lookup found resident KV. Native allocation consequently
received no host hits and chose unwritten pages. The coordinator now queries
retained groups against the original maximum prefix, retaining those pages
for external-hit allocation while keeping the reconciled local hit bounded by
every group. Four real-manager tests cover both group orders and target/draft
host groups; two fail before the change, all four pass after it. Combined
packing/prefix suites pass 206 tests (`b48b3ff624c5`, exit 0), after correcting
four relative-import failures in the initial test mount. Evidence in k3s:
`host-prefix-regression-{before,fixed,suite}.log` and matching launch JSON files.
Coordinator SHA256:
`1a98b24debba97edd65bfbbc728d8a5a758f6b5d725a426a50eb4e2c252288ee`.
The manifest is updated; the weighted test uses a new verified read-only
overlay, not an in-place snapshot edit. Weighted correctness and the separate
uncached chunk/MTP repeatability issue remain unresolved. Serving unchanged.

Scheduler-driven cold load `c64a54bf1407` completed at
2026-09-22T04:08:39 UTC, exit 1 / no OOM. Cold hit: 944 tokens and 1,644,544
load bytes/rank. GDN state hashes before/after corresponding chunks and final
target hidden match reference. Full target chunk hidden is not repeatable
after chunk 512, and MTP differs even between uncached controls. Cold MTP
hidden/logit max differences are 1.16796875 / 8.453125, versus uncached
0.0234375 / 0.15625. Cold host prefix maps to block 26 instead of original
block 1; native resident-hit discovery/adoption is under investigation.
Evidence in k3s: `reduced-tp4-native-prefix-batch-rank0.json` through `rank3.json`
(same prefix). Transfer/replay passes do not qualify this failed scheduler
integration. Serving remains unchanged.

Native InputBatch integration follow-up: `3910a558dd65` exited 1 before a
complete scheduler probe. Repeated remove/add without metadata refresh
violated its native batch-update ordering. The reduced harness now retains
requests across chunks, updates native block rows and calls `refresh_metadata`.
Successor `c64a54bf1407` completed on GPUs 0--3, root
`/tmp/flash-reduced-mtp-20260922-MfkKO2`. Evidence in k3s:
`launch-reduced-tp4-native-prefix-batch.json` and failed
`reduced-tp4-native-prefix-preprocess-rank0.json` through `rank3.json` (same
prefix). Sixteen helper tests and Ruff pass; engine sources and serving stay
unchanged. This does not qualify hot/cold correctness.

First scheduler-driven run `605f15c3ab25` completed at
2026-09-22T03:55:07 UTC, exit 1 / no OOM. Uncached final target hidden matches,
but draft hidden/logits differ (max 0.04296875 / 0.21875). Hot and pressure
repeats hit 944 tokens but differ; zero load bytes means pressure did not
exercise cold restoration. The harness omitted native `preprocess_mamba`.
Successor `3910a558dd65` reuses that helper with native InputBatch,
CachedRequestState and copy buffers, adds state/full-hidden hashes, and uses
ten pressure requests. This is still reduced integration qualification, not
proof of an upstream/cache-transfer bug. Evidence in k3s:
`reduced-tp4-native-prefix-context-rank0.json` through `rank3.json` (same prefix)
and `launch-reduced-tp4-native-prefix-preprocess.json`. No engine-source or
serving change.

Scheduler prefix probe setup correction: `d2a699713dda` exited 1 at
2026-09-22T03:44:39 UTC. Its first target forward selected the draft-only
forward registry and raised a missing-target-layer `KeyError`. The harness
now uses each model's own construction config for its forward context.
This is not evidence of cache corruption. Sixteen local tests and Ruff pass;
successor `605f15c3ab25` completed on GPUs 0--3, with no engine-source or
serving changes. Evidence in k3s: `reduced-tp4-native-prefix-context-failure-rank0.json`
and `launch-reduced-tp4-native-prefix-context.json` in the benchmark directory.

Native-bound forward follow-up: `a6b829702d72` completed at
2026-09-22T03:33:02 UTC, exit 0 / no OOM. TP4/PP1 target GDN layer plus Q8 MTP
now run directly on native planned/allocated/bound cache views. All four ranks
pass 60 rejection/replay cases and three aligned cases. Aligned final state
bytes, draft hidden/logits and accepted tokens match; transfers preserve other
GPU pages and resident RAM KV. Shared GPU groups have disjoint diagnostic IDs
and only owned raw-scratch destination pages are cleared. Rows 1/2/4 transfer
1,644,544 / 3,289,088 / 6,578,176 bytes per rank via native packed-page copies.
Evidence: k3s `reduced-tp4-native-bound-rank0.json` through `rank3.json` (same
prefix), under `docs/benchmarks/flash-session-swap-20260921/`.

The preceding transfer-only probe copied CacheConfig, while QSA side owners
retained its original reference: host pages were 944 tokens but compressed
specs retained 4048. The new run uses the shared native config, resolving 944
consistently, and derives slot mappings from that size. No engine-source or
serving changes were necessary. Scheduler-driven prefix ownership/eviction,
batched target verification, long contexts and full-model correctness remain
unqualified; committed target tokens are still evaluated sequentially here.

Native allocator/connector follow-up: `d9f884cd582c` completed at
2026-09-22T03:23:56 UTC, exit 0 / no OOM. All four TP4 ranks pass transfer-only
tests using real GDN/draft-QSA specs, native BLNHC planning/allocation and native
CPU-offload registration. Auto geometry is 944 tokens/page, 32 GPU/32 host
blocks, packed stride 822,272 bytes. Each eligible GDN/compressed-index group
stores/restores 822,272 bytes to a different block; fixed-seed offset-varying
tensors match, with non-destination pages and RAM KV unchanged. The existing
60 rejection/replay cases per rank also pass. Evidence in k3s:
`reduced-tp4-native-layout-rank0.json` through `rank3.json` (same prefix), under
`docs/benchmarks/flash-session-swap-20260921/`. Native-bound model forward,
batched verification and full scheduler correctness remain unqualified;
the output matrix still uses its independent fixture allocations. No engine
source or serving change was needed for this probe.

Latest reduced diagnostic (2026-09-22): development now uses TP4/PP1, one real
target GDN layer plus the native Q8 MTP head with four draft steps. Container
`f1ad5b846a79` completed 03:06:49--03:10:02 UTC, exit 0 / no OOM. Each of four
ranks passes 60 rejection/replay cases (rows 1/2/4, accepted drafts 0--4,
prefix boundaries 12--15) plus three aligned restore cases. Real draft tokens
use native greedy rejection with controlled target decisions, not measured
acceptance rates. Cold proposals/accepted tokens/redraft outputs are exact
against hot execution; hot execution matches a no-rejected-writes control.
Aligned final persistent-state bytes and non-destination isolation pass, and
transfers do not modify resident RAM KV. Native CPUOffloadingWorker is reused.
The harness token-layout correction needed no engine changes. Evidence and
scripts are in the k3s checkout under `docs/benchmarks/flash-session-swap-20260921/`
(`reduced-tp4-mtp-rejection-rank0.json` through `rank3.json`, same prefix) and
`infra/inference/experiment_flash_layer0_mtp_cold.py`.

These explicit layer-state layouts do not qualify native planner/connector
registration, batched target verification, prefix lifecycle or full serving.
The full-model `93d96274e7a9` run below is terminal: all 36 requests finished;
hot/cold hits cover 4048 tokens and cold loads 125,079,552 bytes, but uncached,
hot and cold comparisons fail exact-output qualification. Initial 2048-token
prefill input/output hashes match at every layer across uncached references;
later chunks/decode remain unresolved. Production is unchanged.

The user reiterated: reuse vLLM offload and patch only model-compatibility gaps
or proven bugs, not a replacement session cache. Weighted `4bb0464e3b3a`
exited 1 / no OOM at 2026-09-22T02:05:53 UTC before inference. CUDA error
state was zero immediately before mmap registration on every rank; ranks
0/1/2 then failed registration with code 1. This is not an inherited reported
CUDA error. The existing `CPUOffloadingWorker(mmap_region=None)` already
supports private pinned tensors with the same transfer handlers. The opt-in
`kv_connector_extra_config.use_shared_memory=false` now selects that native
path; default true preserves existing behavior. Replicated-layout dedup is
disabled through the existing native guard when buffers are private.
Native GPU/factory qualification passes 89 tests / 11 deselected, including
actual transfers with shared/private buffers, chunking, both directions and
dedup-selection guards. Evidence: k3s `native-private-cold-qualified.log`.
This is not a full-model clearance. Successor `93d96274e7a9` is running with
private cold tensors and bounded ordinary-request first-prefill tracing,
snapshot `/tmp/flash-session-gpu-20260921-Bqxl9B`. No weights,
production services or native cache/transfer algorithms changed for this option.

Complete native scheduler/config suite: **272 passed**, CPU-only container
`3a2fba459a0e`, exit 0 / no OOM. The prior offline fixture failures were
resolved by mounting the public `facebook/opt-125m` configuration, without
weights or altered tests. The container remains network-isolated. Evidence:
k3s `native-cold-bound-complete.log`. This strengthens connector regression
coverage but does not establish weighted cache correctness. The full-model
successor below failed during startup as recorded above.

Native cold-hit alignment fix (2026-09-22): `_lookup_complete_chunks` must
reserve the logits token from the prompt length before applying an earlier
resident-prefix cap. Otherwise a cap on a Mamba checkpoint loses a whole
block. Six regression cases reproduce three failures before the fix and all
pass afterward, including local-prefix offsets and last-prompt-token controls.
The broader offline suite has 181 passes / 91 failures caused by unavailable
`facebook/opt-125m` configuration; it is not a full-suite pass. Source manifest
now preserves 65 engine sources / 78 export files. Local CPU helpers remain
111 passed / two dependency skips; Ruff and diff checks pass.
The fixture-free selection passes 82 native tests / 190 deselected; see k3s
`native-cold-bound-selected.log`. Weighted successor `4bb0464e3b3a` ran
on development GPUs only, snapshot `/tmp/flash-session-gpu-20260921-1cTOL7`;
cold restore and output repeatability still require real-model evidence.

Latest completed weighted screen: `55c589ec9232` exited 1 / no OOM at
2026-09-22T01:45:18 UTC. All four host registrations succeeded with zero
CUDA error state before/after. All 36 requests reported finite logprobs, and
stable expert sorting was exercised on all workers (22724/22724/19228/19228
calls). Nevertheless the uncached references and hot hit differ at output
index 6. The hot hit covers 4048 tokens; the final repeat hits zero tokens
and loads zero bytes. Total store bytes: 4,127,625,216. Neither numerical
repeatability nor actual cold restoration is qualified. The k3s evidence is
`screen-native-prefix-registration-trace.json`. Investigating the connector's
aligned resident-prefix limit before launching another model screen.
Production is unchanged.

Full-model successor `9d34280236d9` started at
2026-09-22T01:17:43 UTC (development GPUs 0--3 only). Four-rank no-weight mmap
registration passes, so the weighted run uses CUDA_LAUNCH_BLOCKING=1 to localize
errors plus fail-closed registration. A named worker RPC installs diagnostic
stable token ordering inside native expert groups; it preserves routing,
padding/counts and restores both sorter entrypoints at teardown. Per-rank call
counts must prove use. This is not a serving or performance optimization.
No MTP, native kernel fallbacks, ordinary native prefix cache and existing
strict exact-output/real-cold-load gates. CPU helper suite: 112 discovered /
110 passed / two dependency skips. It exited 1 / no OOM at 01:23:40 UTC,
with rank 0 cudaHostRegister code 1 for 8,588,795,904 bytes before any request.
Stable-layout hooks were not installed because engine startup failed.
Four-rank synthetic controls with 22 GiB device / 32 GiB pinned host per rank
pass, with and without strided UVA views. Thus simple capacity/UVA-view setup
does not reproduce the failure. Next weighted diagnostic adds a non-clearing
cudaPeekAtLastError check before/after registration. Latest CPU helpers:
113 discovered / 111 passed / two dependency skips. Evidence is in k3s
`launch-native-prefix-stable-blocking.json`,
`native-prefix-stable-registration-failure.log` and the `native-registration-*`
logs. Serving is unchanged; no numerical/cold-restoration verdict.
Registration-trace successor `55c589ec9232` ran on development
GPUs 0--3, immutable snapshot `/tmp/flash-session-gpu-20260921-RrrYv0`.
It is terminal; its failed qualification is recorded above.
Launch evidence: k3s `launch-native-prefix-registration-trace.json`.

Latest diagnostics (2026-09-22): no-weight two-process registration succeeds for
the same 8,588,795,904-byte mmap, both whole and chunked. Size alone does not
reproduce the weighted startup failure. Native pinning now raises immediately
on failure: the Triton load path needs registered UVA memory, not a speculative
pageable-DMA fallback. Three injected registration tests and 55 native
canonical-copy/shared-region tests pass. No chunking change was made. Source
manifest now preserves 64 engine sources / 77 exported files; the additional
gpu_worker.py baseline matches both HEAD and the pinned image. CPU helper suite
110 discovered / 108 passed / two dependency skips with the documented path.
An isolated real-layer TP2 diagnostic also reproduces finite output variation
at 128 rows with fixed inputs/routes and no KV/MTP; gate/up already varies.
The native-layout replay follow-up passed at 01:10:44 UTC: holding its first
sorted-token layout fixed makes all 16 repeats exact at 1/4/22/128 rows on both
ranks. At 128 rows all 15 native repeat layouts differ; router choices and
padding counts remain fixed. This identifies layout ordering as one source of
numerical variation, not a proof of cross-batch invariance or cache integrity.
No engine sorter/GEMM change was made. Reports in the k3s benchmark directory:
`expert-layout-rank0.json`, `expert-layout-rank1.json`, `native-expert-layout.log`.
Neither result qualifies full-model cold restoration or resolves prior NaNs.
All diagnostic containers are terminal and serving is unchanged.

Latest capacity fix: direct-host plans can qualify native whole-block GPU
offload for block-outer layouts. The worker verifies allocation identity,
size,stride and per-layer bounds before constructing one transfer region;
RAM/scratch groups remain excluded and empty local PP groups have no refs.
Other layouts retain conservative sizing. Common PP slot width remains the
maximum worker bound. A regression reproduced65536 versus16384 physical bytes.
Native selected suite150 passes, including GPU DMA byte-exact restoration into
different blocks, unchanged neighbouring blocks/RAM sentinel and unsafe-view
rejection. This does not qualify full-model cold hits or numerical repeatability.
The no-MTP weighted successor `7bbdd4c8b525` ran on development GPUs0--3,
2026-09-22T00:44:20--00:50:15UTC,exit1/no OOM before requests. Its8.59GB mmap
selected one packed CPU tensor per worker, but cudaHostRegister returned code1
on ranks0/1/2; native warning-only handling continued into a warmup CUDA error.
No cold restoration was tested. A no-weight registration probe and separate
one-layer expert repeatability test are in progress. Evidence in k3s:
`native-packed-cold-qualified.log` and `launch-native-prefix-packed-no-mtp.json`
under `docs/benchmarks/flash-session-swap-20260921/`. Source manifest now records
63 engine files/76 exported files. No production changes.

Latest isolation:12 GPU QSA reference tests pass with NaN-poisoned unused
pages, direct/staged attention and device/pinned-host backing (seed173,
rows1/3/32,two histories17/7101). A genuinely selected poisoned value fails
the finiteness check as intended. No attention kernel changes. The weighted
hot-prefix mismatch and later NaN remain unresolved. A no-MTP ordinary-prefix
control using the corrected shared PP offload geometry ran on development
GPUs0--3 at2026-09-22T00:22:12--00:30:46UTC,exit1/no OOM. All36 requests
had finite logprobs, but uncached references first differ at token82 and the
4048-token hot hit at token6. The final repeat was a miss (zero cached tokens,
zero load bytes), not a cold-restoration test. No-MTP also changes the block
size from3504 to4048. MTP is not required for output variability; absence of
NaNs in one no-MTP run does not establish the earlier NaN's cause. Corrected
PP geometry agreement/worker initialization ran; payload restoration remains
unqualified. Conservative aliased-layer capacity accounting requires inspection
before the next cold-pressure test. Serving is unchanged. Evidence is in k3s
`docs/benchmarks/flash-session-swap-20260921/`, files
`native-qsa-poison-gpu.log`, `launch-native-prefix-no-mtp.json` and
`screen-native-prefix-no-mtp.json`.

### Authoritative integration direction (2026-09-21 clarification)

- Keep vLLM's chained block hashes and native prefix matching. Do not introduce
  radix trees, explicit session IDs, or require an HTTP stream to remain open.
- Reuse native allocation, references, copy-on-write, hybrid checkpoint
  convergence, connector scheduling, CPU transfer workers, and LRU policy.
- A new request receives fresh sampling/request state. Only reusable model
  prefix state crosses request boundaries; do not restore the old request's
  sampler, penalties, output counters, or streaming queue.
- Preserve QSA KV in its existing RAM tier, without a second RAM-to-RAM copy.
  RAM pages currently use GPU physical block IDs; retain them under independent
  native block-pool ownership before releasing/reusing GPU blocks. A pointer to
  the RAM location is not an ownership guarantee. Native CPU offload should
  transfer the remaining GPU state, not duplicate already resident QSA pages.
- Add the requested 3600-second cold-idle expiry and hit/miss visibility through
  the native cache lifecycle. Never expire or recycle an in-flight transfer.
- A prefix hit is usable only at a boundary supported by every state group.
  Reconstruct speculative scratch through native replay where supported; do
  not transplant request-specific draft state from an unrelated request.

### Source-verified native integration gaps

1. `OffloadingConnectorWorker.register_kv_caches` receives the pre-binding
   allocation dictionary. QSA's `bind_kv_cache` allocates `_qsa_host_kv` and
   replaces the layer's `kv_cache` with its UVA view, but the dictionary retains
   the GPU placeholder. The native registration audit measures 7008 bytes versus
   3588096 actual host bytes per QSA block in the current TP2 geometry.
2. Native CPU capacity derives from `worker_kv_bytes_per_block`. Exclude resident
   QSA pages/placeholders from transfer accounting, and budget their independent
   host pool separately. Keep scheduler/worker group IDs and block mappings
   consistent. Host pool exhaustion must refuse admission, not supply null
   pages: unlike HiSparse, direct RAM QSA has no retained GPU backing fallback.
3. The current run uses prefix caching disabled and effective Mamba `none`.
   Native offloading asserts compatible block/hash granularity. Qualification
   must exercise native prefix caching and aligned checkpoints, not waive this
   assertion with custom whole-session lookup.
4. Native Mamba prefix/COW/offload logic already exists. Reuse it. The raw QSA
   ring's `CircularBufferSpec` is explicitly non-prefix-cacheable and its
   manager does not publish cached blocks. Its restoration/reconstruction at
   native replay boundaries must be proven before exposing a cache hit; simply
   changing the flag or ignoring the ring is not a validated solution.
5. `CPUOffloadingManager` already owns pending/ready lookup results, transfer
   ref-counts, allocation, LRU/ARC policy hooks and eviction events. Extend
   missing TTL behavior there or through its supported policy interface; do
   not carry the separate session coordinator into the production path.

Acceptance requires ordinary independent requests, matching/branching prefixes,
hot and cold hits, physical-block reuse, misses/expiry, and changed sampling
parameters, in addition to the existing byte-integrity and failure tests.
The existing streaming screens do not establish this acceptance criterion.

### Independent-pool admission implementation

The coordinator now aggregates allocation requirements by the native owning
`BlockPool`. It returns only GPU demand to the existing scheduler gate, and
refuses admission if any separate pool lacks capacity. Multiple groups sharing
one RAM pool are checked together. HiSparse external loads report their actual
host demand; its GPU-computed best-effort allocation behavior stays unchanged.

CPU tests use native managers and verify independent GPU/RAM accounting, RAM
exhaustion before partial allocation, GPU exhaustion without consuming RAM,
and HiSparse demand reporting.

The next step adds the internal `DIRECT_HOST` group role and
`direct_host_num_blocks` capacity. It binds strict-capacity full-attention
managers to vLLM's `SharedEventQueueBlockPool`, preserving native prefix hashes,
LRU references, events and independent physical block ownership. Native CPU
offload excludes these resident groups while keeping them as prefix-hit bounds.
Single-group lookup and scheduler capacity/rank agreement checks use the host
pool. Direct-host IDs are excluded from generic GPU zeroing; full-block prefix
hits avoid host CoW, and an unexpected host CoW queue fails closed.

The existing `KVCacheBlock.pool` and `BlockPool.free_blocks` already route mixed
deferred frees to their owners. No replacement release protocol was added.
CPU tests demonstrate RAM-prefix metadata remains valid after GPU block reuse,
becomes a miss after host eviction, and supports a fresh request with different
sampling settings. An actual scheduler fence keeps both pools pinned until the
last in-flight step is processed. Native suite21 unique tests pass (34 executed,
13 repeated because unittest also discovers imported test classes), exit0/no OOM
in container `c1b8de1a4029`; local suite104 pass/two dependency skips.

The metadata changes alone are not a serving-ready feature. The later opt-in
startup integration below selects `DIRECT_HOST`; full-model reuse remains
unqualified.

### Native resident-host tensor binding

`allocate_kv_cache` now uses the existing native layout/view allocator separately
for GPU and pinned RAM backings. An explicit per-worker host byte ceiling is
required before allocation, including conservative pinned-allocator rounding.
QSA binds the pinned BF16 host view through UVA without making a second copy or
allocating a GPU placeholder. The cache dictionary keeps CPU views, so generic
GPU block copies exclude them. Both model runners pass host group IDs to native
GPU zeroing, which must not touch these independent pages. The historical
session-copy protocol rejects this layout rather than applying GPU IDs to RAM.

GPU0 probe `ac3924b85b3a` passes raw-byte checks with seven host blocks versus
four GPU blocks, two strided host-layer views sharing50233344 bytes, and a QSA
write/read at host block5 (outside the GPU block range). Native QSA scatter
writes match the input bytes. GPU zeroing, block copying and a native CPU
offload store/load leave every host byte unchanged. The GPU-state transfer is
28032 bytes; no resident QSA pages enter the transfer. Sparse QSA attention
matches a VRAM reference byte-for-byte. Scope: one real QSA cache path with
synthetic inputs, not a full weighted layer or model. Seed173, BF16/head256,
3504-token pages. The initial two attempts failed on probe helper/import setup,
before this validation ran; those failures are retained.

Native packing regression suite23 pass (CPU,14 upstream deprecation warnings).
Local suite107 discovered/105 pass/two native-dependency skips. Targeted Ruff
passes; `qsa.py` retains the same three pre-existing line-length findings as
HEAD, with no new findings. Export manifest57 sources/70files.

### Resident-host pipeline layout planner (CPU-qualified)

`get_direct_host_kv_cache_configs` now takes explicit global host/device groups,
per-worker device budgets, one shared host block count, and a total RAM budget.
It reuses native packed tensor layout construction for both pools. Projection
preserves host ownership and transfer roles, including stages with no local
host layers. Device capacities converge independently without shrinking host
pages. The RAM ceiling includes each TP worker's separately rounded pinned
backing; workers without host layers reserve zero host bytes. Missing layers,
placeholder pages, inconsistent roles and insufficient host capacity fail
before allocation. The scheduler also combines draft-group flags across PP
stages instead of losing a flag when stage0 owns no draft layers.

Pinned-image CPU container `4991f85fa0b0` passes32 packing tests (23 existing,
nine new), exit0/no OOM. Tests cover unequal TP2/PP2 worker budgets, empty
stages, shared scheduler IDs, pinned-byte budget rounding, and actual native
CPU tensor views with disjoint layer regions and separate device/host backing.
This does not rerun pinned/UVA GPU binding or model inference. Local105 pass,
two dependency skips; targeted Ruff and diff checks pass. Evidence:
`k3s/docs/benchmarks/flash-session-swap-20260921/native-host-planner.log`.

The planner requires an explicit max_model_len (no automatic context fitting).
The opt-in startup integration below supplies real full-attention host groups.
No serving settings changed.

### Experimental model-startup selection

`additional_config.flash_next_direct_host_kv` requires integer
`num_blocks` (including the shared host null block) and `max_bytes` (total
rounded pinned allocations across all TP/PP workers). Omit the key to keep
legacy allocation unchanged. Enabling it requires `VLLM_QSA_KV_OFFLOAD=1`,
DP1, the hybrid cache manager and an explicit context limit. Eager execution
remains the tested GPU path. Optional boolean `allow_cudagraph` defaults false;
its experimental graph path additionally requires positive explicit GPU KV
bytes, compilation mode NONE and FULL_DECODE_ONLY (see current progress).
Automatic CUDA-graph profiling has not been integrated, and host groups
remain rejected by the GPU-only planner.

QSA then emits a `DirectHostAttentionSpec` with real BF16 page geometry, keeping
the configured token block size. Grouping strips the placement marker into
ordinary full-attention host groups. Target and MTP layers remain separate;
the MTP constructor marks its own attention modules explicitly, without name
heuristics. Native `get_kv_cache_configs` selects the independent-pool planner.
The regular model path retains its prior GPU-placeholder specs when disabled.

CPU pinned-image container `488772da04a9` passes40 tests (32 previous plus eight
startup tests), exit0/no OOM. Coverage includes the actual QSA spec method
(7008-byte placeholder versus3588096-byte real page at3504 tokens/head256),
native startup planning, target/draft separation, unchanged token block size,
invalid option/unsupported mode rejection and the GPU-only planner guard.
This is not a weighted model start or prefix-reuse qualification.

GPU0 container `ee58d839678c` uses QSA's spec method and native startup planning
to construct the earlier seven-host/four-device-block primitive. Raw-byte QSA
writes, independent-pool zero/copy/offload safety, no-copy UVA binding and
sparse attention equality versus VRAM all pass.50233344 resident host bytes,
28032 GPU-state transfer bytes,seed173. Exit0/no OOM. Initial attempt failed
before allocation on the probe's use of an init-disabled CacheConfig field;
the setup fix and both attempts are retained. No weighted model or scheduler
was exercised. Evidence `native-host-startup-gpu.json` in the k3s evidence tree.

### MTP retain/replay contract and raw-ring qualification

The user approved retaining historical target and MTP QSA KV in RAM, preserving
GPU-side compressed/index prefix history through native CPU offload, and
recomputing only the short boundary tail and scratch. This is not a full-history
MTP rebuild: historical draft inputs depend on target hidden states that are
not cheaply recoverable from target KV alone. Fresh requests retain native
sampling and speculative metadata; no old stream/sampler state is restored.

The raw ring remains non-prefix-cacheable. An explicit replay_alignment=4
qualifies only CR4/BF16 QSA rings in the direct-host experimental mode. Native
prefix alignment includes this constraint; native offload excludes qualified
scratch rings, not compressed index history. Unqualified rings retain their
legacy path, and qualified/unqualified specs cannot silently merge.

GPU0 test `c25b88a2f8cf` passed26 tests: retained versus relocated/NaN-poisoned
scratch and clean accepted-only references; depths0--4; boundaries16/3504;
two requests; text/M-RoPE positions;13 speculative steps including rejected
drafts; byte-exact accepted query, committed compressed state and final ring.
Negative controls at offsets1/2/3 detect unsafe mid-group rebuilds. Twenty FP8
cases skipped because native e4m3 compilation is unsupported on SM86; FP8 is
not qualified. An initial run hit exactly that compiler limitation (26 pass,
20 fail), not a BF16 mismatch. No model weights or end-to-end MTP were tested.
Native planning initially passed45 CPU tests; a stricter isolated ring guard
was then rerun successfully in the140-test suite below.

### Native CPU idle expiry

The native CPU manager now accepts optional `idle_ttl_seconds`. The direct-host
offload builder defaults it to3600 without mutating caller configuration;
legacy layouts retain the disabled default. Explicit positive finite values
override it. Native LRU remains the default policy; ARC is also tested.

Expiry is lazy: lookup rejects expired entries; request admission, store
preparation and stats collection reclaim due idle chunks. Completed stores
start the clock. In-flight stores and referenced loads cannot expire. The last
load completion restarts the clock. Native touch refreshes a still-present
entry without deleting an offered hit between lookup and prepare_load; it
cannot resurrect a removed key. Capacity eviction and reset discard deadlines.
Native removal events and `vllm:kv_offload_cpu_cache_expired_chunks` expose
expiry. Metrics register the counter even when the builder injects the TTL
default after metric setup.

Pinned-image CPU test `1b8675b4ee09` passed140 tests,14 existing warnings,
exit0/no OOM,2026-09-21T23:36:05--23:36:39UTC. Tests cover pending stores,
overlapping loads, reused slots, failed writes, reset, deadline crossing,
counter reset, configuration propagation and the native layout guards.
Initial138-pass/one-fail run lacked the package path needed by an existing
dynamic factory test; the final container supplies that path, without skipping
the test. Source manifest preserves61 engine sources/74 export files. Local
suite105 pass/two dependency skips; targeted Ruff and diff checks pass.

### Resident-host expiry and connector coverage gate

The direct-host coordinator now selects `IdleExpiringHostBlockPool`, a narrow
extension of the existing shared-event native pool. It preserves native
hashes, refcounts, duplicate-block handling, aliases, LRU/free queues and
reuse callbacks. GPU pools and ordinary HiSparse pools are unchanged.
The planner carries the same configured idle lifetime (default3600 seconds)
to every worker and the scheduler; mismatched worker lifetimes fail merging.

The final reference release starts the idle clock. Referenced pages do not
expire, including pages shared by overlapping requests. Lazy expiry runs on
host lookup/allocation and invalidates every hash alias of an expired idle
page, emitting native CPU BlockRemoved events. Already-free expired pages move
to the front of the native free queue. No tensor copy, zeroing or unmapping
occurs at expiry: native allocation still controls physical buffer reuse and
fires outstanding read/reuse callbacks before overwrite. Successful reset
clears deadlines; native touch protects offered hits across a clock boundary.

Pinned-image CPU suite passed176 tests, then177 with a real connector coverage
regression. That regression holds a native CPU cold copy live for7200 seconds,
expires its required resident QSA prefix at3600, and verifies the connector's
maximum loadable prefix becomes zero and local prefix lookup misses. Thus
different tier lifetimes cannot authorize reading missing RAM pages. Tiers
expire independently; this is a conservative prefix-coverage bound, not a
custom distributed session-expiry transaction.

Evidence: `native-host-idle-ttl-cpu.log` and
`native-host-idle-ttl-bound-cpu.log` in the k3s evidence tree. Final container
`80640d6e2156` ran23:46:43--23:47:18UTC on2026-09-21, exit0/no OOM, CPU-only,
177 pass/15 warnings (14 existing deprecations and one unregistered test mark
in the isolated harness). Local105 pass/two dependency skips;62 engine sources/
75 export files pinned. This is allocator/connector bookkeeping evidence,
not a new weighted-model or GPU DMA run. Pinned memory pools remain allocated;
expiry revokes reusable prefix ownership and allows recycling, not secure
erasure or immediate release of the arena to the OS.

Remaining: ordinary independent-request hot/cold qualification with weighted
target/MTP and recurrent checkpoints, including expiry during real transfers;
graph profiling and optimized-mode qualification; unresolved baseline
numerical nondeterminism. No serving container changed.

### Ordinary-request weighted screen launched

`launch_session_swap_screen.py --native-prefix --source-stopped --execute`
now mounts all62 manifest-pinned engine sources and disables the diagnostic
session-ID coordinator (`VLLM_FLASH_SESSION_SWAP_BYTES=0`). It starts only the
authorized free development quartet; exact source/serving identities, source
model mount and serving start time are guarded. No ports/network are exposed.

The new `prefix_cache_screen.py` uses ordinary completed AsyncLLM requests,
native prefix hashes and native CPU offload with LRU/3600-second TTL. Initial
qualification scope is eager TP2/PP2, MTP2,8K context,7100-token prompts,
128 generated tokens,seed173,512 resident-host blocks bounded by128GiB total
rounded pinned allocations,8GiB native cold tier, unchanged750000000-byte
GPU KV budget per rank. Production context/topology remain unchanged.
Two no-cache-read references, a hot repeat,32 distinct-prefix pressure
requests and a final repeat exercise the path. Native logger counters must
prove stored bytes and a cold load during the final request; cached tokens or
matching text alone cannot pass. Baseline repeatability and exact token
comparisons remain required. This screen alone does not qualify full scope.

Container `1fdf1e2ac1de` (`flash-session-gpu-vb5vuc`) started
2026-09-21T23:55:33.902236863Z; remote root
`/tmp/flash-session-gpu-20260921-VB5vUC`. Engine logs confirm eager TP2/PP2,
MTP2 and native prefix caching. It exited1/no container OOM at
2026-09-22T00:05:25.806554727Z. Launch evidence is k3s `launch-native-prefix.json`.
Its legacy `storage_backend=copy` field is unused in native-prefix mode;
the prototype is disabled. No observation timeout permits a restart.
Local107 pass/two dependency skips, including native-screen config/verdict
guards. The launched helper SHA is
`47a2dd28efcd44a189d3ae5ec13ec908c4be8723694beeebc1046b0e639c19c1`.
After launch the local helper corrected its explicit LRU key from the unused
`cache_policy` to native `eviction_policy`; both use native default LRU, and
the immutable running snapshot was not changed.

### First weighted-prefix result: not qualified

The two uncached7100-token/128-output-token runs matched exactly. The hot
repeat reused3504 tokens but first diverged at output index6 (token271 versus
198). Fourteen pressure requests then completed with no cached tokens.
The next request produced NaN logprobs; strict JSON serialization raised
before persisting that row, and the exception handler encountered the same
NaN again. Saved `screen-native-prefix-incomplete.json` is the last successful
snapshot, not a clean finish: its null error field is stale. Failure traceback
is in `native-prefix-nonfinite-failure.log`. No final cold request ran.
Native store counters increased, but no cold load is proven. Do not attribute
the hot divergence or NaN to offload layout without further isolation.

The harness now encodes nonfinite probabilities as null with explicit
position/token/value failure metadata, saves it, and fails the request. It
does not ignore NaNs or allow them through the pass predicate. Local source
SHA `350d7cb3ef8feb50340603238f8115acdaf089b41bf93da60a60e89058eb5443`.

In parallel, a CPU regression proved uneven PP workers could derive different
shared CPU row sizes/chunk counts: (128,65536,16384) versus (256,32768,8192)
for (chunks,row bytes,rank-slot bytes). This can disagree even when total mmap
size matches. Direct-host planning now carries one conservative maximum slot
size to all worker/scheduler configs. Empty worker groups contribute zero;
undersized slots and mismatched plans fail closed. CPUOffloadingSpec adds a
PP all-rank geometry check before opening a shared mmap region. Native buffer
transfer logic and hashes remain unchanged.

Regression `2f89c5addf62` failed as expected. Corrected pinned-image suite
`966ca088f388` passed181 tests/15 warnings in31.38 seconds, exit0/no OOM at
2026-09-22T00:10:05UTC. It includes the original geometry repro, plan rejection,
and checks that disagreement fails before any shared region is opened.
The collective's real multi-rank execution still needs GPU qualification.
Per-layer slot sizing remains conservative and can waste cold-tier capacity;
deduplicating aliased regions is a separate measured optimization, not claimed
implemented. Local108 pass/two native-dependency skips; source pins updated.

Next: isolate the hot-prefix mismatch and NaN under buffer reuse, test the
corrected real PP transfers, then prove actual cold hits and exact state
restoration. A no-MTP control and poisoned-unused-cache checks can distinguish
draft/recurrent replay from attention masking/allocation failures. Production
remains untouched; dev GPUs0--3 are free and no weighted test is running.

### Historical prototype contract (diagnostic only)

Retain more resumable sessions than fit in the hot GPU cache. Suspend a session
at a drained generation boundary, preserve its full context in bounded RAM,
release its GPU cache blocks and worker request slot, and restore into newly
allocated blocks/slots before scheduling it again. Uninterrupted active decode
must not perform per-token swap transfers. Preserve existing main-QSA RAM
offload, target weights, MTP acceptance/rollback semantics, and normal behavior
when the feature is disabled. Capacity is bounded by bytes, not a promise of
100 arbitrary-length sessions. No public API or serving deployment is changed
by the current work.

## Findings from the actual engine

- Routed-method trace24d3217e5825 ran22:01:29--22:08:39UTC, exit1/no OOM. All15
  spills/nine restores/six cancellations and cleanup pass; numerical FAIL.
  Layer1 router selection weights/IDs and routed-forward tensor inputs match
  across four references. Routed outputs differ in reference3 by
  0.000030517578125 on TP0 and0.0001220703125 on TP1, before the final TP
  reduction. This localizes the issue within the routed path but does not yet
  identify an individual kernel. No new full-model diagnostic was launched.

- Component-trace successor8bbcd277af21 (21:53--22:00UTC) again passes lifecycle
  and cleanup but fails numerical qualification. Layer1 MLP inputs, router
  logits and all shared-expert component outputs match across four references
  on both TP ranks; combined expert output differs in reference3 by
  0.000030517578125. The next diagnostic wraps only this layer's expert
  selection, routed forward and final TP reduction, restoring the original
  methods when collected. Native shared-expert cuBLAS behavior alone does not
  explain this observed boundary. Local suite106 discovered/104 pass/two skips.

- Native-fallback diagnostic094f114fcaae (2026-09-21,21:42--21:49UTC) completed
  15 spills/nine restores and strict cleanup, but failed exact output checks.
  With19 opt-in kernels disabled, all-layer prefill tracing first observes a
  difference in layer1 MLP output (max absolute0.000030517578125) despite equal
  recorded layer inputs. Both stage0 TP ranks agree on this observation. This
  precedes session swapping and does not identify the responsible operation or
  prove harmless rounding. The diagnostic now adds bounded layer1 MLP component
  hooks; CPU readbacks perturb timing, and no tolerance waiver is introduced.
  Local suite105 discovered/103 pass/two native-module skips.

- The scheduler already retains `resumable` requests in
  `WAITING_FOR_STREAMING_REQ`. They currently keep cache blocks **and** worker
  request slots; simply removing them from the running queue does not free
  capacity. Ordinary completed HTTP requests are not automatically sessions.
- The worker uses one block-outermost shared GPU allocation. Recurrent states,
  speculative rollback slots, raw indexer rings, and compressed attention-index
  pages all need preservation. Cache group IDs and request block tables are the
  authoritative mapping, not model-layer iteration order.
- Historical prototype: QSA's `_qsa_host_kv` uses the **same physical block IDs** as the GPU pool.
  Freeing GPU blocks while leaving those RAM pages in place is unsafe: another
  request can overwrite them. The first implementation copies live host pages
  to independently owned cold storage too. A separately indexed host allocator
  eliminates that CPU-to-CPU copy and is required by the clarified production
  direction above. The copying prototype remains diagnostic only.
- MTP accept/reject completion does not prove every recurrent rollback copy
  has already run. The model state carries `num_accepted_tokens_gpu`, and
  `mamba_cache_mode=none` can consume it on the next forward. Preserve all state
  slots and this metadata initially; do not silently keep only one state slot.
- Worker request reconstruction must preserve position, accepted-token count,
  pending drafts, sampling seed/state, and any model-specific state in addition
  to cache bytes. PP output rings and in-flight scheduling must be drained.
- The development command requests `mamba_cache_mode=align`, but the native
  configuration verifier changes it to `none` when prefix caching is disabled.
  Both retained-source and new GPU-trial startup logs confirm this. The extra
  align-mode allocator coverage preserves logical last-state columns, retirement counts, checkpoint
  columns and allocated-request membership along with null table positions.
  Worker alignment indices are logical columns, so physical-page relocation
  does not change them. Prefix caching and pending partial-tail/COW transfers
  remain unsupported. The initial GPU screen uses effective `none` mode.
- Flash-Next uses `Qwen4ExpModelState`, a hybrid-state subclass. Its PLE
  n-gram buffers are batch scratch reconstructed from restored token history,
  not persistent request-slot rows. An empty text-only registration in the
  model's encoder cache is permitted; actual image state remains unsupported.

## Transaction protocol

1. Stop scheduling the selected request and drain in-flight work. Freeze its
   token boundary, block tables, and worker metadata. Do not change live tables.
2. Admit cold storage under explicit per-rank byte/session limits. Copy all
   local GPU state and authoritative host KV; wait for all-rank acknowledgement.
   On failure, retain the original hot allocation and clean partial snapshots.
3. Only after acknowledgement, release hot blocks and worker slots. Keep the
   request's scheduler/token history and cold generation identity.
4. To resume, reserve fresh hot blocks and a worker slot. Restore all cache
   groups, host KV, metadata, and block tables. Keep the cold source until all
   ranks acknowledge; failed destinations must never become runnable.
5. Commit hot ownership, retire the cold copy, and resume normal scheduling.
   Explicit cancellation/expiry retires the corresponding snapshots. Never
   silently evict the sole copy of a resumable session to satisfy RAM pressure.

Initially use synchronous, device-fenced transfers at session transitions.
Later overlap copies only after the correctness and failure gates pass. CUDA
graph buffer addresses must not change. Session keys are internal scoped
request identities plus generations, not arbitrary globally shared user names;
client/tenant identity must remain attached through any API integration.

## Implementation and remaining integration

`vllm/v1/worker/gpu/flash_session_swap.py` implements bounded rank-local cold
storage and the native BLNHC/QSA registration adapter. It copies byte views
without a gathered GPU temporary, rejects missing coverage or unsupported
layouts, preserves null-block positions, supports relocation, retains source
checkpoints through restore failures, and requires explicit retirement.
The memory limit accounts for tensor and metadata payload bytes; normal Python
object/allocator overhead also needs headroom in the process RAM limit.

`flash_session_worker.py` now connects native runner initialization and an
internal worker RPC behind the default-off `VLLM_FLASH_SESSION_SWAP_BYTES`
setting. It captures explicit worker metadata, releases/restores request slots,
and retains cold state on failed restore. `flash_session_allocator.py` provides
private-block release/reservation helpers. `flash_session_transactions.py` now
wires engine utility calls, all-rank acknowledgements, scheduler ownership,
cancellation, and automatic policy hooks. Cold sessions are retained outside
runnable/waiting queues, so they neither occupy hot slots nor keep an idle
engine executing empty batches.

Engine cache-reset, destructive sleep, weight-version changes and arbitrary
collective RPCs now reject while any cold/in-progress session transaction is
retained. Explicitly restore or cancel those sessions first. Session transfer
RPCs and device synchronization remain available. This guard is local code
newer than the initial GPU screen; it is not loaded in that trial.

Remaining: native GPU-worker integration tests, usable client continuation
demonstration, complete configuration/state-coverage audit, GPU/distributed
model-backed qualification, performance measurements, and PR preservation.
The all-rank coordinator tests currently use a simulated executor: they prove
transaction ordering, not real distributed correctness. Do not advertise this
as qualified working session swapping yet.

## Retention policy and diagnostics

The user selected **60 minutes** on 2026-09-21. The experimental defaults are:

- `VLLM_FLASH_SESSION_SWAP_BYTES=0`: feature disabled. Positive values bound
  each rank's independently owned cold tensor/metadata payload.
- `VLLM_FLASH_SESSION_SWAP_MAX_SESSIONS=100`: per-rank cold count ceiling, not
  a guarantee that 100 long contexts fit the byte budget.
- `VLLM_FLASH_SESSION_SWAP_TTL_SECONDS=3600`: idle cold-session TTL, measured
  using a monotonic clock from the completed cold commit. Zero disables expiry.
- Under hot-slot pressure, or waiting work at at least 90% block-pool usage,
  spill the least-recently-used **idle** streaming session to RAM. Exclude
  active generation and sessions with queued new input. Under RAM pressure,
  refuse capture without discarding an unexpired sole checkpoint.
- Automatically drain the pipeline before a policy transaction. On expiry,
  retire all rank copies through the cancellation protocol. A paused active
  generation snapshot is not an idle-session TTL victim.
- An incoming chunk for a cold streaming session requests restoration. A
  retained checkpoint is never passed to the scheduler until all ranks finish
  restore and retirement. Policy failure leaves the engine paused and reports
  `policy_error`; it does not retry indefinitely or resume partially restored
  workers. A manual scheduler pause is not automatically resumed by policy.

The internal engine utility `flash_session_swap` provides `status`, `stats`,
`lookup`, `describe`, `suspend`, `restore`, `complete_suspend`, `complete_restore`,
`evict_lru`, and `expire_idle`. Explicit transfer calls require a completed
`pause_scheduler(mode="keep", clear_cache=False)` first. Restore/retry calls
require the returned session generation and original `client_index`.
This index identifies an engine frontend, **not an authenticated tenant**;
these utilities must not be exposed as an unauthenticated public session API.

`lookup` reports `hot_hit`, `cold_hit`, `miss`, or pending/unavailable state.
Misses include `expired`, `cancelled`, and `not_retained`, with
`requires_prefill=true`. Polling does not extend TTL or alter LRU ordering.
Counters report lookups by tier/reason, successful restores/suspends, LRU
spills, retirements, capacity refusals, and policy failures. These are session
state counters, not vLLM prefix-cache hits or saved-token counts. Expiry history
is bounded to 1024 tombstones; after that history is evicted, the reason can
be `not_retained` instead of `expired`. An expired session needs the full
conversation resubmitted for re-prefill; no public HTTP fallback is implemented
by this patch yet.

When otherwise idle, the engine checks expiry at roughly one-second intervals.
Busy-pipeline expiry waits for a drained transaction boundary. This favors
cache integrity over a strict real-time memory reclamation deadline.

## Mandatory integrity gate

The user requires cache integrity to take priority over swap performance.
The current rank-local implementation therefore:

- Re-reads captured pages and compares their exact bytes before publishing a
  checkpoint. Capture failure preserves the original hot allocation.
- Binds a SHA-256 checksum to the internal request identity and generation,
  token boundary, region descriptors, worker metadata, and saved page bytes.
  Cold-state corruption or checkpoint substitution must fail before restore
  writes. This detects corruption; it cannot repair a corrupted sole copy.
- Rechecks both cache bytes and dynamic worker metadata before releasing the
  original worker slot. A changed source or checkpoint rejects cold commit.
- Reads restored destination pages back and compares exact bytes, then checks
  restored worker metadata. A failed destination is not eligible to resume.
- Retains the cold source through restore until explicit global retirement.
  Verification uses one CPU page of scratch at a time, not another complete
  long-context checkpoint. Copy/readback/hash costs occur at swap boundaries.

These checks do not prove complete state coverage, allocator ownership,
cross-rank atomicity, or correct CUDA ordering. The remaining qualification
gates below are mandatory. A rank failure or ambiguous transaction must stop
scheduling the affected session; successful ranks must not continue alone.
RAM snapshots are not durable recovery from process or host failure. Never
claim an absolute guarantee against all hardware faults or memory corruption.

## Qualification gates

The user explicitly requested deterministic end-to-end output comparison.
Use greedy decoding (`temperature=0`) and the same explicit seed (173), exact
prompt token IDs, tokenizer/template, weights, engine settings and hardware
placement. A seed by itself does not establish determinism.

For each tested topology and MTP depth, run an uninterrupted reference, then
the same prompt with suspend/restore at recorded token boundaries. Require
exact equality of generated token IDs, not just rendered text or semantics.
Repeat after another session has written the original blocks and worker slot;
then repeat several swap cycles and interleave multiple retained sessions.
The reference and swapped runs must match execution/batching conditions for
the strict equality gate. Also run a deliberately varied-interleaving stress
cohort; any divergence needs investigation, not an automatic tolerance waiver.
Compare continuation logits at the first restored step and preserve the
first differing token/position and relevant model/rank configuration on failure.
The logit comparison supplements, not replaces, byte-exact cache checks.

Test MTP disabled and enabled, including partial acceptance and rejection.
Add seeded non-greedy continuation separately to exercise sampling-state
restoration; it is not a substitute for the deterministic greedy release gate.
The bounded model-backed results below cover only part of these gates.

`tools/flash_next/qualify_session_swap.py` implements an offline C1 screening
harness with explicit engine arguments and a local model mount. It first
requires two uninterrupted greedy runs to produce identical token IDs, then
compares a repeatedly swapped run. It records first-token divergence,
top-five logprobs, drained token boundaries, hot ownership and same-group reuse
of the original freed pages by other requests. Failure to prove page reuse is
a coverage failure, not a pass. Top-five logprobs are diagnostics, **not** full
continuation-logit validation. This harness has passed CPU helper tests and
syntax/lint checks and the bounded model-backed screens recorded below.

- Real-tensor CPU tests: GPU-page stand-ins and actual host backing survive
  source overwrite, relocation, repeated cycles, nulls, capacity failure,
  partial transfer failures, and identity/generation mismatch.
- Native allocator integration: restored ownership and state indices match
  every cache group; hot blocks and worker slots are reusable while cold;
  failed admissions/transfers do not leak blocks or permit early scheduling.
- CUDA one-layer test: registered native GDN/QSA/PLE state, dirty block reuse,
  MTP acceptance/rejection boundaries, and graph replay match uninterrupted
  execution, including non-default stream fences.
- Full-model TP4 and TP2/PP2: deterministic continuation/logits against a hot
  reference, more retained sessions than GPU slots, interleaved clients,
  cancellation, restore failure, context growth, and memory-budget exhaustion.
- Report swap bytes/latency, additional RSS/pinned RAM, freed GPU blocks/slots,
  resumed TTFT, steady-state prefill/decode throughput, and MTP acceptance.
  Do not substitute CPU byte-copy tests for these model/distributed gates.

Run the CPU tests with:

```sh
.venv/bin/python -m unittest discover -s tests/flash_next -p test_session_swap.py -v
```

Local results (2026-09-21): all 43 focused storage/metadata/transaction/policy
CPU tests pass. A further admission-telemetry test passes. The broader Flash-Next suite reports 94 tests, with 92 passing
and two native test-module skips because native engine dependencies are unavailable
in the local CPU environment, when run with
`PYTHONPATH=tools/flash_next:vllm/v1/core/sched:vllm/models/qwen4_exp/nvidia`.
Targeted Ruff lint/format checks pass. These are CPU-only gates;
the integration and model-backed gates above remain open. The preservation
manifest/exporter now includes all session modules and modified engine paths
(51 engine files, 64 total source-only build-context files). This records the
experimental source, not a claim that the preserved image contains the patch.

The skipped native allocator tests were separately run successfully in an
isolated CPU-only container using the pinned image:
`flash-session-allocator-sfthz3` / `e550f907a493eae07083fafc5ecf7d2b67075f7e94acf57edb64b427cb6dcd60`.
Both tests passed, covering MTP depths 0--4, hot block reuse, restored allocation
growth, capacity rejection, and complete reclamation. Container exit status is
zero without OOM. No GPU devices, network, model mounts, or published ports;
2 CPU / 2 GiB limits, read-only root and source mounts, temporary venv created
with `uv --offline` and system site-packages. Probe sources/logs remain at
`water-server.lan:/tmp/flash-session-allocator-20260921-Sfthz3` and in the
stopped diagnostic container. This does not test native scheduler transitions
or GPU/model state.

A second CPU-only native probe, `flash-session-scheduler-g8r8ar`
(`5e9b83724953400b2afd0ddeeaa0e4f9a0e8b250ef939fe80bab48f31e64e639`),
passes seven tests: the allocator gates including repeated `align` relocation at
MTP depths 0--4, native streaming suspend/reuse/resume/cancel with synchronous
PP1 and asynchronous PP1/PP2 schedulers, native TTL expiry, and PLE n-gram
reconstruction after slot relocation, and rejection of engine mutations while
cold state is retained. Alignment tests cross multiple state
block boundaries and compare allocation/retirement against an uninterrupted
native allocator. Worker collective replies are still stand-ins: this is not
a distributed or model-output test. It uses the same pinned image with runc,
no GPUs/network/model mounts/ports, read-only root, 2 CPU / 3 GiB limits and
offline uv. Sources and stopped-container logs are retained at
`water-server.lan:/tmp/flash-session-scheduler-20260921-G8r8ar`.

Initial model-backed smoke screen `f99cd2f6634b` / `flash-session-gpu-an8xh4`
started at `2026-09-21T19:05:29.345341516Z`, completed initialization and
exited1/no OOM after two uninterrupted 128-token greedy references diverged at
zero-based token87 (11870 versus6007). No swap ran. The leading two token
logprobs reversed by 0.125 in each run. This blocks numerical qualification;
it is not evidence of swap-induced corruption. Next test the same prompt,
seed and token count with MTP disabled to isolate baseline repeatability.
Root: `water-server.lan:/tmp/flash-session-gpu-20260921-aN8Xh4`.
It preserves TP2/PP2, Q8 MTP2, PP26/22, 750MB hot cache/rank, 56GiB QSA host
tier, 240K context and the existing pinned image/optimization overlays. Nine
read-only patch overlays add session swapping with a 1GiB/rank cold-store cap.
No network or ports; unchanged 240GiB memory/no-swap container limit.
The old development container `d9f6e139ba74` stopped cleanly and is retained;
stop the trial before restarting it for rollback. Serving `7d408c2bc15d` on
GPUs4--7 remains unchanged. Later restored-continuation results are below.

No-MTP isolation screen `818db3f5b39d` / `flash-session-gpu-ef2gg9` started at
`2026-09-21T19:13:42.684391337Z` and exited1/no OOM at the same token87
baseline divergence, with no swaps. Both full token streams match their MTP2
counterparts exactly, but logprobs differ from position0. It used the same TP2/PP2 cache/context settings,
prompt, seed and output budget. It includes the newest mutation guards and
reports the effective Mamba mode. Root:
`water-server.lan:/tmp/flash-session-gpu-20260921-ef2gg9`.

Diagnostic `ee9ee2de2807` / `flash-session-gpu-b7kx1j` started at
`2026-09-21T19:22:08.147021576Z` and passed, exiting0/no OOM. It used the same no-MTP
configuration but runs four references to identify first-run/warmup effects,
then attempts transactions even if the baseline fails. `--diagnose` cannot
waive repeatability failure: the verdict still requires every reference,
candidate output and dirty-reuse gate to pass. A CPU regression test checks
that a successful diagnostic restore cannot turn a failed baseline into a pass.
Root: `water-server.lan:/tmp/flash-session-gpu-20260921-b7kX1J`.
All four reference streams and the twice-restored candidate matched128 tokens.
Every one of39 source pages per cycle was reused by other requests in its own
group. Capture394/475ms, restore385/269ms; two cold hits, zero unresolved
transactions or policy errors. This passing stream matches the second stream
from the earlier failed controls; the cross-run variability is not resolved.
The result is limited to one short-context C1 no-MTP screen.

Q8 MTP2 diagnostic `45a4450f556c` / `flash-session-gpu-jxzcdu` completed,
exit1/no OOM: four unswapped references differ, so the overall screen failed.
Its twice-restored candidate matches the first128-token reference, all113
source pages per cycle were reused, and all byte checks completed. Capture
1316.11/783.94ms; restore540.74/489.35ms; no unresolved transactions or policy
errors. Root: `water-server.lan:/tmp/flash-session-gpu-20260921-JxZCDu`.

The native streaming client harness `tools/flash_next/session_stream_screen.py`
holds input streams open between turns, attempts nine retained sessions with
eight configured slots, resumes by sending new input, and checks lookup isolation
and cleanup. It uses the native streaming semantics: only computed prior output
tokens are retained; final uncomputed sampled tokens are dropped by the existing
scheduler before adding the next input. This is not a new HTTP chat API.

First streaming trial `b1ce683928cd` completed four two-turn references with
inconsistent output, then timed out on the first retained stream's first turn.
No swap ran. Exit1/no OOM; root
`water-server.lan:/tmp/flash-session-gpu-20260921-gsPUMK`. Follow-up instrumentation
records queue/status counts, processed/scheduled steps, free/usable GPU blocks,
per-phase progress and output-event counts. Ordinary stats do not issue worker
RPCs. These counters are scheduler observations, not proof of worker-slot or
rank-local cold-memory reclamation. Serving remains unchanged.

Expanded native CPU lifecycle probe `30239028c2ad` passed eight tests, exit0/no
OOM. It closes four two-turn streams with complete block reclamation, retains
three idle sessions with two hot slots via automatic LRU, restores via queued
input and reclaims all79 usable blocks at final close. Worker replies are still
stand-ins; it does not resolve the real GPU timeout. Root:
`water-server.lan:/tmp/flash-session-lifecycle-20260921-njEHHA`. The first CPU
launch failed before tests on a read-only uv cache; the successor uses `/tmp`.
It inherits prior native-probe source mounts read-only and overrides only the
coordinator and native test with the current telemetry/regression. GPU diagnostic
`a5f5cc0fde67` separately runs the instrumented streaming screen.

That instrumented GPU screen exited1/no OOM: after reference2, scheduler stats
showed zero requests and460/460 blocks free but idle_hot=1. Reference3 then
remained WAITING with no output until timeout. Two new native tests reproduce
overlapping same-session forwards before an earlier result settles, and a
three-token MTP decode with only one returned token left in the turn budget.
Both fail before local scheduler guards and pass after (10 native CPU tests,
`f33db893f4a6`, root `/tmp/flash-session-boundary-20260921-QYynqw`). Guards apply
only to resumable requests while session swapping is enabled: one unsettled
forward per session, and a decode tail bounded by remaining max_tokens.
They are not GPU-qualified; early EOS/stop truncation and its recurrent-state
boundary still require investigation. Do not infer complete state correctness
from fixing the idle counter. No successor GPU run was launched.

## Native offload reuse review (2026-09-21)

The user's architecture question requires evaluating reuse before further
expansion of the standalone cold store. This checkout's `OffloadingConnector`
already handles MambaSpec and AttentionSpec, hybrid cache groups, asynchronous
transfers, native CPU capacity/policies and lookup metrics. It is incorrect to
describe native offload as attention-only. Its ordinary contract is reusable
prefix chunks, not a snapshot of an in-progress request's worker slot.

Current Flash QSA uses custom `_qsa_host_kv` UVA backing and tiny GPU placeholder
pages. Registration must preserve the real host pages, not just placeholders.
Current prefix caching is disabled and effective Mamba mode is none, with
240000/3504-token group sizes; native offload configuration has alignment
requirements. Exact request history/sampling/MTP metadata and session identity,
TTL and transactional ownership also need an explicit contract. Prefer native
offload infrastructure plus narrow model/session adapters if these gaps can be
closed. No native connector has been enabled or live memory budget changed.

### Native transfer and group-layout evidence

CPU audit `flash-native-offload-s8zk34-model` (`1ba3905afff3`) exited0/no OOM.
The installed native CPU manager protects in-flight loads, refuses allocation
when both chunks are busy, implements idle LRU and removes failed stores.
With actual model head_dim256/TP2, standard QSA registration sees7008 bytes
per block but omits3588096 bytes of separate host backing. Current mixed
240000/3504/8-token groups fail native hash alignment; CircularBufferSpec is
not handled by its window helper. This is a compatibility audit, not a pass
for enabling the ordinary prefix connector.

GPU0 transfer-only probes `7392ee21366b` and `411cffc99ddb` both exited0/no OOM.
They used the pinned image with no model, network or ports,4GiB memory/no added
swap, and read-only source mounts. The second additionally mounts the new
`native_group_cache_layouts` adapter in `flash_session_swap.py`.
Remote root: `/tmp/flash-native-transfer-20260921-BZCohA`.
Evidence lives in the k3s repository at
`docs/benchmarks/flash-session-swap-20260921/native-transfer*.json`.

Unmodified native CPUOffloadingWorker successfully copies GPU pages and CUDA
UVA views of pinned QSA host pages. Three cycles overwrite every hot source,
restore to different physical IDs, and verify exact bytes and untouched
non-destinations. The grouped adapter separately passes relocation to nonzero
IDs, represents empty groups without a pool, and rejects missing coverage,
overlapping/duplicate regions, strided pages and unpinned host memory.
One chunk in each of the two representative group pools allocates6836224
bytes instead of10424320 bytes in a shared two-chunk pool. This only describes
the probe geometry, not the full model's cold-memory requirement.

Adapter SHA256:
`c636d815bf7f7bd758f858205a5f1fee3f4cdf98475d345e3752b34418577eaf`.
Grouped probe SHA256:
`832070e17c015efbc927874493514e0b92ce68f5802a79b173984327a243c35a`.
Local preservation/CPU suite:94 discovered,92 passed,two native-module skips.
The adapter is not wired into SessionColdStore or a serving connector yet.
No model-output or all-rank restoration claim follows from these DMA checks.

Next integration contract: reuse native pool allocation/transfer primitives,
but add explicit checkpoint retention and retirement. Native prefix LRU works
at chunk granularity; it cannot silently discard a sole resumable checkpoint.
Session-wide LRU/60-minute TTL must coordinate all-rank removal before chunk
reuse. Do not simulate retention with indefinitely pending native load jobs:
that would conflate ownership with active-transfer metrics. Keep existing
hash/readback and all-rank acknowledgement gates while replacing the storage
backend. Full-model streaming boundary, deterministic baseline and EOS/stop
gates remain open independently of native transfer support.

### Opt-in native checkpoint backend (2026-09-21)

`VLLM_FLASH_SESSION_SWAP_BACKEND=native` now selects NativeSessionColdStore
when session swapping is enabled. `copy` remains the default/reference backend;
unknown values fail startup. The launcher records the selected backend and
accepts `--storage-backend native`; the screen report records it too.

NativeSessionColdStore reuses unmodified CPUOffloadingWorker allocation/DMA and
CPUOffloadingManager bookkeeping. Each checkpoint owns one exact-sized pool
per populated local group. Other checkpoints never admit stores to those
pools, so native chunk eviction cannot partially destroy another session.
The existing aggregate payload/metadata byte limit is enforced before creating
pools. This is private native storage, not enabling the prefix connector and
not a shared native LRU pool. The coordinator retains whole-session idle LRU,
60-minute expiry, explicit misses, and all-rank retirement responsibility.
Pool creation/destruction overhead needs measurement before sharing/recycling
pools is considered.
The byte limit is an admitted payload/metadata limit, not yet a measured bound
on process RSS or PyTorch's cached pinned allocator reservation. Zero live
checkpoint bytes/pools after retirement does not prove those physical host
pages were returned to the OS. Measure allocator allocated/reserved bytes under
varying context lengths before claiming bounded physical cold-RAM overhead.

Existing identity-bound hashes, capture/readback checks and worker metadata
apply directly to native CPU buffers without another checkpoint-sized copy.
A version-pinned binding reads the native store handler's CPU tensor views;
shape/dtype/pinning/layout checks reject incompatible implementations. Failed
capture drains private storage before acknowledging abort. Failed load
acknowledgement fences device writes before releasing its native read lease.
Worker rollback also fences before acknowledging that destination blocks may
be recycled. Failed retirement preserves the checkpoint and charged bytes.
Unresolved transfer/fence failures prohibit subsequent store/restore operations.

Real-GPU backend probe `590ac1ad2dcc` (`flash-native-store-r8ajkn`) passed,
exit0/no OOM,20:26:31--20:27:14UTC. It tests two8460327-byte private checkpoints,
capacity refusal without eviction, byte-exact restore after source reuse,
metadata/null positions, deliberate corruption before destination writes,
idempotent retirement, failed load-ack retry and failed capture-ack cleanup.
Final charged payload/native pool count:0/0. It does not test a model, TTL clock
or all-rank coordination. CPU suite:96 discovered,94 pass,two native skips.
Root: `/tmp/flash-native-store-20260921-r8ajkN`; k3s evidence `native-store.json`.

Source hashes tested:

- Store: `9cf3fcae6ca2ad364eee2216e09ddd83e37812a3481e31ba1fac171dab8b3d9a`.
- GPU probe: `7b6224fc72275716848db5eaafb8ec7611ac9cfb79356727e179e26514018e62`.
- Worker integration, local CPU-tested: `8f4036cc940d931d23b366a7bbf1c15422ac5f630062e4ef3cfbc5c5dc183a74`.

Full-model screen `cf92ea30ca17` used native storage, Q8MTP2/TP2/PP2 and the
retained streaming-boundary guards. It exited1/no OOM at20:35:24UTC after all
nine sessions were admitted and restored. Total15 automatic spills/nine
restores/six cold cancellations; final requests/idle counter/cold records are0,
all460 blocks free, no policy or byte-integrity failure. Reference closures
also leave idle_hot=0. The prior idle-counter failure did not recur.

Numerical verdict remains FAIL: references differ at first-turn token88 or
cumulative second-turn token129, and only one retained session matches both
turns of reference0. These zero-based divergences precede any swap in the
reference controls. Evidence: k3s `screen-native-stream.json` and exact source
pins in `launch-native-stream.json`; root `/tmp/flash-session-gpu-20260921-CbpzGA`.
Serving is unchanged. Do not waive numerical/EOS/stop or memory-accounting gates.

Source audit identifies a separate possible streaming bug: runner.add_requests
removes/re-adds an existing slot; MambaHybridModelState.add_request resets
num_accepted_tokens to1 without copying its last accepted recurrent state to
the base slot in none mode. GDN non-spec/prefill uses base state, while spec
decode selects column(num_accepted_tokens-1). Native-core reproducer
`42b36f0e701c` confirms the error for accepted counts2/3 (output max differences
0.041015625/0.10546875); selecting/copying the correct state restores exact
output. This affects the demonstrated state handoff but does not establish the
cause of all full-model variability, especially first-turn/no-MTP differences.

`materialize_mamba_boundary` was initially an isolated candidate helper.
It validates all regions before writes and implements
native conv-window/temporal-copy semantics, cloning only overlapping conv
slices. GPU probe `8626cdb838ff` passes accepted counts1--3 with SD/DS conv
layouts and exact native recurrent output; exit0/no OOM at20:49:39UTC. CPU
suite98 discovered/96 pass/two native skips. All development containers stopped.

Worker/helper SHA256:
`ead0a5f737e76e081915d248e8cd37dfdce9afef23e2f626bc5094eb8b76a54d`.
Probe SHA256:
`16da1ca0609dabc79a18f78d0f37548fe5b92554ea69dc2bce5174e21dbd6f4a`.
Evidence: k3s `mamba-stream-boundary*.json`; remote root
`/tmp/flash-session-boundary-kernel-20260921-igfyDR`. Earlier attempts failed
before kernels on RO/noexec cache locations; the successful isolated GPU0
probe uses executable temporary JIT cache,4GiB RAM,no weights/network/ports.
The subsequent hook now runs before streaming re-add, settles only the PP
receive prefix needed for the existing slot, checks scheduler/worker boundaries,
materializes state, then resets acceptance. GPU worker-hook probe `f5d35a461dd1`
passes1--3 accepted tokens in both layouts. Local PP queue tests cover selective
prefix draining and stale slot generations. Suite100 discovered/98 pass/2 skips.

Full-model hook screen `ab2fccfe4a9c` exited1/no OOM at21:05:30UTC. Native storage
completed15 spills/nine restores and full block/slot cleanup. Numerical verdict
remains FAIL: unswapped references diverge at token87; some restored second
turns also differ from references with the same first turn. Five of nine full
candidate streams match some reference, which cannot waive reference failure.
Worker SHA `f1f188d4c0f814796bff2d99d4ee00e4b0e7bed2abf07b9c01ea538606b96f3c`;
runner SHA `2f3d0dba1e2168bc406ffae0a4ddb019171dfdae1f6200620e4bd2477bb05e25`.
Evidence in k3s: `launch-native-stream-boundary.json`,
`screen-native-stream-boundary.json`, `mamba-stream-boundary-hook.json`.
This trial stopped; serving unchanged. General spec-to-nonspec
and EOS/stop transitions remain unqualified. The streaming harness now records
per-turn top-five logprobs and checks hot scheduler/block cleanup explicitly;
neither diagnostic changes the exact-output release requirement.
Local suite after harness changes:101 discovered/99 pass/two native skips.
Same-engine diagnostic `314dbae71b95` ran21:08:40--21:15:17UTC on GPUs0--3 and
exited1/no OOM. Lifecycle and strict cleanup pass; numerical verdict FAIL.
Scores show ties/0.125 margins at observed divergence positions87/129/179,
including unswapped controls. Seven candidate streams match reference1, two
match none. These margins do not establish the cause or waive exact matching.
Evidence: k3s `launch-native-stream-logprobs.json` and
`screen-native-stream-logprobs.json`; root
`/tmp/flash-session-gpu-20260921-OQzw8O` on water-server.

The optional `--trace-prefill` diagnostic installs temporary per-decoder-layer
hooks for reference first-prefills, compares valid input/output tensor bytes,
and removes them before swaps. Retained CPU baselines are capped at64MiB/rank;
only hashes/differences leave workers. Readback changes timing, so stable traced
outputs alone cannot rule out an untraced race. Trace coverage is mandatory.
CPU suite102 discovered/100 pass/two native skips. Initial launch `d5d82323668c`
was intentionally stopped during startup (exit137/no OOM at21:17:20UTC) when
source inspection found its1D position filter missed text M-RoPE. Corrected
tracer supports three-channel positions and ignores padded rows. No serving
or engine-source changes are involved in these diagnostic revisions.
Corrected trace trial `964b6ecfb5be` ran21:18:23--21:24:32UTC, exit1/no OOM.
It failed before inference on native secure callable serialization. Evidence:
k3s `launch-native-prefill-trace-rope.json`; root
`/tmp/flash-session-gpu-20260921-o9GueT`; k3s
`screen-prefill-trace-rpc-failure.json`. No trace or swap result. The diagnostic
now uses named worker-extension RPCs with plain integer arguments, keeping
insecure serialization disabled. Pinned-image CPU-only preflight `da19453b2b52`
passes native encoding and8 harness tests; local suite103 discovered/101 pass/
two native skips. No engine production source changes for this diagnostic.

`--native-kernels` is a separate isolated control disabling19 exact opt-in
kernel flags without altering Q8 MTP, PP, scheduling or RAM KV. It does not
revert all engine patches to upstream. User explicitly raised kernel
optimization as a possible cause; neither kernels nor session state are yet
exonerated. Untraced control `0cc7f5ada0c1` started21:29:23UTC with all19 flags0,
otherwise matching the optimized logprob screen. Evidence: k3s
`launch-native-kernel-control.json`; root
`/tmp/flash-session-gpu-20260921-s3aRCl`. It exited1/no OOM at21:38:29UTC.
All four unswapped first-turn references differ (first differences88/30/87
against reference0); logprobs differ from output0. These opt-in kernels are not
necessary for the instability, but neither all patches nor state correctness
are exonerated. Eleven spills/five restores completed before a separate worker
error; cleanup not proven. See k3s `screen-native-kernel-control.json`.

The handoff found no Mamba layers when native groups wrapped their per-layer
specs in UniformTypeKVCacheSpecs; this branch runs only when accepted>1. It now
enumerates actual per-layer specs, preserving group IDs and failing on missing
entries. The prior lifecycle passes did not prove this branch executed.
Worker SHA `1695103980291b04c5cce43d50dc4667a8ca2e87be824fd715e2ecf976a3d6a7`.
GPU0 native-wrapper probe `88ba497fa648` passes accepted1--3, both conv layouts,
and exact recurrent outputs; exit0/no OOM at21:40:49UTC. No model weights/PP.
Evidence: k3s `mamba-stream-boundary-wrapped.json`; root
`/tmp/flash-boundary-wrapped-20260921-hhS17V`. CPU suite104 discovered/102 pass/
two skips; wrapped-group regression and source manifest updated. Full-model
qualification remains required; serving is unchanged.
Full-model successor `094f114fcaae` started21:42:16UTC with native kernel
fallbacks, the wrapper fix and named-RPC prefill tracing. No verdict yet.
Evidence: k3s `launch-native-trace-wrapped.json`; root
`/tmp/flash-session-gpu-20260921-lW36Rc`. Do not restart on an observation timeout.
