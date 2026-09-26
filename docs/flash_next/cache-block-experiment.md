# Cache-block sizing experiment, 2026-09-24

The user authorized trying2048 cache blocks, then suggested1024. Prefill budget
stays2048; internal checkpoint optimization staysOFF. Neither candidate has a
full-model throughput result or production qualification. Default remains944.

## Implementation

Explicit `VLLM_FLASH_QSA_BLOCK_SIZE` selects944/1024/2048 only for TP4/PP1
Qwen4Exp with QSA host offload; rejects combination with the old page-size cap.
Zero/unset preserves automatic sizing. Native packed grouping determines physical
stride and recurrent packing; no cache hash/refcount/LRU/offload logic changes.
Both guarded launchers freeze the platform override only for this experiment.
Three CPU sizing/guard contracts and scoped Ruff pass; CPU checks are not GPU
qualification. Evidence runs/qsa-block-size-cpu-tests-v2.log.

## Completed reduced runs

| Run | Result | Limitation |
|---|---|---|
|2048 v1,0762a365|Native copies/MTP rejection pass; cold-load assertion fails|First suite was incorrectly expanded to7100 with only32 RAM blocks|
|1024 v1,8eb7a16f|Same cold-load assertion fails|Same fixture issue|
|1024 trace,df8863ab|Cold lookup returns0 before CPU backend lookup|RAM prefix evicted by oversized pressure fixture|
|1024 v2,194ea220|Cold loads/byte checks/32 staging probes per rank pass; numerical screen fails|Both suites2080 because inherited long-shape flag wasOFF|
|944 control,284c1649|Native copies/32 staging probes pass; long numerical/index-digest screen fails|First1920, target7100, budget2048|
|1024 v3,75070ae2|Native copies/32 staging probes pass; long numerical/index-digest screen fails|First2080, target7100, budget2048|

Full identities, exact timestamps, image/source hashes, logs and raw failures
are retained in runs/reduced-block*-{launch,collection}.json and results folders.
All terminal runs were collected before a successor; serving identity remained
unchanged. No active GPU job after final collection.

## Cache and numerical findings

The first fixture must use its original two-block-plus32-token prompt so ten
pressure requests evict GPU state while retaining RAM KV. The target-QSA fixture
has a larger host pool and explicitly enables7100-token coverage. No native
eviction behavior was relaxed to pass this condition.

Corrected1024 long requests obtain a5120-token hot/cold prefix and9044992 cold
load bytes. All four ranks match target/draft RAM-KV digests and target compressed
index digests across hot/cold. The MTP compressed index differs; raw screen fails.
The944 control exhibits the same category of mismatch plus fresh numerical
variation. Every recorded cold prefix digest in both long controls matches the
first fresh reference at the same boundary, establishing the restored data's
recorded provenance.944 hot MTP index matches the second fresh reference;
1024 hot index version is not fully attributed. These facts do not overwrite
the failed screen or establish runtime race freedom.

Evidence: runs/reduced-block{944-control-v1,1024-v3}-{analysis,prefix-provenance}.json.
The analyses hash each source rank report. Corrected2048 was not rerun.

## Remaining work

### 2026-09-26 reference-order control

After API provisioning, native944 reduced container
`eb70a77ce04177287b2ebe1ca0f13b1a62b0ec516f5d4acba3b0fcf6876f8525`
exited1 at03:55:15.436900538Z.82 frozen sources/native library and exact live k3s
serving boundary verified. Command: `vllm/.venv/bin/python
context/infra/inference/reduced/launch_local.py --k3s-serving --qsa-staging
--prefill-metadata-cache --qsa-block-size944 --reference-after-cold
--evidence runs/reduced-cache-single-origin-v1-launch.json --execute`
(CLI spelling is `--qsa-block-size 944`).

Only diagnostic order changes: reference-a, hot, pressure, cold, reference-b.
All four ranks and both reduced fixtures now match every recorded hot/cold
prefix digest, including MTP compressed indices, and GDN entry-state digests.
Native layouts/rejection replay/staging exact checks pass. This supports distinct
repeated-prefill versions as the cause of the earlier recorded index mismatch;
it does not change the native runtime or prove general cache/race integrity.

Raw output screen remains failed. Short fixture: reference repeats/cold exact,
hot differs; hot suffix chunks944+32 versus cold/reference976. Long fixture:
hot944+492 versus cold/reference1436; even fresh repeated outputs differ.
Thus matching stored bytes alone does not resolve numerical/batch-shape effects.
No failed verdict relabeled. `runs/reduced-cache-single-origin-v1-analysis.json`
retains all ranks, original verdicts, schedules and numerical differences.
Analysis command: `vllm/.venv/bin/python context/infra/inference/analyze_flash_reference_order.py
runs/reduced-cache-single-origin-v1-results`.

Resolve MTP compressed-index version differences for the same prefix hash and
retain separate numerical and cache-state verdicts. Then measure full-model
memory capacity and the complete C2/C4 prefill/TG matrix before choosing a block
size. Dividing2048 evenly does not prove1024 faster: the observed native scheduler
still inserts a1024-token checkpoint chunk in the7100-token request.

Raw `runs/` evidence references in this record refer to the development archive;
that archive and site orchestration are not bundled with this public source tree.
