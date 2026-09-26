# Bounded prefill tuning — 2026-09-25

Retain the current runtime. The user accepted the remaining historical prefill
gap and requested a bounded set of experiments. This component sweep found no
alternative worth advancing to a full-model benchmark.

On each of GPUs0–3, test160/944/1104 prefill rows at16k/32k context with
1/2/4/8 warps. Keep tile width and the attention formula unchanged. Use the real
944-token cache pages,12-layer host strides, six query heads,256 head dimension,
2052-column packed selection and96MiB staging arena. Three changing-input checks
exercise query, host KV and block-table updates; five alternating timing repeats
follow warmup. All RAM-byte checks and finite-output checks pass.

| Rows | Context | Existing warps | Fastest alternative | Component speed increase |
|---|---:|---:|---:|---:|
|160|16384|1|4|3.9%|
|944|16384|2|8|2.2%|
|1104|16384|2|8|1.8%|
|160|32768|1|4|2.2%|
|944|32768|2|8|2.0%|
|1104|32768|2|8|2.0%|

Ratios use the median across four per-GPU ratios of five-repeat medians.
These are QSA component results, not end-to-end prefill improvements.
Every alternative fails exact-output equality in at least one check; maximum
absolute differences are0.000244–0.000488. Existing settings reproduce themselves
exactly. Other alternatives slow the component substantially. The combined exact
component screen therefore remains false; small numerical differences are not
relabelled as passes or RAM-cache corruption.

Exact container `03cba1edd98a5a8bb16c2235f4c865d8457f776c629f04aa2f3be7ef08610cad`
exited0 at2026-09-25T19:56:02.625708443Z. All four frozen source hashes and exact
stopped-peer identity verified. Four existing fallback tests passed; six cases
completed on each GPU. Evidence: `runs/prefill-warp-sweep-v1-{launch,collection,
analysis}.json` and adjacent results/log. The frozen benchmark hash is recorded
in the launch; subsequent local lint-only edits bind its loop closure explicitly
and wrap two lines. The final benchmark passes Ruff and Python compilation.

No engine defaults, attention kernels, CPU frequency settings or TG paths were
changed. Latest full C4/16k prefill remains3512.2 tok/s, compared with historical
3788.7; no new full-model rate is claimed. The earlier first-use staged-QSA/MoE
warmup gap remains a separate known issue; this sweep does not fix it.
No further steady-prefill tuning is scheduled under this bounded experiment.

Raw `runs/` evidence references in this record refer to the development archive;
that archive and site orchestration are not bundled with this public source tree.
