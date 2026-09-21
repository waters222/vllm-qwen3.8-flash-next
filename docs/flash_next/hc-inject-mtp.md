# Larger-row HC injection experiment

`VLLM_FLASH_HC_INJECT_MTP=1` extends the existing injection opt-in to M5--16.
It does not enable injection by itself. Ordinary execution, exact SM86 BF16
N4/K10240 shapes, native unquantized replicated linear, and existing no-bias,
no-LoRA guards still apply. M1--4 dispatch and M17+ native fallback are unchanged.
No weights, scales, attention state or graph shapes change.

The target TP4 MTP3 trace showed96 native skinny GEMMs per verification pass
at C2/C4, matching two HC injections per layer. The one-layer operator screen
reproduced that exact kernel name. Native M5--16 injection costs approximately
71--75us; independent-row split-K takes2.8--3.7us. Native M17+ selects a faster
split-K GEMM (~6us), so the new serving gate does not replace that path.

Two real TP2 ranks, both layer0 HC blocks, every M1--20, changed inputs and
zero/masked rows passed native/FP32 and CUDA-graph checks. Larger inputs
concatenate normalized rows captured across teacher-forced steps. These are
operator inputs, not actual speculative trajectories.

A whole-layer prototype with actual TP2 reductions passed64-step output/state
and changed-input graph checks at M4/5/8/12/16/20. M4 was unchanged and exact;
recurrent states matched exactly. Native/candidate rank0 layer times (us):

|M|Native|Candidate|
|---|---:|---:|
|4|358.62|359.97|
|5|576.87|380.52|
|8|637.84|440.49|
|12|714.03|509.34|
|16|739.78|526.14|
|20|620.83|614.72|

M20 was tested but is not enabled: its ~1% layer gain is small and its maximum
output difference (.01367, within the existing relative/absolute gate) is
larger than the <=.001953125 difference across both M5--16 windows. Do not read these hot-layer
numbers as full-model gains. The prototype uses ordinary independent GDN
decode states and HC opt-ins, not every full-model projection optimization.

Evidence is retained in the k3s operations repository under
`docs/benchmarks/flash-next-perf-20260918/tp4-current/hc-inject-mtp/`.
Installed custom-op/fake/compile integration subsequently passed on both real
TP2 ranks in a fresh original-image container with only two read-only source
overlays. Both HC blocks and every M1--20 passed native/FP32 and unchanged-path
checks. Each rank passed24 exact compiled checks at M5/8/12/16 with changed
inputs/weights. Actual installed whole-layer on/off graphs and64-step state
checks passed; M4 and M20 are exact unchanged negative controls. No HC methods
were rebound. A restored dispatch counter verified the installed kernel ran.

This qualifies the bounded integration screen, not complete serving-model
compilation. Default is off; no production promotion.

## Completed TP4 Q8 MTP3 full-model screen

Target AWQ unchanged; Q8 draft-expert weights only, BF16 activations. TP4/PP1,
no EP, 750MB GPU cache/rank, 56GiB host QSA, 240K context. The configuration
and old source payloads match previous MTP3 except the two HC source overlays,
one feature flag and generated container hostname. Three512-token repeats per
class/concurrency, seed17 and thinking off:

|Class|C1 decode|C2 aggregate|C3 aggregate|C4 aggregate|
|---|---:|---:|---:|---:|
|Math|164.07|243.06|316.50|372.42|
|Code|167.28|240.74|339.04|406.26|
|Prose|97.74|162.63|239.56|281.10|

All36 batches pass token/timing/counter audits, with zero preemptions and
expected output overlap. Compared with previous same-cache MTP3, C2--4 gain
22.9--36.7%. Math/code C1 change -0.4%/-2.8%, while prose C1 is20% lower.
That lower result remains unresolved; do not claim universal improvement.
Acceptance C1/C2/C3/C4: math72.38/76.75/75.73/74.76%,
code82.47/83.69/82.10/83.35%, prose42.86/44.52/44.41/45.68%.

All12 sequential fixture outputs match the historical result, including its
known case-sensitive `No` mismatch. Separate concurrent checks passed58/58,
including a longer integer-array output with observed overlap1/2/3/4 and
zero preemptions. The concurrent yes/no check is explicitly case-insensitive.
Greedy free-form outputs varied in both old and new runs, including noMTP;
token equality is not asserted and this is not broad model-quality evaluation.

NoMTP comparisons retain a cache-budget difference (275MB versus750MB/rank).
Depth1/2 comparisons and the balanced default remain pending. Full-model
performance evidence and raw data are in the operations repository's
`tp4-current/q8-mtp3-hc/` benchmark directory.

## Completed TP4 Q8 MTP4 full-model screen

The next depth uses identical source payloads, cache paths, memory budgets
and runtime configuration except speculative depth, required graph sizes and
container-specific source roots/hostname. Three512-token repeats per class/C:

|Class|C1 decode|C2 aggregate|C3 aggregate|C4 aggregate|
|---|---:|---:|---:|---:|
|Math|168.13|235.99|313.60|361.54|
|Code|165.47|251.03|352.25|418.28|
|Prose|110.72|167.36|214.48|253.99|

All36 batches pass the same audits with zero preemptions and expected output
overlap. All12 sequential fixture outputs match, and separate concurrent
checks pass58/58, including long-array overlap1/2/3/4. This is bounded
behavior qualification, not broad model-quality evaluation.

Relative to patchedMTP3, concurrent code gains3.0--4.3%; math changes
-0.9 to-2.9%; prose C2 gains2.9% but C3/C4 lose10.5%/9.6%. Fourth-position
acceptance per draft is11.5--12.9% for concurrent prose and59.7--62.6% for
concurrent code. More draft tokens are not uniformly better.

Relative to unpatched same-depth4, C1 improves27.2--42.8%, C2/C3 improve
22.3--29.4%, and C4 changes-1.6 to+4.1%. C4's20-row target verification is
outside the new HC gate. These are end-to-end observations across retained
windows, not isolated kernel measurements.

Keep the variation: math C1 samples are168.13/101.73/169.41tok/s. The slower
sample has higher acceptance but client intervals near41ms rather than22ms.
The intermittent slowdown is not specific to prose. Client intervals cannot
separate GPU work, scheduling or transport; its cause remains unresolved.
Artifacts are under the operations repository's `tp4-current/q8-mtp4-hc/`.
No production promotion or final balanced-depth selection has been made.

## Completed TP4 Q8 MTP2 matrix and postchecks

The same controlled three-repeat512-token matrix completed with zero
preemptions and expected output overlap in every batch. All12 sequential
fixture outputs match. Configuration/source equality against MTP3 permits
only depth, corresponding graph sizes and container-specific roots/hostname.

|Class|C1 decode|C2 aggregate|C3 aggregate|C4 aggregate|
|---|---:|---:|---:|---:|
|Math|138.69|217.74|293.59|372.20|
|Code|124.13|229.19|273.58|391.41|
|Prose|113.42|184.38|243.37|296.88|

Acceptance C1/C2/C3/C4: math84.03/81.32/81.39/82.72%,
code87.90/90.18/89.63/89.27%, prose53.44/56.71/55.04/54.33%.
Higher acceptance does not ensure higher throughput: fewer accepted tokens
per target pass and verification/draft cost also matter.

Across equally weighted math/code/prose C2--4 cases, the geometric mean of
MTP2/MTP3 throughput ratios is0.9682; MTP4/MTP3 is0.9843. These small margins
are not confidence bounds and do not establish a universal best depth.
MTP2 benefits prose relative to MTP3 but loses on several math/code cases.
Retain variability: code C2 aggregates236.96/194.60/229.19; prose C1 decode
113.42/114.23/69.49tok/s. The slower C1 run again has longer client intervals,
not a diagnosed cause. Separate concurrent checks passed58/58 with zero
preemptions and long-array output overlap1/2/3/4. Eight C1 padded-context
measurements (two each at2K/8K/16K/32K) passed with zero preemptions. Their
median input-token/TTFT rates are3935/3912/3817/3807tok/s and decode medians
114.34/116.74/114.36/104.82tok/s. The32K decode samples92.57/117.06 retain
timing variability. This is not240K or concurrent long-context qualification.
Depth1 subsequently completed as recorded below; a same750MB-cache no-MTP
control is still required before selection.

## Completed TP4 Q8 MTP1 matrix and postchecks

The same three-repeat512-token matrix passed all36 batches with zero
preemptions and expected output overlap. All12 sequential fixture outputs
match, and all58 concurrent behavior checks passed, including long-array
overlap1/2/3/4. Runtime/source comparison permits only draft depth, graph sizes
and container-specific source roots/hostname.

|Class|C1 decode|C2 aggregate|C3 aggregate|C4 aggregate|
|---|---:|---:|---:|---:|
|Math|98.08|190.20|247.45|321.62|
|Code|89.01|184.43|250.58|327.00|
|Prose|79.08|174.27|243.48|294.79|

Acceptance C1/C2/C3/C4: math89.62/89.27/88.42/89.62%,
code93.20/92.23/91.67/92.17%, prose70.94/70.44/71.48/69.06%.
On the equal-weight concurrent score, depth1/depth3 is0.8664, compared with
depth2/depth3 at0.9682 and depth4/depth3 at0.9843. Depth3 provisionally leads
the tested workload mix; the small margins among2--4 do not establish
statistical separation or a universal optimum. All repeats, including slow
C1 samples, are retained. Artifacts are in the operations repository under
`tp4-current/q8-mtp1-hc/`.

The matched no-MTP control uses the same750MB cache, source payloads, host
KV allowance and runtime configuration, excluding speculation/Q8 flags and
the graph sizes required by drafting. Its comparison remains pending. No
production setting has changed.

## Matched no-MTP comparison

The same750MB-cache no-MTP control completed the same36 timing batches,
sequential fixture and58 concurrent behavior checks. The full five-arm audit
passes: identical benchmark client/source pins and configured memory budgets,
zero preemptions, expected output overlap, complete streamed token counts and
settled speculative counters. No-MTP has no speculative counter increments.
The differences are draft configuration/Q8 flags and required graph sizes.

|Class|C1 decode|C2 aggregate|C3 aggregate|C4 aggregate|
|---|---:|---:|---:|---:|
|Math|87.33|137.28|194.13|243.67|
|Code|88.12|137.59|194.42|245.25|
|Prose|88.66|137.57|194.38|244.77|

The selection score is the geometric mean of throughput ratios across equally
weighted math/code/prose and C2/C3/C4. C1 decode is reported separately, not
mixed with concurrent aggregate throughput.

|Draft depth|C2--4 score gain|Worst tested concurrent gain|C1 decode score gain|
|---|---:|---:|---:|
|1|29.53%|20.43%|0.39%|
|2|44.75%|21.29%|41.97%|
|3|49.50%|14.84%|57.83%|
|4|47.15%|3.76%|65.27%|

Depth3 is the observed balanced-score leader. Relative to no-MTP, it gains
77.05/63.04/52.84% on math,74.97/74.39/65.65% on code, and
18.22/23.25/14.84% on prose at C2/3/4. Depth2 has a stronger weakest-case
gain in this small workload set and can be preferable for prose-heavy traffic;
depth4 slightly favors concurrent code. The2--4 score margins are small and
not statistical confidence bounds. Keep all slow C1 samples and their
unresolved timing variation; do not claim a universal best depth.

The control also passed eight synthetic padded-math C1 context measurements,
two each at2K/8K/16K/32K with256 output tokens and zero preemptions. Median
input-token/TTFT rates are4063/4075/4060/3996tok/s; decode medians are
77.80/77.49/77.30/77.49tok/s. This prefill proxy includes API/first-token
overhead and is not isolated GPU prefill time. No240K or concurrent long-context
performance claim is made. Raw comparison/audit evidence remains in the operations
repository at `tp4-current/hc-depth-final-comparison.json` and `hc-control/`.

## Selected depth3 context confirmation and recommendation

An intentional restart of the retained depth3 trial preserved its exact runtime
configuration and ten source payloads. The fresh sequential fixture matched
all12 historical outputs. Two C1 repeats at each exact context length passed
token/counter audits with zero preemptions; both client scripts match the
control. Original throughput samples were not replaced.

|Context|No-MTP prefill proxy|MTP3 prefill proxy|No-MTP decode|MTP3 decode|
|---|---:|---:|---:|---:|
|2K|4063|3951|77.80|101.34|
|8K|4075|3905|77.49|116.21|
|16K|4060|3881|77.30|123.81|
|32K|3996|3806|77.49|124.26|

All rates are tok/s; prefill proxy means input tokens divided by TTFT. Decode
uses256 output tokens after synthetic math-context padding. Depth3 acceptance
is80.35/71.34/80.44/80.67% at2K/8K/16K/32K. Keep the2K decode samples
125.06/77.63 rather than claiming a uniform latency benefit. Its timing
variability remains unresolved. Prefill is modestly slower with MTP here.

Recommend **MTP=3 with Q8 draft-expert weights and BF16 activations** for this
tested TP4/PP1/noEP workload mix. This is a balanced empirical choice, not a
statistically unique optimum or whole-head W8A8 qualification. The target AWQ
weights are unchanged. Depth2 is a reasonable prose-heavy alternative; depth4
slightly favors code but has a weaker worst-case result. All five arms passed
the full180-batch timing and290-answer bounded concurrent audit. The selected
experimental trial and unchanged serving peer are healthy/idle after testing.
No production promotion or clean consolidated-image qualification is implied.
