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

| M | Native | Candidate |
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
compilation or speculative GDN/QSA states. Full-model MTP behavior/throughput
and depth1--4 selection remain pending. Default is off; no production promotion.
