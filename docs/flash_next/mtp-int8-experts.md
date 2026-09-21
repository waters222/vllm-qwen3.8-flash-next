# Experimental draft-only INT8 experts

`VLLM_FLASH_MTP_INT8_EXPERTS=1` attaches native online per-channel INT8
weight-only quantization to the separate draft checkpoint quantization config.
It targets only `mtp.layers.48.mlp.experts` and rejects configurations outside
the reviewed Flash-Next TP2/PP2/no-EP, single-head, depth1--4 pilot. Default off
does not inspect the model. Original checkpoint weights are not modified.

Dense draft layers, embeddings and the shared target LM head retain their
checkpoint precision. This is not a fully INT8 draft or an INT8 KV cache.
The first eager draft forward checks method identity, INT8 CUDA weight tensors
and finite scales, and reports resident expert bytes/backend. It must happen
before CUDA graph capture. The existing engine performs conversion and kernels;
the pilot hook does not implement a new quantizer.

The existing bounded phase observer separately accepts MTP only when
`VLLM_FLASH_PP_PHASE_ALLOW_MTP=1`. It records speculative depth and scheduled
token counts without recording token IDs or changing policy. This diagnostic
is not enabled in ordinary timing trials. CPU immutability/privacy tests pass.
Eight full-model MTP2 C2 captures completed: opposite PP phases did not
consistently outperform paired verification. The no-MTP phase-balancing policy
has not been enabled for MTP on this evidence.

The first full-model Q8 attempt hit transient VRAM OOM while creating the
INT8 down matrix alongside both BF16 input matrices and the INT8 gate/up
matrix. A separately opt-in `VLLM_FLASH_MTP_INT8_LOW_PEAK=1` conversion order
processes gate/up first, releases its BF16 storage, then processes down. It
rebinds the old Parameter data because loader arguments retain the object.
This path is restricted to the named MTP module and rejects weight reload.
The arithmetic and TP reduction are unchanged. A real-weight TP2 probe with
all 512 draft experts passed exact equality of both INT8 matrices and both
scales on both ranks. Extra peak allocation fell from 1,278,738,432 to
850,003,456 bytes: 409 MiB saved per rank, including simulated live loader
Parameter references. This validates conversion only, not inference speed.

The source allowlist now adds this native `online/int8.py` file (41 engine
files total), with its original checksum verified against the serving image.
The historical 40-file snapshot/tag remains unchanged.

CPU selection/gate tests pass. The full-model depth2 pilot reached Ready with
both TP ranks attesting 1,258,291,200 INT8 expert weight bytes and the Triton
backend. All 12 target-verified behavior outputs matched the saved fixture.
A three-repeat short-prompt screen admitted four requests with a larger GPU
cache: C1 decode144.61, C2--4 aggregate158.11/209.35/262.87 tokens/s. This is
precision plus cache-budget evidence, not an isolated kernel comparison.
C2 variability remains and C1 draft acceptance dropped versus BF16. Three-repeat
code/prose follow-up (after the bounded observer finished collection) measured
C2 aggregate medians163.73/120.52 and C4 medians277.04/212.36 tokens/s;
acceptance is workload dependent. Two-repeat exact2K/8K synthetic-context
screens completed with four eventually overlapping output streams, including
transient admission delay at8K. These are not long-document quality evaluations.
The balanced-depth comparison and broader qualification remain open. Not promoted.
This AI-assisted experimental change requires human review before an upstream
contribution. Source is preserved separately from the historical image hash.
