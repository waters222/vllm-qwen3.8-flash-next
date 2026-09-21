# Experimental TP4 expert-wise padding and repacking

`VLLM_FLASH_TP4_CHUNKED_REPACK=1` is a default-off initial conversion pilot for
the exact512-expert, hidden2560, local-intermediate160, group32 INT4 TP4/noEP
layout. The native Marlin format pads the intermediate width to192. The pilot
performs identical padding one expert at a time, then invokes the unchanged
native repacker and copies into the final output bank. Original input tensors
remain unchanged. Scales, zero points and bias use the existing conversion.

This targets conversion peak memory, not steady-state size or inference speed.
It must first pass real-layer/all-expert/four-local-shard bitwise comparison
and peak-allocation measurement. It does not establish distributed TP4 model
correctness, runtime memory capacity, or MTP compatibility. No GPU qualification
yet for full-model inference. Disable the flag to recover the original conversion path.

The real-layer/all512-expert/four-local-shard GPU probe passed bitwise comparison
for all weights/scales/zero points and verified original inputs unchanged.
Extra peak allocation decreased657.72 to447.31MiB (210.41MiB saved per shard).
Final allocation is unchanged. This was four logical shards on one GPU, not a
distributed-model validation. A separate55-case TP4 RAM-KV layout screen passed
44 rank-local bit-exact checks and11 full-head comparisons (max error0.0004883).
Full-model qualification remains pending. The explicit `VLLM_FLASH_QSA_TP4=1`
gate permits replicated-KV layout while counting all physical host pools.

The source allowlist now includes `fused_moe/oracle/int_wna16.py`: its historical
hash is the previously unchanged upstream/image source; its revised hash pins
this experiment. The allowlist has42 Python engine files and54 total exported
build inputs. This is source preservation, not production qualification.
