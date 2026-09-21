# Optional QSA offload page cap

`VLLM_QSA_OFFLOAD_BLOCK_SIZE_CAP=3072` bounds the automatically derived
attention page length without changing the recurrent-state page size, model
context limit, checkpoint data, or attention arithmetic. Default zero preserves
the original behavior. The cap must be a multiple of16 and no larger than the
automatically computed length.

Motivation: increasing a shared GPU block pool for speculative recurrent
states also enlarges every host-backed attention tensor. Crossing a pinned
allocator size-class boundary can double reserved host memory. At642 blocks,
3536-token pages require a conservative104GiB across13 layers andTP2;3072-token
pages stay within52GiB. Logical attention capacity still increases relative to
the earlier336-block pool. This does not promise four full240K contexts.

Validation: two CPU contract tests cover sizing/budget arithmetic and invalid
caps. An isolated QSA read probe compares3072/3536 layouts, CPU-UVA/GPU backing,
permuted physical pages, both layouts' boundary tokens, and M1/2/3/4/5/8/12/16/20.
All outputs are bit-exact. The1050MB GPU-cache full-model trial reached Ready
but failed a31-token prefill with GPU OOM allocating QSA scratch. Page-boundary
retrieval and concurrency validation did not run. The cap is not qualified by
this trial. This remains opt-in development
code, not a serving default. AI assistance was used; human review is required
before upstream contribution.
