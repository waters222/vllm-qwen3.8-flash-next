# Publisher prerequisites

The initial preservation commit applies these patches from Minachist's
`Qwen3.8-Flash-Next-INT4-Mixed-AutoRound` repository, revision
`1703ff595285561b141648609291377d3faf3b64`:

1. `vllm-patch/3x3090/flash-next-vllm.patch`
2. `vllm-patch/3x3090/flash-next-decode-01-ple-host-gather.patch`
3. `vllm-patch/3x3090/flash-next-decode-02-model-state-hook.patch`
4. `vllm-patch/4x3090/flash-next-mtp-01-enable.patch`

Source: <https://huggingface.co/Minachist/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound/tree/1703ff595285561b141648609291377d3faf3b64/vllm-patch>.

These are prerequisite code changes, not a claim that all configurations
were validated. The preserved serving configuration does **not** enable MTP
or image input. MTP support code was already present in the installed tree.

The base is upstream vLLM commit
`dc36fcce902a63eab06c1b93a5c4a5ee178a0c56`. The publisher's patches applied
without context changes. Their source bytes are preserved, including existing
formatting. See `NOTICE` for copyright and Apache-2.0 attribution; `LICENSE.MIT`
is retained for the publisher's ancillary deployment materials. Model weights
are not included and are not relicensed by this fork.
