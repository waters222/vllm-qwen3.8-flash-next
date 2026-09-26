"""Reduced real-weight GDN/MTP check and CPU-inclusive metadata benchmark."""

from copy import copy
from dataclasses import fields
import json
from pathlib import Path
import statistics
import time


def run(context, parameters):
    if parameters:
        raise ValueError("Fixed metadata experiment takes no parameters")
    import torch
    from vllm.config.compilation import CUDAGraphMode
    from vllm.forward_context import set_forward_context
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends import flash_gdn_spec_metadata as fused
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder

    target, device = context["layer"], context["device"]
    config = copy(context["config"])
    config.compilation_config = copy(config.compilation_config)
    config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    config.compilation_config.max_cudagraph_capture_size = 30
    config.scheduler_config = copy(config.scheduler_config)
    config.scheduler_config.max_num_seqs = 6
    spec = target.linear_attn.get_kv_cache_spec(config)
    assert spec.num_speculative_blocks == 4
    original_states = target.linear_attn.kv_cache
    enabled_before = fused.ENABLED
    checks, timings = [], []
    try:
        for n in (2, 4, 6):
            builders = [GDNAttentionMetadataBuilder(
                spec, [target.linear_attn.prefix], config, device) for _ in range(37)]
            query_cpu = torch.arange(0, (n + 1) * 5, 5, dtype=torch.int32)
            table = (torch.arange(80, dtype=torch.int32, device=device)[None, :] % 5
                     + torch.arange(n, dtype=torch.int32, device=device)[:, None] * 5 + 1)
            block = spec.block_size
            lengths = torch.tensor([block - 1, block, block + 1, 2 * block,
                                    2 * block + 1, 69 * block][:n],
                                   dtype=torch.int32, device=device)
            common = CommonAttentionMetadata(
                query_start_loc=query_cpu.to(device), query_start_loc_cpu=query_cpu,
                seq_lens=lengths, num_reqs=n, num_actual_tokens=n * 5,
                max_query_len=5, max_seq_len=69 * block, block_table_tensor=table,
                slot_mapping=torch.zeros(n * 5, dtype=torch.int64, device=device),
            )
            accepted = torch.arange(n, dtype=torch.int32, device=device) % 5 + 1
            drafts = torch.full((n,), 4, dtype=torch.int32)
            states = tuple(torch.zeros((n * 5 + 1, *state.shape[1:]),
                                       dtype=state.dtype, device=device)
                           for state in original_states)
            target.linear_attn.kv_cache = states
            inputs = context["inputs"][0]
            inputs = inputs.repeat((n + inputs.shape[0] - 1) // inputs.shape[0], 1)[:n]
            inputs = inputs.repeat_interleave(5, dim=0).contiguous()
            positions = (lengths[:, None].to(torch.int64) - 5
                         + torch.arange(5, device=device)).flatten()
            outputs = []
            metadata_snapshots = []
            attention_outputs = []
            for enabled in (False, False, True, True):
                for state in states:
                    state.zero_()
                fused.ENABLED = enabled
                metadata = builders[0].build(0, common, accepted, drafts)
                if enabled:
                    assert fused.try_build(builders[0], common, type(metadata),
                                           accepted, drafts, False) is not None
                metadata_snapshots.append({
                    field.name: value.clone() if isinstance(value, torch.Tensor) else value
                    for field in fields(metadata)
                    if (value := getattr(metadata, field.name)) is not None
                })
                hook = target.linear_attn.register_forward_hook(
                    lambda module, args, output: attention_outputs.append(output.clone()))
                with set_forward_context({target.linear_attn.prefix: metadata},
                                         context["config"], num_tokens=n * 5):
                    hidden = target.mlp_hyper_connection.combine(*target(
                        inputs.clone(), None, None, positions, input_ids=None,
                        query_start_loc=None, ngram_context=None))
                hook.remove()
                torch.cuda.synchronize()
                outputs.append((hidden.clone(), tuple(s.clone() for s in states)))
            comparisons = []
            for left, right in ((0, 1), (0, 2), (2, 3)):
                metadata_equal = metadata_snapshots[left].keys() == metadata_snapshots[right].keys()
                for name, value in metadata_snapshots[left].items():
                    other = metadata_snapshots[right][name]
                    metadata_equal &= (torch.equal(value, other)
                                       if isinstance(value, torch.Tensor) else value == other)
                comparisons.append(dict(
                    left=left, right=right, metadata_exact=metadata_equal,
                    attention_exact=torch.equal(attention_outputs[left], attention_outputs[right]),
                    output_exact=torch.equal(outputs[left][0], outputs[right][0]),
                    max_output_error=float((outputs[left][0] - outputs[right][0]).abs().max()),
                    states_exact=all(torch.equal(a, b) for a, b in
                                     zip(outputs[left][1], outputs[right][1])),
                ))
            checks.append(dict(requests=n, target_rows=n * 5, comparisons=comparisons))
            Path(f"/results/spec-metadata-control-rank{context['rank']}.json").write_text(
                json.dumps(checks, indent=2))
            assert all(c["metadata_exact"] and c["attention_exact"] and c["states_exact"]
                       for c in comparisons), "Metadata/attention/state control failed"

            samples = {False: [], True: []}
            for iteration in range(8):
                for enabled in ((False, True) if iteration % 2 == 0 else (True, False)):
                    fused.ENABLED = enabled
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    for builder in builders:
                        builder.build(0, common, accepted, drafts)
                    torch.cuda.synchronize()
                    if iteration > 1:
                        samples[enabled].append((time.perf_counter() - start) * 1000)
            timings.append(dict(requests=n, groups=len(builders),
                                native_ms=statistics.median(samples[False]),
                                fused_ms=statistics.median(samples[True]),
                                samples_ms={str(k): v for k, v in samples.items()}))
    finally:
        target.linear_attn.kv_cache = original_states
        fused.ENABLED = enabled_before

    # Retain the existing selected-target + MTP native offload lifecycle screen.
    import experiment_flash_layer0_mtp_cold as lifecycle
    cache = lifecycle.run(context, {})
    return dict(screen_passed=cache["screen_passed"], metadata_checks=checks,
                full_layer_output_exact=all(c["output_exact"] for check in checks
                                            for c in check["comparisons"]),
                metadata_timings=timings, cache_lifecycle=cache,
                scope="Reduced target equality and MTP cache lifecycle; no full-model claim")
