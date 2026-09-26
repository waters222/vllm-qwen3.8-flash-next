"""Isolate pending GDN speculative-state selection at a streaming boundary.

Uses native recurrent CUDA kernels and the native model-state add_request
method, without loading weights. This is a regression reproducer, not a
full-layer or whole-model correctness verdict.
"""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm.third_party.flash_linear_attention.ops import (
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--wrapped", action="store_true")
    args = parser.parse_args()
    if args.wrapped and not args.candidate:
        parser.error("wrapped layouts require --candidate")
    if args.candidate:
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateCopyFuncCalculator,
        )
        from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
        from vllm.v1.kv_cache_interface import (
            KVCacheGroupSpec,
            MambaSpec,
            UniformTypeKVCacheSpecs,
        )
        from vllm.v1.worker.gpu.flash_session_worker import FlashSessionWorker

        state_copy_funcs = (
            MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()
        )
    torch.cuda.set_device(0)
    torch.manual_seed(173)
    common = dict(
        A_log=torch.randn(2, device="cuda"),
        dt_bias=torch.randn(2, device="cuda"),
        inplace_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    def inputs(tokens):
        return dict(
            q=torch.randn(1, tokens, 2, 16, device="cuda", dtype=torch.bfloat16),
            k=torch.randn(1, tokens, 2, 16, device="cuda", dtype=torch.bfloat16),
            v=torch.randn(1, tokens, 2, 16, device="cuda", dtype=torch.bfloat16),
            a=torch.randn(1, tokens, 2, device="cuda"),
            b=torch.randn(1, tokens, 2, device="cuda"),
            cu_seqlens=torch.tensor([0, tokens], dtype=torch.int32, device="cuda"),
        )

    state = torch.randn(5, 2, 16, 16, device="cuda", dtype=torch.float32)
    indices = torch.tensor([[1, 2, 3]], dtype=torch.int32, device="cuda")
    fused_sigmoid_gating_delta_rule_update(
        **common,
        **inputs(3),
        initial_state=state,
        ssm_state_indices=indices,
        num_accepted_tokens=torch.ones(1, dtype=torch.int32, device="cuda"),
    )
    next_inputs = inputs(1)
    rows = []
    for accepted in (1, 2, 3):
        model_state = object.__new__(MambaHybridModelState)
        model_state.rope_state = None
        model_state.prompt_embeds_state = None
        model_state._align_mode = False
        model_state.num_accepted_tokens_gpu = torch.tensor(
            [accepted], dtype=torch.int32, device="cuda"
        )
        expected, _ = fused_sigmoid_gating_delta_rule_update(
            **common,
            **next_inputs,
            initial_state=state.clone(),
            ssm_state_indices=indices,
            num_accepted_tokens=model_state.num_accepted_tokens_gpu,
        )
        # The runner removes/re-adds an existing slot on streaming input.
        # In none mode, add_request resets this value without moving state.
        model_state.add_request(0, SimpleNamespace(num_computed_tokens=128))
        reset = int(model_state.num_accepted_tokens_gpu.item())
        assert reset == 1
        actual, _ = fused_sigmoid_gating_delta_rule_update(
            **common,
            **next_inputs,
            initial_state=state.clone(),
            ssm_state_indices=indices[:, 0],
            num_accepted_tokens=None,
        )
        canonical = state.clone()
        if accepted > 1:
            canonical[1].copy_(canonical[accepted])
        corrected, _ = fused_sigmoid_gating_delta_rule_update(
            **common,
            **next_inputs,
            initial_state=canonical,
            ssm_state_indices=indices[:, 0],
            num_accepted_tokens=None,
        )
        matches = torch.equal(expected, actual)
        assert matches == (accepted == 1)
        assert torch.equal(expected, corrected)
        candidate_checks = []
        if args.candidate:
            for dim_first in (False, True):
                candidate = state.clone()
                conv = torch.arange(5 * 6 * 8, device="cuda").reshape(5, 6, 8).float()
                if dim_first:
                    conv = conv.transpose(1, 2)
                conv_before = conv.clone()
                model_state.num_accepted_tokens_gpu.fill_(accepted)
                spec = MambaSpec(
                    block_size=240000,
                    shapes=(tuple(conv.shape[1:]), tuple(state.shape[1:])),
                    dtypes=(conv.dtype, state.dtype),
                    num_speculative_blocks=2,
                    mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
                    mamba_cache_mode="none",
                )
                if args.wrapped:
                    spec = UniformTypeKVCacheSpecs(
                        block_size=spec.block_size, kv_cache_specs={"gdn": spec}
                    )
                runner = SimpleNamespace(
                    pp_handler=None,
                    req_states=SimpleNamespace(
                        req_id_to_index={"stream": 0},
                        num_computed_tokens=SimpleNamespace(
                            gpu=torch.tensor([128], device="cuda", dtype=torch.int32)
                        ),
                    ),
                    model_state=model_state,
                    vllm_config=SimpleNamespace(
                        cache_config=SimpleNamespace(mamba_cache_mode="none")
                    ),
                    kv_cache_config=SimpleNamespace(
                        kv_cache_groups=[KVCacheGroupSpec(["gdn"], spec)]
                    ),
                    compilation_config=SimpleNamespace(
                        static_forward_context={
                            "gdn": SimpleNamespace(kv_cache=[conv, candidate])
                        }
                    ),
                    model=SimpleNamespace(
                        get_mamba_state_copy_funcs=lambda kinds: {
                            kind: state_copy_funcs for kind in kinds
                        }
                    ),
                )
                with patch(
                    "vllm.model_executor.layers.mamba.mamba_utils.is_conv_state_dim_first",
                    return_value=dim_first,
                ):
                    FlashSessionWorker(runner, None).prepare_streaming_update(
                        SimpleNamespace(
                            req_id="stream",
                            num_computed_tokens=128,
                            block_ids=([1, 2, 3],),
                        )
                    )
                assert model_state.num_accepted_tokens_gpu.item() == 1
                candidate_output, _ = fused_sigmoid_gating_delta_rule_update(
                    **common,
                    **next_inputs,
                    initial_state=candidate,
                    ssm_state_indices=indices[:, 0],
                    num_accepted_tokens=None,
                )
                assert torch.equal(expected, candidate_output)
                offset = accepted - 1
                if dim_first:
                    assert torch.equal(
                        conv[1, :, : 6 - offset], conv_before[1, :, offset:]
                    )
                else:
                    assert torch.equal(conv[1, : 6 - offset], conv_before[1, offset:])
                assert torch.equal(conv[2:], conv_before[2:])
                candidate_checks.append(
                    dict(conv_dim_first=dim_first, byte_exact=True, worker_hook=True)
                )
        rows.append(
            dict(
                accepted_before_reset=accepted,
                accepted_after_add_request=reset,
                unmaterialized_output_equal=matches,
                max_output_difference=float(
                    (expected.float() - actual.float()).abs().max()
                ),
                canonicalized_output_equal=True,
                candidate_checks=candidate_checks,
            )
        )
    report = dict(
        gap_reproduced=True,
        scope="native GDN recurrent core; no conv, PP, model weights, or logits",
        seed=173,
        rows=rows,
        production_fix_qualified=False,
        candidate_tested=args.candidate,
        wrapped_layout=args.wrapped,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    with args.output.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
