# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GDNAttentionMetadataBuilder.build() — specifically the
reclassification of non-spec decodes as prefills when spec decodes exist.
Covers the fix for https://github.com/vllm-project/vllm/issues/34845.
"""

from dataclasses import dataclass

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec

BLOCK_SIZE = 16
DEVICE = torch.device("cpu")


@dataclass
class GDNBuildTestCase:
    """Specification for a GDN metadata builder classification test."""

    seq_lens: list[int]
    query_lens: list[int]
    num_decode_draft_tokens: list[int] | None  # None = no spec config
    num_speculative_tokens: int
    expected_num_decodes: int
    expected_num_prefills: int
    expected_num_prefill_tokens: int
    expected_num_spec_decodes: int


GDN_BUILD_TEST_CASES = {
    # The original #34845 crash: non-spec query_len=1 + spec decode
    "mixed_decode_and_spec_decode": GDNBuildTestCase(
        seq_lens=[65, 20],
        query_lens=[1, 3],
        num_decode_draft_tokens=[-1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=1,
        expected_num_spec_decodes=1,
    ),
    # All requests are spec decodes — no reclassification needed
    "pure_spec_decode": GDNBuildTestCase(
        seq_lens=[50, 30],
        query_lens=[3, 3],
        num_decode_draft_tokens=[2, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=2,
    ),
    # No speculative config at all — standard decode path
    "pure_regular_decode": GDNBuildTestCase(
        seq_lens=[40, 30, 20],
        query_lens=[1, 1, 1],
        num_decode_draft_tokens=None,
        num_speculative_tokens=0,
        expected_num_decodes=3,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=0,
    ),
    # Multi-token prefill alongside spec decode — no decode to reclassify
    "spec_decode_with_real_prefill": GDNBuildTestCase(
        seq_lens=[100, 20],
        query_lens=[50, 3],
        num_decode_draft_tokens=[-1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=50,
        expected_num_spec_decodes=1,
    ),
    # All three types in one batch — decode gets reclassified
    "prefill_decode_and_spec_decode": GDNBuildTestCase(
        seq_lens=[100, 65, 20],
        query_lens=[50, 1, 3],
        num_decode_draft_tokens=[-1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=2,
        expected_num_prefill_tokens=51,
        expected_num_spec_decodes=1,
    ),
    # Multiple non-spec query_len=1 requests all reclassified
    "multiple_decodes_reclassified": GDNBuildTestCase(
        seq_lens=[40, 50, 60, 20],
        query_lens=[1, 1, 1, 3],
        num_decode_draft_tokens=[-1, -1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=3,
        expected_num_prefill_tokens=3,
        expected_num_spec_decodes=1,
    ),
    # Zero-length padded sequence excluded from counts
    "zero_length_padding_with_spec": GDNBuildTestCase(
        seq_lens=[16, 65, 20],
        query_lens=[0, 1, 3],
        num_decode_draft_tokens=[-1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=1,
        expected_num_spec_decodes=1,
    ),
}


def _create_gdn_builder(
    num_speculative_tokens: int = 0,
    full_cuda_graph: bool = False,
) -> GDNAttentionMetadataBuilder:
    """Create a GDNAttentionMetadataBuilder with minimal config."""
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
    )
    if full_cuda_graph:
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    if num_speculative_tokens > 0:
        vllm_config.speculative_config = SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=num_speculative_tokens,
        )
    mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
    )
    return GDNAttentionMetadataBuilder(
        kv_cache_spec=mamba_spec,
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=DEVICE,
    )


def _build(
    builder: GDNAttentionMetadataBuilder,
    batch_spec: BatchSpec,
    num_decode_draft_tokens: list[int] | None = None,
) -> GDNAttentionMetadata:
    """Build GDN attention metadata, optionally with spec-decode kwargs."""
    common = create_common_attn_metadata(batch_spec, BLOCK_SIZE, DEVICE)
    kwargs: dict = {}
    if num_decode_draft_tokens is not None:
        kwargs["num_decode_draft_tokens_cpu"] = torch.tensor(
            num_decode_draft_tokens, dtype=torch.int32
        )
        kwargs["num_accepted_tokens"] = torch.ones(
            batch_spec.batch_size, dtype=torch.int32, device=DEVICE
        )
    return builder.build(common_prefix_len=0, common_attn_metadata=common, **kwargs)


@pytest.mark.parametrize(
    "test_case", GDN_BUILD_TEST_CASES.values(), ids=GDN_BUILD_TEST_CASES.keys()
)
def test_gdn_build_classification(test_case: GDNBuildTestCase):
    """Test that GDN metadata builder classifies requests correctly."""
    builder = _create_gdn_builder(test_case.num_speculative_tokens)
    batch = BatchSpec(seq_lens=test_case.seq_lens, query_lens=test_case.query_lens)
    meta = _build(builder, batch, test_case.num_decode_draft_tokens)

    assert meta.num_decodes == test_case.expected_num_decodes
    assert meta.num_prefills == test_case.expected_num_prefills
    assert meta.num_prefill_tokens == test_case.expected_num_prefill_tokens
    assert meta.num_spec_decodes == test_case.expected_num_spec_decodes


def test_has_initial_state_after_reclassification():
    """After reclassification, num_prefills > 0 so the prefill kernel path
    should compute has_initial_state. For the reclassified request with
    context_lens > 0, the corresponding entry must be True."""
    builder = _create_gdn_builder(num_speculative_tokens=2)
    batch = BatchSpec(seq_lens=[65, 20], query_lens=[1, 3])
    meta = _build(builder, batch, num_decode_draft_tokens=[-1, 2])

    assert meta.num_prefills > 0, "reclassification should produce prefills"
    assert meta.has_initial_state is not None
    # req0 has context_lens = 65 - 1 = 64 > 0, so has_initial_state[0] = True
    assert meta.has_initial_state[0].item() is True


def test_full_cudagraph_spec_metadata_uses_request_count():
    """FULL cudagraph token padding must not pad request-indexed metadata."""
    num_speculative_tokens = 3
    builder = _create_gdn_builder(
        num_speculative_tokens=num_speculative_tokens,
        full_cuda_graph=True,
    )
    batch = BatchSpec(seq_lens=[80, 96], query_lens=[4, 4])
    meta = _build(builder, batch, num_decode_draft_tokens=[3, 3])

    assert meta.num_spec_decodes == batch.batch_size
    assert meta.num_spec_decode_tokens == batch.compute_num_tokens()
    assert meta.spec_state_indices_tensor is not None
    assert meta.spec_state_indices_tensor.shape == (
        batch.batch_size,
        num_speculative_tokens + 1,
    )
    assert meta.spec_sequence_masks is not None
    assert meta.spec_sequence_masks.shape == (batch.batch_size,)
    assert meta.spec_query_start_loc is not None
    assert meta.spec_query_start_loc.shape == (batch.batch_size + 1,)
    assert meta.num_accepted_tokens is not None
    assert meta.num_accepted_tokens.shape == (batch.batch_size,)


def make_uniform_spec_fixture(n, mode, device):
    """Local tensor-only fixture; no model download or checkpoint loading."""
    from types import SimpleNamespace as NS

    from vllm.v1.attention.backend import CommonAttentionMetadata

    config = NS(
        compilation_config=NS(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            max_cudagraph_capture_size=30,
        ),
        speculative_config=NS(num_speculative_tokens=4, parallel_drafting=False),
        scheduler_config=NS(max_num_seqs=6),
        parallel_config=NS(decode_context_parallel_size=1),
        cache_config=NS(mamba_cache_mode=mode),
        additional_config={"gdn_prefill_backend": "triton"},
        model_config=NS(hf_text_config=NS(linear_key_head_dim=128)),
    )
    spec = MambaSpec(
        block_size=472, shapes=((16, 64),), dtypes=(torch.float32,),
        num_speculative_blocks=4,
    )
    builder = GDNAttentionMetadataBuilder(spec, ["layer.0"], config, device)
    query_cpu = torch.arange(0, (n + 1) * 5, 5, dtype=torch.int32)
    columns = 80 if mode == "align" else 5
    table = torch.arange(n * columns, dtype=torch.int32, device=device).view(n, -1)
    lengths = torch.tensor([0, 1, 471, 472, 473, 32768][:n],
                           dtype=torch.int32, device=device)
    common = CommonAttentionMetadata(
        query_start_loc=query_cpu.to(device), query_start_loc_cpu=query_cpu,
        seq_lens=lengths, num_reqs=n, num_actual_tokens=n * 5,
        max_query_len=5, max_seq_len=32768, block_table_tensor=table,
        slot_mapping=torch.zeros(n * 5, dtype=torch.int64, device=device),
    )
    accepted = torch.arange(n, dtype=torch.int32, device=device) % 5 + 1
    drafts = torch.full((n,), 4, dtype=torch.int32)
    return builder, common, accepted, drafts


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA staging kernel")
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("mode", ["none", "align"])
def test_fused_spec_metadata_matches_native_and_graph_replay(n, mode, monkeypatch):
    """State selection, accepted counts and captured pointers survive row updates."""
    from dataclasses import fields

    from vllm.v1.attention.backends import flash_gdn_spec_metadata as fused

    device = torch.device("cuda", torch.cuda.current_device())
    builder, common, accepted, drafts = make_uniform_spec_fixture(n, mode, device)
    monkeypatch.setattr(fused, "ENABLED", False)

    def snapshot(meta):
        return {field.name: (value.clone() if isinstance(value, torch.Tensor) else value)
                for field in fields(meta) if (value := getattr(meta, field.name)) is not None}

    reference = snapshot(builder.build(0, common, accepted, drafts))
    monkeypatch.setattr(fused, "ENABLED", True)
    actual = fused.try_build(builder, common, GDNAttentionMetadata, accepted, drafts, False)
    assert actual is not None
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        captured = snapshot(actual)
    for iteration in range(3):
        if iteration:
            common.block_table_tensor.add_(7)
            common.seq_lens.copy_(torch.roll(common.seq_lens, 1))
            accepted.copy_(accepted % 5 + 1)
            monkeypatch.setattr(fused, "ENABLED", False)
            reference = snapshot(builder.build(0, common, accepted, drafts))
            monkeypatch.setattr(fused, "ENABLED", True)
        actual = fused.try_build(builder, common, GDNAttentionMetadata, accepted, drafts, False)
        assert actual is not None
        graph.replay()
        assert captured.keys() == reference.keys()
        for name, expected in reference.items():
            value = captured[name]
            if isinstance(expected, torch.Tensor):
                assert torch.equal(value, expected), name
            else:
                assert value == expected, name


@pytest.mark.parametrize("unsupported", ["partial_drafts", "token_padding", "fast_build"])
def test_fused_spec_metadata_falls_back_without_writing(unsupported, monkeypatch):
    """Unqualified batches leave persistent buffers untouched for native build."""
    from vllm.v1.attention.backends import flash_gdn_spec_metadata as fused

    builder, common, accepted, drafts = make_uniform_spec_fixture(2, "align", DEVICE)
    builder.spec_state_indices_tensor.fill_(-123)
    monkeypatch.setattr(fused, "ENABLED", True)
    if unsupported == "partial_drafts":
        drafts[0] = 3
    elif unsupported == "token_padding":
        common.num_actual_tokens += 5
    assert fused.try_build(builder, common, GDNAttentionMetadata, accepted, drafts,
                           unsupported == "fast_build") is None
    assert torch.all(builder.spec_state_indices_tensor == -123)
