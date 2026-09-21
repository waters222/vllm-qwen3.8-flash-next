"""Compile isolated SM86 BF16/U4/group32 Marlin K32 tiles for K=160.

Reuse the unchanged arithmetic template; do not replace any installed operator.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import build_flash_marlin_pipeline as base

NAMESPACE = 'flash_marlin_k32'
TILES = ((32, 256, 128),)
M_BLOCKS = ((1, True), (1, False), (2, False), (3, False))


def generate(source):
    base.require(hashlib.sha256(source.encode()).hexdigest() == base.PINS[base.PREFIX+'ops.cu'], 'Source differs')
    source = base.replace_once(source, '#include "kernel.h"', '#include "kernel.h"\n#include "marlin_template.h"')
    selectors = []
    for k, n, threads in TILES:
      for m_blocks, m8 in M_BLOCKS:
        flag = 'true' if m8 else 'false'
        selectors.append(f'''  if (a_type == vllm::kBFloat16 && b_type == vllm::kU4 &&
      c_type == vllm::kBFloat16 && s_type == vllm::kBFloat16 &&
      thread_m_blocks == {m_blocks} && thread_n_blocks == {n//16} && thread_k_blocks == 2 &&
      m_block_size_8 == {flag} && has_zp && group_blocks == 2 && threads == {threads} &&
      !is_zp_float && stages == 4)
    kernel = Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(),
                    vllm::kBFloat16.id(), {threads}, {m_blocks}, {n//16}, 2, {flag}, 4, 2, false>;''')
    source = base.replace_once(source, '#include "kernel_selector.h"', '\n'.join(selectors))
    source = base.replace_once(source, 'th_config.thread_k < min_thread_k', 'th_config.thread_k < 32')
    source = base.replace_once(source, 'thread_config_t{thread_k, thread_n, thread_k * thread_n / 64}',
                               'thread_config_t{thread_k, thread_n, 128}')
    source = base.replace_once(source, '  // Set thread config', '''  STD_TORCH_CHECK(major_capability == 8 && minor_capability == 6,
                  "K32 probe requires SM86");
  STD_TORCH_CHECK(a_type == vllm::kBFloat16 && b_type == vllm::kU4 &&
                  c_type == vllm::kBFloat16 && s_type == vllm::kBFloat16 &&
                  group_size == 32 && has_zp && !is_zp_float && !has_bias &&
                  !use_atomic_add && use_fp32_reduce && num_experts == 512 &&
                  (moe_block_size == 8 || moe_block_size == 16 ||
                   moe_block_size == 32 || moe_block_size == 48),
                  "K32 probe precision/layout differs");
  STD_TORCH_CHECK(prob_n == 2560 && prob_k == 160 && prob_m >= 10 &&
                  prob_m <= 20480 && prob_m % 10 == 0 && top_k == 1 &&
                  mul_topk_weights && thread_k == 32 &&
                  thread_n == 256,
                  "K32 probe requires unpadded TP4 expert-down M1--2048");
  // Set thread config''')
    source = source.replace('moe_wna16_marlin_gemm', NAMESPACE+'_gemm')
    source = base.replace_once(source, 'STABLE_TORCH_LIBRARY_IMPL(_moe_C, CUDA, m)',
                               f'STABLE_TORCH_LIBRARY_IMPL({NAMESPACE}, CUDA, m)')
    source = base.replace_once(source, f'm.impl("{NAMESPACE}_gemm"', 'm.impl("gemm"')
    return '#define MARLIN_NAMESPACE_NAME '+NAMESPACE+'\n'+source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    base.require(args.output.is_dir() and not list(args.output.iterdir()), 'Require empty output')
    base.require(not args.output.resolve().is_relative_to(args.source.resolve()), 'Separate source/output')
    base.require(all(base.digest(args.source/p) == h for p, h in base.PINS.items()), 'Source pin differs')
    base.require(os.environ.get('TORCH_CUDA_ARCH_LIST') == '8.6' and os.environ.get('MAX_JOBS') == '2', 'Build bounds differ')
    import torch
    import vllm
    from torch.utils.cpp_extension import load
    base.require(not torch.cuda.is_available(), 'Build without GPUs')
    extension = Path(vllm.__file__).parent/'_moe_C_stable_libtorch.abi3.so'
    base.require(base.digest(extension) == base.NATIVE_EXTENSION, 'Native library differs')
    unit = args.output/(NAMESPACE+'.cu')
    unit.write_text(generate((args.source/(base.PREFIX+'ops.cu')).read_text()))
    report = dict(status='building', namespace=NAMESPACE, source_pins=base.PINS,
                  builder_sha256=base.digest(__file__), base_sha256=base.digest(base.__file__),
                  generated_sha256=base.digest(unit), native_extension_sha256=base.NATIVE_EXTENSION,
                  private_schema=base.PRIVATE_SCHEMA, torch_version=torch.__version__,
                  cuda_version=torch.version.cuda, m_blocks=M_BLOCKS,
                  gpu_validation=False, serving_changes=False)
    output = args.output/'build-report.json'
    output.write_text(json.dumps(report, indent=2)+'\n')
    owner = torch.library.Library(NAMESPACE, 'DEF')
    owner.define(base.PRIVATE_SCHEMA)
    build = args.output/NAMESPACE
    build.mkdir()
    path = load(name=NAMESPACE, sources=[str(unit)],
                extra_include_paths=[str(args.source), str(args.source/base.PREFIX)],
                extra_cflags=['-O3', '-DUSE_CUDA'],
                extra_cuda_cflags=['-O3', '-DUSE_CUDA', '--expt-relaxed-constexpr',
                                  '--expt-extended-lambda', '-Xptxas=-v'],
                build_directory=str(build), is_python_module=False, verbose=True)
    base.require(str(torch.ops.flash_marlin_k32.gemm.default._schema).startswith(NAMESPACE+'::gemm('), 'Private registration failed')
    base.require(base.digest(extension) == base.NATIVE_EXTENSION, 'Native library changed')
    report.update(status='compiled_and_loaded_no_gpu', library=str(path), library_sha256=base.digest(path))
    output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
