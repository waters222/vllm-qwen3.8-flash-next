"""Build isolated stage2/stage4 SM86 Marlin probes; never replace a vLLM op.

Requires hash-pinned upstream sources and the current image/toolchain. CPU-only
compilation is not numerical validation or evidence of a performance benefit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

COMMIT = 'dc36fcce902a63eab06c1b93a5c4a5ee178a0c56'
PREFIX = 'libtorch_stable/moe/marlin_moe_wna16/'
PINS = {
    PREFIX + 'ops.cu': 'edbfdfdeed22baf84547311f3e1922bb914cf56cdddc29eba3ac25f9c6fe1bf0',
    PREFIX + 'kernel.h': 'f6991bdd8bf5d2c180ebcb8f20c23f58e8de2d1a60507475d41f61eed2ca1de6',
    PREFIX + 'marlin_template.h': '20972ba8eed8d6b3d3ff258dae7c495a5085197482fcd237bf72b3e5ff748c41',
    'libtorch_stable/quantization/marlin/marlin.cuh': '8a129f1f3086d83c7f33ee8fdb38a0d68c0b167c99c9cef60394c3219ede86b7',
    'libtorch_stable/quantization/marlin/marlin_dtypes.cuh': '15b90a65eadb200a3f7165e2e0d7d899f36d4993b20dbd89093ef2cea274da71',
    'libtorch_stable/quantization/marlin/dequant.h': '39c4640d2de39374ef5d96e7fa27701a675922113c3f5b05d45369321090e728',
    'libtorch_stable/quantization/marlin/marlin_mma.h': 'bf863d252bfc468eaff42b2c1bda583c5e6ab97ceacf0ac248c164564ddcce9b',
    'libtorch_stable/torch_utils.h': 'f619e1943039a67a3254f7e0679e1f3e0faa9d8fccae795f3b20df9e91a8a429',
    'core/scalar_type.hpp': '8d3671759c43a0219f51e3a1f32c94ef5d70ecc99e264100f4015a781d94bb99',
}
NATIVE_EXTENSION = 'df2ea61124a7bcbe8f27708c45705a0f2624277df14f4ca914242b99ac299e6c'
TILES = ((128, 128, 256), (64, 128, 128), (128, 64, 128))
# Captured from the exact installed extension; GPU validation must recheck it.
NATIVE_SCHEMA = ('_moe_C::moe_wna16_marlin_gemm(Tensor($0! -> ) a, Tensor? c_or_none, '
    'Tensor($1! -> ) b_q_weight, Tensor? b_bias_or_none, Tensor($2! -> ) b_scales, '
    'Tensor? a_scales, Tensor? global_scale, Tensor? b_zeros_or_none, Tensor($3! -> ) workspace, '
    'Tensor sorted_token_ids, Tensor($4! -> ) expert_ids, Tensor($5! -> ) num_tokens_past_padded, '
    'Tensor($6! -> ) topk_weights, int moe_block_size, int top_k, bool mul_topk_weights, '
    'int b_type_id, int size_m, int size_n, int size_k, bool use_atomic_add, bool use_fp32_reduce, '
    'bool is_zp_float, int thread_k, int thread_n, int blocks_per_sm) -> Tensor')
PRIVATE_SCHEMA = ('gemm(Tensor! a, Tensor? c_or_none, Tensor! b_q_weight, Tensor? b_bias_or_none, '
    'Tensor! b_scales, Tensor? a_scales, Tensor? global_scale, Tensor? b_zeros_or_none, '
    'Tensor! workspace, Tensor sorted_token_ids, Tensor! expert_ids, Tensor! num_tokens_past_padded, '
    'Tensor! topk_weights, int moe_block_size, int top_k, bool mul_topk_weights, int b_type_id, '
    'int size_m, int size_n, int size_k, bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float, '
    'int thread_k, int thread_n, int blocks_per_sm) -> Tensor')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(source, before, after):
    require(source.count(before) == 1, 'Source boundary differs: ' + before[:80])
    return source.replace(before, after)


def generate(source, stages):
    require(stages in (2, 4) and type(stages) is int, 'Only even stage2/stage4 probes are supported')
    require(hashlib.sha256(source.encode()).hexdigest() == PINS[PREFIX + 'ops.cu'], 'Dispatcher pin differs')
    namespace = f'flash_marlin_stage{stages}'
    source = replace_once(source, '#include "kernel.h"', '#include "kernel.h"\n#include "marlin_template.h"')
    selector = []
    for k, n, threads in TILES:
        selector.append(f'''  if (a_type == vllm::kBFloat16 && b_type == vllm::kU4 &&
      c_type == vllm::kBFloat16 && s_type == vllm::kBFloat16 &&
      thread_m_blocks == 1 && thread_n_blocks == {n//16} && thread_k_blocks == {k//16} &&
      m_block_size_8 && has_zp && group_blocks == 2 && threads == {threads} &&
      !is_zp_float && stages == {stages})
    kernel = Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(),
                    vllm::kBFloat16.id(), {threads}, 1, {n//16}, {k//16}, true, {stages}, 2, false>;''')
    source = replace_once(source, '#include "kernel_selector.h"', '\n'.join(selector))
    source = replace_once(source, '  int stages = 4;', f'  int stages = {stages};')
    guard = '''  STD_TORCH_CHECK(major_capability == 8 && minor_capability == 6,
                  "Pipeline probe requires SM86");
  STD_TORCH_CHECK(a_type == vllm::kBFloat16 && b_type == vllm::kU4 &&
                  c_type == vllm::kBFloat16 && s_type == vllm::kBFloat16 &&
                  group_size == 32 && has_zp && !is_zp_float && !has_bias &&
                  !use_atomic_add && use_fp32_reduce && num_experts == 512 &&
                  moe_block_size == 8, "Pipeline probe precision/layout differs");
  STD_TORCH_CHECK(
      (prob_n == 640 && prob_k == 2560 && prob_m >= 1 && prob_m <= 4 &&
       top_k == 10 && !mul_topk_weights) ||
      (prob_n == 2560 && prob_k == 320 && prob_m >= 10 && prob_m <= 40 &&
       prob_m % 10 == 0 && top_k == 1 && mul_topk_weights),
      "Pipeline probe requires real TP2 M1--4 projection shapes");
  // Set thread config'''
    source = replace_once(source, '  // Set thread config', guard)
    # Unique global entry symbol and torch namespace prevent native interposition.
    source = source.replace('moe_wna16_marlin_gemm', f'{namespace}_gemm')
    source = replace_once(source, 'STABLE_TORCH_LIBRARY_IMPL(_moe_C, CUDA, m)',
                          f'STABLE_TORCH_LIBRARY_IMPL({namespace}, CUDA, m)')
    source = replace_once(source, f'm.impl("{namespace}_gemm"', 'm.impl("gemm"')
    return f'#define MARLIN_NAMESPACE_NAME {namespace}\n' + source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='Pinned csrc directory, read-only')
    p.add_argument('--output', type=Path, required=True, help='New empty writable build directory')
    a = p.parse_args()
    require(a.source.resolve() != a.output.resolve() and not a.output.resolve().is_relative_to(a.source.resolve()),
            'Build output must be separate from source')
    require(a.output.is_dir() and not list(a.output.iterdir()), 'Require empty build output')
    for relative, expected in PINS.items():
        require(digest(a.source / relative) == expected, 'Source pin differs: ' + relative)
    require(os.environ.get('TORCH_CUDA_ARCH_LIST') == '8.6' and os.environ.get('MAX_JOBS') == '2',
            'Require SM86-only compilation and two compiler jobs')
    import torch
    import vllm
    from torch.utils.cpp_extension import load
    require(not torch.cuda.is_available(), 'Compile probe without GPU access')
    extension = Path(vllm.__file__).parent / '_moe_C_stable_libtorch.abi3.so'
    require(digest(extension) == NATIVE_EXTENSION, 'Installed native extension differs')
    schema = NATIVE_SCHEMA
    report = dict(status='building', source_commit=COMMIT, source_pins=PINS,
                  builder_sha256=digest(__file__), native_extension_sha256=NATIVE_EXTENSION,
                  native_schema=schema, native_schema_revalidated=False,
                  torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  nvcc=subprocess.check_output(['/usr/local/cuda/bin/nvcc', '--version'], text=True),
                  compile_arch='8.6', compiler_jobs=2, builds=[], gpu_validation=False,
                  serving_changes=False, promotion_eligible=False)
    def save():
        (a.output / 'build-report.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    libraries = []
    for stages in (4, 2):
        namespace = f'flash_marlin_stage{stages}'
        source = generate((a.source / (PREFIX + 'ops.cu')).read_text(), stages)
        unit = a.output / f'{namespace}.cu'
        unit.write_text(source)
        build = a.output / namespace
        build.mkdir()
        library = torch.library.Library(namespace, 'DEF')
        library.define(PRIVATE_SCHEMA)
        libraries.append(library)
        path = load(name=namespace, sources=[str(unit)],
                    extra_include_paths=[str(a.source), str(a.source / PREFIX)],
                    extra_cflags=['-O3', '-DUSE_CUDA'], extra_cuda_cflags=['-O3', '-DUSE_CUDA', '--expt-relaxed-constexpr',
                        '--expt-extended-lambda', '-Xptxas=-v'],
                    build_directory=str(build), is_python_module=False, verbose=True)
        require(str(getattr(torch.ops, namespace).gemm.default._schema).startswith(namespace + '::gemm('),
                'Private operator registration failed')
        report['builds'].append(dict(stages=stages, namespace=namespace, library=str(path),
                                    library_sha256=digest(path), generated_source_sha256=digest(unit)))
        save()
    require(digest(extension) == NATIVE_EXTENSION, 'Native extension changed')
    report['status'] = 'compiled_and_loaded_no_gpu'
    save()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
