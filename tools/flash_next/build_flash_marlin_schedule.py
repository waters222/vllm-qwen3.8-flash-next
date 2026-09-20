"""Compile an isolated SM86 stage4 expert-down whole-tile scheduling probe.

CPU-only build; no native replacement, weight changes or numerical acceptance.
Reuse the pinned upstream arithmetic and change only tile work ownership.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

BASE_SHA = 'b5beb236739b0ed1a5e0a6db0ed453d57a0a73d736f2500aeaf59db585f4f1fc'
NAMESPACE = 'flash_marlin_dp4'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_base():
    path = Path(__file__).with_name('build_flash_marlin_pipeline.py')
    require(digest(path) == BASE_SHA, 'Base builder pin differs')
    spec = importlib.util.spec_from_file_location('schedule_base_' + BASE_SHA, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def generate_kernel(source):
    base = load_base()
    require(hashlib.sha256(source.encode()).hexdigest() == base.PINS[base.PREFIX + 'marlin_template.h'],
            'Kernel source pin differs')
    before = '  int slice_row = 0;'
    after = '''  // Whole output tiles, round-robin over the unchanged launch grid.
  // No output tile is split across CTAs; existing block-local FP32 math stays.
  part2_mn_tiles = 0;
  part1_mn_iters = blockIdx.x < global_mn_tiles
      ? div_ceil(global_mn_tiles - static_cast<int>(blockIdx.x),
                 static_cast<int>(gridDim.x)) : 0;
  iters = 0;

  int slice_row = 0;'''
    source = base.replace_once(source, before, after)
    start = source.index('  auto init_slice = [&]() {')
    end = source.index('\n  init_slice();', start)
    original = source[start:end]
    require(original.count('init_part1_slice();') == original.count('init_part2_slice();') == 1,
            'Slice initializer differs')
    replacement = '''  auto init_slice = [&]() {
    if (part1_mn_iters == 0) {
      slice_iters = 0;
      return;
    }
    init_part1_slice();
  };
'''
    return source[:start] + replacement + source[end:]


def generate_dispatch(source):
    base = load_base()
    generated = base.generate(source, 4).replace('flash_marlin_stage4', NAMESPACE)
    generated = base.replace_once(generated, '#include "marlin_template.h"', '#include "marlin_dp_template.h"')
    return base.replace_once(generated, '  // Set thread config', '''  STD_TORCH_CHECK(prob_n == 2560 && prob_k == 320 && top_k == 1 &&
                  mul_topk_weights, "Whole-tile probe is expert-down only");
  // Set thread config''')


def schedule(global_tiles, k_tiles, blocks, *, whole_tile=False):
    """Integer work partition model, not a timing or occupancy model."""
    require(all(type(v) is int and v > 0 for v in (global_tiles, k_tiles, blocks)), 'Require positive integer geometry')
    assignments = [[] for _ in range(blocks)]
    if whole_tile:
        for tile in range(global_tiles):
            assignments[tile % blocks].append((tile, 0, k_tiles))
    else:
        remainder = global_tiles
        dp_cycles = 0
        if global_tiles > blocks:
            remainder = global_tiles % blocks
            if remainder * 3 <= blocks:
                remainder += blocks
            dp_cycles = (global_tiles - remainder) // blocks
        for block in range(blocks):
            for cycle in range(dp_cycles):
                assignments[block].append((block + cycle * blocks, 0, k_tiles))
        iters = (k_tiles * remainder + blocks - 1) // blocks
        offset = global_tiles - remainder
        for block in range(blocks):
            begin, end = block * iters, min((block + 1) * iters, k_tiles * remainder)
            while begin < end:
                tile, k = divmod(begin, k_tiles)
                length = min(k_tiles - k, end - begin)
                assignments[block].append((offset + tile, k, length))
                begin += length
    coverage = [[] for _ in range(global_tiles)]
    for block in assignments:
        for tile, start, length in block:
            coverage[tile].append((start, length))
    for pieces in coverage:
        cursor = 0
        for start, length in sorted(pieces):
            require(start == cursor, 'Overlapping or missing K work')
            cursor += length
        require(cursor == k_tiles, 'Incomplete output tile')
    work = [sum(length for _, _, length in block) for block in assignments]
    return dict(global_tiles=global_tiles, k_tiles=k_tiles, grid_blocks=blocks, whole_tile=whole_tile,
        active_blocks=sum(bool(a) for a in assignments), total_k_tile_work=sum(work),
        max_k_tiles_per_block=max(work), min_k_tiles_per_active_block=min(v for v in work if v),
        split_output_tiles=sum(len(p) > 1 for p in coverage),
        extra_partial_results=sum(len(p)-1 for p in coverage),
        output_tiles_per_block=[len(a) for a in assignments], assignments=assignments)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = load_base()
    require(args.source.resolve() != args.output.resolve() and not args.output.resolve().is_relative_to(args.source.resolve()),
            'Separate read-only sources from build output')
    require(args.output.is_dir() and not list(args.output.iterdir()), 'Require empty build output')
    require(all(digest(args.source / p) == h for p, h in base.PINS.items()), 'Upstream source differs')
    require(os.environ.get('TORCH_CUDA_ARCH_LIST') == '8.6' and os.environ.get('MAX_JOBS') == '2', 'Build limits differ')
    import torch
    import vllm
    from torch.utils.cpp_extension import load
    require(not torch.cuda.is_available(), 'Compile without GPU access')
    extension = Path(vllm.__file__).parent / '_moe_C_stable_libtorch.abi3.so'
    require(digest(extension) == base.NATIVE_EXTENSION, 'Native extension differs')
    unit = args.output / f'{NAMESPACE}.cu'
    header = args.output / 'marlin_dp_template.h'
    unit.write_text(generate_dispatch((args.source / (base.PREFIX + 'ops.cu')).read_text()))
    header.write_text(generate_kernel((args.source / (base.PREFIX + 'marlin_template.h')).read_text()))
    report = dict(status='building', namespace=NAMESPACE, builder_sha256=digest(__file__), base_builder_sha256=BASE_SHA,
        source_pins=base.PINS, native_extension_sha256=base.NATIVE_EXTENSION, native_schema=base.NATIVE_SCHEMA,
        private_schema=base.PRIVATE_SCHEMA, native_schema_revalidated=False, torch_version=torch.__version__,
        cuda_version=torch.version.cuda, nvcc=subprocess.check_output(['/usr/local/cuda/bin/nvcc', '--version'], text=True),
        generated_dispatch_sha256=digest(unit), generated_kernel_sha256=digest(header), stages=4,
        compile_arch='8.6', compiler_jobs=2, gpu_validation=False, serving_changes=False, promotion_eligible=False)
    def save():
        (args.output / 'build-report.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    owner = torch.library.Library(NAMESPACE, 'DEF')
    owner.define(base.PRIVATE_SCHEMA)
    build = args.output / NAMESPACE
    build.mkdir()
    path = load(name=NAMESPACE, sources=[str(unit)],
        extra_include_paths=[str(args.source), str(args.source / base.PREFIX)],
        extra_cflags=['-O3', '-DUSE_CUDA'], extra_cuda_cflags=['-O3', '-DUSE_CUDA', '--expt-relaxed-constexpr',
            '--expt-extended-lambda', '-Xptxas=-v'], build_directory=str(build), is_python_module=False, verbose=True)
    require(str(getattr(torch.ops, NAMESPACE).gemm.default._schema).startswith(NAMESPACE + '::gemm('), 'Private registration failed')
    require(digest(extension) == base.NATIVE_EXTENSION, 'Native extension changed')
    report.update(status='compiled_and_loaded_no_gpu', library=str(path), library_sha256=digest(path))
    save()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
