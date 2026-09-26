"""Multi-rank, no-weight reproduction of the cold mmap registration boundary."""

import argparse
import json
import os


def main():
    import torch
    import torch.distributed as dist

    from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("whole", "chunked"), required=True)
    parser.add_argument("--engine-id", required=True)
    parser.add_argument("--gpu-gib", type=int, choices=range(23), default=0)
    parser.add_argument("--host-gib", type=int, choices=range(33), default=0)
    parser.add_argument("--sync-population", action="store_true")
    parser.add_argument("--uva-host", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size in (2, 4) and 0 <= rank < world_size
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo")
    region = None
    report = dict(
        rank=rank,
        world_size=world_size,
        mode=args.mode,
        registrations=[],
        passed=False,
        gpu_gib=args.gpu_gib,
        host_gib=args.host_gib,
        sync_population=args.sync_population,
        uva_host=args.uva_host,
        phase="pressure",
    )
    gpu_pressure = None
    host_pressure = None
    uva_views = []
    try:
        if args.gpu_gib:
            gpu_pressure = torch.zeros(
                args.gpu_gib * 1024**3, dtype=torch.uint8, device="cuda"
            )
        if args.host_gib:
            host_pressure = torch.zeros(
                args.host_gib * 1024**3, dtype=torch.uint8, pin_memory=True
            )
        if args.uva_host:
            from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

            if host_pressure is None:
                raise ValueError("UVA probe requires a pinned host allocation")
            for index in range(16):
                view = host_pressure[index::16]
                assert view.is_pinned()
                uva_views.append(get_accelerator_view_from_cpu_tensor(view))
        torch.cuda.synchronize()
        report["gpu_free_bytes_before_region"] = torch.cuda.mem_get_info()[0]
        report["phase"] = "region"
        region = SharedOffloadRegion(
            engine_id=args.engine_id,
            num_chunks=1339,
            rank=rank,
            kv_bytes_per_chunk=6414336,
            cpu_page_size=1603584,
            barrier=dist.barrier,
        )
        if args.sync_population:
            dist.barrier()
        report["phase"] = "registration"
        row = 6414336
        chunk_bytes = (
            region.total_size_bytes if args.mode == "whole" else (1024**3 // row) * row
        )
        runtime = torch.cuda.cudart()
        for offset in range(0, region.total_size_bytes, chunk_bytes):
            size = min(chunk_bytes, region.total_size_bytes - offset)
            address = region.base_tensor.data_ptr() + offset
            status = runtime.cudaHostRegister(address, size, 0)
            report["registrations"].append(
                dict(offset=offset, bytes=size, code=int(status.value))
            )
            if status.value != 0:
                raise RuntimeError(f"Host registration failed with code {status.value}")
            region.pinned_addresses.append(address)
            region.is_pinned = True
        # Exercise CUDA after registration to detect deferred runtime errors.
        report["phase"] = "cuda_check"
        value = torch.empty(1, device="cuda").fill_(7)
        torch.cuda.synchronize()
        assert value.item() == 7
        report["passed"] = True
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error))
    finally:
        # Keep every mapping alive until both registration attempts finish.
        dist.barrier()
        if region is not None:
            region.cleanup()
        uva_views.clear()
        del gpu_pressure, host_pressure
        print(json.dumps(report, sort_keys=True), flush=True)
        dist.destroy_process_group()
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
