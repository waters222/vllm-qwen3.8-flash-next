"""Local controller for the authorized, isolated water-server GPU swap screen.

Run through the checkout's .venv. Inspect/prepare by default; --execute stops
only the exact idle development source and starts one trial. No automatic
restart, cleanup, serving changes, downloads, or public endpoints.
"""

import argparse
import hashlib
import json
import shlex
import shutil
import socket
import subprocess
from pathlib import Path

HOST = "water-server.lan"
LOCAL_HOST = False
SOURCE = "d9f6e139ba74f08277764411ae49170648307842267f8a7854ee47e3b7676aef"
PEER = "7d408c2bc15d08fc7ba4d060da4c136b03a3eaa7aee5d895d27394e9f4287eeb"
PEER_START = "2026-09-20T19:45:53.401210251Z"
IMAGE = "sha256:0f75571247e58cfe09d11fe9c6d5dce6b7e2c7da9dfeb6c3e1a2607a7c7e5599"
SITE = "/usr/local/lib/python3.12/dist-packages/"
SOURCES = (
    "vllm/v1/core/flash_session_allocator.py",
    "vllm/v1/core/flash_session_transactions.py",
    "vllm/v1/core/sched/scheduler.py",
    "vllm/v1/engine/core.py",
    "vllm/v1/request.py",
    "vllm/v1/worker/gpu/flash_session_swap.py",
    "vllm/v1/worker/gpu/flash_session_worker.py",
    "vllm/v1/worker/gpu/model_runner.py",
    "vllm/v1/worker/gpu_worker.py",
)
KERNEL_FLAGS = (
    "VLLM_FLASH_GDN_INPUT_SM86_GEMV",
    "VLLM_FLASH_GDN_DECODE_METADATA",
    "VLLM_FLASH_GDN_INPUT_MULTIROW_SM86",
    "VLLM_FLASH_GDN_BA_SM86",
    "VLLM_FLASH_GDN_INPUT_M3",
    "VLLM_FLASH_GDN_INPUT_TENSOR_SM86",
    "VLLM_FLASH_ATTN_OUT_SM86_GEMV",
    "VLLM_FLASH_ATTN_OUT_MULTIROW_SM86",
    "VLLM_FLASH_SHARED_EPILOGUE_SM86",
    "VLLM_FLASH_MARLIN_DOWN_SCHEDULE_SM86",
    "VLLM_FLASH_MOE_LAYOUT_SM86",
    "VLLM_FLASH_MOE_SUM_C1",
    "VLLM_FLASH_HC_PAIR_SM86",
    "VLLM_FLASH_HC_PAIR_M3",
    "VLLM_FLASH_HC_DOWN_SILU_SM86",
    "VLLM_FLASH_HC_UP_GATE_SM86",
    "VLLM_FLASH_HC_INJECT_SM86",
    "VLLM_FLASH_HC_INJECT_MTP",
    "VLLM_FLASH_HC_SM86_GEMV",
)
EXPERIMENTAL_KERNEL_FLAGS = ("VLLM_FLASH_GDN_TP4_PROJECTION",)


def kernel_environment(env, native, optimized_canonical=False, expert_order_control=None):
    """Disable only opt-in kernels; retain Q8/PP/scheduler/RAM-KV settings."""
    if expert_order_control not in (None, "native", "stable"):
        raise ValueError("Unsupported expert-order control")
    if expert_order_control and (native or optimized_canonical):
        raise ValueError("Expert-order control cannot combine with other kernel controls")
    if optimized_canonical or expert_order_control:
        if native:
            raise ValueError("Mixed diagnostic cannot also disable all optimizations")
        # The ordering hook wraps native expert grouping, not optimized layout.
        key = "VLLM_FLASH_MOE_LAYOUT_SM86"
        return [row for row in env if row.split("=", 1)[0] != key] + [f"{key}=0"]
    if not native:
        return list(env)
    flags = KERNEL_FLAGS + EXPERIMENTAL_KERNEL_FLAGS
    return [row for row in env if row.split("=", 1)[0] not in flags] + [
        f"{name}=0" for name in flags
    ]


def remote(*args):
    return subprocess.check_output(
        list(args)
        if LOCAL_HOST
        else ["ssh", "-o", "BatchMode=yes", HOST, shlex.join(args)],
        text=True,
    ).strip()


def tp4_environment(env):
    """Use the validated unpadded TP4/Q8 gates without inherited PP placement."""
    overrides = {
        "VLLM_FLASH_QSA_TP4": "1",
        "VLLM_FLASH_TP4_MARLIN_K32": "1",
        "VLLM_FLASH_TP4_MTP_Q8": "1",
        "VLLM_FLASH_MTP_INT8_EXPERTS": "1",
        "VLLM_FLASH_MTP_INT8_LOW_PEAK": "1",
        "VLLM_FLASH_TP4_CHUNKED_REPACK": "0",
    }
    return [
        row
        for row in env
        if row.split("=", 1)[0] not in overrides
        and not row.startswith(("VLLM_PP_", "VLLM_FLASH_PP_"))
    ] + [f"{name}={value}" for name, value in overrides.items()]


def prefill_staging_environment(env, enabled):
    """Do not inherit an experimental staging choice into a control run."""
    name = "VLLM_QSA_STAGE_ALL_PREFILL"
    return [row for row in env if row.split("=", 1)[0] != name] + [
        f"{name}={int(enabled)}"
    ]


def inspect(cid):
    fields = {
        name: json.loads(remote("docker", "inspect", cid, "--format", expr))
        for name, expr in (
            ("id", "{{json .Id}}"),
            ("image", "{{json .Image}}"),
            ("state", "{{json .State}}"),
            ("host", "{{json .HostConfig}}"),
            ("mounts", "{{json .Mounts}}"),
        )
    }
    return fields


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def require_idle(cid):
    remote("docker", "exec", cid, "curl", "-fsS", "http://127.0.0.1:8000/health")
    metrics = remote(
        "docker", "exec", cid, "curl", "-fsS", "http://127.0.0.1:8000/metrics"
    )
    rows = [
        row
        for row in metrics.splitlines()
        if row.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{"))
    ]
    require(
        len(rows) == 2 and all(float(r.rsplit(" ", 1)[1]) == 0 for r in rows),
        "Expected a healthy, idle development model",
    )


def require_free_quartet():
    rows = remote(
        "nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"
    ).splitlines()
    usage = {int(i): int(m) for i, m in (row.split(",") for row in rows)}
    require(
        all(usage.get(i, 99999) < 100 for i in range(4)),
        "Development quartet is not free",
    )


def main():
    global LOCAL_HOST
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument(
        "--local-host",
        action="store_true",
        help="Run directly on water-server; retain all identity/GPU guards",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--source-stopped", action="store_true")
    parser.add_argument("--no-mtp", action="store_true")
    parser.add_argument("--native-prefix", action="store_true")
    parser.add_argument("--tp4", action="store_true")
    parser.add_argument("--serving-decode", action="store_true")
    parser.add_argument("--prefix-concurrency", type=int, choices=(1, 2), default=1)
    parser.add_argument("--prefix-prefill-budget", type=int, choices=(1024, 2048))
    parser.add_argument("--prefix-gpu-cache-mib", type=int, choices=(512, 768, 896, 1024))
    parser.add_argument("--balanced-prefix-prefill", action="store_true")
    parser.add_argument("--paired-cold-loads", action="store_true")
    parser.add_argument("--continuation-load-gate", action="store_true")
    parser.add_argument("--idle-expiry", action="store_true")
    parser.add_argument("--pending-expiry", action="store_true")
    parser.add_argument("--continuations", action="store_true")
    parser.add_argument(
        "--continuation-concurrency", type=int, choices=(1, 2, 4), default=1
    )
    parser.add_argument("--active-cancel", action="store_true")
    parser.add_argument(
        "--cancellation-concurrency", type=int, choices=(2, 4), default=2
    )
    parser.add_argument("--audit-active-cancel", action="store_true")
    parser.add_argument("--expire-active-cache", action="store_true")
    parser.add_argument("--performance-repeats", type=int, choices=(0, 3, 5), default=0)
    parser.add_argument("--performance-iteration-details", action="store_true")
    parser.add_argument("--profile-prefix-prefill", action="store_true")
    parser.add_argument("--prefill-staging-ab", action="store_true")
    parser.add_argument("--prefill-staging-concurrency", type=int, choices=(2, 4), default=2)
    parser.add_argument("--native-load-failure", choices=("inject", "recovery"))
    parser.add_argument("--stage-all-prefill", action="store_true")
    parser.add_argument(
        "--continuation-context", type=int, choices=(8192, 32768), default=8192
    )
    parser.add_argument("--untraced-balanced-prefix", action="store_true")
    parser.add_argument("--divergent-prefixes", action="store_true")
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--trace-prefill", action="store_true")
    parser.add_argument("--trace-prefix-prefill", action="store_true")
    parser.add_argument("--trace-prompt-chunks", action="store_true")
    parser.add_argument("--trace-prefix-shapes", action="store_true")
    parser.add_argument("--trace-prefix-pages", action="store_true")
    parser.add_argument("--private-cold-buffers", action="store_true")
    parser.add_argument("--audit-native-copies", action="store_true")
    parser.add_argument("--native-kernels", action="store_true")
    parser.add_argument("--optimized-canonical", action="store_true")
    parser.add_argument("--stable-expert-layout", action="store_true")
    parser.add_argument("--mlp-numerical-control", action="store_true")
    parser.add_argument("--sustained-decode", action="store_true")
    parser.add_argument("--sustained-real-idle-expiry", action="store_true")
    parser.add_argument("--sustained-uniform-outputs", action="store_true")
    parser.add_argument("--sustained-observe-lookups", action="store_true")
    parser.add_argument("--sustained-idle-ttl", type=int, choices=(300, 3600), default=3600)
    parser.add_argument("--sustained-output-tokens", type=int, choices=(1024, 4096),
                        default=1024)
    parser.add_argument("--expert-order-control", choices=("native", "stable"))
    parser.add_argument("--stable-qsa-selection", action="store_true")
    parser.add_argument("--stable-qsa-ties", action="store_true")
    parser.add_argument("--cuda-launch-blocking", action="store_true")
    parser.add_argument("--trace-host-registration", action="store_true")
    parser.add_argument("--storage-backend", choices=("copy", "native"), default="copy")
    args = parser.parse_args()
    if args.local_host:
        require(
            socket.gethostname().split(".")[0] == "water-server",
            "Local transport is restricted to water-server",
        )
    LOCAL_HOST = args.local_host
    from prefix_cache_benchmark import validate_performance
    from prefix_cache_screen import (
        validate_balanced_observation,
        validate_cancellation_observation,
        validate_continuation_observation,
        validate_copy_audit,
        validate_gpu_cache_budget,
        validate_serving_decode,
    )
    from session_transfer_audit import validate_native_load_failure
    from session_sustained_decode import validate as validate_sustained

    validate_sustained(args)
    validate_balanced_observation(args)
    validate_continuation_observation(args)
    validate_cancellation_observation(args)
    validate_serving_decode(args)
    validate_copy_audit(args)
    validate_gpu_cache_budget(args)
    validate_performance(args)
    validate_native_load_failure(args)
    require(
        not args.continuation_load_gate or (
            args.paired_cold_loads and args.continuations
            and args.continuation_concurrency == 4
        ),
        "Continuation load gate requires paired C4 continuations",
    )
    require(
        not args.stage_all_prefill or args.native_prefix,
        "Multi-request staging requires the native-prefix diagnostic",
    )
    require(
        not (args.idle_expiry or args.pending_expiry) or args.paired_cold_loads,
        "Expiry diagnostics require the isolated paired-load scheduler",
    )
    require(
        not args.paired_cold_loads or args.balanced_prefix_prefill,
        "Paired cold loads require balanced prefix prefill",
    )
    require(
        not args.balanced_prefix_prefill
        or (
            args.native_prefix
            and args.tp4
            and args.prefix_concurrency == 2
            and args.prefix_prefill_budget == 2048
            and (
                args.trace_prefix_shapes
                or args.untraced_balanced_prefix
                or args.audit_native_copies
            )
        ),
        "Balanced prefix diagnostic requires TP4/C2, budget2048 "
        "and an observation mode",
    )
    require(
        not args.trace_prefix_pages or args.trace_prefix_shapes,
        "Prefix page tracing requires the phase/shape trace",
    )
    require(
        not (args.tp4 or args.prefix_concurrency > 1) or args.native_prefix,
        "TP4/concurrency screening requires ordinary native-prefix mode",
    )
    require(
        not args.divergent_prefixes or args.prefix_concurrency > 1,
        "Divergent-prefix screening requires concurrent requests",
    )
    require(
        args.prefix_prefill_budget is None or args.native_prefix,
        "Prefill budget diagnostic requires native-prefix mode",
    )
    require(
        not args.private_cold_buffers or args.native_prefix,
        "Private cold buffers require native-prefix screening",
    )
    require(
        not (
            args.trace_prefix_prefill
            or args.trace_prompt_chunks
            or args.trace_prefix_shapes
        )
        or args.native_prefix,
        "Prefix prefill trace requires ordinary native-prefix screening",
    )
    require(
        not (args.trace_prefix_prefill and args.trace_prompt_chunks),
        "Select only one prefix tracing mode",
    )
    require(
        not args.trace_host_registration or args.stable_expert_layout,
        "Host registration trace requires the stable-layout worker extension",
    )
    require(
        not args.optimized_canonical
        or (
            args.native_prefix
            and not args.native_kernels
            and args.stable_expert_layout
            and args.stable_qsa_selection
            and args.stable_qsa_ties
        ),
        "Mixed diagnostic requires native-prefix, all canonical controls, "
        "and no native-kernels override",
    )
    require(
        not args.stable_expert_layout
        or (args.native_prefix and (args.native_kernels or args.optimized_canonical
                                  or args.expert_order_control == "stable")),
        "Stable expert diagnostic requires native-prefix and explicit kernel control",
    )
    require(
        not args.stable_qsa_selection
        or (args.native_prefix and (args.native_kernels or args.optimized_canonical)),
        "Stable QSA diagnostic requires native-prefix and explicit kernel control",
    )
    require(
        not args.stable_qsa_ties or args.stable_qsa_selection,
        "Stable QSA ties require the selection diagnostic",
    )
    require(
        not args.native_prefix or not (args.streaming or args.trace_prefill),
        "Ordinary prefix screening cannot use the streaming prototype",
    )
    require(
        not args.trace_prefill or args.streaming, "Prefill trace requires streaming"
    )
    require(not args.evidence.exists(), "Evidence file already exists")
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "docs/flash_next/source_manifest.json").read_text())
    sources = tuple(manifest["engine_sources"]) if args.native_prefix else SOURCES
    hashes = {}
    for name in sources:
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        expected = manifest["post_snapshot_sources"].get(
            name, manifest["engine_sources"][name]
        )
        require(digest == expected, "Source manifest mismatch: " + name)
        hashes[name] = digest
    source, peer = inspect(SOURCE), inspect(PEER)
    require(
        source["id"] == SOURCE
        and source["image"] == IMAGE
        and (
            source["state"]["Running"]
            or (
                args.source_stopped
                and source["state"]["Status"] == "exited"
                and source["state"]["ExitCode"] == 0
                and not source["state"]["OOMKilled"]
            )
        ),
        "Development identity changed",
    )
    require(
        peer["state"]["Running"] and peer["state"]["StartedAt"] == PEER_START,
        "Serving peer changed",
    )
    host = source["host"]
    require(
        host["DeviceRequests"][0]["DeviceIDs"] == ["0", "1", "2", "3"]
        and host["NetworkMode"] == "none"
        and not host["PortBindings"]
        and host["Memory"] == 240 * 1024**3
        and host["MemorySwap"] == host["Memory"],
        "Source boundary changed",
    )
    if source["state"]["Running"]:
        require_idle(SOURCE)
    else:
        require_free_quartet()
    model = next(m for m in source["mounts"] if m["Destination"] == "/model")
    require(
        not model["RW"]
        and model["Source"] == "/media/storage/models/Qwen/Qwen3.8-Flash-Next-AWQ-INT4",
        "Source model mount changed",
    )
    if not args.execute:
        print("Preflight passed; no remote state changed")
        return
    trial_root = remote("mktemp", "-d", "/tmp/flash-session-gpu-20260921-XXXXXX")
    name = "flash-session-gpu-" + trial_root.rsplit("-", 1)[1].lower()
    # Explicit paths only; no source checkout or credentials enter the trial.
    uploads = list(sources) + [
        "tools/flash_next/qualify_session_swap.py",
        "tools/flash_next/session_swap_engine_args.json",
        "tools/flash_next/session_stream_screen.py",
        "tools/flash_next/session_prefill_trace.py",
        "tools/flash_next/session_numerical_control.py",
        "tools/flash_next/session_sustained_decode.py",
        "tools/flash_next/session_transfer_audit.py",
        "tools/flash_next/prefix_cache_screen.py",
        "tools/flash_next/prefix_cancellation.py",
        "tools/flash_next/prefix_cache_benchmark.py",
        "tools/flash_next/prefix_load_barrier.py",
    ]
    for path in uploads:
        target = trial_root + "/" + path
        remote("mkdir", "-p", str(Path(target).parent))
        if LOCAL_HOST:
            shutil.copyfile(root / path, target)
        else:
            subprocess.run(["scp", str(root / path), HOST + ":" + target], check=True)
        got = remote("sha256sum", target).split()[0]
        require(
            got == hashlib.sha256((root / path).read_bytes()).hexdigest(),
            "Remote source transfer hash mismatch",
        )
        hashes[path] = got
    remote("mkdir", trial_root + "/results")
    # Whitelist names remotely so secret-bearing environment entries never leave
    # the host. The source has no model API credential in this allowlist.
    names = set(KERNEL_FLAGS + EXPERIMENTAL_KERNEL_FLAGS) | {
        "HF_HOME", "HF_HUB_OFFLINE", "GLOO_SOCKET_IFNAME", "VLLM_HOST_IP",
        "PYTORCH_CUDA_ALLOC_CONF", "TRITON_CACHE_DIR", "XDG_CACHE_HOME",
        "VLLM_USE_BREAKABLE_CUDAGRAPH", "VLLM_QSA_KV_OFFLOAD",
        "VLLM_QSA_KV_OFFLOAD_MAX_GIB", "VLLM_QSA_KVO_ARENA",
        "VLLM_FLASH_NO_EP_AB", "VLLM_PP_LAYER_PARTITION",
        "VLLM_FLASH_PP_VOCAB_OWNERS_ONLY", "VLLM_FLASH_PP_BALANCE",
        "VLLM_SKIP_P2P_CHECK", "VLLM_CACHE_ROOT",
        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS",
        "VLLM_FLASH_MTP_INT8_EXPERTS", "VLLM_FLASH_MTP_INT8_LOW_PEAK",
    }
    tests = " ".join('(eq (index (split . "=") 0) ' + json.dumps(n) + ")"
                     for n in sorted(names))
    template = '{{range .Config.Env}}{{if or ' + tests + '}}{{println .}}{{end}}{{end}}'
    env = remote("docker", "inspect", "--format", template, SOURCE).splitlines()
    # The cache gate excludes the independently experimental projection path.
    env = [row for row in env if not row.startswith("VLLM_FLASH_GDN_TP4_PROJECTION=")]
    env.append("VLLM_FLASH_GDN_TP4_PROJECTION=0")
    require(not args.expert_order_control or (
        args.native_prefix and not args.native_kernels and not args.optimized_canonical
        and not args.stable_qsa_selection and not args.stable_qsa_ties
        and args.stable_expert_layout == (args.expert_order_control == "stable")
    ), "Expert-order control requires matching stable flag and no other canonical controls")
    require(not args.mlp_numerical_control or (
        args.tp4 and args.native_prefix and args.stable_expert_layout
        and args.expert_order_control == "stable" and not args.serving_decode
    ), "MLP numerical control requires eager TP4 native-prefix stable-order diagnostic")
    env = kernel_environment(env, args.native_kernels, args.optimized_canonical,
                             args.expert_order_control)
    env = prefill_staging_environment(env, args.stage_all_prefill)
    env = [row for row in env if not row.startswith("FLASH_PREFILL_STAGING_AB=")]
    env.append(f"FLASH_PREFILL_STAGING_AB={int(args.prefill_staging_ab)}")
    env = [row for row in env if not row.startswith("FLASH_NATIVE_LOAD_FAILURE=")]
    env.append(f"FLASH_NATIVE_LOAD_FAILURE={int(args.native_load_failure == 'inject')}")
    if args.serving_decode:
        effective = dict(row.split("=", 1) for row in env)
        require(
            all(effective.get(name) == "1" for name in KERNEL_FLAGS),
            "Serving decode requires every canonical optimization enabled",
        )
    if args.tp4:
        env = tp4_environment(env)
    if args.cuda_launch_blocking:
        env.append("CUDA_LAUNCH_BLOCKING=1")
    if args.trace_host_registration:
        env.append("FLASH_COLD_REGISTER_TRACE=1")
    env += [
        "VLLM_FLASH_SESSION_SWAP_BYTES=1073741824",
        "VLLM_FLASH_SESSION_SWAP_TTL_SECONDS=3600",
        f"VLLM_FLASH_SESSION_SWAP_BACKEND={args.storage_backend}",
    ]
    if args.native_prefix:
        env = [row for row in env if not row.startswith("VLLM_FLASH_SESSION_SWAP_")]
        env.append("VLLM_FLASH_SESSION_SWAP_BYTES=0")
    command = [
        "docker",
        "create",
        "--name",
        name,
        "--network",
        "none",
        "--gpus",
        '"device=0,1,2,3"',
        "--memory",
        "240g",
        "--memory-swap",
        "240g",
        "--shm-size",
        "16g",
        "--ulimit",
        "memlock=-1:-1",
        "--entrypoint",
        "/bin/sh",
    ]
    for value in env:
        command += ["--env", value]
    replaced = {SITE + path for path in sources}
    for mount in source["mounts"]:
        if mount["Destination"] in replaced:
            continue
        spec = mount["Source"] + ":" + mount["Destination"]
        command += ["--volume", spec + (":ro" if not mount["RW"] else "")]
    for path in sources:
        command += ["--volume", trial_root + "/" + path + ":" + SITE + path + ":ro"]
    for path in uploads[len(sources) :]:
        command += [
            "--volume",
            trial_root + "/" + path + ":/probe/" + Path(path).name + ":ro",
        ]
    command += [
        "--volume",
        trial_root + "/results:/results",
        IMAGE,
        "-c",
        "uv venv --offline --system-site-packages --python /usr/bin/python3 "
        "/tmp/session-probe/.venv && exec /tmp/session-probe/.venv/bin/python "
        "/probe/qualify_session_swap.py --engine-args "
        "/probe/session_swap_engine_args.json --output /results/screen.json",
    ]
    if args.no_mtp:
        command[-1] += " --no-mtp"
    if args.native_prefix:
        command[-1] += " --native-prefix --reuse-rounds 32"
        command[-1] += f" --prefix-concurrency {args.prefix_concurrency}"
    if args.divergent_prefixes:
        command[-1] += " --divergent-prefixes"
    if args.prefix_prefill_budget is not None:
        command[-1] += f" --prefix-prefill-budget {args.prefix_prefill_budget}"
    if args.prefix_gpu_cache_mib is not None:
        command[-1] += f" --prefix-gpu-cache-mib {args.prefix_gpu_cache_mib}"
    if args.balanced_prefix_prefill:
        command[-1] += " --balanced-prefix-prefill"
    if args.paired_cold_loads:
        command[-1] += " --paired-cold-loads"
    if args.continuation_load_gate:
        command[-1] += " --continuation-load-gate"
    if args.idle_expiry:
        command[-1] += " --idle-expiry"
    if args.pending_expiry:
        command[-1] += " --pending-expiry"
    if args.active_cancel:
        command[-1] += " --active-cancel"
        command[-1] += f" --cancellation-concurrency {args.cancellation_concurrency}"
        if args.cancellation_concurrency == 4:
            command[-1] += f" --continuation-context {args.continuation_context}"
            command[-1] += " --timeout 3600"
    if args.audit_active_cancel:
        command[-1] += " --audit-active-cancel"
    if args.expire_active_cache:
        command[-1] += " --expire-active-cache"
    if args.performance_repeats:
        command[-1] += (
            f" --performance-repeats {args.performance_repeats} --timeout 3600"
        )
    if args.performance_iteration_details:
        command[-1] += " --performance-iteration-details"
    if args.profile_prefix_prefill:
        command[-1] += " --profile-prefix-prefill"
    if args.prefill_staging_ab:
        command[-1] += " --prefill-staging-ab"
        command[-1] += f" --prefill-staging-concurrency {args.prefill_staging_concurrency}"
        if args.prefill_staging_concurrency == 4:
            command[-1] += " --timeout 3600"
    if args.native_load_failure:
        command[-1] += f" --native-load-failure {args.native_load_failure}"
    if args.continuations:
        command[-1] += " --continuations"
        command[-1] += f" --continuation-context {args.continuation_context}"
        command[-1] += f" --continuation-concurrency {args.continuation_concurrency}"
        if args.continuation_context == 32768:
            command[-1] += " --timeout 3600"
    if args.untraced_balanced_prefix:
        command[-1] += " --untraced-balanced-prefix"
    if args.tp4:
        command[-1] += " --tp4"
    if args.serving_decode:
        command[-1] += " --serving-decode"
    if args.stable_expert_layout:
        command[-1] += " --stable-expert-layout"
    if args.mlp_numerical_control:
        command[-1] += " --mlp-numerical-control"
    if args.sustained_decode:
        command[-1] += (" --sustained-decode --sustained-output-tokens "
                        f"{args.sustained_output_tokens} --continuation-context "
                        f"{args.continuation_context} --timeout 7200")
    if args.sustained_real_idle_expiry:
        command[-1] += " --sustained-real-idle-expiry"
        command[-1] += f" --sustained-idle-ttl {args.sustained_idle_ttl}"
    if args.sustained_uniform_outputs:
        command[-1] += " --sustained-uniform-outputs"
    if args.sustained_observe_lookups:
        command[-1] += " --sustained-observe-lookups"
    if args.stable_qsa_selection:
        command[-1] += " --stable-qsa-selection"
    if args.stable_qsa_ties:
        command[-1] += " --stable-qsa-ties"
    if args.trace_prefix_shapes:
        command[-1] += " --trace-prefix-shapes"
    if args.trace_prefix_pages:
        command[-1] += " --trace-prefix-pages"
    if args.diagnose:
        command[-1] += " --diagnose --reference-runs 4"
    if args.streaming:
        command[-1] += " --streaming"
    if args.trace_prefill:
        command[-1] += " --trace-prefill"
    if args.trace_prefix_prefill:
        command[-1] += " --trace-prefix-prefill"
    if args.trace_prompt_chunks:
        command[-1] += " --trace-prompt-chunks"
    if args.private_cold_buffers:
        command[-1] += " --private-cold-buffers"
    if args.audit_native_copies:
        command[-1] += " --audit-native-copies"
    cid = remote(*command)
    evidence = dict(
        root=trial_root,
        container_id=cid,
        name=name,
        source_id=SOURCE,
        peer_id=PEER,
        peer_started_at=PEER_START,
        hashes=hashes,
        image=IMAGE,
        no_mtp=args.no_mtp,
        native_prefix=args.native_prefix,
        tp4=args.tp4,
        serving_decode=args.serving_decode,
        prefix_concurrency=args.prefix_concurrency,
        divergent_prefixes=args.divergent_prefixes,
        prefix_prefill_budget=args.prefix_prefill_budget,
        prefix_gpu_cache_mib=args.prefix_gpu_cache_mib,
        balanced_prefix_prefill=args.balanced_prefix_prefill,
        paired_cold_loads=args.paired_cold_loads,
        continuation_load_gate=args.continuation_load_gate,
        idle_expiry=args.idle_expiry,
        pending_expiry=args.pending_expiry,
        continuations=args.continuations,
        continuation_concurrency=args.continuation_concurrency,
        active_cancel=args.active_cancel,
        cancellation_concurrency=args.cancellation_concurrency,
        audit_active_cancel=args.audit_active_cancel,
        expire_active_cache=args.expire_active_cache,
        performance_repeats=args.performance_repeats,
        performance_iteration_details=args.performance_iteration_details,
        profile_prefix_prefill=args.profile_prefix_prefill,
        prefill_staging_ab=args.prefill_staging_ab,
        prefill_staging_concurrency=args.prefill_staging_concurrency,
        native_load_failure=args.native_load_failure,
        qsa_stage_all_prefill=args.stage_all_prefill,
        continuation_context=args.continuation_context,
        untraced_balanced_prefix=args.untraced_balanced_prefix,
        diagnose=args.diagnose,
        streaming=args.streaming,
        trace_prefill=args.trace_prefill,
        trace_prefix_prefill=args.trace_prefix_prefill,
        trace_prompt_chunks=args.trace_prompt_chunks,
        trace_prefix_shapes=args.trace_prefix_shapes,
        trace_prefix_pages=args.trace_prefix_pages,
        private_cold_buffers=args.private_cold_buffers,
        audit_native_copies=args.audit_native_copies,
        native_kernel_fallbacks=args.native_kernels,
        optimized_canonical=args.optimized_canonical,
        stable_expert_layout=args.stable_expert_layout,
        mlp_numerical_control=args.mlp_numerical_control,
        sustained_decode=args.sustained_decode,
        sustained_real_idle_expiry=args.sustained_real_idle_expiry,
        sustained_uniform_outputs=args.sustained_uniform_outputs,
        sustained_observe_lookups=args.sustained_observe_lookups,
        sustained_idle_ttl=args.sustained_idle_ttl,
        sustained_output_tokens=args.sustained_output_tokens,
        expert_order_control=args.expert_order_control,
        stable_qsa_selection=args.stable_qsa_selection,
        stable_qsa_ties=args.stable_qsa_ties,
        cuda_launch_blocking=args.cuda_launch_blocking,
        trace_host_registration=args.trace_host_registration,
        kernel_flags={
            key: value
            for key, value in (row.split("=", 1) for row in env)
            if key in KERNEL_FLAGS + EXPERIMENTAL_KERNEL_FLAGS
        },
        storage_backend=args.storage_backend,
        started=False,
    )
    args.evidence.write_text(json.dumps(evidence, indent=2) + "\n")
    if source["state"]["Running"]:
        require_idle(SOURCE)
        print("Stopping and retaining exact idle development source", flush=True)
        remote("docker", "stop", "--time", "60", SOURCE)
    state = inspect(SOURCE)["state"]
    require(
        not state["Running"] and state["ExitCode"] == 0 and not state["OOMKilled"],
        "Development source did not stop cleanly",
    )
    require_free_quartet()
    remote("docker", "start", cid)
    evidence["started"] = True
    args.evidence.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2), flush=True)


if __name__ == "__main__":
    main()
