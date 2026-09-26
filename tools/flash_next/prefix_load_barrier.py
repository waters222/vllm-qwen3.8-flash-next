"""Diagnostic-only native cold-load promotion barrier for one fixed C2 pair."""

import json
import re
import time
from pathlib import Path


def _expiry_components(scheduler, host_module, cpu_module):
    if host_module is None:
        import vllm.v1.hisparse.block_pool as host_module
    if cpu_module is None:
        import vllm.v1.kv_offload.cpu.manager as cpu_module

    connector = scheduler.connector.connector_scheduler
    pools = {
        id(manager.block_pool): manager.block_pool
        for manager in scheduler.kv_cache_manager.coordinator.single_type_managers
        if isinstance(manager.block_pool, host_module.IdleExpiringHostBlockPool)
    }
    cpu = connector.manager
    if (
        not pools
        or cpu.idle_ttl_seconds != 3600
        or any(pool.idle_ttl_seconds != 3600 for pool in pools.values())
    ):
        raise RuntimeError("Expiry requires host/CPU pools with 3600-second TTL")
    return pools, cpu, host_module, cpu_module


def _advance_expiry(pools, cpu, host_module, cpu_module):
    originals = [(module, module.monotonic) for module in (host_module, cpu_module)]
    before = cpu._expired_chunks
    try:
        for module, clock in originals:
            module.monotonic = lambda clock=clock: clock() + 3601
        expired_host = sum(pool.expire_idle() for pool in pools.values())
        cpu._expire_idle()
    finally:
        for module, clock in originals:
            module.monotonic = clock
    return dict(
        configured_ttl_seconds=3600,
        advanced_seconds=3601,
        host_pools=len(pools),
        expired_host_blocks=expired_host,
        expired_cpu_chunks=cpu._expired_chunks - before,
        clocks_restored=all(module.monotonic is clock for module, clock in originals),
    )


def expire_idle_cache(scheduler, host_module=None, cpu_module=None):
    """Exercise native TTL only at a quiescent diagnostic request boundary."""
    connector = scheduler.connector.connector_scheduler
    if scheduler.requests or connector._jobs or connector._req_status:
        raise RuntimeError("Idle expiry requires drained requests and transfers")
    components = _expiry_components(scheduler, host_module, cpu_module)
    pools = components[0]
    if any(
        block.ref_cnt != 0
        for pool in pools.values()
        for block in pool.blocks
        if not block.is_null
    ):
        raise RuntimeError("Idle expiry requires unowned host pages")
    return dict(
        **_advance_expiry(*components),
        scope="Private diagnostic clocks only; native idle expiry at quiescence",
        quiescent=True,
    )


def expire_pending_cache(scheduler, load_jobs, host_module=None, cpu_module=None):
    """Expire idle state after native load preparation, before worker submission."""
    components = _expiry_components(scheduler, host_module, cpu_module)
    pools, cpu = components[:2]
    connector = scheduler.connector.connector_scheduler
    jobs = [connector._jobs[key] for key in load_jobs]
    if not jobs or any(
        job.is_store
        or job.pending_count <= 0
        or scheduler.requests[job.req_id].status.name != "WAITING_FOR_REMOTE_KVS"
        for job in jobs
    ):
        raise RuntimeError("Pending expiry requires unacknowledged native loads")
    host_blocks = [
        block
        for pool in pools.values()
        for block in pool.blocks
        if not block.is_null and block.ref_cnt > 0
    ]
    destinations = [
        block
        for req_id in {job.req_id for job in jobs}
        for group in scheduler.kv_cache_manager.get_blocks(req_id).blocks
        for block in group
        if not block.is_null
    ]
    keys = set().union(*(job.keys for job in jobs))
    chunks = {key: cpu._policy.get(key) for key in keys}
    if (
        not host_blocks
        or not destinations
        or not chunks
        or any(block.ref_cnt <= 0 for block in destinations)
        or any(chunk is None or chunk.ref_cnt <= 0 for chunk in chunks.values())
    ):
        raise RuntimeError("Pending expiry requires pinned host and transfer state")
    block_before = [
        (block, block.ref_cnt, block.block_hash) for block in host_blocks + destinations
    ]
    chunk_before = {key: chunk.ref_cnt for key, chunk in chunks.items()}
    evidence = _advance_expiry(*components)
    preserved = all(
        block.ref_cnt == refs and block.block_hash == block_hash
        for block, refs, block_hash in block_before
    ) and all(
        cpu._policy.get(key) is chunk and chunk.ref_cnt == chunk_before[key]
        for key, chunk in chunks.items()
    )
    if not preserved:
        raise RuntimeError("Expiry changed protected load ownership")
    return dict(
        **evidence,
        scope="Prepared loads before worker submission; not physical DMA failure",
        pending_load_jobs=len(jobs),
        protected_host_blocks=len(host_blocks),
        protected_cpu_chunks=len(chunks),
        owned_destination_blocks=len(destinations),
        ownership_preserved=preserved,
    )


class PairLoadGate:
    """Observe native acknowledgements; never allocate, release or edit requests."""

    def __init__(self, timeout=30):
        self.size = 2
        self.timeout = timeout
        self.started = None
        self.released = False
        self.deferred_calls = 0
        self.ready = []

    @staticmethod
    def branch(request_id):
        match = re.fullmatch(r"batch-cold-([01])(?:-[0-9a-f]{8})?", request_id)
        return int(match[1]) if match else None

    def hold(self, request, requests, finished, failed, now):
        if self.branch(request.request_id) is None or self.released:
            return False
        if request.status.name != "WAITING_FOR_REMOTE_KVS":
            return False
        if self.started is None:
            self.started = now
        if now - self.started >= self.timeout:
            raise TimeoutError("Diagnostic cold pair did not both finish native loads")
        pair = {}
        for candidate in requests.values():
            branch = self.branch(candidate.request_id)
            if branch is not None:
                if branch in pair:
                    raise AssertionError("Ambiguous diagnostic cold request identity")
                pair[branch] = candidate
        if any(r.request_id in failed for r in pair.values()):
            raise RuntimeError("Diagnostic cold pair has a failed native load")
        if len(pair) == self.size and all(
            r.status.name == "WAITING_FOR_REMOTE_KVS" and r.request_id in finished
            for r in pair.values()
        ):
            self.ready = [pair[i].request_id for i in range(self.size)]
            self.released = True
            return False
        self.deferred_calls += 1
        return True


class ContinuationLoadGate(PairLoadGate):
    """Bound one C4 cold continuation turn; observe native completion only."""

    def __init__(self, turn, timeout=30):
        if turn not in (1, 2, 3):
            raise ValueError("Continuation gate requires turn1–3")
        super().__init__(timeout)
        self.size = 4
        self.turn = turn

    def branch(self, request_id):
        match = re.fullmatch(
            rf"turn4-cold-{self.turn}-([0-3])(?:-[0-9a-f]{{8}})?", request_id
        )
        return int(match[1]) if match else None


def owned_request_blocks(scheduler, request_id):
    """Read native pool objects; never retain, release or rewrite a page."""
    groups = scheduler.kv_cache_manager.get_blocks(request_id).blocks
    return {
        (group_id, index): (block, block.block_hash, block.ref_cnt)
        for group_id, blocks in enumerate(groups)
        for index, block in enumerate(blocks)
        if not block.is_null
    }


def snapshot_resident_prefix_ownership(scheduler, request_id, boundary):
    """Observe native request hashes, page objects and membership without lookup."""
    from vllm.v1.core.kv_cache_utils import (
        make_block_hash_with_group_id,
        resolve_block_hashes,
    )

    if type(boundary) is not int or not 0 < boundary < 32768:
        raise ValueError("Resident ownership audit needs a bounded positive prefix")
    request = scheduler.requests[request_id]
    manager = scheduler.kv_cache_manager
    specs = manager.kv_cache_config.kv_cache_groups
    managers = manager.coordinator.single_type_managers
    blocks = manager.get_blocks(request_id).blocks
    if len(specs) != len(managers) or len(blocks) != len(specs):
        raise ValueError("Native ownership groups do not cover the request")
    groups = []
    for gid, (spec, single, owned) in enumerate(zip(specs, managers, blocks)):
        if not spec.host_resident:
            continue
        size, pool = single.block_size, single.block_pool
        if boundary % size or single.kv_cache_group_id != gid:
            raise ValueError("Resident ownership boundary/group is not aligned")
        count = boundary // size
        hashes = resolve_block_hashes(request.block_hashes, pool.hash_block_size, size)
        if len(owned) < count or len(hashes) < count:
            raise ValueError("Resident ownership prefix lacks pages or native hashes")
        pages = []
        for index, block in enumerate(owned[:count]):
            key = make_block_hash_with_group_id(hashes[index], gid)
            valid_id = 0 <= block.block_id < len(pool.blocks)
            checks = dict(
                owned_pool=block.pool is pool,
                owned_object=valid_id and pool.blocks[block.block_id] is block,
                nonnull=not block.is_null,
                live=block.ref_cnt > 0,
                expected_hash=block.block_hash == key,
                expected_hash_boundary=block.block_hash_num_tokens
                == (index + 1) * size,
                # Multiple independently computed versions of one hash are valid.
                # Do not compare only with the first version returned by lookup.
                registered=pool.cached_block_hash_to_block.contain(key, block.block_id),
            )
            pages.append(
                dict(
                    logical_index=index,
                    physical_id=block.block_id,
                    references=block.ref_cnt,
                    expected_key=key.hex(),
                    actual_key=block.block_hash.hex()
                    if block.block_hash is not None
                    else None,
                    hash_tokens=block.block_hash_num_tokens,
                    checks=checks,
                    passed=all(checks.values()),
                )
            )
        groups.append(dict(group_id=gid, block_size=size, pages=pages))
    if not groups:
        raise ValueError("Resident ownership audit found no host groups")
    return dict(
        request_id=request_id,
        boundary=boundary,
        groups=groups,
        passed=all(page["passed"] for group in groups for page in group["pages"]),
        scope="Native resident-prefix ownership metadata, not tensor bytes or "
        "transfer completion. Read-only: no cache lookup, expiry or refcount changes.",
    )


def compare_owned_request_blocks(before, after):
    checks = dict(
        complete_owned_table=bool(before) and before.keys() == after.keys(),
        same_pages_and_hashes=bool(before)
        and all(
            key in after and old[0] is after[key][0] and old[1] == after[key][1]
            for key, old in before.items()
        ),
        live_references=bool(after) and all(row[2] > 0 for row in after.values()),
    )
    return dict(
        checks=checks,
        passed=all(checks.values()),
        blocks=len(before),
        groups=len({group for group, _ in before}),
        references_before=sum(row[2] for row in before.values()),
        references_after=sum(row[2] for row in after.values()),
    )


def collect_scheduled_resident_ownership(scheduler, output, plan, records):
    """Snapshot selected prefill boundaries after native schedule allocation."""
    matches = [re.fullmatch(r"turn([24])-(?:hot|cold)-[0-3]", name) for name in plan]
    if (not matches or len(matches) > 8 or not all(matches)
            or len({match[1] for match in matches}) != 1):
        raise ValueError("Resident ownership requires one bounded C2/C4 plan")
    concurrency = int(matches[0][1])
    added = 0
    for request_id, scheduled in output.num_scheduled_tokens.items():
        request = scheduler.requests[request_id]
        # Native Scheduler._update_after_schedule has already advanced this count.
        boundary = request.num_computed_tokens - scheduled
        phases = [
            name
            for name, boundaries in plan.items()
            if boundary in boundaries
            and re.fullmatch(
                re.escape(name) + rf"-[0-{concurrency - 1}](?:-[0-9a-f]{{8}})?",
                request_id,
            )
        ]
        if not phases:
            continue
        if (
            len(phases) != 1
            or scheduled <= 0
            or request.num_output_tokens != 0
            or boundary >= request.num_prompt_tokens
            or len(records) >= concurrency * 16
            or any(
                row["request_id"] == request_id and row["boundary"] == boundary
                for row in records
            )
        ):
            raise ValueError(
                "Ambiguous, duplicate or excess resident ownership boundary"
            )
        record = snapshot_resident_prefix_ownership(scheduler, request_id, boundary)
        record.update(phase=phases[0], scheduled_tokens=scheduled)
        records.append(record)
        added += 1
    return added


def expire_active_cache(scheduler, request_ids, host_module=None, cpu_module=None):
    """Expire idle entries while paused live requests retain native ownership."""
    if (
        len(request_ids) not in (2, 4)
        or len(set(request_ids)) != len(request_ids)
        or any(
            name not in scheduler.requests
            or scheduler.requests[name].status.name != "RUNNING"
            for name in request_ids
        )
    ):
        raise RuntimeError("Active expiry requires C2/C4 distinct running requests")
    components = _expiry_components(scheduler, host_module, cpu_module)
    pools, cpu = components[:2]
    before = {name: owned_request_blocks(scheduler, name) for name in request_ids}
    host = [
        (block, block.ref_cnt, block.block_hash)
        for pool in pools.values()
        for block in pool.blocks
        if not block.is_null and block.ref_cnt > 0
    ]
    jobs = scheduler.connector.connector_scheduler._jobs
    keys = {key for job in jobs.values() if job.pending_count > 0 for key in job.keys}
    chunks = {key: cpu._policy.get(key) for key in keys}
    if (
        not host
        or any(
            not rows or any(row[2] <= 0 for row in rows.values())
            for rows in before.values()
        )
        or any(chunk is None or chunk.ref_cnt == 0 for chunk in chunks.values())
    ):
        raise RuntimeError("Active expiry requires owned request and transfer state")
    chunk_before = {
        key: (chunk, chunk.ref_cnt, chunk.chunk_id) for key, chunk in chunks.items()
    }
    evidence = _advance_expiry(*components)
    ownership = {
        name: compare_owned_request_blocks(rows, owned_request_blocks(scheduler, name))
        for name, rows in before.items()
    }
    preserved = (
        all(row["passed"] for row in ownership.values())
        and all(
            before[name][key][2] == row[2]
            for name in before
            for key, row in owned_request_blocks(scheduler, name).items()
        )
        and all(
            block.ref_cnt == refs and block.block_hash == value
            for block, refs, value in host
        )
        and all(
            cpu._policy.get(key) is chunk
            and (chunk.ref_cnt, chunk.chunk_id) == (refs, slot)
            for key, (chunk, refs, slot) in chunk_before.items()
        )
    )
    if not preserved:
        raise RuntimeError("Active expiry changed protected ownership")
    return dict(
        **evidence,
        scope="Paused live request ownership across private-clock idle expiry",
        active_requests=len(before),
        protected_host_blocks=len(host),
        protected_cpu_chunks=len(chunks),
        ownership=ownership,
        ownership_preserved=preserved,
    )


class PairedLoadMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = kwargs["vllm_config"]
        self._diagnostic_load_gate = PairLoadGate()
        self._diagnostic_continuation_gates = {
            turn: ContinuationLoadGate(turn) for turn in (1, 2, 3)
        } if config.additional_config.get("flash_next_diagnostic_c4_load_gate") else {}
        self._diagnostic_load_path = Path(
            config.additional_config["flash_next_diagnostic_load_barrier"]
        )
        self._diagnostic_load_report = dict(
            scope="diagnostic promotion timing only; native ownership and transfers",
            async_scheduling=config.scheduler_config.async_scheduling,
            scheduler_class=type(self).__name__,
            released=False,
            promoted=[],
            continuation_turns={},
        )
        with self._diagnostic_load_path.open("x") as stream:
            json.dump(self._diagnostic_load_report, stream)
        path = config.additional_config.get("flash_next_diagnostic_idle_expiry")
        self._diagnostic_expiry_path = Path(path) if path is not None else None
        self._diagnostic_expired = False
        path = config.additional_config.get("flash_next_diagnostic_pending_expiry")
        self._diagnostic_pending_expiry_path = Path(path) if path is not None else None
        self._diagnostic_pending_expired = False
        path = config.additional_config.get("flash_next_diagnostic_active_abort")
        self._diagnostic_active_abort_path = Path(path) if path is not None else None
        self._diagnostic_active_survivors = tuple(
            config.additional_config.get(
                "flash_next_diagnostic_active_survivors", ["cancel-survivor"]
            )
        )
        if (
            len(self._diagnostic_active_survivors) not in (1, 3)
            or len(set(self._diagnostic_active_survivors))
            != len(self._diagnostic_active_survivors)
            or any(
                not re.fullmatch(r"cancel-survivor(?:-[0-2])?", name)
                for name in self._diagnostic_active_survivors
            )
        ):
            raise ValueError("Active audit requires explicit C2/C4 survivor names")
        self._diagnostic_active_expiry = config.additional_config.get(
            "flash_next_diagnostic_active_expiry", False
        )
        path = config.additional_config.get("flash_next_diagnostic_resident_ownership")
        self._diagnostic_resident_path = Path(path) if path is not None else None
        self._diagnostic_resident_plan = config.additional_config.get(
            "flash_next_diagnostic_resident_boundaries", {}
        )
        self._diagnostic_resident_records = []
        if self._diagnostic_resident_path is not None:
            if not self._diagnostic_resident_plan:
                raise ValueError(
                    "Resident ownership audit requires explicit boundaries"
                )
            with self._diagnostic_resident_path.open("x") as stream:
                json.dump(dict(records=[]), stream)
        if (
            self._diagnostic_active_expiry
            and self._diagnostic_active_abort_path is None
        ):
            raise ValueError("Active expiry requires the paused abort audit")

    def finish_requests(self, request_ids, finished_status):
        path = self._diagnostic_active_abort_path
        if path is None:
            return super().finish_requests(request_ids, finished_status)
        ids = (
            list(self.requests)
            if request_ids is None
            else [request_ids]
            if isinstance(request_ids, str)
            else list(request_ids)
        )
        victim = [
            name
            for name in ids
            if re.fullmatch(r"cancel-victim(?:-[0-9a-f]{8})?", name)
            and name in self.requests
        ]
        if len(ids) != 1 or not victim:
            return super().finish_requests(ids, finished_status)
        expected = getattr(self, "_diagnostic_active_survivors", ("cancel-survivor",))
        matches = {
            external: [
                name for name in self.requests
                if re.fullmatch(re.escape(external) + r"(?:-[0-9a-f]{8})?", name)
            ]
            for external in expected
        }
        if (
            len(victim) != 1
            or any(len(ids) != 1 for ids in matches.values())
            or path.exists()
        ):
            raise RuntimeError("Active abort audit requires every expected survivor")
        peers = {external: self.requests[ids[0]] for external, ids in matches.items()}
        if any(peer.status.name != "RUNNING" for peer in peers.values()):
            raise RuntimeError("Active abort audit requires running survivors")
        before = {
            name: owned_request_blocks(self, peer.request_id)
            for name, peer in peers.items()
        }
        expiry = (
            expire_active_cache(
                self, [victim[0], *[p.request_id for p in peers.values()]]
            )
            if getattr(self, "_diagnostic_active_expiry", False)
            else None
        )
        result = super().finish_requests(ids, finished_status)
        survivors = {
            name: dict(
                survivor=peer.request_id,
                survivor_running=peer.status.name == "RUNNING",
                **compare_owned_request_blocks(
                    before[name], owned_request_blocks(self, peer.request_id)
                ),
            )
            for name, peer in peers.items()
        }
        for row in survivors.values():
            row["passed"] &= row["survivor_running"]
        evidence = dict(
            scope="Scheduler ownership immediately across native finish_requests",
            victim=victim[0],
            victim_removed_from_running=all(
                request.request_id != victim[0] for request in self.running
            ),
        )
        if len(survivors) == 1:
            evidence.update(next(iter(survivors.values())))
        else:
            evidence.update(
                survivors=survivors,
                passed=all(row["passed"] for row in survivors.values()),
            )
        if expiry is not None:
            evidence["active_expiry"] = expiry
        evidence["passed"] &= evidence["victim_removed_from_running"]
        path.write_text(json.dumps(evidence, indent=2) + "\n")
        if not evidence["passed"]:
            raise RuntimeError("Native abort changed survivor ownership")
        return result

    def schedule(self, *args, **kwargs):
        output = super().schedule(*args, **kwargs)
        owner_path = getattr(self, "_diagnostic_resident_path", None)
        if owner_path is not None and collect_scheduled_resident_ownership(
            self,
            output,
            self._diagnostic_resident_plan,
            self._diagnostic_resident_records,
        ):
            owner_path.write_text(
                json.dumps(dict(records=self._diagnostic_resident_records), indent=2)
                + "\n"
            )
        path = self._diagnostic_pending_expiry_path
        metadata = output.kv_connector_metadata
        if (
            path is not None
            and not self._diagnostic_pending_expired
            and metadata is not None
            and metadata.load_jobs
        ):
            connector = self.connector.connector_scheduler
            jobs = {
                key: value
                for key, value in metadata.load_jobs.items()
                if PairLoadGate.branch(connector._jobs[key].req_id) is not None
            }
            pair = {
                PairLoadGate.branch(request.request_id): request
                for request in self.requests.values()
                if PairLoadGate.branch(request.request_id) is not None
            }
            # An unacquired follower may legitimately miss after expiry. That
            # lifecycle cannot be held behind a two-cold-load comparison gate.
            pair_acquired = len(pair) == 2 and all(
                request.status.name == "WAITING_FOR_REMOTE_KVS"
                for request in pair.values()
            )
            if jobs and pair_acquired:
                jobs = {
                    key: job
                    for key, job in connector._jobs.items()
                    if not job.is_store
                    and job.pending_count > 0
                    and PairLoadGate.branch(job.req_id) is not None
                }
                evidence = expire_pending_cache(self, jobs)
                evidence["acquired_cold_requests"] = len(pair)
                with path.open("x") as stream:
                    json.dump(evidence, stream, indent=2)
                self._diagnostic_pending_expired = True
        return output

    def add_request(self, request):
        if self._diagnostic_expiry_path is not None and re.fullmatch(
            r"expiry-trigger(?:-[0-9a-f]{8})?", request.request_id
        ):
            if self._diagnostic_expired:
                raise RuntimeError("Diagnostic idle expiry may run only once")
            report = expire_idle_cache(self)
            with self._diagnostic_expiry_path.open("x") as stream:
                json.dump(report, stream, indent=2)
            self._diagnostic_expired = True
        return super().add_request(request)

    def _try_promote_blocked_waiting_request(self, request):
        gate = self._diagnostic_load_gate
        turn = None
        for candidate_turn, candidate in self._diagnostic_continuation_gates.items():
            if candidate.branch(request.request_id) is not None:
                turn, gate = candidate_turn, candidate
                break
        if gate.hold(
            request,
            self.requests,
            self.finished_recving_kv_req_ids,
            self.failed_recving_kv_req_ids,
            time.monotonic(),
        ):
            return False
        promoted = super()._try_promote_blocked_waiting_request(request)
        if promoted and gate.branch(request.request_id) is not None:
            report = self._diagnostic_load_report
            if turn is not None:
                report = report["continuation_turns"].setdefault(
                    str(turn), dict(promoted=[])
                )
            report.update(
                released=gate.released,
                ready_request_ids=gate.ready,
                deferred_calls=gate.deferred_calls,
            )
            report["promoted"].append(request.request_id)
            self._diagnostic_load_path.write_text(
                json.dumps(self._diagnostic_load_report, indent=2) + "\n"
            )
        return promoted


class CacheLookupObserverMixin:
    """Record native lookup decisions without gating or changing their results."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        path = Path(kwargs["vllm_config"].additional_config[
            "flash_next_diagnostic_lookup_observer"])
        self._lookup_observer_stream = path.open("x")
        connector = self.connector.connector_scheduler
        lookup = connector.get_num_new_matched_tokens
        allocate = connector.update_state_after_alloc
        count = 0

        def record(row):
            nonlocal count
            if count >= 4096:
                raise RuntimeError("Cache lookup observation limit exceeded")
            row["monotonic"] = time.monotonic()
            self._lookup_observer_stream.write(json.dumps(row) + "\n")
            self._lookup_observer_stream.flush()
            count += 1

        def observed_lookup(request, num_computed_tokens, *args, **kwargs):
            result = lookup(request, num_computed_tokens, *args, **kwargs)
            if request.request_id.startswith("sustained-"):
                record(dict(event="lookup", request_id=request.request_id,
                            local_tokens=num_computed_tokens,
                            external_tokens=result[0], asynchronous=result[1]))
            return result

        def observed_allocate(request, blocks, num_external_tokens):
            result = allocate(request, blocks, num_external_tokens)
            if request.request_id.startswith("sustained-"):
                state = connector._req_status[request.request_id]
                jobs = [dict(job_id=job_id, pending_workers=job.pending_count,
                             chunks=len(job.keys))
                        for job_id, job in connector._jobs.items()
                        if job.req_id == request.request_id and not job.is_store]
                record(dict(event="allocated", request_id=request.request_id,
                            local_tokens=state.num_locally_computed_tokens,
                            external_tokens=num_external_tokens, loads=jobs))
            return result

        connector.get_num_new_matched_tokens = observed_lookup
        connector.update_state_after_alloc = observed_allocate


def __getattr__(name):
    # Keep the pure gate importable in dependency-light controller tests.
    if name in ("PairedLoadAsyncScheduler", "CacheLookupAsyncScheduler"):
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler as native
    elif name in ("PairedLoadScheduler", "CacheLookupScheduler"):
        from vllm.v1.core.sched.scheduler import Scheduler as native
    else:
        raise AttributeError(name)
    mixin = CacheLookupObserverMixin if name.startswith("CacheLookup") else PairedLoadMixin
    cls = type(name, (mixin, native), {"__module__": __name__})
    globals()[name] = cls
    return cls
