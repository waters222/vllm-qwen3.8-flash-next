# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drained-engine transactions for Flash-Next hot/cold request ownership.

This coordinator runs only on the engine thread. It provides idle-session LRU
and TTL, not a public session API. A failed/ambiguous commit holds both the
scheduler pause and allocation ownership until explicitly recovered.
"""

import time
from collections import Counter, OrderedDict
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any


@dataclass
class SessionTransaction:
    request: Any
    key: dict[str, Any]
    original_status: Any
    layout: Any
    request_data: Any
    phase: str = "capturing"
    destination: Any = None
    cold_since: float | None = None


class FlashSessionTransactions:
    def __init__(
        self,
        engine,
        allocator,
        statuses,
        paused_all,
        new_request_data,
        *,
        ttl_seconds=3600,
        clock=time.monotonic,
        unpaused="unpaused",
    ):
        if ttl_seconds < 0:
            raise ValueError("cold session TTL must be nonnegative")
        self.engine = engine
        self.scheduler = engine.scheduler
        self.allocator = allocator
        self.statuses = statuses
        self.paused_all = paused_all
        self.new_request_data = new_request_data
        self.world_size = engine.vllm_config.parallel_config.world_size
        self.records: dict[str, SessionTransaction] = {}
        self.generation = 0
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self.unpaused = unpaused
        self.automatic_pause = False
        self.policy_error: str | None = None
        self.retry_after = 0.0
        self.counters: Counter[str] = Counter()
        self.hot_last_used: dict[tuple[str, int], float] = {}
        self.hot_generations: dict[tuple[str, int], int] = {}
        self.tombstones: OrderedDict[tuple[str, int, int], str] = OrderedDict()

    def observe_hot_sessions(self) -> None:
        """Observe active work, not status polling, for idle-session LRU order."""
        live = set()
        now = self.clock()
        for request in self.scheduler.requests.values():
            key = (request.request_id, request.client_index)
            live.add(key)
            if request.status == self.statuses.RUNNING:
                self.hot_last_used[key] = now
            elif request.status == self.statuses.WAITING_FOR_STREAMING_REQ:
                self.hot_last_used.setdefault(key, now)
        for key in self.hot_last_used.keys() - live:
            del self.hot_last_used[key]
            self.hot_generations.pop(key, None)

    def _expired(self, record) -> bool:
        return (
            record.phase == "cold"
            and record.original_status == self.statuses.WAITING_FOR_STREAMING_REQ
            and bool(self.ttl_seconds)
            and record.cold_since is not None
            and self.clock() - record.cold_since >= self.ttl_seconds
        )

    def _miss(self, reason):
        self.counters[f"miss_{reason}"] += 1
        return {
            "phase": "miss",
            "cache": "miss",
            "reason": reason,
            "requires_prefill": True,
        }

    def lookup(self, request_id, client_index, generation=None) -> dict:
        """Probe retained state without extending TTL or changing LRU order."""
        tombstone = self.tombstones.get((request_id, client_index, generation))
        if tombstone is not None:
            return self._miss(tombstone)
        request = self.scheduler.requests.get(request_id)
        if request is None or request.client_index != client_index:
            return self._miss("not_retained")
        record = self.records.get(request_id)
        if record is not None:
            if generation is not None and record.key["generation"] != generation:
                return self._miss("not_retained")
            if self._expired(record):
                return self._miss("expired")
            if record.phase != "cold":
                return {"phase": "pending", "cache": "unavailable"}
            self.counters["cold_hits"] += 1
            return {"phase": "cold", "cache": "cold_hit", "key": record.key}
        if request.status in (
            self.statuses.RUNNING,
            self.statuses.WAITING_FOR_STREAMING_REQ,
        ):
            if generation is not None and generation != self.hot_generations.get(
                (request_id, client_index), 0
            ):
                return self._miss("not_retained")
            self.counters["hot_hits"] += 1
            return {"phase": "hot", "cache": "hot_hit"}
        return self._miss("not_retained")

    def evict_lru(self) -> dict:
        """Spill the oldest idle hot session; never evict active generation."""
        self._require_drained()
        self.assert_can_resume()
        self.observe_hot_sessions()
        candidates = [
            request
            for request in self.scheduler.requests.values()
            if request.status == self.statuses.WAITING_FOR_STREAMING_REQ
            and not request.streaming_queue
        ]
        if not candidates:
            return {"phase": "unchanged", "reason": "no_idle_victim"}
        victim = min(
            candidates,
            key=lambda r: (
                self.hot_last_used[(r.request_id, r.client_index)],
                r.request_id,
            ),
        )
        result = self.suspend(victim.request_id, victim.client_index)
        self.counters["lru_spills"] += 1
        return result

    def expire_idle(self) -> dict:
        """Retire expired cold state through the same all-rank cancellation path."""
        self._require_drained()
        self.assert_can_resume()
        expired = []
        for record in list(self.records.values()):
            if not self._expired(record):
                continue
            request_id = record.request.request_id
            self.cancel(request_id, reason="expired")
            finished = self.scheduler.finish_requests(
                request_id, self.statuses.FINISHED_ABORTED
            )
            send_aborts = getattr(self.engine, "_send_abort_outputs", None)
            if send_aborts is not None:
                send_aborts(finished)
            expired.append(record.key)
        return {"phase": "expired", "sessions": expired}

    def stats(self) -> dict:
        scheduler = self.scheduler
        pool = scheduler.kv_cache_manager.block_pool
        return {
            "policy": "idle_lru",
            "cold_idle_ttl_seconds": self.ttl_seconds,
            "cold_sessions": sum(r.phase == "cold" for r in self.records.values()),
            "unresolved_transactions": sum(
                r.phase != "cold" for r in self.records.values()
            ),
            "counters": dict(self.counters),
            "automatic_pause": self.automatic_pause,
            "policy_error": self.policy_error,
            "scheduler": {
                "retained_requests": len(scheduler.requests),
                "running": len(scheduler.running),
                "idle_hot": scheduler.num_waiting_for_streaming_input,
                "waiting": len(scheduler.waiting),
                "skipped_waiting": len(scheduler.skipped_waiting),
                "status_counts": dict(
                    Counter(
                        getattr(r.status, "name", str(r.status))
                        for r in scheduler.requests.values()
                    )
                ),
                "scheduled_step": scheduler.sched_step_seq,
                "processed_step": scheduler.processed_step_seq,
                "pending_batches": len(self.engine.batch_queue or ()),
                "free_gpu_blocks": pool.get_num_free_blocks(),
                "usable_gpu_blocks": pool.num_gpu_blocks - 1,
            },
        }

    def describe(self, request_id: str, client_index: int) -> dict:
        """Internal drained-engine allocation evidence; never return token contents."""
        self._require_drained()
        request = self.scheduler.requests.get(request_id)
        if request is None or request.client_index != client_index:
            return {"phase": "miss", "reason": "not_retained"}
        record = self.records.get(request_id)
        if record is None:
            layout = self.allocator.capture_layout(request_id)
            phase = "hot"
        else:
            layout = record.destination or record.layout
            phase = record.phase
        return {
            "phase": phase,
            "request_status": request.status.name,
            "computed_tokens": request.num_computed_tokens,
            "token_count": len(request.all_token_ids),
            "block_ids": layout.block_ids,
            "null_block_id": layout.null_block_id,
            "owns_hot_blocks": any(
                request_id in group.req_to_blocks for group in self.allocator.groups
            ),
        }

    def _idle_victims(self):
        return [
            r
            for r in self.scheduler.requests.values()
            if r.status == self.statuses.WAITING_FOR_STREAMING_REQ
            and not r.streaming_queue
        ]

    def _hot_slots_full(self):
        return (
            len(self.scheduler.running) + self.scheduler.num_waiting_for_streaming_input
            >= self.scheduler.max_num_running_reqs
        )

    def _waiting_pressure(self):
        return bool(self.scheduler.waiting) and (
            self._hot_slots_full() or self.scheduler.kv_cache_manager.usage >= 0.9
        )

    def maintenance_needed(self) -> bool:
        return any(
            self._expired(r) or (r.phase == "cold" and bool(r.request.streaming_queue))
            for r in self.records.values()
        ) or (self._waiting_pressure() and bool(self._idle_victims()))

    def automatic_tick(self) -> None:
        """Drain first; then expire, restore queued continuations, or spill LRU.

        A policy error leaves the engine paused with its retained allocations.
        It must not trigger repeated retries or resume partially restored ranks.
        """
        if self.policy_error or self.engine._idle_state_callbacks:
            return
        if not self.automatic_pause:
            if (
                self.scheduler.pause_state != self.unpaused
                or self.clock() < self.retry_after
                or not self.maintenance_needed()
            ):
                return
            self.scheduler.set_pause_state(self.paused_all)
            self.automatic_pause = True
        if (
            self.engine.batch_queue
            or self.scheduler.processed_step_seq != self.scheduler.sched_step_seq
        ):
            return
        try:
            self.expire_idle()
            for record in list(self.records.values()):
                if record.phase != "cold" or not record.request.streaming_queue:
                    continue
                if self._hot_slots_full() and self._idle_victims():
                    self.evict_lru()
                key = record.key
                try:
                    self.restore(
                        key["request_id"], key["client_index"], key["generation"]
                    )
                except MemoryError:
                    if not self._idle_victims():
                        raise
                    self.evict_lru()
                    self.restore(
                        key["request_id"], key["client_index"], key["generation"]
                    )
            if self._waiting_pressure() and self._idle_victims():
                self.evict_lru()
        except MemoryError:
            # Capacity refusal is retryable only if every transaction still has
            # an unambiguous cold owner. Integrity/transport errors are not.
            if any(r.phase != "cold" for r in self.records.values()):
                self.policy_error = "capacity_failure_with_unresolved_transaction"
            else:
                self.counters["admission_refusals"] += 1
                self.retry_after = self.clock() + 1.0
        except Exception as exc:
            self.policy_error = type(exc).__name__
            self.counters["policy_failures"] += 1
        self.automatic_pause = False
        if self.policy_error is None:
            self.assert_can_resume()
            self.scheduler.set_pause_state(self.unpaused)

    def _require_drained(self) -> None:
        scheduler = self.scheduler
        if (
            scheduler.pause_state != self.paused_all
            or self.engine.batch_queue
            or self.engine._idle_state_callbacks
            or scheduler.processed_step_seq != scheduler.sched_step_seq
        ):
            raise ValueError("session swap requires a completed keep/no-clear pause")

    def assert_can_resume(self) -> None:
        if self.policy_error or any(
            record.phase != "cold" for record in self.records.values()
        ):
            raise RuntimeError("unresolved session transaction; keep engine paused")

    def _rpc(self, record, operation, expected, **payload):
        responses = self.engine.collective_rpc(
            "flash_session_swap",
            args=(operation, {"key": record.key, **payload}),
        )
        if len(responses) != self.world_size or any(
            not isinstance(response, dict)
            or response.get("key") != record.key
            or response.get("phase") not in expected
            for response in responses
        ):
            raise RuntimeError("incomplete or mismatched session rank acknowledgements")
        return responses

    def _record(self, request_id, client_index, generation):
        record = self.records[request_id]
        if (
            record.key
            != {
                "request_id": request_id,
                "client_index": client_index,
                "generation": generation,
            }
            or self.scheduler.requests.get(request_id) is not record.request
        ):
            raise ValueError("session identity or generation changed")
        return record

    def _detach(self, record):
        request = record.request
        if record.original_status == self.statuses.RUNNING:
            self.scheduler.running.remove(request)
        else:
            self.scheduler.waiting.remove_requests([request])
            self.scheduler.skipped_waiting.remove_requests([request])
            self.scheduler.num_waiting_for_streaming_input -= 1
        request.status = self.statuses.WAITING_FOR_COLD_SESSION
        # Cold sessions are retained in requests, not in runnable/waiting queues.
        # They must not make an otherwise idle engine spin or occupy hot slots.

    def _attach(self, record):
        request = record.request
        request.status = record.original_status
        self.hot_generations[(request.request_id, request.client_index)] = record.key[
            "generation"
        ]
        del self.records[request.request_id]
        if request.status == self.statuses.RUNNING:
            self.scheduler.running.append(request)
        else:
            self.scheduler.num_waiting_for_streaming_input += 1
            if request.streaming_queue:
                update = request.streaming_queue.popleft()
                if update is None:
                    self.scheduler.finish_requests(
                        request.request_id, self.statuses.FINISHED_ABORTED
                    )
                    return
                self.scheduler._update_request_as_session(request, update)
            self.scheduler._enqueue_waiting_request(request)

    def suspend(self, request_id: str, client_index: int) -> dict:
        self._require_drained()
        self.assert_can_resume()
        request = self.scheduler.requests[request_id]
        if request.client_index != client_index or request_id in self.records:
            raise ValueError("session identity mismatch or already suspended")
        if request.status not in (
            self.statuses.RUNNING,
            self.statuses.WAITING_FOR_STREAMING_REQ,
        ):
            raise ValueError(
                "only allocated running or idle-streaming sessions can swap"
            )
        if (
            request.num_in_flight_tokens
            or request.num_stale_output_tokens
            or request.num_output_placeholders
            or request.num_computed_tokens < request.num_prompt_tokens
            or request.mm_features
            or request.prompt_embeds is not None
            or request.structured_output_request is not None
            or request.pooling_params is not None
            or request.lora_request is not None
            or request.sampling_params is None
            or request.sampling_params.prompt_logprobs is not None
        ):
            raise ValueError("unsupported or undrained session state")
        layout = self.allocator.capture_layout(request_id)
        request_data = deepcopy(
            self.new_request_data.from_request(
                request,
                tuple(list(ids) for ids in layout.block_ids),
                prefill_token_ids=list(request.all_token_ids),
            )
        )
        self.generation += 1
        record = SessionTransaction(
            request,
            dict(
                request_id=request_id,
                client_index=client_index,
                generation=self.generation,
            ),
            request.status,
            layout,
            request_data,
        )
        self.records[request_id] = record
        self._detach(record)
        try:
            self._rpc(
                record,
                "capture",
                {"captured"},
                block_ids=layout.block_ids,
                computed_tokens=request.num_computed_tokens,
            )
        except Exception:
            # Recover only after *every* rank confirms hot ownership remains.
            self._rpc(record, "abort_capture", {"retired", "absent"})
            self._attach(record)
            raise
        record.phase = "committing_cold"
        return self.complete_suspend(request_id, client_index, self.generation)

    def complete_suspend(self, request_id, client_index, generation) -> dict:
        """Retry an ambiguous cold commit without repeating capture or freeing early."""
        self._require_drained()
        record = self._record(request_id, client_index, generation)
        if record.phase == "cold":
            return {"phase": "cold", "key": record.key}
        if record.phase != "committing_cold":
            raise ValueError("session is not awaiting a cold commit")
        self._rpc(record, "commit_cold", {"cold"})
        # If local release unexpectedly fails, prohibit any retry that might
        # release twice. The engine must remain paused for explicit recovery.
        record.phase = "releasing_hot"
        self.allocator.release(request_id, record.layout)
        record.phase = "cold"
        record.cold_since = self.clock()
        self.counters["suspends"] += 1
        return {"phase": "cold", "key": record.key}

    def restore(self, request_id, client_index, generation) -> dict:
        self._require_drained()
        state = self.lookup(request_id, client_index, generation)
        if state["phase"] == "hot":
            return state
        if state["phase"] == "miss":
            if state["reason"] == "expired":
                self.expire_idle()
            return state
        record = self._record(request_id, client_index, generation)
        if record.phase != "cold":
            raise ValueError("session is not cold")
        self.assert_can_resume()
        if (
            len(self.scheduler.running) + self.scheduler.num_waiting_for_streaming_input
            >= self.scheduler.max_num_running_reqs
        ):
            raise MemoryError("no scheduler hot request slot available")
        record.destination = self.allocator.reserve(request_id, record.layout)
        record.phase = "restoring"
        request_data = replace(
            record.request_data,
            block_ids=tuple(list(ids) for ids in record.destination.block_ids),
        )
        try:
            self._rpc(record, "restore", {"restored"}, new_request_data=request_data)
        except Exception:
            # No release if any rank might still own or write the destination.
            self._rpc(record, "rollback_restore", {"cold"})
            record.phase = "releasing_destination"
            self.allocator.release(request_id, record.destination)
            record.destination = None
            record.phase = "cold"
            raise
        record.phase = "retiring_cold"
        return self.complete_restore(request_id, client_index, generation)

    def complete_restore(self, request_id, client_index, generation) -> dict:
        self._require_drained()
        record = self._record(request_id, client_index, generation)
        if record.phase != "retiring_cold":
            raise ValueError("session is not awaiting restore commit")
        self._rpc(record, "retire", {"retired"}, expected_phase="restored")
        self._attach(record)
        self.counters["restores"] += 1
        self.hot_last_used[(request_id, client_index)] = self.clock()
        return {"phase": "hot", "key": record.key, "cache": "cold_hit"}

    def cancel(self, request_id: str, reason: str = "cancelled") -> None:
        """Scheduler finish hook. Pure cold checkpoints have no GPU operations."""
        record = self.records.get(request_id)
        if record is None:
            return
        if record.phase not in ("cold", "cancelling"):
            raise RuntimeError("cannot cancel an unresolved session transaction")
        record.phase = "cancelling"
        try:
            self._rpc(record, "retire", {"retired"}, expected_phase="cold")
        except Exception:
            # Some ranks may already have discarded their cold copy. Do not
            # allow restoration or any other transaction until cancel retries.
            self.scheduler.set_pause_state(self.paused_all)
            raise
        del self.records[request_id]
        key = record.key
        self.tombstones[(request_id, key["client_index"], key["generation"])] = reason
        while len(self.tombstones) > 1024:
            self.tombstones.popitem(last=False)
        self.counters[f"retired_{reason}"] += 1
