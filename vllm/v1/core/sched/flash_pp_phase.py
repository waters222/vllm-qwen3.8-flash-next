"""Bounded scheduling observer. No policy changes, GPU calls, or request payloads."""
import json
import os


class PhaseCapture:
    MAX_SLOTS = 16

    @classmethod
    def from_env(cls, scheduler, emit, environ=None):
        env = os.environ if environ is None else environ
        raw = env.get("VLLM_FLASH_PP_PHASE_STEPS", "0")
        if not isinstance(raw, str) or not raw.isascii() or not raw.isdecimal():
            raise ValueError("VLLM_FLASH_PP_PHASE_STEPS must be an integer 0..256")
        steps = int(raw)
        if not 0 <= steps <= 256:
            raise ValueError("VLLM_FLASH_PP_PHASE_STEPS must be an integer 0..256")
        if not steps:
            return None  # Do not inspect scheduler, start flag, or callback.
        raw_start = env.get("VLLM_FLASH_PP_PHASE_START_STEP", "0")
        if not isinstance(raw_start, str) or not raw_start.isascii() or not raw_start.isdecimal():
            raise ValueError("VLLM_FLASH_PP_PHASE_START_STEP must be a nonnegative integer")
        start = int(raw_start)
        if start > 2**31 - 1:
            raise ValueError("VLLM_FLASH_PP_PHASE_START_STEP exceeds 2147483647")
        config = scheduler.vllm_config
        if not (
            scheduler.use_v2_model_runner
            and scheduler.scheduler_config.async_scheduling
            and type(scheduler).__module__ == "vllm.v1.core.sched.async_scheduler"
            and type(scheduler).__name__ == "AsyncScheduler"
            and scheduler.parallel_config.pipeline_parallel_size == 2
            and scheduler.parallel_config.data_parallel_size == 1
            and config.speculative_config is None
            and scheduler.num_spec_tokens == 0
            and getattr(config.model_config.hf_config, "model_type", None) == "qwen4_exp"
        ):
            raise ValueError("PP phase capture requires standard AsyncScheduler, V2, PP2, DP1, Qwen4Exp, no speculation")
        def bounded(name, default, low, high):
            value=env.get(name,default)
            if not isinstance(value,str) or not value.isascii() or not value.isdecimal() or not low <= int(value) <= high:
                raise ValueError(f"{name} must be an integer {low}..{high}")
            return int(value)
        captures=bounded("VLLM_FLASH_PP_PHASE_CAPTURES","1",1,8)
        minimum=bounded("VLLM_FLASH_PP_PHASE_MIN_RUNNING","0",0,16)
        return cls(steps, start, emit, captures, minimum)

    def __init__(self, steps, start, emit, captures=1, minimum=0):
        self.steps, self.start, self.emit = steps, start, emit
        self.capture_limit, self.minimum = captures, minimum
        self.emitted = 0
        self.awaiting_idle = False
        self.burst_mode = captures > 1 or minimum > 0
        self.records = []
        self.slots = {}
        self.next_slot = 0
        self.pending = None
        self.done = False

    def _finish(self, reason):
        self.emitted += 1
        payload=json.dumps(dict(kind="flash-pp-phase-v2", complete=len(self.records)==self.steps,
            requested_steps=self.steps, recorded_steps=len(self.records), capture_ordinal=self.emitted,
            capture_limit=self.capture_limit, stop_reason=reason, records=self.records),separators=(",", ":"))
        self.records.clear(); self.slots.clear(); self.next_slot=0
        self.done=self.emitted == self.capture_limit
        self.awaiting_idle=not self.done
        self.emit(payload)

    def observe_idle(self, scheduler):
        if self.done or not self.burst_mode or self.pending is not None:
            return False
        # Include retained/queued/paused requests: an empty decode step is not idle.
        if (scheduler.requests or scheduler.running or scheduler.waiting or scheduler.skipped_waiting
                or scheduler.num_waiting_for_streaming_input):
            return False
        if self.records:
            self._finish("burst-drained")
        self.awaiting_idle=False
        return True

    def before(self, scheduler, output):
        if self.done or scheduler.current_step < self.start:
            return
        if self.pending is not None:
            raise RuntimeError("PP phase observer called twice before update")
        if self.observe_idle(scheduler):
            return
        if self.awaiting_idle or (not self.records and len(scheduler.running) < self.minimum):
            return
        running = scheduler.running[:self.MAX_SLOTS]
        active = {id(r) for r in running}
        self.slots = {key:value for key,value in self.slots.items() if key in active}
        rows, references = [], []
        for request in running:
            key = id(request)
            if key not in self.slots:
                self.slots[key] = (request, self.next_slot)
                self.next_slot += 1
            slot = self.slots[key][1]
            scheduled = output.num_scheduled_tokens.get(request.request_id, 0)
            prefill = bool(request.is_prefill_chunk)
            prompt_uncomputed = request.num_computed_tokens < request.num_prompt_tokens
            rows.append(dict(slot=slot, selected=scheduled > 0, scheduled_tokens=int(scheduled),
                             prefill_before=prefill,
                             prompt_uncomputed_before=prompt_uncomputed,
                             completes_prompt=bool(prompt_uncomputed and scheduled > 0 and
                                 request.num_computed_tokens + scheduled >= request.num_prompt_tokens),
                             eligible_before=int(request.next_decode_eligible_step)))
            references.append(request)
        record = dict(step=int(scheduler.current_step), phase=int(scheduler.current_step % 2),
                      scheduled_tokens=int(output.total_num_scheduled_tokens),
                      running_count=len(scheduler.running), waiting_count=len(scheduler.waiting),
                      skipped_waiting_count=len(scheduler.skipped_waiting),
                      new_or_resumed_count=len(output.scheduled_new_reqs),
                      truncated=len(scheduler.running) > self.MAX_SLOTS, requests=rows)
        self.pending = (record, references)

    def after(self, scheduler, output):
        if self.done or self.pending is None:
            return
        record, references = self.pending
        if scheduler.current_step != record["step"]:
            raise RuntimeError("PP phase observer step changed during update")
        for row, request in zip(record["requests"], references):
            row.update(prefill_after=bool(request.is_prefill_chunk),
                       prompt_uncomputed_after=request.num_computed_tokens < request.num_prompt_tokens,
                       eligible_after=int(request.next_decode_eligible_step))
        self.records.append(record)
        self.pending = None
        if len(self.records) == self.steps:
            self._finish("step-budget")
