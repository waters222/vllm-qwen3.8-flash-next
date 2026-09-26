"""Native streaming-session capacity/continuation screen; not an HTTP API.

Retain more idle sessions than worker slots, then resume by sending input.
The controller must prove automatic spill/restore and exact output equality;
successful transactions alone cannot waive an unstable reference.
"""

import asyncio

from qualify_session_swap import compare_tokens


def cleanup_complete(stats, lookups):
    """Require scheduler, block, and cold-state reclamation, not just misses."""
    scheduler = stats["scheduler"]
    return bool(
        stats["cold_sessions"] == 0
        and stats["unresolved_transactions"] == 0
        and stats["policy_error"] is None
        and all(row.get("cache") == "miss" for row in lookups)
        and all(
            scheduler[name] == 0
            for name in (
                "retained_requests",
                "running",
                "idle_hot",
                "waiting",
                "skipped_waiting",
                "pending_batches",
            )
        )
        and not scheduler["status_counts"]
        and scheduler["free_gpu_blocks"] == scheduler["usable_gpu_blocks"]
    )


class StreamingSession:
    """Hold a native input stream open while observing completed input turns."""

    def __init__(self, engine, params, name, input_factory):
        self.engine = engine
        self.name = name
        self.params = params
        self.input_factory = input_factory
        self.inputs = asyncio.Queue()
        self.turns = []
        self.turn_logprobs = []
        self.closed = False
        self.output_events = 0
        self.last_output_tokens = 0
        self.changed = asyncio.Condition()
        self.task = asyncio.create_task(self.collect())

    async def input_stream(self):
        while (tokens := await self.inputs.get()) is not None:
            yield self.input_factory(prompt={"prompt_token_ids": tokens})

    async def collect(self):
        try:
            async for output in self.engine.generate(
                prompt=self.input_stream(),
                sampling_params=self.params,
                request_id=self.name,
            ):
                completion = output.outputs[0]
                self.output_events += 1
                self.last_output_tokens = len(completion.token_ids)
                if completion.finish_reason is not None:
                    async with self.changed:
                        self.turns.append(list(completion.token_ids))
                        self.turn_logprobs.append(
                            [
                                {
                                    str(token): value.logprob
                                    for token, value in row.items()
                                }
                                for row in completion.logprobs or []
                            ]
                        )
                        self.changed.notify_all()
        finally:
            async with self.changed:
                self.closed = True
                self.changed.notify_all()

    async def send(self, tokens):
        if self.closed:
            await self.task
            raise RuntimeError("cannot continue a closed session")
        await self.inputs.put(list(tokens))

    async def wait_turn(self, count, timeout=120):
        async def wait():
            async with self.changed:
                await self.changed.wait_for(
                    lambda: len(self.turns) >= count or self.closed
                )
            if len(self.turns) < count:
                await self.task
                raise RuntimeError("session closed before the requested turn")
            return self.turns[count - 1]

        return await asyncio.wait_for(wait(), timeout=timeout)

    def internal_id(self):
        ids = self.engine.output_processor.external_req_ids.get(self.name, [])
        if len(ids) != 1:
            raise RuntimeError("session does not have one retained internal identity")
        return ids[0]

    async def finish(self):
        await self.inputs.put(None)
        await asyncio.wait_for(self.task, timeout=120)


async def run_stream_screen(engine, args, prompt_ids, params, report, save):
    from vllm.engine.protocol import StreamingInput

    client = engine.engine_core.client_index
    sessions = []

    async def utility(operation, request_id="", generation=None, client_index=client):
        return await engine.engine_core.call_utility_async(
            "flash_session_swap", operation, request_id, client_index, generation
        )

    def make(name):
        session = StreamingSession(engine, params, name, StreamingInput)
        sessions.append(session)
        return session

    async def checkpoint(phase):
        report["phase"] = phase
        report["latest_stats"] = await asyncio.wait_for(utility("stats"), 10)
        save()
        print(f"Streaming screen: {phase}", flush=True)

    async def wait_turn(session, count):
        try:
            return await session.wait_turn(count)
        except TimeoutError:
            report["timeout_session"] = dict(
                name=session.name,
                turn=count,
                completed_turns=len(session.turns),
                output_events=session.output_events,
                last_output_tokens=session.last_output_tokens,
            )
            try:
                await checkpoint("generation_timeout")
            except Exception as exc:
                report["timeout_stats_error"] = type(exc).__name__
            save()
            raise

    continuation = engine.get_tokenizer().encode(
        " Continue with another worked example."
    )
    capacity = engine.vllm_config.scheduler_config.max_num_seqs
    count = capacity + 1
    report.update(
        scope="native idle-session LRU/continuation with more sessions than slots",
        slot_capacity=capacity,
        retained_session_target=count,
        continuation_token_ids=continuation,
        references=[],
        sessions=[],
        baseline_repeatable=True,
    )
    save()
    try:
        if args.trace_prefill:
            report["prefill_trace_install"] = await engine.collective_rpc(
                "install_session_prefill_trace",
                args=(len(prompt_ids), args.reference_runs),
            )
            save()
        await checkpoint("before_references")
        for index in range(args.reference_runs):
            session = make(f"stream-reference-{index}")
            await session.send(prompt_ids)
            await wait_turn(session, 1)
            await session.send(continuation)
            await wait_turn(session, 2)
            await session.finish()
            row = dict(turns=session.turns, top_logprobs=session.turn_logprobs)
            if report["references"]:
                row["comparisons"] = [
                    compare_tokens(left, right)
                    for left, right in zip(
                        report["references"][0]["turns"], session.turns, strict=True
                    )
                ]
                report["baseline_repeatable"] &= all(
                    c["equal"] for c in row["comparisons"]
                )
            report["references"].append(row)
            await checkpoint(f"reference_{index}_closed")
            row["stats_after_close"] = report["latest_stats"]
            save()
            closed_state = report["latest_stats"]["scheduler"]
            if closed_state["retained_requests"] or closed_state["idle_hot"]:
                raise RuntimeError("closed reference left retained scheduler state")
        if args.trace_prefill:
            report["prefill_trace"] = await engine.collective_rpc(
                "collect_session_prefill_trace"
            )
            report["prefill_trace_complete"] = all(
                len(trace["layers"].get(name, [])) == args.reference_runs
                and all("outputs" in row for row in trace["layers"][name])
                for installed, trace in zip(
                    report["prefill_trace_install"],
                    report["prefill_trace"],
                    strict=True,
                )
                for name in installed["layers"]
            )
            save()
            if not report["prefill_trace_complete"]:
                raise RuntimeError("prefill diagnostic did not cover all references")
        if not report["baseline_repeatable"] and not args.diagnose:
            raise RuntimeError("streaming references are not repeatable")
        report["stats_before"] = await utility("stats")
        if report["stats_before"]["cold_idle_ttl_seconds"] != 3600:
            raise RuntimeError("expected the user-selected 60-minute TTL")
        retained = []
        for index in range(count):
            session = make(f"retained-{index}")
            retained.append(session)
            await checkpoint(f"admit_{index}")
            await session.send(prompt_ids)
            await wait_turn(session, 1)
            row = dict(
                name=session.name,
                request_id=session.internal_id(),
                turns=[session.turns[0]],
                top_logprobs=[session.turn_logprobs[0]],
            )
            report["sessions"].append(row)
            report["stats_during_admission"] = await utility("stats")
            save()
        admissions = [await utility("lookup", s.internal_id()) for s in retained]
        report["admission_lookups"] = admissions
        report["stats_after_admission"] = await utility("stats")
        save()
        if not any(row.get("cache") == "cold_hit" for row in admissions):
            raise RuntimeError("more-than-slot admission did not retain a cold session")
        if any(row.get("cache") not in ("cold_hit", "hot_hit") for row in admissions):
            raise RuntimeError("an admitted session was lost or left unavailable")
        if report["stats_after_admission"]["counters"].get("lru_spills", 0) < 1:
            raise RuntimeError("automatic LRU did not release hot capacity")
        report["exact_token_match"] = True
        for session, row in zip(retained, report["sessions"], strict=True):
            row["before_resume"] = await utility("lookup", session.internal_id())
            row["wrong_client_lookup"] = await utility(
                "lookup", session.internal_id(), client_index=client + 1
            )
            if row["wrong_client_lookup"].get("cache") != "miss":
                raise RuntimeError("another frontend identity could see a retained hit")
            await session.send(continuation)
            await wait_turn(session, 2)
            row["turns"] = session.turns.copy()
            row["top_logprobs"] = session.turn_logprobs.copy()
            row["comparisons"] = [
                compare_tokens(left, right)
                for left, right in zip(
                    report["references"][0]["turns"], session.turns, strict=True
                )
            ]
            report["exact_token_match"] &= all(c["equal"] for c in row["comparisons"])
            row["after_resume"] = await utility("lookup", session.internal_id())
            report["stats_after_resume"] = await utility("stats")
            save()
        if report["stats_after_resume"]["counters"].get("restores", 0) < 1:
            raise RuntimeError("queued input did not automatically restore cold state")
        for session in retained:
            await session.finish()
        await engine.pause_generation(mode="keep", clear_cache=False)
        report["stats_after_close"] = stats = await utility("stats")
        report["closed_lookups"] = [
            await utility("lookup", row["request_id"]) for row in report["sessions"]
        ]
        clean = cleanup_complete(stats, report["closed_lookups"])
        report["cleanup_passed"] = clean
        report["screen_passed"] = bool(
            report["baseline_repeatable"] and report["exact_token_match"] and clean
        )
        report["complete"] = True
        save()
        await engine.resume_generation()
        if not report["screen_passed"]:
            raise AssertionError("streaming session qualification failed")
    finally:
        for session in sessions:
            if not session.task.done():
                session.task.cancel()
        await asyncio.gather(*(s.task for s in sessions), return_exceptions=True)
