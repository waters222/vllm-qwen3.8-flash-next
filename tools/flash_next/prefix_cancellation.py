"""Native active-abort controller for C2/C4 diagnostic request groups."""

import copy


class ActiveCancellation:
    """Abort once all participants are live; optionally audit every survivor."""

    def __init__(self, engine, victim, survivors, threshold=8, audit=False):
        survivors = (survivors,) if isinstance(survivors, str) else tuple(survivors)
        names = (victim, *survivors)
        if (
            len(survivors) not in (1, 3)
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
            or type(threshold) is not int
            or threshold <= 0
        ):
            raise ValueError(
                "Cancellation requires distinct C2/C4 names and a threshold"
            )
        self.engine = engine
        self.victim = victim
        self.survivors = survivors
        self.threshold = threshold
        self.audit = audit
        self.progress = {}
        self.at_abort = None
        self.acknowledged = False
        self.cache_audit = None

    async def observe(self, name, output):
        names = (self.victim, *self.survivors)
        if name not in names:
            raise ValueError("Unexpected cancellation-fixture request")
        self.progress[name] = dict(
            tokens=len(output.outputs[0].token_ids), finished=output.finished
        )
        if self.at_abort is not None or not all(
            self.progress.get(request, {}).get("tokens", 0) >= self.threshold
            and not self.progress[request]["finished"]
            for request in names
        ):
            return
        self.at_abort = copy.deepcopy(self.progress)
        if not self.audit:
            await self.engine.abort(self.victim)
            self.acknowledged = True
            return
        await self.engine.pause_generation(mode="keep", clear_cache=False)
        try:
            if not await self.engine.is_paused():
                raise RuntimeError("Cache audit requires native scheduling pause")
            before = await self._snapshot_survivors()
            await self.engine.abort(self.victim)
            self.acknowledged = True
            after = await self._snapshot_survivors()
            audits = {
                survivor: dict(before=before[survivor], after=after[survivor])
                for survivor in self.survivors
            }
            self.cache_audit = (
                audits[self.survivors[0]]
                if len(self.survivors) == 1
                else dict(survivors=audits)
            )
        finally:
            await self.engine.resume_generation()

    async def _snapshot_survivors(self):
        snapshots = {}
        for survivor in self.survivors:
            snapshots[survivor] = await self.engine.collective_rpc(
                "snapshot_active_cache_audit", args=(survivor,)
            )
        return snapshots
