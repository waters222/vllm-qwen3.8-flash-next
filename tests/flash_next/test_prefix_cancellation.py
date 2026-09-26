"""Controller-only tests; no claim of runtime cache or GPU race coverage."""

import asyncio
import unittest
from types import SimpleNamespace as NS

from prefix_cancellation import ActiveCancellation


def output(tokens=8, finished=False):
    return NS(outputs=[NS(token_ids=[1] * tokens)], finished=finished)


class CancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_c4_waits_for_every_live_survivor_and_aborts_once(self):
        aborted = []

        async def abort(name):
            aborted.append(name)
            await asyncio.sleep(0)

        controller = ActiveCancellation(NS(abort=abort), "v", ("a", "b", "c"))
        for name in ("v", "a", "b"):
            await controller.observe(name, output())
        await controller.observe("c", output(7))
        self.assertEqual(aborted, [])
        await asyncio.gather(
            controller.observe("c", output()), controller.observe("b", output(9))
        )
        self.assertEqual(aborted, ["v"])
        self.assertEqual(set(controller.at_abort), {"v", "a", "b", "c"})
        self.assertTrue(controller.acknowledged)
        self.assertIsNone(controller.cache_audit)
        for finished in ("v", "a", "b", "c"):
            controller = ActiveCancellation(NS(abort=abort), "v", ("a", "b", "c"))
            for name in ("v", "a", "b", "c"):
                await controller.observe(name, output(finished=name == finished))
            self.assertIsNone(controller.at_abort)
        self.assertEqual(aborted, ["v"])

    async def test_audit_covers_all_survivors_and_resumes_after_failure(self):
        for survivors in (("a",), ("a", "b", "c")):
            for fail_at in (None, len(survivors), 2 * len(survivors)):
                with self.subTest(survivors=survivors, fail_at=fail_at):
                    calls = []
                    snapshots = 0

                    class Engine:
                        async def pause_generation(self, **kwargs):
                            calls.append(("pause", kwargs))

                        async def is_paused(self):
                            return True

                        async def collective_rpc(self, method, args):
                            nonlocal snapshots
                            snapshots += 1
                            calls.append(("snapshot", args[0]))
                            if snapshots == fail_at:
                                raise RuntimeError("snapshot failed")
                            return [{"request": args[0], "rank": i} for i in range(4)]

                        async def abort(self, name):
                            calls.append(("abort", name))

                        async def resume_generation(self):
                            calls.append(("resume",))

                    controller = ActiveCancellation(
                        Engine(), "v", survivors, audit=True
                    )
                    for name in ("v", *survivors[:-1]):
                        await controller.observe(name, output())
                    if fail_at:
                        with self.assertRaisesRegex(RuntimeError, "snapshot failed"):
                            await controller.observe(survivors[-1], output())
                        self.assertIsNone(controller.cache_audit)
                        self.assertEqual(
                            controller.acknowledged, fail_at > len(survivors)
                        )
                    else:
                        await controller.observe(survivors[-1], output())
                        expected = [("pause", {"mode": "keep", "clear_cache": False})]
                        expected += [("snapshot", name) for name in survivors]
                        expected += [("abort", "v")]
                        expected += [("snapshot", name) for name in survivors]
                        expected += [("resume",)]
                        self.assertEqual(calls, expected)
                        audits = (
                            controller.cache_audit["survivors"]
                            if len(survivors) > 1
                            else {"a": controller.cache_audit}
                        )
                        self.assertEqual(set(audits), set(survivors))
                        for audit in audits.values():
                            self.assertEqual(audit["before"], audit["after"])
                    self.assertEqual(calls[-1], ("resume",))


if __name__ == "__main__":
    unittest.main()
