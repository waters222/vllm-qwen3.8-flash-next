# Experimental MTP1 pipeline admission pairs

Default off. Requires both `VLLM_FLASH_PP_BALANCE=1` and
`VLLM_FLASH_PP_MTP_PAIRS=1`. Restricted to standard Qwen4Exp V2 async TP2/PP2,
no EP, MTP depth one, and at most four requests. Existing no-MTP balancing
is unchanged when the pairs flag is absent.

An eight-batch Q8-expert MTP1 screen associated same-phase C2 with 190--209
aggregate tok/s and opposite-phase C2 with 103--116 tok/s, at similar acceptance.
This is scheduler evidence, not proof of device overlap or causation.

The pilot fills a two-request cohort before opening the other PP phase.
It may defer a fresh short prompt once, without changing established requests'
decode eligibility, token state, KV state, or worker ring. Chunked/long prefills,
resumes, preemption, and ambiguous state bypass the policy. A lone request is
not delayed. Cancellation and identity reuse retain existing cleanup behavior.

CPU helper tests cover cold C1--4 admission, staggered arrivals, one-deferral
bounds, unchanged request state, topology gates, and conservative bypasses.
They do not prove full scheduler/worker-ring correctness. GPU qualification
must check behavior fixtures, observed phases, arrival offsets, C1--4 repeated
timings, acceptance counters, and long-context fallback. Not production ready.

The first GPU intervention passed all12 fixture outputs and put all eight C2
arrival cases in the same phase at192--198 aggregate tok/s. However, the
three-repeat math C1--4 screen remains mixed: C1 decode117.23, C2 aggregate
193.92, C3 aggregate143.30 (reference176.14), and C4 repeats172.65/268.50/301.20.
The policy is not qualified as a general concurrency improvement or promoted.

Rollback: disable both flags and restore the historical balance helper.
Do not apply this policy to MTP2+: its phase measurements did not show the
same consistent phase advantage.
