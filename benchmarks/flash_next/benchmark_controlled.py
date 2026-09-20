#!/usr/bin/env python3
"""Fixed-input streaming benchmark for the isolated, loopback Flash trial."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import threading
import time
import urllib.request


def request(slot, barrier, args):
    prompt = (f'Request {slot}. ' + ' a' * args.context_tokens
              + '\nWrite an extended numbered list of distinct mathematical facts. '
                'Continue until stopped.')
    payload = dict(model='Qwen3.8-Flash-Next',
                   messages=[dict(role='user', content=prompt)],
                   temperature=0, seed=17, max_tokens=args.output_tokens,
                   ignore_eos=True, stream=True,
                   stream_options={'include_usage': True},
                   chat_template_kwargs={'enable_thinking': False})
    if args.allowed_token_id is not None:
        payload['allowed_token_ids'] = [args.allowed_token_id]
    req = urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',
                                 data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    barrier.wait()
    if args.stagger_ms:
        time.sleep(slot * args.stagger_ms / 1000)
    start = time.perf_counter()
    arrivals, content, usage = [], [], None
    with urllib.request.urlopen(req, timeout=getattr(args, 'request_timeout_s', 900)) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            raw = line[6:].strip()
            if raw == b'[DONE]':
                break
            obj = json.loads(raw)
            if obj.get('usage'):
                usage = obj['usage']
            for choice in obj.get('choices', []):
                chunk = choice.get('delta', {}).get('content')
                if chunk:
                    arrivals.append(time.perf_counter() - start)
                    content.append(chunk)
    elapsed = time.perf_counter() - start
    if not usage or usage['completion_tokens'] != args.output_tokens:
        raise RuntimeError(f'Unexpected output count: {usage}')
    if not arrivals:
        raise RuntimeError('No text streamed')
    gaps = [b-a for a, b in zip(arrivals, arrivals[1:])]
    return dict(slot=slot, elapsed_s=elapsed, ttft_s=arrivals[0],
                tpot_s=(arrivals[-1]-arrivals[0]) / (usage['completion_tokens']-1),
                chunk_gaps_s=gaps, usage=usage, text=''.join(content))


def batch(c, args):
    barrier = threading.Barrier(c+1)
    with ThreadPoolExecutor(max_workers=c) as pool:
        futures = [pool.submit(request, i, barrier, args) for i in range(c)]
        start = time.perf_counter()
        barrier.wait()
        results = [f.result() for f in futures]
    elapsed = time.perf_counter()-start
    return dict(concurrency=c, elapsed_s=elapsed,
                tokens_per_s=sum(r['usage']['completion_tokens'] for r in results)/elapsed,
                requests=results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--runs', type=int, default=10)
    p.add_argument('--warmups', type=int, default=2)
    p.add_argument('--output-tokens', type=int, default=512)
    p.add_argument('--context-tokens', type=int, default=0)
    p.add_argument('--stagger-ms', type=float, default=0,
                   help='Diagnostic request-arrival spacing; included in batch wall time')
    p.add_argument('--concurrency', nargs='+', type=int, default=[1,2,3,4])
    p.add_argument('--allowed-token-id', type=int,
                   help='Supplementary fixed-output diagnostic, not representative generation')
    a = p.parse_args()
    if a.output_tokens < 2 or a.runs < 1 or min(a.concurrency) < 1:
        p.error('Invalid token/run/concurrency count')
    if a.allowed_token_id is not None and a.allowed_token_id < 0:
        p.error('--allowed-token-id must be nonnegative')
    out = dict(label=a.label, settings=vars(a)|{'output': str(a.output)}, batches=[],
               note='TTFT/chunk timing is client-observed; SSE chunks need not equal tokens.')
    for _ in range(a.warmups):
        for c in a.concurrency:
            batch(c, a)
    for r in range(a.runs):
        order = a.concurrency[r % len(a.concurrency):]+a.concurrency[:r % len(a.concurrency)]
        for c in order:
            result = batch(c, a)
            result['round'] = r
            out['batches'].append(result)
            a.output.write_text(json.dumps(out, indent=2)+'\n')
            print(r, c, round(result['tokens_per_s'],2), flush=True)
    print('Medians', {c: statistics.median(b['tokens_per_s'] for b in out['batches']
                                         if b['concurrency']==c) for c in a.concurrency}, flush=True)


if __name__ == '__main__':
    main()
