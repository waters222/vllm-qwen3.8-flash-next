# TP4 multi-image inputs with MTP4

On September 26, 2026, the existing modified Flash-Next runtime completed a
66-batch image/text benchmark on four RTX 3090 GPUs. Vision and MTP4 ran together
at C2 and C4, including four images per request, 19,043 total prompt tokens per
request, and 1,024 output tokens. No additional image-specific engine patch was
needed. This is a result for this fork and its pinned runtime, not a claim about
an unmodified upstream installation.

## Configuration and method

The baseline retains TP4, MTP with four speculative tokens, host-RAM QSA KV,
native 944-token cache blocks, a 2,048-token prefill budget, and four request slots.
The vision encoder uses the default weight-sharded TP mode. Image support was
enabled in the benchmark with these engine argument overrides:

```json
{
  "language_model_only": false,
  "limit_mm_per_prompt": {"image": 4, "video": 0},
  "mm_processor_kwargs": {
    "size": {"longest_edge": 1048576, "shortest_edge": 65536}
  },
  "mm_processor_cache_gb": 0
}
```

The [complete benchmark engine arguments](image-mtp-engine-args.json) retain
the measured settings with the diagnostic worker-extension import removed.
They complement the existing fork-specific environment gates; they are not a
standalone upstream or production deployment preset. Exact measured engine
sources and the native library are pinned in
[the runtime source manifest](image-mtp-runtime-sources.json).

The processor's `size` fields express pixel-area limits here. The initial raw
Hugging Face probe ignored a `max_pixels` override; the corrected explicit `size`
configuration was checked against actual image grids before GPU measurement.
Four public diagram fixtures produce 980, 999, 918 and 989 visual tokens each.
Two-image requests contain 1,979 visual tokens; four-image requests contain 3,886.

Each shape had one warmup and three measured repeats. Separate one-output-token
runs measure prefill; generation runs force 1,024 tokens with temperature zero,
seed 17 and EOS ignored. Prefix reads and processor caching were disabled.
Distinct image UUIDs prevent encoder-output reuse. Recorded encoder patch counts
confirm actual encoding for every request on every rank.

This is an in-process AsyncLLM benchmark with decoded PIL images. It includes
processor and encoder work, but excludes network transport, image upload and
file/base64 decoding. Encoder CUDA-event hooks are included in the measured
runtime. The nominal 16k text fixture retokenizes to the actual counts below.

## Complete measured comparison

All rates are aggregate across concurrent requests and are medians of three
repeats. TTFT is time until the last request's first token in the prefill batch.
Steady TG starts after all requests have their first token. Overlap TG starts
after the first request's first token and includes prefill of later requests.
First speculative bursts are excluded from the corresponding TG windows.

| C | Input per request | Prompt tokens | Prefill tok/s | TTFT s | Steady TG tok/s | Overlap TG tok/s | Draft acceptance |
|---|---|---:|---:|---:|---:|---:|---:|
| 2 | Text control | 15,149 | 3,486.9 | 8.69 | 124.4 | 98.4 | 26.5% |
| 2 | 2 images + short text | 2,008 | 2,198.2 | 1.83 | 224.3 | 202.1 | 56.3% |
| 2 | 4 images + short text | 3,919 | 1,878.2 | 4.17 | 211.6 | 185.5 | 55.2% |
| 2 | 4 images + long text | 19,043 | 3,036.4 | 12.54 | 214.8 | 136.1 | 56.3% |
| 4 | Text control | 15,149 | 3,452.6 | 17.55 | 165.3 | 109.7 | 29.5% |
| 4 | 2 images + short text | 2,008 | 1,857.6 | 4.32 | 320.3 | 249.0 | 56.7% |
| 4 | 4 images + short text | 3,919 | 1,708.6 | 9.17 | 299.1 | 214.5 | 54.0% |
| 4 | 4 images + long text | 19,043 | 3,049.8 | 24.98 | 286.9 | 119.3 | 55.1% |

These are different prompts from the historical performance tables. Different
MTP acceptance rates also affect TG substantially; these rows do not establish
a historical regression or an image-processing speedup. Machine-readable rates,
repeat ranges, memory observations and source evidence hashes are provided in
[the benchmark summary](image-mtp-results.json).

For C4/four-image short-text prefill, CPU processing totaled 1.059 seconds per
batch, the longest-rank encoder CUDA spans totaled 1.921 seconds, and batch TTFT
was 9.175 seconds. Mixed C4 values were 1.534, 1.999 and 24.976 seconds. These
spans can overlap, and encoder events can include collective/scheduling delays.
They must not be subtracted as disjoint stages. Faster image processing alone
will not eliminate the language-model and scheduling portions of prefill.

## Capacity and qualification limits

All 66 batches completed, with observed peak concurrency matching C2/C4. Measured
repeats had zero preemptions, allocator retries or unrecovered allocation OOMs.
Initial capacity/warmup had 16 allocation retries across ranks, including failed
initial allocations followed by successful cache reclamation. These warnings
remain part of the evidence.

CUDA-unreserved memory reached 47 MiB per GPU, with roughly 600 MiB of unused
PyTorch reservation available for reuse. Vision weights occupy about 220.7 MiB
per rank. Replicating the full encoder would add about 660 MiB per rank and needs
a separate capacity experiment. The existing weight-sharded vision encoder and
MTP already fit this tested workload.

A CPU processor sweep at 1/2/4/8/12 threads retained exact output hashes. The best
component latency was at 8–12 threads; reducing threads did not help. No new
engine optimization was adopted from these experiments. A second serving
quartet remained unchanged; five-second GPU samples and request counters showed
no serving activity during the bracketed measurements, but do not prove exclusive
host-memory bandwidth.

This does not qualify full 32k image contexts, additional images/resolutions,
image-answer accuracy, or long-duration stability. The separate numerical/cache
screen still fails. A reference-order control matched hot/cold prefix and GDN
entry-state digests across all four ranks and both fixtures, while numerical
outputs still differed. That control is not a generalized cache qualification.
The separately provisioned test API remains text-only; this benchmark did not
enable its image configuration. AI assistance was used for this work.
