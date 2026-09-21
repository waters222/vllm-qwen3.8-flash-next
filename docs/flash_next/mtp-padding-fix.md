# MTP padding correctness fix

The original source snapshot's optional SM86 M1--4/top10 expert-layout path
does not handle negative padding IDs correctly. An all-padding M4 batch
produces expert0 and negative token offsets. Marlin down projection then reads
outside its input. This was reproduced independently of checkpoint weights.

The correction normalizes invalid expert IDs before packing sort keys. It also
passes route IDs through the custom and native no-EP combine paths and masks
loads from skipped-route scratch. Multiplying invalid scratch by zero is not
sufficient because that scratch can contain NaNs. The legacy EP specialization
is selected only when an expert map exists.

Valid-route weight arithmetic, checkpoint files, and quantization are unchanged.
The `source_manifest.json` retains the original image hashes and records these
two revised source hashes separately. They are not present in the historical
image or source-snapshot tag.

Validation so far:

- The old kernel reproduces the captured negative offsets with all-`-1` IDs.
- The corrected operators pass 40 eager/changed-input CUDA-graph cases across
  M1/2/3/4/5/8/12/16, including native fallback sizes, empty/mixed/valid/repeated
  routes, boundary experts, and NaN-poisoned invalid output scratch.
- Alignment agrees with an independent CPU mapping; combine agrees exactly
  with native masked summation. Extreme invalid int32 alignment passes too.
- Regression coverage is in `tests/kernels/moe/test_moe_align_block_size.py`.
- Full-model BF16 MTP3 completed graph startup with TP2/PP2/no EP and normal
  custom all-reduce. All 12 saved behavior outputs match the no-MTP fixture
  exactly. The C1--4 submitted-request matrix completed with valid token and
  acceptance accounting. Math C1 median decode was 163.33 tok/s versus the
  earlier matched-configuration control's 91.98 tok/s; draft acceptance was
  74.16%. This compares with a saved control, not a newly patched control.
- C3--4 encountered queueing: only two requests ran at once with GPU cache
  usage around 90%. Those results do not validate true four-way decode.
- All 21 CPU source/build-contract tests and targeted test/exporter Ruff
  checks pass. Existing imported engine formatting debt remains unchanged.

The existing no-MTP serving image was not changed. This correction was developed
with AI assistance and requires human review before any upstream contribution.

Follow-up depth screening also completed BF16 MTP1/2/4 at the same 550 MB
per-rank GPU-cache budget. C1 math medians were 124.89/149.61/109.46 tok/s,
versus MTP3's 163.33. MTP2/4 match the saved behavior outputs exactly; MTP1
has one reviewed JSON-format-only difference with identical parsed content.
MTP1 admits four short requests, but C2 timing varies from 116.84 to 194.14
aggregate tok/s at about 90% acceptance. Concurrency stability is not qualified.
All development test containers were stopped and retained; serving is unchanged.
The CPU preservation/contract suite now has 23 passing tests, including the
separate optional QSA-page-cap candidate, which is not full-model qualified.
