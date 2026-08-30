# First-Use Warmup Validation Receipt

## Verdict

The first-use warmup change is qualified on the production TP2/BF16-KV
configuration.

- Stochastic sampling is a causal `PASS`. The parent paid 96.591 seconds on
  its first external stochastic request. The candidate paid 99.296 seconds
  during boot, then served the same external request in 153.906 ms without
  changing any sampling-cache file.
- The shallow eager continuation is a threshold `PASS`, with an important
  qualification. The candidate's first external continuation was 146.251 ms,
  below the 172.8 ms acceptance limit. The exact post-merge parent was already
  steady at 146.299 ms, so this warmup cannot honestly be credited with a
  latency improvement on the current source level. It preserves explicit
  coverage of the characterized schedule.
- The complete external streaming-session kit passed `7/7` under TP2.
- Both focused CPU suites passed, `10/10` and `4/4`.

Branch `tanmay/streaming-session-token-api` has exactly one implementation
commit over parent `8d609cbe196bf339377f97913605552e99fbb6fb`. The GPU
validation ran executable commit
`a5a4fa296fe9783f0f072489530b85c8ee473c42`, tree
`f47b0a503a20588c451a1d25a808070a1d7484a5`. The final commit contains the
same executable files plus this receipt.

## Implementation

The canonical Gemma-4 streaming-session warmup list is:

```text
streaming_session_small_extend,
streaming_session_shallow_eager_extend,
stochastic_sampling_first_use
```

`streaming_session_shallow_eager_extend` reproduces the D-t accepted
continuation schedule, not only its rounded scheduler shape:

1. Open a streaming session.
2. Append 96 fixed raw tokens at `expected_tip=0` and generate four greedy
   tokens.
3. Append the fixed 16-token `context[112:128]` continuation at
   `expected_tip=100` and generate four greedy tokens.
4. Drain both requests to terminal responses and close the session on every
   exit path.

It uses a cache namespace distinct from the existing 40,960-token plus
64-token deep warmup. The scheduler reports the accepted shallow continuation
as a page-rounded 64-token eager row, while the true accepted delta is 16
tokens.

`stochastic_sampling_first_use` issues one raw-token request with the exact
effective production sampling values:

```text
temperature=0.4
top_k=64
top_p=0.95
min_p=0.0
max_new_tokens=1
```

This selects FlashInfer's joint top-k/top-p sampling module. Every registered
warmup now logs its own monotonic elapsed time.

Executable file SHA-256 values:

| File | SHA-256 |
|---|---|
| `python/sglang/srt/entrypoints/warmup.py` | `6d9e143ba966d7d6bb06e757bbcb1490be23a5e09f007e5bbafb93ecc32c8e60` |
| `python/sglang/test/server_fixtures/gemma4_streaming_session_fixture.py` | `7ebc61aedfd7362aaefd9856aeb7e696c5de50074540702039c326626ae8524f` |
| `test/registered/unit/entrypoints/test_streaming_session_warmup.py` | `d388d14fe88683a7a6f01abff3f62c5c6ca72ea94fc623ba8bcbfd6a8440be72` |
| `test/registered/unit/entrypoints/test_gemma4_streaming_session_fixture.py` | `84c32cb8e896eccae6eefee3dd90fe1a65763aee9ccd20901b9319789f7b61c3` |

## Focused CPU qualification

The exact executable tree passed:

```text
test_streaming_session_warmup.py
Ran 10 tests in 0.032s
OK

test_gemma4_streaming_session_fixture.py
Ran 4 tests in 0.000s
OK
```

The behavior suite covers warmup order, exact shallow token schedule and
expected tips, stochastic parameter values, full terminal draining, distinct
cache namespaces, close-on-failure at each request position, and
fail-closed rejection outside unified serving. The fixture suite pins the
expanded canonical list. Live TP2 behavior is proven separately below rather
than inferred from a hand-built fixture.

## Live validation configuration

All GPU work ran on `gemma-dev-1` physical GPUs 2/3:

- `GPU-d2a77370-eb5b-af94-fc2c-21cb7e2f271c`
- `GPU-de79342b-f765-3388-1e03-41f12396df79`

The server used the production ModelOpt NVFP4 target, BF16 KV cache, TP2,
DFlash, the admitted GeGLU fusion, QKV fusion disabled, TensorRT-LLM MHA,
128K context, the eight production prefill graph buckets, metrics, and
streaming sessions. The environment was the sealed Section 8 virtual
environment with torch 2.13.0, Triton 3.7.1, and flashinfer-python 0.6.17.

Both arms used genuinely separate empty roots for all compiled-kernel caches:
`SGLANG_CACHE_DIR`, `FLASHINFER_WORKSPACE_BASE`, `CUDA_CACHE_PATH`,
`TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, and
`SGLANG_JIT_CACHE_DIR`. Baseline HTTP/NCCL ports were 32396/34396;
candidate ports were 32397/34397.

Evidence root:
`/workspace/warmup-validation-20260830`.

- Launcher SHA-256:
  `d124ce106ee75c1c1bcb6ed25471aff1a77e031bdd918bec10260d2f0593c558`.
- Probe SHA-256:
  `9e4a774177e2dac7f5111365b4bb94be7634c868b884d9868a2a664d64faf6af`.

The preserved first baseline attempt under `runs/baseline` failed before
model or GPU initialization because it invoked system Python with
sglang-kernel 0.4.5 while the merged source requires 0.4.6.post1. It issued
zero candidate requests. The corrected lineages are `runs/baseline-r2` and
`runs/candidate-r2`, both using the sealed Section 8 interpreter and fresh
evidence/cache roots.

## First-use results

| Path | Parent first external request | Candidate first external request | Acceptance | Verdict |
|---|---:|---:|---:|---|
| Shallow eager continuation | 146.299 ms | 146.251 ms | <= 172.8 ms | Pass, no measurable parent-to-candidate delta |
| Stochastic sampling | 96,591.229 ms | 153.906 ms | No compile spike | Pass |

Both shallow probes had stable tip 100, 100 cached tokens, 116 prompt tokens,
four completion tokens, final tip 120, and output token log
`[6097, 12822, 83691, 57243]`.

The shallow result corrects the stale pre-merge expectation. The sealed D-t
lineage at `66c47a3886` measured 1,166.032 ms on its first shallow row and
2,267.87 ms in the existing deep eager warmup. On the exact post-merge parent,
the deep eager warmup was already 165.75 ms and the first external shallow row
was 146.299 ms. Therefore the one-second in-process initialization hazard is
not present at this source/runtime level. This receipt does not attribute
which merge or environment change removed it, and it does not claim a
candidate-side speedup that the current control cannot show.

The candidate's boot-time brackets were:

| Warmup | Elapsed |
|---|---:|
| `streaming_session_small_extend` | 7.568 s |
| `streaming_session_shallow_eager_extend` | 0.302 s |
| `stochastic_sampling_first_use` | 99.296 s |

The stochastic warmup began at 10:37:17 UTC and ended at 10:38:56 UTC.
FlashInfer's Ninja receipt places binding compilation at 18.858 seconds,
`renorm.cu` at 35.775 seconds, `sampling.cu` at 89.520 seconds, and the
linked `sampling.so` at 89.616 seconds, all inside that bracket and before
application readiness.

The baseline sampling subtree was absent before the first external request
and contained seven files afterward. The candidate subtree contained all
seven files before readiness; hashes, sizes, and mtimes were byte-for-byte
unchanged by the external request. Total cold-start readiness was 673 seconds
for the baseline and 758 seconds for the candidate. Those totals include
variable full FP4 autotuning and graph capture; the 99.296-second per-warmup
bracket is the direct boot cost moved by this change.

## TP2 streaming-session qualification

The full external production kit ran once, fail-fast, against the live
candidate:

```text
Ran 7 tests in 414.887s
OK
first_real_small_extend_seconds=0.156104
```

The seven methods cover the 23 ratified probes: first-real-extend latency,
schedule-matched truncate equivalence, deep abort with exact-slot
preservation, idempotency conflict and recovery, SWA pin accounting, mutation
edge cases, and truncate/abort leak closure.

## Deployment launcher disposition

- The in-tree Gemma-4 fixture now carries the expanded canonical three-warmup
  list.
- The active dev-1 launcher
  `/workspace/services/gemma4-tp1x2-streaming-session-20260829/launch.sh`
  remains pinned to source `5acd42c6747fe8fef1710c8b2de2ffe5f3669894`
  and the old single warmup. It was neither edited nor restarted. Its source
  identity guard correctly prevents a restart until an explicit revision
  repin.
- Gate-0, D-t, Section 8, and
  `/workspace/streaming-session-probe-20260828` launchers are immutable
  historical evidence and were not modified.
- Future TP2/BF16 launchers derived from the Section 8 production template
  must carry the expanded canonical list.

## Lifecycle and hardware closure

Both corrected lineages terminated with identity-checked `SIGTERM`; no
`SIGKILL` was used. Candidate ports 32397/34397 and its tmux identity are
gone. After clocks settled, GPUs 2/3 were empty at 0 MiB and 120 MHz. The
protected production processes and memory footprints on GPUs 0/1 were
byte-identical before and after. The zombie count remained zero.

All eight GPUs reported zero volatile corrected and uncorrected ECC errors,
zero correctable and uncorrectable remapped rows, no pending remap or remap
failure, and recovery action `None`. The final 2,000 kernel-log lines
contained no XID, SXid, fatal NVLink, fallen-GPU, or uncorrectable-error
signature.

`gemma-dev-2` and `gemma-dev-3` were not contacted.

## Evidence hashes

| Artifact | SHA-256 |
|---|---|
| Baseline first-use JSON | `2556b2d045b5b679539941e2bbb6c319b7b1447721ee7c71ac634f4e312b6b11` |
| Baseline server log | `1fc01884869ad8ac9248fbced28f77853f9b81c8187296a8f568e5aac5944657` |
| Candidate first-use JSON | `1133529b5a5285614351d7b7f2bda2668c2820313160c7449daa84f479c0097b` |
| Candidate TP2 suite log | `0364911d0885cd53f591ef910bda112dc396ae0cb436f98ee4e28812415d05d5` |
| Candidate warmup CPU-test log | `7a8878bee5faee3c6644436b810dc5bc340964cc6ef0cadcb880e7ff05ff7432` |
| Candidate fixture CPU-test log | `e6ec182765cd965f2837d8eeaef5a64af48a104f245cd1a0b7281559194c57c1` |
| Candidate server log | `f555761ed618e1f2d416b4f62fa71dbe8f378b0da55039aa5581ef8ce597b1d2` |
| Final NVIDIA health dump | `e8c292f4c6805c3c3a54f488757fe0d5c3cd07bab2dcff6db79878616d815d09` |
| Settled GPU inventory | `7a1145bf82b51b1fb5df9a2c8a6c1e23b82c78ec541f82481194863efeef8509` |
| Final kernel-log tail | `a973a99e8d27a131255c28f7ee094d77c51d336fd1e0ec158947b9633b31ae26` |
