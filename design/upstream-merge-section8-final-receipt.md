# Upstream Merge Section 8 Final Qualification Receipt

## Verdict

The Section 8 ladder is `PASS` on executable revision
`881d1cbdfe0d2477d3689328a1e3d20827418928`.

- Merge revision: `0bdf639686b01fedc0db2c25ee7e3b62f1001c63`.
- Pinned upstream parent:
  `6afb5e17712e2e90b60ba8456ca893e529316869`.
- Namespace-port revision:
  `881d1cbdfe0d2477d3689328a1e3d20827418928`.
- Branch: `tanmay/streaming-session-token-api`.
- Host: `gemma-dev-1` only.
- Evidence root:
  `/workspace/upstream-merge-20260830/section8-881d1cbdfe`.
- Continuation root:
  `/workspace/upstream-merge-20260830/section8-881d1cbdfe/ruling5-continuation`.

Gate 0 remains sealed at
`66c47a38863cdb7230d76ad4b0d190733bf13858` and was not rerun. The final
runtime was a clean detached worktree at `881d1cbdfe`; the two receipt-only
commits after it changed no executable source.

## Frozen runtime

The authoritative server used physical GPUs 2/3, UUIDs
`GPU-d2a77370-eb5b-af94-fc2c-21cb7e2f271c` and
`GPU-de79342b-f765-3388-1e03-41f12396df79`, on HTTP port 32384 and NCCL port
34384. It ran the production ModelOpt NVFP4 target with BF16 KV, TP2, DFlash,
the admitted GeGLU fusion, dormant QKV fusion disabled, the production prefill
graph buckets, TensorRT-LLM MHA, 128K context, metrics, and streaming sessions.
The continuation launcher SHA-256 is
`c9906c9ea4ce42927aee2be32a8b03b2ad72db0c1efc2e7f28990fcaa3699f2f`.

The environment retained torch 2.13.0, Triton 3.7.1, and
flashinfer-python 0.6.17. The protected services on GPUs 0/1 were not
contacted or perturbed. `gemma-dev-2` and `gemma-dev-3` were not contacted.

## Rungs 1 and 2, carried green

Rung 1 proved both fork-native JIT kernels against upstream's namespaced API.
The dormant QKV fusion compiled for both `int32` and `int64` positions and its
complete bitwise matrix passed, `12/12`. Normal TP2/BF16 boot compiled and
captured all eight GeGLU prefill graph buckets. The GeGLU harness matched every
packed FP4 nibble and scale byte at 1, 8, 16, 17, 64, 1024, and 8192 tokens.

Rung 2 passed with `1848 passed, 1205 skipped, 196 subtests passed`. Each of
the three required ownership defects went RED in an isolated worktree, then
the exact tests passed after seed removal. Those sealed rungs were not repeated
after the rung-3 design stop.

## Rung 3, deliberate upstream schema adoption

The authoritative merged capture passed all 28 template, prompt-token, output,
reasoning, usage, and stable-response checks. Ladder ruling #5 adopted
upstream's always-present field:

```json
{"weight_versions":[{"version":"default","start":0,"end":1}]}
```

The old Gate-0 reference remains intact with SHA-256
`c9a9fbc4b46a9773e787001b038d64a8f64509c9f0ef60abc60cec2eed9846d4`.
The separately named ruling-5 reference is byte-identical to the authoritative
capture, both SHA-256
`a5ad63d7728f96322bc3916f5c5184b663654109a2c97b33047ac75b041d2b3c`.
Removing exactly that field from all 28 new responses recovers the old stable
corpus byte-for-byte. No serializer suppression, conditional emission, or
comparator normalization was introduced.

## Rung 4, streaming sessions under production TP2

The complete external kit ran once with retries disabled and fail-fast enabled:

```text
Ran 7 tests in 415.905s
OK
```

The seven methods contain the 23 ratified probes: the first-real-extend bound,
schedule-matched truncate equivalence, deep abort with exact-slot preservation,
idempotency conflict and recovery, SWA pin accounting, mutation edge cases,
and truncate/abort leak closure. Every method passed. All logical session
counters remained zero on TP1 and single-emitted on TP0. At closure,
`num_streaming_sessions`, held full-KV tokens, and held SWA-KV tokens were all
zero.

The suite log SHA-256 is
`11ad04aec2a9e61a80a0012e472f8bccbb57076e241238271fc08ed151981957`.

## Rung 5, decode and Spec V2

The unchanged probe ran one 512-token burst at each depth. All histogram,
iteration-count, sampled-token, mean-chunk, and accept/verify conservation
identities reconstructed exactly.

| Depth | Gate-0 accept/verify | Candidate accept/verify | Gate-0 derived verify iteration | Candidate derived verify iteration | Delta |
|---:|---:|---:|---:|---:|---:|
| 8,034 | 3.413333 | 3.820896 | 7.542609 ms | 7.502627 ms | -0.5301% |
| 44,034 | 4.654545 | 4.413793 | 7.998333 ms | 7.790438 ms | -2.5992% |

Pooled accept/verify moved from `3.938462` to `4.096000`. The published
4.1-4.4 language is a regime-level prior rather than a literal per-row band,
because the authoritative Gate-0 rows themselves straddle it. The candidate
is plainly in the same DFlash acceptance regime, and derived verify-iteration
time is non-regressing at both depths. Spec V2 passes.

Candidate TTFT was 4380.998 ms at 8K and 136.111 ms at 44K. Those values and
p50 token timing are descriptive, not gates. The raw JSON-lines SHA-256 is
`9d6f6ad72f673e2320036d55907a5edcd99f1e2b48b58c513332ab9e965e4604`;
the independent adjudication SHA-256 is
`f7a8c28bbac48abd96a68495d954bffc8224002e26010155c6cea4a6dccdd107`.

## Rung 6, same-server live-duel shape

The standing prohibition on dev-3 access required a narrowed spot check. Chat
and raw-token sessions ran sequentially against the same authoritative server.
This qualifies merged-server API-path shape. It does not qualify the production
proxy, compare revisions, require stochastic token equality, or create a causal
performance A/B.

| Arm and bucket | n | Median wall | p90 wall | Maximum wall | Median ms/completion token |
|---|---:|---:|---:|---:|---:|
| Chat, poll | 9 | 534.417 ms | 1919.278 ms | 1919.278 ms | 7.7552 |
| Chat, broadcast action | 3 | 1071.694 ms | 1899.390 ms | 1899.390 ms | 2.4923 |
| Session, poll | 11 | 245.367 ms | 293.032 ms | 1153.626 ms | 4.2380 |
| Session, broadcast action | 5 | 1189.925 ms | 2390.126 ms | 2390.126 ms | 1.9776 |

Both arms began at 8,034 prompt tokens and crossed 40K. Both preserved the
poll/action latency modes. The 18-row session arm reached 51,556 prompt tokens
and reused `99.6313%` of post-deep prompt tokens at median. Its exact deltas
were 17 commits and one close reap on TP0, zero logical events on TP1, and zero
truncate, abort, or conflict events. The session and both held-KV gauges
settled to zero.

The chat arm's first bootstrap request spent 96.112 seconds compiling the cold
FlashInfer stochastic-sampling JIT artifact. This did not enter either semantic
latency bucket and did not require a retry, but it is a real first-use latency
hazard. Packaging or warmup coverage for the sampling artifact is queued as a
post-merge performance item alongside the already characterized shallow eager
continuation warmup gap.

The chat and session JSON SHA-256 values are
`1484b1080c1837194a6ee778be0603092bc8e2bc22bb194206ba749bef2c513c`
and `6b11d4a52573fb1af3587f0857c7ff41fb45827ab2f5aa0e79aab698b11fecbb`.
The duel adjudication SHA-256 is
`8e5679873670cf64368d1eb7a167968b5edd629d3990f336217456fd0102e71d`.

## Lifecycle and hardware closure

The validated server PID, PGID, and SID were all 251027. Every compute client
on the target UUIDs belonged to that process group, and no protected-service
PID did. SIGTERM retired the group gracefully; no SIGKILL was used. Ports
32384/34384 and the exact tmux identity are gone. GPUs 2/3 are empty at 0 MiB
and 120 MHz.

The protected GPU 0/1 process identities and memory footprints are byte-equal
before and after. Eight GPUs enumerate and the zombie count remains zero. All
volatile and aggregate corrected/uncorrected ECC counters are zero; remapped
row counts are zero, pending/failure states are `No`, and the recovery action
is `None` on every GPU. No XID or SXid occurred. The queued benign
`knvlinkSendInbandData_IMPL` signature did not recur in this continuation.

The final runtime source remained clean at `881d1cbdfe`. The server log
SHA-256 is
`e46d6d269aaf78031eb2271ca27bd418c03684f37e2043372d9d1b2286955be0`.
The qualification manifest SHA-256 is
`79bdf543bc8db92fbb340ea5bae51c83d86304bc6781152c88a3af1dca2cf361`.

The upstream merge plus mechanical namespace port is qualified. Post-merge
kernel candidates, the D-t shallow eager warmup extension, and stochastic
sampling JIT warmup remain outside this lineage.
