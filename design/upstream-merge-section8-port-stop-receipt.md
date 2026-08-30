# Upstream Merge Section 8 Namespace-Port Stop Receipt

## Verdict

- Ladder result: `STOP` at rung 3, the 28-case byte-exact parity corpus
- Qualified revision: `881d1cbdfe0d2477d3689328a1e3d20827418928`
- Namespace-port commit: `881d1cbdfe` (`Port Gemma JIT Kernels To Namespaced API`)
- Merge revision: `0bdf639686b01fedc0db2c25ee7e3b62f1001c63`
- Pinned upstream parent: `6afb5e17712e2e90b60ba8456ca893e529316869`
- Branch: `tanmay/streaming-session-token-api`
- Evidence root: `/workspace/upstream-merge-20260830/section8-881d1cbdfe`

Gate 0 remains sealed at `66c47a38863cdb7230d76ad4b0d190733bf13858`
and was not rerun. Section 8 stopped before the streaming-session,
`measure_decode`, and live-duel rungs. No source correction or corpus
normalization was attempted after the rung-3 result.

## Complete JIT delta and port

The exhaustive fork-delta sweep covered native headers and sources,
registrations, bindings, dependencies, model gates, benchmarks, and tests. The
only fork-native files that refer to upstream's moved JIT host/device API are:

- `python/sglang/kernels/jit/csrc/elementwise/gelu_tanh_and_mul_fp4_quant.cuh`
- `python/sglang/kernels/jit/csrc/elementwise/gemma4_qkv_norm_rope.cuh`

The supporting surface is the `flashinfer_trtllm` dependency registration,
the two Python JIT loaders and custom-op registrations, the Gemma model gates,
the two environment switches, two benchmarks, and the registered QKV test.
Upstream's generated wrapper already emits exports inside `namespace sglang`,
so none of those supporting sites required a change.

The port encloses each existing anonymous CUDA-header namespace in
`namespace sglang` and changes nothing else. Includes remain outside the
namespace. Launch geometry, scale plumbing, argument marshaling, wrapper
targets, registration, dependencies, and kernel arithmetic are byte-unchanged.
The source commit was pushed without rewriting the published merge history.

## Rung 1, build, compile, and production boot

Rung 1 passed.

- The dormant QKV fusion compiled explicitly for both `torch.int32` and
  `torch.int64` position specializations from a source-addressed JIT cache.
- Its complete registered bitwise matrix passed: TP1/TP2/TP4, 1/1024 tokens,
  and sliding/proportional RoPE, `12 passed`.
- The production TP2/BF16 server booted once on physical GPUs 2/3 with the
  frozen production flags. Normal startup compiled and captured all eight
  GeGLU prefill graph buckets from 1024 through 8192 tokens.
- The TP2 GeGLU harness passed at 1, 8, 16, 17, 64, 1024, and 8192 tokens.
  Packed FP4 bytes and swizzled scale bytes were exactly equal to the unfused
  chain for every row.

The server-log SHA-256 is
`0305c98be23d0f0d8580cb3669bcdca5c35ea2cff478c2ca8569ecaa22c6c185`.
The QKV compile, QKV bitwise, and GeGLU bitwise log SHA-256 values are
`3fa606d155c6a163acf742086fa4809cfdcaf0525f9c500c07ff096cb6bd2d69`,
`f748a58d4cc90a2243c16447c7ce824e3d33058b0463d69f514530867ad051a3`,
and `e3816f2176756c05254d11c03441595df7cc0ea4006f56a0278d79db06a5e49b`.

## Rung 2, invariants and seeded defects

Rung 2 passed: `1848 passed, 1205 skipped, 196 subtests passed`.

The first invocation exposed only a runner precondition: setting
`CUDA_VISIBLE_DEVICES` to the empty string caused upstream test collection to
index character zero of an empty value. Zero tests collected and no source
behavior was observed. That receipt is retained. With only free GPU 4 visible,
the unchanged revision and unchanged test population produced the green result
above.

Each required seeded defect was run in an isolated detached worktree and went
RED at its intended invariant:

1. omitting the fresh `ReqKvInfo` handoff failed the mid-abort ownership test;
2. keying ownership on `kv.is_released` failed the held-empty resume test;
3. dropping the held-empty SWA no-op failed the pre-allocation eviction test.

After removing the seeds, the three exact tests passed together. The
authoritative source tree remained clean throughout.

## Rung 3 stop, upstream wire-schema expansion

The capture runner passed all 28 serializer and live-token checks:

- fork template bytes equal the golden vLLM template bytes;
- fork and golden prompt token IDs are equal;
- every live server prompt-token sequence equals the golden sequence;
- the Boltzmann reasoning-retention ordering invariant holds.

The byte-exact response comparison then failed on one field, consistently in
all 28 cases. Gate 0 emitted:

```json
{"metadata":{"weight_version":"default"}}
```

The merged server emits:

```json
{"metadata":{"weight_version":"default","weight_versions":[{"start":0,"end":1,"version":"default"}]}}
```

No rendered text, prompt token, generated token, reasoning field, usage value,
or other stable response field differs.

This is an upstream semantic change, not a namespace-port defect. Upstream
introduced `sglang.srt.utils.weight_versions`, made the scheduler produce
per-token version spans, propagated them through tokenizer metadata, and
changed chat response construction from a literal `weight_version` object to
`build_endpoint_weight_version_metadata`. Even without a mid-request weight
update, the helper exposes the one default-version span. Because the frozen
gate is byte-exact, deleting that field from the comparator after seeing it
would be a post-result gate change. The ladder therefore stops for a design
seat disposition: retain the fork's public schema, accept and re-bless the
upstream schema, or define a different prospective compatibility contract.

The Gate-0 reference and merged capture SHA-256 values are
`c9a9fbc4b46a9773e787001b038d64a8f64509c9f0ef60abc60cec2eed9846d4`
and `a5ad63d7728f96322bc3916f5c5184b663654109a2c97b33047ac75b041d2b3c`.

## D-t follow-up

The D-t anomaly is closed as process-first eager-forward initialization, not
conflict debt or downclocking. The existing
`streaming_session_small_extend` warmup covers a deep 40,960-token session and
a cached 64-token extend. The missed family is the shallow cached continuation
that produced the first 64-row eager forward. Preserving deep coverage while
adding the shallow schedule requires either factoring the warmup body and
invoking both schedules or registering a second comma-separated warmup. It is
queued after the merge ladder and did not enter this merge+port lineage.

## Hardware and lifecycle closure

Only dev-1 was contacted. Standing services on GPUs 0/1 retained the same
process identities and memory footprints. GPUs 2/3 hosted the one authoritative
server; GPU 4 hosted the isolated kernel checks. Identity-safe teardown removed
the Section 8 process group, tmux session, and ports 32382/34382. GPUs 2 through
7 are empty at 120 MHz and the zombie count is zero.

Corrected and uncorrected ECC counts are zero, remapped-row counts and failure
flags are zero, and no XID or SXid occurred. The only two new kernel lines are
the already documented benign
`knvlinkSendInbandData_IMPL: Failed to send inband data: 0` signature, once at
startup and once at teardown.
