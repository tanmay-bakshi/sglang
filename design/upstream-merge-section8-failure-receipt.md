# Upstream Merge Section 8 Failure Receipt

## Verdict

- Ladder result: `STOP` at rung 1 (build, import, and authoritative server boot)
- Merge revision: `0bdf639686b01fedc0db2c25ee7e3b62f1001c63`
- Fork parent: `e8a2a54c2923f13b9c7647bc3255c2725497472e`
- Pinned upstream parent: `6afb5e17712e2e90b60ba8456ca893e529316869`
- Branch: `tanmay/streaming-session-token-api`
- Evidence root: `/workspace/upstream-merge-20260830/section8-0bdf639686`

The one-pass Section 8 ladder is stopped. The merged runtime imported cleanly,
loaded both target and DFlash draft weights, allocated the production BF16 KV
pools under TP2, and completed FlashInfer FP4 autotuning. It then failed while
capturing the first prefill CUDA graph, before HTTP readiness and before any
request was served. Per the ratified ladder, no source correction, second boot,
or later rung was attempted.

## Failure

The admitted Gemma GeGLU kernel could not compile its JIT module:

```text
RuntimeError: Capture prefill CUDA graph failed: Failed to build JIT module
sgl_kernel_jit_gelu_tanh_and_mul_fp4_quant_true

gelu_tanh_and_mul_fp4_quant.cuh(172): error: name must be a namespace name
    using namespace host;
gelu_tanh_and_mul_fp4_quant.cuh(174): error: identifier "SymbolicSize" is undefined
gelu_tanh_and_mul_fp4_quant.cuh(182): error: identifier "TensorMatcher" is undefined
gelu_tanh_and_mul_fp4_quant.cuh(195): error: identifier "RuntimeCheck" is undefined
gelu_tanh_and_mul_fp4_quant.cuh(246): error: identifier "LaunchKernel" is undefined
```

The server exited at `2026-08-30T07:09:04Z`. The frozen launcher SHA-256 is
`2bf3bd2156e9e699884802fdb001fed6462cb0e4e9c26bf1f88895eab5ef9dda`;
the complete server log SHA-256 is
`2d1f3d4d511528b24eb53257756207af8c8472c17b4d88f8687cde7f13b4c6a9`.
The generated JIT translation unit and Ninja recipe are retained under
`failure/jit/`.

## Root cause

This is a silent cross-file merge incompatibility, not a model, CUDA, memory,
transport, or hardware failure.

The fork-only GeGLU header was carried forward byte-for-byte. In the fork
parent, JIT host helpers such as `host::TensorMatcher` and
`host::RuntimeCheck` lived at global scope. Upstream moved the JIT API beneath
the outer `sglang` namespace:

```text
fork parent tensor.h: namespace host { ... }
upstream tensor.h:    namespace sglang { namespace host { ... } }
```

The custom header still opens only an anonymous namespace, then uses
`using namespace host;` and unqualified `device::` helpers. Git therefore had
no textual conflict, while the composed translation unit lost every host-side
symbol at compile time. The other fork-only Gemma JIT header,
`gemma4_qkv_norm_rope.cuh`, has the same namespace assumption. It remained
dormant because the production launcher correctly kept QKV fusion disabled,
but it belongs to the same compatibility repair and must not be enabled in the
merged tree as it stands.

The bounded source trace is sealed at
`failure/namespace-compatibility-trace.txt`. Relevant SHA-256 values are:

- GeGLU header: `1d6021b9092fd88911edbae84e02c6aadc9e378c132dec9da7fe93c37a67a917`
- dormant QKV header: `4788433d0bd07767e5f689144e44fadc694b6909f2ee56930d698ff0f8daf8ad`
- merged `tensor.h`: `61747de0460e2075dc6bd298212d860b4bb9812645b33b48316b3648d9ec3d40`
- merged `runtime.cuh`: `f5d625427f6a78ea945adafe1eb0c9af35c5a46ae36ebe20d073756db7b04654`

## Pre-GPU qualification and runtime

The coherent 25-file CPU seal completed before the authoritative boot:
`1848 passed, 1205 skipped, 196 subtests passed`. Its log SHA-256 is
`121ee1f346fcc0fa49fd212ce4e57d9dff9fdbef8a4f89af6c9f8b2d680567e1`.
The three previously required seeded-defect RED proofs remain sealed under
`/workspace/upstream-merge-20260830/merge-cpu/semantic-audit-Tfg89e/`.

The isolated runtime used Python 3.12.3, torch 2.13.0, Triton 3.7.1,
FlashInfer 0.6.17, sglang-kernel 0.4.6.post1, Model Optimizer 0.45.0, and
Run:ai Model Streamer 0.16.1. System-site visibility was disabled, package
compatibility was green, and no version check was bypassed. The final runtime
freeze SHA-256 is
`df60b8599c8572c053ac1134a06b45d1bc7531066f2e42b1740aa8bc7d95e6cb`.

## Ladder disposition

Rungs 2 through 6 were not run. In particular, the 28-case corpus,
23-probe streaming-session suite, `measure_decode` comparison, live-duel spot
check, and non-gating D-t scoping probe remain unexecuted against the merge.
Gate 0 remains sealed green and is not rerun.

## Hardware and lifecycle closure

Only dev-1 physical GPUs 2/3 were acquired. Standing services on GPUs 0/1
were not contacted and their pre/post process inventory is byte-identical.
Dev-2 and dev-3 were not contacted.

The failed process tree exited on its own fail-closed path. The tmux session
and ports 32380/34380 are absent, GPUs 2/3 have no clients and returned to
120 MHz, and the zombie count is zero. The complete eight-GPU pre/post state
is byte-identical (SHA-256
`50994e3e3c622b58572cc8df0533a1ccf072b9e49ab31a1e28fd6461ed6f0e97`).
Corrected and uncorrected ECC, retired/remapped rows, and remap failures are
zero. No XID or SXid occurred. The only kernel lines during the lineage were
two instances of the already documented benign
`knvlinkSendInbandData_IMPL: Failed to send inband data: 0` signature, once
during initialization and once during teardown.
