# Gate 0 D-t Conflict Sweep Receipt

## Scope

- Result: `COMPLETE`, non-gating characterization only
- Source revision: `66c47a38863cdb7230d76ad4b0d190733bf13858`
- Source tree: `b686f307c96dd40a8238db4b4a26f9b34203dd70`
- Evidence root: `/workspace/upstream-merge-20260830/dt-conflict-sweep-66c47a3886`
- Physical GPUs: dev-1 GPUs 2/3

This is the bounded D-t probe registered by Gate-0 ruling #3. It does not
reopen Gate 0, qualify the merge, or create admission evidence. One server
lineage ran the production BF16-KV/TP2/DFlash flag set at the sealed pre-merge
revision. The conflict volley used the original five stale-request shapes and
was truncated or repeated to produce counts 0, 1, 5, and 10. Every accepted
continuation used the same 16-token extend shape and generated four greedy
tokens. Both GPUs' `clocks.sm` values were sampled every 20 ms around each
continuation.

The launcher SHA-256 is
`c8bcbdaa095b6bdc3ff9288e8e143e0a687d11588e7e2b0c13d774cfeb28851b`.
The probe SHA-256 is
`3a285dcc43b0fd7250a7ef1abd7858dcf5250dd73478cff1b04f2902006d192c`.

## Results

| Conflicts | Conflict wall sum (ms) | Continuation wall (ms) | Native continuation forward (ms) |
|---:|---:|---:|---:|
| 0 | 0.000 | 1166.032 | 1160.51 |
| 1 | 147.948 | 149.682 | 144.62 |
| 5 | 731.102 | 148.842 | 143.74 |
| 10 | 1455.295 | 147.209 | 142.45 |

All four continuations produced the exact token log
`[6097, 12822, 83691, 57243]`, with public tip 100, 100 cached tokens, 116
prompt tokens, and four completion tokens. Each stale request was rejected
without mutating the public session state, and the logical conflict counter
advanced exactly by the requested count.

Every one of the 340 clock samples across GPUs 2/3 reported P0 at 2032 MHz.
There was no idle clock state and no clock ramp before or during any
continuation.

## Verdict

Both registered live hypotheses are rejected:

1. **Serialized per-conflict debt:** rejected. Continuation time did not grow
   with conflict count. Counts 1, 5, and 10 were flat at 147-150 ms; the
   conflict waits were paid synchronously by the conflict requests themselves
   at approximately 145-148 ms each.
2. **Idle-downclock ramp:** rejected. SM clocks were pinned at 2032 MHz for
   every sampled point in every row.

The 1.16-second event landed in the zero-conflict row, which was also the
process's first shallow, cached, 64-row eager session extension. Once
that exact forward family had executed, all later schedule-identical
continuations were stable near 144 ms native forward time regardless of
conflict count. All rows remained on the same eager path (`cuda graph: False`),
so the result is not a graph-bucket switch. Queue time on the slow row was only
0.46 ms; the cost was inside the forward. No JIT cache file was created or
modified during the event, so the bounded evidence supports an in-process
first-use/lazy-initialization cost but does not identify a narrower internal
subcomponent.

The exact result JSON SHA-256 is
`6a1bb07166fe6b722a04738001e74b93b81cf417984607aed990957b3fcc99c8`.
The native request excerpt SHA-256 is
`df7a39fd74d69b6e1596dbe18384ad0589f3e2fdc96b1b6244ccac3c4764ef09`.

## Lifecycle closure

The server was identity-checked before termination. GPUs 2/3 released all
clients and returned to 120 MHz; ports 32390/34390 and the tmux session are
absent; the zombie count is zero. The eight-GPU pre/post inventory and the
compute-process inventory are each byte-identical. Corrected and uncorrected
ECC, retired/remapped rows, and remap failures remained zero, with no XID or
SXid. The only kernel lines were the established benign
`knvlinkSendInbandData_IMPL: Failed to send inband data: 0` signature at
initialization and teardown. Standing services on GPUs 0/1 were untouched;
dev-2 and dev-3 were not contacted.
