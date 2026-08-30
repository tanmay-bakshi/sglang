# Upstream Merge Gate 0 Success Receipt

## Verdict

- Gate: fresh pre-merge Gate 0 after the schedule-matched recovery-oracle
  correction
- Result: `PASS`
- Source revision: `66c47a38863cdb7230d76ad4b0d190733bf13858`
- Source tree: `b686f307c96dd40a8238db4b4a26f9b34203dd70`
- Upstream target: `6afb5e17712e2e90b60ba8456ca893e529316869`
- Merge status at qualification: not started

Gate 0 is closed. The production-config TP2 fork passed the serializer corpus,
decode controls, complete streaming-session suite, and hardware closure in one
server lineage with `failfast=True` and retries disabled.

## Schedule-matched oracle qualification

The oracle correction landed alone as `66c47a3886` (`Match Streaming Session
Recovery Oracles By Schedule`). It requires exact equality between the hot
post-conflict continuation and an honestly reconstructed replay with the same
chunk schedule. It separately requires exact equality between the cold
reconstruction and its own same-schedule replay. The field-for-field public
session-state comparison after the five-conflict volley remains unchanged.
Across different schedules, the test requires only equal token count, non-empty
output, and valid tokenizer-vocabulary IDs.

Before GPU qualification, the seven registered session unit files passed:

- 73 tests passed, with all 7 registered subtests green
- CPU receipt SHA-256:
  `b61e3dde931a1fc8d79a052bcf456e3d2b7e4168279c5b1451c9366665b1fd40`
- Session-kit SHA-256:
  `3b9601450ed4be0c04cc78d019ed887cf8aeafe9f61e94ff293c7be7c52fefcd`

## Frozen lineage

The clean detached worktree was
`/workspace/upstream-merge-gate0-ruling3-66c47a3886` on `gemma-dev-1`.
Physical GPUs 2/3 were used; the standing fixtures on GPUs 0/1 were untouched.
The single server used the dev-3 production shape: TP2, BF16 KV, DFlash, the
admitted GeGLU fusion, QKV fusion disabled, production graph buckets, TRT-LLM
MHA, 128K context, streaming sessions, and metrics. Dev-3 was not contacted.

- Frozen plan SHA-256:
  `c16e63d00f280b001e0c8ef532184e22b2ca1aa76efa27ece0d622331347f78a`
- Frozen launcher SHA-256:
  `9549847017775d2bc8b50bb221609615d6c33bb69672ec271610e654bc8576bc`
- Server readiness: 313 seconds, one launch, no retry

## Functional results

The serializer and live prompt-token corpus passed all 28 cases byte-for-byte
and token-for-token. The reference includes the vLLM-parity reasoning-retention
and tool-round cases.

- Corpus: 28/28
- Reference SHA-256:
  `c9a9fbc4b46a9773e787001b038d64a8f64509c9f0ef60abc60cec2eed9846d4`
- Corpus log SHA-256:
  `d01dcbe9ea756d15512029a72b4281a2bd5174eff73b817d980247c37edc5a0a`

The fresh BF16-KV/TP2 decode controls are:

| Depth | Steady ms/token | Mean tokens/chunk | Accept/verify |
|---:|---:|---:|---:|
| 8,034 | 2.209749 | 3.390728 | 3.413333 |
| 44,034 | 1.718392 | 4.612613 | 4.654545 |

The 8,034-depth first continuation retained the already observed approximately
5.2-second warm-path event. The 44,034-depth control was 125.98 ms to first
token. These values are the pre-merge controls; the registered post-conflict
stall remains a separate non-gating performance item.

- Decode JSONL SHA-256:
  `1c24382e0286e4a9d7ad0356353c80b97fff9ea307a233287d61a90884ead88a`

The full external streaming-session suite passed once with
`SGLANG_TEST_MAX_RETRY=0` and `failfast=True`:

- 7/7 tests passed in 416.741 seconds
- first real small extend: 0.154586 seconds
- schedule-matched truncate/SWA equivalence: passed
- deep streamed abort exact-slot preservation: passed
- idempotency and recovery, including both exact replay oracles: passed
- SWA pin accounting and commit release: passed
- zero-token mutation boundaries: passed
- truncate/abort leak closure: passed
- Suite log SHA-256:
  `b8d244eeaabfd6817e4e6ba32a7565340ab8563fcc9750d238eed026b807ba6f`

Terminal metrics remained single-emitting under TP2. TP0/TP1 values were 15/0
truncations, 12/0 commits, 5/0 aborts with the slot preserved, 5/0
idempotency conflicts, 33/0 close reaps, and 1/0 timeout reaps. Held full and
SWA KV gauges both returned to zero.

- Terminal metrics SHA-256:
  `42161af840728c689789ffc479b7886514ea4387d138731a7eba719f9f599663`
- Server log SHA-256:
  `c2c7cef071ac6f2f64e4ea688ce18f4226e6cbd244e0f7e0b7e3105483914d8b`

## Health and closure

The server process group was verified by PID, PGID, model, ports, TP size, and
KV dtype before termination. GPUs 2/3 released every client and returned to
120 MHz. All eight GPUs remained enumerated. Corrected and uncorrected ECC,
retired pages, row remaps, XID/SXid, and zombie counts remained zero. NVLink
protocol and link-recovery errors did not move.

Two kernel lines appeared during the lineage, both the documented benign
`knvlinkSendInbandData_IMPL: Failed to send inband data: 0` signature, once at
NVLink setup and once at teardown. There was no correlated workload or health
failure.

- Preflight SHA-256:
  `72ce7e91015c0870150885b1284b948121a50db65174de464fae93d344b81942`
- Postflight SHA-256:
  `4c24696609fdf8e44a1248288532a67c3a0338ed38187c2374b8c32f8e846983`

The evidence root is
`/workspace/upstream-merge-20260830/gate0/ruling3/fresh-66c47a3886` on
`gemma-dev-1`.
