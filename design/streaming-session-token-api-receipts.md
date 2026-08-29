# Streaming-Session Token API Qualification

## Seal

- Qualified implementation commit: `115ee146fc634e92c80be1552a68214843c1fdae`
- Qualified implementation tree: `1122f928f9dbf5539564b0d7a805b3c8c77e99d7`
- Branch: `tanmay/streaming-session-token-api`
- Base revision: `c10c2167c`
- Ratified design SHA-256: `3c6d54183ae94a5cce5a83ae1bef5616473e4d71365e30abbb89e7c69399934a`
- Qualification host: `gemma-dev-1`
- Receipt root: `/workspace/streaming-session-probe-20260828/receipts`

The branch commit that adds this receipt changes documentation only. All CPU and GPU
qualification below ran against the implementation commit and tree named above.

## CPU qualification

`final-head-cpu-qualification-115ee146.log` completed with 155 passed tests, 19
warnings, and 11 passed subtests in 29.71 seconds. Its SHA-256 is
`a1868210ad0ba35514d9954891d70e3973e1b4b5d3db56efbf4ff0f5e9ffa8e4`.

The final suite explicitly covers the two composition cases required at seal:

- `TestStreamingSessionAdmission::test_queue_admitted_mutation_survives_abort_before_prefill`
  admits truncate, append, and commit mutations at queue entry, aborts before
  prefill, verifies tip/floor/slot consistency, and successfully continues the
  session.
- `TestSessionTokenShare::test_expected_tip_conflict_is_typed_and_non_destructive`
  submits stale `expected_tip` together with `truncate_to=0`, receives the typed
  conflict, and verifies that token arrays, tip, floor, last request identity, and
  activity time remain unchanged.

### Pre-existing registry debt

The read-only registered-test validator exits nonzero for these seven files because
they lack a CI registry call:

- `test/registered/unit/disaggregation/test_nixl_packed_decode.py`
- `test/registered/unit/disaggregation/test_packed_runtime.py`
- `test/registered/unit/entrypoints/openai/test_gemma4_chat_template.py`
- `test/registered/unit/managers/test_tp_worker_prefill_profiling.py`
- `test/registered/unit/mem_cache/test_load_back_result.py`
- `test/registered/unit/mem_cache/test_radix_cache_metrics_rank_labels.py`
- `test/registered/unit/observability/test_req_time_stats_diagnostic_durations.py`

Each file is byte-identical between base `c10c2167c` and the qualified implementation
commit. These failures are baseline registry debt, not implementation failures, and
are not counted among the passing tests above.

## GPU qualification

The four-row matrix ran on physical GPU
`GPU-57b837f8-ce87-421b-e6eb-ddc5f547f911`. Every row used the exact qualified
implementation commit, disabled test retries, passed, and completed a clean teardown.

| Row | Receipt stem | Result | Run SHA-256 | Preflight SHA-256 | Postflight SHA-256 |
|---|---|---:|---|---|---|
| DFlash, one-shot | `stage5-dflash-one-shot-final-115ee146-r1` | 7 passed, 587.933 s | `bce63dbf79290bd7eb7d156c39fa954189912d3096cc9c2803837c96b1161e74` | `453e243cc3697f027185f8c32e7b28784d6073712a8967fbb7a5f43701ce0f12` | `cb2604cc996613aab5c6c90565e2a88c1d2d2c6fe29661e733751a0b2f369b32` |
| DFlash, chunked 1024 | `stage5-dflash-chunked-final-115ee146-r1` | 1 passed, 222.944 s | `4d1d1ff49bf8cf927b4f7c59f9322fe3ac4766be89edb9aba9d8f1be21de4039` | `13ee94f7c007ddad8c49ac63aa7b37e6e2c10f65f7ca146cac14b165f3575e8e` | `3c3fa64cb66cf6e18ad8f4744e95847fdcdb7bbf8ce76b3ec30a73b9256b893c` |
| No speculation, one-shot | `stage5-no-spec-one-shot-final-115ee146-r1` | 1 passed, 181.954 s | `462ce2b079c15f24a7abc107c3260c06769094a27eebf0afe100b8282c544885` | `acf02cdf1a5d304cadf4a59d456366f800f93de6da699965a466ea3e94c5ca86` | `d3bf04be9ef7d1a57b659d6609e8c9ffb102e84629eeb3a270fb9a25276ab01b` |
| No speculation, chunked 1024 | `stage5-no-spec-chunked-final-115ee146-r1` | 1 passed, 183.965 s | `553721cdd870d40e67661a354ed26c94776f60c7c0defd6ff254e75029796dc0` | `fcf1d47a01de0163601752a1b0f2aea725fbc9a7269615b28151c06c7fce8996` | `8b7590436c7ebdae3ea4090fba989ef4d2e85c8894a03d20edbed712100007a7` |

The matrix completion receipt is
`stage5-matrix-final-115ee146-r1-complete.log`, SHA-256
`e9daf311c290586210ca554208113db82987538ce94bdc8834ff0b7deebd86ad`.

## Oracle sensitivity

A scratch-only mutant changed the truncation retention boundary from page-floor to
page-ceiling alignment. The mutant was committed as
`29001951c994a6a07345b122b1321a474abd4771` with tree
`0e59af4599375afd5a9a9e3a74faadea8c02acf3`. It was never composed into the
qualified branch and is not pushed.

The schedule-matched no-speculation one-shot truncate oracle went deterministically
red, with retries disabled, at the intended boundary assertion:

```text
truncate cache boundary mismatch at target=1900:
expected=1856, hot=1920, peer=1920
```

The test exited 1 after 136.446 seconds. The receipt stem is
`stage5-seeded-defect-no-spec-one-shot-29001951-r1`. Its hashes are:

- Preflight: `e6be31ece96e1728b2933e8bf44103aed8a9af789c12668ee29c65872b438680`
- Run: `16232c64d5d8062b147a4598e97c0f88bb23ff19ba55a0c992d60c824dcf54a3`
- Postflight: `2c5f7e9416b26fbcb5cd5dfbc880933bcb613524ec335052d9d1ef4a21318674`

This demonstrates that the redesigned schedule-matched oracle detects a one-page
truncate/SWA retention error rather than merely passing the qualified build.

## Health closure

Every clean-matrix and seeded-defect pre/postflight pair observed eight GPUs, no
resident compute processes after teardown, zero zombies, zero XIDs/SXids, zero ECC
movement, and zero row-remap failures. Historical soft-signal counts did not move:

- multicast join `-22`: 409
- `free_os_event`: 6
- `sm_throttle_assert`: 0
