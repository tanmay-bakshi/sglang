# Upstream Merge Gate 0 Ruling 1 Failure Receipt

## Verdict

- Gate: fresh pre-merge Gate 0 after the session-metric output-rank fix
- Result: `FAIL_STOP`
- Stop trigger: the unchanged TP2 streaming-session suite failed, with
  `failfast=True`, in `test_30_idempotency_and_recovery`
- Classification: a second pre-existing TP>1 fork qualification gap, exposed
  only after the metric fix allowed the suite to pass its earlier truncate
  assertion
- Merge status: not started; upstream
  `6afb5e17712e2e90b60ba8456ca893e529316869` remains unmerged

The session-metric defect named by Gate-0 ruling 1 is fixed and independently
proven. Gate 0 as a whole remains closed because the deeper recovery oracle is
red. No attempt was made to change that oracle or fix the newly exposed source
behavior.

## Output-rank metric fix

The production trace established that `SchedulerMetricsCollector.init_new()`
constructs and attaches a collector on every scheduler when metrics are enabled.
Stock scheduler metrics still emit once because `SchedulerMetricsReporter`
checks `current_scheduler_metrics_enabled` immediately before emission.
Session counters routed through `ReqTimeStats` called the attached collector
directly and bypassed that reporter gate.

The source fix gates the complete logical session-metric family at increment or
log time using the unique session-control output rank, while retaining
`--enable-metrics-for-all-schedulers` as the explicit per-rank escape. The
sweep covers truncations, commits, aborts-with-slot-preserved, idempotency
conflicts, reaps, active-session count, held full-attention tokens, and held SWA
tokens. The reap observer and scheduler-side conflict accounting consume the
same derived capability.

- Fix commit: `13fc2d2c74d46ee544d0e0da2bf9c3467f93eb9a`
- Fix tree: `7cd1fdc58869bb822581bf2515227f3a69eef5d6`
- Commit title: `Gate Streaming Session Metrics By Output Rank`
- Broad CPU qualification: 159 passed, 19 warnings, 16 passed subtests in
  30.77 seconds
- CPU log SHA-256:
  `c75e4b6767bb5b7f66e117c538d441dddec795c1a73af761fc601bbfa9d57cf5`

## Seeded-defect RED proof

A detached scratch commit removed only the truncation counter's output-rank
guard:

- Mutant commit: `e75e5939b6a68f36ad9b28487996098e9f5af768`
- Mutant tree: `c97f6aa8fff8d7f1119f1fe9827693c9917c201e`
- Mutant disposition: detached scratch evidence, never merged

The production-config TP2 suite ran once with `SGLANG_TEST_MAX_RETRY=0` and
`failfast=True`. The warmup test passed. The unchanged truncate oracle then
failed at the intended aggregate metric assertion:

```text
AssertionError: metric sglang:streaming_session_truncations_total did not settle:
expected=11.0, actual=22.0
```

The failure metrics contain 11 truncations on TP0 and 11 on TP1. Artifact
SHA-256 values:

- preflight: `26fa81923364ce42a1e2788c9ae9d03b7b22a0a01e0e673f1f306b40354e04b5`
- suite: `a78bc44e6da50dcddbca90fd4f01a34eb0b01f75fd5377fbe619c3d20a0569b5`
- at-failure metrics: `71cbb94ea7256cff3bd6846fc201581f7ef5c20c594641ce077fdc9af86cd7c5`
- server log: `8126c1b52a29439bb876176cb8d6762f9e4efcf8642e5af514e0a1d60b09d223`
- postflight: `64624a1c08623ddaacd2eb55ea718b8a6be31ce53b9e6a593b72a5dce529134a`

The evidence root is
`/workspace/upstream-merge-20260830/gate0/ruling1/mutant-e75e5939b6` on
`gemma-dev-1`.

## Fresh full Gate 0

The good revision ran from the clean detached worktree
`/workspace/upstream-merge-gate0-ruling1-13fc2d2c74`, on physical GPUs 2/3,
with the dev-3 production shape: TP2, BF16 KV, DFlash, the admitted GeGLU
fusion, QKV fusion disabled, production graph buckets, TRT-LLM MHA, and 128K
context. The single-pass plan was frozen before launch; its SHA-256 is
`389115d34dd5c7c751e46b20065abc27b65afbdd188e70b8418b61438d0461bc`.

Passing rows:

- Static template and live prompt-token corpus: 28/28.
- Re-blessed reference SHA-256:
  `2394078d1a0d85a45c9d49599748302e8df128be2ac98285353bc61ee65ca087`.
- Decode at depth 8,034: 1.934826 ms/token steady state,
  accept/verify 3.908397.
- Decode at depth 44,034: 1.715951 ms/token steady state,
  accept/verify 4.612613.
- Session warmup, schedule-matched truncate equivalence, and deep-abort slot
  preservation passed under TP2.
- The decisive metric state is now single-emitting: TP0/TP1 values were 11/0
  truncations, 8/0 commits, 1/0 abort-with-slot-preserved, 5/0 idempotency
  conflicts, and 22/0 close reaps. Session gauges existed only on TP0.

### Stop trace

The fourth session test failed after all five stale `expected_tip` requests had
returned their typed conflicts and the test had verified that public session
state remained field-for-field unchanged. The subsequent accepted hot
continuation and an equal-token cold reconstruction had the same 116-token
prompt length but different greedy outputs:

```text
Traceback (most recent call last):
  File ".../gemma4_streaming_session_token_api_kit.py", line 2259, in test_30_idempotency_and_recovery
    run_recovery_qualification(self.base_url)
  File ".../gemma4_streaming_session_token_api_kit.py", line 1446, in run_recovery_qualification
    assert cold.output_ids == accepted.output_ids, (
AssertionError: conflict recovery changed greedy content:
hot=[6097, 12822, 83691, 57243], cold=[6097, 1852, 1852, 1852]
```

Native logs show that the hot continuation reused 100 cached tokens and took
1182.78 ms of forward time; the cold arm reused zero tokens and took 141.77 ms.
Both reported input length 116 and output length 4. The matching first output
token followed by divergence at token two leaves two live explanations: a real
TP2 continuation/cache-state defect after conflict handling, or numerical
sensitivity in the remaining non-schedule-matched hot-versus-cold oracle. Gate
0 does not authorize selecting between them by changing source or test, so the
line stopped.

Fresh-run artifact SHA-256 values:

- preflight: `94aad5546728c24b8d08a46922060dbc1673572b3bb8133df4c4d58b2d14d947`
- corpus log: `80d4c33a88d24281aae040c9389e7edd9c59c0ef3cb45a6cb82cb89c52e36eff`
- decode JSONL: `4c9d87d595cc8af3e5c4839f8db44ec38a3a4efd7a2d8a0a30adbabb271272b5`
- session suite: `b320c069261c66844baa49cd76378d7f8caeeafdae8e792bf7777d042b552319`
- at-failure metrics: `914dc087d5a5924ca4af2f102dc772073cde667c84cf16aa64df5e4546b0d8ac`
- server log: `d419dfc73e3df401874398e23307a503216702d4e5de79ed9bdb3bdfed901fd6`
- postflight: `c9eb7227b69fef5b595c636386b0579bfdffd1f39dcec0febb5032945feda035`

The evidence root is
`/workspace/upstream-merge-20260830/gate0/ruling1/fresh-13fc2d2c74` on
`gemma-dev-1`.

## Health and closure

Both GPU acquisitions used identity-checked process groups and completed clean
teardown. GPUs 2/3 are empty and downclocked to 120 MHz. All eight GPUs remain
enumerated; XID/SXid, ECC, row-remap, NVLink protocol/link-recovery error, and
zombie counts are zero. The queued historical soft signatures did not move during either
lineage: multicast-join `-22` stayed at 469, `free_os_event` stayed at 9, and
`sm_throttle_assert` stayed at 0.

The standing dev-1 TP1 fixtures on GPUs 0/1 were not interrupted. Dev-3 was not
contacted. The implementation tree remains the clean fix tree named above and
there is no `MERGE_HEAD`.
