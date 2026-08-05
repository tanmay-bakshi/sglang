# Prefill TP dispatch profiling

This trace contract localizes steady-state arrival skew between tensor-parallel
prefill ranks without stacks, shape capture, or Python wrappers around every
collective. It combines two prefill-only user annotations with existing SGLang
and Kineto events:

| Phase | Trace event | Source |
|---|---|---|
| Scheduler entry and exit | `scheduler.run_batch` | Existing scheduler annotation |
| Schedule-to-forward conversion | `sglang.prefill.forward_batch_init` | TP worker |
| Model dispatch and return | `sglang.prefill.model_runner_dispatch` | TP worker |
| Model execution geometry | `step[EXTEND bs=<n> toks=<n>]` | Existing model-runner annotation |
| Collective host entry and exit | `c10d::*` | Native Kineto/PyTorch events |
| Custom-AR host entry and exit | `sglang::outplace_all_reduce` | Registered SGLang custom op |
| Collective device execution | NCCL kernel events | Native Kineto/CUPTI events |
| Custom-AR device execution | Correlated custom-AR CUDA kernels | Native Kineto/CUPTI events |

Each complete range carries the standard Chrome trace fields `pid`, `tid`,
`ts`, and `dur`. The trace root's `baseTimeNanoseconds` aligns timestamps, and
`distributedInfo.rank` plus the `TP-<rank>` trace filename identifies the TP
rank. Join the rank to `deployment/ranks/prefill.rank-<rank>.json` for the
deployment's authoritative `physical_gpu`; a container-local CUDA device index
is not a physical GPU identity.

`sglang.prefill.model_runner_dispatch` contains the existing `step[...]` span
and native collective events. The first `c10d::*` event nested under each model
dispatch or the first nested `sglang::outplace_all_reduce` event is therefore
the rank's first observed collective entry. Record which family won; c1 can use
the custom-AR path while c4 can first expose skew through NCCL. The selected
event's `ts + dur` is the host-side collective exit. Its corresponding NCCL or
custom-AR CUDA launch and kernel events retain the normal Kineto correlation
fields.

The markers use `torch.profiler.record_function` only while a torch profiler is
active. Non-prefill batches take a shared null context and do not call the
profiler probe. Native c10d/NCCL events and the registered custom-AR op remain
the collective authorities; adding another Python range to every collective
would add overhead without more timing information.

## Matched c1 and c4 capture

Use the production TP2-prefill/TP1-decode runtime with NVFP4 target weights,
DFlash enabled, the same prompt material, and the same packed-transfer settings
as the accepted production path. Capture the prefill engine only.

1. Run an unprofiled warmup cohort after service startup and CUDA-graph warmup.
2. Use profiled cohort matrices with `warmup_requests: 0`; no benchmark warmup
   request belongs inside the trace.
3. Capture `p1024-o128-c1-shared-exact` and
   `p1024-o128-c4-shared-exact` separately. Use at least 32 measured requests
   for c1 and 64 for c4 so steady-state distributions are visible.
4. Set `--engine prefill --no-with-stack --no-record-shapes
   --no-merge-profiles`. Preserve both raw TP-rank traces, the deployment
   manifest, rank evidence, source identity, benchmark request records, and
   profiler-control timestamps.
5. Verify both rank traces contain the same number of matched model-dispatch
   and step spans, and that every analyzed step has a nested c10d or custom-AR
   collective.

Match steps across ranks by forward geometry and ordinal within each capture.
For every matched step and rank, extract:

- scheduler entry `ts`;
- forward-batch-init entry and exit;
- model-runner-dispatch entry and exit;
- first c10d-or-custom-AR collective family, entry, and exit;
- first correlated NCCL-or-custom-AR launch and kernel start/end.

Report median, p95, and maximum cross-rank skew for each entry phase. Also
report per-rank phase offsets (`forward_batch_init - scheduler`,
`model_runner_dispatch - scheduler`, and `first_collective - model_dispatch`)
so an inherited scheduler skew is not misattributed to the model. Report native
c10d or custom-AR duration and its NCCL or custom-AR kernel duration separately;
the host ranges overlap GPU work and are not additive. Stratify the result by
collective family rather than averaging c10d and custom-AR steps together.

Interpret the first phase at which the rank delta appears:

- skew already present at `scheduler.run_batch` points to scheduler/request
  coordination;
- aligned scheduler entry followed by divergent init or dispatch points to
  host dispatch, rank CPU placement, or batch construction;
- aligned model dispatch followed by divergent collective entry points to the
  pre-collective model path;
- aligned collective entry with divergent exit or device execution points to
  collective/topology behavior.

## Placement control

The measured baseline remains prefill GPUs 3 and 4 with decode GPU 1. Sysfs
reports GPU 1 on NUMA node 0, GPUs 4 and 5 on NUMA node 1, and GPU 3 with no
declared NUMA node. NUMA is therefore a hypothesis, not a conclusion.

If steady-state host or collective-entry skew survives both c1 and c4 captures,
repeat the same captures with prefill GPUs 4 and 5 and decode GPU 1. Keep source,
model flags, clocks, workload records, request order, and profiler settings
fixed. This shared-node prefill placement is the clean control against the
3-and-4 baseline. An A/B/A order guards against drift. Only a repeatable shift
in the phase where skew first appears supports a placement or NUMA explanation.
