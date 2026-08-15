import dataclasses
import hashlib
import inspect
import threading
from types import SimpleNamespace

import pytest
from sglang.srt.disaggregation.common.packed_staging_protocol import PackedRequestKey
from sglang.srt.disaggregation.nixl.conn import NixlKVManager
from sglang.srt.disaggregation.nixl.packed_runtime import (
    PackedPrefillSubmission,
    PackedTerminalDFlashAuxiliarySource,
)
from sglang.srt.disaggregation.prefill import (
    PrefillBootstrapQueue,
    SchedulerDisaggregationPrefillMixin,
    _TerminalPrefillBatchLeaseLedger,
    _TerminalPrefillLaunch,
)
from sglang.srt.disaggregation.terminal_progress.dflash_auxiliary import (
    DFlashBoundaryPrefillSource,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalPublicationIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.output_projection import (
    PrefillTerminalGatewayOutputProjection,
)
from sglang.srt.disaggregation.terminal_progress.publisher import (
    FrozenTerminalGatewayOutputProjection,
)
from sglang.srt.disaggregation.terminal_progress.source_plan import (
    PackedTerminalSourceIdentityPlan,
)
from sglang.srt.disaggregation.terminal_progress.source_wiring import (
    PackedTerminalSourceCancellationDisposition,
    PackedTerminalSourceSubmission,
)
from sglang.srt.disaggregation.utils import DisaggregationMode


@dataclasses.dataclass(frozen=True, slots=True)
class _NonPrefillProjection(FrozenTerminalGatewayOutputProjection):
    """Immutable projection from another serving protocol."""

    payload: bytes

    @property
    def digest(self) -> bytes:
        """Return the fixture projection digest.

        :returns: SHA-256 over the exact fixture payload.
        """

        return hashlib.sha256(self.payload).digest()


class _ManagerReactor:
    """Minimal admission owner used at the real manager boundary."""

    admission_open: bool
    require_calls: int
    stop_calls: int

    def __init__(self, *, admission_open: bool = True) -> None:
        """Create one reactor admission fixture.

        :param admission_open: Whether source admission initially succeeds.
        """

        self.admission_open = admission_open
        self.require_calls = 0
        self.stop_calls = 0

    def require_admission_open(self) -> None:
        """Require the source admission boundary to remain open."""

        self.require_calls += 1
        if not self.admission_open:
            raise RuntimeError("synthetic admission closure")

    def inventory(self) -> SimpleNamespace:
        """Return the fields consumed by manager failure recording.

        :returns: Started reactor admission state.
        """

        return SimpleNamespace(started=True, admission_open=self.admission_open)

    def stop_admission(self) -> None:
        """Close fixture admission exactly once."""

        self.stop_calls += 1
        self.admission_open = False


class _ManagerActor:
    """Record actor binding and PREPARE publication boundaries."""

    bind_calls: list[PackedPrefillSubmission]
    prepare_calls: list[PackedPrefillSubmission]
    fail_bind: bool
    fail_prepare: bool

    def __init__(self, *, fail_bind: bool = False, fail_prepare: bool = False) -> None:
        """Create an empty actor ledger.

        :param fail_bind: Whether actor ownership binding raises.
        :param fail_prepare: Whether PREPARE publication raises.
        """

        self.bind_calls = []
        self.prepare_calls = []
        self.fail_bind = fail_bind
        self.fail_prepare = fail_prepare

    def bind_terminal_owner(
        self,
        transport: PackedPrefillSubmission,
        identity: PackedTerminalSourceIdentityPlan,
    ) -> None:
        """Record the transport identity entering actor ownership.

        :param transport: Exact packed source submission.
        :param identity: Exact terminal source identity.
        """

        del identity
        self.bind_calls.append(transport)
        if self.fail_bind:
            raise RuntimeError("synthetic actor bind failure")

    def publish_terminal_owner_prepare(
        self,
        transport: PackedPrefillSubmission,
    ) -> None:
        """Record or reject one actor PREPARE.

        :param transport: Exact packed source submission.
        """

        self.prepare_calls.append(transport)
        if self.fail_prepare:
            raise RuntimeError("synthetic PREPARE failure")


class _ManagerBindingRegistry:
    """Record one external lifecycle binding surface."""

    bindings: list[TerminalRequestBinding]
    fail: bool

    def __init__(self, *, fail: bool = False) -> None:
        """Create an empty binding registry.

        :param fail: Whether registration raises after observing the binding.
        """

        self.bindings = []
        self.fail = fail

    def register_binding(self, binding: TerminalRequestBinding) -> None:
        """Record or reject one binding.

        :param binding: Exact lifecycle identity.
        """

        self.bindings.append(binding)
        if self.fail:
            raise RuntimeError("synthetic binding registry failure")


class _ManagerServing:
    """Model lifecycle commit and fail-closed source drain boundaries."""

    active: set[bytes]
    abort_calls: int
    fail_before_commit: bool
    fail_after_commit: bool

    def __init__(
        self,
        *,
        fail_before_commit: bool = False,
        fail_after_commit: bool = False,
    ) -> None:
        """Create one empty serving lifecycle fixture.

        :param fail_before_commit: Whether binding fails before lifecycle commit.
        :param fail_after_commit: Whether binding fails after lifecycle commit.
        """

        self.active = set()
        self.abort_calls = 0
        self.fail_before_commit = fail_before_commit
        self.fail_after_commit = fail_after_commit

    def bind_submission(
        self,
        submission: PackedTerminalSourceSubmission,
        release_resources: object,
    ) -> None:
        """Commit or reject one source lifecycle.

        :param submission: Exact immutable source submission.
        :param release_resources: Scheduler resource callback.
        """

        del release_resources
        if self.fail_before_commit:
            raise RuntimeError("synthetic pre-commit serving failure")
        self.active.add(submission.identity.local_binding.digest)
        if self.fail_after_commit:
            raise RuntimeError("synthetic post-commit serving failure")

    def inventory(self) -> SimpleNamespace:
        """Return the lifecycle fields consumed by manager recovery.

        :returns: Exact active source binding population.
        """

        return SimpleNamespace(
            wiring=SimpleNamespace(active_binding_digests=tuple(sorted(self.active)))
        )

    def begin_fail_closed_abort(self) -> None:
        """Record one process-fatal lifecycle drain."""

        self.abort_calls += 1


class _DFlashQuarantineOwner:
    """Record exact unpublished DFlash row quarantine calls."""

    leases: list[object]

    def __init__(self) -> None:
        """Create an empty quarantine ledger."""

        self.leases = []

    def quarantine_unpublished_source_row(self, lease: object) -> None:
        """Record one non-reusable device row.

        :param lease: Exact fixture row lease.
        """

        self.leases.append(lease)


def _manager_identity(*, local_rank: int = 0) -> PackedTerminalSourceIdentityPlan:
    """Build one TP2 source identity accepted by the real manager boundary.

    :param local_rank: Source rank selected as local.
    :returns: Complete rank-local source identity plan.
    """

    request_key = PackedRequestKey(room_id=811, request_generation=b"g" * 16)
    sources = tuple(
        TerminalProcessIdentity(
            process_generation=bytes((0x30 + rank,)) * 16,
            role=TerminalOwnerRole.SOURCE,
            tp_rank=rank,
            tp_size=2,
        )
        for rank in range(2)
    )
    decoder = TerminalProcessIdentity(
        process_generation=b"d" * 16,
        role=TerminalOwnerRole.DECODE,
        tp_rank=0,
        tp_size=1,
    )
    bindings = tuple(
        TerminalRequestBinding(
            request_key=request_key,
            owner=source,
            rank_manifest_digest=b"m" * 32,
            allocation_digest=b"a" * 32,
        )
        for source in sources
    )
    return PackedTerminalSourceIdentityPlan(
        local_binding=bindings[local_rank],
        source_bindings=bindings,
        publication_identity=TerminalPublicationIdentity(
            request_key=request_key,
            publisher_process_generation=sources[0].process_generation,
            publication_generation=b"p" * 16,
        ),
        request_ready_issuer=decoder,
        publisher_issuer=sources[0],
    )


def _manager_submission(
    identity: PackedTerminalSourceIdentityPlan,
    *,
    dflash: bool = False,
    non_prefill_projection: bool = False,
) -> tuple[PackedTerminalSourceSubmission, object | None]:
    """Build one concrete packed submission without touching CUDA.

    :param identity: Exact rank-local source identity.
    :param dflash: Whether the transport owns a DFlash device row.
    :param non_prefill_projection: Whether canonical output uses another schema.
    :returns: Complete submission and optional DFlash fixture lease.
    """

    transport = object.__new__(PackedPrefillSubmission)
    object.__setattr__(transport, "plan", object())
    object.__setattr__(transport, "destination", object())
    object.__setattr__(transport, "destination_registration", object())
    object.__setattr__(transport, "control", object())
    object.__setattr__(transport, "components", ())
    object.__setattr__(transport, "producer_event", object())
    dflash_lease: object | None = None
    if dflash:
        dflash_lease = object()
        source = object.__new__(DFlashBoundaryPrefillSource)
        object.__setattr__(source, "lease", dflash_lease)
        object.__setattr__(source, "counters", object())
        auxiliary: object = PackedTerminalDFlashAuxiliarySource(source)
    else:
        auxiliary = object()
    object.__setattr__(transport, "auxiliary_source", auxiliary)

    projection: FrozenTerminalGatewayOutputProjection | None = None
    if identity.local_binding.owner.tp_rank == 0:
        if non_prefill_projection:
            projection = _NonPrefillProjection(payload=b"another protocol")
        else:
            prefill_projection = object.__new__(PrefillTerminalGatewayOutputProjection)
            object.__setattr__(prefill_projection, "shell", object())
            object.__setattr__(prefill_projection, "result_slot", object())
            object.__setattr__(
                prefill_projection,
                "producer_event_generation",
                b"e" * 16,
            )
            projection = prefill_projection
    submission = PackedTerminalSourceSubmission(
        identity=identity,
        output_projection=projection,
        producer_event_generation=b"e" * 16,
        transport_submission=transport,
    )
    return submission, dflash_lease


def _manager_fixture(
    *,
    fault_stage: str | None = None,
) -> tuple[
    NixlKVManager,
    _ManagerReactor,
    _ManagerActor,
    _ManagerBindingRegistry,
    _ManagerServing,
    _ManagerBindingRegistry,
]:
    """Construct a real manager method boundary with explicit collaborators.

    :param fault_stage: Optional collaborator boundary which raises.
    :returns: Manager, reactor, actor, importer, serving, and publication registry.
    """

    reactor = _ManagerReactor(admission_open=fault_stage != "admission")
    actor = _ManagerActor(
        fail_bind=fault_stage == "actor",
        fail_prepare=fault_stage == "prepare",
    )
    importer = _ManagerBindingRegistry(fail=fault_stage == "importer")
    serving = _ManagerServing(
        fail_before_commit=fault_stage == "serving_before",
        fail_after_commit=fault_stage == "serving_after",
    )
    publication = _ManagerBindingRegistry(fail=fault_stage == "publication")
    manager = object.__new__(NixlKVManager)
    manager._terminal_process_reactor = reactor
    manager._terminal_startup_binding = SimpleNamespace(
        advertisement=SimpleNamespace(role=TerminalOwnerRole.SOURCE)
    )
    manager._terminal_source_serving = serving
    manager._packed_prefill_runtime = actor
    manager._terminal_source_receipt_importers = {}
    manager._terminal_source_publication_control = publication
    manager._terminal_unpublished_source_quarantine = {}
    manager._terminal_unpublished_source_quarantine_lock = threading.Lock()
    manager._terminal_dflash_source_owner = None
    manager.disaggregation_mode = DisaggregationMode.PREFILL
    manager._terminal_process_fatal_lock = threading.Lock()
    manager._terminal_process_fatal_reason = None
    manager._terminal_process_fatal_traceback = None
    manager._terminal_runtime_installation = None
    return manager, reactor, actor, importer, serving, publication


class _LeaseManager:
    """Record exact batch-scope source-row dispositions."""

    cancelled: list[object]
    quarantined: list[object]
    cancellations: list[tuple[object, str]]
    process_failures: list[tuple[str, str | None]]

    def __init__(self) -> None:
        """Create empty disposition ledgers."""

        self.cancelled = []
        self.quarantined = []
        self.cancellations = []
        self.process_failures = []

    def cancel_unpublished_terminal_dflash_source(self, source: object) -> None:
        """Record one safe pre-CUDA cancellation.

        :param source: Exact fixture lease.
        """

        self.cancelled.append(source)

    def quarantine_unpublished_terminal_source_submission(
        self,
        submission: object,
    ) -> None:
        """Record one complete CUDA-touched submission quarantine.

        :param submission: Exact fixture submission.
        """

        self.quarantined.append(submission)

    def cancel_terminal_source_request(
        self,
        binding: object,
        reason: str,
    ) -> PackedTerminalSourceCancellationDisposition:
        """Record completion-required client intent.

        :param binding: Exact fixture binding.
        :param reason: Stable client intent.
        :returns: Completion-required disposition.
        """

        self.cancellations.append((binding, reason))
        return PackedTerminalSourceCancellationDisposition.COMPLETION_REQUIRED

    def fail_terminal_source_process(
        self,
        reason: str,
        formatted_traceback: str | None,
    ) -> None:
        """Record one explicit fail-closed process disposition.

        :param reason: Stable failure reason.
        :param formatted_traceback: Optional originating traceback.
        """

        self.process_failures.append((reason, formatted_traceback))


@pytest.mark.parametrize(
    ("fault_stage", "lifecycle_committed"),
    (
        ("admission", False),
        ("actor", False),
        ("importer", False),
        ("serving_before", False),
        ("serving_after", True),
        ("publication", True),
        ("retention", True),
        ("prepare", True),
    ),
)
def test_real_manager_fault_boundary_retains_exact_owner_partition(
    fault_stage: str,
    lifecycle_committed: bool,
) -> None:
    """Every real manager bind failure retains exactly one lifetime authority."""

    identity = _manager_identity()
    submission, _ = _manager_submission(identity)
    manager, reactor, actor, importer, serving, publication = _manager_fixture(
        fault_stage=fault_stage
    )
    manager._terminal_source_receipt_importers = {
        identity.request_ready_issuer: importer
    }

    def commit_retention(retained: PackedTerminalSourceSubmission) -> None:
        """Commit or reject scheduler retention at the manager boundary.

        :param retained: Exact source submission entering scheduler retention.
        """

        assert retained is submission
        if fault_stage == "retention":
            raise RuntimeError("synthetic scheduler retention failure")

    with pytest.raises(RuntimeError, match="synthetic"):
        manager.bind_terminal_source_submission(
            submission,
            lambda retired: None,
            commit_retention,
        )

    digest = identity.local_binding.digest
    quarantine = manager._terminal_unpublished_source_quarantine
    if lifecycle_committed:
        assert quarantine == {}
        assert serving.active == {digest}
    else:
        assert quarantine == {digest: submission}
        assert serving.active == set()
        retained_projection = quarantine[digest].output_projection
        assert retained_projection is submission.output_projection
    assert len(actor.bind_calls) == (0 if fault_stage == "admission" else 1)
    assert len(importer.bindings) == (
        1
        if fault_stage
        in (
            "importer",
            "serving_before",
            "serving_after",
            "publication",
            "retention",
            "prepare",
        )
        else 0
    )
    assert len(publication.bindings) == (
        1 if fault_stage in ("publication", "retention", "prepare") else 0
    )
    assert len(actor.prepare_calls) == (1 if fault_stage == "prepare" else 0)
    assert manager._terminal_process_fatal_reason == (
        "terminal source bind failed after producer submission"
    )
    assert serving.abort_calls == 1
    assert reactor.stop_calls == (0 if fault_stage == "admission" else 1)


def test_real_manager_commits_scheduler_retention_before_prepare() -> None:
    """Native PREPARE cannot outrun scheduler-visible request ownership."""

    identity = _manager_identity()
    submission, _ = _manager_submission(identity)
    manager, _, actor, importer, _, _ = _manager_fixture()
    manager._terminal_source_receipt_importers = {
        identity.request_ready_issuer: importer
    }
    order: list[str] = []

    def commit_retention(retained: PackedTerminalSourceSubmission) -> None:
        """Record exact scheduler retention.

        :param retained: Exact manager-bound source submission.
        """

        assert retained is submission
        order.append("scheduler_retention")

    def publish_prepare(transport: PackedPrefillSubmission) -> None:
        """Reject PREPARE unless scheduler retention already committed.

        :param transport: Exact source transport entering native lifecycle.
        """

        assert transport is submission.transport_submission
        assert order == ["scheduler_retention"]
        order.append("prepare")

    actor.publish_terminal_owner_prepare = publish_prepare  # type: ignore[method-assign]

    manager.bind_terminal_source_submission(
        submission,
        lambda retired: None,
        commit_retention,
    )

    assert order == ["scheduler_retention", "prepare"]


def test_real_manager_rejects_nonprefill_projection_before_actor_publication() -> None:
    """Canonical output schema rejection precedes every distributed side effect."""

    identity = _manager_identity()
    submission, _ = _manager_submission(
        identity,
        non_prefill_projection=True,
    )
    manager, _, actor, importer, serving, publication = _manager_fixture()
    manager._terminal_source_receipt_importers = {
        identity.request_ready_issuer: importer
    }

    with pytest.raises(TypeError, match="pinned prefill output projection"):
        manager.bind_terminal_source_submission(
            submission,
            lambda retired: None,
            lambda retained: None,
        )

    assert actor.bind_calls == []
    assert importer.bindings == []
    assert serving.active == set()
    assert publication.bindings == []
    assert manager._terminal_unpublished_source_quarantine == {
        identity.local_binding.digest: submission
    }
    assert serving.abort_calls == 1


def test_real_manager_quarantines_dflash_submission_only_once() -> None:
    """Repeated fail-closed observation cannot double-quarantine one device row."""

    identity = _manager_identity()
    submission, dflash_lease = _manager_submission(identity, dflash=True)
    assert dflash_lease is not None
    manager, _, _, _, _, _ = _manager_fixture()
    dflash_owner = _DFlashQuarantineOwner()
    manager._terminal_dflash_source_owner = dflash_owner

    manager.quarantine_unpublished_terminal_source_submission(submission)
    manager.quarantine_unpublished_terminal_source_submission(submission)

    assert manager._terminal_unpublished_source_quarantine == {
        identity.local_binding.digest: submission
    }
    assert dflash_owner.leases == [dflash_lease]
    changed_submission = dataclasses.replace(
        submission,
        producer_event_generation=b"f" * 16,
    )
    with pytest.raises(RuntimeError, match="quarantine identity was reused"):
        manager.quarantine_unpublished_terminal_source_submission(changed_submission)
    assert dflash_owner.leases == [dflash_lease]


def test_batch_lease_failure_quarantines_cuda_and_cancels_untouched_once() -> None:
    """Never release CUDA-touched rows or leak later preleased rows."""

    manager = _LeaseManager()
    first = object()
    second = object()
    submission = object()
    ledger = _TerminalPrefillBatchLeaseLedger(manager)  # type: ignore[arg-type]
    ledger.retain(first)  # type: ignore[arg-type]
    ledger.retain(second)  # type: ignore[arg-type]
    ledger.begin_cuda(first, submission)  # type: ignore[arg-type]

    assert ledger.settle_after_failure() == ()
    assert manager.quarantined == [submission]
    assert manager.cancelled == [second]
    assert ledger.settle_after_failure() == ()
    assert manager.quarantined == [submission]
    assert manager.cancelled == [second]


def test_batch_lease_manager_handoff_excludes_lifecycle_owned_row() -> None:
    """Manager-owned rows cannot be cancelled by batch unwind."""

    manager = _LeaseManager()
    handed = object()
    untouched = object()
    submission = object()
    ledger = _TerminalPrefillBatchLeaseLedger(manager)  # type: ignore[arg-type]
    ledger.retain(handed)  # type: ignore[arg-type]
    ledger.retain(untouched)  # type: ignore[arg-type]
    ledger.begin_cuda(handed, submission)  # type: ignore[arg-type]
    ledger.hand_to_manager(handed)  # type: ignore[arg-type]

    assert ledger.settle_after_failure() == ()
    assert manager.quarantined == []
    assert manager.cancelled == [untouched]


def test_terminal_mode_skips_legacy_staging_and_waiting_collective() -> None:
    """Owner-driven requests never enter either legacy prefill hot path."""

    class _TerminalManager:
        def uses_terminal_source_publication(self) -> bool:
            return True

        def _prefetch_staging_reqs(self, room: int) -> None:
            raise AssertionError(f"legacy staging reached room {room}")

    manager = _TerminalManager()
    scheduler = SimpleNamespace(
        disagg_prefill_bootstrap_queue=SimpleNamespace(
            kv_manager=manager,
            terminal_source=True,
        ),
        waiting_queue=[object()],
    )
    batch = SimpleNamespace(requires_hidden_states=False, reqs=[object()])

    SchedulerDisaggregationPrefillMixin.maybe_prefetch_staging_for_batch(
        scheduler,
        batch,
    )
    SchedulerDisaggregationPrefillMixin.resolve_waiting_queue_bootstrap(scheduler)


def test_owner_managed_abort_records_intent_and_retains_completion_owners() -> None:
    """Post-PREPARE client intent cannot revoke or release published ownership."""

    manager = _LeaseManager()
    sender = SimpleNamespace(
        abort=lambda: pytest.fail("legacy sender abort must not run"),
        clear=lambda: pytest.fail("reclaim must not run during client abort"),
    )
    request = SimpleNamespace(rid="request-1", to_finish=None, disagg_kv_sender=sender)
    binding = object()
    scheduler = SimpleNamespace(
        disagg_prefill_terminal_requests={request.rid: request},
        disagg_prefill_terminal_bindings={request.rid: binding},
        disagg_prefill_bootstrap_queue=SimpleNamespace(kv_manager=manager),
    )

    matched = SchedulerDisaggregationPrefillMixin.abort_terminal_prefill_requests(
        scheduler,
        request.rid,
        False,
    )

    assert matched == 1
    assert request.to_finish is None
    assert scheduler.disagg_prefill_terminal_requests == {request.rid: request}
    assert scheduler.disagg_prefill_terminal_bindings == {request.rid: binding}
    assert manager.cancellations == [
        (binding, "client cancelled an owner-managed source request")
    ]
    assert manager.process_failures == []


@pytest.mark.parametrize(
    "loop_name",
    (
        "event_loop_normal_disagg_prefill",
        "event_loop_overlap_disagg_prefill",
    ),
)
def test_terminal_launch_freezes_before_every_model_submission(loop_name: str) -> None:
    """Eager and overlap loops freeze geometry before entering run_batch."""

    source = inspect.getsource(getattr(SchedulerDisaggregationPrefillMixin, loop_name))

    assert "run_terminal_prefill_batch(batch)" in source

    terminal_run = inspect.getsource(
        SchedulerDisaggregationPrefillMixin.run_terminal_prefill_batch
    )
    freeze = terminal_run.index("build_terminal_prefill_launches(batch)")
    submit = terminal_run.index("terminal_bind=terminal_bind")
    assert freeze < submit
    assert "functools.partial(" in terminal_run[freeze:submit]
    assert "self.bind_terminal_prefill_launches" in terminal_run[freeze:submit]


def test_terminal_bind_orders_event_record_before_native_publication() -> None:
    """The exact producing stream owns copies and event before owner bind."""

    source = inspect.getsource(
        SchedulerDisaggregationPrefillMixin.bind_terminal_prefill_launches
    )

    stream = source.index("torch.cuda.current_stream")
    projection = source.index("enqueue_terminal_dflash_source_projection")
    noncanonical_event = source.index("producer_event.record(stream)")
    transport = source.index("bind_producer_event")
    retention_callback = source.index("def commit_scheduler_retention(")
    native_bind = source.index("bind_terminal_source_submission")
    assert stream < transport < projection < retention_callback < native_bind
    assert stream < transport < noncanonical_event < retention_callback < native_bind
    assert ".tolist()" not in source
    assert "output_streamer" not in source
    assert "disagg_prefill_inflight_queue" not in source


def test_terminal_result_path_cannot_reenter_legacy_completion() -> None:
    """Owner-managed results have no legacy sender, streamer, or collective."""

    source = inspect.getsource(
        SchedulerDisaggregationPrefillMixin.process_batch_result_terminal_disagg_prefill
    )
    forbidden = (
        "send_kv_chunk",
        "output_streamer",
        "disagg_prefill_inflight_queue",
        "poll_and_all_reduce_attn_cp_tp_group",
        "TransferKVChunk",
        ".tolist()",
    )

    assert all(value not in source for value in forbidden)
    assert "disagg_prefill_terminal_requests" in source


def test_model_submit_failure_cancels_every_prelaunch_row_once() -> None:
    """A failed model submission never reaches bind or loses prelaunch leases."""

    manager = _LeaseManager()
    source = object()
    launch = object.__new__(_TerminalPrefillLaunch)
    object.__setattr__(launch, "dflash_source", source)
    bind_calls: list[object] = []

    def fail_model_submit(batch: object, *, terminal_bind: object) -> object:
        """Fail before the terminal bind callback is invoked.

        :param batch: Exact fixture batch.
        :param terminal_bind: Callback which must remain uncalled.
        :returns: This function never returns.
        """

        bind_calls.append(terminal_bind)
        raise RuntimeError("synthetic model submission failure")

    scheduler = SimpleNamespace(
        disagg_prefill_bootstrap_queue=SimpleNamespace(kv_manager=manager),
        build_terminal_prefill_launches=lambda batch: (launch,),
        bind_terminal_prefill_launches=lambda owner, result: pytest.fail(
            "terminal bind callback must not run"
        ),
        run_batch=fail_model_submit,
    )

    with pytest.raises(RuntimeError, match="synthetic model submission failure"):
        SchedulerDisaggregationPrefillMixin.run_terminal_prefill_batch(
            scheduler,
            object(),
        )
    assert len(bind_calls) == 1
    assert manager.cancelled == [source]


def test_legacy_sender_fails_closed_for_terminal_source() -> None:
    """An accidental legacy transfer is rejected before sender enqueue."""

    source = inspect.getsource(SchedulerDisaggregationPrefillMixin.send_kv_chunk)

    guard = source.index("uses_terminal_source_publication")
    enqueue = source.index("req.disagg_kv_sender.send")
    assert guard < enqueue
    assert "cannot enter the legacy transfer queue" in source[guard:enqueue]


def test_terminal_bootstrap_uses_no_legacy_metadata_row() -> None:
    """Terminal sender initialization carries no metadata-buffer index."""

    source = inspect.getsource(PrefillBootstrapQueue.finalize_bootstrap)

    assert "if not self.terminal_source" in source
    assert "None if self.terminal_source" in source
    assert "req.disagg_kv_sender.init(num_pages, metadata_index)" in source


def test_scheduler_reclaim_is_the_only_terminal_request_release() -> None:
    """The retained callback releases cache and registry exactly once."""

    source = inspect.getsource(
        SchedulerDisaggregationPrefillMixin.bind_terminal_prefill_launches
    )
    release_start = source.index("def release_resources(")
    release_end = source.index("if launch.dflash_source is not None:", release_start)
    release_body = source[release_start:release_end]

    assert release_body.count("release_kv_cache(retained_req, self.tree_cache)") == 1
    assert release_body.count("del self.disagg_prefill_terminal_requests") == 1
    assert "retired is not expected" in release_body
