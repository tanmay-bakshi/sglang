import concurrent.futures
import dataclasses
import hashlib
import logging
import os
import select
import sys
import threading
import time
from collections.abc import Callable

import pytest
from sglang.srt.disaggregation.common.packed_staging_protocol import PackedRequestKey
from sglang.srt.disaggregation.terminal_progress.evidence import (
    parse_terminal_progress_timing_log_line,
)
from sglang.srt.disaggregation.terminal_progress.grouped_nixl_owner import (
    GroupedNixlTerminalOwner,
    GroupedNixlTerminalOwnerInventory,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalPublicationIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.native_state import (
    NATIVE_SOURCE_RESOURCE_MASK,
    NativeSourceLifecyclePhase,
    NativeTerminalOwnerAction,
    NativeTerminalOwnerActionKind,
    NativeTerminalOwnerEvent,
    NativeTerminalOwnerEventKind,
    NativeTerminalOwnerFatalCode,
    NativeTerminalOwnerInventory,
    NativeTerminalOwnerOutput,
    NativeTerminalOwnerRole,
    NativeTerminalProcessIdentity,
    NativeTerminalProducerClass,
    NativeTerminalProducerRegistration,
    NativeTerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.nixl_adapter import (
    NixlTerminalBackendLifecycleInventory,
    NixlTerminalChannelFatal,
    NixlTerminalChannelInventory,
)
from sglang.srt.disaggregation.terminal_progress.publisher import (
    FrozenTerminalGatewayOutputProjection,
    FrozenTerminalGatewayPublication,
    TerminalGatewayPublicationSuccess,
)
from sglang.srt.disaggregation.terminal_progress.receipts import (
    TerminalReceiptKind,
    TerminalReceiptOutcome,
)
from sglang.srt.disaggregation.terminal_progress.runtime import (
    NativeTerminalDeliveryLeaseDisposition,
    NativeTerminalProducerDelivery,
    NativeTerminalRuntime,
    NativeTerminalRuntimeDisposition,
    NativeTerminalRuntimeProducerSpec,
)
from sglang.srt.disaggregation.terminal_progress.scheduler_inbox import (
    SchedulerReceiptPublishResult,
)
from sglang.srt.disaggregation.terminal_progress.scheduler_serving import (
    TerminalSchedulerActionPublicationError,
    TerminalSchedulerDeliveryLease,
)
from sglang.srt.disaggregation.terminal_progress.serving_reactor import (
    PackedTerminalProcessReactor,
    PackedTerminalProcessReactorFailure,
)
from sglang.srt.disaggregation.terminal_progress.source_gather_worker import (
    PackedTerminalSourceGatherWorkerDisposition,
)
from sglang.srt.disaggregation.terminal_progress.source_plan import (
    PackedTerminalSourceIdentityPlan,
)
from sglang.srt.disaggregation.terminal_progress.source_serving import (
    PackedTerminalSourceResourceInventory,
    PackedTerminalSourceServing,
    PackedTerminalSourceWork,
)
from sglang.srt.disaggregation.terminal_progress.source_wiring import (
    PackedTerminalSourceCancellationDisposition,
    PackedTerminalSourceMetric,
    PackedTerminalSourcePublicationRetentionError,
    PackedTerminalSourceQuarantineRetentionError,
    PackedTerminalSourceSubmission,
)
from sglang.srt.disaggregation.terminal_progress.wire import (
    IssuedTerminalWireReceipt,
    TerminalWireReceiptIssuer,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="native terminal runtime requires Linux eventfd and timerfd",
)

_LOCAL_PRODUCER_ID = 1
_LOCAL_RECEIPT_PRODUCER_ID = 2
_REMOTE_RECEIPT_PRODUCER_ID = 3
_REMOTE_CONTROL_PRODUCER_ID = 4
_NATIVE_PRODUCER_ID = 5
_WAIT_SECONDS = 5.0


@dataclasses.dataclass(frozen=True, slots=True)
class _Projection(FrozenTerminalGatewayOutputProjection):
    """Minimal immutable output projection fixture.

    :ivar payload: Exact bytes represented by this projection.
    """

    payload: bytes

    @property
    def digest(self) -> bytes:
        """Return the projection digest.

        :returns: SHA-256 over the exact payload.
        """

        return hashlib.sha256(self.payload).digest()


class _Metrics:
    """Record non-gating source metrics."""

    values: list[PackedTerminalSourceMetric]

    def __init__(self) -> None:
        """Create an empty metric ledger."""

        self.values = []

    def emit(self, metric: PackedTerminalSourceMetric) -> None:
        """Record one source metric.

        :param metric: Exact metric projection.
        """

        self.values.append(metric)


class _Publisher:
    """Record immutable gateway publications."""

    values: list[FrozenTerminalGatewayPublication]

    def __init__(self) -> None:
        """Create an empty publication ledger."""

        self.values = []

    def submit(self, publication: FrozenTerminalGatewayPublication) -> bool:
        """Accept one exact publication.

        :param publication: Immutable gateway publication.
        :returns: Always ``True`` for a new fixture value.
        """

        self.values.append(publication)
        return True


class _CudaCompletion:
    """Release source completion into native state after authorization."""

    _runtime: NativeTerminalRuntime

    def __init__(self, runtime: NativeTerminalRuntime) -> None:
        """Bind the fixture to its exact native runtime.

        :param runtime: Source runtime receiving callback completion.
        """

        self._runtime = runtime

    def arm(self, binding_digest: bytes) -> None:
        """Accept one source lifecycle.

        :param binding_digest: Exact source binding.
        """

    def submit(self, stream_handle: int, binding_digest: bytes) -> None:
        """Accept one stream-tail callback attachment.

        :param stream_handle: Exact source stream handle.
        :param binding_digest: Exact armed binding.
        """

    def authorize_delivery(self, binding_digest: bytes) -> bool:
        """Deliver the retained callback after decoder allocation.

        :param binding_digest: Exact armed source binding.
        :returns: ``True`` after direct native delivery.
        """

        self._runtime._owner.submit(
            NativeTerminalOwnerEvent(
                producer_id=_NATIVE_PRODUCER_ID,
                binding_digest=binding_digest,
                kind=NativeTerminalOwnerEventKind.SOURCE_PRODUCER_COMPLETED,
                enqueued_ns=1_000,
            )
        )
        return True


class _EmptyGroupedNixlOwner(GroupedNixlTerminalOwner):
    """Nominal empty grouped owner for source-composition tests."""

    _descriptor: int
    _admission_open: bool
    _closed: bool

    def __init__(self) -> None:
        """Create one readable-descriptor-compatible empty owner."""

        self._descriptor = os.eventfd(0, os.EFD_CLOEXEC | os.EFD_NONBLOCK)
        self._admission_open = True
        self._closed = False

    def fileno(self) -> int:
        """Return the borrowed process-reactor descriptor.

        :returns: Open nonblocking eventfd.
        """

        return self._descriptor

    def drain(self) -> tuple[object, ...]:
        """Return the permanently empty aggregate-result population.

        :returns: Empty result tuple.
        """

        return ()

    def stop_admission(self) -> None:
        """Permanently close fixture admission."""

        self._admission_open = False

    def inventory(self) -> GroupedNixlTerminalOwnerInventory:
        """Return exact-zero grouped and native inventory.

        :returns: Empty immutable inventory.
        """

        backend = NixlTerminalBackendLifecycleInventory(
            source_deliveries_outstanding=0,
            source_local_pending=0,
            source_receipt_pending=0,
            destination_pending=0,
            destination_admitting=0,
            destination_committed=0,
            destination_replaying=0,
            destination_quarantined=0,
            active_native_deadlines=0,
            source_deliveries=(),
            destination_deliveries=(),
            native_deadlines=(),
        )
        native = NixlTerminalChannelInventory(
            capacity=8,
            queued_channel_events=0,
            active_channel_subscriptions=0,
            retained_public_subscriptions=0,
            backend_producers=0,
            active_callback_slots=0,
            queued_owner_continuations=0,
            backend_lifecycle=backend,
            accepting_subscriptions=self._admission_open,
            closed=self._closed,
            fatal=NixlTerminalChannelFatal.NONE,
            eventfd_error=0,
        )
        return GroupedNixlTerminalOwnerInventory(
            native=native,
            admission_open=self._admission_open,
            closed=self._closed,
            active_group_count=0,
            sealed_group_count=0,
            pending_result_count=0,
            acknowledged_result_count=0,
            active_transfer_count=0,
            terminal_transfer_count=0,
            settled_transfer_count=0,
            quarantined_transfer_count=0,
            released_transfer_count=0,
            unowned_handle_count=0,
        )

    def close_clean(self) -> GroupedNixlTerminalOwnerInventory:
        """Close the exact-zero fixture owner once.

        :returns: Final clean inventory.
        """

        if self._closed:
            raise RuntimeError("empty grouped owner already closed")
        self._admission_open = False
        os.close(self._descriptor)
        self._closed = True
        return self.inventory()


def _identity(*, local_rank: int = 0) -> PackedTerminalSourceIdentityPlan:
    """Build one TP2 source and TP1 decode identity graph.

    :param local_rank: Source rank selected as local.
    :returns: Exact rank-local source plan.
    """

    key = PackedRequestKey(room_id=301, request_generation=b"g" * 16)
    sources = tuple(
        TerminalProcessIdentity(
            process_generation=bytes((0x40 + rank,)) * 16,
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
            request_key=key,
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
            request_key=key,
            publisher_process_generation=sources[0].process_generation,
            publication_generation=b"p" * 16,
        ),
        request_ready_issuer=decoder,
        publisher_issuer=sources[0],
    )


def _runtime(
    identity: PackedTerminalSourceIdentityPlan,
    *,
    enable_forward_independent_handoff: bool = False,
) -> NativeTerminalRuntime:
    """Construct one source runtime with every authority pre-registered.

    :param identity: Exact source identity graph.
    :param enable_forward_independent_handoff: Whether native action delivery
        acquires request-scoped scheduler exclusion.
    :returns: Dormant process-lifetime runtime.
    """

    owner = NativeTerminalProcessIdentity.from_identity(identity.local_binding.owner)
    remote = NativeTerminalProcessIdentity.from_identity(identity.request_ready_issuer)
    role = NativeTerminalOwnerRole.SOURCE
    specs = (
        NativeTerminalRuntimeProducerSpec(
            registration=NativeTerminalProducerRegistration(
                producer_id=_LOCAL_PRODUCER_ID,
                name="python-local",
                producer_class=NativeTerminalProducerClass.LOCAL,
                allowed_role=role,
                authenticated_issuer=None,
            ),
            delivery=NativeTerminalProducerDelivery.PYTHON,
        ),
        NativeTerminalRuntimeProducerSpec(
            registration=NativeTerminalProducerRegistration(
                producer_id=_LOCAL_RECEIPT_PRODUCER_ID,
                name="python-local-receipt",
                producer_class=NativeTerminalProducerClass.RECEIPT,
                allowed_role=role,
                authenticated_issuer=owner,
            ),
            delivery=NativeTerminalProducerDelivery.PYTHON,
        ),
        NativeTerminalRuntimeProducerSpec(
            registration=NativeTerminalProducerRegistration(
                producer_id=_REMOTE_RECEIPT_PRODUCER_ID,
                name="python-remote-receipt",
                producer_class=NativeTerminalProducerClass.RECEIPT,
                allowed_role=role,
                authenticated_issuer=remote,
            ),
            delivery=NativeTerminalProducerDelivery.PYTHON,
        ),
        NativeTerminalRuntimeProducerSpec(
            registration=NativeTerminalProducerRegistration(
                producer_id=_REMOTE_CONTROL_PRODUCER_ID,
                name="python-remote-control",
                producer_class=NativeTerminalProducerClass.CONTROL,
                allowed_role=role,
                authenticated_issuer=remote,
            ),
            delivery=NativeTerminalProducerDelivery.PYTHON,
        ),
        NativeTerminalRuntimeProducerSpec(
            registration=NativeTerminalProducerRegistration(
                producer_id=_NATIVE_PRODUCER_ID,
                name="native-terminal",
                producer_class=NativeTerminalProducerClass.LOCAL,
                allowed_role=role,
                authenticated_issuer=None,
            ),
            delivery=NativeTerminalProducerDelivery.NATIVE,
        ),
    )
    return NativeTerminalRuntime(
        owner_identity=owner,
        producer_specs=specs,
        fatal_producer_id=_LOCAL_PRODUCER_ID,
        input_capacity=64,
        output_capacity=64,
        maximum_live_lifecycles=8,
        scheduler_capacity=8,
        coordinator_capacity=8,
        lifecycle_capacity=8,
        source_gather_capacity=8,
        source_work_capacity=8,
        decode_scatter_capacity=8,
        decode_work_capacity=8,
        publisher_capacity=8,
        observation_capacity=64,
        enable_forward_independent_handoff=enable_forward_independent_handoff,
    )


def _submission(
    identity: PackedTerminalSourceIdentityPlan,
) -> PackedTerminalSourceSubmission:
    """Build one immutable source submission.

    :param identity: Exact source identity graph.
    :returns: Complete immutable handoff.
    """

    return PackedTerminalSourceSubmission(
        identity=identity,
        output_projection=(
            _Projection(payload=b"output")
            if identity.local_binding.owner.tp_rank == 0
            else None
        ),
        producer_event_generation=b"e" * 16,
        producer_stream_handle=19,
        transport_submission=("packed", identity.local_binding.digest),
    )


def _serving(
    identity: PackedTerminalSourceIdentityPlan,
    *,
    post_gather: (
        Callable[[PackedTerminalSourceSubmission, NativeTerminalOwnerAction], None]
        | None
    ) = None,
    quarantine: Callable[[NativeTerminalOwnerAction], None] | None = None,
    bind_gather_cuda_device: Callable[[], None] | None = None,
    enable_forward_independent_handoff: bool = False,
) -> tuple[
    PackedTerminalSourceServing,
    NativeTerminalRuntime,
    _Publisher,
    _Metrics,
    list[str],
    list[object],
]:
    """Construct one source composition and its observation ledgers.

    :param identity: Exact source identity graph.
    :param post_gather: Optional dedicated-worker gather callback.
    :param quarantine: Optional fail-closed resource-retention callback.
    :param bind_gather_cuda_device: Optional worker-thread device binder.
    :param enable_forward_independent_handoff: Whether to exercise the native
        scheduler-interrupt delivery path.
    :returns: Serving, runtime, publisher, metrics, work labels, and fatal inventories.
    """

    runtime = _runtime(
        identity,
        enable_forward_independent_handoff=enable_forward_independent_handoff,
    )
    publisher = _Publisher()
    metrics = _Metrics()
    work_labels: list[str] = []
    fatal_inventories: list[object] = []
    if post_gather is None:

        def post_gather(
            submission: PackedTerminalSourceSubmission,
            action: NativeTerminalOwnerAction,
        ) -> None:
            """Record one fixture gather action.

            :param submission: Exact source submission.
            :param action: Exact native gather action.
            """

            work_labels.append("gather")

    if bind_gather_cuda_device is None:

        def bind_gather_cuda_device() -> None:
            """Provide a GPU-free worker affinity boundary for CPU tests."""

    if quarantine is None:

        def quarantine(action: NativeTerminalOwnerAction) -> None:
            """Record one fixture quarantine action.

            :param action: Exact native fail-closed authority.
            """

            work_labels.append("quarantine")

    def resource_inventory() -> PackedTerminalSourceResourceInventory:
        """Return exact-zero external ownership for the isolated fixture.

        :returns: Empty actor, DFlash, and quarantine inventory.
        """

        return PackedTerminalSourceResourceInventory(
            actor_active_binding_digests=(),
            actor_quarantined_binding_digests=(),
            actor_waiting_for_ready_binding_digests=(),
            actor_main_handle_binding_digests=(),
            actor_auxiliary_handle_binding_digests=(),
            actor_lane_binding_digests=(),
            request_ready_import_binding_digests=(),
            publication_control_active_binding_digests=(),
            publication_control_terminal_binding_digests=(),
            source_transfer_info_room_ids=(),
            source_prefix_length_room_ids=(),
            source_prefetched_room_ids=(),
            source_prefetch_requested_room_ids=(),
            dflash_active_transfer_count=0,
            dflash_posted_transfer_count=0,
            dflash_settled_transfer_count=0,
            dflash_released_transfer_count=0,
            dflash_quarantined_transfer_count=0,
            dflash_unowned_native_handle_count=0,
            dflash_free_row_count=0,
            dflash_active_row_count=0,
            dflash_quarantined_row_count=0,
            unpublished_quarantined_binding_digests=(),
            unpublished_quarantined_result_slot_binding_digests=(),
        )

    serving = PackedTerminalSourceServing(
        runtime=runtime,
        cuda_completion=_CudaCompletion(runtime),
        local_identity=identity.local_binding.owner,
        publisher=publisher,
        metrics_sink=metrics,
        clock_ns=lambda: 1_000,
        physical_capacity=8,
        process_fatal_handler=fatal_inventories.append,
        grouped_nixl=_EmptyGroupedNixlOwner(),
        work=PackedTerminalSourceWork(
            post_gather=post_gather,
            send_outcomes=lambda submission, action: work_labels.append("outcome"),
            send_ack=lambda submission, action: work_labels.append("ack"),
            quarantine=quarantine,
            observe_output=lambda output: None,
        ),
        bind_gather_cuda_device=bind_gather_cuda_device,
        retire_native_producers=lambda: runtime._owner.retire_python_producer(
            _NATIVE_PRODUCER_ID
        ),
        resource_inventory=resource_inventory,
        retire_submission=lambda submission, action: None,
    )
    return serving, runtime, publisher, metrics, work_labels, fatal_inventories


def _pump(
    serving: PackedTerminalSourceServing,
    runtime: NativeTerminalRuntime,
) -> None:
    """Fence native output and drain every currently earned action.

    :param serving: Exact source composition.
    :param runtime: Its sole native runtime.
    """

    readable, _, _ = select.select(
        list(serving.runtime_filenos),
        [],
        [],
        _WAIT_SECONDS,
    )
    if len(readable) == 0:
        raise TimeoutError("source composition runtime inbox did not wake")
    serving.drain_runtime_actions()


def _wait_for_phase(predicate: Callable[[], bool], description: str) -> None:
    """Keep the main interpreter runnable until a reactor-owned phase lands.

    Yielding between observations lets the serving reactor and its workers own
    downstream progress without coupling the assertion to their conditions.

    :param predicate: Exact phase condition to observe.
    :param description: Stable assertion context when the phase misses its bound.
    """

    deadline = time.monotonic() + _WAIT_SECONDS
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0)
    assert predicate(), description


def _pause_functional_start(
    monkeypatch: pytest.MonkeyPatch,
    serving: PackedTerminalSourceServing,
) -> tuple[threading.Event, threading.Event]:
    """Pause immediately before the delivery registry linearizes side effects.

    :param monkeypatch: Test-owned attribute replacement fixture.
    :param serving: Exact source composition whose gate must pause.
    :returns: Entry and release events for one functional-start attempt.
    """

    entered = threading.Event()
    release = threading.Event()
    original = serving._delivery_leases.begin_functional_action

    def paused(action: NativeTerminalOwnerAction) -> bool:
        """Expose the post-abort-check, pre-functional-start interval.

        :param action: Exact action attempting functional admission.
        :returns: Authoritative registry disposition after release.
        """

        entered.set()
        if not release.wait(timeout=_WAIT_SECONDS):
            raise TimeoutError("functional-start test release was not delivered")
        return original(action)

    monkeypatch.setattr(
        serving._delivery_leases,
        "begin_functional_action",
        paused,
    )
    return entered, release


def _action(
    identity: PackedTerminalSourceIdentityPlan,
    *,
    action_id: int,
    kind: NativeTerminalOwnerActionKind,
) -> NativeTerminalOwnerAction:
    """Build one receipt-free native source action.

    :param identity: Exact source identity graph.
    :param action_id: Unique action identifier.
    :param kind: Receipt-free source work kind.
    :returns: Immutable native owner action.
    """

    return NativeTerminalOwnerAction(
        action_id=action_id,
        kind=kind,
        binding=NativeTerminalRequestBinding.from_binding(identity.local_binding),
        commit_timestamp_ns=1_000 + action_id,
        receipt=None,
    )


def _output(
    action: NativeTerminalOwnerAction,
    *,
    owner_sequence: int,
) -> NativeTerminalOwnerOutput:
    """Wrap one synthetic source action in an immutable native output.

    :param action: Exact action carried by the output.
    :param owner_sequence: Positive process-local output sequence.
    :returns: Structurally valid source output for projection-boundary tests.
    """

    return NativeTerminalOwnerOutput(
        binding=action.binding,
        owner_sequence=owner_sequence,
        producer_id=_NATIVE_PRODUCER_ID,
        producer_sequence=owner_sequence,
        event_kind=NativeTerminalOwnerEventKind.SOURCE_PRODUCER_COMPLETED,
        enqueued_ns=action.commit_timestamp_ns,
        completed_ns=action.commit_timestamp_ns,
        role=NativeTerminalOwnerRole.SOURCE,
        previous_phase=int(NativeSourceLifecyclePhase.WAITING_FOR_PRODUCER),
        phase=int(NativeSourceLifecyclePhase.GATHERING),
        live_resources=NATIVE_SOURCE_RESOURCE_MASK,
        retired_resources=0,
        quarantined_resources=0,
        actions=(action,),
        armed_deadline_mask=0,
        process_fatal=False,
        fatal_code=NativeTerminalOwnerFatalCode.NONE,
    )


def _advance_source_to_request_ready_join(
    serving: PackedTerminalSourceServing,
    runtime: NativeTerminalRuntime,
    identity: PackedTerminalSourceIdentityPlan,
) -> None:
    """Advance one source lifecycle through its teardown ACK.

    :param serving: Started source composition.
    :param runtime: Exact native runtime owned by the composition.
    :param identity: Rank-local request identity.
    """

    submission = _submission(identity)
    serving.bind_submission(submission, lambda submission: None)
    serving.attach_producer_completion(submission)
    digest = identity.local_binding.digest
    assert serving.packed_ready(digest)
    assert runtime.wait_for_output_projection(_WAIT_SECONDS)
    assert serving._gather_worker.wait_until_idle(_WAIT_SECONDS)
    _pump(serving, runtime)
    runtime._owner.submit(
        NativeTerminalOwnerEvent(
            producer_id=_NATIVE_PRODUCER_ID,
            binding_digest=digest,
            kind=NativeTerminalOwnerEventKind.SOURCE_NATIVE_TERMINAL,
            enqueued_ns=2_000,
        )
    )
    _pump(serving, runtime)
    serving.wiring.teardown_received(digest, identity.request_ready_issuer)
    _pump(serving, runtime)


def _request_ready_receipt(
    identity: PackedTerminalSourceIdentityPlan,
) -> IssuedTerminalWireReceipt:
    """Issue one authenticated request-global readiness receipt.

    :param identity: Rank-local request identity.
    :returns: Issuer-owned paired wire and local receipt authority.
    """

    return TerminalWireReceiptIssuer(identity.request_ready_issuer).issue(
        binding=identity.local_binding,
        kind=TerminalReceiptKind.REQUEST_READY,
        outcome=TerminalReceiptOutcome.SUCCESS,
        terminal_timestamp_ns=3_000,
    )


def test_delivery_join_is_order_independent_and_late_ack_does_not_reacquire() -> None:
    """The two durable milestones release once and teardown cannot resurrect it."""

    identity = _identity()
    serving, _, _, _, _, _ = _serving(
        identity,
        enable_forward_independent_handoff=True,
    )
    serving.start()
    try:
        submission = _submission(identity)
        serving.bind_submission(submission, lambda value: None)
        gather = _action(
            identity,
            action_id=80,
            kind=NativeTerminalOwnerActionKind.SOURCE_GATHER_READY,
        )
        serving._delivery_leases.acquire_for_actions((gather,))
        serving._delivery_leases.mark_publication_owned(identity.local_binding)
        inventory = serving.inventory()
        assert inventory.delivery_leases.active_binding_digests == (
            identity.local_binding.digest,
        )
        assert inventory.scheduler_serving.inbox.active_delivery_intents != ()

        serving._delivery_leases.mark_outcomes_sent(identity.local_binding)
        inventory = serving.inventory()
        assert inventory.delivery_leases.active_binding_digests == ()
        assert inventory.scheduler_serving.inbox.active_delivery_intents == ()

        late_ack = _action(
            identity,
            action_id=81,
            kind=NativeTerminalOwnerActionKind.SOURCE_ACK_READY,
        )
        serving._delivery_leases.acquire_for_actions((late_ack,))
        assert serving.inventory().delivery_leases.active_binding_digests == ()
    finally:
        serving.abort_and_close()


def test_delivery_batch_rejects_generation_alias_before_allocating_intent() -> None:
    """Conflicting bindings cannot split one request generation across leases."""

    identity = _identity()
    serving, _, _, _, _, _ = _serving(
        identity,
        enable_forward_independent_handoff=True,
    )
    serving.start()
    try:
        submission = _submission(identity)
        serving.bind_submission(submission, lambda value: None)
        first = _action(
            identity,
            action_id=82,
            kind=NativeTerminalOwnerActionKind.SOURCE_GATHER_READY,
        )
        conflicting_binding = dataclasses.replace(
            identity.local_binding,
            allocation_digest=b"x" * 32,
        )
        conflicting = dataclasses.replace(
            first,
            action_id=83,
            binding=NativeTerminalRequestBinding.from_binding(conflicting_binding),
        )

        with pytest.raises(RuntimeError, match="aliases a request generation"):
            serving._delivery_leases.acquire_for_actions((first, conflicting))

        inventory = serving.inventory()
        assert inventory.delivery_leases.active_binding_digests == ()
        assert inventory.scheduler_serving.inbox.active_delivery_intents == ()
        with pytest.raises(RuntimeError, match="process-fatal"):
            serving._delivery_leases.acquire_for_actions((first,))
        assert serving.inventory().scheduler_serving.inbox.active_delivery_intents == ()
        serving.begin_fail_closed_abort()
        inventory = serving.inventory()
        assert inventory.delivery_leases.active_binding_digests == ()
        assert inventory.scheduler_serving.inbox.active_delivery_intents == ()
    finally:
        serving.abort_and_close()


def test_scheduler_fatal_delivery_state_admits_only_lease_free_abort_drain() -> None:
    """Sticky scheduler death turns every later handoff into abort ownership."""

    identity = _identity()
    serving, _, _, _, _, _ = _serving(
        identity,
        enable_forward_independent_handoff=True,
    )
    serving.start()
    try:
        submission = _submission(identity)
        serving.bind_submission(submission, lambda value: None)
        gather = _action(
            identity,
            action_id=86,
            kind=NativeTerminalOwnerActionKind.SOURCE_GATHER_READY,
        )
        assert serving._delivery_leases.acquire_for_actions((gather,)) is (
            NativeTerminalDeliveryLeaseDisposition.ACQUIRED
        )
        serving.begin_fail_closed_abort()

        for action_id, kind in (
            (87, NativeTerminalOwnerActionKind.SOURCE_OUTCOME_READY),
            (88, NativeTerminalOwnerActionKind.GATEWAY_PUBLICATION_READY),
            (89, NativeTerminalOwnerActionKind.SOURCE_ACK_READY),
        ):
            action = _action(identity, action_id=action_id, kind=kind)
            assert serving._delivery_leases.acquire_for_actions((action,)) is (
                NativeTerminalDeliveryLeaseDisposition.FAIL_CLOSED_DRAIN
            )
        inventory = serving.inventory()
        assert inventory.delivery_leases.active_binding_digests == ()
        assert inventory.scheduler_serving.inbox.active_delivery_intents == ()
    finally:
        serving.abort_and_close()


def test_composed_source_fatal_is_valid_after_binding_before_start() -> None:
    """A publisher startup failure can abort a bound dormant composition."""

    identity = _identity()
    serving, runtime, _, _, _, _ = _serving(
        identity,
        enable_forward_independent_handoff=True,
    )

    serving.begin_fail_closed_abort()

    inventory = serving.inventory()
    assert runtime.disposition is NativeTerminalRuntimeDisposition.ABORT_DRAINING
    assert inventory.owner_dead_marked
    assert inventory.scheduler_serving.inbox.fatal_cause is not None
    assert inventory.delivery_leases.active_binding_digests == ()
    closed = serving.abort_and_close()
    assert runtime.disposition is NativeTerminalRuntimeDisposition.STOPPED
    assert (
        closed.gather_worker.disposition
        is PackedTerminalSourceGatherWorkerDisposition.STOPPED
    )
    assert not closed.gather_worker.thread_alive
    with pytest.raises(RuntimeError, match="cannot restart"):
        serving.start()


def test_scheduler_fatal_reactor_drains_causal_actions_without_side_effects() -> None:
    """Outcome and publication actions reconcile after scheduler death."""

    identity = _identity()
    serving, runtime, publisher, _, work_labels, _ = _serving(identity)
    serving.start()
    try:
        serving.begin_fail_closed_abort()
        actions = (
            _action(
                identity,
                action_id=90,
                kind=NativeTerminalOwnerActionKind.SOURCE_OUTCOME_READY,
            ),
            _action(
                identity,
                action_id=91,
                kind=NativeTerminalOwnerActionKind.GATEWAY_PUBLICATION_READY,
            ),
        )
        for action in actions:
            runtime._route_action(action)

        assert serving.drain_runtime_actions() == len(actions)
        snapshot = runtime.snapshot()
        assert snapshot.fatal_reason is not None
        assert snapshot.consumer_pending_count == 0
        assert snapshot.source_work.queued_count == 0
        assert snapshot.publisher.queued_count == 0
        assert work_labels == []
        assert publisher.values == []
    finally:
        serving.abort_and_close()


def test_scheduler_fatal_gather_drain_never_executes_functional_work() -> None:
    """A worker observes runtime fatality before consuming a claimed gather."""

    identity = _identity()
    serving, runtime, _, _, work_labels, _ = _serving(identity)
    serving.start()
    try:
        serving.begin_fail_closed_abort()
        actions = tuple(
            _action(
                identity,
                action_id=action_id,
                kind=NativeTerminalOwnerActionKind.SOURCE_GATHER_READY,
            )
            for action_id in (92, 93)
        )
        for action in actions:
            runtime._route_action(action)

        assert serving._gather_worker.wait_until_idle(_WAIT_SECONDS)
        snapshot = runtime.snapshot()
        assert snapshot.fatal_reason is not None
        assert snapshot.consumer_pending_count == 0
        assert snapshot.source_gather.queued_count == 0
        assert work_labels == []
        assert serving.inventory().gather_worker.aborted_action_count == len(actions)
    finally:
        serving.abort_and_close()


def test_acquired_delivery_racing_source_fatal_cannot_execute_functional_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime abort wins even when scheduler death follows lease acquisition."""

    identity = _identity()
    serving, runtime, _, _, work_labels, _ = _serving(
        identity,
        enable_forward_independent_handoff=True,
    )
    acquisition_returning = threading.Event()
    release_acquisition = threading.Event()
    original_acquirer = serving._delivery_leases.acquire_for_actions

    def pause_acquired_delivery(
        actions: tuple[NativeTerminalOwnerAction, ...],
    ) -> NativeTerminalDeliveryLeaseDisposition:
        """Pause after the registry linearizes an acquired delivery lease.

        :param actions: Exact action population crossing native handoff.
        :returns: The acquired disposition captured before source fatality.
        """

        disposition = original_acquirer(actions)
        assert disposition is NativeTerminalDeliveryLeaseDisposition.ACQUIRED
        acquisition_returning.set()
        if not release_acquisition.wait(timeout=_WAIT_SECONDS):
            raise TimeoutError("acquired delivery was not released by the test")
        return disposition

    monkeypatch.setattr(
        serving._delivery_leases,
        "acquire_for_actions",
        pause_acquired_delivery,
    )
    reactor_failures: list[PackedTerminalProcessReactorFailure] = []
    reactor = PackedTerminalProcessReactor.for_source(
        serving,
        reactor_failures.append,
    )
    reactor_started = False
    serving.start()
    try:
        reactor.start(_WAIT_SECONDS)
        reactor_started = True
        submission = _submission(identity)
        serving.bind_submission(submission, lambda value: None)
        serving.attach_producer_completion(submission)
        assert serving.packed_ready(identity.local_binding.digest)
        assert acquisition_returning.wait(timeout=_WAIT_SECONDS)

        serving.begin_fail_closed_abort()
        release_acquisition.set()

        assert runtime.wait_for_output_projection(_WAIT_SECONDS)
        assert serving._gather_worker.wait_until_idle(_WAIT_SECONDS)
        _wait_for_phase(
            lambda: runtime.snapshot().consumer_pending_count == 0,
            "source abort population did not drain",
        )
        snapshot = runtime.snapshot()
        assert snapshot.disposition is NativeTerminalRuntimeDisposition.ABORT_DRAINING
        assert snapshot.consumer_pending_count == 0
        assert "gather" not in work_labels
        assert reactor_failures == []
        inventory = serving.inventory()
        assert inventory.delivery_leases.active_binding_digests == ()
        assert inventory.scheduler_serving.inbox.active_delivery_intents == ()
    finally:
        release_acquisition.set()
        if reactor_started:
            reactor.close(_WAIT_SECONDS)
        serving.abort_and_close()


def test_source_fatal_before_delivery_acquisition_allocates_no_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime fatality closes delivery admission before lease allocation."""

    identity = _identity()
    serving, runtime, _, _, work_labels, _ = _serving(
        identity,
        enable_forward_independent_handoff=True,
    )
    acquisition_entered = threading.Event()
    release_acquisition = threading.Event()
    begin_delivery_bindings: list[TerminalRequestBinding] = []
    original_acquirer = serving._delivery_leases.acquire_for_actions
    original_begin_delivery = serving._scheduler_serving.begin_delivery_lease

    def pause_before_acquisition(
        actions: tuple[NativeTerminalOwnerAction, ...],
    ) -> NativeTerminalDeliveryLeaseDisposition:
        """Expose the runtime-fatal to authority-acquisition ordering.

        :param actions: Exact action population crossing native handoff.
        :returns: Delivery disposition established after the race release.
        """

        acquisition_entered.set()
        if not release_acquisition.wait(timeout=_WAIT_SECONDS):
            raise TimeoutError("source delivery acquisition was not released")
        return original_acquirer(actions)

    def record_begin_delivery(
        binding: TerminalRequestBinding,
    ) -> TerminalSchedulerDeliveryLease:
        """Record any forbidden scheduler lease allocation.

        :param binding: Request generation requesting a delivery lease.
        :returns: Newly allocated delivery lease.
        """

        begin_delivery_bindings.append(binding)
        return original_begin_delivery(binding)

    monkeypatch.setattr(
        serving._delivery_leases,
        "acquire_for_actions",
        pause_before_acquisition,
    )
    monkeypatch.setattr(
        serving._scheduler_serving,
        "begin_delivery_lease",
        record_begin_delivery,
    )
    serving.start()
    closed = False
    try:
        submission = _submission(identity)
        serving.bind_submission(submission, lambda value: None)
        serving.attach_producer_completion(submission)
        assert serving.packed_ready(identity.local_binding.digest)
        assert acquisition_entered.wait(timeout=_WAIT_SECONDS)

        with runtime._condition:
            assert runtime._disposition is NativeTerminalRuntimeDisposition.RUNNING
            runtime._enter_runtime_fatal_locked(
                "synthetic runtime fatal before source delivery acquisition"
            )
        assert serving.inventory().scheduler_serving.inbox.fatal_cause is None

        release_acquisition.set()
        assert runtime.wait_for_output_projection(_WAIT_SECONDS)
        assert serving._gather_worker.wait_until_idle(_WAIT_SECONDS)

        final_inventory = serving.abort_and_close()
        closed = True
        assert begin_delivery_bindings == []
        assert work_labels == ["quarantine"]
        assert final_inventory.runtime.owner.source_batch_handoff_count == 0
        assert final_inventory.runtime.consumer_pending_count == 0
        assert final_inventory.runtime.source_preclaimed_count == 0
        assert final_inventory.runtime.source_preclaimed_consumer_count == 0
        assert final_inventory.runtime.decode_publication_preclaimed_count == 0
        assert final_inventory.delivery_leases.active_binding_digests == ()
        assert final_inventory.scheduler_serving.inbox.active_delivery_intents == ()
    finally:
        release_acquisition.set()
        if not closed:
            serving.abort_and_close()


def test_functional_start_before_fatal_retains_delivery_until_callback_return() -> None:
    """An admitted side effect pins its causal lease across scheduler death."""

    identity = _identity()
    serving, _, _, _, _, _ = _serving(
        identity,
        enable_forward_independent_handoff=True,
    )
    action = _action(
        identity,
        action_id=95,
        kind=NativeTerminalOwnerActionKind.SOURCE_GATHER_READY,
    )
    serving.start()
    try:
        assert serving._delivery_leases.acquire_for_actions((action,)) is (
            NativeTerminalDeliveryLeaseDisposition.ACQUIRED
        )
        assert serving._delivery_leases.begin_functional_action(action)

        serving.begin_fail_closed_abort()

        retained = serving.inventory()
        assert retained.delivery_leases.active_binding_digests == (
            identity.local_binding.digest,
        )
        assert retained.delivery_leases.active_functional_actions == (action,)
        assert retained.scheduler_serving.inbox.active_delivery_intents != ()

        serving._delivery_leases.finish_functional_action(action)
        released = serving.inventory()
        assert released.delivery_leases.active_binding_digests == ()
        assert released.delivery_leases.active_functional_actions == ()
        assert released.scheduler_serving.inbox.active_delivery_intents == ()
    finally:
        serving.abort_and_close()


def test_gather_fatal_winning_final_start_gate_aborts_without_side_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gather cannot cross functional start after scheduler fatal wins."""

    identity = _identity()
    serving, runtime, _, _, work_labels, _ = _serving(
        identity,
        enable_forward_independent_handoff=True,
    )
    entered, release = _pause_functional_start(monkeypatch, serving)
    serving.start()
    try:
        submission = _submission(identity)
        serving.bind_submission(submission, lambda value: None)
        serving.attach_producer_completion(submission)
        assert serving.packed_ready(identity.local_binding.digest)
        _wait_for_phase(entered.is_set, "gather did not reach functional admission")

        serving.begin_fail_closed_abort()
        assert serving.inventory().delivery_leases.active_binding_digests == ()
        release.set()

        assert serving._gather_worker.wait_until_idle(_WAIT_SECONDS)
        inventory = serving.inventory()
        assert work_labels == []
        assert inventory.gather_worker.completed_action_count == 0
        assert inventory.gather_worker.aborted_action_count == 1
        assert inventory.delivery_leases.active_functional_actions == ()
    finally:
        release.set()
        serving.abort_and_close()


@pytest.mark.parametrize(
    ("kind", "expected_label"),
    (
        (NativeTerminalOwnerActionKind.SOURCE_OUTCOME_READY, "outcome"),
        (NativeTerminalOwnerActionKind.SOURCE_ACK_READY, "ack"),
    ),
)
def test_source_work_fatal_winning_final_start_gate_aborts_without_side_effect(
    monkeypatch: pytest.MonkeyPatch,
    kind: NativeTerminalOwnerActionKind,
    expected_label: str,
) -> None:
    """Neither source-work callback begins after scheduler fatal wins."""

    identity = _identity()
    serving, runtime, _, _, work_labels, _ = _serving(
        identity,
        enable_forward_independent_handoff=True,
    )
    release = threading.Event()
    serving.start()
    try:
        submission = _submission(identity)
        serving.bind_submission(submission, lambda value: None)
        serving.attach_producer_completion(submission)
        digest = identity.local_binding.digest
        assert serving.packed_ready(digest)
        _wait_for_phase(
            lambda: work_labels == ["gather"],
            "initial gather did not complete",
        )
        assert serving._gather_worker.wait_until_idle(_WAIT_SECONDS)
        assert work_labels == ["gather"]

        if kind is NativeTerminalOwnerActionKind.SOURCE_ACK_READY:
            runtime._owner.submit(
                NativeTerminalOwnerEvent(
                    producer_id=_NATIVE_PRODUCER_ID,
                    binding_digest=digest,
                    kind=NativeTerminalOwnerEventKind.SOURCE_NATIVE_TERMINAL,
                    enqueued_ns=2_000,
                )
            )
            _wait_for_phase(
                lambda: runtime.source_work_actions.snapshot().queued_count == 1,
                "outcome action did not reach its inbox",
            )
            serving.drain_runtime_actions()
            assert work_labels == ["gather", "outcome"]

        entered, release = _pause_functional_start(monkeypatch, serving)
        if kind is NativeTerminalOwnerActionKind.SOURCE_OUTCOME_READY:
            runtime._owner.submit(
                NativeTerminalOwnerEvent(
                    producer_id=_NATIVE_PRODUCER_ID,
                    binding_digest=digest,
                    kind=NativeTerminalOwnerEventKind.SOURCE_NATIVE_TERMINAL,
                    enqueued_ns=2_000,
                )
            )
        else:
            serving.wiring.teardown_received(digest, identity.request_ready_issuer)
        _wait_for_phase(
            lambda: runtime.source_work_actions.snapshot().queued_count == 1,
            "target source-work action did not reach its inbox",
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            drain = executor.submit(serving.drain_runtime_actions)
            assert entered.wait(timeout=_WAIT_SECONDS)

            serving.begin_fail_closed_abort()
            release.set()

            assert drain.result(timeout=_WAIT_SECONDS) >= 1
        inventory = serving.inventory()
        assert expected_label not in work_labels
        assert inventory.delivery_leases.active_binding_digests == ()
        assert inventory.delivery_leases.active_functional_actions == ()
    finally:
        release.set()
        serving.abort_and_close()


def test_publisher_fatal_winning_final_start_gate_aborts_without_side_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publisher submission cannot begin after scheduler fatal wins."""

    identity = _identity()
    serving, runtime, publisher, _, work_labels, _ = _serving(
        identity,
        enable_forward_independent_handoff=True,
    )
    release = threading.Event()
    publication_calls: list[NativeTerminalOwnerAction] = []
    serving.start()
    try:
        submission = _submission(identity)
        serving.bind_submission(submission, lambda value: None)
        serving.attach_producer_completion(submission)
        digest = identity.local_binding.digest
        assert serving.packed_ready(digest)
        _wait_for_phase(
            lambda: work_labels == ["gather"],
            "publisher setup gather did not complete",
        )
        runtime._owner.submit(
            NativeTerminalOwnerEvent(
                producer_id=_NATIVE_PRODUCER_ID,
                binding_digest=digest,
                kind=NativeTerminalOwnerEventKind.SOURCE_NATIVE_TERMINAL,
                enqueued_ns=2_000,
            )
        )
        _wait_for_phase(
            lambda: runtime.source_work_actions.snapshot().queued_count == 1,
            "publisher setup outcome did not reach its inbox",
        )
        serving.drain_runtime_actions()
        serving.wiring.teardown_received(digest, identity.request_ready_issuer)
        _wait_for_phase(
            lambda: runtime.source_work_actions.snapshot().queued_count == 1,
            "publisher setup acknowledgement did not reach its inbox",
        )
        serving.drain_runtime_actions()
        assert work_labels == ["gather", "outcome", "ack"]
        entered, release = _pause_functional_start(monkeypatch, serving)
        monkeypatch.setattr(
            serving._wiring,
            "consume_gateway_publication_ready",
            publication_calls.append,
        )
        ready = _request_ready_receipt(identity)
        serving.request_ready(
            binding_digest=identity.local_binding.digest,
            wire_receipt=ready.wire_receipt,
            local_receipt=ready.local_receipt,
            authenticated_issuer=identity.request_ready_issuer,
        )
        _wait_for_phase(
            lambda: runtime.publisher_actions.snapshot().queued_count == 1,
            "gateway publication did not reach its inbox",
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            drain = executor.submit(serving.drain_runtime_actions)
            assert entered.wait(timeout=_WAIT_SECONDS)

            serving.begin_fail_closed_abort()
            release.set()

            assert drain.result(timeout=_WAIT_SECONDS) >= 1
        inventory = serving.inventory()
        assert publication_calls == []
        assert publisher.values == []
        assert inventory.delivery_leases.active_binding_digests == ()
        assert inventory.delivery_leases.active_functional_actions == ()
    finally:
        release.set()
        serving.abort_and_close()


def test_delivery_batch_allocation_retains_launch_exclusion_until_owner_death(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial batch failure blocks another launch through owner death."""

    identity = _identity()
    serving, runtime, _, _, _, _ = _serving(
        identity,
        enable_forward_independent_handoff=True,
    )
    serving.start()
    second_binding = dataclasses.replace(
        identity.local_binding,
        request_key=PackedRequestKey(
            room_id=302,
            request_generation=b"h" * 16,
        ),
        allocation_digest=b"b" * 32,
    )
    try:
        submission = _submission(identity)
        serving.bind_submission(submission, lambda value: None)
        first = _action(
            identity,
            action_id=84,
            kind=NativeTerminalOwnerActionKind.SOURCE_GATHER_READY,
        )
        second = dataclasses.replace(
            first,
            action_id=85,
            binding=NativeTerminalRequestBinding.from_binding(second_binding),
        )
        outputs = (
            _output(first, owner_sequence=1),
            _output(second, owner_sequence=2),
        )
        original_begin = serving._scheduler_serving.begin_delivery_lease
        allocation_count = 0
        native_claims: list[tuple[NativeTerminalOwnerAction, ...]] = []
        rejected_actions: list[NativeTerminalOwnerAction] = []

        def fail_second(
            binding: TerminalRequestBinding,
        ) -> TerminalSchedulerDeliveryLease:
            """Fail after one real scheduler intent has been allocated.

            :param binding: Exact request selected for allocation.
            :returns: First request's real delivery lease.
            """

            nonlocal allocation_count
            allocation_count += 1
            if allocation_count == 2:
                raise RuntimeError("synthetic second allocation failure")
            return original_begin(binding)

        monkeypatch.setattr(
            serving._scheduler_serving,
            "begin_delivery_lease",
            fail_second,
        )
        monkeypatch.setattr(
            runtime._owner,
            "claim_source_forward_independent_handoffs",
            native_claims.append,
        )
        monkeypatch.setattr(
            runtime._owner,
            "fail_action_delivery",
            lambda action, reason: rejected_actions.append(action),
        )
        with pytest.raises(RuntimeError, match="batch allocation failed") as raised:
            runtime._prepare_output_batch(outputs)

        assert allocation_count == 2
        assert native_claims == []
        inventory = serving.inventory()
        assert inventory.delivery_leases.active_binding_digests == (
            identity.local_binding.digest,
        )
        assert len(inventory.scheduler_serving.inbox.active_delivery_intents) == 1
        assert tuple(serving._delivery_leases._records) == (
            identity.local_binding.request_key,
        )

        launch_submitted = threading.Event()

        def submit() -> str:
            """Record any host launch which crosses the retained intent.

            :returns: Synthetic launch result.
            """

            launch_submitted.set()
            return "submitted"

        handoff_waiting = threading.Event()
        original_wait = serving._scheduler_serving._inbox._wait_for_publication_intents

        def signal_handoff_wait(intents: frozenset[int]) -> None:
            """Expose the scheduler's wait on the retained delivery intent.

            :param intents: Exact intent population blocking host submission.
            """

            handoff_waiting.set()
            original_wait(intents)

        monkeypatch.setattr(
            serving._scheduler_serving._inbox,
            "_wait_for_publication_intents",
            signal_handoff_wait,
        )

        launch_blocked_before_rejection = False
        launch_blocked_after_rejection = False

        def enter_owner_death() -> None:
            """Reject the batch and make scheduler owner death durable."""

            nonlocal launch_blocked_before_rejection
            nonlocal launch_blocked_after_rejection
            if not handoff_waiting.wait(timeout=_WAIT_SECONDS):
                serving.begin_fail_closed_abort()
                raise TimeoutError("scheduler did not wait on the delivery intent")
            launch_blocked_before_rejection = not launch_submitted.is_set()
            runtime._reject_output_batch(
                outputs,
                str(raised.value),
            )
            launch_blocked_after_rejection = not launch_submitted.is_set()
            serving.drain_runtime_actions()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            owner_death = executor.submit(enter_owner_death)
            with pytest.raises(RuntimeError, match="process-fatal"):
                serving._scheduler_serving.launch_handoff(submit)
            owner_death.result(timeout=_WAIT_SECONDS)

        assert rejected_actions == [first, second]
        assert launch_blocked_before_rejection
        assert launch_blocked_after_rejection
        assert not launch_submitted.is_set()
        inventory = serving.inventory()
        assert inventory.owner_dead_marked
        assert inventory.delivery_leases.active_binding_digests == ()
        assert inventory.scheduler_serving.inbox.active_delivery_intents == ()
        snapshot = runtime.snapshot()
        assert snapshot.source_preclaimed_count == 0
        assert snapshot.source_preclaimed_consumer_count == 0
    finally:
        serving.abort_and_close()


def test_native_batch_handoff_progresses_while_scheduler_waits_on_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Source delivery breaks the attempt-29 scheduler-wait cycle."""

    identity = _identity()
    leases_acquired = threading.Event()
    release_acquisition = threading.Event()
    scheduler_waiting = threading.Event()
    launch_submitted = threading.Event()
    gather_entered = threading.Event()
    release_gather = threading.Event()

    def post_gather(
        submission: PackedTerminalSourceSubmission,
        action: NativeTerminalOwnerAction,
    ) -> None:
        """Park after claim so the lock-independent lease remains observable.

        :param submission: Exact source submission.
        :param action: Exact gather authority.
        """

        gather_entered.set()
        if not release_gather.wait(timeout=_WAIT_SECONDS):
            raise TimeoutError("test gather release was not delivered")

    serving, runtime, _, _, _, _ = _serving(
        identity,
        post_gather=post_gather,
        enable_forward_independent_handoff=True,
    )
    original_acquire = serving._delivery_leases.acquire_for_actions

    def acquire_then_pause(
        actions: tuple[NativeTerminalOwnerAction, ...],
    ) -> NativeTerminalDeliveryLeaseDisposition:
        """Expose the exact post-lease, pre-native-handoff boundary.

        :param actions: Complete source batch selected by the output reactor.
        :returns: Real delivery disposition after the test releases the batch.
        """

        disposition = original_acquire(actions)
        assert disposition is NativeTerminalDeliveryLeaseDisposition.ACQUIRED
        leases_acquired.set()
        if not release_acquisition.wait(timeout=_WAIT_SECONDS):
            raise TimeoutError("source batch acquisition release was not delivered")
        return disposition

    monkeypatch.setattr(
        serving._delivery_leases,
        "acquire_for_actions",
        acquire_then_pause,
    )
    reactor_failures: list[PackedTerminalProcessReactorFailure] = []
    reactor = PackedTerminalProcessReactor.for_source(
        serving,
        reactor_failures.append,
    )
    scheduler_inbox = serving._scheduler_serving._inbox
    original_wait = scheduler_inbox._wait_for_publication_intents

    def observe_scheduler_wait(intents: frozenset[int]) -> None:
        """Expose the real launch wait which deadlocked attempt 29.

        :param intents: Exact causal delivery intents blocking host launch.
        """

        scheduler_waiting.set()
        original_wait(intents)

    monkeypatch.setattr(
        scheduler_inbox,
        "_wait_for_publication_intents",
        observe_scheduler_wait,
    )
    reactor_started = False
    serving.start()
    try:
        reactor.start(_WAIT_SECONDS)
        reactor_started = True
        submission = _submission(identity)
        serving.bind_submission(submission, lambda value: None)
        serving.attach_producer_completion(submission)
        assert serving.packed_ready(identity.local_binding.digest)
        assert leases_acquired.wait(timeout=_WAIT_SECONDS)

        def complete_delivery_off_scheduler() -> NativeTerminalOwnerInventory:
            """Drive delivery while the main interpreter owns the launch wait.

            :returns: Native inventory captured before releasing the launch.
            """

            try:
                assert scheduler_waiting.wait(timeout=_WAIT_SECONDS)
                assert not launch_submitted.is_set()
                release_acquisition.set()
                _wait_for_phase(
                    lambda: (
                        runtime.snapshot().owner.source_batch_handoff_action_count
                        == 1
                    ),
                    "native source batch did not claim while the scheduler waited",
                )
                _wait_for_phase(
                    gather_entered.is_set,
                    "source gather remained coupled to the blocked scheduler",
                )
                owner = runtime.snapshot().owner
                assert owner.unclaimed_handoff_action_count == 0
                assert owner.claimed_handoff_action_count == 1
                assert owner.source_batch_handoff_count == 1
                assert owner.source_batch_handoff_action_count == 1
                assert serving._delivery_leases.inventory().active_binding_digests == (
                    identity.local_binding.digest,
                )
                assert not launch_submitted.is_set()

                release_gather.set()
                assert serving._gather_worker.wait_until_idle(_WAIT_SECONDS)
                serving._delivery_leases.mark_outcomes_sent(identity.local_binding)
                assert (
                    serving._delivery_leases.inventory().active_binding_digests
                    == (identity.local_binding.digest,)
                )
                assert not launch_submitted.is_set()
                serving._delivery_leases.mark_publication_owned(identity.local_binding)
                return owner
            finally:
                release_acquisition.set()
                release_gather.set()
                active = serving._delivery_leases.inventory().active_binding_digests
                if identity.local_binding.digest in active:
                    serving._delivery_leases.release_process_fatal()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            delivery = executor.submit(complete_delivery_off_scheduler)
            launch_result = serving._scheduler_serving.launch_handoff(
                lambda: launch_submitted.set() or "submitted"
            )
            owner = delivery.result(timeout=_WAIT_SECONDS)

        assert launch_result == "submitted"
        assert owner.source_batch_handoff_action_count == 1

        assert launch_submitted.is_set()
        assert reactor_failures == []
    finally:
        release_acquisition.set()
        release_gather.set()
        if reactor_started:
            reactor.close(_WAIT_SECONDS)
        serving.abort_and_close()


def test_gather_worker_owns_direct_inbox_and_binds_device_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Blocked gather work cannot delay ACK progress on the process reactor."""

    identity = _identity()
    gather_entered = threading.Event()
    release_gather = threading.Event()
    gather_threads: list[str] = []
    device_threads: list[str] = []

    def post_gather(
        submission: PackedTerminalSourceSubmission,
        action: NativeTerminalOwnerAction,
    ) -> None:
        """Hold one gather while the process reactor consumes an ACK.

        :param submission: Exact source submission.
        :param action: Exact gather authority.
        """

        gather_threads.append(threading.current_thread().name)
        gather_entered.set()
        if not release_gather.wait(timeout=_WAIT_SECONDS):
            raise TimeoutError("test gather release was not delivered")

    def bind_device() -> None:
        """Record the thread which owns production device binding."""

        device_threads.append(threading.current_thread().name)

    serving, runtime, _, _, _, _ = _serving(
        identity,
        post_gather=post_gather,
        bind_gather_cuda_device=bind_device,
    )
    serving.start()
    try:
        worker = serving.inventory().gather_worker
        assert worker.cuda_device_bound
        assert worker.thread_alive
        assert runtime.snapshot().source_gather.capacity == 8
        assert runtime.source_gather_actions.fileno() not in serving.runtime_filenos
        assert device_threads == ["packed-terminal-source-gather-worker"]

        submission = _submission(identity)
        serving.bind_submission(submission, lambda submission: None)
        serving.attach_producer_completion(submission)
        assert serving.packed_ready(identity.local_binding.digest)
        assert runtime.wait_for_output_projection(_WAIT_SECONDS)
        assert gather_entered.wait(timeout=_WAIT_SECONDS)

        ack = _action(
            identity,
            action_id=90,
            kind=NativeTerminalOwnerActionKind.SOURCE_ACK_READY,
        )
        consumed: list[NativeTerminalOwnerActionKind] = []

        def consume_ack(
            action: NativeTerminalOwnerAction,
            send_ack: Callable[
                [PackedTerminalSourceSubmission, NativeTerminalOwnerAction], None
            ],
        ) -> None:
            """Acknowledge one synthetic reactor-owned action.

            :param action: Exact synthetic ACK action.
            :param send_ack: Unused source ACK callback.
            """

            consumed.append(action.kind)
            runtime.acknowledge_consumed_action(action)

        monkeypatch.setattr(serving._wiring, "consume_ack_ready", consume_ack)
        runtime._route_action(ack)
        assert serving.drain_runtime_actions() >= 1
        assert consumed == [NativeTerminalOwnerActionKind.SOURCE_ACK_READY]
        assert gather_threads == ["packed-terminal-source-gather-worker"]
    finally:
        release_gather.set()
        assert serving._gather_worker.wait_until_idle(_WAIT_SECONDS)
        serving.abort_and_close()


def test_source_work_drain_prioritizes_ack_and_preserves_kind_fifo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ACK authority bypasses queued outcomes without reordering either kind."""

    identity = _identity()
    serving, runtime, _, _, _, _ = _serving(identity)
    serving.start()
    consumed: list[int] = []
    try:
        actions = (
            _action(
                identity,
                action_id=101,
                kind=NativeTerminalOwnerActionKind.SOURCE_OUTCOME_READY,
            ),
            _action(
                identity,
                action_id=102,
                kind=NativeTerminalOwnerActionKind.SOURCE_ACK_READY,
            ),
            _action(
                identity,
                action_id=103,
                kind=NativeTerminalOwnerActionKind.SOURCE_OUTCOME_READY,
            ),
            _action(
                identity,
                action_id=104,
                kind=NativeTerminalOwnerActionKind.SOURCE_ACK_READY,
            ),
        )

        def consume(
            action: NativeTerminalOwnerAction,
            callback: Callable[
                [PackedTerminalSourceSubmission, NativeTerminalOwnerAction], None
            ],
        ) -> None:
            """Record and retire one synthetic source-work action.

            :param action: Exact synthetic source-work action.
            :param callback: Unused serving callback.
            """

            consumed.append(action.action_id)
            runtime.acknowledge_consumed_action(action)

        monkeypatch.setattr(serving._wiring, "consume_ack_ready", consume)
        monkeypatch.setattr(serving._wiring, "consume_outcome_ready", consume)
        for action in actions:
            runtime._route_action(action)

        assert serving.drain_runtime_actions() == 4
        assert consumed == [102, 104, 101, 103]
    finally:
        serving.abort_and_close()


def test_runtime_drain_claims_ready_siblings_before_scheduler_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scheduler lock contention cannot strand its publication sibling."""

    identity = _identity()
    serving, runtime, publisher, _, _, _ = _serving(identity)
    scheduler_routed = threading.Event()
    release_split_projection = threading.Event()
    publisher_routed = threading.Event()
    scheduler_delivery_entered = threading.Event()
    late_projection_started = threading.Event()
    late_projection_entered = threading.Event()
    scheduler_state_lock = serving._scheduler_serving._inbox._state_lock
    scheduler_lock_owned_by_test = False
    serving.start()
    try:
        _advance_source_to_request_ready_join(serving, runtime, identity)
        original_route_action = runtime._route_action

        def route_with_split(action: NativeTerminalOwnerAction) -> None:
            """Pause projection between the native sibling action routes.

            :param action: Exact native action being projected.
            """

            original_route_action(action)
            if action.kind is NativeTerminalOwnerActionKind.RECLAIM_AUTHORIZED:
                scheduler_routed.set()
                if not release_split_projection.wait(timeout=_WAIT_SECONDS):
                    raise TimeoutError("sibling projection release was not delivered")
            if action.kind is NativeTerminalOwnerActionKind.GATEWAY_PUBLICATION_READY:
                publisher_routed.set()

        original_publish_action = serving._scheduler_serving.publish_action

        def observe_scheduler_delivery(
            action: NativeTerminalOwnerAction,
        ) -> SchedulerReceiptPublishResult:
            """Expose entry into the scheduler-owned publication lock.

            :param action: Exact reclaim authority being published.
            :returns: Scheduler publication result.
            """

            scheduler_delivery_entered.set()
            return original_publish_action(action)

        def enter_late_projection() -> None:
            """Model a second native output arriving during scheduler delivery."""

            late_projection_started.set()
            with runtime.source_action_delivery_fence():
                late_projection_entered.set()

        monkeypatch.setattr(runtime, "_route_action", route_with_split)
        monkeypatch.setattr(
            serving._scheduler_serving,
            "publish_action",
            observe_scheduler_delivery,
        )
        ready = _request_ready_receipt(identity)
        scheduler_state_lock.acquire()
        scheduler_lock_owned_by_test = True
        serving.request_ready(
            binding_digest=identity.local_binding.digest,
            wire_receipt=ready.wire_receipt,
            local_receipt=ready.local_receipt,
            authenticated_issuer=identity.request_ready_issuer,
        )
        assert scheduler_routed.wait(timeout=_WAIT_SECONDS)
        assert not publisher_routed.is_set()

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            try:
                drain_future = executor.submit(serving.drain_runtime_actions)
                assert not scheduler_delivery_entered.wait(timeout=0.05)
                assert runtime.scheduler_actions.snapshot().queued_count == 1

                release_split_projection.set()
                assert publisher_routed.wait(timeout=_WAIT_SECONDS)
                assert scheduler_delivery_entered.wait(timeout=_WAIT_SECONDS)
                with runtime._condition:
                    publisher_actions = tuple(
                        action
                        for action in runtime._consumer_pending.values()
                        if action.kind
                        is NativeTerminalOwnerActionKind.GATEWAY_PUBLICATION_READY
                    )
                    assert len(publisher_actions) == 1
                    publisher_action = publisher_actions[0]
                    assert (
                        publisher_action.action_id in runtime._inbox_claimed_action_ids
                    )
                assert not drain_future.done()

                late_projection_future = executor.submit(enter_late_projection)
                assert late_projection_started.wait(timeout=_WAIT_SECONDS)
                assert not late_projection_entered.is_set()
                assert not late_projection_future.done()

                scheduler_state_lock.release()
                scheduler_lock_owned_by_test = False
                assert drain_future.result(timeout=_WAIT_SECONDS) >= 2
                late_projection_future.result(timeout=_WAIT_SECONDS)
                assert late_projection_entered.is_set()
            finally:
                release_split_projection.set()
                if scheduler_lock_owned_by_test:
                    scheduler_state_lock.release()
                    scheduler_lock_owned_by_test = False

        assert len(publisher.values) == 1
        serving.drain_scheduler_at_loop_entry()
        publication = publisher.values[0]
        gateway_issuer = TerminalWireReceiptIssuer(identity.publisher_issuer)
        completed_ns = 4_000
        serving.publisher_result(
            TerminalGatewayPublicationSuccess(
                publication=publication,
                completed_ns=completed_ns,
                source_receipts=tuple(
                    gateway_issuer.issue(
                        binding=binding,
                        kind=TerminalReceiptKind.GATEWAY_PUBLISHED,
                        outcome=TerminalReceiptOutcome.SUCCESS,
                        terminal_timestamp_ns=completed_ns,
                    )
                    for binding in identity.source_bindings
                ),
            )
        )
        _pump(serving, runtime)
        assert serving.inventory().retained_resource_count == 0
        serving.stop_admission_and_retire_producers()
        serving.close_clean(_WAIT_SECONDS)
    finally:
        release_split_projection.set()
        if scheduler_lock_owned_by_test:
            scheduler_state_lock.release()
        if runtime.disposition.value != "stopped":
            serving.abort_and_close()


def test_claimed_runtime_batch_failure_reconciles_every_local_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed first consumer cannot orphan later claimed batch members."""

    identity = _identity()
    serving, runtime, _, _, _, _ = _serving(identity)
    serving.start()
    try:
        _advance_source_to_request_ready_join(serving, runtime, identity)
        ready = _request_ready_receipt(identity)
        serving.request_ready(
            binding_digest=identity.local_binding.digest,
            wire_receipt=ready.wire_receipt,
            local_receipt=ready.local_receipt,
            authenticated_issuer=identity.request_ready_issuer,
        )
        assert runtime.wait_for_output_projection(_WAIT_SECONDS)
        with runtime._condition:
            sibling_action_ids = frozenset(
                action.action_id
                for action in runtime._consumer_pending.values()
                if action.kind
                in (
                    NativeTerminalOwnerActionKind.RECLAIM_AUTHORIZED,
                    NativeTerminalOwnerActionKind.GATEWAY_PUBLICATION_READY,
                )
            )
        assert len(sibling_action_ids) == 2

        def fail_scheduler_delivery(
            actions: tuple[NativeTerminalOwnerAction, ...],
            execution: object,
        ) -> None:
            """Fail only the non-empty authoritative scheduler population.

            :param actions: Already-claimed scheduler action population.
            :param execution: Exact process-reactor ownership ledger.
            """

            if len(actions) > 0:
                raise RuntimeError("synthetic claimed-batch delivery failure")

        monkeypatch.setattr(
            serving,
            "_consume_scheduler_actions",
            fail_scheduler_delivery,
        )
        with pytest.raises(
            RuntimeError,
            match="synthetic claimed-batch delivery failure",
        ):
            serving.drain_runtime_actions()

        with runtime._condition:
            assert sibling_action_ids.isdisjoint(runtime._consumer_pending)
            assert sibling_action_ids.isdisjoint(runtime._inbox_claimed_action_ids)
        runtime.snapshot()
    finally:
        serving.abort_and_close()


def test_later_failure_preserves_scheduler_accepted_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Later work cannot reconcile authority retained by the scheduler."""

    identity = _identity()
    serving, runtime, _, _, _, _ = _serving(identity)
    serving.start()
    try:
        _advance_source_to_request_ready_join(serving, runtime, identity)
        ready = _request_ready_receipt(identity)
        serving.request_ready(
            binding_digest=identity.local_binding.digest,
            wire_receipt=ready.wire_receipt,
            local_receipt=ready.local_receipt,
            authenticated_issuer=identity.request_ready_issuer,
        )
        assert runtime.wait_for_output_projection(_WAIT_SECONDS)
        with runtime._condition:
            siblings = {
                action.kind: action
                for action in runtime._consumer_pending.values()
                if action.kind
                in (
                    NativeTerminalOwnerActionKind.RECLAIM_AUTHORIZED,
                    NativeTerminalOwnerActionKind.GATEWAY_PUBLICATION_READY,
                )
            }
        assert len(siblings) == 2

        def fail_later_source_work(
            actions: tuple[NativeTerminalOwnerAction, ...],
            execution: object,
        ) -> None:
            """Fail after scheduler delivery, before publisher delivery.

            :param actions: Already-claimed source-work population.
            :param execution: Exact process-reactor ownership ledger.
            """

            raise RuntimeError("synthetic post-scheduler execution failure")

        with monkeypatch.context() as scoped_patch:
            scoped_patch.setattr(
                serving,
                "_consume_source_work_actions",
                fail_later_source_work,
            )
            with pytest.raises(
                RuntimeError,
                match="synthetic post-scheduler execution failure",
            ):
                serving.drain_runtime_actions()

        scheduler_action = siblings[NativeTerminalOwnerActionKind.RECLAIM_AUTHORIZED]
        publisher_action = siblings[
            NativeTerminalOwnerActionKind.GATEWAY_PUBLICATION_READY
        ]
        with runtime._condition:
            assert scheduler_action.action_id in runtime._consumer_pending
            assert publisher_action.action_id not in runtime._consumer_pending
            assert publisher_action.action_id not in runtime._inbox_claimed_action_ids
        assert scheduler_action.action_id in (
            serving._scheduler_serving.inventory().retained_action_ids
        )
        runtime.snapshot()
    finally:
        serving.abort_and_close()


def test_scheduler_wake_failure_preserves_retained_action_through_abort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-insertion wake failure cannot surrender scheduler authority."""

    identity = _identity()
    serving, runtime, _, _, _, _ = _serving(identity)
    serving.start()
    closed = False
    try:
        _advance_source_to_request_ready_join(serving, runtime, identity)
        ready = _request_ready_receipt(identity)
        serving.request_ready(
            binding_digest=identity.local_binding.digest,
            wire_receipt=ready.wire_receipt,
            local_receipt=ready.local_receipt,
            authenticated_issuer=identity.request_ready_issuer,
        )
        assert runtime.wait_for_output_projection(_WAIT_SECONDS)
        scheduler_write_fd = serving._scheduler_serving._inbox._write_fd
        real_write = os.write

        def fail_scheduler_wake(file_descriptor: int, payload: bytes) -> int:
            """Fail only the scheduler publication wake after insertion.

            :param file_descriptor: Target descriptor.
            :param payload: Exact wake payload.
            :returns: Bytes written for every unrelated descriptor.
            """

            if file_descriptor == scheduler_write_fd:
                raise BrokenPipeError("synthetic scheduler wake failure")
            return real_write(file_descriptor, payload)

        with monkeypatch.context() as scoped_patch:
            scoped_patch.setattr(os, "write", fail_scheduler_wake)
            with pytest.raises(TerminalSchedulerActionPublicationError) as raised:
                serving.drain_runtime_actions()

        action = raised.value.action
        assert raised.value.scheduler_retains_action
        with runtime._condition:
            assert runtime._consumer_pending.get(action.action_id) == action
            assert all(
                action_id == action.action_id
                or pending.kind is NativeTerminalOwnerActionKind.PROCESS_FATAL
                for action_id, pending in runtime._consumer_pending.items()
            )
            assert action.action_id in runtime._inbox_claimed_action_ids
        assert serving._scheduler_serving.inventory().retained_action_ids == (
            action.action_id,
        )

        closed_inventory = serving.abort_and_close()
        closed = True
        assert closed_inventory.scheduler_serving.retained_action_ids == (
            action.action_id,
        )
        assert runtime.disposition.value == "stopped"
    finally:
        if not closed:
            serving.abort_and_close()


def test_lifecycle_failure_preserves_publisher_accepted_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lifecycle failure cannot reconcile publisher-retained authority."""

    identity = _identity()
    serving, runtime, publisher, _, _, _ = _serving(identity)
    serving.start()
    try:
        _advance_source_to_request_ready_join(serving, runtime, identity)
        ready = _request_ready_receipt(identity)
        serving.request_ready(
            binding_digest=identity.local_binding.digest,
            wire_receipt=ready.wire_receipt,
            local_receipt=ready.local_receipt,
            authenticated_issuer=identity.request_ready_issuer,
        )
        assert runtime.wait_for_output_projection(_WAIT_SECONDS)
        with runtime._condition:
            publisher_actions = tuple(
                action
                for action in runtime._consumer_pending.values()
                if action.kind
                is NativeTerminalOwnerActionKind.GATEWAY_PUBLICATION_READY
            )
            maximum_action_id = max(runtime._consumer_pending)
        assert len(publisher_actions) == 1
        publisher_action = publisher_actions[0]
        lifecycle_action = _action(
            identity,
            action_id=maximum_action_id + 1_000,
            kind=NativeTerminalOwnerActionKind.REQUEST_QUARANTINED,
        )
        runtime._route_action(lifecycle_action)

        def fail_lifecycle(
            actions: tuple[NativeTerminalOwnerAction, ...],
            execution: object,
        ) -> None:
            """Fail after the publisher accepts its asynchronous authority.

            :param actions: Already-claimed lifecycle population.
            :param execution: Exact process-reactor ownership ledger.
            """

            assert actions == (lifecycle_action,)
            raise RuntimeError("synthetic post-publisher lifecycle failure")

        with monkeypatch.context() as scoped_patch:
            scoped_patch.setattr(
                serving,
                "_consume_lifecycle_actions",
                fail_lifecycle,
            )
            with pytest.raises(
                RuntimeError,
                match="synthetic post-publisher lifecycle failure",
            ):
                serving.drain_runtime_actions()

        assert len(publisher.values) == 1
        with runtime._condition:
            assert publisher_action.action_id in runtime._consumer_pending
            assert lifecycle_action.action_id not in runtime._consumer_pending
            assert publisher_action.action_id in runtime._inbox_claimed_action_ids
            assert lifecycle_action.action_id not in runtime._inbox_claimed_action_ids
        assert serving.wiring.inventory().pending_publication_action_count == 1
        runtime.snapshot()
    finally:
        serving.abort_and_close()


def test_publication_failure_after_retention_survives_abort_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-retention publisher failure transfers exact downstream ownership."""

    identity = _identity()
    serving, runtime, _, _, _, _ = _serving(identity)
    serving.start()
    closed = False
    try:
        _advance_source_to_request_ready_join(serving, runtime, identity)
        ready = _request_ready_receipt(identity)
        serving.request_ready(
            binding_digest=identity.local_binding.digest,
            wire_receipt=ready.wire_receipt,
            local_receipt=ready.local_receipt,
            authenticated_issuer=identity.request_ready_issuer,
        )
        assert runtime.wait_for_output_projection(_WAIT_SECONDS)

        def fail_retained_publication(*args: object) -> None:
            """Fail after wiring stores the exact publication action.

            :param args: Retained publication inputs.
            """

            del args
            raise RuntimeError("synthetic post-retention publication failure")

        monkeypatch.setattr(
            serving.wiring,
            "_consume_retained_gateway_publication",
            fail_retained_publication,
        )
        with pytest.raises(PackedTerminalSourcePublicationRetentionError) as raised:
            serving.drain_runtime_actions()

        action = raised.value.action
        with runtime._condition:
            assert runtime._consumer_pending.get(action.action_id) == action
            assert action.action_id in runtime._inbox_claimed_action_ids
        assert serving.wiring.inventory().pending_publication_action_ids == (
            action.action_id,
        )

        closed_inventory = serving.abort_and_close()
        closed = True
        assert closed_inventory.wiring.pending_publication_action_ids == (
            action.action_id,
        )
        assert runtime.disposition.value == "stopped"
    finally:
        if not closed:
            serving.abort_and_close()


def test_later_inbox_claim_failure_reconciles_earlier_population(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later inbox failure cannot hide an earlier removed population."""

    identity = _identity()
    serving, runtime, _, _, _, _ = _serving(identity)
    serving.start()
    try:
        _advance_source_to_request_ready_join(serving, runtime, identity)
        ready = _request_ready_receipt(identity)
        serving.request_ready(
            binding_digest=identity.local_binding.digest,
            wire_receipt=ready.wire_receipt,
            local_receipt=ready.local_receipt,
            authenticated_issuer=identity.request_ready_issuer,
        )
        assert runtime.wait_for_output_projection(_WAIT_SECONDS)
        with runtime._condition:
            siblings = {
                action.kind: action
                for action in runtime._consumer_pending.values()
                if action.kind
                in (
                    NativeTerminalOwnerActionKind.RECLAIM_AUTHORIZED,
                    NativeTerminalOwnerActionKind.GATEWAY_PUBLICATION_READY,
                )
            }
        assert len(siblings) == 2
        original_publisher_drain = runtime.publisher_actions.drain
        publisher_drain_count = 0

        def fail_first_publisher_drain(
            maximum_items: int | None = None,
        ) -> tuple[NativeTerminalOwnerAction, ...]:
            """Fail before removing the later publisher population once.

            :param maximum_items: Optional inbox drain bound.
            :returns: Publisher actions after the one injected failure.
            """

            nonlocal publisher_drain_count
            publisher_drain_count += 1
            if publisher_drain_count == 1:
                raise OSError("synthetic later-inbox claim failure")
            return original_publisher_drain(maximum_items)

        monkeypatch.setattr(
            runtime.publisher_actions,
            "drain",
            fail_first_publisher_drain,
        )
        with pytest.raises(
            RuntimeError,
            match="source runtime action batch claim failed",
        ):
            serving.drain_runtime_actions()

        scheduler_action = siblings[NativeTerminalOwnerActionKind.RECLAIM_AUTHORIZED]
        publisher_action = siblings[
            NativeTerminalOwnerActionKind.GATEWAY_PUBLICATION_READY
        ]
        with runtime._condition:
            assert scheduler_action.action_id not in runtime._consumer_pending
            assert publisher_action.action_id in runtime._consumer_pending
        assert runtime.publisher_actions.snapshot().queued_count == 1
        runtime.snapshot()
    finally:
        serving.abort_and_close()


def test_preclaimed_batch_failure_reconciles_without_native_dequeue_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed consumer reconciles a batch without repeating native claim."""

    identity = _identity()
    serving, runtime, publisher, _, _, _ = _serving(
        identity,
        enable_forward_independent_handoff=True,
    )
    actions = (
        _action(
            identity,
            action_id=501,
            kind=NativeTerminalOwnerActionKind.GATEWAY_PUBLICATION_READY,
        ),
        _action(
            identity,
            action_id=502,
            kind=NativeTerminalOwnerActionKind.GATEWAY_PUBLICATION_READY,
        ),
    )

    def fail_publisher_population(
        claimed: tuple[NativeTerminalOwnerAction, ...],
        execution: object,
    ) -> None:
        """Reject the exact already-claimed publisher population.

        :param claimed: Complete preclaimed publisher population.
        :param execution: Source reactor's local ownership ledger.
        """

        assert claimed == actions
        raise RuntimeError("synthetic preclaimed population failure")

    def reject_dequeue_reclaim(action: NativeTerminalOwnerAction) -> None:
        """Fail if the obsolete per-action native claim seam is reached.

        :param action: Unexpected action offered for a second native claim.
        """

        raise AssertionError(f"action {action.action_id} was claimed twice")

    serving.start()
    try:
        runtime._record_source_preclaims(actions)
        for action in actions:
            runtime._project_action(action)
        with monkeypatch.context() as scoped_patch:
            scoped_patch.setattr(
                runtime._owner,
                "claim_forward_independent_handoff",
                reject_dequeue_reclaim,
            )
            scoped_patch.setattr(
                serving,
                "_consume_publisher_actions",
                fail_publisher_population,
            )
            with pytest.raises(
                RuntimeError,
                match="synthetic preclaimed population failure",
            ):
                serving.drain_runtime_actions()

        action_ids = frozenset(action.action_id for action in actions)
        with runtime._condition:
            assert action_ids.isdisjoint(runtime._consumer_pending)
            assert action_ids.isdisjoint(runtime._inbox_claimed_action_ids)
            assert action_ids.isdisjoint(runtime._source_preclaimed_actions)
            assert action_ids.isdisjoint(runtime._source_preclaimed_consumer_action_ids)
        snapshot = runtime.snapshot()
        assert snapshot.source_preclaimed_count == 0
        assert snapshot.source_preclaimed_consumer_count == 0
        assert publisher.values == []
    finally:
        serving.abort_and_close()


def test_replayed_publisher_action_keeps_original_downstream_owner() -> None:
    """A stale queue entry cannot surrender an already accepted publication."""

    identity = _identity()
    serving, runtime, publisher, _, _, _ = _serving(identity)
    serving.start()
    try:
        _advance_source_to_request_ready_join(serving, runtime, identity)
        ready = _request_ready_receipt(identity)
        serving.request_ready(
            binding_digest=identity.local_binding.digest,
            wire_receipt=ready.wire_receipt,
            local_receipt=ready.local_receipt,
            authenticated_issuer=identity.request_ready_issuer,
        )
        assert runtime.wait_for_output_projection(_WAIT_SECONDS)
        serving.drain_runtime_actions()
        assert len(publisher.values) == 1
        with runtime._condition:
            publisher_actions = tuple(
                action
                for action in runtime._consumer_pending.values()
                if action.kind
                is NativeTerminalOwnerActionKind.GATEWAY_PUBLICATION_READY
            )
        assert len(publisher_actions) == 1
        action = publisher_actions[0]
        assert action.action_id in runtime._inbox_claimed_action_ids

        runtime.publisher_actions._enqueue(action)
        with pytest.raises(
            RuntimeError,
            match="source runtime action batch claim failed",
        ):
            serving.drain_runtime_actions()

        with runtime._condition:
            assert runtime._consumer_pending.get(action.action_id) == action
            assert action.action_id in runtime._inbox_claimed_action_ids
        assert serving.wiring.inventory().pending_publication_action_ids == (
            action.action_id,
        )
    finally:
        serving.abort_and_close()


def test_gather_worker_failure_is_process_fatal() -> None:
    """A failed dedicated gather action wakes fail-closed scheduler handling."""

    identity = _identity()

    def fail_gather(
        submission: PackedTerminalSourceSubmission,
        action: NativeTerminalOwnerAction,
    ) -> None:
        """Raise from the worker-owned gather boundary.

        :param submission: Exact source submission.
        :param action: Exact gather authority.
        """

        raise RuntimeError("synthetic gather failure")

    serving, runtime, _, _, _, fatal_inventories = _serving(
        identity,
        post_gather=fail_gather,
    )
    serving.start()
    try:
        submission = _submission(identity)
        serving.bind_submission(submission, lambda submission: None)
        serving.attach_producer_completion(submission)
        assert serving.packed_ready(identity.local_binding.digest)
        assert runtime.wait_for_output_projection(_WAIT_SECONDS)
        assert serving._gather_worker.wait_until_idle(_WAIT_SECONDS)
        inventory = serving.inventory()
        assert inventory.gather_worker.fatal_reason == "source gather action failed"
        assert inventory.gather_worker.thread_alive
        assert inventory.gather_worker.abort_requested
        assert inventory.owner_dead_marked
        with pytest.raises(RuntimeError):
            serving.drain_scheduler_at_loop_entry()
        assert len(fatal_inventories) == 1
        assert runtime.disposition.value == "abort_draining"
    finally:
        serving.abort_and_close()


def test_gather_worker_launch_failure_remains_abortable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve fail-closed teardown when the worker thread never launches."""

    identity = _identity()
    serving, runtime, _, _, _, fatal_inventories = _serving(identity)

    def fail_thread_start() -> None:
        """Reject the process-lifetime thread launch deterministically.

        :raises RuntimeError: Always, before the worker thread begins.
        """

        raise RuntimeError("synthetic gather thread launch failure")

    monkeypatch.setattr(serving._gather_worker._thread, "start", fail_thread_start)

    with pytest.raises(RuntimeError, match="synthetic gather thread launch failure"):
        serving.start()

    inventory = serving.inventory()
    assert inventory.gather_worker.fatal_reason == (
        "source gather worker thread failed to start"
    )
    assert not inventory.gather_worker.thread_alive
    assert inventory.owner_dead_marked
    assert fatal_inventories == []
    assert runtime.disposition.value == "abort_draining"

    final_inventory = serving.abort_and_close()
    assert final_inventory.gather_worker.disposition.value == "process_fatal"
    assert not final_inventory.gather_worker.thread_alive


def test_composition_binds_both_scheduler_owners_before_lifecycle() -> None:
    """The native lifecycle cannot race either scheduler-owned registry."""

    identity = _identity()
    serving, _, _, _, _, _ = _serving(identity)
    serving.start()
    release_calls: list[int] = []
    try:
        submission = _submission(identity)
        serving.bind_submission(
            submission,
            lambda submission: release_calls.append(1),
        )
        serving.attach_producer_completion(submission)
        inventory = serving.inventory()
        assert inventory.runtime.scheduler_live_count == 1
        assert inventory.scheduler_consumer.active_binding_digests == (
            identity.local_binding.digest,
        )
        assert inventory.scheduler_serving.inbox.live_bindings == (
            identity.local_binding,
        )
        assert release_calls == []
    finally:
        serving.abort_and_close()


def test_retained_resource_count_covers_native_subscription_authority() -> None:
    """A native subscription leak keeps the clean-close scalar nonzero."""

    serving, _, _, _, _, _ = _serving(_identity())
    serving.start()
    try:
        inventory = serving.inventory()
        assert inventory.retained_resource_count == 0

        native = dataclasses.replace(
            inventory.grouped_nixl.native,
            active_channel_subscriptions=1,
        )
        grouped_nixl = dataclasses.replace(inventory.grouped_nixl, native=native)
        retained = dataclasses.replace(inventory, grouped_nixl=grouped_nixl)

        assert retained.retained_resource_count == 1
    finally:
        serving.abort_and_close()


def test_bind_failure_pairs_unpublished_scheduler_cancellation() -> None:
    """A pre-publication source mismatch leaves neither scheduler registry live."""

    identity = _identity()
    serving, _, _, _, _, _ = _serving(identity)
    serving.start()
    try:
        with pytest.raises(RuntimeError, match="another runtime"):
            serving.bind_submission(
                _submission(_identity(local_rank=1)),
                lambda submission: None,
            )
        inventory = serving.inventory()
        assert inventory.wiring.active_binding_digests == ()
        assert inventory.scheduler_consumer.active_binding_digests == ()
        assert inventory.scheduler_serving.inbox.live_count == 0
    finally:
        serving.abort_and_close()


def test_full_source_composition_retires_exactly_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Pump every source action through runtime, scheduler, and publisher joins."""

    caplog.set_level(
        logging.INFO,
        logger="sglang.srt.disaggregation.terminal_progress.source_wiring",
    )
    identity = _identity()
    serving, runtime, publisher, _, work_labels, _ = _serving(
        identity,
        enable_forward_independent_handoff=True,
    )
    submission = _submission(identity)
    release_calls: list[PackedTerminalSourceSubmission] = []
    reactor_failures: list[PackedTerminalProcessReactorFailure] = []
    reactor = PackedTerminalProcessReactor.for_source(
        serving,
        reactor_failures.append,
    )
    reactor_started = False
    reactor_closed = False
    serving_closed = False
    serving.start()
    try:
        reactor.start(_WAIT_SECONDS)
        reactor_started = True
        serving.bind_submission(submission, release_calls.append)
        serving.attach_producer_completion(submission)
        digest = identity.local_binding.digest
        assert serving.packed_ready(digest)
        _wait_for_phase(
            lambda: work_labels == ["gather"],
            "source gather phase did not complete",
        )
        assert serving._gather_worker.wait_until_idle(_WAIT_SECONDS)
        assert work_labels == ["gather"]
        delivery = serving.inventory().delivery_leases
        assert delivery.active_binding_digests == (digest,)
        assert delivery.outcomes_sent_binding_digests == ()
        assert delivery.publication_owned_binding_digests == ()
        cancellation = serving.cancel_submission(
            identity.local_binding,
            "client disconnected",
        )
        assert (
            cancellation
            is PackedTerminalSourceCancellationDisposition.COMPLETION_REQUIRED
        )
        assert serving.inventory().wiring.completion_required_binding_digests == (
            digest,
        )
        _wait_for_phase(
            lambda: sum(
                parse_terminal_progress_timing_log_line(record.getMessage()) is not None
                for record in caplog.records
            )
            == 1,
            "source submission timing observation did not complete",
        )
        samples = tuple(
            sample
            for record in caplog.records
            if (sample := parse_terminal_progress_timing_log_line(record.getMessage()))
            is not None
        )
        assert len(samples) == 1
        assert samples[0].binding.digest == digest
        assert samples[0].sample_key == "source-rank-0"
        assert samples[0].field.value == "producer_to_owner_handoff_ms"
        assert samples[0].started_ns == 1_000
        assert samples[0].completed_ns >= samples[0].started_ns

        runtime._owner.submit(
            NativeTerminalOwnerEvent(
                producer_id=_NATIVE_PRODUCER_ID,
                binding_digest=digest,
                kind=NativeTerminalOwnerEventKind.SOURCE_NATIVE_TERMINAL,
                enqueued_ns=2_000,
            )
        )
        _wait_for_phase(
            lambda: work_labels == ["gather", "outcome"]
            and serving._delivery_leases.inventory().outcomes_sent_binding_digests
            == (digest,),
            "source outcome phase did not complete",
        )
        assert work_labels == ["gather", "outcome"]
        delivery = serving.inventory().delivery_leases
        assert delivery.active_binding_digests == (digest,)
        assert delivery.outcomes_sent_binding_digests == (digest,)
        assert delivery.publication_owned_binding_digests == ()

        serving.wiring.teardown_received(digest, identity.request_ready_issuer)
        _wait_for_phase(
            lambda: work_labels == ["gather", "outcome", "ack"],
            "source acknowledgement side effect did not complete",
        )
        _wait_for_phase(
            lambda: runtime.snapshot().consumer_pending_count == 0,
            "source acknowledgement phase did not complete",
        )
        assert work_labels == ["gather", "outcome", "ack"]

        ready = TerminalWireReceiptIssuer(identity.request_ready_issuer).issue(
            binding=identity.local_binding,
            kind=TerminalReceiptKind.REQUEST_READY,
            outcome=TerminalReceiptOutcome.SUCCESS,
            terminal_timestamp_ns=3_000,
        )
        serving.wiring.request_ready(
            binding_digest=digest,
            wire_receipt=ready.wire_receipt,
            local_receipt=ready.local_receipt,
            authenticated_issuer=identity.request_ready_issuer,
        )
        _wait_for_phase(
            lambda: len(publisher.values) == 1
            and serving._delivery_leases.inventory().active_binding_digests == ()
            and serving._scheduler_serving._inbox.inventory().active_delivery_intents
            == (),
            "gateway publication phase did not complete",
        )
        assert len(publisher.values) == 1
        assert serving.inventory().delivery_leases.active_binding_digests == ()
        assert serving.inventory().scheduler_serving.inbox.active_delivery_intents == ()
        serving.drain_scheduler_at_loop_entry()
        assert release_calls == [submission]

        publication = publisher.values[0]
        gateway_issuer = TerminalWireReceiptIssuer(identity.publisher_issuer)
        completed_ns = 4_000
        result = TerminalGatewayPublicationSuccess(
            publication=publication,
            completed_ns=completed_ns,
            source_receipts=tuple(
                gateway_issuer.issue(
                    binding=binding,
                    kind=TerminalReceiptKind.GATEWAY_PUBLISHED,
                    outcome=TerminalReceiptOutcome.SUCCESS,
                    terminal_timestamp_ns=completed_ns,
                )
                for binding in identity.source_bindings
            ),
        )
        serving.publisher_result(result)
        _wait_for_phase(
            lambda: serving.inventory().wiring.active_binding_digests == ()
            and runtime.snapshot().consumer_pending_count == 0,
            "source retirement phase did not complete",
        )
        assert serving.inventory().wiring.active_binding_digests == ()
        assert serving.inventory().wiring.completion_required_binding_digests == ()

        serving.stop_admission_and_retire_producers()
        reactor.close(_WAIT_SECONDS)
        reactor_closed = True
        assert reactor_failures == []
        final_inventory = serving.inventory()
        owner = final_inventory.runtime.owner
        assert owner.source_batch_handoff_count == 5
        assert owner.source_batch_handoff_action_count == 5
        assert final_inventory.runtime.source_preclaimed_count == 0
        assert final_inventory.runtime.source_preclaimed_consumer_count == 0
        assert final_inventory.runtime.decode_publication_preclaimed_count == 0
        serving.close_clean(_WAIT_SECONDS)
        serving_closed = True
    finally:
        if reactor_started and not reactor_closed:
            reactor.close(_WAIT_SECONDS)
        if not serving_closed:
            serving.abort_and_close()


def test_runtime_fatal_marks_scheduler_and_quarantines_retained_release() -> None:
    """Owner death wakes the scheduler and preserves mutable source resources."""

    identity = _identity()
    serving, runtime, _, _, _, fatal_inventories = _serving(identity)
    serving.start()
    try:
        submission = _submission(identity)
        serving.bind_submission(submission, lambda submission: None)
        serving.attach_producer_completion(submission)
        runtime.begin_abort()
        assert runtime.wait_for_output_projection(_WAIT_SECONDS)
        _pump(serving, runtime)
        with pytest.raises(RuntimeError):
            serving.drain_scheduler_at_loop_entry()
        assert len(fatal_inventories) == 1
        inventory = serving.inventory()
        assert inventory.owner_dead_marked
        assert inventory.scheduler_consumer.quarantined_binding_digests == (
            identity.local_binding.digest,
        )
        assert inventory.wiring.quarantined_binding_digests == (
            identity.local_binding.digest,
        )
    finally:
        serving.abort_and_close()


def test_quarantine_callback_failure_retains_exact_action_until_abort_closure() -> None:
    """Post-claim quarantine failure cannot discard lifecycle authority."""

    identity = _identity()

    def fail_quarantine(action: NativeTerminalOwnerAction) -> None:
        """Fail after the source wiring has claimed exact authority.

        :param action: Exact native fail-closed authority.
        """

        raise RuntimeError(f"synthetic quarantine failure for {action.action_id}")

    serving, runtime, _, _, _, _ = _serving(
        identity,
        quarantine=fail_quarantine,
    )
    serving.start()
    closed = False
    try:
        submission = _submission(identity)
        serving.bind_submission(submission, lambda submission: None)
        serving.attach_producer_completion(submission)
        runtime.begin_abort()
        assert runtime.wait_for_output_projection(_WAIT_SECONDS)

        with pytest.raises(PackedTerminalSourceQuarantineRetentionError) as raised:
            serving.drain_runtime_actions()

        action = raised.value.action
        with runtime._condition:
            assert runtime._consumer_pending.get(action.action_id) == action
            assert action.action_id in runtime._inbox_claimed_action_ids
        inventory = serving.wiring.inventory()
        assert inventory.quarantined_binding_digests == (identity.local_binding.digest,)
        assert inventory.retained_quarantine_action_ids == (action.action_id,)

        closed_inventory = serving.abort_and_close()
        closed = True
        assert closed_inventory.wiring.retained_quarantine_action_ids == (
            action.action_id,
        )
        assert runtime.disposition.value == "stopped"
    finally:
        if not closed:
            serving.abort_and_close()


def test_source_python_producer_retirement_resumes_after_mid_roster_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Source teardown skips already-retired producers when a later retry resumes."""

    identity = _identity()
    serving, runtime, _, _, _, _ = _serving(identity)
    serving.start()
    closed = False
    try:
        serving.stop_admission_and_retire_native_producers()
        producer_ids = runtime.python_producer_ids
        failed_producer_id = producer_ids[1]
        original_retire = runtime.retire_python_producer

        def fail_second_producer(producer_id: int) -> None:
            """Retire the first namespace and fail at the next boundary.

            :param producer_id: Exact Python producer namespace.
            """

            if producer_id == failed_producer_id:
                raise RuntimeError("synthetic mid-roster retirement failure")
            original_retire(producer_id)

        with monkeypatch.context() as scoped_patch:
            scoped_patch.setattr(
                runtime,
                "retire_python_producer",
                fail_second_producer,
            )
            with pytest.raises(
                RuntimeError,
                match="synthetic mid-roster retirement failure",
            ):
                serving.retire_python_producers()

        assert runtime.unretired_python_producer_ids == producer_ids[1:]
        serving.retire_python_producers()
        assert runtime.unretired_python_producer_ids == ()
        serving.close_clean(_WAIT_SECONDS)
        closed = True
    finally:
        if not closed:
            serving.abort_and_close()
