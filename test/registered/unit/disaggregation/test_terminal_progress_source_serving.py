import concurrent.futures
import dataclasses
import hashlib
import logging
import os
import select
import sys
import threading
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
    NativeTerminalOwnerAction,
    NativeTerminalOwnerActionKind,
    NativeTerminalOwnerEvent,
    NativeTerminalOwnerEventKind,
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
    NativeTerminalProducerDelivery,
    NativeTerminalRuntime,
    NativeTerminalRuntimeProducerSpec,
)
from sglang.srt.disaggregation.terminal_progress.scheduler_inbox import (
    SchedulerReceiptPublishResult,
)
from sglang.srt.disaggregation.terminal_progress.scheduler_serving import (
    TerminalSchedulerActionPublicationError,
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


def _runtime(identity: PackedTerminalSourceIdentityPlan) -> NativeTerminalRuntime:
    """Construct one source runtime with every authority pre-registered.

    :param identity: Exact source identity graph.
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
        enable_forward_independent_handoff=False,
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
    :returns: Serving, runtime, publisher, metrics, work labels, and fatal inventories.
    """

    runtime = _runtime(identity)
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
        assert serving._drain_source_work_actions() == 1
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

        assert serving._drain_source_work_actions() == 4
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
            assert runtime._consumer_pending == {action.action_id: action}
            assert runtime._inbox_claimed_action_ids == {action.action_id}
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


def test_mid_claim_failure_retains_and_reconciles_complete_population(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed member claim cannot strand its later dequeued siblings."""

    identity = _identity()
    serving, runtime, _, _, _, _ = _serving(identity)
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
    claim_count = 0

    def fail_first_native_claim(action: NativeTerminalOwnerAction) -> None:
        """Fail one member while allowing the later member to be retained.

        :param action: Exact forward-independent action being claimed.
        """

        nonlocal claim_count
        claim_count += 1
        if claim_count == 1:
            raise RuntimeError("synthetic mid-population claim failure")

    serving.start()
    try:
        for action in actions:
            runtime._route_action(action)
        monkeypatch.setattr(
            runtime,
            "_claim_forward_independent_handoff",
            fail_first_native_claim,
        )
        with pytest.raises(
            RuntimeError,
            match="source runtime action batch claim failed",
        ):
            serving.drain_runtime_actions()

        assert claim_count == 2
        action_ids = frozenset(action.action_id for action in actions)
        with runtime._condition:
            assert action_ids.isdisjoint(runtime._consumer_pending)
            assert action_ids.isdisjoint(runtime._inbox_claimed_action_ids)
        runtime.snapshot()
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
    serving, runtime, publisher, _, work_labels, _ = _serving(identity)
    submission = _submission(identity)
    release_calls: list[PackedTerminalSourceSubmission] = []
    serving.start()
    try:
        serving.bind_submission(submission, release_calls.append)
        serving.attach_producer_completion(submission)
        digest = identity.local_binding.digest
        assert serving.packed_ready(digest)
        assert runtime.wait_for_output_projection(_WAIT_SECONDS)
        assert serving._gather_worker.wait_until_idle(_WAIT_SECONDS)
        _pump(serving, runtime)
        assert work_labels == ["gather"]
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
        _pump(serving, runtime)
        assert work_labels == ["gather", "outcome"]

        serving.wiring.teardown_received(digest, identity.request_ready_issuer)
        _pump(serving, runtime)
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
        _pump(serving, runtime)
        assert len(publisher.values) == 1
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
        _pump(serving, runtime)
        assert serving.inventory().wiring.active_binding_digests == ()
        assert serving.inventory().wiring.completion_required_binding_digests == ()

        serving.stop_admission_and_retire_producers()
        serving.close_clean(_WAIT_SECONDS)
    finally:
        if runtime.disposition.value != "stopped":
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
