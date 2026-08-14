import dataclasses
import hashlib
import logging
import os
import select
import sys

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
    NativeTerminalOwnerEvent,
    NativeTerminalOwnerEventKind,
    NativeTerminalOwnerRole,
    NativeTerminalProcessIdentity,
    NativeTerminalProducerClass,
    NativeTerminalProducerRegistration,
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
from sglang.srt.disaggregation.terminal_progress.source_plan import (
    PackedTerminalSourceIdentityPlan,
)
from sglang.srt.disaggregation.terminal_progress.source_serving import (
    PackedTerminalSourceServing,
    PackedTerminalSourceWork,
)
from sglang.srt.disaggregation.terminal_progress.source_wiring import (
    PackedTerminalSourceMetric,
    PackedTerminalSourceSubmission,
)
from sglang.srt.disaggregation.terminal_progress.wire import (
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
        source_work_capacity=8,
        decode_work_capacity=8,
        publisher_capacity=8,
        observation_capacity=64,
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
        output_projection=_Projection(payload=b"output"),
        producer_event_generation=b"e" * 16,
        transport_submission=("packed", identity.local_binding.digest),
    )


def _serving(
    identity: PackedTerminalSourceIdentityPlan,
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
    :returns: Serving, runtime, publisher, metrics, work labels, and fatal inventories.
    """

    runtime = _runtime(identity)
    publisher = _Publisher()
    metrics = _Metrics()
    work_labels: list[str] = []
    fatal_inventories: list[object] = []
    serving = PackedTerminalSourceServing(
        runtime=runtime,
        local_identity=identity.local_binding.owner,
        publisher=publisher,
        metrics_sink=metrics,
        clock_ns=lambda: 1_000,
        physical_capacity=8,
        process_fatal_handler=fatal_inventories.append,
        grouped_nixl=_EmptyGroupedNixlOwner(),
        work=PackedTerminalSourceWork(
            post_gather=lambda submission, action: work_labels.append("gather"),
            send_outcomes=lambda submission, action: work_labels.append("outcome"),
            send_ack=lambda submission, action: work_labels.append("ack"),
            quarantine=lambda action: work_labels.append("quarantine"),
            observe_output=lambda output: None,
        ),
        retire_native_producers=lambda: runtime._owner.retire_python_producer(
            _NATIVE_PRODUCER_ID
        ),
        retire_submission=lambda submission: None,
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


def test_composition_binds_both_scheduler_owners_before_lifecycle() -> None:
    """The native lifecycle cannot race either scheduler-owned registry."""

    identity = _identity()
    serving, _, _, _, _, _ = _serving(identity)
    serving.start()
    release_calls: list[int] = []
    try:
        serving.bind_submission(
            _submission(identity),
            lambda submission: release_calls.append(1),
        )
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
        digest = identity.local_binding.digest
        serving.wiring.producer_completed(digest)
        _pump(serving, runtime)
        assert work_labels == ["gather"]
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
        serving.bind_submission(_submission(identity), lambda submission: None)
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
    finally:
        serving.abort_and_close()
