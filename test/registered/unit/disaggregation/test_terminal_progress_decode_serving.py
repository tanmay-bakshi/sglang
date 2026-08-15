import dataclasses
import inspect
import logging
import select
import sys
import threading
import types

import pytest
from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PackedChunkKey,
    PackedRequestKey,
)
from sglang.srt.disaggregation.common.staging_layout import StagingWriterId
from sglang.srt.disaggregation.nixl.packed_runtime import (
    PackedDecodeOwnerInventory,
    PackedDecodeOwnerSignal,
    PackedDecodeRuntime,
    PackedDecodeScatterBatch,
    PackedDecodeTerminalRegistration,
)
from sglang.srt.disaggregation.nixl.packed_staging_request import (
    PackedDecodeRequestTransaction,
    PackedRequestTransactionState,
)
from sglang.srt.disaggregation.terminal_progress.coordinator import (
    TerminalRequestCoordinatorManifest,
)
from sglang.srt.disaggregation.terminal_progress.decode_adoption import (
    TerminalDFlashDecodeAdoption,
)
from sglang.srt.disaggregation.terminal_progress.decode_scheduler_consumer import (
    PackedTerminalDecodeSchedulerRegistration,
)
from sglang.srt.disaggregation.terminal_progress.decode_serving import (
    PackedTerminalDecodeDeliveryTarget,
    PackedTerminalDecodeServing,
    PackedTerminalDecodeWireDelivery,
    PackedTerminalDecodeWork,
)
from sglang.srt.disaggregation.terminal_progress.evidence import (
    parse_terminal_progress_timing_log_line,
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
    PackedTerminalSourcePlan,
    PackedTerminalSourceWriter,
)
from sglang.srt.disaggregation.terminal_progress.wire import (
    TerminalWireReceiptImportNamespace,
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


class _TerminalDeviceCopyEvent:
    """Immediate terminal event used by CPU-only lifecycle composition tests."""

    def synchronize(self) -> None:
        """Model a completed device-copy synchronization boundary."""

    def query(self) -> bool:
        """Report terminal completion after synchronization.

        :returns: Always ``True`` for this successful fixture.
        """

        return True


def _terminal_dflash_adoption() -> TerminalDFlashDecodeAdoption:
    """Build a type-exact adoption envelope without requiring CUDA.

    :returns: Exact outer adoption authority consumed by decode wiring.
    """

    adoption = object.__new__(TerminalDFlashDecodeAdoption)
    object.__setattr__(adoption, "transaction_adoption", object())
    object.__setattr__(
        adoption,
        "device_value",
        types.SimpleNamespace(completion_event=_TerminalDeviceCopyEvent()),
    )
    return adoption


@dataclasses.dataclass(slots=True)
class _ActorState:
    """Mutable evidence behind a type-exact packed decode actor shell.

    :ivar source_plan: Frozen source identity plan.
    :ivar bindings: Actor-owned transactions by terminal binding digest.
    :ivar transaction_bindings: Reverse transaction-to-binding index.
    :ivar quarantined: Ambiguous actor identities retained against reuse.
    :ivar events: Ordered actor side effects.
    :ivar control_kinds: Native transitions earned by successive controls.
    """

    source_plan: PackedTerminalSourcePlan
    bindings: dict[bytes, PackedDecodeRequestTransaction] = dataclasses.field(
        default_factory=dict
    )
    transaction_bindings: dict[int, TerminalRequestBinding] = dataclasses.field(
        default_factory=dict
    )
    quarantined: set[bytes] = dataclasses.field(default_factory=set)
    events: list[str] = dataclasses.field(default_factory=list)
    control_kinds: list[NativeTerminalOwnerEventKind] = dataclasses.field(
        default_factory=lambda: [
            NativeTerminalOwnerEventKind.DECODE_WRITER_AGGREGATION_STARTED,
            NativeTerminalOwnerEventKind.DECODE_WRITER_MANIFEST_COMPLETED,
            NativeTerminalOwnerEventKind.DECODE_ACK_AGGREGATION_STARTED,
            NativeTerminalOwnerEventKind.DECODE_ACK_MANIFEST_COMPLETED,
        ]
    )
    fail_adoption: bool = False


class _CudaCompletion:
    """Deliver one scatter-tail callback directly into the native runtime."""

    _runtime: NativeTerminalRuntime
    armed: list[bytes]
    submitted: list[tuple[int, bytes]]

    def __init__(self, runtime: NativeTerminalRuntime) -> None:
        """Create one callback producer.

        :param runtime: Runtime receiving terminal scatter events.
        """

        self._runtime = runtime
        self.armed = []
        self.submitted = []

    def arm(self, binding_digest: bytes) -> None:
        """Record callback authority before stream attachment.

        :param binding_digest: Exact lifecycle being armed.
        """

        self.armed.append(binding_digest)

    def submit(self, stream_handle: int, binding_digest: bytes) -> None:
        """Deliver terminality after the already-enqueued start transition.

        :param stream_handle: Fixture scatter stream.
        :param binding_digest: Exact lifecycle receiving terminality.
        """

        self.submitted.append((stream_handle, binding_digest))
        self._runtime._owner.submit(
            NativeTerminalOwnerEvent(
                producer_id=_NATIVE_PRODUCER_ID,
                binding_digest=binding_digest,
                kind=NativeTerminalOwnerEventKind.DECODE_SCATTER_TERMINAL,
                enqueued_ns=2_000,
            )
        )


def _identity(
    marker: int,
    role: TerminalOwnerRole,
    rank: int,
    size: int,
) -> TerminalProcessIdentity:
    """Build one stable process identity.

    :param marker: Process-generation marker.
    :param role: Source or decode role.
    :param rank: Attention TP rank.
    :param size: Attention TP size.
    :returns: Exact process identity.
    """

    return TerminalProcessIdentity(
        process_generation=bytes((marker,)) * 16,
        role=role,
        tp_rank=rank,
        tp_size=size,
    )


def _identity_graph(
    decode_tp_size: int,
) -> tuple[
    tuple[TerminalRequestBinding, ...],
    PackedTerminalSourcePlan,
    TerminalRequestCoordinatorManifest,
]:
    """Build one source-to-decode request identity graph.

    :param decode_tp_size: Destination attention TP width.
    :returns: Destination bindings, source plan, and coordinator manifest.
    """

    key = PackedRequestKey(room_id=401, request_generation=b"g" * 16)
    source = _identity(0x31, TerminalOwnerRole.SOURCE, 0, 1)
    destinations = tuple(
        TerminalRequestBinding(
            request_key=key,
            owner=_identity(
                0x41 + rank, TerminalOwnerRole.DECODE, rank, decode_tp_size
            ),
            rank_manifest_digest=b"m" * 32,
            allocation_digest=b"a" * 32,
        )
        for rank in range(decode_tp_size)
    )
    source_plan = PackedTerminalSourcePlan(
        request_key=key,
        writers=(
            PackedTerminalSourceWriter(
                writer_id=StagingWriterId(
                    transfer_source_rank=0,
                    source_attn_tp_rank=0,
                    source_pp_rank=0,
                    source_cp_rank=0,
                ),
                process_identity=source,
            ),
        ),
        rank_manifest_digest=b"m" * 32,
        allocation_digest=b"a" * 32,
        publication_identity=TerminalPublicationIdentity(
            request_key=key,
            publisher_process_generation=source.process_generation,
            publication_generation=b"p" * 16,
        ),
        request_ready_issuer=destinations[0].owner,
    )
    manifest = TerminalRequestCoordinatorManifest(
        request_key=key,
        destination_bindings=destinations,
        recipient_bindings=(*destinations, *source_plan.source_bindings),
    )
    return destinations, source_plan, manifest


def _transaction(
    key: PackedRequestKey, owner: object
) -> PackedDecodeRequestTransaction:
    """Build the type-exact transaction shell used by the scheduler boundary.

    :param key: Exact request generation.
    :param owner: Mutable scheduler request retained by the transaction.
    :returns: Minimal type-exact transaction.
    """

    transaction = object.__new__(PackedDecodeRequestTransaction)
    transaction._request_key = key
    transaction._request_owner = owner
    transaction._lock = threading.RLock()
    transaction._state = PackedRequestTransactionState.PREPARED
    transaction._protocol = object()
    transaction._chunks = ()
    transaction._teardown_acks = set()
    transaction._auxiliary_outcome = None
    transaction._auxiliary_plan = types.SimpleNamespace(
        canonical_writer_id=StagingWriterId(
            transfer_source_rank=0,
            source_attn_tp_rank=0,
            source_pp_rank=0,
            source_cp_rank=0,
        )
    )
    return transaction


def _actor(
    source_plan: PackedTerminalSourcePlan,
) -> tuple[PackedDecodeRuntime, _ActorState]:
    """Build a type-exact actor with deterministic bound lifecycle methods.

    :param source_plan: Frozen source identity plan.
    :returns: Actor shell and its evidence state.
    """

    actor = object.__new__(PackedDecodeRuntime)
    state = _ActorState(source_plan=source_plan)

    def bind_terminal_owner(
        self: PackedDecodeRuntime,
        transaction: PackedDecodeRequestTransaction,
        binding: TerminalRequestBinding,
        plan: PackedTerminalSourcePlan,
    ) -> PackedDecodeTerminalRegistration:
        del self
        assert plan == state.source_plan
        state.bindings[binding.digest] = transaction
        state.transaction_bindings[id(transaction)] = binding
        state.events.append("bound")
        issuers = tuple(writer.process_identity for writer in plan.writers)
        if plan.request_ready_issuer not in issuers:
            issuers = (*issuers, plan.request_ready_issuer)
        return PackedDecodeTerminalRegistration(binding, issuers)

    def terminal_owner_transaction(
        self: PackedDecodeRuntime,
        binding_digest: bytes,
    ) -> PackedDecodeRequestTransaction:
        del self
        return state.bindings[binding_digest]

    def terminal_owner_binding(
        self: PackedDecodeRuntime,
        transaction: PackedDecodeRequestTransaction,
    ) -> TerminalRequestBinding:
        del self
        return state.transaction_bindings[id(transaction)]

    def terminal_owner_request_ready_issuer(
        self: PackedDecodeRuntime,
        binding_digest: bytes,
    ) -> TerminalProcessIdentity:
        del self
        assert binding_digest in state.bindings
        return state.source_plan.request_ready_issuer

    def bind_publication(
        self: PackedDecodeRuntime,
        transaction: PackedDecodeRequestTransaction,
        publication: object,
        routes: tuple[object, ...],
    ) -> NativeTerminalOwnerEventKind:
        del self, publication, routes
        assert id(transaction) in state.transaction_bindings
        state.events.append("published")
        return NativeTerminalOwnerEventKind.DECODE_ALLOCATION_PUBLISHED

    def handle_control(
        self: PackedDecodeRuntime,
        writer_id: StagingWriterId,
        message: object,
    ) -> tuple[PackedDecodeOwnerSignal, ...]:
        del self, message
        assert writer_id == state.source_plan.writers[0].writer_id
        kind = state.control_kinds.pop(0)
        binding = next(iter(state.transaction_bindings.values()))
        state.events.append(kind.name)
        return (
            PackedDecodeOwnerSignal(
                binding_digest=binding.digest,
                kind=kind,
                issuer=state.source_plan.writers[0].process_identity,
            ),
        )

    def begin_terminal_owner_scatter(
        self: PackedDecodeRuntime,
        transaction: PackedDecodeRequestTransaction,
    ) -> PackedDecodeScatterBatch:
        del self
        binding = state.transaction_bindings[id(transaction)]
        state.events.append("scatter")
        return PackedDecodeScatterBatch(
            binding_digest=binding.digest,
            chunk_keys=(
                PackedChunkKey(
                    room_id=binding.request_key.room_id,
                    chunk_id=0,
                    request_generation=binding.request_key.request_generation,
                ),
            ),
            stream_handle=17,
        )

    def confirm_terminal_owner_scatter_callback(
        self: PackedDecodeRuntime,
        transaction: PackedDecodeRequestTransaction,
        batch: PackedDecodeScatterBatch,
    ) -> None:
        del self
        assert (
            batch.binding_digest == state.transaction_bindings[id(transaction)].digest
        )
        state.events.append("callback")

    def complete_terminal_owner_scatter(
        self: PackedDecodeRuntime,
        transaction: PackedDecodeRequestTransaction,
    ) -> None:
        del self
        assert id(transaction) in state.transaction_bindings
        state.events.append("scatter_complete")

    def begin_terminal_owner_teardown(
        self: PackedDecodeRuntime,
        transaction: PackedDecodeRequestTransaction,
    ) -> None:
        del self
        assert id(transaction) in state.transaction_bindings
        state.events.append("teardown")

    def consume_terminal_owner_adoption(
        self: PackedDecodeRuntime,
        transaction: PackedDecodeRequestTransaction,
    ) -> object:
        del self
        state.events.append("adopt")
        if state.fail_adoption:
            raise RuntimeError("injected scheduler adoption failure")
        return transaction.request_owner

    def complete_terminal_owner_metadata_consumption(
        self: PackedDecodeRuntime,
        transaction: PackedDecodeRequestTransaction,
        *,
        dflash_adoption: object,
    ) -> None:
        del self
        assert id(transaction) in state.transaction_bindings
        assert dflash_adoption is not None
        state.events.append("metadata")

    def retire_terminal_owner_request(
        self: PackedDecodeRuntime,
        transaction: PackedDecodeRequestTransaction,
    ) -> None:
        del self
        binding = state.transaction_bindings.pop(id(transaction))
        del state.bindings[binding.digest]
        state.quarantined.discard(binding.digest)
        state.events.append("retired")

    def quarantine(
        self: PackedDecodeRuntime,
        transaction: PackedDecodeRequestTransaction,
        reason: str,
    ) -> None:
        del self
        assert len(reason) > 0
        state.quarantined.add(state.transaction_bindings[id(transaction)].digest)

    def terminal_owner_inventory(
        self: PackedDecodeRuntime,
    ) -> PackedDecodeOwnerInventory:
        del self
        return PackedDecodeOwnerInventory(
            active_bindings=tuple(sorted(state.bindings)),
            quarantined_bindings=tuple(sorted(state.quarantined)),
            in_flight_scatter_count=0,
            pending_adoption_count=0,
        )

    methods = {
        "bind_terminal_owner": bind_terminal_owner,
        "terminal_owner_transaction": terminal_owner_transaction,
        "terminal_owner_binding": terminal_owner_binding,
        "terminal_owner_request_ready_issuer": terminal_owner_request_ready_issuer,
        "bind_publication": bind_publication,
        "handle_control": handle_control,
        "begin_terminal_owner_scatter": begin_terminal_owner_scatter,
        "confirm_terminal_owner_scatter_callback": (
            confirm_terminal_owner_scatter_callback
        ),
        "complete_terminal_owner_scatter": complete_terminal_owner_scatter,
        "begin_terminal_owner_teardown": begin_terminal_owner_teardown,
        "consume_terminal_owner_adoption": consume_terminal_owner_adoption,
        "complete_terminal_owner_metadata_consumption": (
            complete_terminal_owner_metadata_consumption
        ),
        "retire_terminal_owner_request": retire_terminal_owner_request,
        "quarantine": quarantine,
        "terminal_owner_inventory": terminal_owner_inventory,
    }
    for name, method in methods.items():
        setattr(actor, name, types.MethodType(method, actor))
    return actor, state


def _runtime(
    local_identity: TerminalProcessIdentity,
    source_identity: TerminalProcessIdentity,
) -> NativeTerminalRuntime:
    """Construct one decode runtime with a complete frozen producer directory.

    :param local_identity: Decode process owned by the runtime.
    :param source_identity: Authenticated source control peer.
    :returns: Dormant process-lifetime runtime.
    """

    owner = NativeTerminalProcessIdentity.from_identity(local_identity)
    source = NativeTerminalProcessIdentity.from_identity(source_identity)
    role = NativeTerminalOwnerRole.DECODE
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
                name="python-source-receipt",
                producer_class=NativeTerminalProducerClass.RECEIPT,
                allowed_role=role,
                authenticated_issuer=source,
            ),
            delivery=NativeTerminalProducerDelivery.PYTHON,
        ),
        NativeTerminalRuntimeProducerSpec(
            registration=NativeTerminalProducerRegistration(
                producer_id=_REMOTE_CONTROL_PRODUCER_ID,
                name="python-source-control",
                producer_class=NativeTerminalProducerClass.CONTROL,
                allowed_role=role,
                authenticated_issuer=source,
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


def _registration(
    binding: TerminalRequestBinding,
    source_plan: PackedTerminalSourcePlan,
    scheduler_events: list[str],
) -> PackedTerminalDecodeSchedulerRegistration:
    """Build one callback-bearing scheduler registration.

    :param binding: Local decode lifecycle identity.
    :param source_plan: Complete source identity plan.
    :param scheduler_events: Mutable callback evidence ledger.
    :returns: Exact scheduler registration.
    """

    owner = object()

    def adopt_request(request: object) -> TerminalDFlashDecodeAdoption:
        """Record scheduler adoption and return exact DFlash authority.

        :param request: Exact retained scheduler request.
        :returns: Exact device-copy completion authority.
        """

        scheduler_events.append("adopt")
        return _terminal_dflash_adoption()

    return PackedTerminalDecodeSchedulerRegistration(
        binding=binding,
        source_plan=source_plan,
        transaction=_transaction(binding.request_key, owner),
        request_owner=owner,
        adopt_request=adopt_request,
        finalize_request=lambda request: scheduler_events.append("finalize"),
        cancel_request=lambda request: scheduler_events.append("cancel"),
        quarantine_request=lambda request, reason: scheduler_events.append(
            "quarantine"
        ),
    )


def _serving(
    decode_tp_size: int,
) -> tuple[
    PackedTerminalDecodeServing,
    NativeTerminalRuntime,
    _CudaCompletion,
    _ActorState,
    PackedTerminalDecodeSchedulerRegistration,
    TerminalRequestCoordinatorManifest,
    list[PackedTerminalDecodeWireDelivery],
    list[str],
    list[object],
]:
    """Construct one decode serving composition and all evidence ledgers.

    :param decode_tp_size: Destination attention TP width.
    :returns: Composition, owners, registration, manifest, and ledgers.
    """

    bindings, source_plan, manifest = _identity_graph(decode_tp_size)
    local_identity = bindings[0].owner
    runtime = _runtime(local_identity, source_plan.writers[0].process_identity)
    actor, actor_state = _actor(source_plan)
    completion = _CudaCompletion(runtime)
    deliveries: list[PackedTerminalDecodeWireDelivery] = []
    scheduler_events: list[str] = []
    fatal_inventories: list[object] = []
    serving = PackedTerminalDecodeServing(
        actor=actor,
        runtime=runtime,
        cuda_completion=completion,
        local_identity=local_identity,
        coordinator_issuer=TerminalWireReceiptIssuer(local_identity),
        coordinator_importers=tuple(
            TerminalWireReceiptImportNamespace(binding.owner) for binding in bindings
        ),
        clock_ns=lambda: 10_000,
        physical_capacity=8,
        process_fatal_handler=fatal_inventories.append,
        work=PackedTerminalDecodeWork(
            send_delivery=deliveries.append,
            observe_output=lambda output: None,
        ),
        retire_native_producers=lambda: runtime._owner.retire_python_producer(
            _NATIVE_PRODUCER_ID
        ),
    )
    registration = _registration(bindings[0], source_plan, scheduler_events)
    return (
        serving,
        runtime,
        completion,
        actor_state,
        registration,
        manifest,
        deliveries,
        scheduler_events,
        fatal_inventories,
    )


def _pump(serving: PackedTerminalDecodeServing) -> int:
    """Wait for and consume one complete projected runtime wave.

    :param serving: Open decode composition.
    :returns: Total actions and observations drained.
    """

    readable, _, _ = select.select(
        list(serving.runtime_filenos),
        [],
        [],
        _WAIT_SECONDS,
    )
    if len(readable) == 0:
        raise TimeoutError("decode composition runtime inbox did not wake")
    return serving.drain_runtime_actions()


def _drive_to_adoption(
    serving: PackedTerminalDecodeServing,
    registration: PackedTerminalDecodeSchedulerRegistration,
) -> None:
    """Drive allocation, authenticated control, scatter, and ACK completion.

    :param serving: Open composition carrying the registered request.
    :param registration: Exact scheduler and packed-actor ownership.
    """

    serving.allocation_published(
        registration.transaction,
        object.__new__(object),
        (),
    )
    writer_id = registration.source_plan.writers[0].writer_id
    serving.control_received(writer_id, object())
    serving.control_received(writer_id, object())
    _pump(serving)
    _pump(serving)
    serving.control_received(writer_id, object())
    serving.control_received(writer_id, object())
    _pump(serving)


def test_tp1_full_success_retires_every_authority_exactly_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Registration through local fan-out reaches an exact-zero clean close."""

    (
        serving,
        runtime,
        completion,
        actor_state,
        registration,
        manifest,
        deliveries,
        scheduler_events,
        _,
    ) = _serving(1)
    with caplog.at_level(logging.INFO):
        serving.start()
        try:
            serving.register_request(registration, manifest)
            registered = serving.inventory()
            assert registered.runtime.scheduler_live_count == 1
            assert registered.actor.active_bindings == (registration.binding.digest,)
            assert registered.scheduler_consumer.active_binding_digests == (
                registration.binding.digest,
            )
            assert registered.scheduler_serving.inbox.live_bindings == (
                registration.binding,
            )
            _drive_to_adoption(serving, registration)
            serving.drain_scheduler_at_loop_entry()
            _pump(serving)
            _pump(serving)

            inventory = serving.inventory()
            assert inventory.active_binding_digests == ()
            assert inventory.active_coordinator_manifest_digests == ()
            assert inventory.actor.active_bindings == ()
            assert inventory.scheduler_consumer.active_binding_digests == ()
            assert inventory.scheduler_serving.inbox.live_count == 0
            assert inventory.ready_coordinator_count == 1
            assert completion.armed == [registration.binding.digest]
            assert completion.submitted == [(17, registration.binding.digest)]
            assert scheduler_events == ["adopt", "finalize"]
            assert actor_state.events == [
                "bound",
                "published",
                "DECODE_WRITER_AGGREGATION_STARTED",
                "DECODE_WRITER_MANIFEST_COMPLETED",
                "scatter",
                "callback",
                "scatter_complete",
                "teardown",
                "DECODE_ACK_AGGREGATION_STARTED",
                "DECODE_ACK_MANIFEST_COMPLETED",
                "adopt",
                "metadata",
                "retired",
            ]
            assert len(deliveries) == 1
            assert deliveries[0].target is PackedTerminalDecodeDeliveryTarget.OWNER
            assert deliveries[0].recipient.role is TerminalOwnerRole.SOURCE

            timing_samples = tuple(
                sample
                for record in caplog.records
                if (
                    sample := parse_terminal_progress_timing_log_line(
                        record.getMessage()
                    )
                )
                is not None
            )
            assert len(timing_samples) == 5
            assert {sample.field.value for sample in timing_samples} == {
                "scatter_callback_delivery_ms",
                "ack_aggregation_ms",
                "request_global_coordination_ms",
                "scheduler_inbox_delay_ms",
                "metadata_consumption_ms",
            }
            assert {sample.sample_key for sample in timing_samples} == {
                "decode-rank-0"
            }

            serving.stop_admission_and_retire_producers()
            serving.close_clean(_WAIT_SECONDS)
        finally:
            if runtime.disposition.value != "stopped":
                serving.abort_and_close()


def test_runtime_fds_and_launch_binding_are_stable() -> None:
    """Reactor descriptors stay stable and post-forward binding stays gated."""

    serving, runtime, _, _, _, _, _, _, _ = _serving(1)
    serving.start()
    try:
        runtime_fds = serving.runtime_filenos
        scheduler_fd = serving.scheduler_fileno
        assert serving.runtime_filenos == runtime_fds
        assert serving.scheduler_fileno == scheduler_fd
        assert len(runtime_fds) == 5
        assert len({*runtime_fds, scheduler_fd}) == 6

        calls: list[str] = []
        result = serving.launch_and_bind_handoff(
            lambda: calls.append("launch") or 41,
            lambda value: calls.append("bind") or value + 1,
        )
        assert result == 42
        assert calls == ["launch", "bind"]
    finally:
        serving.abort_and_close()
        assert runtime.disposition.value == "stopped"


def test_serving_control_paths_contain_no_polling_sleep_or_collective() -> None:
    """The process composition exposes only reactor-driven progress paths."""

    source = inspect.getsource(PackedTerminalDecodeServing)
    forbidden = (
        "time.sleep(",
        ".poll(",
        "all_reduce(",
        "all_gather(",
        "barrier(",
    )
    assert tuple(token for token in forbidden if token in source) == ()


def test_tp2_coordinator_waits_for_every_rank_without_collective() -> None:
    """Point-to-point rank receipts fan out only after complete TP2 readiness."""

    serving, _, _, _, registration, manifest, deliveries, _, _ = _serving(2)
    serving.start()
    try:
        serving.register_request(registration, manifest)
        rank_issuers = tuple(
            TerminalWireReceiptIssuer(binding.owner)
            for binding in manifest.destination_bindings
        )
        rank_zero = rank_issuers[0].issue(
            binding=manifest.destination_bindings[0],
            kind=TerminalReceiptKind.LOCAL_DECODE_READY,
            outcome=TerminalReceiptOutcome.SUCCESS,
            terminal_timestamp_ns=100,
        )
        rank_one = rank_issuers[1].issue(
            binding=manifest.destination_bindings[1],
            kind=TerminalReceiptKind.LOCAL_DECODE_READY,
            outcome=TerminalReceiptOutcome.SUCCESS,
            terminal_timestamp_ns=110,
        )

        serving.coordinator_receipt_received(
            rank_zero.wire_receipt,
            manifest.destination_bindings[0].owner,
        )
        assert deliveries == []
        assert serving.inventory().active_coordinator_manifest_digests == (
            manifest.digest,
        )

        serving.coordinator_receipt_received(
            rank_one.wire_receipt,
            manifest.destination_bindings[1].owner,
        )
        assert serving.inventory().active_coordinator_manifest_digests == ()
        assert tuple(delivery.recipient for delivery in deliveries) == (
            manifest.destination_bindings[1].owner,
            registration.source_plan.writers[0].process_identity,
        )
        assert all(
            delivery.target is PackedTerminalDecodeDeliveryTarget.OWNER
            for delivery in deliveries
        )
    finally:
        serving.abort_and_close()


def test_runtime_fatal_abort_preserves_ambiguous_actor_and_scheduler_resources() -> (
    None
):
    """Owner death quarantines rather than releases a registered generation."""

    (
        serving,
        runtime,
        _,
        _,
        registration,
        manifest,
        _,
        scheduler_events,
        fatal_inventories,
    ) = _serving(1)
    serving.start()
    serving.register_request(registration, manifest)

    runtime.begin_abort()
    _pump(serving)
    inventory = serving.abort_and_close()

    assert inventory.owner_dead_marked
    assert inventory.active_binding_digests == (registration.binding.digest,)
    assert inventory.actor.active_bindings == (registration.binding.digest,)
    assert inventory.actor.quarantined_bindings == (registration.binding.digest,)
    assert inventory.scheduler_consumer.active_binding_digests == (
        registration.binding.digest,
    )
    assert inventory.scheduler_consumer.quarantined_binding_digests == (
        registration.binding.digest,
    )
    assert scheduler_events == ["quarantine"]
    assert len(fatal_inventories) == 1


def test_scheduler_adoption_failure_aborts_and_quarantines_exact_request() -> None:
    """A scheduler-boundary exception preserves every ambiguous resource."""

    (
        serving,
        runtime,
        _,
        actor_state,
        registration,
        manifest,
        _,
        scheduler_events,
        fatal_inventories,
    ) = _serving(1)
    actor_state.fail_adoption = True
    serving.start()
    serving.register_request(registration, manifest)
    _drive_to_adoption(serving, registration)

    with pytest.raises(RuntimeError, match="injected scheduler adoption failure"):
        serving.drain_scheduler_at_loop_entry()
    inventory = serving.abort_and_close()

    assert runtime.disposition.value == "stopped"
    assert inventory.owner_dead_marked
    assert inventory.actor.active_bindings == (registration.binding.digest,)
    assert inventory.actor.quarantined_bindings == (registration.binding.digest,)
    assert inventory.scheduler_consumer.quarantined_binding_digests == (
        registration.binding.digest,
    )
    assert scheduler_events == ["quarantine"]
    assert len(fatal_inventories) == 1


def test_scheduler_callback_failure_after_native_completion_stays_fail_closed() -> None:
    """Post-completion scheduler failure uses request-local fatal authority."""

    (
        serving,
        runtime,
        _,
        _,
        registration,
        manifest,
        _,
        scheduler_events,
        fatal_inventories,
    ) = _serving(1)

    def fail_adoption(request: object) -> None:
        """Fail after native adoption accounting has completed.

        :param request: Exact scheduler request owner.
        """

        assert request is registration.request_owner
        scheduler_events.append("adopt_failed")
        raise RuntimeError("injected scheduler callback failure")

    registration = dataclasses.replace(registration, adopt_request=fail_adoption)
    serving.start()
    serving.register_request(registration, manifest)
    _drive_to_adoption(serving, registration)

    with pytest.raises(RuntimeError, match="injected scheduler callback failure"):
        serving.drain_scheduler_at_loop_entry()
    inventory = serving.abort_and_close()

    assert runtime.disposition.value == "stopped"
    assert inventory.owner_dead_marked
    assert inventory.actor.active_bindings == (registration.binding.digest,)
    assert inventory.actor.quarantined_bindings == (registration.binding.digest,)
    assert inventory.scheduler_consumer.quarantined_binding_digests == (
        registration.binding.digest,
    )
    assert scheduler_events == ["adopt_failed", "quarantine"]
    assert len(fatal_inventories) == 1
