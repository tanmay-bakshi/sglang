import concurrent.futures
import contextlib
import dataclasses
import inspect
import json
import logging
import os
import select
import sys
import threading
import time
import types
from collections.abc import Callable

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
from sglang.srt.disaggregation.terminal_progress.clock import SystemTerminalOwnerClock
from sglang.srt.disaggregation.terminal_progress.coordinator import (
    TerminalRequestCoordinatorManifest,
)
from sglang.srt.disaggregation.terminal_progress.decode_adoption import (
    TerminalDFlashDecodeAdoption,
)
from sglang.srt.disaggregation.terminal_progress.decode_scatter_worker import (
    TERMINAL_DECODE_SCATTER_TIMING_LOG_PREFIX,
    PackedTerminalDecodeScatterWorker,
    PackedTerminalDecodeScatterWorkerDisposition,
    PackedTerminalDecodeScatterWorkerError,
    PackedTerminalDecodeScatterWorkerInventory,
)
from sglang.srt.disaggregation.terminal_progress.decode_scheduler_consumer import (
    PackedTerminalDecodeSchedulerRegistration,
)
from sglang.srt.disaggregation.terminal_progress.decode_serving import (
    PackedTerminalDecodeDeliveryTarget,
    PackedTerminalDecodeServing,
    PackedTerminalDecodeServingInventory,
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
    NativeTerminalOwnerAction,
    NativeTerminalOwnerActionKind,
    NativeTerminalOwnerEvent,
    NativeTerminalOwnerEventKind,
    NativeTerminalOwnerRole,
    NativeTerminalProcessIdentity,
    NativeTerminalProducerClass,
    NativeTerminalProducerRegistration,
    NativeTerminalReceipt,
    NativeTerminalReceiptKind,
    NativeTerminalReceiptOutcome,
    NativeTerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.receipts import (
    TerminalReceiptKind,
    TerminalReceiptOutcome,
)
from sglang.srt.disaggregation.terminal_progress.runtime import (
    NativeTerminalProducerDelivery,
    NativeTerminalRuntime,
    NativeTerminalRuntimeDisposition,
    NativeTerminalRuntimeOverflowError,
    NativeTerminalRuntimeProducerSpec,
)
from sglang.srt.disaggregation.terminal_progress.scheduler_serving import (
    TerminalSchedulerActionPublicationError,
)
from sglang.srt.disaggregation.terminal_progress.source_plan import (
    PackedTerminalSourcePlan,
    PackedTerminalSourceWriter,
)
from sglang.srt.disaggregation.terminal_progress.wire import (
    TerminalWireReceipt,
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
    :ivar scatter_entered: Set when the dedicated worker starts submission.
    :ivar scatter_release: Optional test barrier for a blocking submission.
    :ivar scatter_thread_ids: Threads which executed scatter submission.
    :ivar cuda_binding_thread_ids: Threads which established CUDA affinity.
    :ivar retirement_worker_inventories: Worker state at CUDA retirement.
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
    fail_scatter: bool = False
    block_scatter: bool = False
    scatter_entered: threading.Event = dataclasses.field(
        default_factory=threading.Event
    )
    scatter_release: threading.Event = dataclasses.field(
        default_factory=threading.Event
    )
    scatter_thread_ids: list[int] = dataclasses.field(default_factory=list)
    cuda_binding_thread_ids: list[int] = dataclasses.field(default_factory=list)
    retirement_worker_inventories: list[PackedTerminalDecodeScatterWorkerInventory] = (
        dataclasses.field(default_factory=list)
    )


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

    def authorize_delivery(self, binding_digest: bytes) -> bool:
        """Reject source-only authorization on a decode producer.

        :param binding_digest: Candidate decode binding.
        :returns: This method does not return.
        :raises RuntimeError: Always, because decode delivery is ungated.
        """

        raise RuntimeError("decode callback delivery is not source-authorized")


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
        assert plan.writers == state.source_plan.writers
        assert plan.request_ready_issuer == state.source_plan.request_ready_issuer
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
        state.scatter_thread_ids.append(threading.get_ident())
        state.scatter_entered.set()
        if state.block_scatter and not state.scatter_release.wait(_WAIT_SECONDS):
            raise TimeoutError("test scatter release timed out")
        if state.fail_scatter:
            raise RuntimeError("injected decode scatter failure")
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
    *,
    enable_forward_independent_handoff: bool,
    testing: bool = False,
) -> NativeTerminalRuntime:
    """Construct one decode runtime with a complete frozen producer directory.

    :param local_identity: Decode process owned by the runtime.
    :param source_identity: Authenticated source control peer.
    :param enable_forward_independent_handoff: Whether native delivery authority
        gates scheduler launches.
    :param testing: Whether to expose deterministic native test controls.
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
        source_gather_capacity=8,
        source_work_capacity=8,
        decode_scatter_capacity=8,
        decode_work_capacity=8,
        publisher_capacity=8,
        observation_capacity=64,
        enable_forward_independent_handoff=enable_forward_independent_handoff,
        testing=testing,
    )


def _standalone_scatter_runtime() -> tuple[
    NativeTerminalRuntime,
    TerminalRequestBinding,
]:
    """Start one runtime for direct scatter-worker lifecycle qualification.

    :returns: Running runtime and its stable decode binding fixture.
    """

    bindings, source_plan, _ = _identity_graph(1)
    runtime = _runtime(
        bindings[0].owner,
        source_plan.writers[0].process_identity,
        enable_forward_independent_handoff=False,
    )
    runtime.start()
    return runtime, bindings[0]


def _scatter_action(
    binding: TerminalRequestBinding,
    *,
    action_id: int,
    commit_timestamp_ns: int,
) -> NativeTerminalOwnerAction:
    """Build one direct-inbox scatter authority.

    :param binding: Exact decode request binding.
    :param action_id: Globally unique native action identity.
    :param commit_timestamp_ns: Native action commit timestamp.
    :returns: Immutable decode scatter action.
    """

    return NativeTerminalOwnerAction(
        action_id=action_id,
        kind=NativeTerminalOwnerActionKind.DECODE_SCATTER_READY,
        binding=NativeTerminalRequestBinding.from_binding(binding),
        commit_timestamp_ns=commit_timestamp_ns,
        receipt=None,
    )


def _adoption_action(
    binding: TerminalRequestBinding,
    *,
    action_id: int,
) -> NativeTerminalOwnerAction:
    """Build one exact scheduler adoption authority.

    :param binding: Exact decode lifecycle targeted by the action.
    :param action_id: Globally unique native action identity.
    :returns: Immutable owner-minted adoption action.
    """

    native_binding = NativeTerminalRequestBinding.from_binding(binding)
    return NativeTerminalOwnerAction(
        action_id=action_id,
        kind=NativeTerminalOwnerActionKind.ADOPTION_READY,
        binding=native_binding,
        commit_timestamp_ns=action_id,
        receipt=NativeTerminalReceipt(
            binding=native_binding,
            issuer=NativeTerminalProcessIdentity.from_identity(binding.owner),
            kind=NativeTerminalReceiptKind.ADOPTION_READY,
            outcome=NativeTerminalReceiptOutcome.SUCCESS,
            terminal_timestamp_ns=action_id,
            nonce=action_id.to_bytes(16, "big"),
        ),
    )


def _additional_decode_request(
    registration: PackedTerminalDecodeSchedulerRegistration,
    scheduler_events: list[str],
) -> tuple[
    PackedTerminalDecodeSchedulerRegistration,
    TerminalRequestCoordinatorManifest,
]:
    """Build another generation owned by the same decode process.

    :param registration: Existing request supplying the frozen process topology.
    :param scheduler_events: Mutable callback evidence ledger.
    :returns: Additional scheduler registration and coordinator manifest.
    """

    request_key = PackedRequestKey(
        room_id=402,
        request_generation=b"h" * 16,
    )
    binding = TerminalRequestBinding(
        request_key=request_key,
        owner=registration.binding.owner,
        rank_manifest_digest=b"n" * 32,
        allocation_digest=b"b" * 32,
    )
    original_plan = registration.source_plan
    source_plan = PackedTerminalSourcePlan(
        request_key=request_key,
        writers=original_plan.writers,
        rank_manifest_digest=b"n" * 32,
        allocation_digest=b"b" * 32,
        publication_identity=TerminalPublicationIdentity(
            request_key=request_key,
            publisher_process_generation=(
                original_plan.publication_identity.publisher_process_generation
            ),
            publication_generation=b"q" * 16,
        ),
        request_ready_issuer=original_plan.request_ready_issuer,
    )
    manifest = TerminalRequestCoordinatorManifest(
        request_key=request_key,
        destination_bindings=(binding,),
        recipient_bindings=(binding, *source_plan.source_bindings),
    )
    return _registration(binding, source_plan, scheduler_events), manifest


def _enqueue_scatter_action(
    runtime: NativeTerminalRuntime,
    action: NativeTerminalOwnerAction,
) -> None:
    """Publish an action through the runtime's authoritative retention order.

    :param runtime: Runtime owning the direct scatter inbox.
    :param action: Exact one-shot scatter authority.
    """

    runtime._enqueue_consumer_action(runtime.decode_scatter_actions, action)


def _close_standalone_runtime(
    runtime: NativeTerminalRuntime,
    *,
    abort: bool,
) -> None:
    """Retire every fixture producer and close an already-drained runtime.

    :param runtime: Runtime whose worker inbox owns exact zero actions.
    :param abort: Whether the runtime is closing fail closed.
    """

    if runtime.disposition is NativeTerminalRuntimeDisposition.STOPPED:
        return
    if abort:
        runtime.begin_abort("standalone scatter worker fixture abort")
    else:
        runtime.stop_admission()
    for producer_id in runtime.python_producer_ids:
        runtime.retire_python_producer(producer_id)
    runtime._owner.retire_python_producer(_NATIVE_PRODUCER_ID)
    runtime.join_producers()
    if abort:
        runtime.finish_abort_close()
    else:
        runtime.close_clean()


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
    *,
    testing: bool = False,
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
    :param testing: Whether to expose deterministic native test controls.
    :returns: Composition, owners, registration, manifest, and ledgers.
    """

    bindings, source_plan, manifest = _identity_graph(decode_tp_size)
    local_identity = bindings[0].owner
    runtime = _runtime(
        local_identity,
        source_plan.writers[0].process_identity,
        enable_forward_independent_handoff=True,
        testing=testing,
    )
    actor, actor_state = _actor(source_plan)
    completion = _CudaCompletion(runtime)
    deliveries: list[PackedTerminalDecodeWireDelivery] = []
    scheduler_events: list[str] = []
    fatal_inventories: list[object] = []
    clock_ns = SystemTerminalOwnerClock().now_ns
    serving: PackedTerminalDecodeServing

    def retire_native_producers() -> None:
        """Retire CUDA authority only after scatter ownership is exact zero."""

        state = actor_state
        state.retirement_worker_inventories.append(serving.inventory().scatter_worker)
        runtime._owner.retire_python_producer(_NATIVE_PRODUCER_ID)

    serving = PackedTerminalDecodeServing(
        actor=actor,
        runtime=runtime,
        cuda_completion=completion,
        local_identity=local_identity,
        coordinator_issuer=TerminalWireReceiptIssuer(local_identity),
        coordinator_importers=tuple(
            TerminalWireReceiptImportNamespace(binding.owner) for binding in bindings
        ),
        clock_ns=clock_ns,
        physical_capacity=8,
        process_fatal_handler=fatal_inventories.append,
        work=PackedTerminalDecodeWork(
            send_delivery=deliveries.append,
            observe_output=lambda output: None,
        ),
        bind_scatter_cuda_device=lambda: actor_state.cuda_binding_thread_ids.append(
            threading.get_ident()
        ),
        retire_native_producers=retire_native_producers,
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


def _pump_until(
    serving: PackedTerminalDecodeServing,
    predicate: Callable[[PackedTerminalDecodeServingInventory], bool],
) -> None:
    """Drain asynchronous runtime waves until one inventory fact is true.

    :param serving: Open decode composition.
    :param predicate: Exact post-drain state required by the caller.
    """

    deadline = time.monotonic() + _WAIT_SECONDS
    while not predicate(serving.inventory()):
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise TimeoutError("decode composition state predicate expired")
        readable, _, _ = select.select(
            list(serving.runtime_filenos),
            [],
            [],
            remaining,
        )
        if len(readable) == 0:
            raise TimeoutError("decode composition state predicate did not wake")
        serving.drain_runtime_actions()


def _drive_to_adoption(
    serving: PackedTerminalDecodeServing,
    registration: PackedTerminalDecodeSchedulerRegistration,
    actor_state: _ActorState,
) -> None:
    """Drive allocation, authenticated control, scatter, and ACK completion.

    :param serving: Open composition carrying the registered request.
    :param registration: Exact scheduler and packed-actor ownership.
    :param actor_state: Mutable actor evidence advanced by worker and reactor.
    """

    serving.allocation_published(
        registration.transaction,
        object.__new__(object),
        (),
    )
    writer_id = registration.source_plan.writers[0].writer_id
    serving.control_received(writer_id, object())
    serving.control_received(writer_id, object())
    _pump_until(
        serving,
        lambda inventory: "teardown" in actor_state.events,
    )
    serving.control_received(writer_id, object())
    serving.control_received(writer_id, object())
    _pump_until(
        serving,
        lambda inventory: len(inventory.scheduler_serving.retained_action_ids) == 1,
    )


def _stage_runtime_adoption(
    serving: PackedTerminalDecodeServing,
    runtime: NativeTerminalRuntime,
    registration: PackedTerminalDecodeSchedulerRegistration,
    *,
    expected_scheduler_count: int,
) -> None:
    """Drive one reservation-backed adoption into the runtime scheduler inbox.

    :param serving: Open composition whose worker owns scatter execution.
    :param runtime: Running decode runtime owning the lifecycle.
    :param registration: Exact registered request generation.
    :param expected_scheduler_count: Total queued scheduler actions after this
        lifecycle reaches adoption.
    """

    binding_digest = registration.binding.digest
    runtime.submit(
        _LOCAL_PRODUCER_ID,
        binding_digest,
        NativeTerminalOwnerEventKind.DECODE_ALLOCATION_PUBLISHED,
    )
    runtime.submit(
        _REMOTE_CONTROL_PRODUCER_ID,
        binding_digest,
        NativeTerminalOwnerEventKind.DECODE_WRITER_AGGREGATION_STARTED,
    )
    runtime.submit(
        _REMOTE_CONTROL_PRODUCER_ID,
        binding_digest,
        NativeTerminalOwnerEventKind.DECODE_WRITER_MANIFEST_COMPLETED,
    )
    expires_at = time.monotonic() + _WAIT_SECONDS
    while (
        runtime.decode_work_actions.snapshot().queued_count != 1
        and time.monotonic() < expires_at
    ):
        pass
    assert runtime.decode_work_actions.snapshot().queued_count == 1
    assert serving.drain_decode_work_actions() == 1
    runtime.submit(
        _REMOTE_CONTROL_PRODUCER_ID,
        binding_digest,
        NativeTerminalOwnerEventKind.DECODE_ACK_AGGREGATION_STARTED,
    )
    runtime.submit(
        _REMOTE_CONTROL_PRODUCER_ID,
        binding_digest,
        NativeTerminalOwnerEventKind.DECODE_ACK_MANIFEST_COMPLETED,
    )
    expires_at = time.monotonic() + _WAIT_SECONDS
    while (
        runtime.scheduler_actions.snapshot().queued_count != expected_scheduler_count
        and time.monotonic() < expires_at
    ):
        pass
    assert runtime.scheduler_actions.snapshot().queued_count == expected_scheduler_count


def _assert_fail_closed_handoff_zero(
    runtime: NativeTerminalRuntime,
    inventory: PackedTerminalDecodeServingInventory,
) -> None:
    """Assert exact-zero delayed delivery authority after fail-closed close.

    :param runtime: Closed runtime supplying final disposition.
    :param inventory: Last pre-close serving inventory.
    """

    assert inventory.runtime.decode_scheduler_publication_handoff_count == 0
    assert inventory.runtime.owner.active_handoff_action_count == 0
    assert inventory.runtime.owner.abort_settled_handoff_delivery_count == 0
    assert runtime.disposition is NativeTerminalRuntimeDisposition.STOPPED


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
            _drive_to_adoption(serving, registration, actor_state)
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
            assert actor_state.scatter_thread_ids == (
                actor_state.cuda_binding_thread_ids
            )
            assert actor_state.scatter_thread_ids != [threading.get_ident()]
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
            assert {sample.sample_key for sample in timing_samples} == {"decode-rank-0"}

            serving.stop_admission_and_retire_producers()
            assert len(actor_state.retirement_worker_inventories) == 1
            retirement_worker = actor_state.retirement_worker_inventories[0]
            assert not retirement_worker.thread_alive
            assert retirement_worker.retained_action_count == 0
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


def test_adoption_settles_only_after_qualified_scheduler_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A waiting scheduler cannot consume before atomic native settlement.

    :param monkeypatch: Scoped settlement barrier on the publication seam.
    """

    (
        serving,
        runtime,
        _,
        actor_state,
        registration,
        manifest,
        _,
        scheduler_events,
        _,
    ) = _serving(1, testing=True)
    settlement_entered = threading.Event()
    release_settlement = threading.Event()
    consumer_entered = threading.Event()
    dispatch_captured = False
    settle_handoff = runtime.settle_decode_scheduler_publication_handoff

    def record_settlement(action: NativeTerminalOwnerAction) -> None:
        """Hold the commit after insertion but before scheduler visibility.

        :param action: Published adoption action releasing launch exclusion.
        """

        settlement_entered.set()
        if not release_settlement.wait(_WAIT_SECONDS):
            raise TimeoutError("decode publication settlement was not released")
        settle_handoff(action)

    monkeypatch.setattr(
        runtime,
        "settle_decode_scheduler_publication_handoff",
        record_settlement,
    )
    drain_scheduler_inbox = serving._scheduler_serving._inbox.drain_at_loop_entry

    def observe_scheduler_drain(
        consume: Callable[[TerminalWireReceipt], None],
    ) -> tuple[TerminalWireReceipt, ...]:
        """Expose the scheduler-thread drain before it acquires inbox state.

        :param consume: Exact scheduler receipt consumer.
        :returns: Exact receipts consumed by the real inbox drain.
        """

        consumer_entered.set()
        return drain_scheduler_inbox(consume)

    monkeypatch.setattr(
        serving._scheduler_serving._inbox,
        "drain_at_loop_entry",
        observe_scheduler_drain,
    )

    def release_after_scheduler_contention() -> None:
        """Release publication only after the scheduler reaches its drain."""

        assert consumer_entered.wait(_WAIT_SECONDS)
        assert scheduler_events == []
        release_settlement.set()

    serving.start()
    try:
        serving.register_request(registration, manifest)
        serving.allocation_published(
            registration.transaction,
            object.__new__(object),
            (),
        )
        writer_id = registration.source_plan.writers[0].writer_id
        serving.control_received(writer_id, object())
        serving.control_received(writer_id, object())
        _pump_until(
            serving,
            lambda inventory: "teardown" in actor_state.events,
        )
        serving.control_received(writer_id, object())
        serving.control_received(writer_id, object())
        expires_at = time.monotonic() + _WAIT_SECONDS
        while (
            runtime.scheduler_actions.snapshot().queued_count != 1
            and time.monotonic() < expires_at
        ):
            pass
        assert runtime.scheduler_actions.snapshot().queued_count == 1
        transition_floor = runtime.snapshot().owner.transition_count
        submit = runtime.submit

        def serialize_local_ready_submit(
            producer_id: int,
            binding_digest: bytes,
            kind: NativeTerminalOwnerEventKind,
            *,
            receipt: NativeTerminalReceipt | None = None,
            reason: str | None = None,
            enqueued_ns: int | None = None,
        ) -> None:
            """Let prior scheduler events commit before the held local-ready event.

            :param producer_id: Exact producer authority.
            :param binding_digest: Target lifecycle digest.
            :param kind: Native lifecycle event kind.
            :param receipt: Optional authenticated receipt.
            :param reason: Optional failure evidence.
            :param enqueued_ns: Optional producer timestamp.
            """

            nonlocal dispatch_captured
            if kind is NativeTerminalOwnerEventKind.DECODE_LOCAL_READY_ISSUED:
                expires_at = time.monotonic() + _WAIT_SECONDS
                while (
                    runtime.snapshot().owner.transition_count < transition_floor + 2
                    and time.monotonic() < expires_at
                ):
                    pass
                if runtime.snapshot().owner.transition_count < transition_floor + 2:
                    raise TimeoutError(
                        "decode adoption predecessors did not become quiescent"
                    )
                runtime._owner.hold_decode_delivery_dispatch_for_testing()
            submit(
                producer_id,
                binding_digest,
                kind,
                receipt=receipt,
                reason=reason,
                enqueued_ns=enqueued_ns,
            )
            if kind is NativeTerminalOwnerEventKind.DECODE_LOCAL_READY_ISSUED:
                dispatch_captured = True

        monkeypatch.setattr(runtime, "submit", serialize_local_ready_submit)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            publisher = executor.submit(serving.drain_scheduler_actions)
            try:
                assert settlement_entered.wait(_WAIT_SECONDS)
                release = executor.submit(release_after_scheduler_contention)
                consumed = serving.drain_scheduler_at_loop_entry()
                release.result(timeout=_WAIT_SECONDS)
            finally:
                release_settlement.set()

            assert publisher.result(timeout=_WAIT_SECONDS) == 1

        assert tuple(action.kind for action in consumed) == (
            NativeTerminalOwnerActionKind.ADOPTION_READY,
        )
        assert scheduler_events == ["adopt", "finalize"]

        post_adoption = runtime.snapshot().owner
        assert post_adoption.registered_decode_delivery_reservation_count == 2
        assert post_adoption.decode_delivery_reservation_count == 1
        assert post_adoption.decode_reservation_backed_handoff_action_count == 0
        assert post_adoption.queued_output_count == 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            launch = executor.submit(runtime.begin_scheduler_launch_handoff)
            expires_at = time.monotonic() + _WAIT_SECONDS
            while (
                not runtime.snapshot().owner.scheduler_launch_handoff_begin_active
                and time.monotonic() < expires_at
            ):
                pass
            waiting = runtime.snapshot().owner
            assert waiting.scheduler_launch_handoff_begin_active
            action_watermark = waiting.scheduler_launch_handoff_begin_watermark
            reservation_watermark = (
                waiting.scheduler_launch_handoff_begin_decode_reservation_watermark
            )
            assert action_watermark is not None
            assert reservation_watermark is not None
            assert reservation_watermark > 0
            assert not launch.done()

            runtime._owner.release_decode_delivery_dispatch_for_testing()
            dispatch_captured = False
            expires_at = time.monotonic() + _WAIT_SECONDS
            while (
                runtime.coordinator_actions.snapshot().queued_count != 1
                and time.monotonic() < expires_at
            ):
                pass
            assert runtime.coordinator_actions.snapshot().queued_count == 1
            transferred = runtime.snapshot().owner
            assert transferred.decode_delivery_reservation_count == 0
            assert transferred.decode_reservation_backed_handoff_action_count == 1
            with runtime._condition:
                local_ready_actions = tuple(
                    action
                    for action in runtime._consumer_pending.values()
                    if action.kind is NativeTerminalOwnerActionKind.LOCAL_DECODE_READY
                )
            assert len(local_ready_actions) == 1
            assert local_ready_actions[0].action_id > action_watermark
            assert not launch.done()

            assert serving.drain_coordinator_actions() == 1
            token = launch.result(timeout=_WAIT_SECONDS)

        runtime.end_scheduler_launch_handoff(token)
        local_ready_delivered = runtime.snapshot().owner
        assert local_ready_delivered.decode_delivery_reservation_count == 0
        assert local_ready_delivered.decode_reservation_backed_handoff_action_count == 0
        assert local_ready_delivered.transferred_decode_delivery_reservation_count == 2
    finally:
        release_settlement.set()
        if dispatch_captured:
            runtime._owner.release_decode_delivery_dispatch_for_testing()
        serving.abort_and_close()
        assert runtime.disposition.value == "stopped"


@pytest.mark.parametrize(
    "failure_after_retention",
    (False, True),
    ids=("caller-retains", "scheduler-retains"),
)
def test_scheduler_batch_failure_conserves_current_and_later_actions(
    monkeypatch: pytest.MonkeyPatch,
    failure_after_retention: bool,
) -> None:
    """A failed first publication cannot orphan its already-claimed sibling.

    :param monkeypatch: Scoped publication-failure injection.
    :param failure_after_retention: Whether scheduler ownership linearizes first.
    """

    (
        serving,
        runtime,
        _,
        _,
        registration,
        manifest,
        _,
        scheduler_events,
        _,
    ) = _serving(1)
    second_registration, second_manifest = _additional_decode_request(
        registration,
        scheduler_events,
    )
    serving.start()
    closed = False
    try:
        serving.register_request(registration, manifest)
        serving.register_request(second_registration, second_manifest)
        _stage_runtime_adoption(
            serving,
            runtime,
            registration,
            expected_scheduler_count=1,
        )
        _stage_runtime_adoption(
            serving,
            runtime,
            second_registration,
            expected_scheduler_count=2,
        )
        drained_actions: list[NativeTerminalOwnerAction] = []
        drain_scheduler_actions = runtime.scheduler_actions.drain

        def record_scheduler_actions(
            maximum_items: int | None = None,
        ) -> tuple[NativeTerminalOwnerAction, ...]:
            """Record the exact candidate batch crossing the runtime inbox.

            :param maximum_items: Optional FIFO-prefix bound.
            :returns: Exact actions claimed by the real runtime inbox.
            """

            actions = drain_scheduler_actions(maximum_items)
            drained_actions.extend(actions)
            return actions

        monkeypatch.setattr(
            runtime.scheduler_actions,
            "drain",
            record_scheduler_actions,
        )

        with monkeypatch.context() as scoped_patch:
            if failure_after_retention:
                scheduler_write_fd = serving._scheduler_serving._inbox._write_fd
                real_write = os.write

                def fail_scheduler_wake(
                    file_descriptor: int,
                    payload: bytes,
                ) -> int:
                    """Fail only the scheduler wake after action retention.

                    :param file_descriptor: Target wake descriptor.
                    :param payload: Exact wake payload.
                    :returns: Bytes written for unrelated descriptors.
                    """

                    if file_descriptor == scheduler_write_fd:
                        raise BrokenPipeError("synthetic scheduler wake failure")
                    return real_write(file_descriptor, payload)

                scoped_patch.setattr(os, "write", fail_scheduler_wake)
            else:

                def fail_before_retention(
                    receipt: object,
                    retain: Callable[[], None],
                    commit: Callable[[], None],
                ) -> None:
                    """Reject publication before scheduler authority transfers.

                    :param receipt: Candidate scheduler receipt.
                    :param retain: Uncalled action-retention boundary.
                    :param commit: Uncalled causal-authority transfer boundary.
                    """

                    del receipt, retain, commit
                    raise OSError("synthetic pre-retention publication failure")

                scoped_patch.setattr(
                    serving._scheduler_serving._inbox,
                    "publish_after_retention",
                    fail_before_retention,
                )

            with pytest.raises(TerminalSchedulerActionPublicationError) as raised:
                serving.drain_scheduler_actions()

        assert raised.value.scheduler_retains_action is failure_after_retention
        assert len(drained_actions) == 2
        candidate_action_ids = frozenset(action.action_id for action in drained_actions)
        expected_retained = (
            frozenset((drained_actions[0].action_id,))
            if failure_after_retention
            else frozenset()
        )
        with runtime._condition:
            pending_action_ids = frozenset(runtime._consumer_pending)
            claimed_action_ids = frozenset(runtime._inbox_claimed_action_ids)
        assert pending_action_ids & candidate_action_ids == expected_retained
        assert claimed_action_ids & candidate_action_ids == expected_retained
        runtime_snapshot = runtime.snapshot()
        assert runtime_snapshot.decode_scheduler_publication_handoff_count == 0
        assert runtime_snapshot.owner.active_handoff_action_count == 0
        scheduler_retained = serving._scheduler_serving.inventory().retained_action_ids
        assert frozenset(scheduler_retained) == expected_retained

        inventory = serving.abort_and_close()
        closed = True
        assert runtime.disposition is NativeTerminalRuntimeDisposition.STOPPED
        assert inventory.scheduler_serving.retained_action_ids == scheduler_retained
    finally:
        if not closed:
            serving.abort_and_close()


def test_precommit_publication_failure_closes_without_delayed_handoffs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-commit rejection settles every delayed handoff fail closed.

    :param monkeypatch: Scoped pre-commit publication failure.
    """

    serving, runtime, _, _, registration, manifest, _, _, _ = _serving(1)

    def reject_before_commit(
        receipt: object,
        retain: Callable[[], None],
        commit: Callable[[], None],
    ) -> None:
        """Reject before either scheduler retention callback can execute.

        :param receipt: Candidate scheduler receipt.
        :param retain: Uncalled action-retention boundary.
        :param commit: Uncalled handoff-settlement boundary.
        """

        del receipt, retain, commit
        raise OSError("synthetic pre-commit publication failure")

    serving.start()
    closed = False
    try:
        serving.register_request(registration, manifest)
        _stage_runtime_adoption(
            serving,
            runtime,
            registration,
            expected_scheduler_count=1,
        )
        monkeypatch.setattr(
            serving._scheduler_serving._inbox,
            "publish_after_retention",
            reject_before_commit,
        )
        with pytest.raises(TerminalSchedulerActionPublicationError):
            serving.drain_scheduler_actions()
        after_failure = runtime.snapshot()
        assert after_failure.decode_scheduler_publication_handoff_count == 0
        assert after_failure.owner.active_handoff_action_count == 0
        assert after_failure.owner.abort_settled_handoff_delivery_count == 0

        inventory = serving.abort_and_close()
        closed = True
        _assert_fail_closed_handoff_zero(runtime, inventory)
    finally:
        if not closed:
            serving.abort_and_close()


def test_postinsertion_commit_failure_closes_without_delayed_handoffs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed commit after inbox insertion is reconciled fail closed.

    :param monkeypatch: Scoped post-insertion commit failure.
    """

    serving, runtime, _, _, registration, manifest, _, _, _ = _serving(1)

    def reject_commit(action: NativeTerminalOwnerAction) -> None:
        """Reject the handoff commit after qualified receipt insertion.

        :param action: Exact delayed adoption authority.
        """

        assert action.kind is NativeTerminalOwnerActionKind.ADOPTION_READY
        raise RuntimeError("synthetic post-insertion commit failure")

    serving.start()
    closed = False
    try:
        serving.register_request(registration, manifest)
        _stage_runtime_adoption(
            serving,
            runtime,
            registration,
            expected_scheduler_count=1,
        )
        monkeypatch.setattr(
            runtime,
            "settle_decode_scheduler_publication_handoff",
            reject_commit,
        )
        with pytest.raises(TerminalSchedulerActionPublicationError):
            serving.drain_scheduler_actions()
        after_failure = runtime.snapshot()
        assert after_failure.decode_scheduler_publication_handoff_count == 1
        assert after_failure.owner.active_handoff_action_count == 0
        assert after_failure.owner.abort_settled_handoff_delivery_count == 1

        inventory = serving.abort_and_close()
        closed = True
        _assert_fail_closed_handoff_zero(runtime, inventory)
    finally:
        if not closed:
            serving.abort_and_close()


def test_abort_racing_publication_commit_closes_without_delayed_handoffs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Abort settlement and a racing publication commit reconcile once.

    :param monkeypatch: Scoped commit barrier exposing the abort race.
    """

    serving, runtime, _, _, registration, manifest, _, _, _ = _serving(1)
    commit_entered = threading.Event()
    release_commit = threading.Event()
    settle_handoff = runtime.settle_decode_scheduler_publication_handoff

    def hold_commit(action: NativeTerminalOwnerAction) -> None:
        """Hold after insertion until native abort settles active authority.

        :param action: Exact delayed adoption authority.
        """

        commit_entered.set()
        if not release_commit.wait(_WAIT_SECONDS):
            raise TimeoutError("publication commit was not released")
        settle_handoff(action)

    serving.start()
    closed = False
    try:
        serving.register_request(registration, manifest)
        _stage_runtime_adoption(
            serving,
            runtime,
            registration,
            expected_scheduler_count=1,
        )
        monkeypatch.setattr(
            runtime,
            "settle_decode_scheduler_publication_handoff",
            hold_commit,
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            publication = executor.submit(serving.drain_scheduler_actions)
            assert commit_entered.wait(_WAIT_SECONDS)
            before_abort = runtime.snapshot()
            assert before_abort.decode_scheduler_publication_handoff_count == 1
            assert before_abort.owner.active_handoff_action_count == 1

            runtime.begin_abort("synthetic abort racing publication commit")
            during_abort = runtime.snapshot()
            assert during_abort.decode_scheduler_publication_handoff_count == 1
            assert during_abort.owner.active_handoff_action_count == 0
            assert during_abort.owner.abort_settled_handoff_delivery_count == 1
            release_commit.set()
            assert publication.result(timeout=_WAIT_SECONDS) == 1
        after_commit = runtime.snapshot()
        assert after_commit.decode_scheduler_publication_handoff_count == 0
        assert after_commit.owner.active_handoff_action_count == 0
        assert after_commit.owner.abort_settled_handoff_delivery_count == 0

        inventory = serving.abort_and_close()
        closed = True
        _assert_fail_closed_handoff_zero(runtime, inventory)
    finally:
        release_commit.set()
        if not closed:
            serving.abort_and_close()


def test_scatter_submission_isolated_and_drains_before_cuda_retirement() -> None:
    """A blocked host submission cannot occupy the decode serving reactor."""

    (
        serving,
        runtime,
        _,
        actor_state,
        registration,
        manifest,
        _,
        _,
        _,
    ) = _serving(1)
    actor_state.block_scatter = True
    projection_states: list[PackedTerminalDecodeScatterWorkerInventory] = []
    original_projection_fence = runtime.wait_for_output_projection

    def record_projection_fence(timeout_seconds: float) -> bool:
        """Capture worker ownership at each native output fence.

        :param timeout_seconds: Production shutdown bound.
        :returns: Whether the original fence reached quiescence.
        """

        projection_states.append(serving.inventory().scatter_worker)
        return original_projection_fence(timeout_seconds)

    runtime.wait_for_output_projection = record_projection_fence
    serving.start()
    try:
        serving.register_request(registration, manifest)
        serving.allocation_published(
            registration.transaction,
            object.__new__(object),
            (),
        )
        writer_id = registration.source_plan.writers[0].writer_id
        serving.control_received(writer_id, object())
        serving.control_received(writer_id, object())
        assert actor_state.scatter_entered.wait(_WAIT_SECONDS)

        worker_inventory = serving.inventory().scatter_worker
        assert worker_inventory.active_binding_digest == registration.binding.digest
        assert worker_inventory.thread_alive
        assert actor_state.scatter_thread_ids == actor_state.cuda_binding_thread_ids
        assert runtime.decode_scatter_actions.fileno() not in serving.runtime_filenos
        assert serving.drain_decode_work_actions() == 0

        shutdown_finished = threading.Event()
        shutdown_error: list[BaseException] = []

        def stop_serving() -> None:
            """Drive the production shutdown boundary from another owner."""

            try:
                serving.stop_admission_and_retire_producers()
            except BaseException as error:  # noqa: BLE001
                shutdown_error.append(error)
            finally:
                shutdown_finished.set()

        shutdown_thread = threading.Thread(target=stop_serving, daemon=False)
        shutdown_thread.start()
        assert not shutdown_finished.wait(0.05)
        assert actor_state.retirement_worker_inventories == []

        actor_state.scatter_release.set()
        shutdown_thread.join(timeout=_WAIT_SECONDS)
        assert not shutdown_thread.is_alive()
        assert shutdown_error == []
        assert len(actor_state.retirement_worker_inventories) == 1
        retired_worker = actor_state.retirement_worker_inventories[0]
        assert retired_worker.retained_action_count == 0
        assert not retired_worker.thread_alive
        assert retired_worker.completed_action_count == 1
        assert retired_worker.last_queue_residence_ns is not None
        assert retired_worker.last_submission_duration_ns is not None
        assert len(projection_states) == 2
        assert projection_states[0].active_binding_digest == (
            registration.binding.digest
        )
        assert projection_states[0].thread_alive
        assert projection_states[1].retained_action_count == 0
        assert not projection_states[1].thread_alive
    finally:
        actor_state.scatter_release.set()
        if runtime.disposition.value != "stopped":
            serving.abort_and_close()


def test_scatter_worker_accounts_and_logs_queue_and_submission_time(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One direct action exposes exact inventory and parser-stable timing."""

    runtime, binding = _standalone_scatter_runtime()
    action = _scatter_action(binding, action_id=701, commit_timestamp_ns=100)
    clock_values = iter((125, 425))
    consumed_actions: list[NativeTerminalOwnerAction] = []

    def consume_action(value: NativeTerminalOwnerAction) -> None:
        """Complete the runtime authority after deterministic submission.

        :param value: Exact worker-owned action.
        """

        consumed_actions.append(value)
        runtime.acknowledge_consumed_action(value)

    worker = PackedTerminalDecodeScatterWorker(
        runtime=runtime,
        consume_action=consume_action,
        bind_cuda_device=lambda: None,
        fatal_listener=lambda reason, formatted_traceback: None,
        clock_ns=lambda: next(clock_values),
    )
    worker.start()
    try:
        with caplog.at_level(logging.INFO):
            _enqueue_scatter_action(runtime, action)
            inventory = worker.stop_and_join(_WAIT_SECONDS, abort=False)

        assert consumed_actions == [action]
        assert inventory.disposition is (
            PackedTerminalDecodeScatterWorkerDisposition.STOPPED
        )
        assert inventory.completed_action_count == 1
        assert inventory.aborted_action_count == 0
        assert inventory.last_queue_residence_ns == 25
        assert inventory.maximum_queue_residence_ns == 25
        assert inventory.last_submission_duration_ns == 300
        assert inventory.maximum_submission_duration_ns == 300
        records = tuple(
            record.getMessage()
            for record in caplog.records
            if TERMINAL_DECODE_SCATTER_TIMING_LOG_PREFIX in record.getMessage()
        )
        assert len(records) == 1
        payload = json.loads(
            records[0].split(TERMINAL_DECODE_SCATTER_TIMING_LOG_PREFIX, 1)[1]
        )
        assert payload == {
            "binding_digest": binding.digest.hex(),
            "host_submission_ms": 0.0003,
            "queue_residence_ms": 0.000025,
        }
        _close_standalone_runtime(runtime, abort=False)
    finally:
        if runtime.disposition is not NativeTerminalRuntimeDisposition.STOPPED:
            worker.stop_and_join(_WAIT_SECONDS, abort=True)
            _close_standalone_runtime(runtime, abort=True)


def test_scatter_worker_action_failure_is_process_fatal_and_clears_active() -> None:
    """A failed scatter preserves its first cause and loses no authority."""

    runtime, binding = _standalone_scatter_runtime()
    action = _scatter_action(binding, action_id=702, commit_timestamp_ns=100)
    failure_published = threading.Event()
    failures: list[tuple[str, str | None]] = []

    def fail_action(value: NativeTerminalOwnerAction) -> None:
        """Raise from the exact worker execution boundary.

        :param value: Expected scatter action.
        """

        assert value == action
        raise RuntimeError("injected direct scatter failure")

    def record_failure(reason: str, formatted_traceback: str | None) -> None:
        """Capture the process-fatal worker notification.

        :param reason: Stable first-cause text.
        :param formatted_traceback: Complete originating traceback.
        """

        failures.append((reason, formatted_traceback))
        failure_published.set()

    worker = PackedTerminalDecodeScatterWorker(
        runtime=runtime,
        consume_action=fail_action,
        bind_cuda_device=lambda: None,
        fatal_listener=record_failure,
        clock_ns=lambda: 125,
    )
    worker.start()
    try:
        _enqueue_scatter_action(runtime, action)
        assert failure_published.wait(_WAIT_SECONDS)
        inventory = worker.stop_and_join(_WAIT_SECONDS, abort=True)

        assert len(failures) == 1
        assert failures[0][0] == "decode scatter action failed"
        assert failures[0][1] is not None
        assert "injected direct scatter failure" in failures[0][1]
        assert inventory.disposition is (
            PackedTerminalDecodeScatterWorkerDisposition.PROCESS_FATAL
        )
        assert inventory.active_action_id is None
        assert inventory.retained_action_count == 0
        assert inventory.completed_action_count == 0
        assert inventory.aborted_action_count == 1
        assert runtime.snapshot().consumer_pending_count == 0
    finally:
        if worker.inventory().thread_alive:
            worker.stop_and_join(_WAIT_SECONDS, abort=True)
        _close_standalone_runtime(runtime, abort=True)


def test_scatter_worker_clock_and_abort_ack_failures_preserve_first_cause(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A secondary abort failure cannot hide timing failure or active state."""

    runtime, binding = _standalone_scatter_runtime()
    action = _scatter_action(binding, action_id=703, commit_timestamp_ns=100)
    failure_published = threading.Event()
    failures: list[str] = []
    original_acknowledge = runtime.acknowledge_aborted_action_if_pending

    def fail_clock() -> int:
        """Fail at the first worker timing boundary.

        :returns: This function does not return.
        :raises RuntimeError: Always.
        """

        raise RuntimeError("injected worker clock failure")

    def fail_acknowledgement(value: NativeTerminalOwnerAction) -> bool:
        """Fail reconciliation after the timing failure is already sticky.

        :param value: Exact action awaiting fail-closed reconciliation.
        :returns: This function does not return.
        :raises RuntimeError: Always.
        """

        assert value == action
        raise RuntimeError("injected abort acknowledgement failure")

    def record_failure(reason: str, formatted_traceback: str | None) -> None:
        """Retain the first worker failure notification.

        :param reason: Stable first-cause text.
        :param formatted_traceback: Complete first-cause traceback.
        """

        del formatted_traceback
        failures.append(reason)
        failure_published.set()

    runtime.acknowledge_aborted_action_if_pending = fail_acknowledgement
    worker = PackedTerminalDecodeScatterWorker(
        runtime=runtime,
        consume_action=lambda value: None,
        bind_cuda_device=lambda: None,
        fatal_listener=record_failure,
        clock_ns=fail_clock,
    )
    worker.start()
    try:
        with caplog.at_level(logging.CRITICAL):
            _enqueue_scatter_action(runtime, action)
            assert failure_published.wait(_WAIT_SECONDS)
            inventory = worker.stop_and_join(_WAIT_SECONDS, abort=True)

        assert failures == ["decode scatter worker timing or ownership failed"]
        assert inventory.fatal_reason == failures[0]
        assert inventory.active_action_id is None
        assert inventory.retained_action_count == 0
        assert runtime.snapshot().consumer_pending_count == 1
        messages = tuple(record.getMessage() for record in caplog.records)
        assert any("injected worker clock failure" in message for message in messages)
        assert any(
            "injected abort acknowledgement failure" in message for message in messages
        )

        runtime.acknowledge_aborted_action_if_pending = original_acknowledge
        assert runtime.acknowledge_aborted_action_if_pending(action)
        assert runtime.snapshot().consumer_pending_count == 0
    finally:
        runtime.acknowledge_aborted_action_if_pending = original_acknowledge
        if worker.inventory().thread_alive:
            worker.stop_and_join(_WAIT_SECONDS, abort=True)
        if runtime.snapshot().consumer_pending_count != 0:
            runtime.acknowledge_aborted_action_if_pending(action)
        _close_standalone_runtime(runtime, abort=True)


def test_scatter_worker_unexpected_exit_is_process_fatal() -> None:
    """A live worker cannot disappear outside explicit drain ownership."""

    runtime, _ = _standalone_scatter_runtime()
    failure_published = threading.Event()
    failures: list[str] = []
    worker = PackedTerminalDecodeScatterWorker(
        runtime=runtime,
        consume_action=lambda action: None,
        bind_cuda_device=lambda: None,
        fatal_listener=lambda reason, formatted_traceback: (
            failures.append(reason),
            failure_published.set(),
        ),
        clock_ns=lambda: 0,
    )

    def exit_without_drain(
        self: PackedTerminalDecodeScatterWorker,
        selector: object,
    ) -> None:
        """Return from the owner loop without a lifecycle request.

        :param self: Exact worker fixture.
        :param selector: Worker-owned selector.
        """

        del self, selector

    worker._run_loop = types.MethodType(exit_without_drain, worker)
    try:
        with contextlib.suppress(PackedTerminalDecodeScatterWorkerError):
            worker.start()
        assert failure_published.wait(_WAIT_SECONDS)
        inventory = worker.stop_and_join(_WAIT_SECONDS, abort=True)

        assert failures == ["decode scatter worker exited without a shutdown request"]
        assert inventory.fatal_reason == failures[0]
        assert not inventory.thread_alive
    finally:
        if worker.inventory().thread_alive:
            worker.stop_and_join(_WAIT_SECONDS, abort=True)
        _close_standalone_runtime(runtime, abort=True)


def test_scatter_worker_bounded_inbox_abort_drains_active_and_queued() -> None:
    """The physical bound rejects overflow and abort drains exact ownership."""

    runtime, binding = _standalone_scatter_runtime()
    release_active = threading.Event()
    active_entered = threading.Event()
    consumed_count = 0

    def consume_action(action: NativeTerminalOwnerAction) -> None:
        """Block the first action while the direct inbox reaches capacity.

        :param action: Exact action retained by the worker.
        """

        nonlocal consumed_count
        consumed_count += 1
        if consumed_count == 1:
            active_entered.set()
            if not release_active.wait(_WAIT_SECONDS):
                raise TimeoutError("active scatter was not released")
        runtime.acknowledge_consumed_action(action)

    worker = PackedTerminalDecodeScatterWorker(
        runtime=runtime,
        consume_action=consume_action,
        bind_cuda_device=lambda: None,
        fatal_listener=lambda reason, formatted_traceback: None,
        clock_ns=SystemTerminalOwnerClock().now_ns,
    )
    worker.start()
    shutdown_finished = threading.Event()
    shutdown_results: list[PackedTerminalDecodeScatterWorkerInventory] = []
    shutdown_errors: list[BaseException] = []
    try:
        first = _scatter_action(
            binding,
            action_id=800,
            commit_timestamp_ns=0,
        )
        _enqueue_scatter_action(runtime, first)
        assert active_entered.wait(_WAIT_SECONDS)
        for offset in range(8):
            _enqueue_scatter_action(
                runtime,
                _scatter_action(
                    binding,
                    action_id=801 + offset,
                    commit_timestamp_ns=0,
                ),
            )
        assert runtime.decode_scatter_actions.snapshot().queued_count == 8
        with pytest.raises(
            NativeTerminalRuntimeOverflowError,
            match="exceeded capacity 8",
        ):
            _enqueue_scatter_action(
                runtime,
                _scatter_action(
                    binding,
                    action_id=809,
                    commit_timestamp_ns=0,
                ),
            )

        runtime.begin_abort("decode scatter direct inbox overflow")

        def stop_worker() -> None:
            """Join the abort drain while the first action remains blocked."""

            try:
                shutdown_results.append(worker.stop_and_join(_WAIT_SECONDS, abort=True))
            except BaseException as error:  # noqa: BLE001
                shutdown_errors.append(error)
            finally:
                shutdown_finished.set()

        shutdown_thread = threading.Thread(target=stop_worker, daemon=False)
        shutdown_thread.start()
        with worker._condition:
            assert worker._condition.wait_for(
                lambda: worker._admission_closed,
                timeout=_WAIT_SECONDS,
            )
        assert not shutdown_finished.is_set()

        release_active.set()
        shutdown_thread.join(timeout=_WAIT_SECONDS)
        assert not shutdown_thread.is_alive()
        assert shutdown_errors == []
        assert len(shutdown_results) == 1
        inventory = shutdown_results[0]
        assert inventory.retained_action_count == 0
        assert inventory.completed_action_count == 1
        assert inventory.aborted_action_count == 8
        assert runtime.snapshot().consumer_pending_count == 0
    finally:
        release_active.set()
        if worker.inventory().thread_alive:
            worker.stop_and_join(_WAIT_SECONDS, abort=True)
        _close_standalone_runtime(runtime, abort=True)


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
    _drive_to_adoption(serving, registration, actor_state)

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
        actor_state,
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
    _drive_to_adoption(serving, registration, actor_state)

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
