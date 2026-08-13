import dataclasses
import logging
import traceback
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from sglang.srt.disaggregation.common.packed_staging_wire import PackedWireMessage
from sglang.srt.disaggregation.common.staging_layout import StagingWriterId
from sglang.srt.disaggregation.nixl.packed_runtime import (
    PackedControlSender,
    PackedDecodeOwnerSignal,
    PackedDecodeRuntime,
    PackedDecodeScatterBatch,
    PackedDecodeScatterCompletionProducer,
)
from sglang.srt.disaggregation.nixl.packed_staging_request import (
    PackedDecodeRequestTransaction,
    PackedRequestPublication,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.native_state import (
    NativeTerminalLifecycleRegistration,
    NativeTerminalOwnerAction,
    NativeTerminalOwnerActionKind,
    NativeTerminalOwnerEventKind,
    NativeTerminalProcessIdentity,
    NativeTerminalProducerClass,
    NativeTerminalReceipt,
    NativeTerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.receipts import (
    TerminalReceiptKind,
    TerminalReceiptOutcome,
)
from sglang.srt.disaggregation.terminal_progress.source_plan import (
    PackedTerminalSourcePlan,
)
from sglang.srt.disaggregation.terminal_progress.wire import TerminalWireReceipt

logger = logging.getLogger(__name__)


@runtime_checkable
class NativeTerminalDecodeRuntime(Protocol):
    """Process-lifetime native runtime surface consumed by decode wiring."""

    def python_producer_id(
        self,
        producer_class: NativeTerminalProducerClass,
        authenticated_issuer: NativeTerminalProcessIdentity | None = None,
    ) -> int:
        """Resolve one pre-registered Python producer authority.

        :param producer_class: Required local, control, or receipt authority.
        :param authenticated_issuer: Route-authenticated issuer when required.
        :returns: Exact registered producer identity.
        """

    def register_lifecycle(
        self,
        registration: NativeTerminalLifecycleRegistration,
    ) -> None:
        """Register one decode lifecycle before its first event.

        :param registration: Complete native decode registration.
        """

    def submit(
        self,
        producer_id: int,
        binding_digest: bytes,
        kind: NativeTerminalOwnerEventKind,
        *,
        receipt: NativeTerminalReceipt | None = None,
        reason: str | None = None,
        enqueued_ns: int | None = None,
    ) -> None:
        """Submit one producer-ordered event to the native owner.

        :param producer_id: Exact registered producer identity.
        :param binding_digest: Exact local lifecycle identity.
        :param kind: Closed native event kind.
        :param receipt: Imported or owner-minted authority when required.
        :param reason: Stable failure evidence when required.
        :param enqueued_ns: Optional exact native-clock timestamp.
        """

    def complete_work_action(
        self,
        producer_id: int,
        action: NativeTerminalOwnerAction,
        followup_kind: NativeTerminalOwnerEventKind,
        *,
        receipt: NativeTerminalReceipt | None = None,
        reason: str | None = None,
        enqueued_ns: int | None = None,
    ) -> None:
        """Complete one exact decode-worker action.

        :param producer_id: Exact local producer identity.
        :param action: Native action drained by the decode worker.
        :param followup_kind: Successful or failed transition earned by work.
        :param receipt: Optional receipt required by the followup.
        :param reason: Stable failure evidence when required.
        :param enqueued_ns: Optional exact native-clock timestamp.
        """

    def complete_scheduler_action(
        self,
        producer_id: int,
        action: NativeTerminalOwnerAction,
        followup_kind: NativeTerminalOwnerEventKind,
        *,
        completion_receipt: NativeTerminalReceipt | None = None,
        enqueued_ns: int | None = None,
    ) -> None:
        """Consume one exact scheduler-affine action.

        :param producer_id: Exact local receipt producer identity.
        :param action: Adoption authority drained by the scheduler.
        :param followup_kind: Exact adoption-consumed transition.
        :param completion_receipt: Optional source-only completion receipt.
        :param enqueued_ns: Optional exact native-clock timestamp.
        """

    def submit_imported_receipt(
        self,
        producer_id: int,
        receipt: NativeTerminalReceipt,
        kind: NativeTerminalOwnerEventKind,
        *,
        reason: str | None = None,
        enqueued_ns: int | None = None,
    ) -> None:
        """Submit one route-authenticated imported receipt.

        :param producer_id: Producer bound to the receipt issuer.
        :param receipt: Validated native receipt.
        :param kind: Receipt-consuming lifecycle event.
        :param reason: Stable failure evidence when required.
        :param enqueued_ns: Optional exact native-clock timestamp.
        """

    def acknowledge_consumed_action(
        self,
        action: NativeTerminalOwnerAction,
    ) -> None:
        """Retire one non-scheduler action after exact consumer acceptance.

        :param action: Exact native action whose side effect completed.
        """

    def fail_scheduler_action(
        self,
        action: NativeTerminalOwnerAction,
        reason: str,
    ) -> None:
        """Fail one scheduler action whose ownership became ambiguous.

        :param action: Exact adoption action which could not complete.
        :param reason: Stable process-fatal failure evidence.
        """


@dataclasses.dataclass(frozen=True, slots=True)
class PackedDecodeScatterHandoff:
    """Scatter work whose direct callback was attached successfully.

    :ivar action: Exact one-shot native scatter-work authority.
    :ivar batch: Actor-owned scatter batch covered by the callback.
    """

    action: NativeTerminalOwnerAction
    batch: PackedDecodeScatterBatch

    def __post_init__(self) -> None:
        """Validate one exact scatter handoff."""

        if type(self.action) is not NativeTerminalOwnerAction:
            raise TypeError("action must be NativeTerminalOwnerAction")
        if self.action.kind is not NativeTerminalOwnerActionKind.DECODE_SCATTER_READY:
            raise ValueError("scatter handoff requires DECODE_SCATTER_READY")
        if type(self.batch) is not PackedDecodeScatterBatch:
            raise TypeError("batch must be PackedDecodeScatterBatch")
        if self.action.binding.digest != self.batch.binding_digest:
            raise ValueError("scatter action and batch bindings differ")


class PackedTerminalDecodeWiring:
    """Bind packed decode side effects to authoritative native transitions.

    This component owns no lifecycle reducer and never observes CUDA progress.
    Native owner actions authorize every continuation. Destination scatter
    terminality travels directly from the CUDA callback producer into the
    native v1 owner ABI.
    """

    _actor: PackedDecodeRuntime
    _runtime: NativeTerminalDecodeRuntime
    _cuda_completion: PackedDecodeScatterCompletionProducer
    _local_producer_id: int
    _local_receipt_producer_id: int

    def __init__(
        self,
        *,
        actor: PackedDecodeRuntime,
        runtime: NativeTerminalDecodeRuntime,
        cuda_completion: PackedDecodeScatterCompletionProducer,
        local_identity: TerminalProcessIdentity,
    ) -> None:
        """Construct decode orchestration around process-lifetime owners.

        :param actor: Existing packed decode transaction actor.
        :param runtime: Sole authoritative native lifecycle runtime.
        :param cuda_completion: Direct CUDA callback-to-owner producer.
        :param local_identity: Exact decode process owned by this wiring.
        """

        if type(actor) is not PackedDecodeRuntime:
            raise TypeError("actor must be PackedDecodeRuntime")
        if not isinstance(runtime, NativeTerminalDecodeRuntime):
            raise TypeError("runtime must satisfy NativeTerminalDecodeRuntime")
        if not isinstance(
            cuda_completion,
            PackedDecodeScatterCompletionProducer,
        ):
            raise TypeError(
                "cuda_completion must satisfy PackedDecodeScatterCompletionProducer"
            )
        if type(local_identity) is not TerminalProcessIdentity:
            raise TypeError("local_identity must be TerminalProcessIdentity")
        if local_identity.role is not TerminalOwnerRole.DECODE:
            raise ValueError("local_identity must belong to decode")
        self._actor = actor
        self._runtime = runtime
        self._cuda_completion = cuda_completion
        self._local_producer_id = runtime.python_producer_id(
            NativeTerminalProducerClass.LOCAL
        )
        self._local_receipt_producer_id = runtime.python_producer_id(
            NativeTerminalProducerClass.RECEIPT,
            NativeTerminalProcessIdentity.from_identity(local_identity),
        )

    def bind_transaction(
        self,
        transaction: PackedDecodeRequestTransaction,
        binding: TerminalRequestBinding,
        source_plan: PackedTerminalSourcePlan,
    ) -> None:
        """Bind actor and native lifecycle identity before publication.

        :param transaction: Exact prepared packed request transaction.
        :param binding: Exact local decode lifecycle identity.
        :param source_plan: Decoder-authored cross-rank source plan.
        """

        registration = self._actor.bind_terminal_owner(
            transaction,
            binding,
            source_plan,
        )
        self._runtime.register_lifecycle(
            NativeTerminalLifecycleRegistration(
                binding=NativeTerminalRequestBinding.from_binding(registration.binding),
                publication_identity=None,
                trusted_issuers=tuple(
                    NativeTerminalProcessIdentity.from_identity(issuer)
                    for issuer in registration.trusted_issuers
                ),
            )
        )

    def allocation_published(
        self,
        transaction: PackedDecodeRequestTransaction,
        publication: PackedRequestPublication,
        routes: tuple[PackedControlSender, ...],
    ) -> None:
        """Bind transport routes and publish the local allocation transition.

        :param transaction: Exact owner-bound transaction.
        :param publication: Matching irreversible packed publication.
        :param routes: Complete authenticated writer routes.
        """

        kind = self._actor.bind_publication(transaction, publication, routes)
        if kind is None:
            raise RuntimeError("owner-bound publication emitted no native transition")
        binding_digest = self._binding_digest(transaction)
        actor_transaction = self._actor.terminal_owner_transaction(binding_digest)
        if actor_transaction is not transaction:
            raise RuntimeError("decode transaction registry changed during publication")
        self._runtime.submit(
            self._local_producer_id,
            binding_digest,
            kind,
        )

    def control_received(
        self,
        authenticated_writer_id: StagingWriterId,
        message: PackedWireMessage,
    ) -> tuple[PackedDecodeOwnerSignal, ...]:
        """Apply authenticated control and submit every earned transition.

        :param authenticated_writer_id: Writer proved by the control route.
        :param message: Validated packed control payload.
        :returns: Exact native transitions submitted for this message.
        """

        signals = self._actor.handle_control(authenticated_writer_id, message)
        for signal in signals:
            producer_id = self._runtime.python_producer_id(
                NativeTerminalProducerClass.CONTROL,
                NativeTerminalProcessIdentity.from_identity(signal.issuer),
            )
            self._runtime.submit(
                producer_id,
                signal.binding_digest,
                signal.kind,
            )
        return signals

    def consume_scatter_action(
        self,
        action: NativeTerminalOwnerAction,
    ) -> PackedDecodeScatterHandoff:
        """Launch scatter and attach direct native terminal delivery.

        ``DECODE_SCATTER_STARTED`` enters the ordered native queue before the
        CUDA callback is attached. The callback therefore cannot overtake the
        start transition even when the scatter has already completed.

        :param action: Exact native scatter-work authority.
        :returns: Actor batch covered by the direct callback.
        """

        self._require_action(action, NativeTerminalOwnerActionKind.DECODE_SCATTER_READY)
        transaction = self._actor.terminal_owner_transaction(action.binding.digest)
        try:
            batch = self._actor.begin_terminal_owner_scatter(transaction)
        except (OSError, RuntimeError, TypeError, ValueError):
            self._complete_failed_work(action, "decode scatter submission failed")
            raise
        try:
            self._runtime.complete_work_action(
                self._local_producer_id,
                action,
                NativeTerminalOwnerEventKind.DECODE_SCATTER_STARTED,
            )
            self._cuda_completion.arm(batch.binding_digest)
            self._cuda_completion.submit(batch.stream_handle, batch.binding_digest)
            self._actor.confirm_terminal_owner_scatter_callback(transaction, batch)
        except (OSError, RuntimeError, TypeError, ValueError):
            self._actor.quarantine(
                transaction,
                "decode scatter callback attachment failed",
            )
            self._submit_local_failure(
                batch.binding_digest,
                "decode scatter callback attachment failed",
            )
            raise
        return PackedDecodeScatterHandoff(action=action, batch=batch)

    def consume_teardown_action(
        self,
        action: NativeTerminalOwnerAction,
    ) -> None:
        """Consume native scatter terminality and send exact teardown.

        :param action: Exact native teardown-work authority.
        """

        self._require_action(
            action,
            NativeTerminalOwnerActionKind.DECODE_TEARDOWN_READY,
        )
        transaction = self._actor.terminal_owner_transaction(action.binding.digest)
        try:
            self._actor.complete_terminal_owner_scatter(transaction)
            self._actor.begin_terminal_owner_teardown(transaction)
            self._runtime.complete_work_action(
                self._local_producer_id,
                action,
                NativeTerminalOwnerEventKind.DECODE_TEARDOWN_SENT,
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            self._complete_failed_work(action, "decode teardown dispatch failed")
            raise

    def consume_adoption_action(
        self,
        action: NativeTerminalOwnerAction,
        adopt_request: Callable[[object], None],
        finalize_request: Callable[[object], None],
    ) -> object:
        """Adopt pages and publish readiness under scheduler authority.

        The first callback copies request metadata and installs the exact
        retained request into scheduler-owned structures while the auxiliary
        row remains pinned. The actor then releases that row and publishes its
        metadata-consumed transition. The second callback clears the
        scheduler's transaction-local fields before local-ready authority can
        enter the native owner. This ordering prevents a runnable request from
        being announced before its mutable scheduler state is complete.

        :param action: Exact owner-minted adoption authority.
        :param adopt_request: Scheduler-affine metadata-copy and request-adoption
            operation.
        :param finalize_request: Scheduler-affine transaction finalization
            operation.
        :returns: Retained request owner adopted by the scheduler.
        """

        self._require_action(action, NativeTerminalOwnerActionKind.ADOPTION_READY)
        if not callable(adopt_request):
            raise TypeError("adopt_request must be callable")
        if not callable(finalize_request):
            raise TypeError("finalize_request must be callable")
        transaction = self._actor.terminal_owner_transaction(action.binding.digest)
        scheduler_action_completed = False
        try:
            owner = self._actor.consume_terminal_owner_adoption(transaction)
            self._runtime.complete_scheduler_action(
                self._local_receipt_producer_id,
                action,
                NativeTerminalOwnerEventKind.DECODE_ADOPTION_CONSUMED,
            )
            scheduler_action_completed = True
            adopt_request(owner)
            self._actor.complete_terminal_owner_metadata_consumption(transaction)
            self._runtime.submit(
                self._local_producer_id,
                action.binding.digest,
                NativeTerminalOwnerEventKind.DECODE_METADATA_CONSUMED,
            )
            finalize_request(owner)
            self._runtime.submit(
                self._local_producer_id,
                action.binding.digest,
                NativeTerminalOwnerEventKind.DECODE_LOCAL_READY_ISSUED,
            )
            return owner
        except Exception:
            reason = "decode scheduler adoption failed"
            formatted_traceback = traceback.format_exc()
            try:
                self._actor.quarantine(transaction, reason)
            except Exception:  # noqa: BLE001
                logger.error(
                    "Decode transaction quarantine failed after adoption error:\n%s",
                    traceback.format_exc(),
                )
            try:
                if scheduler_action_completed:
                    self._submit_local_failure(action.binding.digest, reason)
                else:
                    self._runtime.fail_scheduler_action(action, reason)
            except Exception:  # noqa: BLE001
                logger.error(
                    "Decode native failure publication failed after adoption "
                    "error:\n%s",
                    traceback.format_exc(),
                )
            logger.error(
                "Decode scheduler adoption failed closed:\n%s",
                formatted_traceback,
            )
            raise

    def quarantine_transaction(
        self,
        transaction: PackedDecodeRequestTransaction,
        reason: str,
    ) -> None:
        """Retain one ambiguous decode transaction against resource reuse.

        This boundary is used when scheduler-inbox or composition failure makes
        adoption authority ambiguous before a concrete action reaches the
        scheduler. It mutates no scheduler queue and publishes no replacement
        lifecycle event.

        :param transaction: Exact request-scoped packed actor transaction.
        :param reason: Stable fail-closed evidence.
        """

        if type(transaction) is not PackedDecodeRequestTransaction:
            raise TypeError("transaction must be PackedDecodeRequestTransaction")
        if type(reason) is not str or len(reason) == 0:
            raise ValueError("reason must be a non-empty string")
        self._actor.quarantine(transaction, reason)

    def request_ready(
        self,
        *,
        binding_digest: bytes,
        wire_receipt: TerminalWireReceipt,
        authenticated_issuer: TerminalProcessIdentity,
    ) -> None:
        """Submit request-global readiness from the authenticated coordinator.

        :param binding_digest: Exact local decode lifecycle identity.
        :param wire_receipt: Imported one-shot request-ready authority.
        :param authenticated_issuer: Coordinator proved by the control route.
        """

        if type(wire_receipt) is not TerminalWireReceipt:
            raise TypeError("wire_receipt must be TerminalWireReceipt")
        if type(authenticated_issuer) is not TerminalProcessIdentity:
            raise TypeError("authenticated_issuer must be TerminalProcessIdentity")
        if (
            wire_receipt.kind is not TerminalReceiptKind.REQUEST_READY
            or wire_receipt.outcome is not TerminalReceiptOutcome.SUCCESS
        ):
            raise RuntimeError("request-ready ingress requires successful readiness")
        self._submit_request_terminal_receipt(
            binding_digest=binding_digest,
            wire_receipt=wire_receipt,
            authenticated_issuer=authenticated_issuer,
            event_kind=NativeTerminalOwnerEventKind.DECODE_REQUEST_READY,
            reason=None,
        )

    def request_failed(
        self,
        *,
        binding_digest: bytes,
        wire_receipt: TerminalWireReceipt,
        authenticated_issuer: TerminalProcessIdentity,
        reason: str,
    ) -> None:
        """Submit request-global failure from the authenticated coordinator.

        :param binding_digest: Exact local decode lifecycle identity.
        :param wire_receipt: Imported one-shot request-failure authority.
        :param authenticated_issuer: Coordinator proved by the control route.
        :param reason: Stable request-global failure evidence.
        """

        if type(wire_receipt) is not TerminalWireReceipt:
            raise TypeError("wire_receipt must be TerminalWireReceipt")
        if type(authenticated_issuer) is not TerminalProcessIdentity:
            raise TypeError("authenticated_issuer must be TerminalProcessIdentity")
        if type(reason) is not str or len(reason) == 0:
            raise ValueError("reason must be a non-empty string")
        if (
            wire_receipt.kind is not TerminalReceiptKind.FAILURE
            or wire_receipt.outcome is not TerminalReceiptOutcome.FAILURE
        ):
            raise RuntimeError("request-failure ingress requires failure authority")
        self._submit_request_terminal_receipt(
            binding_digest=binding_digest,
            wire_receipt=wire_receipt,
            authenticated_issuer=authenticated_issuer,
            event_kind=NativeTerminalOwnerEventKind.DECODE_REQUEST_FAILED,
            reason=reason,
        )

    def _submit_request_terminal_receipt(
        self,
        *,
        binding_digest: bytes,
        wire_receipt: TerminalWireReceipt,
        authenticated_issuer: TerminalProcessIdentity,
        event_kind: NativeTerminalOwnerEventKind,
        reason: str | None,
    ) -> None:
        """Join route authentication with one request-global wire authority.

        :param binding_digest: Exact local decode lifecycle identity.
        :param wire_receipt: Validated request-global receipt.
        :param authenticated_issuer: Identity proved by the receive route.
        :param event_kind: Native ready or failed transition.
        :param reason: Failure evidence, otherwise ``None``.
        """

        expected = self._actor.terminal_owner_request_ready_issuer(binding_digest)
        if authenticated_issuer != expected:
            raise RuntimeError("request-ready route authenticated another coordinator")
        if wire_receipt.issuer != authenticated_issuer:
            raise RuntimeError("request-ready receipt asserts another coordinator")
        if wire_receipt.binding.digest != binding_digest:
            raise RuntimeError("request-ready receipt targets another binding")
        native_issuer = NativeTerminalProcessIdentity.from_identity(
            authenticated_issuer
        )
        producer_id = self._runtime.python_producer_id(
            NativeTerminalProducerClass.RECEIPT,
            native_issuer,
        )
        self._runtime.submit_imported_receipt(
            producer_id,
            NativeTerminalReceipt.from_wire_receipt(wire_receipt),
            event_kind,
            reason=reason,
        )

    def retire(self, action: NativeTerminalOwnerAction) -> None:
        """Retire actor identity after native request-global completion.

        :param action: Exact native request-retirement action.
        """

        self._require_action(action, NativeTerminalOwnerActionKind.REQUEST_RETIRED)
        transaction = self._actor.terminal_owner_transaction(action.binding.digest)
        self._actor.retire_terminal_owner_request(transaction)
        self._runtime.acknowledge_consumed_action(action)

    def cancel_unpublished(
        self,
        transaction: PackedDecodeRequestTransaction,
        reason: str,
    ) -> object:
        """Cancel an owner-bound request before external publication.

        :param transaction: Exact prepared transaction.
        :param reason: Stable cancellation evidence.
        :returns: Retained request owner released by cancellation.
        """

        if type(reason) is not str or len(reason) == 0:
            raise ValueError("reason must be a non-empty string")
        binding_digest = self._binding_digest(transaction)
        owner = self._actor.cancel_terminal_owner_unpublished(transaction)
        self._runtime.submit(
            self._local_producer_id,
            binding_digest,
            NativeTerminalOwnerEventKind.DECODE_CANCEL_UNPUBLISHED,
            reason=reason,
        )
        return owner

    def _complete_failed_work(
        self,
        action: NativeTerminalOwnerAction,
        reason: str,
    ) -> None:
        """Consume one work action through its native failure transition.

        :param action: Exact unconsumed work action.
        :param reason: Stable request-local failure evidence.
        """

        self._runtime.complete_work_action(
            self._local_producer_id,
            action,
            NativeTerminalOwnerEventKind.DECODE_REQUEST_FAILED,
            reason=reason,
        )

    def _submit_local_failure(self, binding_digest: bytes, reason: str) -> None:
        """Submit request-local failure after a work action was consumed.

        :param binding_digest: Exact local lifecycle identity.
        :param reason: Stable request-local failure evidence.
        """

        self._runtime.submit(
            self._local_producer_id,
            binding_digest,
            NativeTerminalOwnerEventKind.DECODE_REQUEST_FAILED,
            reason=reason,
        )

    def _binding_digest(
        self,
        transaction: PackedDecodeRequestTransaction,
    ) -> bytes:
        """Resolve one actor transaction to its sole native binding.

        :param transaction: Exact registered actor transaction.
        :returns: Exact local native lifecycle digest.
        """

        return self._actor.terminal_owner_binding(transaction).digest

    @staticmethod
    def _require_action(
        action: NativeTerminalOwnerAction,
        kind: NativeTerminalOwnerActionKind,
    ) -> None:
        """Validate one exact native action dispatch boundary.

        :param action: Candidate action drained from a runtime inbox.
        :param kind: Required action kind for the operation.
        """

        if type(action) is not NativeTerminalOwnerAction:
            raise TypeError("action must be NativeTerminalOwnerAction")
        if action.kind is not kind:
            raise ValueError(f"decode operation requires {kind.name}")
