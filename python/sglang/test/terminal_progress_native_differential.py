import dataclasses
import time

from sglang.srt.disaggregation.common.packed_staging_protocol import PackedRequestKey
from sglang.srt.disaggregation.terminal_progress.deadlines import (
    TerminalDeadlineKind,
    terminal_deadline_spec,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalPublicationIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.lifecycle import (
    DecodeLifecycleEventKind,
    SourceLifecycleEventKind,
    TerminalResourceKind,
)
from sglang.srt.disaggregation.terminal_progress.native_owner import (
    NativeTerminalLifecycleSnapshot,
    NativeTerminalOwner,
)
from sglang.srt.disaggregation.terminal_progress.native_state import (
    NativeDecodeLifecyclePhase,
    NativeSourceLifecyclePhase,
    NativeTerminalDeadlineKind,
    NativeTerminalLifecycleRegistration,
    NativeTerminalOwnerActionKind,
    NativeTerminalOwnerEvent,
    NativeTerminalOwnerEventKind,
    NativeTerminalOwnerFatalCode,
    NativeTerminalOwnerRole,
    NativeTerminalProcessIdentity,
    NativeTerminalProducerClass,
    NativeTerminalProducerRegistration,
    NativeTerminalPublicationIdentity,
    NativeTerminalReceipt,
    NativeTerminalReceiptKind,
    NativeTerminalReceiptOutcome,
    NativeTerminalRequestBinding,
    NativeTerminalResource,
)
from sglang.srt.disaggregation.terminal_progress.receipts import (
    TerminalReceiptKind,
)
from sglang.test.terminal_progress_native_oracle import (
    OracleEventSpec,
    OracleLifecyclePath,
    OracleLifecycleProjection,
    OracleOwnerAction,
    OracleReceiptBinding,
    OracleReceiptIssuer,
    OracleReceiptSpec,
    OracleReductionError,
    OracleTransitionCase,
    OracleTransitionEvaluation,
    decode_oracle_paths,
    evaluate_oracle_transition,
    make_oracle_event,
    source_oracle_paths,
)

_SOURCE_PROCESS_GENERATION = bytes.fromhex("102132435465768798a9bacbdcedfe0f")
_DECODE_PROCESS_GENERATION = bytes.fromhex("ffeeddccbbaa99887766554433221100")
_UNTRUSTED_PROCESS_GENERATION = bytes.fromhex("aabbccddeeff00112233445566778899")
_REQUEST_GENERATION = bytes.fromhex("00112233445566778899aabbccddeeff")
_PUBLICATION_GENERATION = bytes.fromhex("0123456789abcdeffedcba9876543210")
_RANK_MANIFEST_DIGEST = b"r" * 32
_ALLOCATION_DIGEST = b"a" * 32
_LOCAL_PRODUCER_ID = 1
_TRUSTED_PRODUCER_ID = 2
_UNTRUSTED_PRODUCER_ID = 3
_CONTROL_PRODUCER_ID = 4
_WAIT_SECONDS = 2.0

_SOURCE_EVENT_KINDS = {
    SourceLifecycleEventKind.SUBMISSION_ACCEPTED: (
        NativeTerminalOwnerEventKind.SOURCE_SUBMISSION_ACCEPTED
    ),
    SourceLifecycleEventKind.PRODUCER_COMPLETED: (
        NativeTerminalOwnerEventKind.SOURCE_PRODUCER_COMPLETED
    ),
    SourceLifecycleEventKind.GATHER_COMPLETED_AND_NATIVE_POSTED: (
        NativeTerminalOwnerEventKind.SOURCE_GATHER_POSTED
    ),
    SourceLifecycleEventKind.NATIVE_TRANSFER_TERMINAL: (
        NativeTerminalOwnerEventKind.SOURCE_NATIVE_TERMINAL
    ),
    SourceLifecycleEventKind.OUTCOMES_SENT: (
        NativeTerminalOwnerEventKind.SOURCE_OUTCOMES_SENT
    ),
    SourceLifecycleEventKind.TEARDOWN_RECEIVED: (
        NativeTerminalOwnerEventKind.SOURCE_TEARDOWN_RECEIVED
    ),
    SourceLifecycleEventKind.ACK_SENT: NativeTerminalOwnerEventKind.SOURCE_ACK_SENT,
    SourceLifecycleEventKind.REQUEST_READY_RECEIVED: (
        NativeTerminalOwnerEventKind.SOURCE_REQUEST_READY
    ),
    SourceLifecycleEventKind.RECLAIM_CONSUMED: (
        NativeTerminalOwnerEventKind.SOURCE_RECLAIM_CONSUMED
    ),
    SourceLifecycleEventKind.GATEWAY_PUBLISHED: (
        NativeTerminalOwnerEventKind.SOURCE_GATEWAY_PUBLISHED
    ),
    SourceLifecycleEventKind.PUBLICATION_FAILED: (
        NativeTerminalOwnerEventKind.SOURCE_PUBLICATION_FAILED
    ),
    SourceLifecycleEventKind.REQUEST_FAILED: (
        NativeTerminalOwnerEventKind.SOURCE_REQUEST_FAILED
    ),
    SourceLifecycleEventKind.OWNER_DIED: (
        NativeTerminalOwnerEventKind.SOURCE_OWNER_DIED
    ),
    SourceLifecycleEventKind.PUBLISHER_DIED: (
        NativeTerminalOwnerEventKind.SOURCE_PUBLISHER_DIED
    ),
    SourceLifecycleEventKind.SCHEDULER_INBOX_OVERFLOW: (
        NativeTerminalOwnerEventKind.SOURCE_INBOX_OVERFLOW
    ),
}
_DECODE_EVENT_KINDS = {
    DecodeLifecycleEventKind.ALLOCATION_PUBLISHED: (
        NativeTerminalOwnerEventKind.DECODE_ALLOCATION_PUBLISHED
    ),
    DecodeLifecycleEventKind.WRITER_AGGREGATION_STARTED: (
        NativeTerminalOwnerEventKind.DECODE_WRITER_AGGREGATION_STARTED
    ),
    DecodeLifecycleEventKind.WRITER_MANIFEST_COMPLETED: (
        NativeTerminalOwnerEventKind.DECODE_WRITER_MANIFEST_COMPLETED
    ),
    DecodeLifecycleEventKind.SCATTER_STARTED: (
        NativeTerminalOwnerEventKind.DECODE_SCATTER_STARTED
    ),
    DecodeLifecycleEventKind.SCATTER_TERMINAL: (
        NativeTerminalOwnerEventKind.DECODE_SCATTER_TERMINAL
    ),
    DecodeLifecycleEventKind.TEARDOWN_SENT: (
        NativeTerminalOwnerEventKind.DECODE_TEARDOWN_SENT
    ),
    DecodeLifecycleEventKind.ACK_AGGREGATION_STARTED: (
        NativeTerminalOwnerEventKind.DECODE_ACK_AGGREGATION_STARTED
    ),
    DecodeLifecycleEventKind.ACK_MANIFEST_COMPLETED: (
        NativeTerminalOwnerEventKind.DECODE_ACK_MANIFEST_COMPLETED
    ),
    DecodeLifecycleEventKind.ADOPTION_CONSUMED: (
        NativeTerminalOwnerEventKind.DECODE_ADOPTION_CONSUMED
    ),
    DecodeLifecycleEventKind.METADATA_CONSUMED: (
        NativeTerminalOwnerEventKind.DECODE_METADATA_CONSUMED
    ),
    DecodeLifecycleEventKind.LOCAL_DECODE_READY_ISSUED: (
        NativeTerminalOwnerEventKind.DECODE_LOCAL_READY_ISSUED
    ),
    DecodeLifecycleEventKind.REQUEST_READY_RECEIVED: (
        NativeTerminalOwnerEventKind.DECODE_REQUEST_READY
    ),
    DecodeLifecycleEventKind.CANCEL_UNPUBLISHED: (
        NativeTerminalOwnerEventKind.DECODE_CANCEL_UNPUBLISHED
    ),
    DecodeLifecycleEventKind.REQUEST_FAILED: (
        NativeTerminalOwnerEventKind.DECODE_REQUEST_FAILED
    ),
    DecodeLifecycleEventKind.OWNER_DIED: (
        NativeTerminalOwnerEventKind.DECODE_OWNER_DIED
    ),
    DecodeLifecycleEventKind.SCHEDULER_INBOX_OVERFLOW: (
        NativeTerminalOwnerEventKind.DECODE_INBOX_OVERFLOW
    ),
}
_PROCESS_FATAL_EVENT_KINDS = frozenset(
    (
        SourceLifecycleEventKind.OWNER_DIED,
        SourceLifecycleEventKind.PUBLISHER_DIED,
        SourceLifecycleEventKind.SCHEDULER_INBOX_OVERFLOW,
        DecodeLifecycleEventKind.OWNER_DIED,
        DecodeLifecycleEventKind.SCHEDULER_INBOX_OVERFLOW,
    )
)
_CONTROL_EVENT_KINDS = frozenset(
    (
        SourceLifecycleEventKind.TEARDOWN_RECEIVED,
        DecodeLifecycleEventKind.WRITER_AGGREGATION_STARTED,
        DecodeLifecycleEventKind.WRITER_MANIFEST_COMPLETED,
        DecodeLifecycleEventKind.ACK_AGGREGATION_STARTED,
        DecodeLifecycleEventKind.ACK_MANIFEST_COMPLETED,
    )
)


@dataclasses.dataclass(frozen=True, slots=True)
class NativeDifferentialStep:
    """Observed native disposition for one semantic event.

    :ivar submitted: Whether native admission accepted the event record.
    :ivar before: Native lifecycle before the event.
    :ivar after: Native lifecycle after processing or admission rejection.
    :ivar actions: Externally actionable native side effects.
    :ivar receipts: Authority receipts carried by those actions.
    :ivar fatal_code: Sticky native process-fatal disposition.
    :ivar output_previous_phases: Pre-transition phases recorded on outputs.
    """

    submitted: bool
    before: NativeTerminalLifecycleSnapshot
    after: NativeTerminalLifecycleSnapshot
    actions: tuple[NativeTerminalOwnerActionKind, ...]
    receipts: tuple[NativeTerminalReceiptKind, ...]
    fatal_code: NativeTerminalOwnerFatalCode
    output_previous_phases: tuple[int, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class NativeDifferentialResult:
    """Complete native-versus-canonical verdict for one transition case.

    :ivar case: Exact canonical state/event pair.
    :ivar expected: Canonical Python reducer verdict.
    :ivar observed: Native owner observation.
    :ivar mismatches: Complete semantic mismatch descriptions.
    """

    case: OracleTransitionCase
    expected: OracleTransitionEvaluation
    observed: NativeDifferentialStep
    mismatches: tuple[str, ...]


class NativeDifferentialPathError(RuntimeError):
    """A declared canonical path cannot be represented by native process state."""


@dataclasses.dataclass(frozen=True, slots=True)
class NativePublisherDeathBlastRadius:
    """Native sibling lifecycles after one shared publisher dies.

    :ivar target: Ready target named by the publisher-death event.
    :ivar reclaimed_sibling: Ready sibling whose reclaim was consumed first.
    :ivar live_sibling: Ready sibling whose reclaimables remain live.
    :ivar fatal_code: Sticky process-wide publisher-death disposition.
    """

    target: NativeTerminalLifecycleSnapshot
    reclaimed_sibling: NativeTerminalLifecycleSnapshot
    live_sibling: NativeTerminalLifecycleSnapshot
    fatal_code: NativeTerminalOwnerFatalCode


@dataclasses.dataclass(frozen=True, slots=True)
class NativeDeadlineArmContext:
    """One role-specific lifecycle context which arms a native deadline.

    :ivar name: Stable reader-facing context identity.
    :ivar kind: Exact native deadline kind armed by the context.
    :ivar role: Lifecycle role which owns the deadline.
    :ivar path_name: Canonical accepted path reaching the armed context.
    """

    name: str
    kind: NativeTerminalDeadlineKind
    role: TerminalOwnerRole
    path_name: str

    def __post_init__(self) -> None:
        """Validate one complete role-specific deadline context."""

        if type(self.name) is not str or len(self.name) == 0:
            raise ValueError("deadline context name must be non-empty")
        if type(self.kind) is not NativeTerminalDeadlineKind:
            raise TypeError("deadline context kind must be native")
        if type(self.role) is not TerminalOwnerRole:
            raise TypeError("deadline context role must be TerminalOwnerRole")
        if type(self.path_name) is not str or len(self.path_name) == 0:
            raise ValueError("deadline context path name must be non-empty")

    @property
    def duration_ns(self) -> int:
        """Return the frozen duration for this native deadline.

        :returns: Hash-bound deadline duration in nanoseconds.
        """

        return terminal_deadline_spec(TerminalDeadlineKind[self.kind.name]).duration_ns

    @property
    def armed_mask(self) -> int:
        """Return the single deadline bit expected in this context.

        :returns: Native deadline bit mask.
        """

        return 1 << (int(self.kind) - 1)


class _NativeOracleRuntime:
    """Fresh real native owner and deterministic receipt namespace."""

    _role: TerminalOwnerRole
    _owner: NativeTerminalOwner
    _binding: NativeTerminalRequestBinding
    _other_binding: NativeTerminalRequestBinding
    _bindings_by_room: dict[int, NativeTerminalRequestBinding]
    _owner_identity: NativeTerminalProcessIdentity
    _untrusted_identity: NativeTerminalProcessIdentity
    _deterministic_clock_ns: int | None
    _producer_sequences: dict[int, int]
    _receipts: dict[tuple[bytes, str], tuple[OracleReceiptSpec, NativeTerminalReceipt]]

    def __init__(
        self,
        role: TerminalOwnerRole,
        *,
        additional_source_room_ids: tuple[int, ...] = (),
        deterministic_clock_ns: int | None = None,
    ) -> None:
        """Construct and start one native owner with all test authorities.

        :param role: Source or decode lifecycle role.
        :param additional_source_room_ids: Extra source lifecycles sharing the
            same owner and publisher process.
        :param deterministic_clock_ns: Optional positive test-clock origin.
        """

        if role is not TerminalOwnerRole.SOURCE and len(additional_source_room_ids) > 0:
            raise ValueError("additional lifecycles require a source owner")
        if len(set(additional_source_room_ids)) != len(additional_source_room_ids):
            raise ValueError("additional source room identities must be unique")
        if 71 in additional_source_room_ids or 72 in additional_source_room_ids:
            raise ValueError("additional source rooms overlap oracle identities")
        self._role = role
        if deterministic_clock_ns is not None and (
            type(deterministic_clock_ns) is not int or deterministic_clock_ns <= 0
        ):
            raise ValueError("deterministic_clock_ns must be a positive integer")
        binding = _make_binding(role, room_id=71)
        other_binding = _make_binding(role, room_id=72)
        owner_identity = NativeTerminalProcessIdentity.from_identity(binding.owner)
        untrusted_identity = NativeTerminalProcessIdentity.from_identity(
            TerminalProcessIdentity(
                process_generation=_UNTRUSTED_PROCESS_GENERATION,
                role=role,
                tp_rank=1,
                tp_size=4,
            )
        )
        self._binding = NativeTerminalRequestBinding.from_binding(binding)
        self._other_binding = NativeTerminalRequestBinding.from_binding(other_binding)
        self._bindings_by_room = {71: self._binding}
        self._owner_identity = owner_identity
        self._untrusted_identity = untrusted_identity
        self._deterministic_clock_ns = deterministic_clock_ns
        self._producer_sequences = {
            _LOCAL_PRODUCER_ID: 0,
            _TRUSTED_PRODUCER_ID: 0,
            _UNTRUSTED_PRODUCER_ID: 0,
            _CONTROL_PRODUCER_ID: 0,
        }
        self._receipts = {}
        self._owner = NativeTerminalOwner(
            input_capacity=64,
            output_capacity=64,
            owner_identity=owner_identity,
            testing=True,
        )
        if deterministic_clock_ns is not None:
            self._owner.enable_test_clock(deterministic_clock_ns)
        native_role = owner_identity.role
        self._owner.register_producer(
            NativeTerminalProducerRegistration(
                producer_id=_LOCAL_PRODUCER_ID,
                name="oracle-local",
                producer_class=NativeTerminalProducerClass.LOCAL,
                allowed_role=native_role,
                authenticated_issuer=None,
            )
        )
        self._owner.register_producer(
            NativeTerminalProducerRegistration(
                producer_id=_TRUSTED_PRODUCER_ID,
                name="oracle-trusted-receipt",
                producer_class=NativeTerminalProducerClass.RECEIPT,
                allowed_role=native_role,
                authenticated_issuer=owner_identity,
            )
        )
        self._owner.register_producer(
            NativeTerminalProducerRegistration(
                producer_id=_UNTRUSTED_PRODUCER_ID,
                name="oracle-untrusted-receipt",
                producer_class=NativeTerminalProducerClass.RECEIPT,
                allowed_role=native_role,
                authenticated_issuer=untrusted_identity,
            )
        )
        self._owner.register_producer(
            NativeTerminalProducerRegistration(
                producer_id=_CONTROL_PRODUCER_ID,
                name="oracle-trusted-control",
                producer_class=NativeTerminalProducerClass.CONTROL,
                allowed_role=native_role,
                authenticated_issuer=owner_identity,
            )
        )
        publication = None
        if role is TerminalOwnerRole.SOURCE:
            publication = NativeTerminalPublicationIdentity.from_identity(
                TerminalPublicationIdentity(
                    request_key=binding.request_key,
                    publisher_process_generation=_SOURCE_PROCESS_GENERATION,
                    publication_generation=_PUBLICATION_GENERATION,
                )
            )
        self._owner.register_lifecycle(
            NativeTerminalLifecycleRegistration(
                binding=self._binding,
                publication_identity=publication,
                trusted_issuers=(owner_identity,),
            )
        )
        for room_id in additional_source_room_ids:
            additional_binding = _make_binding(role, room_id=room_id)
            native_binding = NativeTerminalRequestBinding.from_binding(
                additional_binding
            )
            self._bindings_by_room[room_id] = native_binding
            self._owner.register_lifecycle(
                NativeTerminalLifecycleRegistration(
                    binding=native_binding,
                    publication_identity=(
                        NativeTerminalPublicationIdentity.from_identity(
                            TerminalPublicationIdentity(
                                request_key=additional_binding.request_key,
                                publisher_process_generation=(
                                    _SOURCE_PROCESS_GENERATION
                                ),
                                publication_generation=room_id.to_bytes(16, "big"),
                            )
                        )
                    ),
                    trusted_issuers=(owner_identity,),
                )
            )
        self._owner.start()
        for registered_binding in self._bindings_by_room.values():
            if not self._owner.wait_for_lifecycle_registration(
                registered_binding.digest, _WAIT_SECONDS
            ):
                raise TimeoutError("native lifecycle registration did not commit")

    def abort_and_close(self) -> None:
        """Release the bounded test owner without requiring retirement."""

        self._owner.abort_and_close()

    def apply(
        self,
        spec: OracleEventSpec,
        *,
        owner_minted_local_failure: bool = False,
        room_id: int = 71,
    ) -> NativeDifferentialStep:
        """Submit one semantic event and retain its native disposition.

        :param spec: Implementation-independent event specification.
        :param owner_minted_local_failure: Exercise the receipt-less local
            failure path whose authority is minted inside native state.
        :param room_id: Exact registered lifecycle receiving the event.
        :returns: Complete native state and side-effect observation.
        """

        binding = self._binding_for_room(room_id)
        before = self._snapshot(binding)
        before_inventory = self._owner.inventory()
        event_kind = _native_event_kind(spec)
        producer_id = self._producer_id(spec, owner_minted_local_failure)
        producer_sequence = self._producer_sequences[producer_id]
        receipt = None
        if not owner_minted_local_failure:
            receipt = self._materialize_receipt(spec.receipt, binding)
        enqueued_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        if self._deterministic_clock_ns is not None:
            enqueued_ns = self._deterministic_clock_ns
        event_mapping: dict[str, object] = {
            "producer_id": producer_id,
            "producer_sequence": producer_sequence,
            "binding_digest": binding.digest,
            "kind": int(event_kind),
            "enqueued_ns": enqueued_ns,
            "receipt": None if receipt is None else receipt.to_native(),
            "reason": spec.reason,
        }
        structurally_adversarial = owner_minted_local_failure or (
            spec.receipt is not None
            and (
                spec.receipt.binding is OracleReceiptBinding.OTHER_REQUEST
                or (
                    receipt is not None
                    and (
                        receipt.kind is not _expected_native_receipt_kind(spec)
                        or receipt.outcome is not _expected_native_receipt_outcome(spec)
                    )
                )
            )
        )
        submitted = True
        try:
            if structurally_adversarial:
                self._owner.submit_unchecked_for_testing(event_mapping)
            else:
                self._owner.submit(
                    NativeTerminalOwnerEvent(
                        producer_id=producer_id,
                        binding_digest=binding.digest,
                        kind=event_kind,
                        enqueued_ns=int(event_mapping["enqueued_ns"]),
                        receipt=receipt,
                        reason=spec.reason,
                    )
                )
        except OSError:
            submitted = False
        if submitted:
            self._producer_sequences[producer_id] += 1
            self._wait_for_dispatch(before_inventory.transition_count)
        outputs = self._owner.drain_outputs()
        actions = tuple(action.kind for output in outputs for action in output.actions)
        receipts = tuple(
            action.receipt.kind
            for output in outputs
            for action in output.actions
            if action.receipt is not None
        )
        previous_phases = tuple(output.previous_phase for output in outputs)
        for output in outputs:
            for action in output.actions:
                self._owner.acknowledge_action(action)
        inventory = self._owner.inventory()
        return NativeDifferentialStep(
            submitted=submitted,
            before=before,
            after=self._snapshot(binding),
            actions=actions,
            receipts=receipts,
            fatal_code=inventory.fatal_code,
            output_previous_phases=previous_phases,
        )

    def expire_deadline_at(self, now_ns: int) -> NativeDifferentialStep:
        """Expire armed deadlines and return their lifecycle disposition.

        :param now_ns: Deterministic test timestamp.
        :returns: Native state and emitted fail-closed actions.
        """

        binding = self._binding
        before = self._snapshot(binding)
        self.advance_clock_and_expire(now_ns)
        outputs = self._owner.drain_outputs()
        actions = tuple(action.kind for output in outputs for action in output.actions)
        receipts = tuple(
            action.receipt.kind
            for output in outputs
            for action in output.actions
            if action.receipt is not None
        )
        previous_phases = tuple(output.previous_phase for output in outputs)
        for output in outputs:
            for action in output.actions:
                self._owner.acknowledge_action(action)
        inventory = self._owner.inventory()
        return NativeDifferentialStep(
            submitted=True,
            before=before,
            after=self._snapshot(binding),
            actions=actions,
            receipts=receipts,
            fatal_code=inventory.fatal_code,
            output_previous_phases=previous_phases,
        )

    def snapshot(self, room_id: int = 71) -> NativeTerminalLifecycleSnapshot:
        """Return one registered lifecycle projection.

        :param room_id: Exact registered source room identity.
        :returns: Immutable native lifecycle snapshot.
        """

        return self._snapshot(self._binding_for_room(room_id))

    def advance_clock_and_expire(self, now_ns: int) -> None:
        """Advance deterministic time and evaluate armed deadline boundaries.

        :param now_ns: Positive monotonic test timestamp.
        """

        self._owner.set_test_clock(now_ns)
        self._deterministic_clock_ns = now_ns
        self._owner.expire_deadlines_for_testing()

    def _snapshot(
        self, binding: NativeTerminalRequestBinding
    ) -> NativeTerminalLifecycleSnapshot:
        """Return this runtime's exact lifecycle projection.

        :param binding: Exact registered native request binding.
        :returns: Immutable native lifecycle snapshot.
        """

        return self._owner.lifecycle_snapshot_for_testing(binding.digest)

    def _binding_for_room(self, room_id: int) -> NativeTerminalRequestBinding:
        """Return one exact registered lifecycle binding.

        :param room_id: Stable packed room identity.
        :returns: Native request binding.
        """

        binding = self._bindings_by_room.get(room_id)
        if binding is None:
            raise KeyError(f"native oracle room is absent: {room_id}")
        return binding

    def _producer_id(
        self, spec: OracleEventSpec, owner_minted_local_failure: bool
    ) -> int:
        """Select the independently authenticated route for one event.

        :param spec: Semantic event specification.
        :param owner_minted_local_failure: Whether native owns failure authority.
        :returns: Registered native producer identity.
        """

        if spec.kind in _CONTROL_EVENT_KINDS:
            return _CONTROL_PRODUCER_ID
        if owner_minted_local_failure or spec.receipt is None:
            return _LOCAL_PRODUCER_ID
        if spec.receipt.issuer is OracleReceiptIssuer.TRUSTED:
            return _TRUSTED_PRODUCER_ID
        return _UNTRUSTED_PRODUCER_ID

    def _materialize_receipt(
        self,
        spec: OracleReceiptSpec | None,
        target_binding: NativeTerminalRequestBinding,
    ) -> NativeTerminalReceipt | None:
        """Mint or replay one deterministic native receipt.

        :param spec: Optional semantic authority specification.
        :param target_binding: Exact lifecycle receiving the authority.
        :returns: Fixed-width native receipt, when required.
        """

        if spec is None:
            return None
        cache_key = (target_binding.digest, spec.key)
        cached = self._receipts.get(cache_key)
        if cached is not None:
            cached_spec, receipt = cached
            if cached_spec != spec:
                raise ValueError("one receipt key cannot describe two authorities")
            return receipt
        binding = (
            target_binding
            if spec.binding is OracleReceiptBinding.TARGET
            else self._other_binding
        )
        issuer = (
            self._owner_identity
            if spec.issuer is OracleReceiptIssuer.TRUSTED
            else self._untrusted_identity
        )
        receipt = NativeTerminalReceipt(
            binding=binding,
            issuer=issuer,
            kind=NativeTerminalReceiptKind[spec.kind.name],
            outcome=NativeTerminalReceiptOutcome[spec.outcome.name],
            terminal_timestamp_ns=len(self._receipts) + 1,
            nonce=spec.deterministic_nonce,
        )
        self._receipts[cache_key] = (spec, receipt)
        return receipt

    def _wait_for_dispatch(self, before_transition_count: int) -> None:
        """Wait until the real native reactor accepts or fail-closes an event.

        :param before_transition_count: Commit count before submission.
        :raises TimeoutError: If the native reactor makes no bounded progress.
        """

        deadline = time.monotonic() + _WAIT_SECONDS
        while time.monotonic() < deadline:
            inventory = self._owner.inventory()
            if (
                inventory.transition_count > before_transition_count
                or inventory.fatal_code is not NativeTerminalOwnerFatalCode.NONE
            ):
                return
            time.sleep(0)
        raise TimeoutError("native terminal owner did not dispatch a test event")


def evaluate_native_differential_case(
    case: OracleTransitionCase,
    *,
    owner_minted_local_failure: bool = False,
) -> NativeDifferentialResult:
    """Evaluate one canonical state/event pair through the real native owner.

    :param case: Exact accepted history plus candidate event.
    :param owner_minted_local_failure: Exercise native-local failure authority.
    :returns: Complete verdict with every observed semantic mismatch.
    """

    if type(case) is not OracleTransitionCase:
        raise TypeError("case must be OracleTransitionCase")
    expected = evaluate_oracle_transition(case)
    runtime = _NativeOracleRuntime(case.path.role)
    path = OracleLifecyclePath(
        name=f"{case.path.name}-native-prefix",
        role=case.path.role,
        events=(),
    )
    last_path_step: NativeDifferentialStep | None = None
    try:
        for index, event in enumerate(case.path.events):
            prefix_case = OracleTransitionCase(
                name=f"{case.name}-path-{index}",
                path=path,
                event=event,
            )
            prefix_expected = evaluate_oracle_transition(prefix_case)
            if not prefix_expected.accepted:
                raise RuntimeError(f"declared oracle path rejected {event.kind.value}")
            prefix_observed = runtime.apply(event)
            last_path_step = prefix_observed
            prefix_mismatches = _compare_step(prefix_expected, prefix_observed)
            if len(prefix_mismatches) > 0:
                raise NativeDifferentialPathError(
                    f"{case.path.name} diverged at path[{index}] "
                    f"{event.kind.value}: {'; '.join(prefix_mismatches)}"
                )
            if prefix_observed.after.process_fatal and index + 1 < len(
                case.path.events
            ):
                raise NativeDifferentialPathError(
                    f"{case.path.name} continues after process-fatal "
                    f"path[{index}] {event.kind.value}"
                )
            path = OracleLifecyclePath(
                name=f"{case.path.name}-native-prefix-{index}",
                role=case.path.role,
                events=(*path.events, event),
            )
        if last_path_step is not None and last_path_step.after.process_fatal:
            raise NativeDifferentialPathError(
                f"{case.path.name} ends in process-fatal state and cannot "
                "accept a candidate event"
            )
        observed = runtime.apply(
            case.event,
            owner_minted_local_failure=owner_minted_local_failure,
        )
        return NativeDifferentialResult(
            case=case,
            expected=expected,
            observed=observed,
            mismatches=_compare_step(expected, observed),
        )
    finally:
        runtime.abort_and_close()


def native_deadline_arm_contexts() -> tuple[NativeDeadlineArmContext, ...]:
    """Return every role-specific context which arms native deadline kinds 3-9.

    Deadline kinds shared by source and decode have one context per role. The
    source scheduler-consumption and publication contexts complete the other
    join first so exactly one same-anchor deadline remains armed.

    :returns: Ten contexts ordered by native deadline kind and lifecycle role.
    """

    return (
        NativeDeadlineArmContext(
            name="source-producer-and-gather",
            kind=NativeTerminalDeadlineKind.OWNER_PRODUCER_AND_GATHER,
            role=TerminalOwnerRole.SOURCE,
            path_name="source-waiting_for_producer",
        ),
        NativeDeadlineArmContext(
            name="source-native-transfer",
            kind=NativeTerminalDeadlineKind.OWNER_NATIVE_TRANSFER,
            role=TerminalOwnerRole.SOURCE,
            path_name="source-native_in_flight",
        ),
        NativeDeadlineArmContext(
            name="decode-scatter",
            kind=NativeTerminalDeadlineKind.OWNER_DECODE_SCATTER,
            role=TerminalOwnerRole.DECODE,
            path_name="decode-scatter_in_flight",
        ),
        NativeDeadlineArmContext(
            name="source-teardown-ack",
            kind=NativeTerminalDeadlineKind.OWNER_TEARDOWN_ACK,
            role=TerminalOwnerRole.SOURCE,
            path_name="source-outcomes_sent",
        ),
        NativeDeadlineArmContext(
            name="decode-teardown-ack",
            kind=NativeTerminalDeadlineKind.OWNER_TEARDOWN_ACK,
            role=TerminalOwnerRole.DECODE,
            path_name="decode-teardown_sent",
        ),
        NativeDeadlineArmContext(
            name="source-request-global-ready",
            kind=NativeTerminalDeadlineKind.OWNER_REQUEST_GLOBAL_READY,
            role=TerminalOwnerRole.SOURCE,
            path_name="source-ack_sent",
        ),
        NativeDeadlineArmContext(
            name="decode-request-global-ready",
            kind=NativeTerminalDeadlineKind.OWNER_REQUEST_GLOBAL_READY,
            role=TerminalOwnerRole.DECODE,
            path_name="decode-local-ready",
        ),
        NativeDeadlineArmContext(
            name="source-scheduler-receipt-consumption",
            kind=NativeTerminalDeadlineKind.OWNER_SCHEDULER_RECEIPT_CONSUMPTION,
            role=TerminalOwnerRole.SOURCE,
            path_name="source-ready-gateway-published",
        ),
        NativeDeadlineArmContext(
            name="decode-scheduler-receipt-consumption",
            kind=NativeTerminalDeadlineKind.OWNER_SCHEDULER_RECEIPT_CONSUMPTION,
            role=TerminalOwnerRole.DECODE,
            path_name="decode-adoption_ready",
        ),
        NativeDeadlineArmContext(
            name="source-gateway-publication",
            kind=NativeTerminalDeadlineKind.OWNER_GATEWAY_PUBLICATION,
            role=TerminalOwnerRole.SOURCE,
            path_name="source-ready-reclaim-consumed",
        ),
    )


def evaluate_native_deadline_boundary(
    context: NativeDeadlineArmContext,
    *,
    started_ns: int,
    now_ns: int,
) -> NativeDifferentialStep:
    """Evaluate one armed native deadline at a deterministic timestamp.

    :param context: Exact role-specific deadline context.
    :param started_ns: Deterministic timestamp used by the arming transition.
    :param now_ns: Deterministic timestamp at which expiry is evaluated.
    :returns: Native lifecycle disposition and emitted actions.
    """

    if type(context) is not NativeDeadlineArmContext:
        raise TypeError("context must be NativeDeadlineArmContext")
    if type(started_ns) is not int or started_ns <= 0:
        raise ValueError("started_ns must be a positive integer")
    if type(now_ns) is not int or now_ns < started_ns:
        raise ValueError("now_ns must not precede started_ns")
    paths = source_oracle_paths()
    if context.role is TerminalOwnerRole.DECODE:
        paths = decode_oracle_paths()
    matches = tuple(path for path in paths if path.name == context.path_name)
    if len(matches) != 1:
        raise RuntimeError("deadline context path must resolve exactly once")
    path = matches[0]
    runtime = _NativeOracleRuntime(
        context.role,
        deterministic_clock_ns=started_ns,
    )
    try:
        for event in path.events:
            step = runtime.apply(event)
            if step.fatal_code is not NativeTerminalOwnerFatalCode.NONE:
                raise RuntimeError("deadline setup became process-fatal")
        armed = runtime.snapshot()
        if armed.armed_deadline_mask != context.armed_mask:
            raise RuntimeError(
                "deadline context did not isolate its exact native deadline"
            )
        return runtime.expire_deadline_at(now_ns)
    finally:
        runtime.abort_and_close()


def evaluate_native_publisher_death_blast_radius() -> NativePublisherDeathBlastRadius:
    """Exercise one publisher death across ready sibling source lifecycles.

    All three requests share the publisher process. The target and one sibling
    retain live reclaimables; the other sibling consumed reclaim authority
    before the publisher died. Only each publication identity may be
    quarantined because request-global readiness already made decode adoption
    independent of gateway publication.

    :returns: Exact post-fatal state of all three source lifecycles.
    """

    runtime = _NativeOracleRuntime(
        TerminalOwnerRole.SOURCE,
        additional_source_room_ids=(73, 74),
    )
    ready_path = next(
        path for path in source_oracle_paths() if path.name == "source-ready"
    )
    try:
        for room_id in (71, 73, 74):
            for event in ready_path.events:
                step = runtime.apply(event, room_id=room_id)
                if step.fatal_code is not NativeTerminalOwnerFatalCode.NONE:
                    raise RuntimeError("ready setup became process-fatal")
        runtime.apply(
            make_oracle_event(
                SourceLifecycleEventKind.RECLAIM_CONSUMED,
                receipt_key="publisher-blast-reclaimed-sibling",
            ),
            room_id=73,
        )
        fatal = runtime.apply(
            make_oracle_event(SourceLifecycleEventKind.PUBLISHER_DIED),
            room_id=71,
        )
        return NativePublisherDeathBlastRadius(
            target=runtime.snapshot(71),
            reclaimed_sibling=runtime.snapshot(73),
            live_sibling=runtime.snapshot(74),
            fatal_code=fatal.fatal_code,
        )
    finally:
        runtime.abort_and_close()


def evaluate_native_post_publication_quarantine_request_failure() -> (
    NativeDifferentialResult
):
    """Exercise request failure after publication identity quarantine.

    The publication identity is already quarantined and remains so. A later
    authenticated request failure may only quarantine the still-live source
    resources; it cannot emit a second request-quarantine notification.

    :returns: Canonical and native verdict for the post-publication failure.
    """

    path = next(
        candidate
        for candidate in source_oracle_paths()
        if candidate.name == "source-publication-quarantined"
    )
    case = OracleTransitionCase(
        name="source-publication-quarantined--request-failed-action-idempotence",
        path=path,
        event=make_oracle_event(
            SourceLifecycleEventKind.REQUEST_FAILED,
            receipt_key="post-publication-quarantine-request-failure",
        ),
    )
    return evaluate_native_differential_case(case)


def _compare_step(
    expected: OracleTransitionEvaluation,
    observed: NativeDifferentialStep,
) -> tuple[str, ...]:
    """Compare canonical acceptance before native fail-closed disposition.

    :param expected: Canonical pure-reducer verdict.
    :param observed: Real native owner observation.
    :returns: Complete semantic mismatch descriptions.
    """

    mismatches: list[str] = []
    native_before = _project_native_snapshot(observed.before)
    if native_before != expected.before:
        mismatches.append(
            f"before state differs: expected={expected.before!r}, "
            f"native={native_before!r}"
        )
    event_is_process_fatal = (
        expected.accepted and expected.case.event.kind in _PROCESS_FATAL_EVENT_KINDS
    )
    native_accepted = observed.submitted and (
        observed.fatal_code is NativeTerminalOwnerFatalCode.NONE
        or (
            event_is_process_fatal
            and observed.fatal_code is NativeTerminalOwnerFatalCode.DEPENDENCY_DEATH
        )
    )
    if native_accepted != expected.accepted:
        mismatches.append(
            f"acceptance differs: expected={expected.accepted}, "
            f"native={native_accepted}, submitted={observed.submitted}, "
            f"fatal={observed.fatal_code.name}"
        )
    if expected.accepted:
        native_after = _project_native_snapshot(observed.after)
        if native_after != expected.after:
            mismatches.append(
                f"after state differs: expected={expected.after!r}, "
                f"native={native_after!r}"
            )
        expected_actions = tuple(
            action
            for action in expected.actions
            if action
            not in (
                OracleOwnerAction.STATE_COMMITTED,
                OracleOwnerAction.REQUEST_QUARANTINED,
                OracleOwnerAction.PROCESS_FATAL,
            )
        )
        native_actions = tuple(
            OracleOwnerAction[action.name]
            for action in observed.actions
            if action
            not in (
                NativeTerminalOwnerActionKind.REQUEST_QUARANTINED,
                NativeTerminalOwnerActionKind.PROCESS_FATAL,
            )
        )
        if native_actions != expected_actions:
            mismatches.append(
                f"actions differ: expected={expected_actions!r}, "
                f"native={native_actions!r}"
            )
        native_receipts = tuple(
            TerminalReceiptKind[receipt.name] for receipt in observed.receipts
        )
        if native_receipts != expected.emitted_receipts:
            mismatches.append(
                f"receipts differ: expected={expected.emitted_receipts!r}, "
                f"native={native_receipts!r}"
            )
    else:
        if observed.submitted:
            if observed.fatal_code is NativeTerminalOwnerFatalCode.NONE:
                mismatches.append("native rejection did not fail closed")
            if not observed.after.process_fatal:
                mismatches.append("native rejection retained a healthy process")
            if expected.case.event.kind in _PROCESS_FATAL_EVENT_KINDS:
                if (
                    observed.fatal_code
                    is not NativeTerminalOwnerFatalCode.DEPENDENCY_DEATH
                ):
                    mismatches.append(
                        "rejected process-fatal event did not preserve its fatal cause"
                    )
            else:
                expected_fatal = _expected_rejection_fatal_code(expected)
                if observed.fatal_code is not expected_fatal:
                    mismatches.append(
                        f"rejection code differs: expected={expected_fatal.name}, "
                        f"native={observed.fatal_code.name}"
                    )
            permission_actions = tuple(
                action
                for action in observed.actions
                if action is not NativeTerminalOwnerActionKind.PROCESS_FATAL
            )
            if len(permission_actions) > 0 or len(observed.receipts) > 0:
                mismatches.append(
                    "native rejection earned permission-bearing side effects"
                )
        elif not observed.before.process_fatal:
            mismatches.append("healthy native owner rejected admission synchronously")
    if len(observed.output_previous_phases) > 0 and any(
        phase != observed.before.phase for phase in observed.output_previous_phases
    ):
        mismatches.append(
            f"output previous phase differs from {observed.before.phase}: "
            f"{observed.output_previous_phases!r}"
        )
    return tuple(mismatches)


def _expected_rejection_fatal_code(
    expected: OracleTransitionEvaluation,
) -> NativeTerminalOwnerFatalCode:
    """Map a canonical rejection class to its native fail-closed code.

    :param expected: Canonical rejected transition.
    :returns: Exact native fatal code.
    """

    if expected.error is OracleReductionError.RECEIPT:
        if expected.error_message is not None and "already consumed" in (
            expected.error_message
        ):
            return NativeTerminalOwnerFatalCode.RECEIPT_REPLAY
        return NativeTerminalOwnerFatalCode.RECEIPT_AUTHORITY
    return NativeTerminalOwnerFatalCode.ILLEGAL_TRANSITION


def _project_native_snapshot(
    snapshot: NativeTerminalLifecycleSnapshot,
) -> OracleLifecycleProjection:
    """Project one native lifecycle into the implementation-neutral schema.

    :param snapshot: Exact native lifecycle projection.
    :returns: Canonical phase and resource projection.
    """

    if snapshot.role is NativeTerminalOwnerRole.SOURCE:
        role = TerminalOwnerRole.SOURCE
        phase = NativeSourceLifecyclePhase(snapshot.phase).name.lower()
    else:
        role = TerminalOwnerRole.DECODE
        phase = NativeDecodeLifecyclePhase(snapshot.phase).name.lower()
    return OracleLifecycleProjection(
        role=role,
        phase=phase,
        live_resources=_resources_from_mask(snapshot.live_resources),
        retired_resources=_resources_from_mask(snapshot.retired_resources),
        quarantined_resources=_resources_from_mask(snapshot.quarantined_resources),
        process_fatal=snapshot.process_fatal,
    )


def _resources_from_mask(mask: int) -> frozenset[TerminalResourceKind]:
    """Convert native resource bits into canonical resource identities.

    :param mask: Native resource bit mask.
    :returns: Canonical resource identity set.
    """

    return frozenset(
        TerminalResourceKind[resource.name]
        for resource in NativeTerminalResource
        if mask & int(resource) != 0
    )


def _native_event_kind(spec: OracleEventSpec) -> NativeTerminalOwnerEventKind:
    """Return the stable native event code for one semantic event.

    :param spec: Semantic event specification.
    :returns: Native event code.
    """

    if type(spec.kind) is SourceLifecycleEventKind:
        return _SOURCE_EVENT_KINDS[spec.kind]
    return _DECODE_EVENT_KINDS[spec.kind]


def _expected_native_receipt_kind(
    spec: OracleEventSpec,
) -> NativeTerminalReceiptKind | None:
    """Return the native receipt kind required by one semantic event.

    :param spec: Semantic event specification.
    :returns: Required native receipt kind, when receipt-bearing.
    """

    if spec.receipt is None:
        return None
    requirements = {
        SourceLifecycleEventKind.REQUEST_READY_RECEIVED: (
            NativeTerminalReceiptKind.REQUEST_READY
        ),
        SourceLifecycleEventKind.RECLAIM_CONSUMED: (
            NativeTerminalReceiptKind.RECLAIM_CONSUMED
        ),
        SourceLifecycleEventKind.GATEWAY_PUBLISHED: (
            NativeTerminalReceiptKind.GATEWAY_PUBLISHED
        ),
        SourceLifecycleEventKind.PUBLICATION_FAILED: (
            NativeTerminalReceiptKind.FAILURE
        ),
        SourceLifecycleEventKind.REQUEST_FAILED: NativeTerminalReceiptKind.FAILURE,
        DecodeLifecycleEventKind.ADOPTION_CONSUMED: (
            NativeTerminalReceiptKind.ADOPTION_READY
        ),
        DecodeLifecycleEventKind.REQUEST_READY_RECEIVED: (
            NativeTerminalReceiptKind.REQUEST_READY
        ),
        DecodeLifecycleEventKind.REQUEST_FAILED: NativeTerminalReceiptKind.FAILURE,
    }
    return requirements[spec.kind]


def _expected_native_receipt_outcome(
    spec: OracleEventSpec,
) -> NativeTerminalReceiptOutcome | None:
    """Return the native receipt outcome required by one semantic event.

    :param spec: Semantic event specification.
    :returns: Required native receipt outcome, when receipt-bearing.
    """

    if spec.receipt is None:
        return None
    if spec.kind in (
        SourceLifecycleEventKind.PUBLICATION_FAILED,
        SourceLifecycleEventKind.REQUEST_FAILED,
        DecodeLifecycleEventKind.REQUEST_FAILED,
    ):
        return NativeTerminalReceiptOutcome.FAILURE
    return NativeTerminalReceiptOutcome.SUCCESS


def _make_binding(role: TerminalOwnerRole, room_id: int) -> TerminalRequestBinding:
    """Construct one deterministic native-differential request binding.

    :param role: Source or decode owner role.
    :param room_id: Stable packed room identity.
    :returns: Complete canonical request binding.
    """

    process_generation = _SOURCE_PROCESS_GENERATION
    if role is TerminalOwnerRole.DECODE:
        process_generation = _DECODE_PROCESS_GENERATION
    return TerminalRequestBinding(
        request_key=PackedRequestKey(
            room_id=room_id,
            request_generation=_REQUEST_GENERATION,
        ),
        owner=TerminalProcessIdentity(
            process_generation=process_generation,
            role=role,
            tp_rank=1,
            tp_size=4,
        ),
        rank_manifest_digest=_RANK_MANIFEST_DIGEST,
        allocation_digest=_ALLOCATION_DIGEST,
    )
