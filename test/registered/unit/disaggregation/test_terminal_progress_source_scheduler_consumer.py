import concurrent.futures
import dataclasses
import hashlib
import threading

import pytest
from sglang.srt.disaggregation.common.packed_staging_protocol import PackedRequestKey
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalPublicationIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.inbox import (
    SchedulerInboxFatalCause,
)
from sglang.srt.disaggregation.terminal_progress.native_state import (
    NativeTerminalOwnerAction,
    NativeTerminalOwnerActionKind,
    NativeTerminalProcessIdentity,
    NativeTerminalReceipt,
    NativeTerminalReceiptKind,
    NativeTerminalReceiptOutcome,
    NativeTerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.publisher import (
    FrozenTerminalGatewayOutputProjection,
)
from sglang.srt.disaggregation.terminal_progress.receipts import (
    TerminalReceiptKind,
    TerminalReceiptOutcome,
)
from sglang.srt.disaggregation.terminal_progress.scheduler_inbox import (
    SchedulerReceiptInboxInventory,
)
from sglang.srt.disaggregation.terminal_progress.source_plan import (
    PackedTerminalSourceIdentityPlan,
)
from sglang.srt.disaggregation.terminal_progress.source_scheduler_consumer import (
    PackedTerminalSourceSchedulerConsumer,
    PackedTerminalSourceSchedulerRelease,
)
from sglang.srt.disaggregation.terminal_progress.source_wiring import (
    PackedTerminalSourceSubmission,
)
from sglang.srt.disaggregation.terminal_progress.wire import (
    IssuedTerminalWireReceipt,
    TerminalWireReceiptIssuer,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@dataclasses.dataclass(frozen=True, slots=True)
class _Projection(FrozenTerminalGatewayOutputProjection):
    """Minimal immutable gateway projection."""

    payload: bytes

    @property
    def digest(self) -> bytes:
        """Return the fixture projection digest.

        :returns: SHA-256 over the exact fixture payload.
        """

        return hashlib.sha256(self.payload).digest()


@dataclasses.dataclass(frozen=True, slots=True)
class _OpaqueTransportPayload:
    """Payload deliberately carrying no algorithm-specific schema."""

    marker: int


@dataclasses.dataclass(slots=True)
class _MutableSchedulerOwner:
    """Mutable request-owned resources which must remain scheduler-affine."""

    release_count: int = 0
    release_thread_id: int | None = None


class _SourceWiring:
    """Minimal source wiring which exercises the exact release callback."""

    submission: PackedTerminalSourceSubmission
    issuer: TerminalWireReceiptIssuer
    fail_after_release: bool
    calls: int

    def __init__(
        self,
        submission: PackedTerminalSourceSubmission,
        *,
        fail_after_release: bool = False,
    ) -> None:
        """Construct one deterministic reclaim-wiring fixture.

        :param submission: Exact immutable source handoff.
        :param fail_after_release: Whether receipt publication fails after the
            scheduler-owned resource operation returns.
        """

        self.submission = submission
        self.issuer = TerminalWireReceiptIssuer(
            submission.identity.local_binding.owner
        )
        self.fail_after_release = fail_after_release
        self.calls = 0

    def consume_reclaim_authorized(
        self,
        action: NativeTerminalOwnerAction,
        release_resources,
    ) -> IssuedTerminalWireReceipt:
        """Run the retained callback and optionally fail after resource release.

        :param action: Exact reclaim authority.
        :param release_resources: Scheduler-owned source release operation.
        :returns: Deterministic reclaim-consumed authority.
        """

        self.calls += 1
        release_resources(self.submission)
        if self.fail_after_release:
            raise RuntimeError("synthetic post-release receipt failure")
        return self.issuer.issue(
            binding=self.submission.identity.local_binding,
            kind=TerminalReceiptKind.RECLAIM_CONSUMED,
            outcome=TerminalReceiptOutcome.SUCCESS,
            terminal_timestamp_ns=700,
        )


def _submission(
    marker: int = 1,
) -> tuple[PackedTerminalSourceSubmission, _OpaqueTransportPayload]:
    """Build a source submission without an algorithm-specific payload.

    :param marker: Stable generation marker.
    :returns: Immutable submission and its opaque transport payload.
    """

    key = PackedRequestKey(
        room_id=marker,
        request_generation=bytes((marker,)) * 16,
    )
    source = TerminalProcessIdentity(
        process_generation=bytes((marker + 1,)) * 16,
        role=TerminalOwnerRole.SOURCE,
        tp_rank=0,
        tp_size=1,
    )
    decode = TerminalProcessIdentity(
        process_generation=bytes((marker + 2,)) * 16,
        role=TerminalOwnerRole.DECODE,
        tp_rank=0,
        tp_size=1,
    )
    binding = TerminalRequestBinding(
        request_key=key,
        owner=source,
        rank_manifest_digest=bytes((marker + 3,)) * 32,
        allocation_digest=bytes((marker + 4,)) * 32,
    )
    publication = TerminalPublicationIdentity(
        request_key=key,
        publisher_process_generation=source.process_generation,
        publication_generation=bytes((marker + 5,)) * 16,
    )
    identity = PackedTerminalSourceIdentityPlan(
        local_binding=binding,
        source_bindings=(binding,),
        publication_identity=publication,
        request_ready_issuer=decode,
        publisher_issuer=source,
    )
    payload = _OpaqueTransportPayload(marker=marker)
    return (
        PackedTerminalSourceSubmission(
            identity=identity,
            output_projection=_Projection(payload=b"output"),
            producer_event_generation=bytes((marker + 6,)) * 16,
            producer_stream_handle=marker + 7,
            transport_submission=payload,
        ),
        payload,
    )


def _action(
    binding: TerminalRequestBinding,
    action_id: int = 1,
) -> NativeTerminalOwnerAction:
    """Build one exact owner-minted reclaim action.

    :param binding: Source lifecycle earning reclaim authority.
    :param action_id: Stable one-shot action identity.
    :returns: Validated native reclaim action.
    """

    native_binding = NativeTerminalRequestBinding.from_binding(binding)
    native_issuer = NativeTerminalProcessIdentity.from_identity(binding.owner)
    receipt = NativeTerminalReceipt(
        binding=native_binding,
        issuer=native_issuer,
        kind=NativeTerminalReceiptKind.RECLAIM_AUTHORIZED,
        outcome=NativeTerminalReceiptOutcome.SUCCESS,
        terminal_timestamp_ns=action_id,
        nonce=action_id.to_bytes(16, "big"),
    )
    return NativeTerminalOwnerAction(
        action_id=action_id,
        kind=NativeTerminalOwnerActionKind.RECLAIM_AUTHORIZED,
        binding=native_binding,
        commit_timestamp_ns=action_id,
        receipt=receipt,
    )


def _fatal_inventory(
    binding: TerminalRequestBinding,
) -> SchedulerReceiptInboxInventory:
    """Build sticky owner-death evidence for one live source generation.

    :param binding: Exact retained source generation.
    :returns: Valid scheduler-inbox fatal inventory.
    """

    return SchedulerReceiptInboxInventory(
        physical_capacity=1,
        live_bindings=(binding,),
        pending_request_keys=(),
        consuming_request_keys=(),
        outstanding_publications=0,
        wake_armed=True,
        fatal_cause=SchedulerInboxFatalCause.OWNER_DEATH,
        closed=False,
    )


def test_reclaim_consumes_exact_scheduler_owned_release() -> None:
    """Owner authority releases opaque transport resources exactly once."""

    submission, payload = _submission()
    binding = submission.identity.local_binding
    wiring = _SourceWiring(submission)
    fatal_inventories: list[SchedulerReceiptInboxInventory] = []
    consumer = PackedTerminalSourceSchedulerConsumer(
        wiring=wiring,
        process_fatal_handler=fatal_inventories.append,
    )
    mutable_owner = _MutableSchedulerOwner()
    scheduler_thread_id = threading.get_ident()

    def release_resources(actual: PackedTerminalSourceSubmission) -> None:
        """Release mutable request ownership only on the scheduler thread.

        :param actual: Exact immutable source submission from wiring.
        """

        assert threading.get_ident() == scheduler_thread_id
        assert actual.transport_submission is payload
        mutable_owner.release_count += 1
        mutable_owner.release_thread_id = threading.get_ident()

    consumer.register_release(
        PackedTerminalSourceSchedulerRelease(
            binding=binding,
            release_resources=release_resources,
        )
    )
    consumer.consume_reclaim_authorized(_action(binding))

    assert mutable_owner.release_count == 1
    assert mutable_owner.release_thread_id == scheduler_thread_id
    assert wiring.calls == 1
    assert fatal_inventories == []
    inventory = consumer.inventory()
    assert inventory.active_binding_digests == ()
    assert inventory.quarantined_binding_digests == ()
    assert inventory.resource_release_completed_binding_digests == ()
    assert inventory.released_count == 1


def test_reclaim_rejects_owner_thread_mutation() -> None:
    """The immutable native action cannot move release work off scheduler."""

    submission, _ = _submission()
    binding = submission.identity.local_binding
    wiring = _SourceWiring(submission)
    consumer = PackedTerminalSourceSchedulerConsumer(
        wiring=wiring,
        process_fatal_handler=lambda inventory: None,
    )
    release_calls: list[int] = []
    consumer.register_release(
        PackedTerminalSourceSchedulerRelease(
            binding=binding,
            release_resources=lambda actual: release_calls.append(1),
        )
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            consumer.consume_reclaim_authorized,
            _action(binding),
        )
        with pytest.raises(
            RuntimeError,
            match="source scheduler resources crossed thread affinity",
        ):
            future.result()

    assert release_calls == []
    assert wiring.calls == 0
    assert consumer.inventory().active_binding_digests == (binding.digest,)


def test_post_release_failure_is_quarantined_without_double_release() -> None:
    """A receipt failure retains identity while never reusing freed storage."""

    submission, _ = _submission()
    binding = submission.identity.local_binding
    wiring = _SourceWiring(submission, fail_after_release=True)
    consumer = PackedTerminalSourceSchedulerConsumer(
        wiring=wiring,
        process_fatal_handler=lambda inventory: None,
    )
    release_calls: list[int] = []
    consumer.register_release(
        PackedTerminalSourceSchedulerRelease(
            binding=binding,
            release_resources=lambda actual: release_calls.append(1),
        )
    )
    action = _action(binding)

    with pytest.raises(RuntimeError, match="post-release receipt failure"):
        consumer.consume_reclaim_authorized(action)

    inventory = consumer.inventory()
    assert release_calls == [1]
    assert inventory.active_binding_digests == (binding.digest,)
    assert inventory.quarantined_binding_digests == (binding.digest,)
    assert inventory.resource_release_completed_binding_digests == (
        binding.digest,
    )
    with pytest.raises(RuntimeError, match="quarantined source resources"):
        consumer.consume_reclaim_authorized(action)
    assert release_calls == [1]


def test_process_fatal_quarantines_every_retained_release() -> None:
    """Inbox failure preserves every unresolved scheduler-owned identity."""

    submission, _ = _submission()
    binding = submission.identity.local_binding
    fatal_inventories: list[SchedulerReceiptInboxInventory] = []
    consumer = PackedTerminalSourceSchedulerConsumer(
        wiring=_SourceWiring(submission),
        process_fatal_handler=fatal_inventories.append,
    )
    consumer.register_release(
        PackedTerminalSourceSchedulerRelease(
            binding=binding,
            release_resources=lambda actual: None,
        )
    )
    fatal = _fatal_inventory(binding)

    consumer.process_fatal(fatal)
    consumer.process_fatal(fatal)

    assert fatal_inventories == [fatal]
    inventory = consumer.inventory()
    assert inventory.active_binding_digests == (binding.digest,)
    assert inventory.quarantined_binding_digests == (binding.digest,)
    assert inventory.fatal_inventory == fatal
    with pytest.raises(RuntimeError, match="process-fatal"):
        consumer.register_release(
            PackedTerminalSourceSchedulerRelease(
                binding=binding,
                release_resources=lambda actual: None,
            )
        )
