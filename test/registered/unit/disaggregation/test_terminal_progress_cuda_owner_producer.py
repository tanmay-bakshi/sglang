import gc
import time

import pytest

from sglang.srt.disaggregation.common.packed_staging_protocol import PackedRequestKey
from sglang.srt.disaggregation.terminal_progress.cuda_owner_producer import (
    CudaTerminalEventKind,
    CudaTerminalProducer,
    cuda_terminal_producer_abi,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalPublicationIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.native_owner import (
    NativeTerminalOwner,
    native_terminal_owner_producer_abi,
)
from sglang.srt.disaggregation.terminal_progress.native_state import (
    NativeDecodeLifecyclePhase,
    NativeSourceLifecyclePhase,
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
    NativeTerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.runtime import (
    NativeTerminalNativeProducerBinding,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

_LOCAL_PRODUCER_ID = 1
_CUDA_PRODUCER_ID = 2
_CONTROL_PRODUCER_ID = 3
_WAIT_SECONDS = 2.0


def test_independent_owner_and_cuda_dsos_compile_the_exact_same_abi() -> None:
    """Structure sizes and offsets match across independent native builds."""

    owner_abi = native_terminal_owner_producer_abi(testing=True)
    cuda_abi = cuda_terminal_producer_abi(testing=True)

    assert owner_abi == cuda_abi
    assert owner_abi == {
        "abi_version": 1,
        "api_struct_size": 40,
        "event_struct_size": 168,
        "required_flags": 3,
        "event_offsets": {
            "abi_version": 0,
            "struct_size": 4,
            "binding_digest": 8,
            "event_kind": 40,
            "enqueued_ns": 48,
            "receipt_binding_digest": 80,
            "receipt_nonce": 152,
        },
    }


def _make_owner() -> tuple[
    NativeTerminalOwner,
    NativeTerminalProcessIdentity,
    NativeTerminalProcessIdentity,
]:
    """Create one decode owner with local and CUDA producer namespaces.

    :returns: Native owner plus its local and authenticated source identities.
    """

    process = TerminalProcessIdentity(
        process_generation=bytes.fromhex("102132435465768798a9bacbdcedfe0f"),
        role=TerminalOwnerRole.DECODE,
        tp_rank=0,
        tp_size=1,
    )
    identity = NativeTerminalProcessIdentity.from_identity(process)
    source_identity = NativeTerminalProcessIdentity.from_identity(
        TerminalProcessIdentity(
            process_generation=bytes.fromhex("ffeeddccbbaa99887766554433221100"),
            role=TerminalOwnerRole.SOURCE,
            tp_rank=0,
            tp_size=1,
        )
    )
    owner = NativeTerminalOwner(
        input_capacity=128,
        output_capacity=128,
        observation_capacity=128,
        owner_identity=identity,
        testing=True,
    )
    for registration in (
        NativeTerminalProducerRegistration(
            producer_id=_LOCAL_PRODUCER_ID,
            name="decode-local",
            producer_class=NativeTerminalProducerClass.LOCAL,
            allowed_role=NativeTerminalOwnerRole.DECODE,
            authenticated_issuer=None,
        ),
        NativeTerminalProducerRegistration(
            producer_id=_CUDA_PRODUCER_ID,
            name="decode-cuda-scatter",
            producer_class=NativeTerminalProducerClass.LOCAL,
            allowed_role=NativeTerminalOwnerRole.DECODE,
            authenticated_issuer=None,
        ),
        NativeTerminalProducerRegistration(
            producer_id=_CONTROL_PRODUCER_ID,
            name="source-control",
            producer_class=NativeTerminalProducerClass.CONTROL,
            allowed_role=NativeTerminalOwnerRole.DECODE,
            authenticated_issuer=source_identity,
        ),
    ):
        owner.register_producer(registration)
    return owner, identity, source_identity


def _register_scatter_inflight(
    owner: NativeTerminalOwner,
    identity: NativeTerminalProcessIdentity,
    source_identity: NativeTerminalProcessIdentity,
    room_id: int,
) -> NativeTerminalRequestBinding:
    """Register and advance one decode lifecycle to scatter in flight.

    :param owner: Process-lifetime native owner.
    :param identity: Exact decode owner identity.
    :param source_identity: Authenticated source control identity.
    :param room_id: Stable test request room.
    :returns: Exact native request binding.
    """

    process = TerminalProcessIdentity(
        process_generation=identity.process_generation,
        role=TerminalOwnerRole.DECODE,
        tp_rank=identity.tp_rank,
        tp_size=identity.tp_size,
    )
    request_key = PackedRequestKey(
        room_id=room_id,
        request_generation=room_id.to_bytes(16, "big"),
    )
    binding = NativeTerminalRequestBinding.from_binding(
        TerminalRequestBinding(
            request_key=request_key,
            owner=process,
            rank_manifest_digest=b"r" * 32,
            allocation_digest=room_id.to_bytes(32, "big"),
        )
    )
    owner.register_lifecycle(
        NativeTerminalLifecycleRegistration(
            binding=binding,
            publication_identity=None,
            trusted_issuers=(identity, source_identity),
        )
    )
    for kind in (
        NativeTerminalOwnerEventKind.DECODE_ALLOCATION_PUBLISHED,
        NativeTerminalOwnerEventKind.DECODE_WRITER_AGGREGATION_STARTED,
        NativeTerminalOwnerEventKind.DECODE_WRITER_MANIFEST_COMPLETED,
        NativeTerminalOwnerEventKind.DECODE_SCATTER_STARTED,
    ):
        producer_id = _LOCAL_PRODUCER_ID
        if kind in (
            NativeTerminalOwnerEventKind.DECODE_WRITER_AGGREGATION_STARTED,
            NativeTerminalOwnerEventKind.DECODE_WRITER_MANIFEST_COMPLETED,
        ):
            producer_id = _CONTROL_PRODUCER_ID
        owner.submit(
            NativeTerminalOwnerEvent(
                producer_id=producer_id,
                binding_digest=binding.digest,
                kind=kind,
                enqueued_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
            )
        )
    return binding


def _cuda_binding(owner: NativeTerminalOwner) -> NativeTerminalNativeProducerBinding:
    """Build the runtime-shaped binding for the registered CUDA producer.

    :param owner: Native owner containing the producer registration.
    :returns: Opaque producer API and context binding.
    """

    return NativeTerminalNativeProducerBinding(
        producer_id=_CUDA_PRODUCER_ID,
        producer_api=owner.producer_api(),
        producer_context=owner.producer_capsule(_CUDA_PRODUCER_ID),
    )


def _make_waiting_source_owner() -> tuple[
    NativeTerminalOwner,
    NativeTerminalRequestBinding,
]:
    """Create one source lifecycle waiting on direct CUDA completion.

    :returns: Native source owner and its registered request binding.
    """

    process = TerminalProcessIdentity(
        process_generation=bytes.fromhex("1234567890abcdef1234567890abcdef"),
        role=TerminalOwnerRole.SOURCE,
        tp_rank=0,
        tp_size=1,
    )
    identity = NativeTerminalProcessIdentity.from_identity(process)
    owner = NativeTerminalOwner(
        input_capacity=128,
        output_capacity=128,
        observation_capacity=128,
        owner_identity=identity,
        testing=True,
    )
    for registration in (
        NativeTerminalProducerRegistration(
            producer_id=_LOCAL_PRODUCER_ID,
            name="source-local",
            producer_class=NativeTerminalProducerClass.LOCAL,
            allowed_role=NativeTerminalOwnerRole.SOURCE,
            authenticated_issuer=None,
        ),
        NativeTerminalProducerRegistration(
            producer_id=_CUDA_PRODUCER_ID,
            name="source-cuda-completion",
            producer_class=NativeTerminalProducerClass.LOCAL,
            allowed_role=NativeTerminalOwnerRole.SOURCE,
            authenticated_issuer=None,
        ),
        NativeTerminalProducerRegistration(
            producer_id=_CONTROL_PRODUCER_ID,
            name="source-control",
            producer_class=NativeTerminalProducerClass.CONTROL,
            allowed_role=NativeTerminalOwnerRole.SOURCE,
            authenticated_issuer=identity,
        ),
    ):
        owner.register_producer(registration)
    request_key = PackedRequestKey(
        room_id=301,
        request_generation=bytes.fromhex("00112233445566778899aabbccddeeff"),
    )
    binding = NativeTerminalRequestBinding.from_binding(
        TerminalRequestBinding(
            request_key=request_key,
            owner=process,
            rank_manifest_digest=b"r" * 32,
            allocation_digest=b"a" * 32,
        )
    )
    owner.register_lifecycle(
        NativeTerminalLifecycleRegistration(
            binding=binding,
            publication_identity=NativeTerminalPublicationIdentity.from_identity(
                TerminalPublicationIdentity(
                    request_key=request_key,
                    publisher_process_generation=process.process_generation,
                    publication_generation=b"p" * 16,
                )
            ),
            trusted_issuers=(identity,),
        )
    )
    owner.submit(
        NativeTerminalOwnerEvent(
            producer_id=_LOCAL_PRODUCER_ID,
            binding_digest=binding.digest,
            kind=NativeTerminalOwnerEventKind.SOURCE_SUBMISSION_ACCEPTED,
            enqueued_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
        )
    )
    return owner, binding


def _wait_for_source_phase(
    owner: NativeTerminalOwner,
    binding_digest: bytes,
    phase: NativeSourceLifecyclePhase,
) -> None:
    """Wait for one direct source lifecycle transition.

    :param owner: Process-lifetime native source owner.
    :param binding_digest: Exact lifecycle identity.
    :param phase: Required source lifecycle phase.
    """

    deadline = time.monotonic() + _WAIT_SECONDS
    snapshot = None
    while time.monotonic() < deadline:
        snapshot = owner.lifecycle_snapshot_for_testing(binding_digest)
        if snapshot.phase == int(phase):
            return
        time.sleep(0.001)
    raise TimeoutError(
        "native source lifecycle did not reach "
        f"{phase.name}: snapshot={snapshot}, inventory={owner.inventory()}"
    )


def _wait_for_phase(
    owner: NativeTerminalOwner,
    binding_digest: bytes,
    phase: NativeDecodeLifecyclePhase,
) -> None:
    """Wait for one actionless native state commit.

    :param owner: Process-lifetime native owner.
    :param binding_digest: Exact lifecycle binding digest.
    :param phase: Required decode lifecycle phase.
    """

    deadline = time.monotonic() + _WAIT_SECONDS
    snapshot = None
    while time.monotonic() < deadline:
        try:
            snapshot = owner.lifecycle_snapshot_for_testing(binding_digest)
        except KeyError:
            time.sleep(0.001)
            continue
        if snapshot.phase == int(phase):
            return
        time.sleep(0.001)
    inventory = owner.inventory()
    raise TimeoutError(
        "native decode lifecycle did not reach "
        f"{phase.name}: snapshot={snapshot}, inventory={inventory}"
    )


def _retire_and_abort(
    owner: NativeTerminalOwner,
    producer: CudaTerminalProducer,
) -> None:
    """Retire every producer namespace and release the test lifecycle.

    :param owner: Process-lifetime native owner.
    :param producer: Direct CUDA producer bound to the owner.
    """

    owner.retire_python_producer(_LOCAL_PRODUCER_ID)
    owner.retire_python_producer(_CONTROL_PRODUCER_ID)
    assert producer.join(_WAIT_SECONDS)
    producer.close()
    assert owner.wait_for_producer_retirement(_LOCAL_PRODUCER_ID, _WAIT_SECONDS)
    assert owner.join_producers()
    owner.abort_and_close()


def test_direct_callback_reaches_native_owner_without_python_drain() -> None:
    """A CUDA terminal callback mutates native state without an intermediate queue."""

    owner, identity, source_identity = _make_owner()
    binding = _register_scatter_inflight(
        owner,
        identity,
        source_identity,
        room_id=401,
    )
    producer = CudaTerminalProducer(
        _cuda_binding(owner),
        CudaTerminalEventKind.DECODE_SCATTER_TERMINAL,
        testing=True,
    )
    producer.arm(binding.digest)
    owner.start()
    producer.complete_synchronously_for_testing(binding.digest)

    _wait_for_phase(owner, binding.digest, NativeDecodeLifecyclePhase.SCATTER_TERMINAL)
    inventory = producer.inventory()
    assert inventory.total_submissions == 1
    assert inventory.total_delivered == 1
    assert inventory.retained_count == 0
    assert inventory.fatal_code == "none"
    assert owner.inventory().fatal_code is NativeTerminalOwnerFatalCode.NONE
    _retire_and_abort(owner, producer)


def test_source_callback_reaches_producer_completed_without_python_drain() -> None:
    """A source CUDA callback directly earns native gather authority."""

    owner, binding = _make_waiting_source_owner()
    producer = CudaTerminalProducer(
        _cuda_binding(owner),
        CudaTerminalEventKind.SOURCE_PRODUCER_COMPLETED,
        testing=True,
    )
    producer.arm(binding.digest)
    owner.start()
    _wait_for_source_phase(
        owner,
        binding.digest,
        NativeSourceLifecyclePhase.WAITING_FOR_PRODUCER,
    )

    producer.complete_synchronously_for_testing(binding.digest)

    _wait_for_source_phase(
        owner,
        binding.digest,
        NativeSourceLifecyclePhase.GATHERING,
    )
    outputs = owner.drain_outputs()
    assert len(outputs) == 1
    assert tuple(action.kind for action in outputs[0].actions) == (
        NativeTerminalOwnerActionKind.SOURCE_GATHER_READY,
    )
    inventory = producer.inventory()
    assert inventory.total_submissions == 1
    assert inventory.total_delivered == 1
    assert inventory.retained_count == 0
    _retire_and_abort(owner, producer)


def test_concurrent_callbacks_preserve_owner_assigned_queue_order() -> None:
    """Concurrent native callbacks cannot forge or reorder producer sequences."""

    owner, identity, source_identity = _make_owner()
    bindings = tuple(
        _register_scatter_inflight(
            owner,
            identity,
            source_identity,
            room_id=500 + index,
        )
        for index in range(16)
    )
    producer = CudaTerminalProducer(
        _cuda_binding(owner),
        CudaTerminalEventKind.DECODE_SCATTER_TERMINAL,
        testing=True,
    )
    for binding in bindings:
        producer.arm(binding.digest)
    owner.start()
    producer.complete_concurrently_for_testing(
        tuple(binding.digest for binding in bindings)
    )

    for binding in bindings:
        _wait_for_phase(
            owner,
            binding.digest,
            NativeDecodeLifecyclePhase.SCATTER_TERMINAL,
        )
    inventory = producer.inventory()
    assert inventory.total_delivered == len(bindings)
    assert inventory.retained_count == 0
    assert inventory.fatal_code == "none"
    assert owner.inventory().fatal_code is NativeTerminalOwnerFatalCode.NONE
    _retire_and_abort(owner, producer)


def test_close_with_live_callback_fails_closed_and_keeps_capsule_lifetime() -> None:
    """A live callback prevents producer and owner context destruction."""

    owner, identity, source_identity = _make_owner()
    binding = _register_scatter_inflight(
        owner,
        identity,
        source_identity,
        room_id=601,
    )
    producer = CudaTerminalProducer(
        _cuda_binding(owner),
        CudaTerminalEventKind.DECODE_SCATTER_TERMINAL,
        testing=True,
    )
    producer.arm(binding.digest)
    owner.start()
    producer.begin_held_callback_for_testing(binding.digest)

    assert not producer.join(0.01)
    with pytest.raises(RuntimeError, match="close_with_active_callbacks"):
        producer.close()
    producer.complete_held_callback_for_testing(binding.digest)
    _wait_for_phase(owner, binding.digest, NativeDecodeLifecyclePhase.SCATTER_TERMINAL)
    inventory = producer.inventory()
    assert inventory.fatal_code == "close_with_active_callbacks"
    assert inventory.total_delivered == 1
    assert inventory.retained_count == 0
    owner.abort_and_close()
    del producer
    gc.collect()
