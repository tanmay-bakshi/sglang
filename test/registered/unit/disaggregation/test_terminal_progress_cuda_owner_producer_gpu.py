import time

import pytest
import torch

from sglang.srt.disaggregation.common.packed_staging_protocol import PackedRequestKey
from sglang.srt.disaggregation.terminal_progress.cuda_owner_producer import (
    CudaTerminalEventKind,
    CudaTerminalProducer,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.native_owner import (
    NativeTerminalOwner,
)
from sglang.srt.disaggregation.terminal_progress.native_state import (
    NativeDecodeLifecyclePhase,
    NativeTerminalLifecycleRegistration,
    NativeTerminalOwnerEvent,
    NativeTerminalOwnerEventKind,
    NativeTerminalOwnerRole,
    NativeTerminalProcessIdentity,
    NativeTerminalProducerClass,
    NativeTerminalProducerRegistration,
    NativeTerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.runtime import (
    NativeTerminalNativeProducerBinding,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA terminal producer requires one CUDA device",
)

_LOCAL_PRODUCER_ID = 1
_CUDA_PRODUCER_ID = 2
_CONTROL_PRODUCER_ID = 3
_WAIT_SECONDS = 2.0


def _build_scatter_owner() -> tuple[
    NativeTerminalOwner,
    CudaTerminalProducer,
    NativeTerminalRequestBinding,
]:
    """Build one started decode owner at scatter-in-flight.

    :returns: Native owner, direct CUDA producer, and request binding.
    """

    process = TerminalProcessIdentity(
        process_generation=bytes.fromhex("ffeeddccbbaa99887766554433221100"),
        role=TerminalOwnerRole.DECODE,
        tp_rank=0,
        tp_size=1,
    )
    identity = NativeTerminalProcessIdentity.from_identity(process)
    source_identity = NativeTerminalProcessIdentity.from_identity(
        TerminalProcessIdentity(
            process_generation=bytes.fromhex("102132435465768798a9bacbdcedfe0f"),
            role=TerminalOwnerRole.SOURCE,
            tp_rank=0,
            tp_size=1,
        )
    )
    owner = NativeTerminalOwner(
        input_capacity=64,
        output_capacity=64,
        observation_capacity=64,
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
    request_key = PackedRequestKey(
        room_id=701,
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
    producer = CudaTerminalProducer(
        NativeTerminalNativeProducerBinding(
            producer_id=_CUDA_PRODUCER_ID,
            producer_api=owner.producer_api(),
            producer_context=owner.producer_capsule(_CUDA_PRODUCER_ID),
        ),
        CudaTerminalEventKind.DECODE_SCATTER_TERMINAL,
        testing=True,
    )
    producer.arm(binding.digest)
    owner.start()
    return owner, producer, binding


def test_real_cuda_callback_delivers_directly_into_native_owner() -> None:
    """A real stream callback reaches scatter-terminal without a Python drain."""

    owner, producer, binding = _build_scatter_owner()
    stream = torch.cuda.Stream()
    producer.submit(stream.cuda_stream, binding.digest)
    stream.synchronize()

    deadline = time.monotonic() + _WAIT_SECONDS
    while time.monotonic() < deadline:
        snapshot = owner.lifecycle_snapshot_for_testing(binding.digest)
        if snapshot.phase == int(NativeDecodeLifecyclePhase.SCATTER_TERMINAL):
            break
        time.sleep(0.001)
    else:
        raise TimeoutError("real CUDA callback did not reach scatter-terminal")

    inventory = producer.inventory()
    assert inventory.total_submissions == 1
    assert inventory.total_delivered == 1
    assert inventory.retained_count == 0
    owner.retire_python_producer(_LOCAL_PRODUCER_ID)
    owner.retire_python_producer(_CONTROL_PRODUCER_ID)
    assert producer.join(_WAIT_SECONDS)
    producer.close()
    assert owner.wait_for_producer_retirement(_LOCAL_PRODUCER_ID, _WAIT_SECONDS)
    assert owner.join_producers()
    owner.abort_and_close()
