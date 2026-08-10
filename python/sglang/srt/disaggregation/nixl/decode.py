import uuid
from collections.abc import Callable

import torch
from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.decode_allocation_lease import (
    DecodeAllocationLease,
    DecodeAllocationLeaseAuthority,
)
from sglang.srt.disaggregation.common.packed_staging_wire import PackedWireMessage
from sglang.srt.disaggregation.common.staging_layout import (
    StagingComponentGeometry,
    StagingWriterId,
)
from sglang.srt.disaggregation.nixl.packed_runtime import (
    PACKED_PREPARED_GRANT_PROTOCOL,
    PackedControlSender,
    PackedDecodeRuntime,
    PackedMetadataIndexAllocator,
    PackedRegistrationAdvertisement,
    PackedRuntimeManager,
    build_same_host_visibility_policy,
    decode_packed_control_frames,
    encode_packed_control_frames,
    load_exact_nixl_runtime_artifacts,
)
from sglang.srt.disaggregation.nixl.packed_staging import (
    PackedPeerIdentity,
    PackedStagingArena,
)
from sglang.srt.disaggregation.nixl.packed_staging_request import (
    PackedDecodeRequestTransaction,
    PackedRequestPublication,
)


class PackedNixlDecodeController:
    """Bind the packed decode actor to one process-lifetime NIXL manager."""

    _arena: PackedStagingArena
    _manager: PackedRuntimeManager
    _peer: PackedPeerIdentity
    _runtime: PackedDecodeRuntime

    def __init__(
        self,
        manager: PackedRuntimeManager,
        staging_tensor: torch.Tensor,
        staging_registration: object,
    ) -> None:
        """Initialize the persistent decode actor over legacy staging storage.

        The legacy staging pool remains the registration owner. The packed
        arena adopts that exact registration and therefore never deregisters
        it during its own lifecycle.

        :param manager: Owning TP1 or TP2 NIXL decode manager.
        :param staging_tensor: Existing process-lifetime staging byte tensor.
        :param staging_registration: Existing NIXL registration for the tensor.
        """

        if manager.attn_tp_size not in (1, 2):
            raise ValueError("packed decode controller supports only TP1 and TP2")
        if manager.attn_tp_rank < 0 or manager.attn_tp_rank >= manager.attn_tp_size:
            raise ValueError(
                "packed decode controller has an invalid attention TP rank"
            )
        if manager.attn_cp_rank != 0 or manager.pp_rank != 0:
            raise ValueError("packed decode controller requires CP1 and PP1")
        process_generation = uuid.UUID(manager.process_generation)
        artifacts = load_exact_nixl_runtime_artifacts()
        visibility_policy = build_same_host_visibility_policy(artifacts)
        peer = PackedPeerIdentity(
            agent_name=manager.agent.name,
            agent_generation=process_generation.bytes,
        )
        arena = PackedStagingArena(
            agent=manager.agent,
            tensor=staging_tensor,
            gpu_id=manager.kv_args.gpu_id,
            peer=peer,
            arena_generation=process_generation.bytes,
            registration=staging_registration,
        )
        runtime = PackedDecodeRuntime(
            manager,
            arena,
            artifacts,
            visibility_policy,
        )
        self._arena = arena
        self._manager = manager
        self._peer = peer
        self._runtime = runtime

    @property
    def advertisement(self) -> PackedRegistrationAdvertisement:
        """Return the persistent arena and native-runtime advertisement.

        :returns: Exact decode registration metadata.
        """

        return self._runtime.advertisement

    @property
    def ready(self) -> bool:
        """Return whether scheduler metadata ownership is attached.

        :returns: Whether packed request admission is safe.
        """

        return self._runtime.ready

    @property
    def prepared_grant_protocol(self) -> str | None:
        """Return the protocol only after the complete actor is live.

        :returns: Closed protocol identifier, otherwise ``None``.
        """

        if not self.ready:
            return None
        return PACKED_PREPARED_GRANT_PROTOCOL

    def attach_scheduler(
        self,
        metadata_allocator: PackedMetadataIndexAllocator,
        consumer_authority: object,
    ) -> None:
        """Attach the existing scheduler metadata allocator once.

        :param metadata_allocator: Existing decode metadata-row allocator.
        :param consumer_authority: Queue consuming metadata row contents.
        """

        self._runtime.attach_scheduler(metadata_allocator, consumer_authority)

    def prepare_transaction(
        self,
        *,
        room_id: int,
        request_owner: object,
        metadata_buffer_index: int,
        allocation_lease: DecodeAllocationLease,
        allocation_authority: DecodeAllocationLeaseAuthority,
        lifecycle_authority: object,
        source_tp_size: int,
        source_registration: tuple[StagingComponentGeometry, ...],
    ) -> PackedDecodeRequestTransaction:
        """Construct one actor-owned request transaction.

        :param room_id: Decoder-minted room.
        :param request_owner: Exact retained decode request.
        :param metadata_buffer_index: Already reserved metadata row.
        :param allocation_lease: Exact pinned decode allocation.
        :param allocation_authority: Exact allocation authority.
        :param lifecycle_authority: Trusted transport lifecycle authority.
        :param source_tp_size: Supported packed source writer width.
        :param source_registration: Bootstrap-pinned source component geometry.
        :returns: Prepared request transaction.
        """

        return self._runtime.prepare_transaction(
            room_id=room_id,
            request_owner=request_owner,
            metadata_buffer_index=metadata_buffer_index,
            allocation_lease=allocation_lease,
            allocation_authority=allocation_authority,
            lifecycle_authority=lifecycle_authority,
            source_tp_size=source_tp_size,
            source_registration=source_registration,
        )

    def bind_publication(
        self,
        transaction: PackedDecodeRequestTransaction,
        publication: PackedRequestPublication,
        routes: tuple[PackedControlSender, ...],
    ) -> None:
        """Bind a publication to its authenticated source-writer routes.

        :param transaction: Exact published transaction.
        :param publication: Matching irreversible publication.
        :param routes: Complete canonical source-writer route set.
        """

        self._runtime.bind_publication(transaction, publication, routes)

    def build_control_sender(
        self,
        writer_id: StagingWriterId,
        send_frames: Callable[[list[bytes]], None],
    ) -> PackedControlSender:
        """Build one generation-authenticated decoder-to-prefill route.

        :param writer_id: Exact source writer reached by the route.
        :param send_frames: Serialized multipart socket send operation.
        :returns: Actor control sender for that writer.
        """

        def send_message(message: PackedWireMessage) -> None:
            frames = encode_packed_control_frames(
                self._peer.agent_name,
                self._manager.process_generation,
                message,
            )
            send_frames(frames)

        return PackedControlSender(writer_id=writer_id, send_message=send_message)

    def handle_control_frames(
        self,
        frames: list[bytes],
        authenticated_peer: PackedPeerIdentity,
        authenticated_writer_id: StagingWriterId,
    ) -> None:
        """Authenticate and dispatch one source-to-decode control message.

        :param frames: Valid bounded PACKED_V4 multipart message.
        :param authenticated_peer: Generation-bound native source peer.
        :param authenticated_writer_id: Writer identity bound to that peer.
        """

        agent_name, process_generation, message = decode_packed_control_frames(frames)
        if agent_name != authenticated_peer.agent_name:
            raise RuntimeError("packed control agent name differs from native peer")
        if uuid.UUID(process_generation).bytes != authenticated_peer.agent_generation:
            raise RuntimeError("packed control generation differs from native peer")
        self._runtime.handle_control(authenticated_writer_id, message)

    def poll(self, transaction: PackedDecodeRequestTransaction) -> KVPoll:
        """Advance scatter, teardown, and scheduler commit work.

        :param transaction: Exact actor-owned transaction.
        :returns: Current transfer state.
        """

        return self._runtime.poll(transaction)

    def cancel_unpublished(
        self,
        transaction: PackedDecodeRequestTransaction,
    ) -> object:
        """Cancel and retire one unpublished actor transaction.

        :param transaction: Exact prepared transaction.
        :returns: Exact retained request owner.
        """

        return self._runtime.cancel_unpublished(transaction)

    def complete_metadata_consumption(
        self,
        transaction: PackedDecodeRequestTransaction,
    ) -> None:
        """Release consumed metadata and retire actor request state.

        :param transaction: Exact committed transaction.
        """

        self._runtime.complete_metadata_consumption(transaction)

    def quarantine(
        self,
        transaction: PackedDecodeRequestTransaction,
        reason: str,
    ) -> None:
        """Quarantine every resource retained by one request.

        :param transaction: Exact actor-owned transaction.
        :param reason: Stable failure reason.
        """

        self._runtime.quarantine(transaction, reason)
