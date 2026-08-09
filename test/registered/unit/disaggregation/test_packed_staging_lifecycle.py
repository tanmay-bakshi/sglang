import concurrent.futures
import dataclasses
import sys
import threading

import numpy as np
import pytest

from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PackedChunkKey,
)
from sglang.srt.disaggregation.common.staging_layout import (
    StagingComponentId,
    StagingWriterId,
)
from sglang.srt.disaggregation.nixl.packed_staging_lifecycle import (
    PACKED_AUTHENTICATOR_BYTES,
    PackedAmbiguousTransportError,
    PackedArenaCapability,
    PackedArenaCapabilityAuthority,
    PackedArenaGrant,
    PackedAuthenticatedPeer,
    PackedAuthenticatedPeerAuthority,
    PackedCapabilityState,
    PackedCohortResources,
    PackedCompletedScatter,
    PackedCompletedTransfer,
    PackedExpectedNixlUcxRoute,
    PackedLifecycleError,
    PackedNativeAttestationUnavailable,
    PackedNativeCompletionContract,
    PackedNativePollState,
    PackedNativePostState,
    PackedNativeTransferState,
    PackedNixlBackend,
    PackedNixlUcxNativeDriver,
    PackedNixlUcxTransferLifecycle,
    PackedPageLease,
    PackedPageLeaseAllocator,
    PackedPageLeaseRole,
    PackedProcessLifetimeQuarantine,
    PackedProcessQuarantinedError,
    PackedResourceCohort,
    PackedScatterEventDriver,
    PackedScatterLifecycle,
    PackedScatterPollState,
    PackedTeardownAck,
    PackedTeardownCoordinator,
    PackedTeardownState,
    PackedUcpTransport,
    PackedWriterCohortCompletion,
    PackedWriterCohortManifest,
    PackedWriterProjection,
    require_nixl_ucx_native_attestation,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ALIGNMENT_BYTES = 256
REQUEST_GENERATION = bytes.fromhex("00112233445566778899aabbccddeeff")
ARENA_GENERATION = bytes.fromhex("102132435465768798a9bacbdcedfe0f")
ROUTE_DIGEST = b"r" * 32
RUNTIME_IDENTITY = b"n" * 32
MAIN_KV = StagingComponentId(state_index=None, state_type=None)


class OpaqueOwner:
    """Weakly descriptive exact-identity test owner."""


class RecordingNativeDriver(PackedNixlUcxNativeDriver):
    """CPU-only exact-handle driver with deterministic fault injection."""

    post_results: dict[object, PackedNativePostState | BaseException]
    poll_results: dict[object, PackedNativePollState | BaseException]
    release_failures: set[object]
    released: list[object]
    transports: frozenset[PackedUcpTransport] | None

    def __init__(self) -> None:
        """Initialize an exact attested successful route."""

        self.post_results = {}
        self.poll_results = {}
        self.release_failures = set()
        self.released = []
        self.transports = frozenset(
            (
                PackedUcpTransport.CUDA_IPC,
                PackedUcpTransport.CUDA,
            )
        )

    def post(self, handle: object) -> PackedNativePostState:
        """Post or inject one exact-handle failure.

        :param handle: Exact prepared handle.
        :returns: Native post state.
        """

        result = self.post_results.get(
            handle,
            PackedNativePostState.SUBMITTED,
        )
        if isinstance(result, BaseException):
            raise result
        return result

    def poll(self, handle: object) -> PackedNativePollState:
        """Poll or inject one exact-handle failure.

        :param handle: Exact submitted handle.
        :returns: Native poll state.
        """

        result = self.poll_results.get(
            handle,
            PackedNativePollState.DONE,
        )
        if isinstance(result, BaseException):
            raise result
        return result

    def query_backend(self, handle: object) -> PackedNixlBackend:
        """Return the exact UCX backend.

        :param handle: Exact transfer handle.
        :returns: UCX backend identity.
        """

        return PackedNixlBackend.UCX

    def query_ucp_transports(
        self,
        handle: object,
    ) -> frozenset[PackedUcpTransport] | None:
        """Return configured per-handle transport attestation.

        :param handle: Exact transfer handle.
        :returns: Configured transport set.
        """

        return self.transports

    def query_runtime_identity(self, handle: object) -> bytes:
        """Return the expected runtime artifact identity.

        :param handle: Exact transfer handle.
        :returns: Runtime digest.
        """

        return RUNTIME_IDENTITY

    def query_completion_contract(
        self,
        handle: object,
    ) -> PackedNativeCompletionContract | None:
        """Return the exact endpoint-flush completion contract.

        :param handle: Exact completed transfer handle.
        :returns: Endpoint-flush contract.
        """

        return PackedNativeCompletionContract.NIXL_UCX_ENDPOINT_FLUSH

    def release_completed(self, handle: object) -> None:
        """Release or inject failure for one exact completed handle.

        :param handle: Exact completed transfer handle.
        """

        if handle in self.release_failures:
            raise RuntimeError("injected native release failure")
        self.released.append(handle)


class RecordingScatterDriver(PackedScatterEventDriver):
    """CPU-only CUDA event driver with deterministic fault injection."""

    results: dict[object, PackedScatterPollState | BaseException]
    submit_failure: BaseException | None
    submitted_events: list[object]

    def __init__(self) -> None:
        """Initialize terminal event progress."""

        self.results = {}
        self.submit_failure = None
        self.submitted_events = []

    def submit(
        self,
        destination_lease: object,
        resources: tuple[object, ...],
    ) -> object:
        """Create an exact fake event or inject submission failure.

        :param destination_lease: Exact destination staging lease.
        :param resources: Exact retained kernel owners.
        :returns: Exact recorded fake event.
        """

        if destination_lease is None or len(resources) == 0:
            raise ValueError("invalid fake scatter submission")
        if self.submit_failure is not None:
            raise self.submit_failure
        event = OpaqueOwner()
        self.submitted_events.append(event)
        return event

    def poll(self, event: object) -> PackedScatterPollState:
        """Poll or inject one exact-event failure.

        :param event: Exact scatter event.
        :returns: Native scatter state.
        """

        result = self.results.get(event, PackedScatterPollState.DONE)
        if isinstance(result, BaseException):
            raise result
        return result


class CoordinatedPostDriver(RecordingNativeDriver):
    """Driver holding one successful post until a peer post fails."""

    blocked_handle: object | None
    release_success: threading.Event
    success_entered: threading.Event

    def __init__(self) -> None:
        """Initialize an unbound successful post gate."""

        super().__init__()
        self.blocked_handle = None
        self.release_success = threading.Event()
        self.success_entered = threading.Event()

    def post(self, handle: object) -> PackedNativePostState:
        """Block the selected handle and fail every peer handle.

        :param handle: Exact prepared handle.
        :returns: Submitted for the selected handle, otherwise error.
        """

        if handle is not self.blocked_handle:
            return PackedNativePostState.ERROR
        self.success_entered.set()
        if not self.release_success.wait(timeout=5):
            raise RuntimeError("coordinated post test timed out")
        return PackedNativePostState.SUBMITTED


class StockNixlAgent:
    """Representative stock surface lacking per-handle lane proof."""

    def query_xfer_backend(self, handle: object) -> str:
        """Return only the backend exposed by stock NIXL.

        :param handle: Exact transfer handle.
        :returns: Backend name.
        """

        return "UCX"


class LifecycleHarness:
    """Complete CPU-only request cohort for lifecycle fault injection."""

    arena_owner: object
    capability: PackedArenaCapability
    capability_authority: PackedArenaCapabilityAuthority
    cohort: PackedResourceCohort
    completed: tuple[PackedCompletedTransfer, ...]
    destination_allocator: PackedPageLeaseAllocator
    destination_lease: PackedPageLease
    driver: RecordingNativeDriver
    endpoints: tuple[object, ...]
    grant: PackedArenaGrant
    handles: tuple[object, ...]
    key: PackedChunkKey
    lifecycle: PackedNixlUcxTransferLifecycle
    manifest: PackedWriterCohortManifest
    page_lease_bindings: tuple[
        tuple[PackedPageLeaseAllocator, PackedPageLease],
        ...,
    ]
    peer_authority: PackedAuthenticatedPeerAuthority
    peers: tuple[PackedAuthenticatedPeer, ...]
    prepared: tuple[object, ...]
    quarantine: PackedProcessLifetimeQuarantine
    registration: object
    scatter: PackedScatterLifecycle
    scatter_driver: RecordingScatterDriver
    source_allocator: PackedPageLeaseAllocator
    source_lease: PackedPageLease
    staging_lease: object
    writers: tuple[StagingWriterId, ...]

    def __init__(
        self,
        source_tp_size: int,
        driver: RecordingNativeDriver | None = None,
    ) -> None:
        """Build one complete request cohort without posting.

        :param source_tp_size: Number of authenticated source writers.
        :param driver: Optional exact native test driver.
        """

        self.writers = tuple(
            StagingWriterId(
                transfer_source_rank=rank,
                source_attn_tp_rank=rank,
                source_pp_rank=0,
                source_cp_rank=0,
            )
            for rank in range(source_tp_size)
        )
        peer_generations = tuple(
            rank.to_bytes(16, "big") for rank in range(source_tp_size)
        )
        self.endpoints = tuple(OpaqueOwner() for _ in self.writers)
        self.peer_authority = PackedAuthenticatedPeerAuthority()
        self.peers = tuple(
            self.peer_authority.bind_native_endpoint(
                endpoint,
                writer,
                generation,
            )
            for endpoint, writer, generation in zip(
                self.endpoints,
                self.writers,
                peer_generations,
                strict=True,
            )
        )
        projections = tuple(
            PackedWriterProjection(
                writer_id=writer,
                peer_generation=generation,
                destination_offset=rank * ALIGNMENT_BYTES,
                length_bytes=ALIGNMENT_BYTES,
            )
            for rank, (writer, generation) in enumerate(
                zip(self.writers, peer_generations, strict=True)
            )
        )
        self.manifest = PackedWriterCohortManifest(
            request_generation=REQUEST_GENERATION,
            source_tp_size=source_tp_size,
            destination_tp_size=1,
            destination_tp_rank=0,
            total_bytes=source_tp_size * ALIGNMENT_BYTES,
            alignment_bytes=ALIGNMENT_BYTES,
            projections=projections,
        )
        self.registration = OpaqueOwner()
        self.arena_owner = OpaqueOwner()
        self.capability_authority = PackedArenaCapabilityAuthority(
            peer_authority=self.peer_authority,
            arena_generation=ARENA_GENERATION,
            registration=self.registration,
            arena_owner=self.arena_owner,
            secret=b"s" * PACKED_AUTHENTICATOR_BYTES,
        )
        self.capability, self.grant = self.capability_authority.issue(
            peers=self.peers,
            manifest=self.manifest,
            route_digest=ROUTE_DIGEST,
            base_address=0x100000,
            total_size=self.manifest.total_bytes,
            destination_gpu_id=3,
            alignment_bytes=ALIGNMENT_BYTES,
        )
        self.source_allocator = PackedPageLeaseAllocator(PackedPageLeaseRole.SOURCE)
        self.destination_allocator = PackedPageLeaseAllocator(
            PackedPageLeaseRole.DESTINATION
        )
        self.source_lease = self.source_allocator.claim(
            owner=OpaqueOwner(),
            pages={MAIN_KV: np.array([1, 2], dtype=np.int32)},
        )
        self.destination_lease = self.destination_allocator.claim(
            owner=OpaqueOwner(),
            pages={MAIN_KV: np.array([7, 8], dtype=np.int32)},
        )
        self.page_lease_bindings = (
            (self.source_allocator, self.source_lease),
            (self.destination_allocator, self.destination_lease),
        )
        self.staging_lease = OpaqueOwner()
        resources = PackedCohortResources(
            capability=self.capability,
            source_page_leases=(self.page_lease_bindings[0],),
            destination_page_leases=(self.page_lease_bindings[1],),
            staging_leases=(self.staging_lease,),
            tensors=(OpaqueOwner(),),
            registrations=(self.registration,),
            endpoints=self.endpoints,
            arenas=(self.arena_owner,),
        )
        self.cohort = self.capability_authority.create_resource_cohort(resources)
        self.quarantine = PackedProcessLifetimeQuarantine()
        self.driver = RecordingNativeDriver() if driver is None else driver
        route = PackedExpectedNixlUcxRoute(
            route_digest=ROUTE_DIGEST,
            runtime_identity=RUNTIME_IDENTITY,
            transports=frozenset(
                (
                    PackedUcpTransport.CUDA_IPC,
                    PackedUcpTransport.CUDA,
                )
            ),
        )
        self.lifecycle = PackedNixlUcxTransferLifecycle(
            driver=self.driver,
            capability_authority=self.capability_authority,
            route=route,
            quarantine=self.quarantine,
        )
        self.handles = tuple(OpaqueOwner() for _ in self.writers)
        self.prepared = tuple(
            self.lifecycle.register_prepared(
                writer_id=writer,
                peer=peer,
                capability=self.capability,
                handle=handle,
                cohort=self.cohort,
            )
            for writer, peer, handle in zip(
                self.writers,
                self.peers,
                self.handles,
                strict=True,
            )
        )
        self.completed = ()
        self.scatter_driver = RecordingScatterDriver()
        self.scatter = PackedScatterLifecycle(
            driver=self.scatter_driver,
            quarantine=self.quarantine,
        )
        self.key = PackedChunkKey(
            room_id=41,
            chunk_id=2,
            request_generation=REQUEST_GENERATION,
        )

    def complete_native(self) -> tuple[PackedCompletedTransfer, ...]:
        """Complete every exact native writer transfer.

        :returns: Completions in canonical writer order.
        """

        completed: list[PackedCompletedTransfer] = []
        for prepared in self.prepared:
            submit_receipt = self.lifecycle.post(prepared)
            inflight = self.lifecycle.accept_submission(
                prepared,
                submit_receipt,
            )
            completion_receipt = self.lifecycle.poll(inflight)
            if completion_receipt is None:
                raise RuntimeError("test driver unexpectedly returned pending")
            completed.append(
                self.lifecycle.accept_completion(
                    inflight,
                    completion_receipt,
                )
            )
        self.completed = tuple(completed)
        return self.completed

    def complete_scatter(self) -> PackedCompletedScatter:
        """Complete one exact destination scatter.

        :returns: Exact completed scatter owner.
        """

        reported, receipt = self.scatter.submit(
            destination_lease=self.staging_lease,
            resources=(OpaqueOwner(),),
            cohort=self.cohort,
        )
        inflight = self.scatter.accept_submission(
            reported,
            receipt,
        )
        completion_receipt = self.scatter.poll(inflight)
        if completion_receipt is None:
            raise RuntimeError("test scatter unexpectedly returned pending")
        return self.scatter.accept_completion(
            inflight,
            completion_receipt,
        )

    def teardown(
        self,
        completed_scatter: PackedCompletedScatter,
    ) -> PackedTeardownCoordinator:
        """Build exact teardown after native and scatter completion.

        :param completed_scatter: Exact terminal destination scatter.
        :returns: Teardown coordinator.
        """

        return PackedTeardownCoordinator(
            key=self.key,
            capability=self.capability,
            capability_authority=self.capability_authority,
            page_leases=self.page_lease_bindings,
            completed_transfers=self.completed,
            native_lifecycle=self.lifecycle,
            completed_scatter=completed_scatter,
            scatter_lifecycle=self.scatter,
            cohort=self.cohort,
            quarantine=self.quarantine,
        )


@pytest.mark.parametrize("source_tp_size", (1, 2, 4))
def test_manifest_and_completion_require_every_tp_writer(
    source_tp_size: int,
) -> None:
    """Require every supported packed writer before cohort completion."""

    harness = LifecycleHarness(source_tp_size)
    completed = harness.complete_native()
    barrier = PackedWriterCohortCompletion(
        harness.capability,
        harness.lifecycle,
    )
    for writer_completion in completed[:-1]:
        assert barrier.record(writer_completion) is None
    receipt = barrier.record(completed[-1])
    assert receipt is not None
    assert barrier.consume(receipt) == completed
    with pytest.raises(PackedLifecycleError, match="already consumed"):
        barrier.consume(receipt)


def test_writer_completion_rejects_same_writer_from_another_request() -> None:
    """Reject equal writer IDs whose completion belongs to another capability."""

    harness = LifecycleHarness(2)
    second_generation = b"x" * 16
    second_manifest = PackedWriterCohortManifest(
        request_generation=second_generation,
        source_tp_size=2,
        destination_tp_size=1,
        destination_tp_rank=0,
        total_bytes=2 * ALIGNMENT_BYTES,
        alignment_bytes=ALIGNMENT_BYTES,
        projections=tuple(
            PackedWriterProjection(
                writer_id=projection.writer_id,
                peer_generation=projection.peer_generation,
                destination_offset=projection.destination_offset,
                length_bytes=projection.length_bytes,
            )
            for projection in harness.manifest.projections
        ),
    )
    second_capability, _ = harness.capability_authority.issue(
        peers=harness.peers,
        manifest=second_manifest,
        route_digest=ROUTE_DIGEST,
        base_address=0x200000,
        total_size=second_manifest.total_bytes,
        destination_gpu_id=3,
        alignment_bytes=ALIGNMENT_BYTES,
    )
    second_source_lease = harness.source_allocator.claim(
        owner=OpaqueOwner(),
        pages={MAIN_KV: np.array([3, 4], dtype=np.int32)},
    )
    second_destination_lease = harness.destination_allocator.claim(
        owner=OpaqueOwner(),
        pages={MAIN_KV: np.array([9, 10], dtype=np.int32)},
    )
    second_cohort = harness.capability_authority.create_resource_cohort(
        PackedCohortResources(
            capability=second_capability,
            source_page_leases=((harness.source_allocator, second_source_lease),),
            destination_page_leases=(
                (
                    harness.destination_allocator,
                    second_destination_lease,
                ),
            ),
            staging_leases=(OpaqueOwner(),),
            tensors=(OpaqueOwner(),),
            registrations=(harness.registration,),
            endpoints=harness.endpoints,
            arenas=(harness.arena_owner,),
        )
    )
    second_handles = tuple(OpaqueOwner() for _ in harness.writers)
    second_prepared = tuple(
        harness.lifecycle.register_prepared(
            writer_id=writer,
            peer=peer,
            capability=second_capability,
            handle=handle,
            cohort=second_cohort,
        )
        for writer, peer, handle in zip(
            harness.writers,
            harness.peers,
            second_handles,
            strict=True,
        )
    )
    second_completed: list[PackedCompletedTransfer] = []
    for prepared in second_prepared:
        submit_receipt = harness.lifecycle.post(prepared)
        inflight = harness.lifecycle.accept_submission(
            prepared,
            submit_receipt,
        )
        completion_receipt = harness.lifecycle.poll(inflight)
        if completion_receipt is None:
            raise RuntimeError("test driver unexpectedly returned pending")
        second_completed.append(
            harness.lifecycle.accept_completion(
                inflight,
                completion_receipt,
            )
        )
    barrier = PackedWriterCohortCompletion(
        harness.capability,
        harness.lifecycle,
    )
    with pytest.raises(PackedLifecycleError, match="another capability"):
        barrier.record(second_completed[0])


def test_manifest_rejects_overlapping_or_incomplete_projection_geometry() -> None:
    """Reject overlapping, gapped, and partially covered writer projections."""

    writers = tuple(StagingWriterId(rank, rank, 0, 0) for rank in range(2))
    generations = (b"a" * 16, b"b" * 16)

    def build(
        offset: int,
        total_bytes: int = 2 * ALIGNMENT_BYTES,
    ) -> None:
        """Build candidate geometry.

        :param offset: Second writer offset.
        :param total_bytes: Destination lease capacity.
        """

        PackedWriterCohortManifest(
            request_generation=REQUEST_GENERATION,
            source_tp_size=2,
            destination_tp_size=1,
            destination_tp_rank=0,
            total_bytes=total_bytes,
            alignment_bytes=ALIGNMENT_BYTES,
            projections=(
                PackedWriterProjection(
                    writers[0],
                    generations[0],
                    0,
                    ALIGNMENT_BYTES,
                ),
                PackedWriterProjection(
                    writers[1],
                    generations[1],
                    offset,
                    ALIGNMENT_BYTES,
                ),
            ),
        )

    with pytest.raises(ValueError, match="overlap"):
        build(0)
    with pytest.raises(ValueError, match="gap"):
        build(2 * ALIGNMENT_BYTES, 3 * ALIGNMENT_BYTES)
    with pytest.raises(ValueError, match="cover"):
        build(ALIGNMENT_BYTES, 3 * ALIGNMENT_BYTES)


def test_capability_authentication_peer_binding_replay_and_revocation() -> None:
    """Reject grant tampering, wrong peers, replay, and revoked authority."""

    harness = LifecycleHarness(2)
    tampered = dataclasses.replace(
        harness.grant,
        destination_gpu_id=4,
    )
    with pytest.raises(PackedLifecycleError, match="modified"):
        harness.capability_authority.resolve(tampered, harness.peers)
    with pytest.raises(PackedLifecycleError, match="another writer cohort"):
        harness.capability_authority.resolve(
            harness.grant,
            tuple(reversed(harness.peers)),
        )
    assert (
        harness.capability_authority.resolve(
            harness.grant,
            harness.peers,
        )
        is harness.capability
    )
    with pytest.raises(PackedLifecycleError, match="already resolved"):
        harness.capability_authority.resolve(
            harness.grant,
            harness.peers,
        )

    completed = harness.complete_native()
    assert len(completed) == 2
    scatter = harness.complete_scatter()
    coordinator = harness.teardown(scatter)
    requests = coordinator.begin()
    receipt = None
    for request, peer in zip(requests, harness.peers, strict=True):
        receipt = coordinator.acknowledge(
            PackedTeardownAck(
                key=request.key,
                writer_id=request.writer_id,
                capability_id=request.capability_id,
                teardown_generation=request.teardown_generation,
            ),
            peer,
        )
    if receipt is None:
        raise RuntimeError("final teardown ack produced no receipt")
    coordinator.finalize(receipt)
    assert (
        harness.capability_authority.state(harness.capability)
        is PackedCapabilityState.REVOKED
    )
    with pytest.raises(PackedLifecycleError):
        harness.capability_authority.resolve(
            harness.grant,
            harness.peers,
        )


def test_page_leases_are_exact_allocator_owned_and_irreversibly_read_only() -> None:
    """Reject cross-allocator leases and writable immutable snapshots."""

    harness = LifecycleHarness(2)
    pages = harness.source_allocator.pages(harness.source_lease)
    snapshot = pages[0][1]
    assert not snapshot.flags.writeable
    with pytest.raises(ValueError):
        snapshot.setflags(write=True)
    another_allocator = PackedPageLeaseAllocator(PackedPageLeaseRole.SOURCE)
    with pytest.raises(PackedLifecycleError, match="another allocator"):
        another_allocator.pages(harness.source_lease)
    with pytest.raises(TypeError, match="allocator owned"):
        PackedPageLease(object(), object(), object())


def test_resource_cohort_requires_authority_owned_anchors() -> None:
    """Reject public construction and omitted registration authority."""

    harness = LifecycleHarness(2)
    resources = PackedCohortResources(
        capability=harness.capability,
        source_page_leases=(harness.page_lease_bindings[0],),
        destination_page_leases=(harness.page_lease_bindings[1],),
        staging_leases=(OpaqueOwner(),),
        tensors=(OpaqueOwner(),),
        registrations=(OpaqueOwner(),),
        endpoints=harness.endpoints,
        arenas=(harness.arena_owner,),
    )
    with pytest.raises(PackedLifecycleError, match="registration owner"):
        harness.capability_authority.create_resource_cohort(resources)
    with pytest.raises(TypeError, match="lifecycle owned"):
        PackedResourceCohort(
            resources,
            harness.writers,
            harness.capability_authority,
            object(),
            object(),
        )


def test_submit_and_completion_receipts_are_one_shot_and_cross_bound() -> None:
    """Reject submit and completion receipt replay or cross-writer binding."""

    harness = LifecycleHarness(2)
    first_submit = harness.lifecycle.post(harness.prepared[0])
    second_submit = harness.lifecycle.post(harness.prepared[1])
    with pytest.raises(PackedLifecycleError, match="another operation"):
        harness.lifecycle.accept_submission(
            harness.prepared[1],
            first_submit,
        )
    first_inflight = harness.lifecycle.accept_submission(
        harness.prepared[0],
        first_submit,
    )
    second_inflight = harness.lifecycle.accept_submission(
        harness.prepared[1],
        second_submit,
    )
    first_completion = harness.lifecycle.poll(first_inflight)
    second_completion = harness.lifecycle.poll(second_inflight)
    if first_completion is None or second_completion is None:
        raise RuntimeError("test driver unexpectedly returned pending")
    with pytest.raises(PackedLifecycleError, match="another operation"):
        harness.lifecycle.accept_completion(
            second_inflight,
            first_completion,
        )
    harness.lifecycle.accept_completion(first_inflight, first_completion)
    harness.lifecycle.accept_completion(second_inflight, second_completion)
    with pytest.raises(PackedLifecycleError):
        harness.lifecycle.accept_completion(first_inflight, first_completion)


@pytest.mark.parametrize(
    ("operation", "failure"),
    (
        ("post", RuntimeError("injected post exception")),
        ("post", PackedNativePostState.ERROR),
        ("poll", RuntimeError("injected poll exception")),
        ("poll", PackedNativePollState.ERROR),
    ),
)
def test_native_ambiguity_quarantines_complete_cohort_without_outcome(
    operation: str,
    failure: Exception | PackedNativePostState | PackedNativePollState,
) -> None:
    """Quarantine every request object for post or poll ambiguity."""

    harness = LifecycleHarness(4)
    handle = harness.handles[0]
    if operation == "post":
        harness.driver.post_results[handle] = failure
        with pytest.raises(PackedAmbiguousTransportError):
            harness.lifecycle.post(harness.prepared[0])
    else:
        submit = harness.lifecycle.post(harness.prepared[0])
        inflight = harness.lifecycle.accept_submission(
            harness.prepared[0],
            submit,
        )
        harness.driver.poll_results[handle] = failure
        with pytest.raises(PackedAmbiguousTransportError):
            harness.lifecycle.poll(inflight)

    assert harness.quarantine.count == 1
    assert harness.cohort.quarantined
    assert (
        harness.capability_authority.state(harness.capability)
        is PackedCapabilityState.QUARANTINED
    )
    assert not harness.source_allocator.is_reusable(harness.source_lease)
    assert not harness.destination_allocator.is_reusable(harness.destination_lease)
    for retained in (
        harness.capability,
        harness.registration,
        harness.arena_owner,
        harness.staging_lease,
        *harness.endpoints,
        *harness.handles,
    ):
        assert harness.quarantine.retains(retained)
    for prepared in harness.prepared:
        assert (
            harness.lifecycle.state(prepared) is PackedNativeTransferState.QUARANTINED
        )


def test_connection_loss_quarantines_without_writer_completion() -> None:
    """Treat connection loss as process-lifetime ambiguity."""

    harness = LifecycleHarness(2)
    submit = harness.lifecycle.post(harness.prepared[0])
    inflight = harness.lifecycle.accept_submission(
        harness.prepared[0],
        submit,
    )
    harness.lifecycle.abandon(inflight, "authenticated connection lost")
    assert harness.quarantine.count == 1
    assert harness.cohort.quarantined
    with pytest.raises(PackedLifecycleError):
        harness.lifecycle.poll(inflight)


@pytest.mark.parametrize("operation", ("post", "poll"))
def test_native_async_interruption_quarantines_before_propagation(
    operation: str,
) -> None:
    """Quarantine an ambiguous native operation before propagating interrupt."""

    harness = LifecycleHarness(2)
    handle = harness.handles[0]
    if operation == "post":
        harness.driver.post_results[handle] = KeyboardInterrupt()
        with pytest.raises(KeyboardInterrupt):
            harness.lifecycle.post(harness.prepared[0])
    else:
        submit = harness.lifecycle.post(harness.prepared[0])
        inflight = harness.lifecycle.accept_submission(
            harness.prepared[0],
            submit,
        )
        harness.driver.poll_results[handle] = KeyboardInterrupt()
        with pytest.raises(KeyboardInterrupt):
            harness.lifecycle.poll(inflight)
    assert harness.quarantine.count == 1
    assert harness.cohort.quarantined


def test_concurrent_writer_failure_withholds_peer_submit_receipt() -> None:
    """Withhold a concurrent success once any writer quarantines the cohort."""

    driver = CoordinatedPostDriver()
    harness = LifecycleHarness(2, driver)
    driver.blocked_handle = harness.handles[0]
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        successful_post = executor.submit(
            harness.lifecycle.post,
            harness.prepared[0],
        )
        assert driver.success_entered.wait(timeout=5)
        with pytest.raises(PackedAmbiguousTransportError):
            harness.lifecycle.post(harness.prepared[1])
        driver.release_success.set()
        with pytest.raises(PackedAmbiguousTransportError):
            successful_post.result(timeout=5)
    assert harness.quarantine.count == 1
    for prepared in harness.prepared:
        assert (
            harness.lifecycle.state(prepared) is PackedNativeTransferState.QUARANTINED
        )


def test_stock_nixl_and_missing_per_handle_lane_proof_fail_closed() -> None:
    """Keep stock NIXL and missing transport attestation hard-gated."""

    with pytest.raises(
        PackedNativeAttestationUnavailable,
        match="query_xfer_ucp_transports",
    ):
        require_nixl_ucx_native_attestation(StockNixlAgent())

    harness = LifecycleHarness(2)
    harness.driver.transports = None
    with pytest.raises(PackedAmbiguousTransportError):
        harness.lifecycle.post(harness.prepared[0])
    assert harness.quarantine.count == 1


def test_scatter_submission_failure_retains_prelaunch_resources() -> None:
    """Quarantine planned resources when trusted scatter submit is ambiguous."""

    harness = LifecycleHarness(2)
    harness.complete_native()
    temporary = OpaqueOwner()
    harness.scatter_driver.submit_failure = RuntimeError(
        "injected scatter submit failure"
    )
    with pytest.raises(PackedAmbiguousTransportError):
        harness.scatter.submit(
            destination_lease=harness.staging_lease,
            resources=(temporary,),
            cohort=harness.cohort,
        )
    assert harness.quarantine.count == 1
    assert harness.quarantine.retains(temporary)
    assert (
        harness.capability_authority.state(harness.capability)
        is PackedCapabilityState.QUARANTINED
    )


def test_scatter_receipts_are_cross_bound_and_ambiguity_quarantines() -> None:
    """Reject cross-event receipts and quarantine ambiguous CUDA progress."""

    harness = LifecycleHarness(2)
    harness.complete_native()
    first_temporary = OpaqueOwner()
    second_temporary = OpaqueOwner()
    first_reported, first_receipt = harness.scatter.submit(
        destination_lease=harness.staging_lease,
        resources=(first_temporary,),
        cohort=harness.cohort,
    )
    second_reported, second_receipt = harness.scatter.submit(
        destination_lease=harness.staging_lease,
        resources=(second_temporary,),
        cohort=harness.cohort,
    )
    first_event, second_event = harness.scatter_driver.submitted_events
    with pytest.raises(PackedLifecycleError, match="another operation"):
        harness.scatter.accept_submission(
            second_reported,
            first_receipt,
        )
    first_inflight = harness.scatter.accept_submission(
        first_reported,
        first_receipt,
    )
    harness.scatter.accept_submission(
        second_reported,
        second_receipt,
    )
    harness.scatter_driver.results[first_event] = PackedScatterPollState.ERROR
    with pytest.raises(PackedAmbiguousTransportError):
        harness.scatter.poll(first_inflight)
    assert harness.quarantine.count == 1
    assert (
        harness.capability_authority.state(harness.capability)
        is PackedCapabilityState.QUARANTINED
    )
    for retained in (
        first_event,
        second_event,
        first_temporary,
        second_temporary,
    ):
        assert harness.quarantine.retains(retained)


def test_scatter_completion_receipts_are_one_shot_and_cross_bound() -> None:
    """Reject scatter-completion receipt replay and cross-event binding."""

    harness = LifecycleHarness(2)
    harness.complete_native()
    submissions = tuple(
        harness.scatter.submit(
            destination_lease=harness.staging_lease,
            resources=(OpaqueOwner(),),
            cohort=harness.cohort,
        )
        for _ in range(2)
    )
    inflight = tuple(
        harness.scatter.accept_submission(
            reported,
            receipt,
        )
        for reported, receipt in submissions
    )
    completion_receipts = tuple(harness.scatter.poll(owner) for owner in inflight)
    if any(receipt is None for receipt in completion_receipts):
        raise RuntimeError("test scatter unexpectedly returned pending")
    first_receipt = completion_receipts[0]
    second_receipt = completion_receipts[1]
    if first_receipt is None or second_receipt is None:
        raise RuntimeError("completion receipt narrowing failed")
    with pytest.raises(PackedLifecycleError, match="another operation"):
        harness.scatter.accept_completion(
            inflight[1],
            first_receipt,
        )
    harness.scatter.accept_completion(inflight[0], first_receipt)
    harness.scatter.accept_completion(inflight[1], second_receipt)
    with pytest.raises(PackedLifecycleError):
        harness.scatter.accept_completion(inflight[0], first_receipt)


@pytest.mark.parametrize("source_tp_size", (1, 2, 4))
def test_teardown_waits_for_every_exact_authenticated_ack(
    source_tp_size: int,
) -> None:
    """Release nothing until every supported writer acks exact teardown."""

    harness = LifecycleHarness(source_tp_size)
    harness.complete_native()
    completed_scatter = harness.complete_scatter()
    coordinator = harness.teardown(completed_scatter)
    requests = coordinator.begin()
    for request, peer in zip(
        requests[:-1],
        harness.peers[:-1],
        strict=True,
    ):
        assert (
            coordinator.acknowledge(
                PackedTeardownAck(
                    key=request.key,
                    writer_id=request.writer_id,
                    capability_id=request.capability_id,
                    teardown_generation=request.teardown_generation,
                ),
                peer,
            )
            is None
        )
    assert coordinator.state is PackedTeardownState.WAITING_FOR_ACKS
    assert harness.driver.released == []
    assert not harness.source_allocator.is_reusable(harness.source_lease)
    assert (
        harness.capability_authority.state(harness.capability)
        is PackedCapabilityState.ACTIVE
    )

    final_request = requests[-1]
    receipt = coordinator.acknowledge(
        PackedTeardownAck(
            key=final_request.key,
            writer_id=final_request.writer_id,
            capability_id=final_request.capability_id,
            teardown_generation=final_request.teardown_generation,
        ),
        harness.peers[-1],
    )
    if receipt is None:
        raise RuntimeError("final teardown ack produced no receipt")
    assert harness.driver.released == []
    coordinator.finalize(receipt)
    assert coordinator.state is PackedTeardownState.RELEASED
    assert harness.driver.released == list(harness.handles)
    assert harness.source_allocator.is_reusable(harness.source_lease)
    assert harness.destination_allocator.is_reusable(harness.destination_lease)
    assert (
        harness.capability_authority.state(harness.capability)
        is PackedCapabilityState.REVOKED
    )
    with pytest.raises(PackedLifecycleError):
        coordinator.finalize(receipt)


def test_teardown_rejects_wrong_peer_and_quarantines_connection_loss() -> None:
    """Authenticate every ack and quarantine an unprovable remaining ack."""

    harness = LifecycleHarness(2)
    harness.complete_native()
    completed_scatter = harness.complete_scatter()
    coordinator = harness.teardown(completed_scatter)
    requests = coordinator.begin()
    first = requests[0]
    ack = PackedTeardownAck(
        key=first.key,
        writer_id=first.writer_id,
        capability_id=first.capability_id,
        teardown_generation=first.teardown_generation,
    )
    with pytest.raises(PackedLifecycleError, match="authenticated peer"):
        coordinator.acknowledge(ack, harness.peers[1])
    coordinator.connection_lost(harness.peers[1])
    assert coordinator.state is PackedTeardownState.QUARANTINED
    assert harness.quarantine.count == 1
    assert not harness.source_allocator.is_reusable(harness.source_lease)
    assert (
        harness.capability_authority.state(harness.capability)
        is PackedCapabilityState.QUARANTINED
    )


def test_teardown_release_failure_quarantines_every_resource() -> None:
    """Quarantine exact ownership when completed-handle release fails."""

    harness = LifecycleHarness(2)
    harness.complete_native()
    completed_scatter = harness.complete_scatter()
    harness.driver.release_failures.add(harness.handles[0])
    coordinator = harness.teardown(completed_scatter)
    requests = coordinator.begin()
    receipt = None
    for request, peer in zip(requests, harness.peers, strict=True):
        receipt = coordinator.acknowledge(
            PackedTeardownAck(
                key=request.key,
                writer_id=request.writer_id,
                capability_id=request.capability_id,
                teardown_generation=request.teardown_generation,
            ),
            peer,
        )
    if receipt is None:
        raise RuntimeError("final teardown ack produced no receipt")
    with pytest.raises(PackedProcessQuarantinedError):
        coordinator.finalize(receipt)
    assert coordinator.state is PackedTeardownState.QUARANTINED
    assert harness.quarantine.count == 1
    assert not harness.source_allocator.is_reusable(harness.source_lease)
    for retained in (
        *harness.handles,
        harness.staging_lease,
        harness.registration,
        harness.arena_owner,
    ):
        assert harness.quarantine.retains(retained)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
