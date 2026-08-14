import dataclasses
import hashlib
import json
import uuid
from collections.abc import Callable

import pytest

from sglang.srt.disaggregation.nixl.startup_enrollment_ack import (
    TERMINAL_STARTUP_ENROLLMENT_ACK_MAX_BYTES,
    TERMINAL_STARTUP_ENROLLMENT_ACK_SCHEMA,
    TERMINAL_STARTUP_ENROLLMENT_ACK_TAG,
    TerminalStartupEnrollmentAck,
    TerminalStartupEnrollmentAckError,
    build_terminal_startup_enrollment_ack,
    decode_terminal_startup_enrollment_ack,
    encode_terminal_startup_enrollment_ack,
    terminal_decoder_registration_multipart_sha256,
)
from sglang.srt.disaggregation.terminal_progress.identity import TerminalOwnerRole
from sglang.srt.disaggregation.terminal_progress.startup_cohort import (
    TerminalStartupCohortMatrix,
    TerminalStartupRankAdvertisement,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_COHORT_SHA256 = bytes.fromhex("11" * 32)
_ROUTE_TABLE_SHA256 = bytes.fromhex("22" * 32)
_REGISTRATION_FRAMES = (
    b"NIXL-KV",
    b"None",
    b"10.0.0.3",
    b"32003",
    b"decode-agent-0",
    b"decoder-native-metadata",
    b"",
    b"packed-v4",
)


def _rank(
    *,
    service_id: str,
    role: TerminalOwnerRole,
    tensor_parallel_rank: int,
    tensor_parallel_size: int,
    generation: int,
    agent_name: str,
    metadata: bytes,
    launch_instance: int,
    port: int,
) -> TerminalStartupRankAdvertisement:
    """Build one exact startup matrix row.

    :param service_id: Static service identifier.
    :param role: Source or decoder role.
    :param tensor_parallel_rank: Rank within the service.
    :param tensor_parallel_size: Exact service TP width.
    :param generation: Non-nil process-generation UUID integer.
    :param agent_name: Exact native agent name.
    :param metadata: Complete native metadata represented by the row.
    :param launch_instance: Non-nil service-incarnation UUID integer.
    :param port: Exact service port.
    :returns: Validated startup rank.
    """

    return TerminalStartupRankAdvertisement(
        group_id="group-a",
        cohort_sha256=_COHORT_SHA256,
        service_id=service_id,
        service_origin=f"http://127.0.0.1:{port}",
        role=role,
        launch_instance_id=uuid.UUID(int=launch_instance).bytes,
        tensor_parallel_rank=tensor_parallel_rank,
        tensor_parallel_size=tensor_parallel_size,
        process_generation=uuid.UUID(int=generation).bytes,
        nixl_agent_name=agent_name,
        nixl_agent_metadata_sha256=hashlib.sha256(metadata).digest(),
    )


def _matrix() -> TerminalStartupCohortMatrix:
    """Build a TP2 source with two TP1 decoder services.

    :returns: Complete source-first startup matrix.
    """

    return TerminalStartupCohortMatrix(
        group_id="group-a",
        cohort_sha256=_COHORT_SHA256,
        ranks=(
            _rank(
                service_id="prefill-a",
                role=TerminalOwnerRole.SOURCE,
                tensor_parallel_rank=0,
                tensor_parallel_size=2,
                generation=101,
                agent_name="prefill-agent-0",
                metadata=b"prefill-native-metadata-0",
                launch_instance=1,
                port=32001,
            ),
            _rank(
                service_id="prefill-a",
                role=TerminalOwnerRole.SOURCE,
                tensor_parallel_rank=1,
                tensor_parallel_size=2,
                generation=102,
                agent_name="prefill-agent-1",
                metadata=b"prefill-native-metadata-1",
                launch_instance=1,
                port=32001,
            ),
            _rank(
                service_id="decode-a",
                role=TerminalOwnerRole.DECODE,
                tensor_parallel_rank=0,
                tensor_parallel_size=1,
                generation=201,
                agent_name="decode-agent-0",
                metadata=b"decoder-native-metadata",
                launch_instance=2,
                port=32002,
            ),
            _rank(
                service_id="decode-b",
                role=TerminalOwnerRole.DECODE,
                tensor_parallel_rank=0,
                tensor_parallel_size=1,
                generation=301,
                agent_name="decode-agent-1",
                metadata=b"decoder-native-metadata-1",
                launch_instance=3,
                port=32003,
            ),
        ),
    )


def _acknowledgement() -> TerminalStartupEnrollmentAck:
    """Build one valid source-to-decoder acknowledgement.

    :returns: Matrix-authenticated immutable acknowledgement.
    """

    matrix = _matrix()
    return build_terminal_startup_enrollment_ack(
        matrix,
        matrix.rank("prefill-a", 0),
        matrix.rank("decode-a", 0),
        _REGISTRATION_FRAMES,
        _ROUTE_TABLE_SHA256,
    )


def _wire_object() -> dict[str, object]:
    """Return one valid acknowledgement as a mutable JSON object.

    :returns: Parsed canonical wire object.
    """

    decoded = json.loads(encode_terminal_startup_enrollment_ack(_acknowledgement()))
    assert type(decoded) is dict
    return decoded


def _wire(payload: dict[str, object]) -> bytes:
    """Encode a test JSON object with production separators.

    :param payload: Mutable test payload.
    :returns: Compact JSON bytes.
    """

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()


def test_multipart_digest_binds_order_boundaries_and_empty_frames() -> None:
    """The registration digest commits to every exact frame boundary."""

    assert (
        terminal_decoder_registration_multipart_sha256((b"ab", b"", b"c")).hex()
        == "9c24d27974bb3b173e762a999df89274e93d1e37cba113b996aa47139a011d4a"
    )
    assert terminal_decoder_registration_multipart_sha256(
        (b"ab", b"c")
    ) != terminal_decoder_registration_multipart_sha256((b"a", b"bc"))
    assert terminal_decoder_registration_multipart_sha256(
        (b"ab", b"c")
    ) != terminal_decoder_registration_multipart_sha256((b"c", b"ab"))

    with pytest.raises(ValueError, match="nonempty tuple"):
        terminal_decoder_registration_multipart_sha256(())
    with pytest.raises(ValueError, match="nonempty tuple"):
        terminal_decoder_registration_multipart_sha256([b"registration"])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="contain bytes"):
        terminal_decoder_registration_multipart_sha256((b"registration", "x"))  # type: ignore[arg-type]


def test_builder_binds_matrix_endpoints_and_exact_registration() -> None:
    """Construction retains every required startup and registration field."""

    matrix = _matrix()
    source = matrix.rank("prefill-a", 1)
    decoder = matrix.rank("decode-b", 0)
    acknowledgement = build_terminal_startup_enrollment_ack(
        matrix,
        source,
        decoder,
        _REGISTRATION_FRAMES,
        _ROUTE_TABLE_SHA256,
    )

    assert acknowledgement.startup_matrix_sha256 == matrix.digest
    assert acknowledgement.source_service_id == source.service_id
    assert acknowledgement.source_tensor_parallel_rank == 1
    assert acknowledgement.source_process_generation == source.process_generation
    assert acknowledgement.source_nixl_agent_name == source.nixl_agent_name
    assert (
        acknowledgement.source_nixl_agent_metadata_sha256
        == source.nixl_agent_metadata_sha256
    )
    assert acknowledgement.target_decoder_service_id == decoder.service_id
    assert acknowledgement.target_decoder_tensor_parallel_rank == 0
    assert (
        acknowledgement.target_decoder_process_generation == decoder.process_generation
    )
    assert acknowledgement.decoder_registration_multipart_sha256 == (
        terminal_decoder_registration_multipart_sha256(_REGISTRATION_FRAMES)
    )
    assert acknowledgement.decoder_control_route_table_sha256 == (
        _ROUTE_TABLE_SHA256
    )
    acknowledgement.require_matrix(matrix)
    acknowledgement.require_decoder_registration(_REGISTRATION_FRAMES)


def test_acknowledgement_is_frozen_and_schema_bearing() -> None:
    """The admitted proof is immutable and its digest includes the schema."""

    acknowledgement = _acknowledgement()
    wire = encode_terminal_startup_enrollment_ack(acknowledgement)

    assert wire.startswith(
        b'{"schema":"' + TERMINAL_STARTUP_ENROLLMENT_ACK_SCHEMA.encode() + b'"'
    )
    assert acknowledgement.digest == hashlib.sha256(wire).digest()
    with pytest.raises(dataclasses.FrozenInstanceError):
        acknowledgement.source_service_id = "prefill-b"  # type: ignore[misc]


def test_canonical_codec_round_trip_is_byte_stable() -> None:
    """The sole wire schema round-trips to exactly the same bytes."""

    acknowledgement = _acknowledgement()
    encoded = encode_terminal_startup_enrollment_ack(acknowledgement)
    multipart = (TERMINAL_STARTUP_ENROLLMENT_ACK_TAG, encoded)
    decoded = decode_terminal_startup_enrollment_ack(multipart[1])

    assert multipart[0] == b"TERMINAL_STARTUP_ENROLLMENT_ACK"
    assert decoded == acknowledgement
    assert encode_terminal_startup_enrollment_ack(decoded) == encoded


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("startup_matrix_sha256", b"x" * 32, "another startup matrix"),
        ("source_service_id", "missing-source", "rank is absent"),
        ("source_tensor_parallel_rank", 1, "source differs"),
        (
            "source_process_generation",
            uuid.UUID(int=999).bytes,
            "source differs",
        ),
        ("source_nixl_agent_name", "other-source", "source differs"),
        ("source_nixl_agent_metadata_sha256", b"m" * 32, "source differs"),
        ("target_decoder_service_id", "missing-decoder", "rank is absent"),
        ("target_decoder_tensor_parallel_rank", 1, "rank is absent"),
        (
            "target_decoder_process_generation",
            uuid.UUID(int=999).bytes,
            "decoder differs",
        ),
    ),
)
def test_matrix_validation_rejects_every_bound_identity_drift(
    field: str,
    replacement: object,
    message: str,
) -> None:
    """No endpoint identity may drift from the sealed matrix.

    :param field: Dataclass field under mutation.
    :param replacement: Structurally valid conflicting value.
    :param message: Expected fail-closed diagnostic.
    """

    acknowledgement = dataclasses.replace(
        _acknowledgement(),
        **{field: replacement},
    )
    with pytest.raises(TerminalStartupEnrollmentAckError, match=message):
        acknowledgement.require_matrix(_matrix())


def test_matrix_validation_rejects_role_inversion() -> None:
    """A matrix identity cannot be replayed in the opposite endpoint role."""

    matrix = _matrix()
    source = matrix.rank("prefill-a", 0)
    decoder = matrix.rank("decode-a", 0)
    acknowledgement = _acknowledgement()
    decoder_as_source = dataclasses.replace(
        acknowledgement,
        source_service_id=decoder.service_id,
        source_tensor_parallel_rank=decoder.tensor_parallel_rank,
        source_process_generation=decoder.process_generation,
        source_nixl_agent_name=decoder.nixl_agent_name,
        source_nixl_agent_metadata_sha256=decoder.nixl_agent_metadata_sha256,
    )
    with pytest.raises(TerminalStartupEnrollmentAckError, match="not a source"):
        decoder_as_source.require_matrix(matrix)

    source_as_decoder = dataclasses.replace(
        acknowledgement,
        target_decoder_service_id=source.service_id,
        target_decoder_tensor_parallel_rank=source.tensor_parallel_rank,
        target_decoder_process_generation=source.process_generation,
    )
    with pytest.raises(TerminalStartupEnrollmentAckError, match="not a decoder"):
        source_as_decoder.require_matrix(matrix)


def test_builder_rejects_rows_outside_the_exact_matrix() -> None:
    """Construction cannot bless a stale row or reverse endpoint roles."""

    matrix = _matrix()
    source = matrix.rank("prefill-a", 0)
    decoder = matrix.rank("decode-a", 0)
    stale_source = dataclasses.replace(
        source,
        process_generation=uuid.UUID(int=999).bytes,
    )

    with pytest.raises(TerminalStartupEnrollmentAckError, match="exact matrix source"):
        build_terminal_startup_enrollment_ack(
            matrix,
            stale_source,
            decoder,
            _REGISTRATION_FRAMES,
            _ROUTE_TABLE_SHA256,
        )
    with pytest.raises(TerminalStartupEnrollmentAckError, match="matrix source"):
        build_terminal_startup_enrollment_ack(
            matrix,
            decoder,
            source,
            _REGISTRATION_FRAMES,
            _ROUTE_TABLE_SHA256,
        )


def test_registration_validation_rejects_any_frame_drift() -> None:
    """An ACK for one multipart registration cannot commit another."""

    acknowledgement = _acknowledgement()
    changed = (*_REGISTRATION_FRAMES[:-1], b"packed-v5")

    with pytest.raises(TerminalStartupEnrollmentAckError, match="another decoder"):
        acknowledgement.require_decoder_registration(changed)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda payload: payload.update({"unexpected": "field"}),
        lambda payload: payload.pop("target_decoder"),
        lambda payload: payload.update({"schema": "packed-terminal-ack-v0"}),
        lambda payload: payload["source"].update({"unexpected": "field"}),  # type: ignore[union-attr]
        lambda payload: payload["target_decoder"].pop("service_id"),  # type: ignore[union-attr]
        lambda payload: payload.update(
            {"startup_matrix_sha256": str(payload["startup_matrix_sha256"]).upper()}
        ),
        lambda payload: payload["source"].update(  # type: ignore[union-attr]
            {"process_generation": "00000000-0000-0000-0000-000000000000"}
        ),
        lambda payload: payload["source"].update(  # type: ignore[union-attr]
            {"tensor_parallel_rank": 0.0}
        ),
    ),
)
def test_codec_rejects_unknown_missing_and_noncanonical_fields(
    mutation: Callable[[dict[str, object]], object],
) -> None:
    """The codec accepts one exact schema without permissive alternatives.

    :param mutation: In-place payload mutation.
    """

    payload = _wire_object()
    mutation(payload)
    with pytest.raises(TerminalStartupEnrollmentAckError):
        decode_terminal_startup_enrollment_ack(_wire(payload))


def test_codec_rejects_duplicate_noncanonical_and_unbounded_wire() -> None:
    """Duplicate keys, alternate JSON bytes, and oversized input fail closed."""

    encoded = encode_terminal_startup_enrollment_ack(_acknowledgement())
    duplicate = encoded.replace(
        b'{"schema":',
        b'{"schema":"duplicate","schema":',
        1,
    )

    with pytest.raises(TerminalStartupEnrollmentAckError, match="duplicate"):
        decode_terminal_startup_enrollment_ack(duplicate)
    with pytest.raises(TerminalStartupEnrollmentAckError, match="not canonical"):
        decode_terminal_startup_enrollment_ack(b" " + encoded)
    with pytest.raises(TerminalStartupEnrollmentAckError, match="size"):
        decode_terminal_startup_enrollment_ack(
            b"x" * (TERMINAL_STARTUP_ENROLLMENT_ACK_MAX_BYTES + 1)
        )
    with pytest.raises(TerminalStartupEnrollmentAckError, match="size"):
        decode_terminal_startup_enrollment_ack(b"")
    with pytest.raises(TerminalStartupEnrollmentAckError, match="size"):
        decode_terminal_startup_enrollment_ack("not bytes")  # type: ignore[arg-type]


def test_codec_rejects_nonfinite_json_constant() -> None:
    """The parser does not admit JSON constants outside the typed model."""

    encoded = encode_terminal_startup_enrollment_ack(_acknowledgement())
    malformed = encoded.replace(
        b'"tensor_parallel_rank":0',
        b'"tensor_parallel_rank":NaN',
        1,
    )

    with pytest.raises(TerminalStartupEnrollmentAckError, match="non-finite"):
        decode_terminal_startup_enrollment_ack(malformed)
