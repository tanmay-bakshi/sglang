import dataclasses
import hashlib
import json
import os
import uuid
from pathlib import Path

import pytest
from sglang.srt.disaggregation.terminal_progress.deployment_cohort import (
    MAX_TERMINAL_DEPLOYMENT_COHORT_BYTES,
    TERMINAL_DEPLOYMENT_COHORT_SCHEMA,
    TerminalDeploymentBootstrapEndpoint,
    TerminalDeploymentCohort,
    TerminalDeploymentCohortError,
    TerminalDeploymentDecoder,
    TerminalDeploymentPrefill,
    TerminalDeploymentRequestPlan,
    TerminalDeploymentRole,
    decode_terminal_deployment_cohort,
    encode_terminal_deployment_cohort,
    load_terminal_deployment_cohort,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _uuid(marker: int) -> uuid.UUID:
    """Build one readable non-nil UUID fixture.

    :param marker: Positive low-order UUID value.
    :returns: Deterministic UUID.
    """

    return uuid.UUID(int=marker)


def _cohort() -> TerminalDeploymentCohort:
    """Build one immutable per-group deployment fixture.

    :returns: Valid deployment cohort.
    """

    return TerminalDeploymentCohort(
        group_id="group-a",
        model_fingerprint="11" * 32,
        logical_kv_layout_fingerprint="22" * 32,
        prefill=TerminalDeploymentPrefill(
            service_id="prefill-a",
            launch_instance_id=_uuid(2),
            origin="http://127.0.0.1:32001",
            bootstrap_endpoint=TerminalDeploymentBootstrapEndpoint(
                host="gemma-dev-1",
                port=32150,
            ),
            tensor_parallel_size=2,
        ),
        decoders=(
            TerminalDeploymentDecoder(
                service_id="decode-a",
                launch_instance_id=_uuid(3),
                origin="http://127.0.0.1:32002",
                tensor_parallel_size=1,
            ),
            TerminalDeploymentDecoder(
                service_id="decode-b",
                launch_instance_id=_uuid(4),
                origin="http://127.0.0.1:32003",
                tensor_parallel_size=1,
            ),
        ),
    )


def _request_plan(
    cohort: TerminalDeploymentCohort | None = None,
) -> TerminalDeploymentRequestPlan:
    """Select the first decoder in the fixture group.

    :param cohort: Optional cohort whose exact objects should be selected.
    :returns: Valid request-selected subset.
    """

    selected = _cohort() if cohort is None else cohort
    return TerminalDeploymentRequestPlan(
        cohort_digest=selected.digest,
        group_id=selected.group_id,
        prefill_service_id=selected.prefill.service_id,
        prefill_launch_instance_id=selected.prefill.launch_instance_id,
        decoder_service_id=selected.decoders[0].service_id,
        decoder_launch_instance_id=selected.decoders[0].launch_instance_id,
    )


def test_cohort_matches_canonical_launcher_schema_bytes() -> None:
    """Preserve the exact launcher-owned schema and field ordering."""

    cohort = _cohort()
    payload = encode_terminal_deployment_cohort(cohort)
    expected = {
        "schema": "packed-terminal-deployment-cohort-v1",
        "group_id": "group-a",
        "model_fingerprint": "11" * 32,
        "logical_kv_layout_fingerprint": "22" * 32,
        "prefill": {
            "id": "prefill-a",
            "launch_instance_id": str(_uuid(2)),
            "origin": "http://127.0.0.1:32001",
            "bootstrap_endpoint": {"host": "gemma-dev-1", "port": 32150},
            "tensor_parallel_size": 2,
        },
        "decoders": [
            {
                "id": "decode-a",
                "launch_instance_id": str(_uuid(3)),
                "origin": "http://127.0.0.1:32002",
                "tensor_parallel_size": 1,
            },
            {
                "id": "decode-b",
                "launch_instance_id": str(_uuid(4)),
                "origin": "http://127.0.0.1:32003",
                "tensor_parallel_size": 1,
            },
        ],
    }

    assert (
        payload
        == json.dumps(
            expected,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode()
    )
    assert decode_terminal_deployment_cohort(payload) == cohort
    assert cohort.digest == hashlib.sha256(payload).digest()
    assert payload.startswith(
        b'{"schema":"' + TERMINAL_DEPLOYMENT_COHORT_SCHEMA.encode() + b'"'
    )
    assert not payload.endswith(b"\n")


def test_local_service_requires_exact_static_membership() -> None:
    """Bind service name, generation, role, width, origin, and bootstrap."""

    cohort = _cohort()
    prefill = cohort.require_local_service(
        service_id=cohort.prefill.service_id,
        role=TerminalDeploymentRole.PREFILL,
        launch_instance_id=cohort.prefill.launch_instance_id,
        tensor_parallel_size=cohort.prefill.tensor_parallel_size,
        origin=cohort.prefill.origin,
        bootstrap_endpoint=cohort.prefill.bootstrap_endpoint,
    )
    decoder_record = cohort.decoders[1]
    decoder = cohort.require_local_service(
        service_id=decoder_record.service_id,
        role=TerminalDeploymentRole.DECODE,
        launch_instance_id=decoder_record.launch_instance_id,
        tensor_parallel_size=decoder_record.tensor_parallel_size,
        origin=decoder_record.origin,
        bootstrap_endpoint=None,
    )

    assert prefill.cohort_digest == cohort.digest
    assert decoder.service_id == decoder_record.service_id
    with pytest.raises(TerminalDeploymentCohortError, match="differs"):
        cohort.require_local_service(
            service_id=cohort.prefill.service_id,
            role=TerminalDeploymentRole.PREFILL,
            launch_instance_id=cohort.prefill.launch_instance_id,
            tensor_parallel_size=4,
            origin=cohort.prefill.origin,
            bootstrap_endpoint=cohort.prefill.bootstrap_endpoint,
        )
    with pytest.raises(TerminalDeploymentCohortError, match="absent"):
        cohort.require_local_service(
            service_id="decode-z",
            role=TerminalDeploymentRole.DECODE,
            launch_instance_id=_uuid(99),
            tensor_parallel_size=1,
            origin="http://127.0.0.1:32099",
            bootstrap_endpoint=None,
        )


def test_request_plan_must_be_an_exact_same_cohort_subset() -> None:
    """Reject stale epoch digests, cross-group plans, and service drift."""

    cohort = _cohort()
    plan = _request_plan(cohort)
    prefill, decoder = cohort.require_request_plan(plan)

    assert prefill == cohort.prefill
    assert decoder == cohort.decoders[0]
    with pytest.raises(TerminalDeploymentCohortError, match="stale"):
        cohort.require_request_plan(
            dataclasses.replace(plan, cohort_digest=bytes.fromhex("90" * 32))
        )
    with pytest.raises(TerminalDeploymentCohortError, match="another"):
        cohort.require_request_plan(dataclasses.replace(plan, group_id="group-b"))
    with pytest.raises(TerminalDeploymentCohortError, match="prefill differs"):
        cohort.require_request_plan(
            dataclasses.replace(plan, prefill_launch_instance_id=_uuid(90))
        )
    with pytest.raises(TerminalDeploymentCohortError, match="absent"):
        cohort.require_request_plan(
            dataclasses.replace(
                plan,
                decoder_service_id="decode-z",
                decoder_launch_instance_id=_uuid(99),
            )
        )


def test_request_plan_contains_only_its_exact_local_members() -> None:
    """Expose explicit request subset validation for local composition."""

    cohort = _cohort()
    plan = _request_plan(cohort)
    prefill = cohort.require_local_service(
        service_id=cohort.prefill.service_id,
        role=TerminalDeploymentRole.PREFILL,
        launch_instance_id=cohort.prefill.launch_instance_id,
        tensor_parallel_size=cohort.prefill.tensor_parallel_size,
        origin=cohort.prefill.origin,
        bootstrap_endpoint=cohort.prefill.bootstrap_endpoint,
    )
    selected_decoder = cohort.require_local_service(
        service_id=cohort.decoders[0].service_id,
        role=TerminalDeploymentRole.DECODE,
        launch_instance_id=cohort.decoders[0].launch_instance_id,
        tensor_parallel_size=cohort.decoders[0].tensor_parallel_size,
        origin=cohort.decoders[0].origin,
        bootstrap_endpoint=None,
    )
    other_decoder = cohort.require_local_service(
        service_id=cohort.decoders[1].service_id,
        role=TerminalDeploymentRole.DECODE,
        launch_instance_id=cohort.decoders[1].launch_instance_id,
        tensor_parallel_size=cohort.decoders[1].tensor_parallel_size,
        origin=cohort.decoders[1].origin,
        bootstrap_endpoint=None,
    )

    assert plan.contains(prefill)
    assert plan.contains(selected_decoder)
    assert not plan.contains(other_decoder)


def test_cohort_rejects_order_and_identity_collisions() -> None:
    """Keep decoder order and every launcher identity collision-free."""

    cohort = _cohort()
    with pytest.raises(ValueError, match="canonical service_id order"):
        dataclasses.replace(cohort, decoders=tuple(reversed(cohort.decoders)))
    duplicate_launch = dataclasses.replace(
        cohort.decoders[0],
        launch_instance_id=cohort.prefill.launch_instance_id,
    )
    with pytest.raises(ValueError, match="launch_instance_id"):
        dataclasses.replace(
            cohort,
            decoders=(duplicate_launch, cohort.decoders[1]),
        )
    duplicate_origin = dataclasses.replace(
        cohort.decoders[0],
        origin=cohort.prefill.origin,
    )
    with pytest.raises(ValueError, match="origin"):
        dataclasses.replace(
            cohort,
            decoders=(duplicate_origin, cohort.decoders[1]),
        )


def test_decoder_rejects_unknown_duplicate_and_noncanonical_json() -> None:
    """Give every accepted static cohort exactly one byte representation."""

    payload = encode_terminal_deployment_cohort(_cohort())
    decoded = json.loads(payload)
    decoded["surprise"] = True
    unknown = json.dumps(decoded, separators=(",", ":")).encode()
    with pytest.raises(TerminalDeploymentCohortError, match="field set"):
        decode_terminal_deployment_cohort(unknown)

    duplicate = payload.replace(
        b'{"schema":',
        b'{"schema":"packed-terminal-deployment-cohort-v1","schema":',
        1,
    )
    with pytest.raises(TerminalDeploymentCohortError, match="duplicate JSON field"):
        decode_terminal_deployment_cohort(duplicate)

    with pytest.raises(TerminalDeploymentCohortError, match="not canonical"):
        decode_terminal_deployment_cohort(payload + b"\n")
    with pytest.raises(TerminalDeploymentCohortError, match="bounded size"):
        decode_terminal_deployment_cohort(
            b"{" + b" " * MAX_TERMINAL_DEPLOYMENT_COHORT_BYTES
        )


def test_loader_requires_regular_canonical_artifact_and_exact_digest(
    tmp_path: Path,
) -> None:
    """Bind filesystem loading to launcher-attested canonical bytes."""

    payload = encode_terminal_deployment_cohort(_cohort())
    path = tmp_path / "cohort.json"
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).digest()

    assert load_terminal_deployment_cohort(path, digest) == _cohort()
    with pytest.raises(TerminalDeploymentCohortError, match="digest differs"):
        load_terminal_deployment_cohort(path, bytes.fromhex("ff" * 32))

    link = tmp_path / "cohort-link.json"
    os.symlink(path, link)
    with pytest.raises(TerminalDeploymentCohortError, match="opened safely"):
        load_terminal_deployment_cohort(link, digest)


def test_decoder_rejects_bootstrap_and_boolean_integer_drift() -> None:
    """Keep decoder roles and JSON booleans out of trusted static facts."""

    cohort = _cohort()
    decoder = cohort.decoders[0]
    with pytest.raises(TerminalDeploymentCohortError, match="differs"):
        cohort.require_local_service(
            service_id=decoder.service_id,
            role=TerminalDeploymentRole.DECODE,
            launch_instance_id=decoder.launch_instance_id,
            tensor_parallel_size=decoder.tensor_parallel_size,
            origin=decoder.origin,
            bootstrap_endpoint=cohort.prefill.bootstrap_endpoint,
        )

    payload = encode_terminal_deployment_cohort(cohort)
    decoded = json.loads(payload)
    decoded["prefill"]["tensor_parallel_size"] = True
    invalid = json.dumps(decoded, separators=(",", ":")).encode()
    with pytest.raises(TerminalDeploymentCohortError, match="integer"):
        decode_terminal_deployment_cohort(invalid)
