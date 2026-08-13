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
    TerminalDeploymentLocalService,
    TerminalDeploymentRequestPlan,
    TerminalDeploymentRole,
    TerminalDeploymentService,
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


def _service(
    service_id: str,
    role: TerminalDeploymentRole,
    marker: int,
    tp_size: int,
    port: int,
) -> TerminalDeploymentService:
    """Build one exact launcher service fixture.

    :param service_id: Stable service name.
    :param role: Prefill or decode role.
    :param marker: Launch UUID marker.
    :param tp_size: Complete service TP width.
    :param port: Exact loopback HTTP port.
    :returns: Valid service identity.
    """

    return TerminalDeploymentService(
        service_id=service_id,
        role=role,
        launch_instance_id=_uuid(marker),
        tensor_parallel_size=tp_size,
        origin=f"http://127.0.0.1:{port}",
        port=port,
    )


def _cohort() -> TerminalDeploymentCohort:
    """Build one immutable per-group deployment fixture.

    :returns: Valid deployment cohort.
    """

    return TerminalDeploymentCohort(
        group_id="group-a",
        model_fingerprint="11" * 32,
        logical_kv_layout_fingerprint="22" * 32,
        bootstrap_endpoint=TerminalDeploymentBootstrapEndpoint(
            host="gemma-dev-1",
            port=32150,
        ),
        services=(
            _service("prefill-a", TerminalDeploymentRole.PREFILL, 2, 2, 32001),
            _service("decode-a", TerminalDeploymentRole.DECODE, 3, 1, 32002),
            _service("decode-b", TerminalDeploymentRole.DECODE, 4, 1, 32003),
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


def _require_local(
    cohort: TerminalDeploymentCohort,
    service: TerminalDeploymentService,
) -> TerminalDeploymentLocalService:
    """Select one exact service from its fixture cohort.

    :param cohort: Owning deployment cohort.
    :param service: Exact cohort member.
    :returns: Validated local membership.
    """

    bootstrap = (
        cohort.bootstrap_endpoint
        if service.role is TerminalDeploymentRole.PREFILL
        else None
    )
    return cohort.require_local_service(
        service_id=service.service_id,
        role=service.role,
        launch_instance_id=service.launch_instance_id,
        tensor_parallel_size=service.tensor_parallel_size,
        origin=service.origin,
        port=service.port,
        bootstrap_endpoint=bootstrap,
    )


def test_cohort_matches_launcher_schema_bytes() -> None:
    """Preserve the exact launcher-owned schema and field ordering."""

    cohort = _cohort()
    payload = encode_terminal_deployment_cohort(cohort)
    expected = {
        "schema": "pd-terminal-deployment-cohort-v1",
        "group_id": "group-a",
        "model_fingerprint": "11" * 32,
        "logical_kv_layout_fingerprint": "22" * 32,
        "bootstrap_endpoint": {"host": "gemma-dev-1", "port": 32150},
        "services": [
            {
                "id": "prefill-a",
                "role": "prefill",
                "launch_instance_id": str(_uuid(2)),
                "tensor_parallel_size": 2,
                "origin": "http://127.0.0.1:32001",
                "port": 32001,
            },
            {
                "id": "decode-a",
                "role": "decode",
                "launch_instance_id": str(_uuid(3)),
                "tensor_parallel_size": 1,
                "origin": "http://127.0.0.1:32002",
                "port": 32002,
            },
            {
                "id": "decode-b",
                "role": "decode",
                "launch_instance_id": str(_uuid(4)),
                "tensor_parallel_size": 1,
                "origin": "http://127.0.0.1:32003",
                "port": 32003,
            },
        ],
    }

    assert json.loads(payload) == expected
    assert payload == json.dumps(expected, separators=(",", ":")).encode()
    assert decode_terminal_deployment_cohort(payload) == cohort
    assert cohort.digest == hashlib.sha256(payload).digest()
    assert payload.startswith(
        b'{"schema":"' + TERMINAL_DEPLOYMENT_COHORT_SCHEMA.encode() + b'"'
    )


def test_local_service_requires_exact_static_membership() -> None:
    """Bind service name, generation, role, width, origin, and bootstrap."""

    cohort = _cohort()
    prefill = _require_local(cohort, cohort.prefill)
    decoder = _require_local(cohort, cohort.decoders[1])

    assert prefill.service == cohort.prefill
    assert decoder.service == cohort.decoders[1]
    with pytest.raises(TerminalDeploymentCohortError, match="differs"):
        cohort.require_local_service(
            service_id=cohort.prefill.service_id,
            role=cohort.prefill.role,
            launch_instance_id=cohort.prefill.launch_instance_id,
            tensor_parallel_size=4,
            origin=cohort.prefill.origin,
            port=cohort.prefill.port,
            bootstrap_endpoint=cohort.bootstrap_endpoint,
        )
    with pytest.raises(TerminalDeploymentCohortError, match="absent"):
        cohort.require_local_service(
            service_id="decode-z",
            role=TerminalDeploymentRole.DECODE,
            launch_instance_id=_uuid(99),
            tensor_parallel_size=1,
            origin="http://127.0.0.1:32099",
            port=32099,
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
    prefill = _require_local(cohort, cohort.prefill)
    selected_decoder = _require_local(cohort, cohort.decoders[0])
    other_decoder = _require_local(cohort, cohort.decoders[1])

    assert plan.contains(prefill)
    assert plan.contains(selected_decoder)
    assert not plan.contains(other_decoder)


def test_cohort_rejects_order_and_identity_collisions() -> None:
    """Keep prefill first and every launcher identity collision-free."""

    cohort = _cohort()
    with pytest.raises(ValueError, match="prefill before"):
        dataclasses.replace(cohort, services=tuple(reversed(cohort.services)))
    duplicate_launch = dataclasses.replace(
        cohort.decoders[0],
        launch_instance_id=cohort.prefill.launch_instance_id,
    )
    with pytest.raises(ValueError, match="launch_instance_id"):
        dataclasses.replace(
            cohort,
            services=(cohort.prefill, duplicate_launch, cohort.decoders[1]),
        )
    duplicate_origin = dataclasses.replace(
        cohort.decoders[0],
        origin=cohort.prefill.origin,
        port=cohort.prefill.port,
    )
    with pytest.raises(ValueError, match="origin"):
        dataclasses.replace(
            cohort,
            services=(cohort.prefill, duplicate_origin, cohort.decoders[1]),
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
        b'{"schema":"pd-terminal-deployment-cohort-v1","schema":',
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
    with pytest.raises(TerminalDeploymentCohortError, match="cannot claim"):
        cohort.require_local_service(
            service_id=decoder.service_id,
            role=decoder.role,
            launch_instance_id=decoder.launch_instance_id,
            tensor_parallel_size=decoder.tensor_parallel_size,
            origin=decoder.origin,
            port=decoder.port,
            bootstrap_endpoint=cohort.bootstrap_endpoint,
        )

    payload = encode_terminal_deployment_cohort(cohort)
    decoded = json.loads(payload)
    decoded["services"][0]["tensor_parallel_size"] = True
    invalid = json.dumps(decoded, separators=(",", ":")).encode()
    with pytest.raises(TerminalDeploymentCohortError, match="integer"):
        decode_terminal_deployment_cohort(invalid)
