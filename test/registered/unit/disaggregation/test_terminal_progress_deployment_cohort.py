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
    TerminalDeploymentGroup,
    TerminalDeploymentLocalService,
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
    """Build a two-group immutable deployment fixture.

    :returns: Valid deployment cohort.
    """

    return TerminalDeploymentCohort(
        groups=(
            TerminalDeploymentGroup(
                group_id="group-a",
                launch_epoch=_uuid(1),
                prefill=TerminalDeploymentPrefill(
                    launch_instance_id=_uuid(2),
                    bootstrap_endpoint=TerminalDeploymentBootstrapEndpoint(
                        host="gemma-dev-1",
                        port=32150,
                    ),
                    tensor_parallel_size=2,
                ),
                decoders=(
                    TerminalDeploymentDecoder(
                        launch_instance_id=_uuid(3),
                        tensor_parallel_size=1,
                    ),
                    TerminalDeploymentDecoder(
                        launch_instance_id=_uuid(4),
                        tensor_parallel_size=1,
                    ),
                ),
            ),
            TerminalDeploymentGroup(
                group_id="group-b",
                launch_epoch=_uuid(5),
                prefill=TerminalDeploymentPrefill(
                    launch_instance_id=_uuid(6),
                    bootstrap_endpoint=TerminalDeploymentBootstrapEndpoint(
                        host="gemma-dev-2",
                        port=32250,
                    ),
                    tensor_parallel_size=4,
                ),
                decoders=(
                    TerminalDeploymentDecoder(
                        launch_instance_id=_uuid(7),
                        tensor_parallel_size=2,
                    ),
                ),
            ),
        )
    )


def _request_plan(
    cohort: TerminalDeploymentCohort | None = None,
) -> TerminalDeploymentRequestPlan:
    """Select the first decoder in the first fixture group.

    :param cohort: Optional cohort whose exact objects should be selected.
    :returns: Valid request-selected subset.
    """

    selected = _cohort() if cohort is None else cohort
    group = selected.groups[0]
    decoder = group.decoders[0]
    return TerminalDeploymentRequestPlan(
        group_id=group.group_id,
        group_launch_epoch=group.launch_epoch,
        prefill_launch_instance_id=group.prefill.launch_instance_id,
        prefill_bootstrap_endpoint=group.prefill.bootstrap_endpoint,
        prefill_tensor_parallel_size=group.prefill.tensor_parallel_size,
        decoder_launch_instance_id=decoder.launch_instance_id,
        decoder_tensor_parallel_size=decoder.tensor_parallel_size,
    )


def test_cohort_round_trip_is_canonical_and_hash_bound() -> None:
    """Preserve every deployment fact under one deterministic digest."""

    cohort = _cohort()
    payload = encode_terminal_deployment_cohort(cohort)

    assert decode_terminal_deployment_cohort(payload) == cohort
    assert cohort.digest == hashlib.sha256(payload).digest()
    assert payload.startswith(
        b'{"schema":"' + TERMINAL_DEPLOYMENT_COHORT_SCHEMA.encode() + b'"'
    )
    assert not payload.endswith(b"\n")


def test_local_service_requires_exact_group_membership() -> None:
    """Bind local role, launch generation, TP width, and bootstrap exactly."""

    cohort = _cohort()
    group = cohort.groups[0]
    prefill = cohort.require_local_service(
        group_id=group.group_id,
        role=TerminalDeploymentRole.PREFILL,
        launch_instance_id=group.prefill.launch_instance_id,
        tensor_parallel_size=group.prefill.tensor_parallel_size,
        bootstrap_endpoint=group.prefill.bootstrap_endpoint,
    )
    decoder_record = group.decoders[1]
    decoder = cohort.require_local_service(
        group_id=group.group_id,
        role=TerminalDeploymentRole.DECODE,
        launch_instance_id=decoder_record.launch_instance_id,
        tensor_parallel_size=decoder_record.tensor_parallel_size,
        bootstrap_endpoint=None,
    )

    assert prefill.group_launch_epoch == group.launch_epoch
    assert decoder.launch_instance_id == decoder_record.launch_instance_id
    with pytest.raises(TerminalDeploymentCohortError, match="differs"):
        cohort.require_local_service(
            group_id=group.group_id,
            role=TerminalDeploymentRole.PREFILL,
            launch_instance_id=group.prefill.launch_instance_id,
            tensor_parallel_size=4,
            bootstrap_endpoint=group.prefill.bootstrap_endpoint,
        )
    with pytest.raises(TerminalDeploymentCohortError, match="absent"):
        cohort.require_local_service(
            group_id=group.group_id,
            role=TerminalDeploymentRole.DECODE,
            launch_instance_id=_uuid(99),
            tensor_parallel_size=1,
            bootstrap_endpoint=None,
        )


def test_request_plan_must_be_an_exact_same_group_subset() -> None:
    """Reject stale epochs, cross-group decoders, and route drift."""

    cohort = _cohort()
    plan = _request_plan(cohort)
    group, prefill, decoder = cohort.require_request_plan(plan)

    assert group == cohort.groups[0]
    assert prefill == group.prefill
    assert decoder == group.decoders[0]
    with pytest.raises(TerminalDeploymentCohortError, match="stale"):
        cohort.require_request_plan(
            dataclasses.replace(plan, group_launch_epoch=_uuid(90))
        )
    with pytest.raises(TerminalDeploymentCohortError, match="prefill differs"):
        cohort.require_request_plan(
            dataclasses.replace(
                plan,
                prefill_bootstrap_endpoint=TerminalDeploymentBootstrapEndpoint(
                    host="gemma-dev-1",
                    port=32151,
                ),
            )
        )
    with pytest.raises(TerminalDeploymentCohortError, match="decoder is absent"):
        cohort.require_request_plan(
            dataclasses.replace(
                plan,
                decoder_launch_instance_id=cohort.groups[1]
                .decoders[0]
                .launch_instance_id,
                decoder_tensor_parallel_size=2,
            )
        )


def test_request_plan_contains_only_its_exact_local_members() -> None:
    """Expose an explicit membership check for local request composition."""

    cohort = _cohort()
    group = cohort.groups[0]
    plan = _request_plan(cohort)
    prefill = cohort.require_local_service(
        group_id=group.group_id,
        role=TerminalDeploymentRole.PREFILL,
        launch_instance_id=group.prefill.launch_instance_id,
        tensor_parallel_size=group.prefill.tensor_parallel_size,
        bootstrap_endpoint=group.prefill.bootstrap_endpoint,
    )
    selected_decoder = cohort.require_local_service(
        group_id=group.group_id,
        role=TerminalDeploymentRole.DECODE,
        launch_instance_id=group.decoders[0].launch_instance_id,
        tensor_parallel_size=group.decoders[0].tensor_parallel_size,
        bootstrap_endpoint=None,
    )
    other_decoder = cohort.require_local_service(
        group_id=group.group_id,
        role=TerminalDeploymentRole.DECODE,
        launch_instance_id=group.decoders[1].launch_instance_id,
        tensor_parallel_size=group.decoders[1].tensor_parallel_size,
        bootstrap_endpoint=None,
    )

    assert plan.contains(prefill)
    assert plan.contains(selected_decoder)
    assert not plan.contains(other_decoder)


def test_cohort_rejects_identity_collisions_and_noncanonical_group_order() -> None:
    """Keep launch epochs and service generations deployment-global."""

    cohort = _cohort()
    with pytest.raises(ValueError, match="canonical group_id order"):
        dataclasses.replace(cohort, groups=tuple(reversed(cohort.groups)))
    duplicate_epoch = dataclasses.replace(
        cohort.groups[1],
        launch_epoch=cohort.groups[0].launch_epoch,
    )
    with pytest.raises(ValueError, match="launch_epoch"):
        dataclasses.replace(cohort, groups=(cohort.groups[0], duplicate_epoch))
    duplicate_service = dataclasses.replace(
        cohort.groups[1],
        decoders=(
            dataclasses.replace(
                cohort.groups[1].decoders[0],
                launch_instance_id=cohort.groups[0].decoders[0].launch_instance_id,
            ),
        ),
    )
    with pytest.raises(ValueError, match="deployment-global"):
        dataclasses.replace(cohort, groups=(cohort.groups[0], duplicate_service))


def test_decoder_rejects_unknown_duplicate_and_noncanonical_json() -> None:
    """Give one byte representation to every accepted deployment fact set."""

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
    """Bind filesystem loading to the launcher-attested canonical bytes."""

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


def test_local_membership_shape_rejects_decode_bootstrap() -> None:
    """Prevent a decoder claim from borrowing a prefill route."""

    with pytest.raises(ValueError, match="decode membership"):
        TerminalDeploymentLocalService(
            group_id="group-a",
            group_launch_epoch=_uuid(1),
            role=TerminalDeploymentRole.DECODE,
            launch_instance_id=_uuid(3),
            tensor_parallel_size=1,
            bootstrap_endpoint=TerminalDeploymentBootstrapEndpoint(
                host="gemma-dev-1",
                port=32150,
            ),
        )


def test_service_widths_reject_boolean_json_scalars() -> None:
    """Keep JSON booleans out of integer tensor-parallel fields."""

    payload = encode_terminal_deployment_cohort(_cohort())
    decoded = json.loads(payload)
    decoded["groups"][0]["prefill"]["tensor_parallel_size"] = True
    invalid = json.dumps(decoded, separators=(",", ":")).encode()

    with pytest.raises(TerminalDeploymentCohortError, match="integer"):
        decode_terminal_deployment_cohort(invalid)
