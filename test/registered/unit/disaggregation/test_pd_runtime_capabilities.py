import dataclasses
import sys

import pytest

from sglang.srt.disaggregation.runtime_capabilities import (
    PD_RUNTIME_CAPABILITIES_FIELD,
    PdProcessRuntimeCapabilities,
    validate_scheduler_runtime_capabilities,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_runtime_capabilities_round_trip() -> None:
    """The scheduler handshake preserves canonical runtime-owned values."""

    capabilities = PdProcessRuntimeCapabilities(
        kv_dtype="bfloat16",
        page_size=64,
        kv_transfer_protocol="packed-v4",
        prepared_grant_protocol="control-v1",
    )

    assert capabilities.kv_dtype == "bf16"
    assert capabilities.advertises_pd_process
    assert (
        PdProcessRuntimeCapabilities.from_dict(capabilities.to_dict()) == capabilities
    )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("kv_dtype", "auto", "runtime KV dtype"),
        ("page_size", 0, "positive page_size"),
        ("kv_transfer_protocol", "configured-nixl", "unknown KV-transfer"),
        ("prepared_grant_protocol", "configured-http", "unknown prepared-grant"),
    ),
)
def test_runtime_capabilities_reject_unattested_values(
    field: str,
    value: object,
    error: str,
) -> None:
    """Configuration labels cannot masquerade as initialized protocols."""

    capabilities = PdProcessRuntimeCapabilities(
        kv_dtype="bf16",
        page_size=64,
        kv_transfer_protocol=None,
        prepared_grant_protocol=None,
    )

    with pytest.raises(ValueError, match=error):
        dataclasses.replace(capabilities, **{field: value})


def test_runtime_capabilities_reject_schema_drift() -> None:
    """The parent process rejects incomplete scheduler capability reports."""

    with pytest.raises(ValueError, match="fields"):
        PdProcessRuntimeCapabilities.from_dict(
            {
                "kv_dtype": "bf16",
                "page_size": 64,
                "kv_transfer_protocol": None,
            }
        )


def test_scheduler_ranks_must_agree() -> None:
    """A process generation has one capability contract across all ranks."""

    capabilities = PdProcessRuntimeCapabilities(
        kv_dtype="bf16",
        page_size=64,
        kv_transfer_protocol=None,
        prepared_grant_protocol=None,
    ).to_dict()
    scheduler_infos = [
        {PD_RUNTIME_CAPABILITIES_FIELD: capabilities},
        {PD_RUNTIME_CAPABILITIES_FIELD: dict(capabilities)},
    ]

    assert validate_scheduler_runtime_capabilities(scheduler_infos) == capabilities

    scheduler_infos[1][PD_RUNTIME_CAPABILITIES_FIELD] = {
        **capabilities,
        "page_size": 32,
    }
    with pytest.raises(RuntimeError, match="disagree"):
        validate_scheduler_runtime_capabilities(scheduler_infos)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
