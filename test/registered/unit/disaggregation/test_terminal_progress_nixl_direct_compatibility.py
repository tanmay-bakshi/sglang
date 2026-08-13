import os
import pathlib
import subprocess

import pytest
from nixl._api import nixl_agent, terminal_owner_producer_abi
from sglang.srt.disaggregation.terminal_progress.nixl_adapter import (
    QUALIFIED_NIXL_DIRECT_OWNER_REVISION,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

QUALIFIED_DIRECT_SOURCE_ENV = "SGLANG_QUALIFIED_NIXL_DIRECT_SOURCE_ROOT"


def _qualified_source_root() -> pathlib.Path:
    """Resolve the exact direct-owner source tree backing this run.

    :returns: Qualified NIXL source root.
    """

    source_root = os.environ.get(QUALIFIED_DIRECT_SOURCE_ENV)
    if source_root is None:
        pytest.skip(
            f"{QUALIFIED_DIRECT_SOURCE_ENV} is required for compatibility testing"
        )
    return pathlib.Path(source_root).resolve(strict=True)


def test_direct_adapter_binds_exact_sealed_owner_producer_surface() -> None:
    source_root = _qualified_source_root()
    revision = subprocess.run(
        ("git", "-C", str(source_root), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()

    assert revision == QUALIFIED_NIXL_DIRECT_OWNER_REVISION
    assert callable(nixl_agent.create_terminal_owner_producer)
    assert callable(nixl_agent.subscribe_xfer_terminal_owner)
    assert callable(nixl_agent.take_xfer_completion_receipt)
    assert terminal_owner_producer_abi() == {
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
        "header_sha256": (
            "f8be2fe5e2f92a78f7cc51f133102f16ce9540fe0159b9d92c73ee855240b297"
        ),
    }
