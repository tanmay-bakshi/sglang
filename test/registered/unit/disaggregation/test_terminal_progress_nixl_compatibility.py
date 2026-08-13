import os
import pathlib
import subprocess

import pytest
from nixl import _bindings as nixl_bindings
from nixl._api import nixl_agent
from sglang.srt.disaggregation.terminal_progress.nixl_adapter import (
    QUALIFIED_NIXL_REVISION,
    NixlTerminalEventAdapter,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

QUALIFIED_SOURCE_ENV = "SGLANG_QUALIFIED_NIXL_SOURCE_ROOT"


def _qualified_source_root() -> pathlib.Path:
    """Resolve the exact source tree backing this compatibility run.

    :returns: Qualified NIXL source root.
    """

    source_root = os.environ.get(QUALIFIED_SOURCE_ENV)
    if source_root is None:
        pytest.skip(f"{QUALIFIED_SOURCE_ENV} is required for compatibility testing")
    return pathlib.Path(source_root).resolve(strict=True)


def test_adapter_binds_exact_qualified_nixl_channel_surface() -> None:
    source_root = _qualified_source_root()
    revision = subprocess.run(
        ("git", "-C", str(source_root), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    assert revision == QUALIFIED_NIXL_REVISION

    config = nixl_bindings.nixlAgentConfig()
    config.useProgThread = False
    config.useListenThread = False
    native_agent = nixl_bindings.nixlAgent(
        "sglang-terminal-adapter-compatibility",
        config,
    )
    owner = object.__new__(nixl_agent)
    owner.agent = native_agent
    owner._terminal_event_channel = None
    owner._leaked_xfer_handles = []

    adapter = NixlTerminalEventAdapter.from_nixl_agent(owner, capacity=8)
    assert adapter.fileno() >= 0
    assert adapter.query_inventory().capacity == 8
    assert adapter.drain().events == ()
    assert adapter.close().is_clean_closed
