from unittest.mock import MagicMock, patch

import pytest

from sglang.srt.distributed.device_communicators import custom_all_reduce_utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_resolve_numeric_visible_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7,3")

    assert custom_all_reduce_utils.resolve_physical_device_id(0) == 7
    assert custom_all_reduce_utils.resolve_physical_device_id(1) == 3


def test_resolve_uuid_visible_device(monkeypatch: pytest.MonkeyPatch) -> None:
    gpu_uuid = "GPU-e61900b2-5402-7f7e-e24c-887ab5d5287f"
    handle = object()
    nvml = MagicMock()
    nvml.nvmlDeviceGetHandleByUUID.return_value = handle
    nvml.nvmlDeviceGetIndex.return_value = 4
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", gpu_uuid)

    with (
        patch.object(custom_all_reduce_utils, "_is_cuda", True),
        patch.object(
            custom_all_reduce_utils,
            "pynvml",
            nvml,
            create=True,
        ),
    ):
        physical_device_id = custom_all_reduce_utils.resolve_physical_device_id(0)

    assert physical_device_id == 4
    nvml.nvmlInit.assert_called_once_with()
    nvml.nvmlDeviceGetHandleByUUID.assert_called_once_with(gpu_uuid)
    nvml.nvmlDeviceGetIndex.assert_called_once_with(handle)
    nvml.nvmlShutdown.assert_called_once_with()


def test_reject_missing_logical_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a")

    with pytest.raises(ValueError, match="is not present"):
        custom_all_reduce_utils.resolve_physical_device_id(1)
