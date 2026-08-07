import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_components.request_receiver import (  # noqa: E402
    SchedulerRequestReceiver,
)

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class RecordingGroup:
    def __init__(self, name: str, calls: list[tuple[str, Any, int]]) -> None:
        self._name = name
        self._calls = calls

    def broadcast_object(self, value: Any, src: int = 0) -> Any:
        self._calls.append((self._name, value, src))
        return value


def make_receiver(
    *,
    calls: list[tuple[str, Any, int]],
    enable_dp_attention: bool,
    local_control: bool = False,
) -> SchedulerRequestReceiver:
    return SchedulerRequestReceiver(
        recv_from_tokenizer=None,
        recv_from_rpc=None,
        recv_skipper=None,
        input_blocker=None,
        mm_receiver=None,
        ps=SimpleNamespace(
            pp_rank=0,
            tp_size=2,
            attn_tp_rank=0,
            attn_tp_size=2,
            attn_cp_rank=0,
            attn_cp_size=2,
        ),
        tp_group=RecordingGroup("tp", calls),
        tp_cpu_group=object(),
        attn_tp_group=RecordingGroup("attn_tp", calls),
        attn_tp_cpu_group=object(),
        attn_cp_group=RecordingGroup("attn_cp", calls),
        attn_cp_cpu_group=object(),
        world_group=SimpleNamespace(cpu_group=object()),
        server_args=SimpleNamespace(
            enable_dp_attention=enable_dp_attention,
            enable_dp_attention_local_control_broadcast=local_control,
            is_ep_scale_joiner=False,
        ),
        model_config=SimpleNamespace(is_multimodal=False),
        max_recv_per_poll=-1,
        stream_output=lambda *args, **kwargs: None,
        get_last_batch=lambda: None,
    )


class TestRequestReceiverBroadcast(unittest.TestCase):
    def test_tensor_parallel_requests_use_group_broadcaster(self) -> None:
        calls: list[tuple[str, Any, int]] = []
        receiver = make_receiver(calls=calls, enable_dp_attention=False)
        requests: list[str] = ["request"]

        result = receiver._broadcast_reqs_across_ranks(requests)

        self.assertIs(result, requests)
        self.assertEqual(calls, [("tp", requests, 0)])

    def test_dp_attention_uses_matching_work_and_control_groups(self) -> None:
        calls: list[tuple[str, Any, int]] = []
        receiver = make_receiver(calls=calls, enable_dp_attention=True)
        work = object()
        control = object()

        with patch.object(
            SchedulerRequestReceiver,
            "_split_work_and_control_reqs",
            return_value=([work], [control]),
        ):
            result = receiver._broadcast_reqs_across_ranks([work, control])

        self.assertEqual(result, [work, control])
        self.assertEqual(
            calls,
            [
                ("attn_tp", [work], 0),
                ("attn_cp", [work], 0),
                ("tp", [control], 0),
            ],
        )

    def test_local_control_stays_with_attention_groups(self) -> None:
        calls: list[tuple[str, Any, int]] = []
        receiver = make_receiver(
            calls=calls,
            enable_dp_attention=True,
            local_control=True,
        )
        work = object()
        control = object()

        with patch.object(
            SchedulerRequestReceiver,
            "_split_work_and_control_reqs",
            return_value=([work], [control]),
        ):
            result = receiver._broadcast_reqs_across_ranks([work, control])

        self.assertEqual(result, [work, control])
        self.assertEqual(
            calls,
            [
                ("attn_tp", [work], 0),
                ("attn_cp", [work], 0),
                ("attn_tp", [control], 0),
                ("attn_cp", [control], 0),
            ],
        )


if __name__ == "__main__":
    unittest.main()
