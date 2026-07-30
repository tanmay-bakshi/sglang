import dataclasses
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.pd_disaggregation_hook import (
    PdProcessRuntimeCapabilities,
    build_pd_process_advertisement,
)
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64
_CAPABILITIES = PdProcessRuntimeCapabilities(
    kv_dtype="bfloat16",
    page_size=64,
    kv_transfer_protocol="packed-v4",
    prepared_grant_protocol="control-v1",
)


def _pd_args(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "api_key": "secret",
        "tokenizer_worker_num": 1,
        "disaggregation_mode": "prefill",
        "disaggregation_transfer_backend": "mooncake",
        "disaggregation_decode_enable_radix_cache": False,
        "disaggregation_decode_extra_slots": None,
        "max_running_requests": None,
        "dp_size": 1,
        "tp_size": 4,
        "pd_model_fingerprint": _DIGEST_A,
        "pd_logical_kv_layout_fingerprint": _DIGEST_B,
        "pd_prefill_bootstrap_advertise_host": "10.20.30.40",
        "disaggregation_bootstrap_port": 8998,
        "kv_cache_dtype": "bfloat16",
        "page_size": 64,
        "launch_instance_id": str(uuid.uuid4()),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestLaunchInstanceId(CustomTestCase):
    def test_launch_uuid_is_stable_and_unique(self):
        first = ServerArgs(model_path="dummy")
        second = ServerArgs(model_path="dummy")

        self.assertEqual(first.launch_instance_id, first.launch_instance_id)
        self.assertNotEqual(first.launch_instance_id, second.launch_instance_id)
        self.assertEqual(
            str(uuid.UUID(first.launch_instance_id)), first.launch_instance_id
        )

    def test_port_args_reuse_server_generation(self):
        server_args = ServerArgs(model_path="dummy")

        with patch("sglang.srt.server_args.tempfile.NamedTemporaryFile") as temporary:
            temporary.return_value.name = "/tmp/sglang-pd-test"
            ports = PortArgs.init_new(server_args)

        self.assertEqual(ports.instance_id, server_args.launch_instance_id)

    def test_launch_uuid_has_no_cli_surface(self):
        field = next(
            field
            for field in dataclasses.fields(ServerArgs)
            if field.name == "launch_instance_id"
        )

        self.assertTrue(field.default_factory is not dataclasses.MISSING)


class TestPdProcessAdvertisement(CustomTestCase):
    def test_prefill_advertisement_shape(self):
        advertisement = build_pd_process_advertisement(
            _pd_args(), runtime_capabilities=_CAPABILITIES
        )

        self.assertEqual(
            advertisement,
            {
                "schema": "v1",
                "launch_instance_id": advertisement["launch_instance_id"],
                "role": "prefill",
                "tensor_parallel_size": 4,
                "data_parallel_size": 1,
                "model_fingerprint": _DIGEST_A,
                "logical_kv_layout_fingerprint": _DIGEST_B,
                "kv_dtype": "bf16",
                "page_size": 64,
                "kv_transfer_protocol": "packed-v4",
                "prepared_grant_protocol": "control-v1",
                "prefill_bootstrap_endpoint": {
                    "host": "10.20.30.40",
                    "port": 8998,
                },
            },
        )

    def test_decode_advertisement_has_no_bootstrap_endpoint(self):
        advertisement = build_pd_process_advertisement(
            _pd_args(
                disaggregation_mode="decode",
                tp_size=1,
                pd_prefill_bootstrap_advertise_host=None,
            ),
            runtime_capabilities=_CAPABILITIES,
        )

        self.assertEqual(advertisement["role"], "decode")
        self.assertIsNone(advertisement["prefill_bootstrap_endpoint"])

    def test_non_pd_server_has_no_advertisement(self):
        self.assertIsNone(
            build_pd_process_advertisement(
                _pd_args(disaggregation_mode="null"),
                runtime_capabilities=None,
            )
        )

    def test_pd_server_has_no_advertisement_without_runtime_capabilities(self):
        self.assertIsNone(
            build_pd_process_advertisement(_pd_args(), runtime_capabilities=None)
        )

    def test_missing_api_key_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "--api-key"):
            build_pd_process_advertisement(
                _pd_args(api_key=None), runtime_capabilities=_CAPABILITIES
            )

    def test_multiple_tokenizers_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "--tokenizer-worker-num 1"):
            build_pd_process_advertisement(
                _pd_args(tokenizer_worker_num=2),
                runtime_capabilities=_CAPABILITIES,
            )

    def test_invalid_fingerprints_are_rejected(self):
        for fingerprint in (None, "A" * 64, "a" * 63, "g" * 64):
            with self.subTest(fingerprint=fingerprint):
                with self.assertRaisesRegex(ValueError, "canonical lowercase"):
                    build_pd_process_advertisement(
                        _pd_args(pd_model_fingerprint=fingerprint),
                        runtime_capabilities=_CAPABILITIES,
                    )

    def test_local_prefill_hosts_are_rejected(self):
        for host in ("localhost", "api.localhost", "127.0.0.1", "[::1]"):
            with self.subTest(host=host):
                with self.assertRaisesRegex(ValueError, "non-local|localhost"):
                    build_pd_process_advertisement(
                        _pd_args(pd_prefill_bootstrap_advertise_host=host),
                        runtime_capabilities=_CAPABILITIES,
                    )

    def test_decode_bootstrap_host_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "decode processes"):
            build_pd_process_advertisement(
                _pd_args(disaggregation_mode="decode"),
                runtime_capabilities=_CAPABILITIES,
            )

    def test_runtime_kv_dtype_is_required(self):
        with self.assertRaisesRegex(ValueError, "runtime KV dtype"):
            build_pd_process_advertisement(
                _pd_args(),
                runtime_capabilities=dataclasses.replace(
                    _CAPABILITIES, kv_dtype="auto"
                ),
            )

    def test_resolved_page_size_is_required(self):
        with self.assertRaisesRegex(ValueError, "resolved positive page_size"):
            build_pd_process_advertisement(
                _pd_args(),
                runtime_capabilities=dataclasses.replace(_CAPABILITIES, page_size=0),
            )


if __name__ == "__main__":
    unittest.main()
