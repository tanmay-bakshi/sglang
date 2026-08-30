import asyncio
import json
import os
import tempfile
import unittest

from sglang.srt import runtime_context
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.entrypoints.engine import Engine, SchedulerInitResult
from sglang.srt.entrypoints.grpc_bridge import RuntimeHandle
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestApiKeyFile(CustomTestCase):
    """Validate file-backed API key loading and its filesystem boundary."""

    def _write_secret(self, directory: str, value: str, mode: int = 0o600) -> str:
        """Create a secret fixture with explicit permissions.

        :param directory: Fixture directory.
        :param value: Secret file contents.
        :param mode: Resulting permission bits.
        :returns: Secret file path.
        """
        path = os.path.join(directory, "api-key")
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as secret_file:
            secret_file.write(value)
        os.chmod(path, mode)
        return path

    def test_secure_regular_file_is_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_secret(directory, "file-secret\n")

            server_args = ServerArgs(model_path="dummy", api_key_file=path)
            server_args.resolve_once()

            self.assertEqual(resolution_result(server_args, "api_key"), "file-secret")
            self.assertEqual(server_args.api_key_file, path)

    def test_inline_and_file_keys_are_mutually_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_secret(directory, "file-secret")

            server_args = ServerArgs(
                model_path="dummy",
                api_key="inline-secret",
                api_key_file=path,
            )
            with self.assertRaisesRegex(ValueError, "cannot be combined"):
                server_args.resolve_once()

    def test_group_or_other_permissions_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_secret(directory, "file-secret", mode=0o640)

            server_args = ServerArgs(model_path="dummy", api_key_file=path)
            with self.assertRaisesRegex(ValueError, "group or other"):
                server_args.resolve_once()

    def test_symbolic_link_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = self._write_secret(directory, "file-secret")
            link = os.path.join(directory, "api-key-link")
            os.symlink(target, link)

            server_args = ServerArgs(model_path="dummy", api_key_file=link)
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                server_args.resolve_once()

    def test_empty_and_multiline_files_are_rejected(self) -> None:
        for value, message in (("", "must not be empty"), ("a\nb\n", "one line")):
            with (
                self.subTest(value=value),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = self._write_secret(directory, value)
                server_args = ServerArgs(model_path="dummy", api_key_file=path)
                with self.assertRaisesRegex(ValueError, message):
                    server_args.resolve_once()


class TestServerArgumentSecretRedaction(CustomTestCase):
    """Ensure every diagnostic representation is credential-safe."""

    def setUp(self) -> None:
        runtime_context.reset_context()

    def tearDown(self) -> None:
        runtime_context.reset_context()

    @staticmethod
    def _secret_server_args() -> ServerArgs:
        server_args = ServerArgs(model_path="dummy")
        server_args.api_key = "api-secret"
        server_args.admin_api_key = "admin-secret"
        server_args.ssl_keyfile_password = "tls-secret"
        return server_args

    def test_repr_excludes_all_secret_values(self) -> None:
        representation = repr(self._secret_server_args())

        self.assertNotIn("api-secret", representation)
        self.assertNotIn("admin-secret", representation)
        self.assertNotIn("tls-secret", representation)

    def test_public_mapping_nulls_secrets_and_reports_configuration(self) -> None:
        public_values = self._secret_server_args().public_server_args_dict()

        for field_name in ("api_key", "admin_api_key", "ssl_keyfile_password"):
            self.assertIsNone(public_values[field_name])
            self.assertTrue(public_values[f"{field_name}_configured"])

    def test_runtime_context_resolved_mapping_remains_redacted(self) -> None:
        server_args = self._secret_server_args()
        runtime_context.get_context().set_server_args(server_args)
        runtime_context.get_context().override("test", page_size=64)

        public_values = runtime_context.get_context().resolved_server_args_dict()

        self.assertEqual(public_values["page_size"], 64)
        self.assertIsNone(public_values["api_key"])
        self.assertTrue(public_values["api_key_configured"])

    def test_grpc_server_info_remains_redacted(self) -> None:
        handle = object.__new__(RuntimeHandle)
        handle.tokenizer_manager = _TokenizerManagerStub(self._secret_server_args())
        handle.scheduler_info = {"max_req_input_len": 1024}
        handle.scheduler_info["api_key"] = "scheduler-secret"

        server_info = json.loads(handle.get_server_info())

        self.assertIsNone(server_info["api_key"])
        self.assertIsNone(server_info["admin_api_key"])
        self.assertIsNone(server_info["ssl_keyfile_password"])
        self.assertTrue(server_info["api_key_configured"])
        self.assertNotIn("scheduler-secret", json.dumps(server_info))

    def test_embedded_engine_server_info_remains_redacted(self) -> None:
        engine = object.__new__(Engine)
        engine.loop = asyncio.new_event_loop()
        engine.tokenizer_manager = _TokenizerManagerStub(self._secret_server_args())
        engine._scheduler_init_result = SchedulerInitResult(
            scheduler_infos=[
                {
                    "api_key": "scheduler-secret",
                }
            ]
        )
        try:
            server_info = engine.get_server_info()
        finally:
            engine.loop.close()

        self.assertIsNone(server_info["api_key"])
        self.assertIsNone(server_info["admin_api_key"])
        self.assertIsNone(server_info["ssl_keyfile_password"])
        self.assertTrue(server_info["api_key_configured"])
        self.assertNotIn("scheduler-secret", json.dumps(server_info))


class _TokenizerManagerStub:
    """Minimal embedded-engine tokenizer-manager boundary."""

    server_args: ServerArgs
    startup_time: dict[str, object] | None = None

    def __init__(self, server_args: ServerArgs) -> None:
        """
        :param server_args: Server arguments returned through engine diagnostics.
        """
        self.server_args = server_args

    async def get_internal_state(self) -> list[dict[str, object]]:
        """Return an empty scheduler-state fixture.

        :returns: Empty internal-state collection.
        """
        return []


if __name__ == "__main__":
    unittest.main()
