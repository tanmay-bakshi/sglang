import types
import unittest
import uuid
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from sglang.srt.entrypoints.decode_reservation_control import (
    attach_native_generate_request,
    handle_attached_openai_request,
    install_decode_reservation_routes,
)
from sglang.srt.entrypoints.openai.protocol import CompletionRequest
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.utils.auth import add_api_key_middleware
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _ControlManager:
    server_args: types.SimpleNamespace
    bound_authorization: str | None
    bound_body: bytes | None
    attached_request: GenerateReqInput | None
    reserve_response: dict[str, object]

    def __init__(self) -> None:
        self.server_args = types.SimpleNamespace(api_key="process-api-key")
        self.bound_authorization = None
        self.bound_body = None
        self.attached_request = None
        self.reserve_response = {"state": "prepared"}

    async def bind_decode_reservation(
        self,
        grant_id: uuid.UUID,
        authorization_header: str | None,
        request_body: bytes,
    ) -> dict[str, object]:
        self.bound_authorization = authorization_header
        self.bound_body = request_body
        return {
            "operation": "bind",
            "state": "prepared",
            "grant_id": str(grant_id),
        }

    async def reserve_decode_reservation(
        self,
        *,
        attempt: object,
        attempt_wire: dict[str, object],
        obj: GenerateReqInput,
    ) -> dict[str, object]:
        del attempt, attempt_wire, obj
        return self.reserve_response

    async def attach_decode_inference(
        self,
        inference_route: str,
        request_body: bytes,
    ) -> GenerateReqInput | None:
        del inference_route, request_body
        return self.attached_request


class DecodeReservationControlHttpTest(unittest.TestCase):
    def test_grant_bearer_bypasses_process_middleware_and_preserves_body(self) -> None:
        manager = _ControlManager()
        app = FastAPI()
        install_decode_reservation_routes(app, lambda: manager)

        @app.post("/normal")
        async def normal_route() -> dict[str, bool]:
            return {"ok": True}

        add_api_key_middleware(
            app,
            api_key="process-api-key",
            admin_api_key=None,
        )
        request_body = b'{ "model": "m", "prompt": "exact bytes" }'
        grant_id = uuid.uuid4()

        with TestClient(app) as client:
            response = client.post(
                (f"/_internal/pd/v1/decode-reservations/{grant_id}/bind"),
                content=request_body,
                headers={
                    "Authorization": "Bearer request-scoped-grant-token",
                    "Content-Type": "application/json",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(manager.bound_body, request_body)
        self.assertEqual(
            manager.bound_authorization,
            "Bearer request-scoped-grant-token",
        )

        with TestClient(app) as client:
            normal_response = client.post(
                "/normal",
                headers={
                    "Authorization": "Bearer request-scoped-grant-token",
                },
            )
        self.assertEqual(normal_response.status_code, 401)

    def test_reserve_refusal_status_preserves_authoritative_receipt(self) -> None:
        for disposition, expected_status in (
            ("retry_same_decoder", 429),
            ("retry_another_decoder", 429),
            ("terminal", 409),
        ):
            with self.subTest(disposition=disposition):
                manager = _ControlManager()
                manager.reserve_response = {
                    "state": "refused",
                    "operation": "reserve",
                    "disposition": disposition,
                    "receipt_digest": "ab" * 32,
                }
                app = FastAPI()
                install_decode_reservation_routes(
                    app,
                    lambda manager=manager: manager,
                )
                add_api_key_middleware(
                    app,
                    api_key="process-api-key",
                    admin_api_key=None,
                )
                normalized_request = GenerateReqInput(text="prompt")
                with (
                    patch(
                        (
                            "sglang.srt.entrypoints.decode_reservation_control."
                            "DecodeReservationAttempt.from_value"
                        ),
                        return_value=types.SimpleNamespace(),
                    ),
                    patch(
                        (
                            "sglang.srt.entrypoints.decode_reservation_control."
                            "_normalize_reserve_request"
                        ),
                        return_value=normalized_request,
                    ),
                    TestClient(app) as client,
                ):
                    response = client.post(
                        "/_internal/pd/v1/decode-reservations/reserve",
                        content=b"{}",
                        headers={
                            "Authorization": "Bearer process-api-key",
                            "Content-Type": "application/json",
                        },
                    )

                self.assertEqual(response.status_code, expected_status)
                self.assertEqual(response.json(), manager.reserve_response)


class DecodeReservationAttachedHttpTest(unittest.IsolatedAsyncioTestCase):
    async def test_native_attachment_returns_retained_request(self) -> None:
        manager = _ControlManager()
        retained = GenerateReqInput(text="retained")
        manager.attached_request = retained
        parsed = GenerateReqInput(text="parsed again")
        raw_request = AsyncMock(spec=Request)
        raw_request.body.return_value = b'{"text":"retained"}'

        result = await attach_native_generate_request(
            manager,
            raw_request,
            parsed,
        )

        self.assertIs(result, retained)

    async def test_openai_attachment_skips_conversion_and_uses_response_path(
        self,
    ) -> None:
        manager = _ControlManager()
        retained = GenerateReqInput(text="retained")
        manager.attached_request = retained
        serving = types.SimpleNamespace(
            tokenizer_manager=manager,
            _handle_streaming_request=AsyncMock(),
            _handle_non_streaming_request=AsyncMock(return_value={"attached": True}),
        )
        protocol_request = CompletionRequest(model="model", prompt="prompt")
        raw_request = AsyncMock(spec=Request)
        raw_request.body.return_value = b'{"model":"model","prompt":"prompt"}'

        attached, response = await handle_attached_openai_request(
            serving,
            protocol_request,
            raw_request,
            "/v1/completions",
        )

        self.assertTrue(attached)
        self.assertEqual(response, {"attached": True})
        serving._handle_non_streaming_request.assert_awaited_once_with(
            retained,
            protocol_request,
            raw_request,
        )
        serving._handle_streaming_request.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
