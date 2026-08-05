import asyncio
import unittest
import uuid
from array import array
from types import SimpleNamespace

from sglang.srt.disaggregation.decode_reservations import (
    DecodeInferenceAttachmentRegistry,
    DecodeInferenceAttachmentState,
    DecodeReservationConflictError,
)
from sglang.srt.managers.io_struct import (
    DecodeInferenceAttachReqInput,
    DecodeReservationBindReqInput,
    DecodeReservationControlReqOutput,
    DecodeReservationExpiryReqOutput,
    DecodeReservationPrepareReqInput,
    GenerateReqInput,
    TokenizedGenerateReqInput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.scheduler_components.request_receiver import (
    SchedulerRequestReceiver,
)
from sglang.srt.managers.tokenizer_manager import (
    DecodeReservationSchedulerError,
    TokenizerManager,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class DecodeReservationTokenizerIpcTest(unittest.IsolatedAsyncioTestCase):
    def test_expiry_output_round_trips_owning_tokenizer_route(self) -> None:
        cancelled_grant_id: str = str(uuid.uuid4())
        quarantined_grant_id: str = str(uuid.uuid4())
        output: DecodeReservationExpiryReqOutput = DecodeReservationExpiryReqOutput(
            http_worker_ipc="ipc:///tmp/tokenizer-worker",
            cancelled_grant_ids=(cancelled_grant_id,),
            quarantined_grant_ids=(quarantined_grant_id,),
        )

        decoded = msgpack_decode(msgpack_encode(output))

        self.assertIsInstance(decoded, DecodeReservationExpiryReqOutput)
        self.assertEqual(decoded.http_worker_ipc, output.http_worker_ipc)
        self.assertEqual(decoded.cancelled_grant_ids, (cancelled_grant_id,))
        self.assertEqual(decoded.quarantined_grant_ids, (quarantined_grant_id,))
        self.assertNotIn("request_body", repr(decoded))

    def test_scheduler_receiver_unwraps_every_prepared_request(self) -> None:
        expected_mm_inputs: tuple[dict[str, int], ...] = (
            {"request": 0},
            {"request": 1},
        )
        tokenized_requests: tuple[TokenizedGenerateReqInput, ...] = tuple(
            TokenizedGenerateReqInput(
                input_text=f"request-{index}",
                input_ids=array("l", [index]),
                input_embeds=None,
                mm_inputs=mm_inputs,
                token_type_ids=None,
                sampling_params=SamplingParams(max_new_tokens=1),
                return_logprob=False,
                logprob_start_len=-1,
                top_logprobs_num=0,
                token_ids_logprob=None,
                stream=False,
            )
            for index, mm_inputs in enumerate(expected_mm_inputs)
        )
        for tokenized_request in tokenized_requests:
            tokenized_request.wrap_pickle_fields()

        prepare: DecodeReservationPrepareReqInput = DecodeReservationPrepareReqInput(
            correlation_id="prepare-correlation",
            grant_id=str(uuid.uuid4()),
            attempt={},
            tokenized_requests=tokenized_requests,
        )
        receiver: SchedulerRequestReceiver = object.__new__(SchedulerRequestReceiver)

        receiver.unwrap_pickle_wrapper([prepare])

        self.assertEqual(
            tuple(request.mm_inputs for request in tokenized_requests),
            expected_mm_inputs,
        )

    def test_control_structs_round_trip_exact_bytes_and_tuple(self) -> None:
        prepare = DecodeReservationPrepareReqInput(
            correlation_id="prepare-correlation",
            grant_id=str(uuid.uuid4()),
            attempt={
                "schema_version": 1,
                "base_request_body_json": "prompt-secret",
            },
            tokenized_requests=(),
        )
        decoded_prepare = msgpack_decode(msgpack_encode(prepare))
        self.assertIsInstance(decoded_prepare, DecodeReservationPrepareReqInput)
        self.assertIsInstance(decoded_prepare.tokenized_requests, tuple)
        self.assertEqual(decoded_prepare.attempt, prepare.attempt)
        self.assertNotIn("prompt-secret", repr(decoded_prepare))

        request_body = b'{ "prompt": "exact bytes", "stream": false }'
        bind = DecodeReservationBindReqInput(
            correlation_id="bind-correlation",
            grant_id=str(uuid.uuid4()),
            request_body=request_body,
        )
        decoded_bind = msgpack_decode(msgpack_encode(bind))
        self.assertIsInstance(decoded_bind, DecodeReservationBindReqInput)
        self.assertEqual(decoded_bind.request_body, request_body)
        self.assertNotIn("exact bytes", repr(decoded_bind))

    async def test_correlated_output_resolves_only_matching_operation(self) -> None:
        manager = object.__new__(TokenizerManager)
        manager.decode_control_futures = {}

        async def dispatch(request: DecodeReservationBindReqInput) -> None:
            manager._handle_decode_reservation_control_output(
                DecodeReservationControlReqOutput(
                    correlation_id=request.correlation_id,
                    operation="bind",
                    success=True,
                    response={"state": "prepared"},
                )
            )

        manager._async_dispatch_to_scheduler = dispatch
        response = await manager._request_decode_control(
            "bind",
            DecodeReservationBindReqInput(
                correlation_id="correlation",
                grant_id=str(uuid.uuid4()),
                request_body=b"body",
            ),
        )
        self.assertEqual(response, {"state": "prepared"})
        self.assertEqual(manager.decode_control_futures, {})

    async def test_mismatched_operation_fails_closed(self) -> None:
        manager = object.__new__(TokenizerManager)
        manager.decode_control_futures = {}

        async def dispatch(request: DecodeReservationBindReqInput) -> None:
            manager._handle_decode_reservation_control_output(
                DecodeReservationControlReqOutput(
                    correlation_id=request.correlation_id,
                    operation="promote",
                    success=True,
                    response={"state": "promoted"},
                )
            )

        manager._async_dispatch_to_scheduler = dispatch
        with self.assertRaises(DecodeReservationSchedulerError):
            await manager._request_decode_control(
                "bind",
                DecodeReservationBindReqInput(
                    correlation_id="correlation",
                    grant_id=str(uuid.uuid4()),
                    request_body=b"body",
                ),
            )

    async def test_exact_promoted_body_attaches_once_without_retokenization(
        self,
    ) -> None:
        manager = object.__new__(TokenizerManager)
        manager.decode_control_futures = {}
        manager.decode_inference_attachments = DecodeInferenceAttachmentRegistry()
        manager.decode_bound_inference_bodies = {}
        manager.decode_consumed_inference_bodies = set()
        manager.decode_grant_inference_bodies = {}
        grant_id = uuid.uuid4()
        child_id = uuid.uuid4()
        request_obj = GenerateReqInput(text="already tokenized", rid=str(child_id))
        request_body = b'{ "text": "already tokenized" }'
        manager.decode_inference_attachments.register(
            grant_id=grant_id,
            reservation_attempt_id=uuid.uuid4(),
            reserve_attempt_digest=b"r" * 32,
            inference_route="/generate",
            child_request_ids=(child_id,),
            opaque_request=request_obj,
        )
        manager.decode_inference_attachments.bind(grant_id, request_body)
        manager.decode_inference_attachments.promote(grant_id)
        body_key = manager._decode_inference_body_key("/generate", request_body)
        manager.decode_bound_inference_bodies[body_key] = grant_id
        captured_body: bytes | None = None

        async def dispatch(request: DecodeInferenceAttachReqInput) -> None:
            nonlocal captured_body
            captured_body = request.request_body
            manager._handle_decode_reservation_control_output(
                DecodeReservationControlReqOutput(
                    correlation_id=request.correlation_id,
                    operation="inference_attach",
                    success=True,
                    response={"state": "attached"},
                )
            )

        manager._async_dispatch_to_scheduler = dispatch
        attached = await manager.attach_decode_inference("/generate", request_body)
        self.assertIs(attached, request_obj)
        self.assertEqual(captured_body, request_body)
        self.assertEqual(attached.decode_reservation_grant_id, str(grant_id))

        with self.assertRaises(DecodeReservationConflictError):
            await manager.attach_decode_inference("/generate", request_body)

    async def test_attach_failure_terminalizes_state_and_keeps_body_tombstone(
        self,
    ) -> None:
        manager = object.__new__(TokenizerManager)
        manager.decode_control_futures = {}
        manager.decode_inference_attachments = DecodeInferenceAttachmentRegistry()
        manager.decode_bound_inference_bodies = {}
        manager.decode_consumed_inference_bodies = set()
        manager.decode_grant_inference_bodies = {}
        grant_id = uuid.uuid4()
        child_id = uuid.uuid4()
        request_obj = GenerateReqInput(text="quarantine me", rid=str(child_id))
        request_body = b'{"text":"quarantine me"}'
        manager.decode_inference_attachments.register(
            grant_id=grant_id,
            reservation_attempt_id=uuid.uuid4(),
            reserve_attempt_digest=b"q" * 32,
            inference_route="/generate",
            child_request_ids=(child_id,),
            opaque_request=request_obj,
        )
        manager.decode_inference_attachments.bind(grant_id, request_body)
        manager.decode_inference_attachments.promote(grant_id)
        body_key = manager._decode_inference_body_key("/generate", request_body)
        manager.decode_bound_inference_bodies[body_key] = grant_id
        discarded: list[GenerateReqInput] = []

        def discard(obj: GenerateReqInput) -> None:
            discarded.append(obj)

        manager._discard_pending_req_states = discard

        async def dispatch(request: DecodeInferenceAttachReqInput) -> None:
            manager._handle_decode_reservation_control_output(
                DecodeReservationControlReqOutput(
                    correlation_id=request.correlation_id,
                    operation="inference_attach",
                    success=False,
                    error_type="DecodeReservationConflictError",
                    error_message="scheduler quarantined ambiguous attach",
                )
            )

        manager._async_dispatch_to_scheduler = dispatch
        with self.assertRaises(DecodeReservationSchedulerError):
            await manager.attach_decode_inference("/generate", request_body)

        self.assertEqual(discarded, [request_obj])
        self.assertNotIn(body_key, manager.decode_bound_inference_bodies)
        self.assertIn(body_key, manager.decode_consumed_inference_bodies)
        with self.assertRaises(DecodeReservationConflictError):
            await manager.attach_decode_inference("/generate", request_body)

    async def test_expiry_terminalizes_exact_requests_and_rejects_retry(
        self,
    ) -> None:
        manager = object.__new__(TokenizerManager)
        manager.decode_inference_attachments = DecodeInferenceAttachmentRegistry()
        manager.decode_bound_inference_bodies = {}
        manager.decode_consumed_inference_bodies = set()
        manager.decode_grant_inference_bodies = {}
        manager.decode_pending_expiry_grants = set()
        manager.decode_reserve_refusals = {}
        manager.rid_to_state = {}
        manager.auto_create_handle_loop = lambda: None

        cancelled_grant_id = uuid.uuid4()
        cancelled_attempt_id = uuid.uuid4()
        cancelled_digest = b"e" * 32
        cancelled_child_id = uuid.uuid4()
        cancelled_request = GenerateReqInput(
            text="expired prompt",
            rid=str(cancelled_child_id),
        )
        manager.decode_inference_attachments.register(
            grant_id=cancelled_grant_id,
            reservation_attempt_id=cancelled_attempt_id,
            reserve_attempt_digest=cancelled_digest,
            inference_route="/generate",
            child_request_ids=(cancelled_child_id,),
            opaque_request=cancelled_request,
        )
        manager.decode_inference_attachments.publish_reserve_response(
            cancelled_grant_id,
            {"state": "prepared", "grant_id": str(cancelled_grant_id)},
        )
        cancelled_body = b'{"text":"expired prompt"}'
        manager.decode_inference_attachments.bind(
            cancelled_grant_id,
            cancelled_body,
        )
        cancelled_body_key = manager._decode_inference_body_key(
            "/generate",
            cancelled_body,
        )
        manager.decode_bound_inference_bodies[cancelled_body_key] = cancelled_grant_id
        manager.decode_grant_inference_bodies[cancelled_grant_id] = cancelled_body_key
        manager.rid_to_state[str(cancelled_child_id)] = object()

        quarantined_grant_id = uuid.uuid4()
        quarantined_attempt_id = uuid.uuid4()
        quarantined_digest = b"f" * 32
        quarantined_child_id = uuid.uuid4()
        quarantined_request = GenerateReqInput(
            text="quarantined prompt",
            rid=str(quarantined_child_id),
        )
        manager.decode_inference_attachments.register(
            grant_id=quarantined_grant_id,
            reservation_attempt_id=quarantined_attempt_id,
            reserve_attempt_digest=quarantined_digest,
            inference_route="/generate",
            child_request_ids=(quarantined_child_id,),
            opaque_request=quarantined_request,
        )
        manager.decode_inference_attachments.publish_reserve_response(
            quarantined_grant_id,
            {"state": "prepared", "grant_id": str(quarantined_grant_id)},
        )
        manager.rid_to_state[str(quarantined_child_id)] = object()

        output = DecodeReservationExpiryReqOutput(
            http_worker_ipc="ipc:///tmp/tokenizer-worker",
            cancelled_grant_ids=(str(cancelled_grant_id),),
            quarantined_grant_ids=(str(quarantined_grant_id),),
        )

        manager._handle_decode_reservation_expiry_output(output)
        manager._handle_decode_reservation_expiry_output(output)

        for attempt_id, digest in (
            (cancelled_attempt_id, cancelled_digest),
            (quarantined_attempt_id, quarantined_digest),
        ):
            snapshot = manager.decode_inference_attachments.find_reserve_attempt(
                attempt_id,
                digest,
            )
            self.assertIsNotNone(snapshot)
            self.assertIs(snapshot.state, DecodeInferenceAttachmentState.TERMINAL)
            self.assertIsNone(snapshot.opaque_request)

        self.assertNotIn(str(cancelled_child_id), manager.rid_to_state)
        self.assertNotIn(str(quarantined_child_id), manager.rid_to_state)
        self.assertNotIn(
            cancelled_body_key,
            manager.decode_bound_inference_bodies,
        )
        self.assertNotIn(
            cancelled_grant_id,
            manager.decode_grant_inference_bodies,
        )
        self.assertIn(
            cancelled_body_key,
            manager.decode_consumed_inference_bodies,
        )

        exact_retry = SimpleNamespace(
            reservation_attempt_id=cancelled_attempt_id,
            reserve_attempt_digest=cancelled_digest,
        )
        with self.assertRaises(DecodeReservationConflictError):
            await manager.reserve_decode_reservation(
                attempt=exact_retry,
                attempt_wire={},
                obj=cancelled_request,
            )

    async def test_expiry_before_prepare_publication_reconciles_exact_grant(
        self,
    ) -> None:
        manager = object.__new__(TokenizerManager)
        manager.decode_inference_attachments = DecodeInferenceAttachmentRegistry()
        manager.decode_bound_inference_bodies = {}
        manager.decode_consumed_inference_bodies = set()
        manager.decode_grant_inference_bodies = {}
        manager.decode_pending_expiry_grants = set()
        manager.rid_to_state = {}

        attempt_id = uuid.uuid4()
        child_id = uuid.uuid4()
        digest = b"g" * 32
        request_obj = GenerateReqInput(
            text="expires during prepare",
            rid=str(child_id),
        )
        manager.rid_to_state[str(child_id)] = object()
        attempt = SimpleNamespace(
            reservation_attempt_id=attempt_id,
            reserve_attempt_digest=digest,
            inference_route="/generate",
            child_request_ids=(child_id,),
        )

        async def tokenize(
            attempt_value: object,
            request_value: GenerateReqInput,
        ) -> tuple[TokenizedGenerateReqInput, ...]:
            del attempt_value, request_value
            return ()

        async def request_control(
            operation: str,
            request: DecodeReservationPrepareReqInput,
        ) -> dict[str, object]:
            self.assertEqual(operation, "reserve")
            manager._handle_decode_reservation_expiry_output(
                DecodeReservationExpiryReqOutput(
                    http_worker_ipc="ipc:///tmp/tokenizer-worker",
                    cancelled_grant_ids=(request.grant_id,),
                    quarantined_grant_ids=(),
                )
            )
            return {"state": "prepared", "grant_id": request.grant_id}

        manager._tokenize_decode_reservation_request = tokenize
        manager._request_decode_control = request_control

        with self.assertRaises(DecodeReservationConflictError):
            await manager._prepare_decode_reservation(
                attempt=attempt,
                attempt_wire={},
                obj=request_obj,
            )

        retained = manager.decode_inference_attachments.find_reserve_attempt(
            attempt_id,
            digest,
        )
        self.assertIsNotNone(retained)
        self.assertIs(retained.state, DecodeInferenceAttachmentState.TERMINAL)
        self.assertIsNone(retained.opaque_request)
        self.assertEqual(manager.decode_pending_expiry_grants, set())
        self.assertNotIn(str(child_id), manager.rid_to_state)

    async def test_cancelled_reserve_waiter_does_not_retain_finished_task(
        self,
    ) -> None:
        manager = object.__new__(TokenizerManager)
        manager.decode_inference_attachments = DecodeInferenceAttachmentRegistry()
        manager.decode_reserve_refusals = {}
        manager.decode_reserve_tasks = {}
        manager.decode_reserve_lock = asyncio.Lock()
        manager.auto_create_handle_loop = lambda: None
        started = asyncio.Event()
        release = asyncio.Event()

        async def prepare(**kwargs: object) -> dict[str, object]:
            del kwargs
            started.set()
            await release.wait()
            return {"state": "refused", "disposition": "terminal"}

        manager._prepare_decode_reservation = prepare
        attempt_id = uuid.uuid4()
        attempt = SimpleNamespace(
            reservation_attempt_id=attempt_id,
            reserve_attempt_digest=b"a" * 32,
        )
        waiter = asyncio.create_task(
            manager.reserve_decode_reservation(
                attempt=attempt,
                attempt_wire={},
                obj=GenerateReqInput(text="prompt"),
            )
        )
        await started.wait()
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        release.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        self.assertNotIn(attempt_id, manager.decode_reserve_tasks)


if __name__ == "__main__":
    unittest.main()
