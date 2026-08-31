import asyncio
import unittest

from sglang.srt.session.event_journal import (
    JournalBehindError,
    SessionEventCursor,
    SessionEventJournal,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_DONE_EVENT = b"data: [DONE]\n\n"


class SessionEventCursorTest(unittest.TestCase):
    """Stable streaming-session event cursor encoding."""

    def test_cursor_round_trips_unusual_request_identity(self) -> None:
        """Preserve request identities without delimiter ambiguity."""
        cursor = SessionEventCursor(
            lineage_generation=7,
            request_id="request:/?= weird ☃",
            start=41,
            end=44,
        )

        self.assertEqual(SessionEventCursor.decode(cursor.encode()), cursor)

    def test_cursor_rejects_invalid_base64(self) -> None:
        """Reject malformed opaque request identities as cursor errors."""
        with self.assertRaisesRegex(ValueError, "Malformed"):
            SessionEventCursor.decode("v1:1:!not-base64!:2:3")


class SessionEventJournalTest(unittest.IsolatedAsyncioTestCase):
    """Bounded exact-byte replay and live-tail behavior."""

    async def test_replay_bytes_are_identical_to_original_suffix(self) -> None:
        """Return stored bytes rather than reserializing replay events."""
        journal = SessionEventJournal(max_events=4)
        original = journal.begin_stream("session-a", "request-a")
        first = await journal.append(
            "session-a",
            "request-a",
            2,
            100,
            101,
            b'data: {"token":1}\n\n',
        )
        second = await journal.append(
            "session-a",
            "request-a",
            2,
            100,
            103,
            b'data: {"token":2,"spacing":"exact"}\n\n',
        )
        await journal.finish("session-a", "request-a")

        original_bytes = [event async for event in journal.stream(original)]
        replay = journal.prepare_resume(
            "session-a",
            "request-a",
            SessionEventCursor(2, "request-a", 100, 101).encode(),
        )
        replay_bytes = [event async for event in journal.stream(replay)]

        self.assertEqual(original_bytes, [first, second, _DONE_EVENT])
        self.assertEqual(replay_bytes, [second, _DONE_EVENT])
        self.assertIs(replay_bytes[0], second)

    async def test_live_resume_follows_events_appended_after_subscription(self) -> None:
        """Follow a retained cursor into the live stream tail."""
        journal = SessionEventJournal(max_events=4)
        journal.begin_stream("session-a", "request-a")
        await journal.append(
            "session-a",
            "request-a",
            0,
            10,
            11,
            b"data: first\n\n",
        )
        replay = journal.prepare_resume(
            "session-a",
            "request-a",
            SessionEventCursor(0, "request-a", 10, 11).encode(),
        )
        iterator = journal.stream(replay)
        waiting = asyncio.create_task(anext(iterator))
        await asyncio.sleep(0)

        second = await journal.append(
            "session-a",
            "request-a",
            0,
            10,
            12,
            b"data: second\n\n",
        )
        await journal.finish("session-a", "request-a")

        self.assertEqual(await waiting, second)
        self.assertEqual(await anext(iterator), _DONE_EVENT)
        with self.assertRaises(StopAsyncIteration):
            await anext(iterator)

    async def test_cursor_at_retained_boundary_replays_full_tail(self) -> None:
        """Accept the immediately preceding cursor after its event is evicted."""
        journal = SessionEventJournal(max_events=2)
        journal.begin_stream("session-a", "request-a")
        first_cursor = SessionEventCursor(1, "request-a", 10, 11)
        await journal.append("session-a", "request-a", 1, 10, 11, b"data: first\n\n")
        second = await journal.append(
            "session-a", "request-a", 1, 10, 12, b"data: second\n\n"
        )
        third = await journal.append(
            "session-a", "request-a", 1, 10, 13, b"data: third\n\n"
        )
        await journal.finish("session-a", "request-a")

        replay = journal.prepare_resume(
            "session-a",
            "request-a",
            first_cursor.encode(),
        )

        self.assertEqual(
            [event async for event in journal.stream(replay)],
            [second, third, _DONE_EVENT],
        )

    async def test_cursor_behind_retained_boundary_is_rejected(self) -> None:
        """Require reconciliation when a cursor precedes the retained tail."""
        journal = SessionEventJournal(max_events=2)
        journal.begin_stream("session-a", "request-a")
        first_cursor = SessionEventCursor(1, "request-a", 10, 11)
        for end in (11, 12, 13, 14):
            await journal.append(
                "session-a",
                "request-a",
                1,
                10,
                end,
                f"data: {end}\n\n".encode(),
            )

        with self.assertRaises(JournalBehindError):
            journal.prepare_resume(
                "session-a",
                "request-a",
                first_cursor.encode(),
            )

    async def test_malformed_and_wrong_request_cursors_are_rejected(self) -> None:
        """Treat invalid or cross-request cursors as unavailable journal state."""
        journal = SessionEventJournal(max_events=2)
        journal.begin_stream("session-a", "request-a")
        await journal.append("session-a", "request-a", 1, 10, 11, b"data: first\n\n")

        with self.assertRaises(JournalBehindError):
            journal.prepare_resume("session-a", "request-a", "not-a-cursor")
        with self.assertRaises(JournalBehindError):
            journal.prepare_resume(
                "session-a",
                "request-a",
                SessionEventCursor(1, "request-b", 10, 11).encode(),
            )

    async def test_slow_live_subscriber_does_not_silently_skip_events(self) -> None:
        """Fail a live subscriber when bounded retention overtakes it."""
        journal = SessionEventJournal(max_events=1)
        original = journal.begin_stream("session-a", "request-a")
        await journal.append("session-a", "request-a", 0, 0, 1, b"data: first\n\n")
        await journal.append("session-a", "request-a", 0, 0, 2, b"data: second\n\n")

        with self.assertRaises(JournalBehindError):
            await anext(journal.stream(original))

    async def test_producer_failure_does_not_emit_false_done_sentinel(self) -> None:
        """Surface an internal producer failure after all retained data."""
        journal = SessionEventJournal(max_events=2)
        original = journal.begin_stream("session-a", "request-a")
        event = await journal.append(
            "session-a", "request-a", 0, 0, 1, b"data: first\n\n"
        )
        await journal.finish(
            "session-a",
            "request-a",
            failure_message="producer failed",
        )
        iterator = journal.stream(original)

        self.assertEqual(await anext(iterator), event)
        with self.assertRaisesRegex(RuntimeError, "producer failed"):
            await anext(iterator)


if __name__ == "__main__":
    unittest.main()
