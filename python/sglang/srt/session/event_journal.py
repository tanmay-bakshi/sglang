import asyncio
import base64
import binascii
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

_EVENT_ID_VERSION = "v1"
_DONE_EVENT = b"data: [DONE]\n\n"
type _StreamKey = tuple[str, str]


class JournalBehindError(ValueError):
    """Reports a resume cursor outside the retained event window."""


@dataclass(frozen=True, slots=True)
class SessionEventCursor:
    """Stable identity of one streaming-session SSE event.

    :ivar lineage_generation: Session lineage containing the event.
    :ivar request_id: Request that produced the event.
    :ivar start: Inclusive absolute token offset covered by the event.
    :ivar end: Exclusive absolute token offset covered by the event.
    """

    lineage_generation: int
    request_id: str
    start: int
    end: int

    def encode(self) -> str:
        """Encode the cursor as an SSE-compatible event identifier.

        :returns: Versioned opaque cursor string.
        """
        request_id = base64.urlsafe_b64encode(self.request_id.encode()).rstrip(b"=")
        return ":".join(
            (
                _EVENT_ID_VERSION,
                str(self.lineage_generation),
                request_id.decode(),
                str(self.start),
                str(self.end),
            )
        )

    @classmethod
    def decode(cls, value: str) -> "SessionEventCursor":
        """Decode and validate an SSE event identifier.

        :param value: Cursor received in ``Last-Event-ID``.
        :returns: Parsed stable event cursor.
        :raises ValueError: If the cursor is malformed.
        """
        parts = value.split(":")
        if len(parts) != 5 or parts[0] != _EVENT_ID_VERSION:
            raise ValueError("Malformed streaming-session event cursor.")
        try:
            lineage_generation = int(parts[1])
            padding = "=" * (-len(parts[2]) % 4)
            request_id = base64.b64decode(
                parts[2] + padding,
                altchars=b"-_",
                validate=True,
            ).decode()
            start = int(parts[3])
            end = int(parts[4])
        except (binascii.Error, UnicodeDecodeError, ValueError) as error:
            raise ValueError("Malformed streaming-session event cursor.") from error
        if lineage_generation < 0 or start < 0 or end < start:
            raise ValueError("Malformed streaming-session event cursor.")
        return cls(
            lineage_generation=lineage_generation,
            request_id=request_id,
            start=start,
            end=end,
        )


@dataclass(frozen=True, slots=True)
class _JournalEvent:
    cursor: SessionEventCursor
    ordinal: int
    payload: bytes


@dataclass(slots=True)
class _StreamState:
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    events: deque[_JournalEvent] = field(default_factory=deque)
    lineage_generation: int | None = None
    next_offset: int | None = None
    next_ordinal: int = 0
    complete: bool = False
    failure_message: str | None = None
    producer_task: asyncio.Task[None] | None = None


@dataclass(frozen=True, slots=True)
class JournalSubscription:
    """Prepared subscription insulated from subsequent retention eviction."""

    state: _StreamState
    pending: tuple[_JournalEvent, ...]
    after_ordinal: int


class SessionEventJournal:
    """Bounded byte journal for streaming-session SSE delivery."""

    _max_events: int
    _streams: dict[_StreamKey, _StreamState]
    _retention_order: deque[tuple[_StreamKey, _JournalEvent]]

    def __init__(self, max_events: int) -> None:
        """Initialize a globally bounded event tail.

        :param max_events: Maximum retained SSE data events across the engine.
        """
        if max_events < 1:
            raise ValueError("Session event journal size must be positive.")
        self._max_events = max_events
        self._streams = {}
        self._retention_order = deque()

    def begin_stream(self, session_id: str, request_id: str) -> JournalSubscription:
        """Create a new stream and subscribe its original client.

        :param session_id: Owning streaming session.
        :param request_id: Stable request identity.
        :returns: Subscription positioned before the first event.
        :raises ValueError: If the same stream identity is still retained.
        """
        key = (session_id, request_id)
        existing = self._streams.get(key)
        if existing is not None:
            raise ValueError(f"Session stream {request_id} already exists.")
        state = _StreamState()
        self._streams[key] = state
        return JournalSubscription(state=state, pending=(), after_ordinal=-1)

    def set_producer(
        self,
        session_id: str,
        request_id: str,
        task: asyncio.Task[None],
    ) -> None:
        """Hold the producer task independently of HTTP response lifetime.

        :param session_id: Owning streaming session.
        :param request_id: Stable request identity.
        :param task: Background generation consumer.
        """
        self._require_state((session_id, request_id)).producer_task = task

    async def append(
        self,
        session_id: str,
        request_id: str,
        lineage_generation: int,
        prompt_offset: int,
        end_offset: int,
        data_event: bytes,
    ) -> bytes:
        """Append one serialized data event and return its identified bytes.

        :param session_id: Owning streaming session.
        :param request_id: Stable request identity.
        :param lineage_generation: Authoritative scheduler lineage generation.
        :param prompt_offset: Absolute output start for the request.
        :param end_offset: Absolute cumulative output end after this event.
        :param data_event: Serialized ``data:`` line including its blank line.
        :returns: Exact retained SSE bytes including the stable ``id:`` line.
        """
        key = (session_id, request_id)
        state = self._require_state(key)
        async with state.condition:
            if state.complete:
                raise RuntimeError("Cannot append to a completed session stream.")
            if state.lineage_generation is None:
                state.lineage_generation = lineage_generation
                state.next_offset = prompt_offset
            if state.lineage_generation != lineage_generation:
                raise RuntimeError("Session stream changed lineage generation.")
            assert state.next_offset is not None
            if end_offset < state.next_offset:
                raise RuntimeError("Session stream token offsets moved backwards.")

            cursor = SessionEventCursor(
                lineage_generation=lineage_generation,
                request_id=request_id,
                start=state.next_offset,
                end=end_offset,
            )
            payload = b"id: " + cursor.encode().encode() + b"\n" + data_event
            event = _JournalEvent(
                cursor=cursor,
                ordinal=state.next_ordinal,
                payload=payload,
            )
            state.next_offset = end_offset
            state.next_ordinal += 1
            state.events.append(event)
            self._retention_order.append((key, event))
            self._evict_oldest()
            state.condition.notify_all()
            return payload

    async def finish(
        self,
        session_id: str,
        request_id: str,
        failure_message: str | None = None,
    ) -> None:
        """Seal a stream and wake every replay subscriber.

        :param session_id: Owning streaming session.
        :param request_id: Stable request identity.
        :param failure_message: Internal failure to surface instead of ``[DONE]``.
        """
        key = (session_id, request_id)
        state = self._require_state(key)
        async with state.condition:
            state.complete = True
            state.failure_message = failure_message
            state.producer_task = None
            state.condition.notify_all()
            if len(state.events) == 0:
                self._streams.pop(key, None)

    def prepare_resume(
        self,
        session_id: str,
        request_id: str,
        last_event_id: str,
    ) -> JournalSubscription:
        """Validate a cursor and snapshot the retained replay suffix.

        :param session_id: Owning streaming session.
        :param request_id: Stable request identity.
        :param last_event_id: Client's last fully received SSE event.
        :returns: Replay subscription after the supplied cursor.
        :raises JournalBehindError: If reconciliation is required.
        """
        try:
            cursor = SessionEventCursor.decode(last_event_id)
        except ValueError as error:
            raise JournalBehindError(str(error)) from error
        if cursor.request_id != request_id:
            raise JournalBehindError("Resume cursor belongs to another request.")

        state = self._streams.get((session_id, request_id))
        if state is None or state.lineage_generation != cursor.lineage_generation:
            raise JournalBehindError("Resume cursor lineage is not retained.")

        events = tuple(state.events)
        if len(events) == 0:
            raise JournalBehindError("Resume cursor is behind the retained journal.")

        matched_index = next(
            (index for index, event in enumerate(events) if event.cursor == cursor),
            None,
        )
        if matched_index is not None:
            pending = events[matched_index + 1 :]
            after_ordinal = events[matched_index].ordinal
        elif cursor.end == events[0].cursor.start:
            pending = events
            after_ordinal = events[0].ordinal - 1
        else:
            raise JournalBehindError("Resume cursor is behind the retained journal.")

        if len(pending) > 0:
            after_ordinal = pending[-1].ordinal
        return JournalSubscription(
            state=state,
            pending=pending,
            after_ordinal=after_ordinal,
        )

    async def stream(self, subscription: JournalSubscription) -> AsyncIterator[bytes]:
        """Yield retained events, follow the live tail, then emit ``[DONE]``.

        :param subscription: Prepared initial or resumed subscription.
        :yields: Exact original SSE event bytes.
        """
        after_ordinal = subscription.after_ordinal
        for event in subscription.pending:
            yield event.payload
        if len(subscription.pending) > 0:
            after_ordinal = subscription.pending[-1].ordinal

        state = subscription.state
        while True:
            async with state.condition:
                available = tuple(
                    event for event in state.events if event.ordinal > after_ordinal
                )
                if len(available) > 0 and available[0].ordinal > after_ordinal + 1:
                    raise JournalBehindError(
                        "Live subscriber fell behind the retained session journal."
                    )
                if len(available) == 0 and not state.complete:
                    await state.condition.wait()
                    continue
                complete = state.complete

            for event in available:
                yield event.payload
                after_ordinal = event.ordinal
            if complete and len(available) == 0:
                break

        if state.failure_message is not None:
            raise RuntimeError(state.failure_message)
        yield _DONE_EVENT

    def _evict_oldest(self) -> None:
        while len(self._retention_order) > self._max_events:
            key, event = self._retention_order.popleft()
            state = self._streams.get(key)
            if state is None:
                continue
            if len(state.events) > 0 and state.events[0] is event:
                state.events.popleft()
            if state.complete and len(state.events) == 0:
                self._streams.pop(key, None)

    def _require_state(self, key: _StreamKey) -> _StreamState:
        state = self._streams.get(key)
        if state is None:
            raise RuntimeError(f"Unknown session stream {key[1]}.")
        return state
