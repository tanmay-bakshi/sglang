class StreamingSessionConflictError(Exception):
    """Reports an idempotency conflict for a streaming-session request."""

    correlation_id: str

    def __init__(self, message: str, correlation_id: str) -> None:
        """Initialize a streaming-session conflict.

        :param message: Human-readable conflict description.
        :param correlation_id: Correlation identifier of the rejected request.
        """
        super().__init__(message)
        self.correlation_id = correlation_id


STREAMING_SESSION_CONFLICT_ERROR_TYPE = StreamingSessionConflictError.__name__
