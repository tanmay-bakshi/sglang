class StreamingSessionConflictError(Exception):
    """Reports an idempotency conflict for a streaming-session request."""

    correlation_id: str
    observed_tip: int
    observed_digest: str

    def __init__(
        self,
        message: str,
        correlation_id: str,
        observed_tip: int,
        observed_digest: str,
    ) -> None:
        """Initialize a streaming-session conflict.

        :param message: Human-readable conflict description.
        :param correlation_id: Correlation identifier of the rejected request.
        :param observed_tip: Durable session tip observed by the scheduler.
        :param observed_digest: Durable lineage digest observed by the scheduler.
        """
        super().__init__(message)
        self.correlation_id = correlation_id
        self.observed_tip = observed_tip
        self.observed_digest = observed_digest


STREAMING_SESSION_CONFLICT_ERROR_TYPE = StreamingSessionConflictError.__name__


class StreamingSessionInfoUnavailableError(ValueError):
    """Reports introspection of an existing non-streaming session."""
