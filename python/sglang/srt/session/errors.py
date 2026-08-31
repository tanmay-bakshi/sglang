class StreamingSessionConflictError(Exception):
    """Reports an idempotency conflict for a streaming-session request."""

    correlation_id: str
    observed_tip: int
    observed_digest: str
    lineage_generation: int

    def __init__(
        self,
        message: str,
        correlation_id: str,
        observed_tip: int,
        observed_digest: str,
        lineage_generation: int,
    ) -> None:
        """Initialize a streaming-session conflict.

        :param message: Human-readable conflict description.
        :param correlation_id: Correlation identifier of the rejected request.
        :param observed_tip: Durable session tip observed by the scheduler.
        :param observed_digest: Durable lineage digest observed by the scheduler.
        :param lineage_generation: Durable lineage generation observed by the scheduler.
        """
        super().__init__(message)
        self.correlation_id = correlation_id
        self.observed_tip = observed_tip
        self.observed_digest = observed_digest
        self.lineage_generation = lineage_generation


class StreamingSessionJournalBehindError(Exception):
    """Reports a resume cursor outside the retained stream journal."""

    current_tip: int
    current_digest: str
    required_action: str

    def __init__(
        self,
        current_tip: int,
        current_digest: str,
        required_action: str = "full_state_reconciliation",
    ) -> None:
        """Initialize a typed journal-behind error.

        :param current_tip: Current durable session tip.
        :param current_digest: Current durable lineage digest.
        :param required_action: Recovery action required of the client.
        """
        super().__init__("Resume cursor is behind the retained session journal.")
        self.current_tip = current_tip
        self.current_digest = current_digest
        self.required_action = required_action


STREAMING_SESSION_CONFLICT_ERROR_TYPE = StreamingSessionConflictError.__name__


class StreamingSessionInfoUnavailableError(ValueError):
    """Reports introspection of an existing non-streaming session."""
