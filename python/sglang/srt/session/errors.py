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


class StreamingSessionStaleEpochError(Exception):
    """Reports a session mutation fenced by the installed engine epoch."""

    request_epoch: int
    registered_epoch: int
    cluster_incarnation: int
    lineage_generation: int
    observed_tip: int

    def __init__(
        self,
        request_epoch: int,
        registered_epoch: int,
        cluster_incarnation: int,
        lineage_generation: int = 0,
        observed_tip: int = 0,
    ) -> None:
        """Initialize a typed stale-epoch rejection.

        :param request_epoch: Epoch supplied by the rejected mutation.
        :param registered_epoch: Minimum epoch installed on the engine.
        :param cluster_incarnation: Installed cluster incarnation identity.
        :param lineage_generation: Current session lineage generation.
        :param observed_tip: Current durable session tip.
        """
        super().__init__(
            f"Stale session epoch {request_epoch}: engine fencing register is "
            f"({registered_epoch}, {cluster_incarnation})."
        )
        self.request_epoch = request_epoch
        self.registered_epoch = registered_epoch
        self.cluster_incarnation = cluster_incarnation
        self.lineage_generation = lineage_generation
        self.observed_tip = observed_tip


STREAMING_SESSION_CONFLICT_ERROR_TYPE = StreamingSessionConflictError.__name__
STREAMING_SESSION_STALE_EPOCH_ERROR_TYPE = StreamingSessionStaleEpochError.__name__


class StreamingSessionInfoUnavailableError(ValueError):
    """Reports introspection of an existing non-streaming session."""


class StreamingSessionDemotionError(ValueError):
    """Reports a session state that cannot enter the host-resident transaction."""


class StreamingSessionBusyError(RuntimeError):
    """Reports a lifecycle transition blocked by a live cache owner."""

    status_code: int = 409


class StreamingSessionNamespaceError(RuntimeError):
    """Reports a demoted session resumed outside its seeded cache namespace."""

    status_code: int = 409

    seeded_extra_key: str | None
    seeded_cache_salt: str | None
    request_extra_key: str | None
    request_cache_salt: str | None

    def __init__(
        self,
        message: str,
        *,
        seeded_extra_key: str | None,
        seeded_cache_salt: str | None,
        request_extra_key: str | None,
        request_cache_salt: str | None,
    ) -> None:
        """Build the refusal with the seeded and requested namespaces.

        :param message: Human-readable refusal.
        :param seeded_extra_key: Extra key the session was demoted under.
        :param seeded_cache_salt: Cache salt the session was demoted under.
        :param request_extra_key: Extra key the resume carried.
        :param request_cache_salt: Cache salt the resume carried.
        """
        super().__init__(message)
        self.seeded_extra_key = seeded_extra_key
        self.seeded_cache_salt = seeded_cache_salt
        self.request_extra_key = request_extra_key
        self.request_cache_salt = request_cache_salt


STREAMING_SESSION_DEMOTION_ERROR_TYPE = StreamingSessionDemotionError.__name__
STREAMING_SESSION_BUSY_ERROR_TYPE = StreamingSessionBusyError.__name__
STREAMING_SESSION_NAMESPACE_ERROR_TYPE = StreamingSessionNamespaceError.__name__
