from dataclasses import dataclass

from sglang.srt.session.errors import StreamingSessionStaleEpochError

FENCING_EPOCH_HEADER = "X-SGLang-Fencing-Epoch"
CLUSTER_INCARNATION_HEADER = "X-SGLang-Cluster-Incarnation"


@dataclass(frozen=True, slots=True)
class SessionFencingState:
    """Current epoch fencing register value.

    :ivar epoch: Minimum accepted session mutation epoch.
    :ivar cluster_incarnation: Installed cluster incarnation identity.
    """

    epoch: int = 0
    cluster_incarnation: int = 0

    @property
    def installed(self) -> bool:
        """Return whether the register is enforcing a fence.

        :returns: Whether the register differs from the unfenced zero pair.
        """
        return self.epoch != 0 or self.cluster_incarnation != 0

    def response_headers(self) -> dict[str, str]:
        """Build the public response-header echo.

        :returns: Current register encoded as decimal HTTP headers.
        """
        return {
            FENCING_EPOCH_HEADER: str(self.epoch),
            CLUSTER_INCARNATION_HEADER: str(self.cluster_incarnation),
        }


class SessionFencingRegister:
    """Scheduler-authoritative fencing register for session mutations."""

    _state: SessionFencingState

    def __init__(self) -> None:
        """Initialize the register in its unfenced zero state."""
        self._state = SessionFencingState()

    @property
    def state(self) -> SessionFencingState:
        """Return the immutable current register value.

        :returns: Current fencing state.
        """
        return self._state

    def install(self, epoch: int, cluster_incarnation: int) -> SessionFencingState:
        """Atomically install an exact fencing-register pair.

        :param epoch: Minimum accepted session mutation epoch.
        :param cluster_incarnation: Cluster incarnation identity to echo.
        :returns: Installed immutable state.
        :raises ValueError: If either integer is negative.
        """
        if epoch < 0 or cluster_incarnation < 0:
            raise ValueError("Session fencing register values must be non-negative.")
        self._state = SessionFencingState(
            epoch=epoch,
            cluster_incarnation=cluster_incarnation,
        )
        return self._state

    def validate(
        self,
        request_epoch: int | None,
        *,
        lineage_generation: int = 0,
        observed_tip: int = 0,
    ) -> None:
        """Reject a session mutation below the installed epoch.

        An omitted header has epoch zero, so installing a positive fence also
        fences clients that do not yet send the header.

        :param request_epoch: Epoch carried by the mutation request.
        :param lineage_generation: Current session lineage for SSE identity.
        :param observed_tip: Current session tip for SSE identity.
        :raises ValueError: If the request epoch is negative.
        :raises StreamingSessionStaleEpochError: If the request is fenced out.
        """
        effective_epoch = 0 if request_epoch is None else request_epoch
        if effective_epoch < 0:
            raise ValueError("Session fencing epoch must be non-negative.")
        if not self._state.installed or effective_epoch >= self._state.epoch:
            return
        raise StreamingSessionStaleEpochError(
            request_epoch=effective_epoch,
            registered_epoch=self._state.epoch,
            cluster_incarnation=self._state.cluster_incarnation,
            lineage_generation=lineage_generation,
            observed_tip=observed_tip,
        )
