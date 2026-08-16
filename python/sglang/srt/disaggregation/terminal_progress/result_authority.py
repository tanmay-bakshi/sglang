import dataclasses
import enum


class TerminalPrefillRowDisposition(enum.Enum):
    """Immutable scheduler disposition for one submitted prefill result row."""

    SCHEDULER_LOCAL_INTERMEDIATE = enum.auto()
    SCHEDULER_LOCAL_FINAL = enum.auto()
    OWNER_MANAGED_INTERMEDIATE = enum.auto()
    OWNER_MANAGED_FINAL = enum.auto()

    @property
    def is_scheduler_local(self) -> bool:
        """Return whether the scheduler completes this row locally.

        :returns: Whether the row belongs to a scheduler-local fake request.
        """

        return self in (
            TerminalPrefillRowDisposition.SCHEDULER_LOCAL_INTERMEDIATE,
            TerminalPrefillRowDisposition.SCHEDULER_LOCAL_FINAL,
        )

    @property
    def is_final(self) -> bool:
        """Return whether the submitted row completes prefill.

        :returns: Whether the exact submitted row is a final prefill row.
        """

        return self in (
            TerminalPrefillRowDisposition.SCHEDULER_LOCAL_FINAL,
            TerminalPrefillRowDisposition.OWNER_MANAGED_FINAL,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalPrefillResultAuthority:
    """Exact row dispositions retained with one model result.

    :ivar rows: Row dispositions in model-result order.
    """

    rows: tuple[TerminalPrefillRowDisposition, ...]

    def __post_init__(self) -> None:
        """Validate the immutable result authority."""

        if type(self.rows) is not tuple or len(self.rows) == 0:
            raise ValueError("terminal prefill result authority requires rows")
        if any(type(row) is not TerminalPrefillRowDisposition for row in self.rows):
            raise TypeError("terminal prefill result authority contains an invalid row")
