import dataclasses

from sglang.srt.disaggregation.nixl.packed_staging_request import (
    PackedDFlashBoundaryDecodeAdoption,
)
from sglang.srt.disaggregation.terminal_progress.dflash_auxiliary import (
    DFlashBoundaryAdoptedValue,
)


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalDFlashDecodeAdoption:
    """Exact scheduler adoption and device-copy completion authority.

    :ivar transaction_adoption: Authenticated row generation and scalar metadata.
    :ivar device_value: Request-owned token and row-copy completion event.
    """

    transaction_adoption: PackedDFlashBoundaryDecodeAdoption
    device_value: DFlashBoundaryAdoptedValue

    def __post_init__(self) -> None:
        """Validate exact cross-layer adoption authority."""

        if type(self.transaction_adoption) is not PackedDFlashBoundaryDecodeAdoption:
            raise TypeError(
                "transaction_adoption must be PackedDFlashBoundaryDecodeAdoption"
            )
        if type(self.device_value) is not DFlashBoundaryAdoptedValue:
            raise TypeError("device_value must be DFlashBoundaryAdoptedValue")

    def await_device_copy_completion(self) -> PackedDFlashBoundaryDecodeAdoption:
        """Prove the request-owned clone is terminal before row release.

        :returns: Exact transaction adoption accepted by row-release authority.
        """

        completion_event = self.device_value.completion_event
        completion_event.synchronize()
        if not completion_event.query():
            raise RuntimeError(
                "DFlash destination copy event is not terminal after synchronization"
            )
        return self.transaction_adoption
