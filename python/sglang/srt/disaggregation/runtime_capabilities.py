import dataclasses
from typing import Literal

KvTransferProtocol = Literal["packed-v4"]
PreparedGrantProtocol = Literal["control-v1"]
SUPPORTED_PACKED_SOURCE_TP_SIZES: tuple[int, ...] = (1, 2, 4, 8)

PD_RUNTIME_CAPABILITIES_FIELD = "pd_runtime_capabilities"
_KV_DTYPE_ALIASES = {
    "bfloat16": "bf16",
    "fp4_mx_block16": "nvfp4",
}
_RUNTIME_CAPABILITY_FIELDS = {
    "kv_dtype",
    "page_size",
    "kv_transfer_protocol",
    "prepared_grant_protocol",
}


@dataclasses.dataclass(frozen=True)
class PdProcessRuntimeCapabilities:
    """Runtime-attested PD data-plane and control-plane capabilities.

    :ivar kv_dtype: Canonical transfer-visible KV storage dtype.
    :ivar page_size: Runtime KV allocator page size.
    :ivar kv_transfer_protocol: Implemented KV-transfer wire protocol.
    :ivar prepared_grant_protocol: Implemented decoder-grant control protocol.
    """

    kv_dtype: str
    page_size: int
    kv_transfer_protocol: KvTransferProtocol | None
    prepared_grant_protocol: PreparedGrantProtocol | None

    def __post_init__(self) -> None:
        """Canonicalize and validate one initialized runtime report."""

        kv_dtype = _KV_DTYPE_ALIASES.get(self.kv_dtype, self.kv_dtype)
        if (
            len(kv_dtype) == 0
            or kv_dtype == "auto"
            or kv_dtype.lower() != kv_dtype
            or not kv_dtype.replace("_", "").isalnum()
        ):
            raise ValueError(
                "PD runtime capabilities require a canonical runtime KV dtype"
            )
        object.__setattr__(self, "kv_dtype", kv_dtype)

        if type(self.page_size) is not int or self.page_size <= 0:
            raise ValueError(
                "PD runtime capabilities require a resolved positive page_size"
            )
        if self.kv_transfer_protocol not in (None, "packed-v4"):
            raise ValueError(
                "PD runtime capabilities contain an unknown KV-transfer protocol"
            )
        if self.prepared_grant_protocol not in (None, "control-v1"):
            raise ValueError(
                "PD runtime capabilities contain an unknown prepared-grant protocol"
            )

    @property
    def advertises_pd_process(self) -> bool:
        """Return whether both required protocol implementations are live.

        :returns: Whether the process may publish the PD capability document.
        """

        return (
            self.kv_transfer_protocol == "packed-v4"
            and self.prepared_grant_protocol == "control-v1"
        )

    def to_dict(self) -> dict[str, object]:
        """Return the process-safe scheduler handshake representation.

        :returns: Strict runtime capability fields.
        """

        return {
            "kv_dtype": self.kv_dtype,
            "page_size": self.page_size,
            "kv_transfer_protocol": self.kv_transfer_protocol,
            "prepared_grant_protocol": self.prepared_grant_protocol,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "PdProcessRuntimeCapabilities":
        """Reconstruct a strict runtime report from a scheduler handshake.

        :param value: Scheduler-owned capability fields.
        :returns: Validated runtime capabilities.
        """

        if set(value) != _RUNTIME_CAPABILITY_FIELDS:
            raise ValueError("invalid PD runtime capability fields")

        kv_dtype = value["kv_dtype"]
        page_size = value["page_size"]
        kv_transfer_protocol = value["kv_transfer_protocol"]
        prepared_grant_protocol = value["prepared_grant_protocol"]
        if type(kv_dtype) is not str:
            raise TypeError("PD runtime kv_dtype must be a string")
        if type(page_size) is not int:
            raise TypeError("PD runtime page_size must be an integer")
        if kv_transfer_protocol is not None and type(kv_transfer_protocol) is not str:
            raise TypeError("PD runtime KV-transfer protocol must be a string or null")
        if (
            prepared_grant_protocol is not None
            and type(prepared_grant_protocol) is not str
        ):
            raise TypeError(
                "PD runtime prepared-grant protocol must be a string or null"
            )

        return cls(
            kv_dtype=kv_dtype,
            page_size=page_size,
            kv_transfer_protocol=kv_transfer_protocol,
            prepared_grant_protocol=prepared_grant_protocol,
        )


def runtime_capabilities_from_scheduler_info(
    scheduler_info: dict[str, object],
) -> PdProcessRuntimeCapabilities | None:
    """Read and validate one scheduler's runtime capability report.

    :param scheduler_info: Scheduler initialization handshake.
    :returns: Validated capabilities for a PD process, otherwise ``None``.
    """

    value = scheduler_info[PD_RUNTIME_CAPABILITIES_FIELD]
    if value is None:
        return None
    if type(value) is not dict:
        raise TypeError("PD runtime capabilities must be an object or null")
    return PdProcessRuntimeCapabilities.from_dict(value)


def validate_scheduler_runtime_capabilities(
    scheduler_infos: list[dict[str, object]],
) -> dict[str, object] | None:
    """Require every scheduler rank to attest the same runtime capabilities.

    :param scheduler_infos: Initialization handshakes from one process generation.
    :returns: The agreed process-safe capability representation.
    """

    if len(scheduler_infos) == 0:
        raise ValueError("scheduler initialization produced no rank information")

    reference = scheduler_infos[0][PD_RUNTIME_CAPABILITIES_FIELD]
    for rank, scheduler_info in enumerate(scheduler_infos[1:], start=1):
        candidate = scheduler_info[PD_RUNTIME_CAPABILITIES_FIELD]
        if candidate != reference:
            raise RuntimeError(
                "scheduler ranks disagree on PD runtime capabilities: "
                f"rank 0 reported {reference!r}, rank {rank} reported {candidate!r}"
            )

    runtime_capabilities_from_scheduler_info(scheduler_infos[0])
    if reference is None:
        return None
    if type(reference) is not dict:
        raise TypeError("PD runtime capabilities must be an object or null")
    return reference
