import dataclasses

from sglang.srt.distributed.parallel_state_wrapper import ParallelState


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class SchedulerOutputIdentity:
    """Immutable process identity governing scheduler output publication.

    :ivar dp_rank: Data-parallel producer rank, or ``None`` without DP identity.
    :ivar dp_size: Configured data-parallel width.
    :ivar tp_rank: Tensor-parallel rank of this scheduler process.
    :ivar tp_size: Tensor-parallel width of this scheduler process.
    :ivar pp_rank: Pipeline-parallel rank of this scheduler process.
    :ivar pp_size: Pipeline-parallel width, restricted to the supported PP1 domain.
    :ivar attn_tp_rank: Attention tensor-parallel rank governing output authority.
    :ivar attn_tp_size: Attention tensor-parallel width.
    :ivar attn_cp_rank: Attention context-parallel rank governing output authority.
    :ivar attn_cp_size: Attention context-parallel width.
    """

    dp_rank: int | None
    dp_size: int
    tp_rank: int
    tp_size: int
    pp_rank: int
    pp_size: int
    attn_tp_rank: int
    attn_tp_size: int
    attn_cp_rank: int
    attn_cp_size: int

    def __post_init__(self) -> None:
        """Validate the complete scheduler output rank domain."""

        size_fields = (
            (self.dp_size, "dp_size"),
            (self.tp_size, "tp_size"),
            (self.pp_size, "pp_size"),
            (self.attn_tp_size, "attn_tp_size"),
            (self.attn_cp_size, "attn_cp_size"),
        )
        for value, label in size_fields:
            if type(value) is not int:
                raise TypeError(f"{label} must be an integer")
            if value <= 0:
                raise ValueError(f"{label} must be positive")
        if self.pp_size != 1:
            raise ValueError("scheduler output publication requires pp_size == 1")

        rank_fields = (
            (self.tp_rank, self.tp_size, "tp_rank"),
            (self.pp_rank, self.pp_size, "pp_rank"),
            (self.attn_tp_rank, self.attn_tp_size, "attn_tp_rank"),
            (self.attn_cp_rank, self.attn_cp_size, "attn_cp_rank"),
        )
        for rank, size, label in rank_fields:
            if type(rank) is not int:
                raise TypeError(f"{label} must be an integer")
            if rank < 0 or rank >= size:
                raise ValueError(f"{label} must identify one rank within its size")

        if self.dp_rank is not None:
            if type(self.dp_rank) is not int:
                raise TypeError("dp_rank must be an integer or None")
            if self.dp_rank < 0:
                raise ValueError("dp_rank must be non-negative")
        if self.dp_rank is None and self.dp_size != 1:
            raise ValueError("dp_rank may be None only when dp_size == 1")
        if (
            self.dp_rank is not None
            and self.dp_size > 1
            and self.dp_rank >= self.dp_size
        ):
            raise ValueError("dp_rank must identify one rank within dp_size")

        attention_partition_size = self.attn_tp_size * self.attn_cp_size
        if self.tp_size % attention_partition_size != 0:
            raise ValueError(
                "attention TP and CP widths must partition the tensor-parallel width"
            )
        expected_attn_tp_rank = self.tp_rank % self.attn_tp_size
        expected_attn_cp_rank = (self.tp_rank // self.attn_tp_size) % self.attn_cp_size
        if self.attn_tp_rank != expected_attn_tp_rank:
            raise ValueError("attn_tp_rank is inconsistent with tp_rank")
        if self.attn_cp_rank != expected_attn_cp_rank:
            raise ValueError("attn_cp_rank is inconsistent with tp_rank")

        attention_dp_size = self.tp_size // attention_partition_size
        if attention_dp_size > 1:
            expected_dp_rank = self.tp_rank // attention_partition_size
            if self.dp_size != attention_dp_size:
                raise ValueError("dp_size is inconsistent with attention partitioning")
            if self.dp_rank != expected_dp_rank:
                raise ValueError("dp_rank is inconsistent with attention partitioning")

    @classmethod
    def from_parallel_state(
        cls,
        parallel_state: ParallelState,
    ) -> "SchedulerOutputIdentity":
        """Construct output identity from the scheduler's authoritative ranks.

        :param parallel_state: Exact process-local parallel state.
        :returns: Validated immutable scheduler output identity.
        """

        if type(parallel_state) is not ParallelState:
            raise TypeError("parallel_state must be ParallelState")
        return cls(
            dp_rank=parallel_state.dp_rank,
            dp_size=parallel_state.dp_size,
            tp_rank=parallel_state.tp_rank,
            tp_size=parallel_state.tp_size,
            pp_rank=parallel_state.pp_rank,
            pp_size=parallel_state.pp_size,
            attn_tp_rank=parallel_state.attn_tp_rank,
            attn_tp_size=parallel_state.attn_tp_size,
            attn_cp_rank=parallel_state.attn_cp_rank,
            attn_cp_size=parallel_state.attn_cp_size,
        )

    @property
    def is_gateway_publisher(self) -> bool:
        """Return whether this PP1 process owns gateway output publication.

        :returns: Whether the process owns its attention partition's output socket.
        """

        return self.attn_tp_rank == 0 and self.attn_cp_rank == 0

    def require_terminal_binding(
        self,
        *,
        tensor_parallel_rank: int,
        tensor_parallel_size: int,
    ) -> None:
        """Require an authenticated terminal rank to match this identity.

        :param tensor_parallel_rank: Rank from the sealed terminal startup binding.
        :param tensor_parallel_size: Width from the sealed terminal startup binding.
        :raises TypeError: If a terminal coordinate is not an integer.
        :raises ValueError: If the binding or publication authority disagrees.
        """

        if type(tensor_parallel_rank) is not int:
            raise TypeError("terminal tensor_parallel_rank must be an integer")
        if type(tensor_parallel_size) is not int:
            raise TypeError("terminal tensor_parallel_size must be an integer")
        if tensor_parallel_rank != self.tp_rank or tensor_parallel_size != self.tp_size:
            raise ValueError(
                "terminal startup binding disagrees with scheduler output identity"
            )
        terminal_gateway_publisher = tensor_parallel_rank == 0
        if terminal_gateway_publisher != self.is_gateway_publisher:
            raise ValueError(
                "terminal and scheduler output publication authorities disagree"
            )
