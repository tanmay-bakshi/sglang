from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from math import prod
from types import MappingProxyType

TGV_TACTIC_COUNT: int = 29
GEMMA4_TP2_PCG_PREFILL_M_BUCKETS: tuple[int, ...] = tuple(range(1024, 8193, 1024))


@dataclass(frozen=True, slots=True)
class Gemma4PrefillBf16Shape:
    """Exact BF16 GEMM shape and bias ABI.

    :ivar m: Flattened token rows.
    :ivar n: Output features.
    :ivar k: Input features.
    :ivar has_bias: Whether the GEMM includes an output-feature bias.
    """

    m: int
    n: int
    k: int
    has_bias: bool

    def __post_init__(self) -> None:
        """Validate positive matrix dimensions."""

        for name, value in (("m", self.m), ("n", self.n), ("k", self.k)):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")


@dataclass(frozen=True, slots=True)
class Gemma4PrefillBf16Tactic:
    """One exact shape-to-tactic entry.

    :ivar shape: Exact GEMM shape and bias ABI.
    :ivar tactic: CuTe DSL TGV tactic identifier.
    """

    shape: Gemma4PrefillBf16Shape
    tactic: int

    def __post_init__(self) -> None:
        """Validate the tactic against the TGV custom-op ABI."""

        if self.tactic < 0 or self.tactic >= TGV_TACTIC_COUNT:
            raise ValueError(
                f"tactic must be in [0, {TGV_TACTIC_COUNT}), got {self.tactic}"
            )


@dataclass(frozen=True, slots=True)
class StaticGemma4PrefillBf16TacticCache:
    """Immutable exact-shape tactic cache with strict ABI fallback.

    :ivar entries: Ordered immutable cache entries.
    """

    entries: tuple[Gemma4PrefillBf16Tactic, ...]
    _tactics: Mapping[Gemma4PrefillBf16Shape, int] = field(
        init=False,
        repr=False,
    )

    def __init__(self, entries: Iterable[Gemma4PrefillBf16Tactic]) -> None:
        """Build an immutable cache.

        :param entries: Exact shape-to-tactic entries.
        :raises ValueError: If the same shape ABI appears more than once.
        """

        frozen_entries = tuple(entries)
        tactics: dict[Gemma4PrefillBf16Shape, int] = {}
        for entry in frozen_entries:
            if entry.shape in tactics:
                raise ValueError(f"duplicate BF16 tactic entry for {entry.shape}")
            tactics[entry.shape] = entry.tactic
        object.__setattr__(self, "entries", frozen_entries)
        object.__setattr__(self, "_tactics", MappingProxyType(tactics))

    def resolve(
        self,
        shape: Gemma4PrefillBf16Shape,
        *,
        x_is_contiguous: bool,
        weight_is_contiguous: bool,
        bias_is_contiguous: bool,
    ) -> int | None:
        """Resolve a tactic only for the exact supported tensor ABI.

        :param shape: Runtime GEMM shape and bias ABI.
        :param x_is_contiguous: Whether the input is contiguous.
        :param weight_is_contiguous: Whether the row-major weight is contiguous.
        :param bias_is_contiguous: Whether a present bias is contiguous.
        :returns: Exact tactic identifier, or ``None`` for the incumbent path.
        """

        if not x_is_contiguous or not weight_is_contiguous:
            return None
        if shape.has_bias and not bias_is_contiguous:
            return None
        return self._tactics.get(shape)


def gemma4_prefill_bf16_shape(
    input_shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
    *,
    has_bias: bool,
) -> Gemma4PrefillBf16Shape | None:
    """Derive the flattened linear shape without accepting an incompatible ABI.

    :param input_shape: Input tensor shape ending in input features.
    :param weight_shape: Row-major linear weight shape ``(N, K)``.
    :param has_bias: Whether the linear operation includes bias.
    :returns: Flattened exact GEMM shape, or ``None`` for the incumbent path.
    """

    if len(input_shape) == 0 or len(weight_shape) != 2:
        return None
    k = input_shape[-1]
    if k <= 0 or weight_shape[0] <= 0 or weight_shape[1] <= 0:
        return None
    if k != weight_shape[1]:
        return None
    m = prod(input_shape[:-1])
    if m <= 0:
        return None
    return Gemma4PrefillBf16Shape(
        m=m,
        n=weight_shape[0],
        k=k,
        has_bias=has_bias,
    )


GEMMA4_TP2_PCG_PREFILL_BF16_TACTICS = StaticGemma4PrefillBf16TacticCache(
    Gemma4PrefillBf16Tactic(
        shape=Gemma4PrefillBf16Shape(
            m=m,
            n=8192,
            k=5376,
            has_bias=False,
        ),
        tactic=24,
    )
    for m in GEMMA4_TP2_PCG_PREFILL_M_BUCKETS
)
