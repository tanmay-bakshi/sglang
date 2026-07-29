import dataclasses
import enum

import numpy as np
import numpy.typing as npt

from sglang.srt.disaggregation.common.staging_layout import (
    StagingChunkLayout,
    StagingComponentGeometry,
    StagingComponentId,
    StagingComponentSpan,
    validate_staging_component_geometry,
)


class StagingEndpoint(str, enum.Enum):
    """Endpoint whose request-local component buffers are being bound."""

    SOURCE = "source"
    DESTINATION = "destination"


@dataclasses.dataclass(frozen=True, eq=False)
class StagingComponentBuffer:
    """Registered tensors and request-local pages for one KV component.

    Entry tuples retain the registration order supplied by the memory pool. In
    particular, K entries remain before V entries when that is the pool's
    registration contract.

    :ivar component_id: Exact main-KV or auxiliary-state identity.
    :ivar tensor_ptrs: Registered tensor base pointers.
    :ivar data_lens: Registered allocation bytes for each tensor.
    :ivar item_lens: Bytes occupied by one page in each tensor.
    :ivar layer_ids: Global layer identifier for every tensor entry.
    :ivar page_size: Complete token rows in one component page.
    :ivar page_array: Request-local physical page indices.
    """

    component_id: StagingComponentId
    tensor_ptrs: tuple[int, ...]
    data_lens: tuple[int, ...]
    item_lens: tuple[int, ...]
    layer_ids: tuple[int, ...]
    page_size: int
    page_array: npt.NDArray[np.int32]

    def __post_init__(self) -> None:
        """Validate one complete component registration.

        :raises TypeError: If the page array is not a one-dimensional int32
            NumPy array.
        :raises ValueError: If tensor metadata is incomplete or invalid.
        """

        if not isinstance(self.page_array, np.ndarray):
            raise TypeError("component page_array must be a NumPy array")
        if self.page_array.dtype != np.dtype(np.int32):
            raise TypeError(
                "component page_array must have dtype int32, got "
                f"{self.page_array.dtype}"
            )
        if self.page_array.ndim != 1:
            raise TypeError(
                "component page_array must be one-dimensional, got "
                f"rank {self.page_array.ndim}"
            )
        if not self.page_array.flags.c_contiguous:
            raise TypeError("component page_array must be C-contiguous")
        if len(self.layer_ids) == 0:
            raise ValueError("global layer IDs are required for packed registration")
        entry_count = len(self.tensor_ptrs)
        if entry_count != len(self.data_lens) or entry_count != len(self.item_lens):
            raise ValueError(
                "tensor-pointer/data-length/item-length count mismatch: "
                f"{entry_count}, {len(self.data_lens)}, and {len(self.item_lens)}"
            )
        for tensor_ptr in self.tensor_ptrs:
            if type(tensor_ptr) is not int or tensor_ptr <= 0:
                raise ValueError(
                    f"tensor pointers must be positive integers, got {tensor_ptr!r}"
                )
        validate_staging_component_geometry(self.geometry, "registered")
        for data_len, item_len in zip(self.data_lens, self.item_lens, strict=True):
            if type(data_len) is not int or data_len <= 0:
                raise ValueError(
                    f"tensor data lengths must be positive integers, got {data_len!r}"
                )
            if data_len % item_len != 0:
                raise ValueError(
                    f"tensor data length {data_len} is not divisible by page item "
                    f"length {item_len}"
                )

    @property
    def geometry(self) -> StagingComponentGeometry:
        """Return transport-independent geometry for this registration.

        :returns: Immutable component geometry in registration order.
        """

        return StagingComponentGeometry(
            component_id=self.component_id,
            item_lens=self.item_lens,
            layer_ids=self.layer_ids,
            page_size=self.page_size,
        )

    @property
    def page_capacity(self) -> int:
        """Return the physical page capacity shared by all tensor entries.

        The smallest registered allocation is authoritative because every active
        page index is applied to every tensor in the component.

        :returns: Minimum complete-page capacity across registered tensors.
        """

        return min(
            data_len // item_len
            for data_len, item_len in zip(self.data_lens, self.item_lens, strict=True)
        )


class StagingComponentBufferRegistry:
    """Exact component registry retaining caller registration order."""

    _components: tuple[StagingComponentBuffer, ...]
    _by_component: dict[StagingComponentId, StagingComponentBuffer]

    def __init__(self, components: tuple[StagingComponentBuffer, ...]) -> None:
        """Build an exact component index.

        :param components: Registrations in memory-pool order.
        :raises ValueError: If a component identity or state index is duplicated.
        """

        by_component: dict[StagingComponentId, StagingComponentBuffer] = {}
        state_indices: set[int] = set()
        for component in components:
            if type(component) is not StagingComponentBuffer:
                raise TypeError(
                    f"expected StagingComponentBuffer, got {type(component)!r}"
                )
            if component.component_id in by_component:
                raise ValueError(
                    "component buffer registration is duplicated: "
                    f"{_component_label(component.component_id)}"
                )
            state_index = component.component_id.state_index
            if state_index is not None:
                if state_index in state_indices:
                    raise ValueError(
                        f"state_index {state_index} is registered more than once"
                    )
                state_indices.add(state_index)
            by_component[component.component_id] = component
        self._components = components
        self._by_component = by_component

    @property
    def components(self) -> tuple[StagingComponentBuffer, ...]:
        """Return registrations in their original order.

        :returns: Immutable ordered registrations.
        """

        return self._components

    def require(self, component_id: StagingComponentId) -> StagingComponentBuffer:
        """Return an exactly identified component registration.

        :param component_id: Exact main-KV or state component identity.
        :returns: Matching registration.
        :raises ValueError: If the component is not registered.
        """

        try:
            return self._by_component[component_id]
        except KeyError as error:
            raise ValueError(
                "active component is not registered: "
                f"{_component_label(component_id)}"
            ) from error


@dataclasses.dataclass(frozen=True, eq=False)
class StagingActiveComponentBuffer:
    """One layout span bound to an endpoint-local page-array slice.

    :ivar component: Complete registered component buffers.
    :ivar page_offset: Component-local offset into the request page array.
    :ivar page_count: Number of physical pages used by the span.
    :ivar page_array: Exact request-local page-array slice used by the span.
    """

    component: StagingComponentBuffer
    page_offset: int
    page_count: int
    page_array: npt.NDArray[np.int32]


@dataclasses.dataclass(frozen=True)
class StagingEndpointBufferBinding:
    """All active buffers for one endpoint in immutable layout order.

    :ivar endpoint: Source or destination binding.
    :ivar components: Active components in the layout's canonical order.
    """

    endpoint: StagingEndpoint
    components: tuple[StagingActiveComponentBuffer, ...]

    def require(self, component_id: StagingComponentId) -> StagingActiveComponentBuffer:
        """Return one exactly identified active component.

        :param component_id: Exact main-KV or state component identity.
        :returns: Matching active component.
        :raises ValueError: If the component is inactive.
        """

        for component in self.components:
            if component.component.component_id == component_id:
                return component
        raise ValueError(f"component is not active: {_component_label(component_id)}")


def _component_label(component_id: StagingComponentId) -> str:
    """Return a reader-facing exact component identity.

    :param component_id: Component identity.
    :returns: Stable diagnostic label.
    """

    if component_id.state_index is None:
        return "main-kv"
    state_type = component_id.state_type
    if state_type is None:
        return f"state[{component_id.state_index}]"
    return f"state[{component_id.state_index}]={state_type.value}"


def _endpoint_layout(
    layout: StagingChunkLayout,
    endpoint: StagingEndpoint,
) -> tuple[
    tuple[StagingComponentGeometry, ...],
    tuple[StagingComponentSpan, ...],
]:
    """Return endpoint geometry together with the common component spans.

    :param layout: Immutable packed staging layout.
    :param endpoint: Source or destination endpoint.
    :returns: Endpoint geometries and canonical spans.
    :raises ValueError: If the immutable layout is internally inconsistent.
    """

    if endpoint is StagingEndpoint.SOURCE:
        geometries = layout.source_components
    elif endpoint is StagingEndpoint.DESTINATION:
        geometries = layout.destination_components
    else:
        raise ValueError(f"unsupported staging endpoint: {endpoint!r}")
    spans = layout.component_spans
    if len(geometries) != len(spans):
        raise ValueError(
            f"{endpoint.value} layout geometry/span count mismatch: "
            f"{len(geometries)} and {len(spans)}"
        )
    for geometry, span in zip(geometries, spans, strict=True):
        validate_staging_component_geometry(geometry, f"{endpoint.value} layout")
        if geometry.component_id != span.component_id:
            raise ValueError(
                f"{endpoint.value} layout component order does not match spans"
            )
    return geometries, spans


def bind_staging_endpoint_buffers(
    layout: StagingChunkLayout,
    endpoint: StagingEndpoint,
    registry: StagingComponentBufferRegistry,
) -> StagingEndpointBufferBinding:
    """Bind one endpoint's exact active registrations to an immutable layout.

    Bounds are checked independently for each active component and endpoint as
    ``offset + physical_token_count / page_size <= len(page_array)``.

    :param layout: Immutable component-aware staging layout.
    :param endpoint: Endpoint whose page offsets and geometry are applied.
    :param registry: Endpoint-local component buffer registry.
    :returns: Active component buffers in immutable layout order.
    :raises ValueError: If a component is missing, its registered geometry
        differs from the layout, or its request-local page array is too short.
    """

    geometries, spans = _endpoint_layout(layout, endpoint)
    active_components: list[StagingActiveComponentBuffer] = []
    for geometry, span in zip(geometries, spans, strict=True):
        component = registry.require(span.component_id)
        if component.geometry != geometry:
            raise ValueError(
                f"{endpoint.value} registration geometry differs from immutable "
                f"layout for {_component_label(span.component_id)}: expected "
                f"{geometry}, got {component.geometry}"
            )
        if span.physical_token_count % geometry.page_size != 0:
            raise ValueError(
                f"{endpoint.value} physical token count is not divisible by the "
                f"immutable page size for {_component_label(span.component_id)}"
            )
        page_count = span.physical_token_count // geometry.page_size
        page_offset = (
            span.source_index_offset
            if endpoint is StagingEndpoint.SOURCE
            else span.destination_index_offset
        )
        if page_offset < 0:
            raise ValueError(
                f"{endpoint.value} page-array offset must be non-negative for "
                f"{_component_label(span.component_id)}"
            )
        page_end = page_offset + page_count
        page_array_length = len(component.page_array)
        if page_end > page_array_length:
            raise ValueError(
                f"{endpoint.value} page-array bounds overflow for "
                f"{_component_label(span.component_id)}: offset {page_offset} + "
                f"{page_count} pages = {page_end}, array length "
                f"{page_array_length}"
            )
        active_page_array = component.page_array[page_offset:page_end]
        if np.any(active_page_array < 0):
            raise ValueError(
                f"{endpoint.value} active page array contains a negative index for "
                f"{_component_label(span.component_id)}"
            )
        page_capacity = component.page_capacity
        if np.any(active_page_array >= page_capacity):
            raise ValueError(
                f"{endpoint.value} active page array exceeds registered page "
                f"capacity {page_capacity} for "
                f"{_component_label(span.component_id)}"
            )
        active_components.append(
            StagingActiveComponentBuffer(
                component=component,
                page_offset=page_offset,
                page_count=page_count,
                page_array=active_page_array,
            )
        )
    return StagingEndpointBufferBinding(
        endpoint=endpoint,
        components=tuple(active_components),
    )
