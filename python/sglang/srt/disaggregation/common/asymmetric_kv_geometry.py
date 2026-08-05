from collections.abc import Sequence


def require_uniform_asymmetric_kv_entry_geometry(
    *,
    source_item_lens: Sequence[int],
    destination_item_lens: Sequence[int],
    source_tp_size: int,
    destination_tp_size: int,
) -> None:
    """Require geometry supported by a uniform-entry asymmetric KV slicer.

    :param source_item_lens: Source bytes per page for aligned main-KV entries.
    :param destination_item_lens: Destination bytes per page for the same entries.
    :param source_tp_size: Source attention tensor-parallel width.
    :param destination_tp_size: Destination attention tensor-parallel width.
    :raises ValueError: If the geometry is invalid or needs per-entry slicing.
    """

    if source_tp_size <= 0:
        raise ValueError("source_tp_size must be positive")
    if destination_tp_size <= 0:
        raise ValueError("destination_tp_size must be positive")

    source_lens = tuple(source_item_lens)
    destination_lens = tuple(destination_item_lens)
    if len(source_lens) != len(destination_lens):
        raise ValueError(
            "aligned source and destination KV entry counts differ: "
            f"{len(source_lens)} != {len(destination_lens)}"
        )
    if any(item_len <= 0 for item_len in (*source_lens, *destination_lens)):
        raise ValueError("KV entry item lengths must be positive")
    if source_tp_size == destination_tp_size or len(source_lens) == 0:
        return

    source_is_uniform = len(set(source_lens)) == 1
    destination_is_uniform = len(set(destination_lens)) == 1
    if source_is_uniform and destination_is_uniform:
        return

    raise ValueError(
        "asymmetric TP with heterogeneous per-entry KV geometry requires "
        "per-entry byte-range slicing: "
        f"source_tp_size={source_tp_size}, "
        f"destination_tp_size={destination_tp_size}, "
        f"source_item_lens={source_lens}, "
        f"destination_item_lens={destination_lens}"
    )
