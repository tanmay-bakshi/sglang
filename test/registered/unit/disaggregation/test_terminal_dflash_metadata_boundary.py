from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from sglang.srt.disaggregation.common.decode_allocation_lease import (
    DecodeAllocationLeaseError,
)
from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeTransferQueue
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    KVClassType,
    TransferBackend,
)
from sglang.srt.managers.scheduler import (
    Scheduler,
    _uses_terminal_dflash_boundary,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _server_args(role: str) -> SimpleNamespace:
    """Build the disaggregation fields consumed by scheduler initialization.

    :param role: Disaggregation role under test.
    :returns: Minimal terminal server configuration.
    """

    return SimpleNamespace(
        disaggregation_mode=role,
        disaggregation_transfer_backend="nixl",
        pd_terminal_local_membership=object(),
        dp_size=1,
        disaggregation_bootstrap_port=33000,
        num_reserved_decode_tokens=512,
        language_only=False,
        encoder_transfer_backend="zmq_to_scheduler",
    )


def _spec_algorithm(*, dflash: bool) -> MagicMock:
    """Build a speculative-algorithm double with explicit schema identity.

    :param dflash: Whether the algorithm owns the DFlash boundary.
    :returns: Configured algorithm double.
    """

    algorithm = MagicMock()
    algorithm.is_dflash.return_value = dflash
    algorithm.carries_draft_hidden_states.return_value = False
    return algorithm


@pytest.mark.parametrize("role", ("prefill", "decode"))
def test_terminal_dflash_scheduler_allocates_no_legacy_metadata(role: str) -> None:
    """Both terminal roles pass absent legacy resources into their queues."""

    scheduler = SimpleNamespace(
        server_args=_server_args(role),
        spec_algorithm=_spec_algorithm(dflash=True),
        draft_worker=None,
        model_config=SimpleNamespace(hf_config=object()),
        req_to_token_pool=SimpleNamespace(size=8),
        token_to_kv_pool_allocator=MagicMock(),
        attn_tp_cpu_group=object(),
        ps=SimpleNamespace(tp_rank=0, tp_size=1, gpu_id=0, pp_rank=0, pp_size=1),
        tree_cache=object(),
        max_total_num_tokens=4096,
        max_running_requests=8,
        _activate_terminal_kv_manager=MagicMock(),
    )
    transfer_queue = SimpleNamespace()
    prealloc_manager = object()
    prealloc_queue = SimpleNamespace(kv_manager=prealloc_manager)
    prefill_manager = object()
    prefill_queue = SimpleNamespace(kv_manager=prefill_manager)

    with (
        patch(
            "sglang.srt.managers.scheduler.kv_cache_builder.get_draft_kv_pool",
            return_value=None,
        ),
        patch(
            "sglang.srt.managers.scheduler.get_dsa_seed_metadata_dim",
            return_value=0,
        ) as dsa_metadata_factory,
        patch(
            "sglang.srt.managers.scheduler.is_minimax_sparse",
            return_value=False,
        ),
        patch(
            "sglang.srt.managers.scheduler.MetadataBuffers",
            side_effect=AssertionError("terminal DFlash allocated MetadataBuffers"),
        ),
        patch(
            "sglang.srt.managers.scheduler.ReqToMetadataIdxAllocator",
            side_effect=AssertionError("terminal DFlash allocated a metadata index"),
        ),
        patch(
            "sglang.srt.managers.scheduler.DecodeTransferQueue",
            return_value=transfer_queue,
        ) as decode_transfer_factory,
        patch(
            "sglang.srt.managers.scheduler.DecodePreallocQueue",
            return_value=prealloc_queue,
        ) as decode_prealloc_factory,
        patch(
            "sglang.srt.managers.scheduler.PrefillBootstrapQueue",
            return_value=prefill_queue,
        ) as prefill_factory,
    ):
        Scheduler.init_disaggregation(scheduler)

    assert scheduler.req_to_metadata_buffer_idx_allocator is None
    assert scheduler.disagg_metadata_buffers is None
    dsa_metadata_factory.assert_not_called()
    if role == "decode":
        assert (
            decode_transfer_factory.call_args.kwargs[
                "req_to_metadata_buffer_idx_allocator"
            ]
            is None
        )
        assert decode_transfer_factory.call_args.kwargs["metadata_buffers"] is None
        assert (
            decode_prealloc_factory.call_args.kwargs[
                "req_to_metadata_buffer_idx_allocator"
            ]
            is None
        )
        assert decode_prealloc_factory.call_args.kwargs["metadata_buffers"] is None
        scheduler._activate_terminal_kv_manager.assert_called_once_with(
            prealloc_manager
        )
        prefill_factory.assert_not_called()
        return

    assert (
        prefill_factory.call_args.kwargs["req_to_metadata_buffer_idx_allocator"] is None
    )
    assert prefill_factory.call_args.kwargs["metadata_buffers"] is None
    scheduler._activate_terminal_kv_manager.assert_called_once_with(prefill_manager)
    decode_transfer_factory.assert_not_called()
    decode_prealloc_factory.assert_not_called()


def test_terminal_dflash_boundary_rejects_eagle_schema_reuse() -> None:
    """A terminal deployment cannot silently reuse EAGLE auxiliary metadata."""

    with pytest.raises(ValueError, match="EAGLE auxiliary state"):
        _uses_terminal_dflash_boundary(
            _server_args("prefill"),
            _spec_algorithm(dflash=False),
        )


def test_nonterminal_serving_retains_legacy_metadata_schema() -> None:
    """The boundary rule leaves existing nonterminal disaggregation unchanged."""

    server_args = _server_args("prefill")
    server_args.pd_terminal_local_membership = None

    assert not _uses_terminal_dflash_boundary(
        server_args,
        _spec_algorithm(dflash=False),
    )


def test_terminal_queues_reject_legacy_metadata_entry_points() -> None:
    """Absent resources fail closed before a legacy row can be touched."""

    prealloc_queue = object.__new__(DecodePreallocQueue)
    prealloc_queue.req_to_metadata_buffer_idx_allocator = None
    transfer_queue = object.__new__(DecodeTransferQueue)
    transfer_queue.req_to_metadata_buffer_idx_allocator = None
    transfer_queue.metadata_buffers = None

    with pytest.raises(DecodeAllocationLeaseError, match="cannot allocate"):
        prealloc_queue._require_legacy_metadata_allocator()
    with pytest.raises(DecodeAllocationLeaseError, match="legacy metadata path"):
        transfer_queue._require_legacy_metadata_resources()


def test_terminal_decode_kv_args_keep_draft_kv_without_auxiliary_dram() -> None:
    """The decode manager advertises complete draft KV and zero legacy aux rows."""

    target_pool = SimpleNamespace(
        page_size=64,
        get_contiguous_buf_infos=MagicMock(return_value=([0x1000], [1024], [64])),
    )
    draft_pool = SimpleNamespace(
        get_contiguous_buf_infos=MagicMock(return_value=([0x2000], [512], [32])),
    )
    kv_args = SimpleNamespace()
    kv_args_factory = MagicMock(return_value=kv_args)
    manager = object()
    manager_factory = MagicMock(return_value=manager)
    queue = object.__new__(DecodePreallocQueue)
    queue.transfer_backend = TransferBackend.NIXL
    queue.tp_rank = 0
    queue.pp_rank = 0
    queue.scheduler = SimpleNamespace(
        ps=SimpleNamespace(dp_rank=0, gpu_id=0),
        enable_hisparse=False,
        model_config=SimpleNamespace(num_hidden_layers=60),
        server_args=SimpleNamespace(disaggregation_ib_device=None),
    )
    queue.token_to_kv_pool = target_pool
    queue.draft_token_to_kv_pool = draft_pool
    queue.req_to_token_pool = SimpleNamespace(size=48)
    queue.metadata_buffers = None
    queue.is_mla_backend = False
    queue.enable_staging = False

    def kv_class(_backend: TransferBackend, class_type: KVClassType) -> object:
        """Return the exact fake class for one requested NIXL role.

        :param _backend: Transfer backend selected by the queue.
        :param class_type: Requested KV class role.
        :returns: Matching fake class factory.
        """

        if class_type is KVClassType.KVARGS:
            return kv_args_factory
        if class_type is KVClassType.MANAGER:
            return manager_factory
        raise AssertionError(f"unexpected KV class type {class_type}")

    with (
        patch(
            "sglang.srt.disaggregation.decode.get_parallel",
            return_value=SimpleNamespace(attn_tp_size=1),
        ),
        patch(
            "sglang.srt.disaggregation.decode.get_kv_class",
            side_effect=kv_class,
        ),
        patch(
            "sglang.srt.disaggregation.decode.resolve_kv_layer_ids",
            side_effect=([3], [59]),
        ),
        patch("sglang.srt.disaggregation.decode.setup_state_kv_args") as setup_state,
    ):
        actual_manager = queue._init_kv_manager()

    assert actual_manager is manager
    assert kv_args.kv_data_ptrs == [0x1000, 0x2000]
    assert kv_args.kv_data_lens == [1024, 512]
    assert kv_args.kv_item_lens == [64, 32]
    assert kv_args.kv_layer_ids == [3, 59]
    assert kv_args.kv_data_mem_kinds == ["VRAM", "VRAM"]
    assert kv_args.aux_data_ptrs == []
    assert kv_args.aux_data_lens == []
    assert kv_args.aux_item_lens == []
    assert kv_args.terminal_request_capacity == 48
    setup_state.assert_called_once()


def test_terminal_dflash_rejects_legacy_sampling_mask_metadata() -> None:
    """Sampling-mask requests fail cleanly without dereferencing absent buffers."""

    scheduler = SimpleNamespace(
        _maybe_namespace_elastic_radix_cache=MagicMock(),
        spec_algorithm=MagicMock(),
        enable_overlap=True,
        disaggregation_mode=DisaggregationMode.DECODE,
        disagg_metadata_buffers=None,
        init_req_max_new_tokens=MagicMock(),
    )
    scheduler.spec_algorithm.is_dflash_family.return_value = True
    request = SimpleNamespace(
        return_sampling_mask=True,
        sampling_params=SimpleNamespace(top_k=4),
        set_finish_with_abort=MagicMock(),
    )

    with patch(
        "sglang.srt.managers.scheduler.validate_dflash_request",
        return_value=None,
    ):
        accepted = Scheduler._prepare_generate_request(
            scheduler,
            SimpleNamespace(),
            request,
        )

    assert not accepted
    request.set_finish_with_abort.assert_called_once_with(
        "return_sampling_mask is unavailable with terminal DFlash because its "
        "boundary schema carries no sampling-mask metadata."
    )
    scheduler.init_req_max_new_tokens.assert_called_once_with(request)
