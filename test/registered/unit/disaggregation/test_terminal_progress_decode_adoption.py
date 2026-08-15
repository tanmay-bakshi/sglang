from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from sglang.srt.disaggregation.terminal_progress.decode_adoption import (
    TerminalDFlashDecodeAdoption,
)
from sglang.srt.disaggregation.terminal_progress.decode_wiring import (
    PackedTerminalDecodeWiring,
)


class _CompletionEvent:
    """Deterministic CUDA-event surface with observable terminality."""

    _order: list[str]
    _terminal_after_sync: bool

    def __init__(self, order: list[str], *, terminal_after_sync: bool = True) -> None:
        """Create one synthetic completion event.

        :param order: Shared lifecycle order ledger.
        :param terminal_after_sync: Value returned by :meth:`query`.
        """

        self._order = order
        self._terminal_after_sync = terminal_after_sync

    def synchronize(self) -> None:
        """Record the exact blocking completion boundary."""

        self._order.append("event_synchronize")

    def query(self) -> bool:
        """Return whether synchronization earned terminal success.

        :returns: Configured terminal state.
        """

        self._order.append("event_query")
        return self._terminal_after_sync


def _adoption(
    order: list[str],
    *,
    terminal_after_sync: bool = True,
) -> tuple[TerminalDFlashDecodeAdoption, object]:
    """Build a type-exact envelope around synthetic device authorities.

    :param order: Shared lifecycle order ledger.
    :param terminal_after_sync: Synthetic event terminal state.
    :returns: Adoption envelope and exact transaction row authority.
    """

    transaction_adoption = object()
    adoption = object.__new__(TerminalDFlashDecodeAdoption)
    object.__setattr__(adoption, "transaction_adoption", transaction_adoption)
    object.__setattr__(
        adoption,
        "device_value",
        SimpleNamespace(
            completion_event=_CompletionEvent(
                order,
                terminal_after_sync=terminal_after_sync,
            )
        ),
    )
    return adoption, transaction_adoption


def _wiring(order: list[str]) -> tuple[PackedTerminalDecodeWiring, MagicMock]:
    """Build the scheduler adoption boundary without native side effects.

    :param order: Shared lifecycle order ledger.
    :returns: Wiring instance and exact actor fixture.
    """

    owner = object()
    transaction = object()
    actor = MagicMock()
    actor.terminal_owner_transaction.return_value = transaction
    actor.consume_terminal_owner_adoption.return_value = owner

    def complete_metadata(
        actual_transaction: object,
        *,
        dflash_adoption: object,
    ) -> None:
        """Record row release only after terminal device-copy proof.

        :param actual_transaction: Exact actor transaction.
        :param dflash_adoption: Exact transaction-issued row authority.
        """

        assert actual_transaction is transaction
        assert order[-2:] == ["event_synchronize", "event_query"]
        order.append("metadata_row_release")

    actor.complete_terminal_owner_metadata_consumption.side_effect = complete_metadata
    wiring = object.__new__(PackedTerminalDecodeWiring)
    wiring._actor = actor
    wiring._runtime = MagicMock()
    wiring._timing = MagicMock()
    wiring._local_receipt_producer_id = 1
    wiring._local_producer_id = 2
    wiring._require_action = MagicMock()
    wiring._submit_local_failure = MagicMock()
    action_binding = SimpleNamespace(
        digest=b"d" * 32,
        owner=SimpleNamespace(tp_rank=0),
        to_binding=lambda: object(),
    )
    wiring._test_action = SimpleNamespace(binding=action_binding)
    wiring._test_owner = owner
    return wiring, actor


def test_decode_row_release_waits_for_exact_device_copy_adoption() -> None:
    """The exact adoption reaches row release only after its event is terminal."""

    order: list[str] = []
    adoption, transaction_adoption = _adoption(order)
    wiring, actor = _wiring(order)

    def adopt(owner: object) -> TerminalDFlashDecodeAdoption:
        """Return exact DFlash authority after request adoption.

        :param owner: Exact retained decode request.
        :returns: Exact device-copy adoption authority.
        """

        assert owner is wiring._test_owner
        order.append("scheduler_adopt")
        return adoption

    def finalize(owner: object) -> None:
        """Record scheduler visibility after metadata row release.

        :param owner: Exact retained decode request.
        """

        assert owner is wiring._test_owner
        order.append("scheduler_finalize")

    result = wiring.consume_adoption_action(
        wiring._test_action,
        adopt,
        finalize,
    )

    assert result is wiring._test_owner
    assert order == [
        "scheduler_adopt",
        "event_synchronize",
        "event_query",
        "metadata_row_release",
        "scheduler_finalize",
    ]
    keyword_adoption = actor.complete_terminal_owner_metadata_consumption.call_args.kwargs[
        "dflash_adoption"
    ]
    assert keyword_adoption is transaction_adoption


def test_nonterminal_device_copy_never_releases_the_dflash_row() -> None:
    """A nonterminal copy event fails closed before metadata row release."""

    order: list[str] = []
    adoption, _ = _adoption(order, terminal_after_sync=False)
    wiring, actor = _wiring(order)
    finalize = MagicMock()

    with pytest.raises(RuntimeError, match="not terminal"):
        wiring.consume_adoption_action(
            wiring._test_action,
            lambda owner: adoption,
            finalize,
        )

    assert order == ["event_synchronize", "event_query"]
    actor.complete_terminal_owner_metadata_consumption.assert_not_called()
    wiring._runtime.submit.assert_not_called()
    finalize.assert_not_called()
    actor.quarantine.assert_called_once()
