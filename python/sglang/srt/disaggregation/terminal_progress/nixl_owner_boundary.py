import abc
from collections.abc import Callable

from sglang.srt.disaggregation.terminal_progress.native_state import (
    NativeTerminalOwnerAction,
)


class NixlTerminalOwnerBoundary(abc.ABC):
    """Nominal process-lifetime owner for direct NIXL terminal delivery."""

    @abc.abstractmethod
    def arm_transfer(self, handle: object, binding_digest: bytes) -> object:
        """Arm terminal delivery before one exact transfer is posted.

        :param handle: Initialized but unposted NIXL transfer handle.
        :param binding_digest: Exact registered source lifecycle digest.
        :returns: Opaque exact-generation transfer authority.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def post_transfer(
        self,
        transfer: object,
        post: Callable[[object], object],
    ) -> object:
        """Post an already armed exact transfer.

        :param transfer: Exact-generation authority returned by
            :meth:`arm_transfer`.
        :param post: Existing NIXL post operation.
        :returns: Existing post operation result.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def settle_success(
        self,
        transfer: object,
        action: NativeTerminalOwnerAction,
    ) -> object:
        """Take completion authority after the matching owner action.

        :param transfer: Exact-generation transfer authority.
        :param action: Authoritative owner action for this lifecycle.
        :returns: Native completion receipt.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def settle_failure(
        self,
        transfer: object,
        action: NativeTerminalOwnerAction,
    ) -> None:
        """Settle terminal failure after the matching owner action.

        :param transfer: Exact-generation transfer authority.
        :param action: Authoritative quarantine or process-fatal action.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def cancel_transfer(self, transfer: object) -> None:
        """Request cancellation without releasing ambiguous authority.

        :param transfer: Exact-generation transfer authority.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def release_transfer(self, transfer: object) -> None:
        """Release one exact handle after terminal settlement.

        :param transfer: Settled exact-generation transfer authority.
        """

        raise NotImplementedError
