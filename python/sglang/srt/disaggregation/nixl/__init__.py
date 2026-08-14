from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.disaggregation.nixl.conn import (
        NixlKVBootstrapServer,
        NixlKVManager,
        NixlKVReceiver,
        NixlKVSender,
    )

__all__ = (
    "NixlKVBootstrapServer",
    "NixlKVManager",
    "NixlKVReceiver",
    "NixlKVSender",
)


def __getattr__(name: str) -> object:
    """Load public manager classes without importing them for every submodule.

    :param name: Requested package attribute.
    :returns: Exact public NIXL implementation class.
    :raises AttributeError: If the package does not export ``name``.
    """

    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from sglang.srt.disaggregation.nixl.conn import (
        NixlKVBootstrapServer,
        NixlKVManager,
        NixlKVReceiver,
        NixlKVSender,
    )

    exports = {
        "NixlKVBootstrapServer": NixlKVBootstrapServer,
        "NixlKVManager": NixlKVManager,
        "NixlKVReceiver": NixlKVReceiver,
        "NixlKVSender": NixlKVSender,
    }
    return exports[name]
