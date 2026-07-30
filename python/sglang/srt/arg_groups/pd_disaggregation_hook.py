import ipaddress
import logging
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

_CANONICAL_DIGEST = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class PdProcessRuntimeCapabilities:
    """Implementation-owned PD transfer and control capabilities.

    :ivar kv_dtype: Canonical runtime KV storage dtype.
    :ivar page_size: Runtime KV page size.
    :ivar kv_transfer_protocol: Implemented KV-transfer wire protocol.
    :ivar prepared_grant_protocol: Implemented decoder-grant control protocol.
    """

    kv_dtype: str
    page_size: int
    kv_transfer_protocol: str
    prepared_grant_protocol: str


def _validate_fingerprint(value: str | None, flag: str) -> str:
    """Validate one operator-owned compatibility fingerprint.

    :param value: Candidate fingerprint.
    :param flag: CLI flag used in validation errors.
    :returns: The canonical fingerprint.
    """
    if value is None or _CANONICAL_DIGEST.fullmatch(value) is None:
        raise ValueError(
            f"{flag} must be a canonical lowercase 64-character hex digest"
        )
    return value


def _validate_bootstrap_host(value: str | None) -> str:
    """Validate an explicitly advertised non-local bootstrap host.

    :param value: Candidate DNS name or IP literal.
    :returns: The validated host.
    """
    if (
        value is None
        or len(value) == 0
        or value.strip() != value
        or any(
            ord(character) < 32 or 127 <= ord(character) <= 159 for character in value
        )
        or "://" in value
        or "/" in value
    ):
        raise ValueError(
            "--pd-prefill-bootstrap-advertise-host must be an explicit "
            "non-local host without a scheme or port"
        )

    normalized = value.lower().rstrip(".")
    if normalized == "localhost" or normalized.endswith(".localhost"):
        raise ValueError(
            "--pd-prefill-bootstrap-advertise-host cannot advertise localhost"
        )

    address_value = (
        value[1:-1] if value.startswith("[") and value.endswith("]") else value
    )
    try:
        address = ipaddress.ip_address(address_value)
    except ValueError:
        if ":" in value:
            raise ValueError(
                "--pd-prefill-bootstrap-advertise-host must bracket IPv6 literals"
            ) from None
        return value

    if address.is_loopback or address.is_unspecified or address.is_multicast:
        raise ValueError(
            "--pd-prefill-bootstrap-advertise-host must be non-local and usable"
        )
    return value


def build_pd_process_advertisement(
    server_args: "ServerArgs",
    *,
    runtime_capabilities: PdProcessRuntimeCapabilities | None,
) -> dict[str, object] | None:
    """Build the versioned PD process-generation capability contract.

    :param server_args: Fully resolved server configuration.
    :param runtime_capabilities: Runtime-proven transfer and control capabilities.
    :returns: The advertisement for a PD process, otherwise ``None``.
    """
    if server_args.disaggregation_mode not in ("prefill", "decode"):
        return None
    if runtime_capabilities is None:
        return None
    if server_args.api_key is None or len(server_args.api_key) == 0:
        raise ValueError("PD process advertisement requires --api-key")
    if server_args.tokenizer_worker_num != 1:
        raise ValueError("PD process advertisement requires --tokenizer-worker-num 1")

    model_fingerprint = _validate_fingerprint(
        server_args.pd_model_fingerprint, "--pd-model-fingerprint"
    )
    logical_kv_layout_fingerprint = _validate_fingerprint(
        server_args.pd_logical_kv_layout_fingerprint,
        "--pd-logical-kv-layout-fingerprint",
    )
    if runtime_capabilities.kv_transfer_protocol != "packed-v4":
        raise ValueError(
            "PD process advertisement requires implemented packed-v4 transfer"
        )
    if runtime_capabilities.prepared_grant_protocol != "control-v1":
        raise ValueError(
            "PD process advertisement requires implemented control-v1 grants"
        )
    kv_dtype = runtime_capabilities.kv_dtype
    if (
        len(kv_dtype) == 0
        or kv_dtype == "auto"
        or not kv_dtype.replace("_", "").isalnum()
        or kv_dtype.lower() != kv_dtype
    ):
        raise ValueError(
            "PD process advertisement requires a canonical runtime KV dtype"
        )
    canonical_kv_dtype = {
        "bfloat16": "bf16",
        "fp4_mx_block16": "nvfp4",
    }.get(kv_dtype, kv_dtype)
    page_size = runtime_capabilities.page_size
    if not isinstance(page_size, int) or page_size <= 0:
        raise ValueError(
            "PD process advertisement requires a resolved positive page_size"
        )
    if server_args.tp_size <= 0:
        raise ValueError("PD process advertisement requires a positive tp_size")
    if server_args.dp_size != 1:
        raise ValueError(
            "PD process advertisement supports only DP1, "
            f"received DP{server_args.dp_size}"
        )

    bootstrap_endpoint = None
    if server_args.disaggregation_mode == "prefill":
        bootstrap_host = _validate_bootstrap_host(
            server_args.pd_prefill_bootstrap_advertise_host
        )
        if not 1 <= server_args.disaggregation_bootstrap_port <= 65535:
            raise ValueError(
                "--disaggregation-bootstrap-port must be between 1 and 65535"
            )
        bootstrap_endpoint = {
            "host": bootstrap_host,
            "port": server_args.disaggregation_bootstrap_port,
        }
    elif server_args.pd_prefill_bootstrap_advertise_host is not None:
        raise ValueError(
            "decode processes cannot set --pd-prefill-bootstrap-advertise-host"
        )

    return {
        "schema": "v1",
        "launch_instance_id": server_args.launch_instance_id,
        "role": server_args.disaggregation_mode,
        "tensor_parallel_size": server_args.tp_size,
        "data_parallel_size": server_args.dp_size,
        "model_fingerprint": model_fingerprint,
        "logical_kv_layout_fingerprint": logical_kv_layout_fingerprint,
        "kv_dtype": canonical_kv_dtype,
        "page_size": page_size,
        "kv_transfer_protocol": runtime_capabilities.kv_transfer_protocol,
        "prepared_grant_protocol": runtime_capabilities.prepared_grant_protocol,
        "prefill_bootstrap_endpoint": bootstrap_endpoint,
    }


def handle_pd_disaggregation(server_args: "ServerArgs") -> None:
    """Validate and normalize PD-disaggregation server args.

    :param server_args: Fully resolved server configuration.
    """
    # "mooncake_tcp" is mooncake with the TCP transport forced: set MC_FORCE_TCP
    # so mooncake installs TcpTransport instead of RDMA, rewrite the backend to
    # mooncake, and skip RDMA HCA selection. Must run before backend-name checks.
    if server_args.disaggregation_transfer_backend == "mooncake_tcp":
        os.environ.setdefault("MC_FORCE_TCP", "1")
        server_args.disaggregation_transfer_backend = "mooncake"
        server_args.disaggregation_ib_device = None
        logger.info(
            "disaggregation transfer backend 'mooncake_tcp' -> mooncake "
            "with MC_FORCE_TCP=1 (TCP transport, no RDMA)"
        )

    if server_args.disaggregation_mode == "decode":
        if server_args.disaggregation_decode_enable_radix_cache:
            if server_args.enable_hisparse:
                raise ValueError(
                    "--disaggregation-decode-enable-radix-cache is incompatible "
                    "with --enable-hisparse"
                )
            if server_args.disaggregation_transfer_backend == "fake":
                raise ValueError(
                    "--disaggregation-decode-enable-radix-cache is incompatible "
                    "with --disaggregation-transfer-backend fake"
                )
            if server_args.speculative_algorithm is not None:
                raise ValueError(
                    "--disaggregation-decode-enable-radix-cache is incompatible "
                    "with speculative decoding "
                    f"(--speculative-algorithm {server_args.speculative_algorithm})"
                )
            from sglang.srt.arg_groups.overrides import resolved_view

            if resolved_view(server_args).enable_dp_attention:
                logger.warning(
                    "EXPERIMENTAL: Decode radix cache with DP attention. "
                    "Requires prefix-aware DP rank routing for optimal cache hits."
                )
            server_args.disable_radix_cache = False
            logger.warning("EXPERIMENTAL: Radix cache is enabled for decode server")
        else:
            server_args.disable_radix_cache = True
            logger.warning("KV cache is forced as chunk cache for decode server")

        # Default the number of *extra* decode req_to_token slots reserved for
        # in-transfer (being-received-from-prefill) requests, on top of the
        # max_running_requests-derived pool. Large batches get none; small
        # per-worker batches reserve 2x the batch as cheap overlap headroom.
        if server_args.disaggregation_decode_extra_slots is None:
            extra_slots = 0
            if server_args.max_running_requests is not None:
                per_worker = server_args.max_running_requests // max(
                    1, server_args.dp_size
                )
                if per_worker <= 32:
                    extra_slots = per_worker * 2
            server_args.disaggregation_decode_extra_slots = extra_slots

    elif server_args.disaggregation_mode == "prefill":
        assert (
            server_args.disaggregation_transfer_backend != "fake"
        ), "Prefill server does not support 'fake' as the transfer backend"

    if (
        server_args.disaggregation_mode in ("prefill", "decode")
        and envs.SGLANG_DISAGG_STAGING_BUFFER.get()
        and server_args.disaggregation_transfer_backend not in ("mooncake", "nixl")
    ):
        raise ValueError(
            f"SGLANG_DISAGG_STAGING_BUFFER requires "
            f"disaggregation_transfer_backend='mooncake' or 'nixl', "
            f"got '{server_args.disaggregation_transfer_backend}'."
        )
