import asyncio

import requests
from aiohttp import web
from sglang.srt.disaggregation.terminal_progress.startup_cohort import (
    TerminalStartupCohortError,
    TerminalStartupCohortMatrix,
    TerminalStartupCohortRegistry,
    TerminalStartupRankAdvertisement,
    decode_terminal_startup_cohort_matrix,
    decode_terminal_startup_rank_advertisement,
    encode_terminal_startup_cohort_matrix,
    encode_terminal_startup_rank_advertisement,
)

TERMINAL_STARTUP_ROUTE: str = "/terminal-startup"


async def handle_terminal_startup_join(
    registry: TerminalStartupCohortRegistry,
    request: web.Request,
) -> web.Response:
    """Join one authenticated native rank to the deployment epoch.

    The registry wait runs outside the HTTP event loop. Other rank joins must
    remain serviceable while earlier handlers sleep on the event-driven cohort
    barrier.

    :param registry: Exact source-owned startup registry.
    :param request: Incoming canonical rank advertisement.
    :returns: Canonical sealed rank matrix or bounded failure evidence.
    """

    if type(registry) is not TerminalStartupCohortRegistry:
        raise TypeError("registry must be TerminalStartupCohortRegistry")
    try:
        advertisement = decode_terminal_startup_rank_advertisement(await request.read())
        matrix = await asyncio.to_thread(
            registry.register_and_wait,
            advertisement,
        )
    except TerminalStartupCohortError as error:
        return web.Response(text=str(error), status=409)
    return web.Response(
        body=encode_terminal_startup_cohort_matrix(matrix),
        content_type="application/json",
        status=200,
    )


def join_terminal_startup_cohort(
    endpoint: str,
    advertisement: TerminalStartupRankAdvertisement,
    timeout_seconds: float,
) -> TerminalStartupCohortMatrix:
    """Join one rank and require the complete canonical observed matrix.

    The launcher proves listener liveness before model ranks enter this call,
    so connection failure is a deployment failure. There is deliberately no
    retry loop or mutable registration fallback.

    :param endpoint: Exact HTTP route owned by the cohort's prefill service.
    :param advertisement: Local generation-bound native identity.
    :param timeout_seconds: Hash-bound startup control deadline.
    :returns: Complete deployment-epoch matrix.
    :raises TerminalStartupCohortError: If transport or cohort admission fails.
    """

    if type(endpoint) is not str or not endpoint.endswith(TERMINAL_STARTUP_ROUTE):
        raise ValueError("endpoint must select the terminal startup route")
    if type(advertisement) is not TerminalStartupRankAdvertisement:
        raise TypeError("advertisement must be TerminalStartupRankAdvertisement")
    if type(timeout_seconds) is not float or timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be a positive float")
    try:
        response = requests.post(
            endpoint,
            data=encode_terminal_startup_rank_advertisement(advertisement),
            headers={"Content-Type": "application/json"},
            timeout=timeout_seconds,
        )
    except requests.RequestException as error:
        raise TerminalStartupCohortError(
            "terminal startup join transport failed"
        ) from error
    if response.status_code != 200:
        reason = response.text[:512]
        raise TerminalStartupCohortError(
            f"terminal startup join failed with HTTP {response.status_code}: {reason}"
        )
    matrix = decode_terminal_startup_cohort_matrix(response.content)
    if matrix.cohort_sha256 != advertisement.cohort_sha256:
        raise TerminalStartupCohortError(
            "terminal startup response belongs to another deployment epoch"
        )
    return matrix
