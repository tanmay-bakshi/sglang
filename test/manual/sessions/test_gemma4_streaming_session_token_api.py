"""Production Gemma-4 raw-token streaming-session qualification matrix."""

import unittest

from sglang.test.kits.gemma4_streaming_session_token_api_kit import (
    Gemma4StreamingSessionFullKitMixin,
    Gemma4StreamingSessionOracleKitMixin,
)
from sglang.test.server_fixtures.gemma4_streaming_session_fixture import (
    Gemma4StreamingSessionArm,
    Gemma4StreamingSessionServerBase,
)


class TestGemma4DFlashOneShotStreamingSession(
    Gemma4StreamingSessionServerBase,
    Gemma4StreamingSessionFullKitMixin,
):
    """Full production qualification with DFlash and one-shot prefill."""

    arm = Gemma4StreamingSessionArm(
        name="dflash-one-shot",
        port=32_310,
        chunked_prefill_size=16_384,
        use_dflash=True,
    )


class TestGemma4DFlashChunkedStreamingSession(
    Gemma4StreamingSessionServerBase,
    Gemma4StreamingSessionOracleKitMixin,
):
    """Greedy oracle with DFlash and 1,024-token prefill chunks."""

    arm = Gemma4StreamingSessionArm(
        name="dflash-chunked",
        port=32_311,
        chunked_prefill_size=1_024,
        use_dflash=True,
    )


class TestGemma4NoSpecOneShotStreamingSession(
    Gemma4StreamingSessionServerBase,
    Gemma4StreamingSessionOracleKitMixin,
):
    """Greedy oracle without speculation and with one-shot prefill."""

    arm = Gemma4StreamingSessionArm(
        name="no-spec-one-shot",
        port=32_312,
        chunked_prefill_size=16_384,
        use_dflash=False,
    )


class TestGemma4NoSpecChunkedStreamingSession(
    Gemma4StreamingSessionServerBase,
    Gemma4StreamingSessionOracleKitMixin,
):
    """Greedy oracle without speculation and with 1,024-token prefill chunks."""

    arm = Gemma4StreamingSessionArm(
        name="no-spec-chunked",
        port=32_313,
        chunked_prefill_size=1_024,
        use_dflash=False,
    )


if __name__ == "__main__":
    unittest.main()
