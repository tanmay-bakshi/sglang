import unittest

from sglang.srt.session.errors import StreamingSessionStaleEpochError
from sglang.srt.session.fencing import (
    CLUSTER_INCARNATION_HEADER,
    FENCING_EPOCH_HEADER,
    SessionFencingRegister,
    SessionFencingState,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class SessionFencingRegisterTest(unittest.TestCase):
    """Atomic session mutation fencing semantics."""

    def test_zero_pair_is_unfenced(self) -> None:
        """Allow epoch-zero and omitted headers before a register is installed."""
        register = SessionFencingRegister()

        register.validate(None)
        register.validate(0)

        self.assertEqual(register.state, SessionFencingState())
        self.assertFalse(register.state.installed)

    def test_missing_and_lower_epochs_are_stale_after_install(self) -> None:
        """Treat a missing header as epoch zero under an installed fence."""
        register = SessionFencingRegister()
        register.install(epoch=5, cluster_incarnation=17)

        for request_epoch in (None, 0, 4):
            with self.subTest(request_epoch=request_epoch):
                with self.assertRaises(StreamingSessionStaleEpochError) as raised:
                    register.validate(
                        request_epoch,
                        lineage_generation=3,
                        observed_tip=128,
                    )
                error = raised.exception
                self.assertEqual(error.request_epoch, request_epoch or 0)
                self.assertEqual(error.registered_epoch, 5)
                self.assertEqual(error.cluster_incarnation, 17)
                self.assertEqual(error.lineage_generation, 3)
                self.assertEqual(error.observed_tip, 128)

    def test_equal_and_higher_epochs_pass_without_advancing_register(self) -> None:
        """Keep request epochs separate from administrative installation."""
        register = SessionFencingRegister()
        installed = register.install(epoch=5, cluster_incarnation=17)

        register.validate(5)
        register.validate(99)

        self.assertEqual(register.state, installed)

    def test_install_moves_the_exact_pair_and_zero_resets_to_unfenced(self) -> None:
        """Install only through the explicit administrative operation."""
        register = SessionFencingRegister()

        self.assertEqual(
            register.install(epoch=7, cluster_incarnation=41),
            SessionFencingState(epoch=7, cluster_incarnation=41),
        )
        self.assertEqual(register.install(0, 0), SessionFencingState())
        self.assertFalse(register.state.installed)

    def test_negative_values_are_rejected(self) -> None:
        """Reject invalid request and register domains."""
        register = SessionFencingRegister()

        with self.assertRaises(ValueError):
            register.install(-1, 0)
        with self.assertRaises(ValueError):
            register.install(0, -1)
        with self.assertRaises(ValueError):
            register.validate(-1)

    def test_response_headers_echo_both_register_integers(self) -> None:
        """Encode the current pair consistently for every session response."""
        state = SessionFencingState(epoch=12, cluster_incarnation=34)

        self.assertEqual(
            state.response_headers(),
            {
                FENCING_EPOCH_HEADER: "12",
                CLUSTER_INCARNATION_HEADER: "34",
            },
        )


if __name__ == "__main__":
    unittest.main()
