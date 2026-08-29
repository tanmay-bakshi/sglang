"""Manual entrypoint for raw-token streaming-session qualification stages."""

import argparse

from sglang.test.kits.gemma4_streaming_session_token_api_kit import (
    run_commit_qualification,
    run_recovery_qualification,
    run_truncate_qualification,
)


def main() -> None:
    """Run one requested live qualification stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:32300")
    parser.add_argument(
        "--stage",
        choices=("truncate", "commit", "recovery"),
        required=True,
    )
    args = parser.parse_args()

    if args.stage == "truncate":
        run_truncate_qualification(args.base_url)
    elif args.stage == "commit":
        run_commit_qualification(args.base_url)
    else:
        run_recovery_qualification(args.base_url)
    print(f"streaming-session {args.stage} qualification passed")


if __name__ == "__main__":
    main()
