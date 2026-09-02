"""Test-only lifecycle pause points and idle ownership evidence.

Everything here is inert unless ``SGLANG_PHASEC_INJECTION_DIR`` names a
directory. With the gate present, a qualification harness arms exactly one
``(trial, point, rank)`` by writing ``arm.json``; the armed scheduler rank
writes a marker binding that arm, then waits for the trial's release file so
the harness can suspend and resume the process at an exact transition seam.
The harness may also write an ``evidence-request`` tag; each rank answers once
per tag with one ``PHASEC_IDLE_EVIDENCE`` log line carrying its complete pool,
tree, and session ownership state.

These hooks supply evidence and a pause. They never alter ownership,
allocation, votes, commits, or publication, and the module must never be
enabled in a production launch.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_GATE_ENV = "SGLANG_PHASEC_INJECTION_DIR"
_SOURCE_IDENTITY_ENV = "PHASEC_SOURCE_IDENTITY_SHA256"
_MAX_WAIT_SECONDS = 180.0
_POLL_SECONDS = 0.002

_injection_dir: Path | None = (
    Path(os.environ[_GATE_ENV]) if len(os.environ.get(_GATE_ENV, "")) > 0 else None
)
_source_identity: str = os.environ.get(_SOURCE_IDENTITY_ENV, "unset")
_answered_evidence_tags: dict[int, str] = {}


def injection_enabled() -> bool:
    """Return whether the environment gate is present.

    :returns: Whether pause points and evidence emission are active.
    """
    return _injection_dir is not None


def process_start_ticks() -> int:
    """Read this process's start time in clock ticks since boot.

    :returns: Field 22 of ``/proc/self/stat``, or 0 where unavailable.
    """
    try:
        stat = Path("/proc/self/stat").read_text(encoding="utf-8")
    except OSError:
        return 0
    fields = stat[stat.rfind(")") + 2 :].split()
    return int(fields[19]) if len(fields) > 19 else 0


def _read_arm() -> dict[str, Any] | None:
    """Read the current arm record, if any.

    :returns: The arm record, or ``None`` when absent or unreadable.
    """
    assert _injection_dir is not None
    try:
        raw = (_injection_dir / "arm.json").read_text(encoding="utf-8")
        record = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(record, dict):
        return None
    return record


def _write_atomically(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON document through a temporary file and rename.

    :param path: Final destination.
    :param payload: JSON-serializable record.
    """
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def pause_point(point: str, tp_rank: int, state: dict[str, Any]) -> None:
    """Pause at one named lifecycle seam if the harness armed it for this rank.

    :param point: Stable seam name known to the harness.
    :param tp_rank: This scheduler's tensor-parallel rank.
    :param state: Point-specific evidence recorded in the marker.
    """
    if _injection_dir is None:
        return
    arm = _read_arm()
    if arm is None:
        return
    if arm.get("point") != point or arm.get("target_rank") != tp_rank:
        return
    trial_id = arm.get("trial_id")
    if not isinstance(trial_id, str) or len(trial_id) == 0:
        return
    markers = _injection_dir / "markers"
    markers.mkdir(parents=True, exist_ok=True)
    claim = markers / f"{trial_id}-tp{tp_rank}.claim"
    try:
        with claim.open("x", encoding="utf-8") as handle:
            handle.write(f"{os.getpid()}\n")
    except FileExistsError:
        return

    marker = {
        "trial_id": trial_id,
        "point": point,
        "tp_rank": tp_rank,
        "nonce": arm.get("nonce"),
        "arm_sha256": arm.get("arm_sha256"),
        "engine_commit": arm.get("engine_commit"),
        "engine_diff_sha256": arm.get("engine_diff_sha256"),
        "source_identity_sha256": arm.get("source_identity_sha256"),
        "pid": os.getpid(),
        "process_start_ticks": process_start_ticks(),
        "utc": datetime.now(timezone.utc).isoformat(),
        "monotonic_ns": time.monotonic_ns(),
        "state": state,
    }
    _write_atomically(markers / f"{trial_id}-tp{tp_rank}.json", marker)

    release = _injection_dir / "releases" / f"{trial_id}.release"
    requested_wait = arm.get("max_wait_seconds")
    wait_seconds = (
        min(float(requested_wait), _MAX_WAIT_SECONDS)
        if isinstance(requested_wait, (int, float)) and requested_wait > 0
        else _MAX_WAIT_SECONDS
    )
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if release.exists():
            return
        time.sleep(_POLL_SECONDS)
    logger.error(
        "Lifecycle pause point %s on TP rank %d for trial %s was never released "
        "within %.0fs; continuing without the harness.",
        point,
        tp_rank,
        trial_id,
        wait_seconds,
    )


def take_idle_evidence_request(tp_rank: int) -> str | None:
    """Return a pending evidence tag this rank has not answered yet.

    :param tp_rank: This scheduler's tensor-parallel rank.
    :returns: The tag to answer, or ``None`` when nothing new is requested.
    """
    if _injection_dir is None:
        return None
    try:
        tag = (_injection_dir / "evidence-request").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if len(tag) == 0 or _answered_evidence_tags.get(tp_rank) == tag:
        return None
    _answered_evidence_tags[tp_rank] = tag
    return tag


def emit_idle_evidence(tag: str, tp_rank: int, record: dict[str, Any]) -> None:
    """Log one rank's idle evidence on a single line the harness can parse.

    :param tag: Evidence request tag being answered.
    :param tp_rank: This scheduler's tensor-parallel rank.
    :param record: Complete ownership evidence for this rank.
    """
    payload = {
        "tag": tag,
        "tp_rank": tp_rank,
        "pid": os.getpid(),
        "process_start_ticks": process_start_ticks(),
        "source_identity_sha256": _source_identity,
        "utc": datetime.now(timezone.utc).isoformat(),
        "monotonic_ns": time.monotonic_ns(),
        **record,
    }
    logger.info(
        "PHASEC_IDLE_EVIDENCE %s",
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
    )
