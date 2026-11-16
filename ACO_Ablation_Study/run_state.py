"""Read-only inspection of one ablation run directory.

The batch runner uses this to decide between skip / resume / conflict without
launching a solve.  A run counts as complete only when:

* ``run.json`` exists, parses, and carries ``status == "completed"``;
* the recorded run-configuration fingerprint matches today's configuration;
* the recorded source-code fingerprint matches the current study sources;
* the recorded dataset hash matches the dataset file on disk;
* ``makespan_s`` is finite and ``feasible`` is true.

Anything else is reported as partial or conflicting; the caller must never
silently mix mismatched results.
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, Optional

from ablation_config import (
    HISTORY_FILENAME,
    RUN_FILENAME,
    SOLUTION_FILENAME,
    STATUS_COMPLETED,
    file_sha256,
)

MISSING = "missing"
PARTIAL = "partial"
COMPLETE = "complete"
CONFIG_MISMATCH = "config_mismatch"
CODE_MISMATCH = "code_mismatch"
DATASET_MISMATCH = "dataset_mismatch"
INVALID = "invalid"


def read_run_json(run_dir: str) -> Optional[Dict[str, object]]:
    path = os.path.join(run_dir, RUN_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def inspect_run(
    run_dir: str,
    expected_run_fingerprint: Optional[str] = None,
    expected_code_fingerprint: Optional[str] = None,
    expected_dataset_sha256: Optional[str] = None,
) -> Dict[str, object]:
    """Return ``{state, detail, payload}`` for a run directory."""
    if not os.path.isdir(run_dir):
        return {"state": MISSING, "detail": "run directory does not exist", "payload": None}
    payload = read_run_json(run_dir)
    if payload is None:
        return {"state": PARTIAL, "detail": "run.json missing or unreadable", "payload": None}
    if payload.get("status") != STATUS_COMPLETED:
        return {
            "state": PARTIAL,
            "detail": "status=%r" % payload.get("status"),
            "payload": payload,
        }
    # run.json is written last; these two files are part of a complete record.
    for name in (SOLUTION_FILENAME, HISTORY_FILENAME):
        if not os.path.isfile(os.path.join(run_dir, name)):
            return {
                "state": PARTIAL,
                "detail": "completed run is missing %s" % name,
                "payload": payload,
            }
    if expected_run_fingerprint is not None:
        recorded = payload.get("run_config_fingerprint")
        if recorded != expected_run_fingerprint:
            return {
                "state": CONFIG_MISMATCH,
                "detail": "recorded run fingerprint %r != expected %r"
                          % (recorded, expected_run_fingerprint),
                "payload": payload,
            }
    if expected_code_fingerprint is not None:
        recorded = payload.get("code_fingerprint")
        if recorded != expected_code_fingerprint:
            return {
                "state": CODE_MISMATCH,
                "detail": "recorded code fingerprint %r != current %r"
                          % (recorded, expected_code_fingerprint),
                "payload": payload,
            }
    if expected_dataset_sha256 is not None:
        recorded = payload.get("dataset_sha256")
        if recorded != expected_dataset_sha256:
            return {
                "state": DATASET_MISMATCH,
                "detail": "recorded dataset hash %r != current %r"
                          % (recorded, expected_dataset_sha256),
                "payload": payload,
            }
    makespan = payload.get("makespan_s")
    if not isinstance(makespan, (int, float)) or not math.isfinite(float(makespan)):
        return {
            "state": INVALID,
            "detail": "non-finite makespan_s=%r" % (makespan,),
            "payload": payload,
        }
    if not bool(payload.get("feasible")):
        return {
            "state": INVALID,
            "detail": "recorded run is infeasible",
            "payload": payload,
        }
    return {"state": COMPLETE, "detail": "ok", "payload": payload}


def dataset_hash_of(payload: Dict[str, object]) -> Optional[str]:
    value = payload.get("dataset_sha256")
    return str(value) if value else None


def current_dataset_hash(path: str) -> str:
    return file_sha256(path)
