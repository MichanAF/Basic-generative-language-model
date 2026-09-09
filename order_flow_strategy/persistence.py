"""On-disk state and audit journals for anything that runs unattended.

Two things a scheduled bot must get right, and both are boring:

*Writing state.* A crash halfway through a write leaves a truncated file. The
next start then reads garbage, and a bot that cannot read its own state must
either guess or refuse. Writing to a temporary file and renaming makes the
swap atomic on POSIX, so the file on disk is always a complete state.

*Reading state.* On a parse error the tempting thing is to start fresh. Do not.
Starting fresh means declaring yourself flat, and if you were in fact holding a
position you now have an untracked one in the market with nothing watching it.
Failing loudly is strictly better than a confident wrong answer.

The journal is append-only JSONL: one self-describing record per line, readable
by anything, and no rewrite path that could lose earlier lines.
"""

import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict, List, Optional, TypeVar

T = TypeVar("T")


def atomic_write_json(path: str, payload: Any) -> None:
    """Write JSON so that a crash mid-write cannot corrupt the file."""
    if is_dataclass(payload) and not isinstance(payload, type):
        payload = asdict(payload)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read_json_strict(path: str, what: str = "state") -> Optional[Dict[str, Any]]:
    """Return the object at ``path``, ``None`` if absent, raise if unreadable.

    The three outcomes are deliberately distinct. Absent means "first run".
    Unreadable means "stop, a human needs to look at this" -- never "assume
    nothing was there".
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(
            f"{path} exists but could not be read ({exc}). Refusing to start "
            f"with an unknown {what} -- inspect the file, do not delete it "
            "blindly."
        ) from exc
    if not isinstance(raw, dict):
        raise RuntimeError(
            f"{path} does not contain a {what} object. Refusing to start with "
            f"an unknown {what}."
        )
    return raw


def load_dataclass(path: str, factory: Callable[..., T], what: str = "state") -> Optional[T]:
    """Rebuild a dataclass from ``path``, dropping fields it no longer has.

    Dropping unknown keys is what lets a state file written by an older version
    still load. The alternative -- refusing -- turns every added field into a
    manual migration on a live box.
    """
    raw = read_json_strict(path, what)
    if raw is None:
        return None
    known = set(factory.__dataclass_fields__)  # type: ignore[attr-defined]
    return factory(**{k: v for k, v in raw.items() if k in known})


def append_jsonl(path: str, record: Any) -> None:
    """Append one record. Creates the directory and file as needed."""
    if is_dataclass(record) and not isinstance(record, type):
        record = asdict(record)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read every record. A malformed line raises rather than being skipped."""
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path} line {n} is not valid JSON: {exc}") from exc
    return out
