"""Small helpers for timing runs and preserving their input decks."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path


class Timer:
    """Accumulate wall time between matching ``start`` and ``stop`` calls."""

    def __init__(self) -> None:
        self._run_start = time.perf_counter()
        self._starts: dict[str, float] = {}
        self._seconds: dict[str, float] = {}
        self._calls: dict[str, int] = {}

    def start(self, name: str) -> None:
        if name in self._starts:
            raise RuntimeError(f"Timer {name!r} is already running.")
        self._starts[name] = time.perf_counter()

    def stop(self, name: str) -> None:
        try:
            started = self._starts.pop(name)
        except KeyError as error:
            raise RuntimeError(f"Timer {name!r} is not running.") from error
        self._seconds[name] = self._seconds.get(name, 0.0) + (
            time.perf_counter() - started
        )
        self._calls[name] = self._calls.get(name, 0) + 1

    def summary(self) -> dict:
        if self._starts:
            raise RuntimeError("Cannot summarize while a timer is running.")
        total = time.perf_counter() - self._run_start
        measured = sum(self._seconds.values())
        sections = {
            name: {"total_seconds": seconds, "calls": self._calls[name]}
            for name, seconds in self._seconds.items()
        }
        sections["other"] = {
            "total_seconds": max(0.0, total - measured),
            "calls": 1,
        }
        for values in sections.values():
            values["percent_of_total"] = (
                100.0 * values["total_seconds"] / total if total else 0.0
            )
        return {"total_seconds": total, "sections": sections}


def save_timing_summary(
    summary: dict,
    path: str | Path,
    *,
    metadata: dict[str, int | str],
) -> Path:
    """Write a timing summary to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({**metadata, **summary}, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def copy_used_config(source: str | Path, destination: str | Path) -> None:
    """Copy one input TOML unless it is already at the destination."""
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)


def format_timing_summary(summary: dict) -> str:
    """Return a compact timing table for terminal output."""
    lines = ["Timing summary:"]
    for name, values in summary["sections"].items():
        label = name.replace("_", " ").capitalize()
        lines.append(
            f"  {label:<24} {values['total_seconds']:9.3f} s "
            f"({values['percent_of_total']:5.1f}%)"
        )
    lines.append(f"  {'Total':<24} {summary['total_seconds']:9.3f} s")
    return "\n".join(lines)
