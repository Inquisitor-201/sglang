"""Structured diagnostic events for the L2/L3 cache path.

The hot path only checks a flag, extracts already available fields, and
enqueues a dict. JSON encoding, file rotation, and disk I/O happen on a
background thread. When no directory is configured every entry point is a
no-op and no thread or file is created.

This module intentionally imports only the standard library so that it can be
unit tested and imported from the cache paths without pulling in torch.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

DEFAULT_QUEUE_EVENTS = 65536
DEFAULT_QUEUE_BYTES = 256 << 20
DEFAULT_MAX_FILE_BYTES = 512 << 20
DEFAULT_KEEP_FILES = 8


@dataclass(frozen=True)
class DiagConfig:
    directory: str
    enabled: bool = True
    overflow_policy: str = "drop"
    queue_events: int = DEFAULT_QUEUE_EVENTS
    queue_bytes: int = DEFAULT_QUEUE_BYTES
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    keep_files: int = DEFAULT_KEEP_FILES

    def __post_init__(self):
        if self.overflow_policy not in ("drop", "block"):
            raise ValueError(
                f"overflow_policy must be 'drop' or 'block', got "
                f"{self.overflow_policy!r}"
            )
        if self.queue_events < 1 or self.queue_bytes < 1:
            raise ValueError("queue limits must be positive")


def estimate_event_bytes(event: Dict[str, Any]) -> int:
    """Cheap upper-bound-ish size estimate without serializing."""
    total = 32
    for value in event.values():
        if isinstance(value, str):
            total += len(value)
        elif isinstance(value, (list, tuple)):
            total += 8 * len(value)
            for item in value:
                if isinstance(item, str):
                    total += len(item)
        else:
            total += 8
    return total


def resolve_diag_directory(directory: str) -> str:
    """Resolve a relative diagnostic directory against the HyperX app root.

    Deployments share one env value across pods whose install root differs, so
    the configured directory is relative and each pod lands under its own
    Themis-readable logs tree.
    """
    if not directory or os.path.isabs(directory):
        return directory
    root = os.environ.get("HIPPO_APP_INST_ROOT")
    if root:
        return str(Path(root) / directory)
    return directory


def compact_indices(indices) -> str:
    """Render sorted page indices as a compact range list: 1-3,7,9-10."""
    values = sorted({int(i) for i in indices})
    if not values:
        return ""
    parts = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = value
    parts.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(parts)


def summarize_codes(codes, kind: str, max_failed: int = 32) -> Dict[str, Any]:
    """Summarize per-key Mooncake result codes without one line per key.

    kind="exists": 1 present, 0 absent, negative error.
    kind="put":    0 success, negative error (e.g. -705 already exists).
    kind="get":    positive bytes read, 0 not found, negative error.
    """
    summary: Dict[str, Any] = {"n": len(codes)}
    errors: Dict[str, int] = {}
    failed = []
    failed_total = 0
    if kind == "exists":
        ok = missing = 0
        first_missing = None
        missing_idx = []
        for index, code in enumerate(codes):
            if code == 1:
                ok += 1
            elif code == 0:
                missing += 1
                if first_missing is None:
                    first_missing = index
                if len(missing_idx) < max_failed:
                    missing_idx.append(index)
            else:
                failed_total += 1
                errors[str(int(code))] = errors.get(str(int(code)), 0) + 1
                if len(failed) < max_failed:
                    failed.append(index)
        summary.update(
            ok=ok,
            missing=missing,
            missing_total=missing,
            missing_idx=missing_idx,
            first_missing_idx=first_missing,
        )
    elif kind == "put":
        ok = 0
        for index, code in enumerate(codes):
            if code == 0:
                ok += 1
            else:
                failed_total += 1
                errors[str(int(code))] = errors.get(str(int(code)), 0) + 1
                if len(failed) < max_failed:
                    failed.append(index)
        summary["ok"] = ok
    elif kind == "get":
        ok = zero = 0
        first_zero = None
        first_err = None
        for index, code in enumerate(codes):
            if code > 0:
                ok += 1
            elif code == 0:
                zero += 1
                if first_zero is None:
                    first_zero = index
            else:
                failed_total += 1
                errors[str(int(code))] = errors.get(str(int(code)), 0) + 1
                if first_err is None:
                    first_err = index
                if len(failed) < max_failed:
                    failed.append(index)
        summary.update(ok=ok, zero=zero, first_zero_idx=first_zero)
    else:
        raise ValueError(f"unknown result kind: {kind!r}")

    summary["err"] = errors
    summary["failed_idx"] = failed
    summary["failed_total"] = failed_total
    if kind == "get":
        summary["first_err_idx"] = first_err
    return summary


class BoundedEventQueue:
    """Event queue bounded by both item count and estimated bytes."""

    def __init__(self, *, max_events: int, max_bytes: int, policy: str):
        if policy not in ("drop", "block"):
            raise ValueError(f"unsupported policy: {policy!r}")
        self._max_events = max_events
        self._max_bytes = max_bytes
        self._policy = policy
        self._queue: deque = deque()
        self._queued_bytes = 0
        self._condition = threading.Condition()
        self._closed = False
        self._dropped = 0
        self._blocked = 0
        self._blocked_seconds = 0.0

    def put(self, item: Any, size: int) -> bool:
        with self._condition:
            if self._closed:
                self._dropped += 1
                return False
            if not self._has_room(size):
                if self._policy == "drop":
                    self._dropped += 1
                    return False
                self._blocked += 1
                started = time.monotonic()
                while not self._has_room(size) and not self._closed:
                    self._condition.wait()
                self._blocked_seconds += time.monotonic() - started
                if self._closed:
                    self._dropped += 1
                    return False
            self._queue.append((item, size))
            self._queued_bytes += size
            self._condition.notify_all()
            return True

    def _has_room(self, size: int) -> bool:
        return (
            len(self._queue) < self._max_events
            and (self._queued_bytes + size) <= self._max_bytes
        )

    def get(self, timeout: Optional[float] = None) -> Optional[Any]:
        with self._condition:
            if not self._queue:
                self._condition.wait(timeout)
            if not self._queue:
                return None
            item, size = self._queue.popleft()
            self._queued_bytes -= size
            self._condition.notify_all()
            return item

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def stats(self) -> Dict[str, float]:
        with self._condition:
            return {
                "dropped": self._dropped,
                "blocked": self._blocked,
                "blocked_seconds": self._blocked_seconds,
                "queued_events": len(self._queue),
                "queued_bytes": self._queued_bytes,
            }


_context_fields: contextvars.ContextVar[tuple] = contextvars.ContextVar(
    "cache_diag_context", default=()
)


class CacheDiagnostics:
    """Process-wide writer for cache.* diagnostic events."""

    def __init__(
        self,
        config: DiagConfig,
        *,
        tp_rank: Optional[int] = None,
        pp_rank: Optional[int] = None,
        dp_rank: Optional[int] = None,
    ):
        self._config = config
        self._enabled = bool(config.enabled and config.directory)
        self._rank_fields: Dict[str, int] = {}
        if tp_rank is not None:
            self._rank_fields["tp_rank"] = int(tp_rank)
        if pp_rank is not None:
            self._rank_fields["pp_rank"] = int(pp_rank)
        if dp_rank is not None:
            self._rank_fields["dp_rank"] = int(dp_rank)
        self._pid = os.getpid()
        self._seq = 0
        self._seq_lock = threading.Lock()
        self._written = 0
        self._serialize_errors = 0
        self._pending = 0
        self._drain_condition = threading.Condition()
        self._current_path: Optional[str] = None
        self._file = None
        self._file_bytes = 0
        self._closed = False

        if not self._enabled:
            self._queue = BoundedEventQueue(
                max_events=1, max_bytes=1, policy="drop"
            )
            return

        Path(config.directory).mkdir(parents=True, exist_ok=True)
        self._queue = BoundedEventQueue(
            max_events=config.queue_events,
            max_bytes=config.queue_bytes,
            policy=config.overflow_policy,
        )
        self._base_path = self._build_base_path(config.directory)
        self._writer = threading.Thread(
            target=self._writer_loop, name="cache-diag-writer", daemon=True
        )
        self._writer.start()

    def _build_base_path(self, directory: str) -> str:
        suffix = "_".join(f"{k}{v}" for k, v in sorted(self._rank_fields.items()))
        if suffix:
            suffix = "_" + suffix
        return str(Path(directory) / f"cache_diag{suffix}_pid{self._pid}.jsonl")

    @property
    def enabled(self) -> bool:
        return self._enabled and not self._closed

    @property
    def current_path(self) -> Optional[str]:
        return self._current_path

    def emit(self, event: str, **fields: Any) -> None:
        if not self.enabled:
            return
        payload: Dict[str, Any] = {"event": event}
        for ctx in _context_fields.get():
            payload.update(ctx)
        payload.update(fields)
        payload["ts_ns"] = time.time_ns()
        payload["pid"] = self._pid
        payload.update(self._rank_fields)
        with self._seq_lock:
            self._seq += 1
            payload["seq"] = self._seq
        size = estimate_event_bytes(payload)
        with self._drain_condition:
            self._pending += 1
        if not self._queue.put(payload, size):
            with self._drain_condition:
                self._pending -= 1
                self._drain_condition.notify_all()

    @contextlib.contextmanager
    def context(self, **fields: Any) -> Iterator[None]:
        token = _context_fields.set(_context_fields.get() + (fields,))
        try:
            yield
        finally:
            _context_fields.reset(token)

    def _writer_loop(self) -> None:
        while True:
            item = self._queue.get(timeout=0.5)
            if item is None:
                if self._closed:
                    break
                continue
            try:
                line = json.dumps(item, separators=(",", ":"), ensure_ascii=False)
            except (TypeError, ValueError):
                self._serialize_errors += 1
                self._finish_one()
                continue
            try:
                self._write_line(line + "\n")
                self._written += 1
            except OSError:
                self._serialize_errors += 1
            self._finish_one()
            if self._pending == 0:
                self._flush_file()

    def _finish_one(self) -> None:
        with self._drain_condition:
            self._pending -= 1
            self._drain_condition.notify_all()

    def _write_line(self, line: str) -> None:
        encoded_size = len(line.encode("utf-8"))
        if self._file is None or (
            self._file_bytes > 0
            and self._file_bytes + encoded_size > self._config.max_file_bytes
        ):
            self._rotate()
        self._file.write(line)
        self._file_bytes += encoded_size

    def _flush_file(self) -> None:
        if self._file is None:
            return
        try:
            self._file.flush()
        except OSError:
            self._serialize_errors += 1

    def _rotate(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
        if self._current_path is not None:
            self._shift_history()
        self._current_path = self._base_path
        self._file = open(self._current_path, "a", encoding="utf-8")
        self._file_bytes = os.path.getsize(self._current_path)
        self._prune_history()

    def _shift_history(self) -> None:
        base = self._base_path
        for index in range(self._config.keep_files - 1, 0, -1):
            source = f"{base}.{index}"
            target = f"{base}.{index + 1}"
            if os.path.exists(source):
                os.replace(source, target)
        os.replace(base, f"{base}.1")

    def _prune_history(self) -> None:
        base = self._base_path
        for index in range(self._config.keep_files, self._config.keep_files + 100):
            stale = f"{base}.{index}"
            if not os.path.exists(stale):
                continue
            os.remove(stale)

    def flush(self, timeout: float = 5.0) -> bool:
        if not self._enabled:
            return True
        deadline = time.monotonic() + timeout
        with self._drain_condition:
            while self._pending > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._drain_condition.wait(remaining)
        self._flush_file()
        return True

    def stats(self) -> Dict[str, Any]:
        stats = dict(self._queue.stats())
        stats.update(
            written=self._written,
            serialize_errors=self._serialize_errors,
            pending=self._pending,
        )
        return stats

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.close()
        if not self._enabled:
            return
        self.flush(timeout=10.0)
        if threading.current_thread() is not getattr(self, "_writer", None):
            self._writer.join(timeout=10.0)
        if self._file is not None:
            self._file.close()
            self._file = None


_recorder: Optional[CacheDiagnostics] = None
_recorder_lock = threading.Lock()


def _config_from_env() -> Optional[DiagConfig]:
    from sglang.srt.environ import envs

    directory = envs.SGLANG_CACHE_DIAG_DIR.get()
    if not directory:
        return None
    return DiagConfig(
        directory=resolve_diag_directory(directory),
        enabled=True,
        overflow_policy=envs.SGLANG_CACHE_DIAG_OVERFLOW_POLICY.get(),
        queue_events=envs.SGLANG_CACHE_DIAG_QUEUE_EVENTS.get(),
        queue_bytes=envs.SGLANG_CACHE_DIAG_QUEUE_BYTES.get(),
        max_file_bytes=envs.SGLANG_CACHE_DIAG_MAX_FILE_BYTES.get(),
        keep_files=envs.SGLANG_CACHE_DIAG_KEEP_FILES.get(),
    )


def get_cache_diagnostics(
    *,
    tp_rank: Optional[int] = None,
    pp_rank: Optional[int] = None,
    dp_rank: Optional[int] = None,
) -> CacheDiagnostics:
    """Return the process-wide recorder, creating it on first use."""
    global _recorder
    if _recorder is not None:
        return _recorder
    with _recorder_lock:
        if _recorder is None:
            config = _config_from_env()
            if config is None:
                config = DiagConfig(directory="", enabled=False)
            _recorder = CacheDiagnostics(
                config, tp_rank=tp_rank, pp_rank=pp_rank, dp_rank=dp_rank
            )
    return _recorder


def reset_cache_diagnostics_for_test() -> None:
    global _recorder
    with _recorder_lock:
        if _recorder is not None:
            _recorder.close()
        _recorder = None
