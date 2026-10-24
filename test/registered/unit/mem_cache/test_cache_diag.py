"""Unit tests for the cache diagnostic event recorder.

The recorder is deliberately free of sglang imports so it can be exercised
without torch; tests load the module by path.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "python"
    / "sglang"
    / "srt"
    / "mem_cache"
    / "diagnostics"
    / "cache_diag.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("cache_diag_under_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def diag_mod():
    return _load_module()


def _read_events(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _all_events(directory: Path):
    events = []
    for path in directory.iterdir():
        events.extend(_read_events(path))
    return events


def test_disabled_recorder_writes_nothing(diag_mod, tmp_path):
    recorder = diag_mod.CacheDiagnostics(
        diag_mod.DiagConfig(directory=str(tmp_path), enabled=False)
    )
    recorder.emit("cache.lookup", rid="r1")
    assert recorder.flush(timeout=2.0)
    assert list(tmp_path.iterdir()) == []
    assert recorder.stats()["written"] == 0


def test_emit_writes_jsonl_with_common_fields(diag_mod, tmp_path):
    recorder = diag_mod.CacheDiagnostics(
        diag_mod.DiagConfig(directory=str(tmp_path)),
        tp_rank=1,
        pp_rank=0,
        dp_rank=2,
    )
    recorder.emit("cache.lookup", rid="r1", hit_pages=7)
    assert recorder.flush(timeout=2.0)

    events = _read_events(Path(recorder.current_path))
    assert len(events) == 1
    event = events[0]
    assert event["event"] == "cache.lookup"
    assert event["rid"] == "r1"
    assert event["hit_pages"] == 7
    assert event["tp_rank"] == 1
    assert event["pp_rank"] == 0
    assert event["dp_rank"] == 2
    assert event["pid"] == os.getpid()
    assert isinstance(event["ts_ns"], int)
    recorder.close()


def test_drop_queue_returns_false_and_counts_dropped(diag_mod):
    queue = diag_mod.BoundedEventQueue(
        max_events=1, max_bytes=1 << 20, policy="drop"
    )
    assert queue.put({"event": "a"}, 10)
    assert not queue.put({"event": "b"}, 10)
    assert queue.stats()["dropped"] == 1
    assert queue.get(timeout=0.1) == {"event": "a"}


def test_block_queue_waits_for_capacity(diag_mod):
    queue = diag_mod.BoundedEventQueue(
        max_events=1, max_bytes=1 << 20, policy="block"
    )
    assert queue.put({"event": "a"}, 10)
    done = threading.Event()

    def producer():
        queue.put({"event": "b"}, 10)
        done.set()

    thread = threading.Thread(target=producer)
    thread.start()
    time.sleep(0.15)
    assert not done.is_set()
    queue.get(timeout=0.1)
    assert done.wait(timeout=2.0)
    thread.join(timeout=2.0)
    assert queue.stats()["dropped"] == 0
    assert queue.stats()["blocked"] == 1


def test_queue_frees_bytes_after_consumption(diag_mod):
    queue = diag_mod.BoundedEventQueue(
        max_events=10, max_bytes=100, policy="drop"
    )
    assert queue.put({"event": "a"}, 60)
    assert not queue.put({"event": "b"}, 60)
    queue.get(timeout=0.1)
    assert queue.put({"event": "c"}, 60)


def test_rotation_creates_new_file_and_preserves_events(diag_mod, tmp_path):
    recorder = diag_mod.CacheDiagnostics(
        diag_mod.DiagConfig(
            directory=str(tmp_path),
            max_file_bytes=600,
            keep_files=64,
        )
    )
    for i in range(40):
        recorder.emit("cache.io", rid=f"r{i}", payload="x" * 20)
    assert recorder.flush(timeout=5.0)
    recorder.close()

    files = list(tmp_path.iterdir())
    assert len(files) > 1
    assert len(_all_events(tmp_path)) == 40


def test_keep_files_limits_retained_history(diag_mod, tmp_path):
    recorder = diag_mod.CacheDiagnostics(
        diag_mod.DiagConfig(
            directory=str(tmp_path),
            max_file_bytes=300,
            keep_files=2,
        )
    )
    for i in range(40):
        recorder.emit("cache.io", rid=f"r{i}", payload="x" * 20)
    assert recorder.flush(timeout=5.0)
    recorder.close()
    assert len(list(tmp_path.iterdir())) <= 2


def test_context_fields_are_merged(diag_mod, tmp_path):
    recorder = diag_mod.CacheDiagnostics(diag_mod.DiagConfig(directory=str(tmp_path)))
    with recorder.context(rid="req-9", op_id=3):
        recorder.emit("cache.lookup", hit_pages=1)
    assert recorder.flush(timeout=2.0)
    event = _read_events(Path(recorder.current_path))[0]
    assert event["rid"] == "req-9"
    assert event["op_id"] == 3
    assert event["hit_pages"] == 1
    recorder.close()


def test_emit_after_close_is_dropped_without_raising(diag_mod, tmp_path):
    recorder = diag_mod.CacheDiagnostics(diag_mod.DiagConfig(directory=str(tmp_path)))
    recorder.close()
    recorder.emit("cache.io", rid="late")
    assert recorder.stats()["written"] == 0


def test_compact_indices_ranges(diag_mod):
    assert diag_mod.compact_indices([]) == ""
    assert diag_mod.compact_indices([5]) == "5"
    assert diag_mod.compact_indices([1, 2, 3, 7, 9, 10]) == "1-3,7,9-10"


def test_summarize_exists_reports_first_missing(diag_mod):
    summary = diag_mod.summarize_codes([1, 1, 0, 1], kind="exists")
    assert summary["ok"] == 3
    assert summary["missing"] == 1
    assert summary["first_missing_idx"] == 2
    assert summary["failed_idx"] == []


def test_summarize_put_groups_error_codes(diag_mod):
    summary = diag_mod.summarize_codes([0, -705, 0], kind="put")
    assert summary["ok"] == 2
    assert summary["err"] == {"-705": 1}
    assert summary["failed_idx"] == [1]


def test_summarize_get_separates_zero_and_errors(diag_mod):
    summary = diag_mod.summarize_codes([4096, 0, -704, 8192], kind="get")
    assert summary["ok"] == 2
    assert summary["zero"] == 1
    assert summary["first_zero_idx"] == 1
    assert summary["err"] == {"-704": 1}
    assert summary["first_err_idx"] == 2


def test_summarize_failed_indices_are_capped(diag_mod):
    summary = diag_mod.summarize_codes([-704] * 50, kind="put", max_failed=4)
    assert summary["failed_idx"] == [0, 1, 2, 3]
    assert summary["failed_total"] == 50


def test_summarize_exists_lists_missing_indices_capped(diag_mod):
    summary = diag_mod.summarize_codes([1, 0, 0, 1, 0], kind="exists", max_failed=2)
    assert summary["missing_total"] == 3
    assert summary["missing_idx"] == [1, 2]
