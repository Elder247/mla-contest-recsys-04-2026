"""Tests for src.inference.validate_submission (Phase F.4 / C1)."""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from src.inference.validate_submission import validate_submission


@pytest.fixture
def users_csv(tmp_path: Path) -> Path:
    p = tmp_path / "users.csv"
    pl.DataFrame({"uid": [100, 200, 300]}).write_csv(p)
    return p


def _write_sub(path: Path, rows: list[tuple[int, str]]) -> Path:
    pl.DataFrame(
        {"uid": [r[0] for r in rows], "item_ids": [r[1] for r in rows]}
    ).write_csv(path)
    return path


def test_valid_submission_passes(tmp_path, users_csv):
    sub = _write_sub(tmp_path / "sub.csv", [
        (100, "1 2 3"),
        (200, "4 5 6"),
        (300, "7 8 9"),
    ])
    r = validate_submission(sub, users_csv, max_items=100)
    assert r.ok
    assert r.errors == []
    assert r.n_rows == 3
    assert r.n_unique_uids == 3
    assert r.items_per_row_min == 3
    assert r.items_per_row_max == 3


def test_missing_uid_is_error(tmp_path, users_csv):
    sub = _write_sub(tmp_path / "sub.csv", [
        (100, "1 2 3"),
        (200, "4 5 6"),
    ])
    r = validate_submission(sub, users_csv, max_items=100)
    assert not r.ok
    assert any("missing from submission" in e for e in r.errors)


def test_extra_uid_is_error(tmp_path, users_csv):
    sub = _write_sub(tmp_path / "sub.csv", [
        (100, "1"), (200, "2"), (300, "3"), (999, "4"),
    ])
    r = validate_submission(sub, users_csv, max_items=100)
    assert not r.ok
    assert any("not present in users.csv" in e for e in r.errors)


def test_duplicate_uids_are_error(tmp_path, users_csv):
    # 100 appears twice, 200/300 missing
    sub = _write_sub(tmp_path / "sub.csv", [(100, "1"), (100, "2")])
    r = validate_submission(sub, users_csv, max_items=100)
    assert not r.ok
    assert any("duplicate uids" in e for e in r.errors)


def test_too_many_items_is_error(tmp_path, users_csv):
    items = " ".join(str(i) for i in range(101))
    sub = _write_sub(tmp_path / "sub.csv", [
        (100, items), (200, "1"), (300, "1"),
    ])
    r = validate_submission(sub, users_csv, max_items=100)
    assert not r.ok
    assert any(">100 item_ids" in e for e in r.errors)


def test_within_row_dups_are_error(tmp_path, users_csv):
    sub = _write_sub(tmp_path / "sub.csv", [
        (100, "1 2 2 3"),  # 2 dup'd
        (200, "1"), (300, "1"),
    ])
    r = validate_submission(sub, users_csv, max_items=100)
    assert not r.ok
    assert any("duplicate item_ids" in e for e in r.errors)


def test_non_integer_item_is_error(tmp_path, users_csv):
    sub = _write_sub(tmp_path / "sub.csv", [
        (100, "1 abc 3"),
        (200, "1"), (300, "1"),
    ])
    r = validate_submission(sub, users_csv, max_items=100)
    assert not r.ok
    assert any("non-integer" in e for e in r.errors)


def test_short_rows_are_warning_not_error(tmp_path, users_csv):
    sub = _write_sub(tmp_path / "sub.csv", [
        (100, "1 2 3"),  # 3 < 100, but allowed
        (200, "4"), (300, "5"),
    ])
    r = validate_submission(sub, users_csv, max_items=100)
    assert r.ok
    assert any("fewer than" in w for w in r.warnings)


def test_missing_file_is_error(tmp_path, users_csv):
    r = validate_submission(tmp_path / "nope.csv", users_csv)
    assert not r.ok
    assert any("submission file not found" in e for e in r.errors)


def test_summary_string_mentions_status(tmp_path, users_csv):
    sub = _write_sub(tmp_path / "sub.csv", [
        (100, "1 2 3"), (200, "1"), (300, "1"),
    ])
    r = validate_submission(sub, users_csv)
    s = r.summary()
    assert "[OK]" in s
    bad_sub = _write_sub(tmp_path / "bad.csv", [(100, "1")])
    r2 = validate_submission(bad_sub, users_csv)
    assert "[FAIL]" in r2.summary()
