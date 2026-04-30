"""Submission format validator (Phase F.4).

The contest expects a CSV ``uid,item_ids`` where each row has the user id
and a space-separated list of item ids. We enforce the contract documented
in CLAUDE.md and learnt empirically from past submissions:

  * exactly one row per uid in ``users.csv``
  * each row has between 1 and ``max_items`` item ids (default 100)
  * item ids are positive integers
  * no duplicate item ids within a row
  * no duplicate uids

Errors are returned as a list (caller decides whether to raise); warnings
flag soft issues like rows shorter than ``max_items`` (always allowed by
the metric, but worth knowing — happens when the candidate pool is too
small for cold users).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Result of :func:`validate_submission` — structured for programmatic use."""

    submission_path: Path
    n_rows: int
    n_unique_uids: int
    n_expected_users: int
    items_per_row_min: int
    items_per_row_max: int
    items_per_row_mean: float
    n_short_rows: int                       # rows with len(item_ids) < max_items
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        status = "OK" if self.ok else "FAIL"
        lines = [
            f"[{status}] {self.submission_path}",
            f"  rows = {self.n_rows} ({self.n_unique_uids} unique uids; "
            f"expected {self.n_expected_users})",
            f"  items per row: min={self.items_per_row_min} "
            f"mean={self.items_per_row_mean:.2f} max={self.items_per_row_max} "
            f"(short={self.n_short_rows})",
        ]
        if self.warnings:
            lines.append("  warnings:")
            lines.extend(f"    - {w}" for w in self.warnings)
        if self.errors:
            lines.append("  errors:")
            lines.extend(f"    - {e}" for e in self.errors)
        return "\n".join(lines)


def validate_submission(
    submission_path: str | Path,
    users_csv: str | Path,
    max_items: int = 100,
    min_items: int = 1,
) -> ValidationReport:
    """Validate a submission CSV against the eval users list.

    Args:
        submission_path: path to the ``sub_*.csv`` file.
        users_csv: path to ``submissions/users.csv``.
        max_items: hard upper bound on ``len(item_ids)`` per row.
        min_items: hard lower bound; <1 means an empty row, never useful.

    Returns:
        :class:`ValidationReport`. Caller can inspect ``ok`` /
        ``errors`` / ``warnings`` and decide how to react.
    """
    submission_path = Path(submission_path)
    users_csv = Path(users_csv)

    errors: list[str] = []
    warnings: list[str] = []

    if not submission_path.exists():
        return ValidationReport(
            submission_path=submission_path,
            n_rows=0, n_unique_uids=0, n_expected_users=0,
            items_per_row_min=0, items_per_row_max=0, items_per_row_mean=0.0,
            n_short_rows=0,
            errors=[f"submission file not found: {submission_path}"],
        )

    expected_users = pl.read_csv(users_csv)
    if "uid" not in expected_users.columns:
        errors.append(f"users.csv missing 'uid' column: cols={expected_users.columns}")
        # bail early: we can't compare uids without the source
        return ValidationReport(
            submission_path=submission_path,
            n_rows=0, n_unique_uids=0, n_expected_users=0,
            items_per_row_min=0, items_per_row_max=0, items_per_row_mean=0.0,
            n_short_rows=0,
            errors=errors,
        )
    expected_uid_set = set(expected_users["uid"].cast(pl.Int64).to_list())
    n_expected = len(expected_uid_set)

    # ── Read submission ──────────────────────────────────────────────────────
    sub = pl.read_csv(submission_path, schema_overrides={"uid": pl.Int64})
    cols_ok = {"uid", "item_ids"}.issubset(set(sub.columns))
    if not cols_ok:
        errors.append(
            f"missing required columns; got {sub.columns}, expected superset of "
            f"{sorted(cols_ok)}"
        )
        return ValidationReport(
            submission_path=submission_path,
            n_rows=len(sub),
            n_unique_uids=0, n_expected_users=n_expected,
            items_per_row_min=0, items_per_row_max=0, items_per_row_mean=0.0,
            n_short_rows=0,
            errors=errors,
        )

    n_rows = len(sub)
    sub_uids = sub["uid"].cast(pl.Int64)
    n_unique_uids = sub_uids.n_unique()

    if n_rows != n_unique_uids:
        errors.append(f"duplicate uids: rows={n_rows} unique={n_unique_uids}")

    sub_uid_set = set(sub_uids.to_list())
    missing_uids = expected_uid_set - sub_uid_set
    extra_uids = sub_uid_set - expected_uid_set
    if missing_uids:
        sample = sorted(missing_uids)[:5]
        errors.append(
            f"{len(missing_uids)} uids from users.csv missing from submission "
            f"(sample: {sample})"
        )
    if extra_uids:
        sample = sorted(extra_uids)[:5]
        errors.append(
            f"{len(extra_uids)} uids in submission not present in users.csv "
            f"(sample: {sample})"
        )

    # ── Parse item_ids per row, count, dedup-check ──────────────────────────
    parsed = (
        sub
        .with_columns(pl.col("item_ids").cast(pl.Utf8).str.split(" ").alias("_items"))
        .with_columns([
            pl.col("_items").list.len().cast(pl.Int32).alias("_n_items"),
            pl.col("_items").list.unique().list.len().cast(pl.Int32).alias("_n_unique"),
        ])
    )

    # length stats
    n_items = parsed["_n_items"]
    n_min = int(n_items.min()) if n_rows else 0
    n_max = int(n_items.max()) if n_rows else 0
    n_mean = float(n_items.mean()) if n_rows else 0.0
    n_short = int((n_items < max_items).sum()) if n_rows else 0

    if n_max > max_items:
        n_over = int((n_items > max_items).sum())
        errors.append(
            f"{n_over} row(s) have >{max_items} item_ids "
            f"(max observed = {n_max})"
        )
    if n_min < min_items:
        n_under = int((n_items < min_items).sum())
        errors.append(
            f"{n_under} row(s) have <{min_items} item_ids "
            f"(min observed = {n_min})"
        )

    # within-row duplicates
    n_dup_rows = int((parsed["_n_items"] != parsed["_n_unique"]).sum())
    if n_dup_rows > 0:
        errors.append(f"{n_dup_rows} row(s) contain duplicate item_ids")

    # parseability — rows where any token is non-numeric become null in cast
    bad_token = (
        parsed
        .select(
            pl.col("_items")
            .list.eval(pl.element().cast(pl.Int64, strict=False).is_null())
            .list.any()
            .alias("_has_bad")
        )
    )
    n_bad_token_rows = int(bad_token["_has_bad"].sum())
    if n_bad_token_rows > 0:
        errors.append(
            f"{n_bad_token_rows} row(s) contain non-integer / unparseable item ids"
        )

    if n_short > 0:
        warnings.append(
            f"{n_short}/{n_rows} row(s) have fewer than {max_items} items "
            f"(min={n_min}) — usually cold/sparse users; metric still valid"
        )

    return ValidationReport(
        submission_path=submission_path,
        n_rows=n_rows,
        n_unique_uids=n_unique_uids,
        n_expected_users=n_expected,
        items_per_row_min=n_min,
        items_per_row_max=n_max,
        items_per_row_mean=n_mean,
        n_short_rows=n_short,
        errors=errors,
        warnings=warnings,
    )
