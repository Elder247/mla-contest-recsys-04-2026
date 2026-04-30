"""CLI wrapper around :func:`src.inference.validate_submission.validate_submission`.

Usage:
    python scripts/validate_submission.py submissions/sub_008_ranker_a3_audioknn.csv
    python scripts/validate_submission.py submissions/sub_*.csv  # several at once

Exit code 0 if every submission is valid, 1 otherwise. Prints a structured
report per file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.inference.validate_submission import validate_submission


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submissions", nargs="+", help="path(s) to sub_*.csv")
    parser.add_argument(
        "--users-csv",
        default="submissions/users.csv",
        help="path to the eval users CSV (default: submissions/users.csv)",
    )
    parser.add_argument(
        "--max-items", type=int, default=100,
        help="hard upper bound on items per row (default: 100)",
    )
    args = parser.parse_args()

    any_failure = False
    for sub_path in args.submissions:
        report = validate_submission(
            Path(sub_path),
            Path(args.users_csv),
            max_items=args.max_items,
        )
        print(report.summary())
        print()
        if not report.ok:
            any_failure = True

    return 1 if any_failure else 0


if __name__ == "__main__":
    sys.exit(main())
