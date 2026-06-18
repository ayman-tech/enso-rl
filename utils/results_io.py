"""Small helper to write result tables as tidy CSV alongside the .npz outputs.

CSVs are written in long/tidy format (one row per observation) so they load
straight into pandas for notebook analysis, and survive lost cluster logs.
"""
import csv
from pathlib import Path


def save_csv(path, rows, fieldnames=None):
    """Write a list of dict rows to CSV (long/tidy format).

    Args:
        path: output path (parent dirs assumed to exist, as for the sibling .npz).
        rows: list of dicts; keys become columns. No-op if empty.
        fieldnames: optional explicit column order (defaults to first row's keys).
    Returns:
        the Path written, or None if rows was empty.
    """
    if not rows:
        return None
    path = Path(path)
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {path}")
    return path
