"""
ingestor.py
-----------
Auto-detects review text, date, and rating columns from any uploaded CSV/JSON.
Works as a standalone module — no hardcoded column names.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ─── Heuristics ───────────────────────────────────────────────────────────────

TEXT_KEYWORDS = [
    "review", "text", "comment", "feedback", "body", "content",
    "description", "message", "opinion", "response", "note"
]

DATE_KEYWORDS = [
    "date", "time", "created", "posted", "submitted", "timestamp",
    "reviewed_at", "review_date", "at", "when"
]

RATING_KEYWORDS = [
    "rating", "score", "star", "stars", "grade", "rank",
    "overall", "evaluation", "rate", "points"
]


def _score_column(col_name: str, keywords: list[str]) -> int:
    """Score how likely a column name matches a keyword list."""
    col_lower = col_name.lower()
    return sum(kw in col_lower for kw in keywords)


def _detect_text_column(df: pd.DataFrame) -> Optional[str]:
    """Find the most likely free-text review column."""
    candidates = []
    for col in df.columns:
        if df[col].dtype != object:
            continue
        name_score = _score_column(col, TEXT_KEYWORDS)
        avg_len = df[col].dropna().apply(len).mean()
        # Prefer long-text columns with matching names
        length_score = min(avg_len / 50, 5)  # cap at 5
        candidates.append((col, name_score + length_score))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_col, best_score = candidates[0]
    logger.info(f"  Text column detected: '{best_col}' (score={best_score:.1f})")
    return best_col


def _detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """Find the most likely date/timestamp column."""
    candidates = []
    for col in df.columns:
        name_score = _score_column(col, DATE_KEYWORDS)
        # Try parsing as date
        parse_score = 0
        try:
            sample = df[col].dropna().head(50)
            pd.to_datetime(sample, infer_datetime_format=True, errors="raise")
            parse_score = 3
        except Exception:
            pass
        total = name_score + parse_score
        if total > 0:
            candidates.append((col, total))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_col, best_score = candidates[0]
    logger.info(f"  Date column detected:  '{best_col}' (score={best_score:.1f})")
    return best_col


def _detect_rating_column(df: pd.DataFrame) -> Optional[str]:
    """Find the most likely numeric rating column."""
    candidates = []
    for col in df.columns:
        name_score = _score_column(col, RATING_KEYWORDS)
        # Numeric column with small range (e.g. 1–5, 1–10)
        range_score = 0
        if pd.api.types.is_numeric_dtype(df[col]):
            col_range = df[col].max() - df[col].min()
            if 1 <= col_range <= 10:
                range_score = 3
        total = name_score + range_score
        if total > 0:
            candidates.append((col, total))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_col, best_score = candidates[0]
    logger.info(f"  Rating column detected: '{best_col}' (score={best_score:.1f})")
    return best_col


# ─── Main loader ──────────────────────────────────────────────────────────────

class ReviewDataset:
    """
    Holds a cleaned DataFrame with standardized columns:
      - review_text : str
      - review_date : datetime (optional)
      - review_rating : float (optional)
      - [all original columns preserved]
    """

    def __init__(
        self,
        df: pd.DataFrame,
        text_col: str,
        date_col: Optional[str] = None,
        rating_col: Optional[str] = None,
        source: str = "unknown"
    ):
        self.source = source
        self.text_col = text_col
        self.date_col = date_col
        self.rating_col = rating_col
        self.df = self._standardize(df)

    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["review_text"] = out[self.text_col].astype(str).str.strip()

        if self.date_col:
            out["review_date"] = pd.to_datetime(
                out[self.date_col], infer_datetime_format=True, errors="coerce"
            )

        if self.rating_col:
            out["review_rating"] = pd.to_numeric(out[self.rating_col], errors="coerce")

        # Drop rows with empty text
        out = out[out["review_text"].str.len() > 10].reset_index(drop=True)
        return out

    @property
    def texts(self) -> list[str]:
        return self.df["review_text"].tolist()

    def summary(self) -> dict:
        info = {
            "source": self.source,
            "total_reviews": len(self.df),
            "text_column": self.text_col,
            "date_column": self.date_col,
            "rating_column": self.rating_col,
        }
        if "review_date" in self.df:
            info["date_range"] = (
                str(self.df["review_date"].min().date()),
                str(self.df["review_date"].max().date()),
            )
        if "review_rating" in self.df:
            info["avg_rating"] = round(self.df["review_rating"].mean(), 2)
        return info


def load(
    filepath: str | Path,
    text_col: Optional[str] = None,
    date_col: Optional[str] = None,
    rating_col: Optional[str] = None,
    sample: Optional[int] = None,
) -> ReviewDataset:
    """
    Load any CSV or JSON file and return a ReviewDataset.

    Parameters
    ----------
    filepath   : path to .csv or .json file
    text_col   : override auto-detection for text column
    date_col   : override auto-detection for date column
    rating_col : override auto-detection for rating column
    sample     : randomly sample N rows (useful for fast dev)
    """
    filepath = Path(filepath)
    logger.info(f"Loading: {filepath.name}")

    if filepath.suffix == ".csv":
        df = pd.read_csv(filepath, low_memory=False)
    elif filepath.suffix in (".json", ".jsonl"):
        df = pd.read_json(filepath, lines=filepath.suffix == ".jsonl")
    else:
        raise ValueError(f"Unsupported file type: {filepath.suffix}. Use .csv or .json")

    logger.info(f"  Rows: {len(df):,} | Columns: {list(df.columns)}")

    if sample:
        df = df.sample(min(sample, len(df)), random_state=42).reset_index(drop=True)
        logger.info(f"  Sampled: {len(df):,} rows")

    # Auto-detect columns if not overridden
    logger.info("Detecting columns...")
    text_col = text_col or _detect_text_column(df)
    date_col = date_col or _detect_date_column(df)
    rating_col = rating_col or _detect_rating_column(df)

    if not text_col:
        raise ValueError(
            "Could not auto-detect a text column. "
            "Pass text_col='your_column_name' explicitly."
        )

    dataset = ReviewDataset(
        df=df,
        text_col=text_col,
        date_col=date_col,
        rating_col=rating_col,
        source=filepath.stem,
    )

    logger.info(f"Dataset ready: {dataset.summary()}")
    return dataset


# ─── CLI quick test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingestor.py path/to/reviews.csv")
        sys.exit(1)

    ds = load(sys.argv[1])
    print("\n── Dataset Summary ──")
    for k, v in ds.summary().items():
        print(f"  {k}: {v}")
    print(f"\nFirst review:\n  {ds.texts[0][:300]}")
