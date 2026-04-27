"""
trend_scorer.py
---------------
Scores whether each consumer theme is rising, stable, or declining over time.
"""

import pandas as pd
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class TrendScorer:
    def __init__(self, time_col="review_date", min_periods=3):
        self.time_col = time_col
        self.min_periods = min_periods

    def score(self, doc_df, dataset_df):
        valid_idx = doc_df.index
        doc_df = doc_df.copy()
        doc_df["review_date"] = dataset_df.loc[valid_idx, "review_date"].values

        doc_df = doc_df.dropna(subset=["review_date"])
        doc_df = doc_df[doc_df["topic_id"] != -1]

        if len(doc_df) == 0:
            return pd.DataFrame(columns=["topic_id", "topic_name", "slope", "momentum_score", "trend_label", "total_reviews"])

        doc_df["month"] = doc_df["review_date"].dt.to_period("M")
        monthly = (
            doc_df.groupby(["topic_id", "topic_name", "month"])
            .size()
            .reset_index(name="count")
        )
        monthly["month_num"] = monthly["month"].apply(lambda p: p.ordinal)

        results = []
        for (topic_id, topic_name), group in monthly.groupby(["topic_id", "topic_name"]):
            if len(group) < self.min_periods:
                continue
            x = group["month_num"].values
            y = group["count"].values
            slope, _, r_value, p_value, _ = stats.linregress(x, y)
            results.append({
                "topic_id": topic_id,
                "topic_name": topic_name,
                "slope": slope,
                "r_squared": r_value ** 2,
                "p_value": p_value,
                "total_reviews": y.sum(),
                "recent_avg": y[-3:].mean(),
                "early_avg": y[:3].mean(),
            })

        if not results:
            return pd.DataFrame(columns=["topic_id", "topic_name", "slope", "momentum_score", "trend_label", "total_reviews"])

        trend_df = pd.DataFrame(results)
        max_slope = trend_df["slope"].abs().max()
        trend_df["momentum_score"] = trend_df["slope"] / max_slope if max_slope > 0 else 0
        trend_df["trend_label"] = trend_df["momentum_score"].apply(
            lambda s: "↑ Rising" if s > 0.2 else ("↓ Declining" if s < -0.2 else "→ Stable")
        )

        return trend_df.sort_values("momentum_score", ascending=False)
