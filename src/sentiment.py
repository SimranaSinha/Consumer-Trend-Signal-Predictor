"""
sentiment.py
------------
Scores sentiment for each consumer theme using RoBERTa.
"""

from transformers import pipeline
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class ThemeSentimentScorer:
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        logger.info(f"Loading sentiment model: {model_name}")
        self.pipe = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            max_length=512,
            truncation=True,
            device=0
        )
        self.label_map = {"positive": 1, "neutral": 0, "negative": -1}

    def score_texts(self, texts, batch_size=64):
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            outputs = self.pipe(batch)
            results.extend(outputs)
        return results

    def score_by_theme(self, doc_df):
        valid = doc_df[doc_df["topic_id"] != -1].copy()
        logger.info(f"Scoring sentiment for {len(valid)} documents...")

        raw_scores = self.score_texts(valid["text"].tolist())
        valid["sentiment_label"] = [r["label"].lower() for r in raw_scores]
        valid["sentiment_score"] = [
            r["score"] * self.label_map.get(r["label"].lower(), 0)
            for r in raw_scores
        ]

        summary = valid.groupby(["topic_id", "topic_name"]).agg(
            review_count=("text", "count"),
            avg_sentiment=("sentiment_score", "mean"),
            pct_positive=("sentiment_label", lambda x: (x == "positive").mean()),
            pct_negative=("sentiment_label", lambda x: (x == "negative").mean()),
            pct_neutral=("sentiment_label", lambda x: (x == "neutral").mean()),
        ).reset_index()

        summary["sentiment_category"] = summary["avg_sentiment"].apply(
            lambda s: "Positive" if s > 0.2 else ("Negative" if s < -0.1 else "Mixed")
        )

        return summary.sort_values("review_count", ascending=False)
