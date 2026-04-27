"""
narrator.py
-----------
Uses GPT-4o to convert model results into business-ready insights.
"""

from openai import OpenAI
import pandas as pd
import json
import os

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def generate_insights(master_df: pd.DataFrame, category: str = "skincare") -> str:
    top_themes = master_df.nlargest(10, "review_count")[[
        "theme", "sentiment_category", "trend_label", "momentum_score", "review_count"
    ]].to_dict(orient="records")

    mixed_negative = master_df[
        master_df["sentiment_category"].isin(["Mixed", "Negative"])
    ][["theme", "sentiment_category", "avg_sentiment"]].to_dict(orient="records")

    prompt = f"""You are a consumer insights analyst. Based on this data from {category} product reviews, generate a concise business intelligence report.

TOP CONSUMER THEMES:
{json.dumps(top_themes, indent=2)}

MIXED/NEGATIVE SENTIMENT THEMES (pain points):
{json.dumps(mixed_negative, indent=2)}

Write a report with these sections:
1. KEY FINDINGS (3 bullet points, most important signals)
2. RISING OPPORTUNITIES (themes with rising momentum — what brands should invest in)
3. PAIN POINTS (mixed/negative themes — unmet consumer needs)
4. STRATEGIC RECOMMENDATIONS (2-3 specific actions for a brand)

Be specific, data-driven, and concise. Use the actual theme names. No fluff."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )

    return response.choices[0].message.content
