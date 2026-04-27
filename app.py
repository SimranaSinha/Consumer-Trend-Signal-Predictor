import streamlit as st
import pandas as pd
import plotly.express as px
import json, os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

st.set_page_config(page_title="Consumer Trend Signal Predictor", layout="wide", page_icon="📊")

st.markdown("""
<style>
    .main { background-color: #0f0f0f; }
    .block-container { padding-top: 2rem; }
    h1 { color: #00ff88; font-family: monospace; }
    h2, h3 { color: #ffffff; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Consumer Trend Signal Predictor")
st.caption("Upload any product review dataset → get emerging themes, sentiment, and trend momentum")

with st.sidebar:
    st.header("⚙️ Configuration")
    openai_key = st.text_input("OpenAI API Key", type="password")
    category = st.text_input("Product Category", value="skincare")
    sample_size = st.slider("Sample Size", 1000, 10000, 5000, 500)
    run_btn = st.button("🚀 Run Analysis", use_container_width=True)

st.subheader("1. Upload Your Data")
uploaded_file = st.file_uploader("Drop any CSV with product reviews", type=["csv"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file, on_bad_lines="skip", engine="python")
    st.success(f"✅ Loaded {len(df_raw):,} rows · {len(df_raw.columns)} columns")
    st.write("**Columns detected:**", df_raw.columns.tolist())

    col1, col2, col3 = st.columns(3)
    with col1:
        text_col = st.selectbox("Review Text Column", df_raw.columns)
    with col2:
        date_col = st.selectbox("Date Column", ["None"] + df_raw.columns.tolist())
    with col3:
        rating_col = st.selectbox("Rating Column", ["None"] + df_raw.columns.tolist())

    if run_btn:
        if not openai_key:
            st.error("Please enter your OpenAI API key in the sidebar")
            st.stop()

        os.environ["OPENAI_API_KEY"] = openai_key
        df = df_raw.sample(min(sample_size, len(df_raw)), random_state=42).reset_index(drop=True)

        with st.spinner("🧹 Cleaning text..."):
            from preprocessor import preprocess
            cleaned_texts, valid_idx = preprocess(df[text_col].tolist())
            st.success(f"✅ Preprocessed {len(cleaned_texts):,} reviews")

        with st.spinner("🔍 Discovering consumer themes (2-3 mins)..."):
            from topic_model import ConsumerTopicModel
            tm = ConsumerTopicModel(min_topic_size=15).build()
            tm.fit(cleaned_texts)
            topic_table = tm.get_topic_table()
            st.success(f"✅ Found {len(topic_table)} consumer themes")

        with st.spinner("💬 Scoring sentiment per theme..."):
            from sentiment import ThemeSentimentScorer
            doc_df = tm.get_topic_per_doc(cleaned_texts)
            scorer = ThemeSentimentScorer()
            sentiment_df = scorer.score_by_theme(doc_df)
            st.success("✅ Sentiment scored")

        with st.spinner("📈 Computing trend momentum..."):
            if date_col != "None":
                from trend_scorer import TrendScorer
                df["review_date"] = pd.to_datetime(df[date_col], errors="coerce")
                trend_df = TrendScorer().score(doc_df, df)
                if len(trend_df) > 0:
                    master_df = sentiment_df.merge(
                        trend_df[["topic_id", "trend_label", "momentum_score"]],
                        on="topic_id", how="left"
                    )
                else:
                    master_df = sentiment_df.copy()
                    master_df["trend_label"] = "→ Stable"
                    master_df["momentum_score"] = 0.0
            else:
                master_df = sentiment_df.copy()
                master_df["trend_label"] = "→ Stable"
                master_df["momentum_score"] = 0.0

            master_df["momentum_score"] = master_df["momentum_score"].fillna(0)
            master_df["trend_label"] = master_df["trend_label"].fillna("→ Stable")
            master_df["theme"] = master_df["topic_name"].apply(
                lambda x: " ".join(x.split("_")[1:4]).title()
            )
            st.success("✅ Trend momentum computed")

        with st.spinner("🤖 Generating business insights..."):
            from narrator import generate_insights
            report = generate_insights(master_df, category=category)

        st.divider()
        st.subheader("2. Consumer Theme Intelligence")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Themes", len(master_df))
        m2.metric("Rising Themes", len(master_df[master_df["trend_label"] == "↑ Rising"]))
        m3.metric("Avg Sentiment", f"{master_df['avg_sentiment'].mean():.2f}")
        m4.metric("Reviews Analyzed", f"{len(cleaned_texts):,}")

        st.dataframe(
            master_df[["theme", "sentiment_category", "trend_label", "momentum_score", "review_count", "avg_sentiment"]]
            .sort_values("momentum_score", ascending=False)
            .reset_index(drop=True),
            use_container_width=True
        )

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                master_df.nlargest(15, "review_count"),
                x="review_count", y="theme", orientation="h",
                color="sentiment_category",
                color_discrete_map={"Positive": "#00ff88", "Mixed": "#ffaa00", "Negative": "#ff4444"},
                title="Top Themes by Volume"
            )
            fig.update_layout(paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e", font_color="white")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.scatter(
                master_df,
                x="momentum_score", y="avg_sentiment",
                size="review_count", color="sentiment_category",
                hover_name="theme",
                color_discrete_map={"Positive": "#00ff88", "Mixed": "#ffaa00", "Negative": "#ff4444"},
                title="Momentum vs Sentiment"
            )
            fig2.update_layout(paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e", font_color="white")
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        st.subheader("3. AI-Generated Business Intelligence Report")
        st.markdown(report)

        st.divider()
        st.download_button(
            "⬇️ Download Full Results CSV",
            master_df.to_csv(index=False),
            file_name="consumer_trend_report.csv",
            mime="text/csv"
        )
