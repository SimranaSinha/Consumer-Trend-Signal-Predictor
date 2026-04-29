# 🔍 Consumer Trend Signal Predictor
### AI-Powered Market Intelligence from Product Reviews

> Upload any product review dataset → automatically discover emerging consumer themes, track sentiment, predict trend momentum, and generate business-ready insights.

---

## 🚀 What it does

Most companies read reviews manually. This model does it automatically at scale.

**Input:** Any CSV with product reviews (skincare, food, electronics, wellness — anything)

**Output:**

| Theme | Sentiment | Trend | Momentum |
|---|---|---|---|
| Lips Lip Balm | Positive | ↑ Rising | 1.0 |
| Acne Treatment | Mixed | ↑ Rising | 0.71 |
| Sensitive Skin | Mixed | → Stable | 0.12 |

---

## ⚙️ How it works
```
Raw Reviews (CSV)
↓
Text Preprocessing
↓
BERTopic — discover hidden consumer themes
↓
RoBERTa — sentiment per theme
↓
Trend Momentum Scoring — rising / stable / declining
↓
GPT-4o — business strategy report
↓
Streamlit Dashboard
```

---

## 🏗️ Project Structure

```
consumer-trend-predictor/
├── src/
│   ├── ingestor.py        # Auto-detects columns from any CSV
│   ├── preprocessor.py    # Text cleaning pipeline
│   ├── topic_model.py     # BERTopic wrapper
│   ├── sentiment.py       # RoBERTa sentiment scoring
│   ├── trend_scorer.py    # Temporal momentum scoring
│   └── narrator.py        # GPT-4o insight generator
├── app.py                 # Streamlit dashboard
├── requirements.txt
└── README.md
```
---

## 🛠️ Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/consumer-trend-signal-predictor.git
cd consumer-trend-signal-predictor
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Topic Modeling | BERTopic + UMAP + HDBSCAN |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Sentiment | cardiffnlp/twitter-roberta-base-sentiment |
| Trend Scoring | scipy linear regression |
| LLM Narration | GPT-4o |
| Dashboard | Streamlit + Plotly |

---

## 📊 Dataset

Tested on [Sephora Skincare Reviews](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews) — 270k+ reviews from 2008–2023.

Works with any CSV containing a review text column. Date and rating columns are optional.



