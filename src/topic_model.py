"""
topic_model.py
--------------
BERTopic wrapper for consumer theme discovery.
"""

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class ConsumerTopicModel:
    def __init__(self, n_topics="auto", min_topic_size=15, top_n_words=10):
        self.n_topics = n_topics
        self.min_topic_size = min_topic_size
        self.top_n_words = top_n_words
        self.model = None
        self.topics = None
        self.probs = None

    def build(self):
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
        hdbscan_model = HDBSCAN(min_cluster_size=self.min_topic_size, metric="euclidean", prediction_data=True)

        self.model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            top_n_words=self.top_n_words,
            verbose=True
        )
        return self

    def fit(self, texts):
        logger.info(f"Fitting BERTopic on {len(texts)} texts...")
        self.topics, self.probs = self.model.fit_transform(texts)
        logger.info(f"Found {len(self.model.get_topic_info()) - 1} topics")
        return self

    def get_topic_table(self):
        info = self.model.get_topic_info()
        info = info[info["Topic"] != -1]
        info["top_words"] = info["Topic"].apply(
            lambda t: ", ".join([w for w, _ in self.model.get_topic(t)[:5]])
        )
        return info[["Topic", "Count", "Name", "top_words"]].reset_index(drop=True)

    def get_topic_per_doc(self, texts):
        df = pd.DataFrame({"text": texts, "topic_id": self.topics})
        topic_info = self.model.get_topic_info().set_index("Topic")["Name"].to_dict()
        df["topic_name"] = df["topic_id"].map(topic_info)
        return df
