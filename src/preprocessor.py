"""
preprocessor.py
---------------
Cleans raw review text before it hits the topic model.
Handles noise common in user-generated content.
"""

import re
import unicodedata
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ─── Cleaning functions ────────────────────────────────────────────────────────

def _normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)

def _remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", " ", text)

def _remove_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)

def _remove_emails(text: str) -> str:
    return re.sub(r"\S+@\S+", " ", text)

def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def _remove_repeated_chars(text: str) -> str:
    # "sooooo good" → "soo good" (keep max 2 repeats for expressiveness)
    return re.sub(r"(.)\1{2,}", r"\1\1", text)

def _remove_emojis(text: str) -> str:
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F9FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(" ", text)


# ─── Main preprocessor ────────────────────────────────────────────────────────

class TextPreprocessor:
    """
    Configurable text cleaner for review data.

    Parameters
    ----------
    remove_urls    : strip http links
    remove_html    : strip HTML tags
    remove_emails  : strip email addresses
    remove_emojis  : strip emoji characters
    lowercase      : convert to lowercase
    min_length     : drop texts shorter than N chars after cleaning
    """

    def __init__(
        self,
        remove_urls: bool = True,
        remove_html: bool = True,
        remove_emails: bool = True,
        remove_emojis: bool = True,
        lowercase: bool = True,
        min_length: int = 20,
    ):
        self.remove_urls = remove_urls
        self.remove_html = remove_html
        self.remove_emails = remove_emails
        self.remove_emojis = remove_emojis
        self.lowercase = lowercase
        self.min_length = min_length

    def clean(self, text: str) -> Optional[str]:
        """Clean a single text string. Returns None if too short after cleaning."""
        if not isinstance(text, str):
            return None

        text = _normalize_unicode(text)

        if self.remove_html:
            text = _remove_html(text)
        if self.remove_urls:
            text = _remove_urls(text)
        if self.remove_emails:
            text = _remove_emails(text)
        if self.remove_emojis:
            text = _remove_emojis(text)

        text = _remove_repeated_chars(text)
        text = _normalize_whitespace(text)

        if self.lowercase:
            text = text.lower()

        if len(text) < self.min_length:
            return None

        return text

    def clean_batch(self, texts: list[str]) -> tuple[list[str], list[int]]:
        """
        Clean a list of texts.

        Returns
        -------
        cleaned_texts : list of cleaned strings (None entries removed)
        valid_indices : original indices that survived cleaning
        """
        cleaned = []
        valid_indices = []

        for i, text in enumerate(texts):
            result = self.clean(text)
            if result is not None:
                cleaned.append(result)
                valid_indices.append(i)

        dropped = len(texts) - len(cleaned)
        if dropped > 0:
            logger.info(f"Preprocessing: dropped {dropped} short/invalid texts ({len(cleaned)} remaining)")

        return cleaned, valid_indices


# ─── Convenience function ─────────────────────────────────────────────────────

def preprocess(texts: list[str], **kwargs) -> tuple[list[str], list[int]]:
    """Shortcut: preprocess a list of texts with default settings."""
    return TextPreprocessor(**kwargs).clean_batch(texts)
