"""Normalisation locale et sans jointure des deux formats Kaggle pris en charge."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


BOT_SOURCE_COLUMNS = {
    "User ID",
    "Tweet",
    "Retweet Count",
    "Mention Count",
    "Follower Count",
    "Verified",
    "Bot Label",
    "Hashtags",
}
BOT_COLUMNS = {
    "account_id",
    "text",
    "is_bot",
    "retweet_count",
    "mention_count",
    "follower_count",
    "verified",
    "hashtag_count",
}
NEWS_SOURCE_COLUMNS = {"title", "text", "date", "label"}
NEWS_COLUMNS = {"article_id", "timestamp", "text", "is_fake"}

_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_MENTION = re.compile(r"(?<![\w])@[A-Z0-9_]{1,30}\b", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)")
_HASHTAG = re.compile(r"(?<!\w)#\w+")


def _redact_contact_details(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = _URL.sub("[URL]", text)
    text = _EMAIL.sub("[EMAIL]", text)
    text = _MENTION.sub("[MENTION]", text)
    return _PHONE.sub("[PHONE]", text).strip()


def _binary_labels(
    values: pd.Series, column: str, positive_label: str, negative_label: str
) -> np.ndarray:
    mapping = {
        "1": 1,
        "true": 1,
        "yes": 1,
        "0": 0,
        "false": 0,
        "no": 0,
        positive_label: 1,
        negative_label: 0,
    }
    labels = values.astype("string").str.strip().str.lower().map(mapping)
    if labels.isna().any():
        raise ValueError(
            "La colonne {0} contient des etiquettes inconnues; seules les classes "
            "bot/human ou fake/real sont prises en charge.".format(column)
        )
    return labels.to_numpy(dtype=np.int64)


def _nonnegative_numbers(values: pd.Series, column: str) -> np.ndarray:
    numbers = pd.to_numeric(values, errors="coerce")
    if numbers.isna().any() or numbers.lt(0).any():
        raise ValueError(
            "La colonne {0} doit contenir des nombres positifs ou nuls.".format(column)
        )
    return numbers.to_numpy(dtype=np.float64)


def _verified_values(values: pd.Series) -> np.ndarray:
    mapping = {
        "1": 1,
        "true": 1,
        "yes": 1,
        "verified": 1,
        "0": 0,
        "false": 0,
        "no": 0,
        "unverified": 0,
    }
    result = values.astype("string").str.strip().str.lower().map(mapping)
    if result.isna().any():
        raise ValueError("La colonne Verified doit contenir des valeurs vrai/faux.")
    return result.to_numpy(dtype=np.int64)


def normalize_bot_dataset(events: pd.DataFrame) -> pd.DataFrame:
    """Convertit les lignes du dataset bot en enregistrements compte pseudonymises."""
    missing_normalized = BOT_COLUMNS.difference(events.columns)
    if not missing_normalized:
        normalized = events[list(BOT_COLUMNS)].copy()
        normalized["account_id"] = normalized["account_id"].astype("string").str.strip()
        if normalized["account_id"].isna().any() or normalized["account_id"].eq("").any():
            raise ValueError("Chaque ligne bot doit avoir un account_id pseudonymise.")
        if normalized["account_id"].duplicated().any():
            raise ValueError("Le fichier bot normalise doit avoir une ligne par compte.")
        normalized["text"] = normalized["text"].map(_redact_contact_details)
        normalized["is_bot"] = _binary_labels(
            normalized["is_bot"], "is_bot", "bot", "human"
        )
        for column in ("retweet_count", "mention_count", "follower_count", "hashtag_count"):
            normalized[column] = _nonnegative_numbers(normalized[column], column)
        normalized["verified"] = _verified_values(normalized["verified"])
        return normalized.reset_index(drop=True)

    missing_source = BOT_SOURCE_COLUMNS.difference(events.columns)
    if missing_source:
        raise ValueError(
            "Colonnes du dataset bot manquantes: {0}".format(
                ", ".join(sorted(missing_source))
            )
        )
    source = events.reset_index(drop=True)
    source_ids = source["User ID"].astype("string").str.strip()
    if source_ids.isna().any() or source_ids.eq("").any():
        raise ValueError("Le dataset bot contient des User ID vides.")
    if source_ids.duplicated().any():
        raise ValueError(
            "Ce format bot doit avoir une ligne par compte; regroupez les publications "
            "avant de calculer des motifs temporels."
        )

    text = source["Tweet"].map(_redact_contact_details)
    hashtags = source["Hashtags"].fillna("").astype(str).map(
        lambda value: len(_HASHTAG.findall(value))
    )
    normalized = pd.DataFrame(
        {
            "account_id": ["bot-{0:06d}".format(index + 1) for index in range(len(source))],
            "text": text,
            "is_bot": _binary_labels(source["Bot Label"], "Bot Label", "bot", "human"),
            "retweet_count": _nonnegative_numbers(source["Retweet Count"], "Retweet Count"),
            "mention_count": _nonnegative_numbers(source["Mention Count"], "Mention Count"),
            "follower_count": _nonnegative_numbers(source["Follower Count"], "Follower Count"),
            "verified": _verified_values(source["Verified"]),
            "hashtag_count": hashtags.to_numpy(dtype=np.float64),
        }
    )
    return normalized


def normalize_news_dataset(events: pd.DataFrame) -> pd.DataFrame:
    """Convertit les articles en lignes independantes, sans les joindre aux comptes."""
    missing_normalized = NEWS_COLUMNS.difference(events.columns)
    if not missing_normalized:
        normalized = events[["article_id", "timestamp", "text", "is_fake"]].copy()
        normalized["article_id"] = normalized["article_id"].astype("string").str.strip()
        if normalized["article_id"].isna().any() or normalized["article_id"].eq("").any():
            raise ValueError("Chaque article doit avoir un article_id.")
        normalized["text"] = normalized["text"].map(_redact_contact_details)
        normalized["is_fake"] = _binary_labels(
            normalized["is_fake"], "is_fake", "fake", "real"
        )
    else:
        missing_source = NEWS_SOURCE_COLUMNS.difference(events.columns)
        if missing_source:
            raise ValueError(
                "Colonnes du dataset fake news manquantes: {0}".format(
                    ", ".join(sorted(missing_source))
                )
            )
        source = events.reset_index(drop=True)
        dates = pd.to_datetime(source["date"], errors="coerce", utc=True)
        if dates.isna().any():
            raise ValueError("Chaque article doit avoir une date ISO-8601 valide.")
        title = source["title"].fillna("").astype(str).str.strip()
        body = source["text"].fillna("").astype(str).str.strip()
        text = (title + " " + body).str.strip().map(_redact_contact_details)
        if text.eq("").any():
            raise ValueError("Chaque article doit contenir un titre ou un texte.")
        normalized = pd.DataFrame(
            {
                "article_id": [
                    "article-{0:06d}".format(index + 1) for index in range(len(source))
                ],
                "timestamp": dates,
                "text": text,
                "is_fake": _binary_labels(source["label"], "label", "fake", "real"),
            }
        )

    normalized["timestamp"] = pd.to_datetime(
        normalized["timestamp"], errors="coerce", utc=True
    )
    if normalized["timestamp"].isna().any():
        raise ValueError("Chaque article doit avoir une date ISO-8601 valide.")
    if normalized["text"].eq("").any():
        raise ValueError("Chaque article doit contenir du texte.")
    if normalized["article_id"].duplicated().any():
        raise ValueError("Le fichier fake news doit avoir un article_id unique par ligne.")
    normalized["timestamp"] = normalized["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return normalized.reset_index(drop=True)
