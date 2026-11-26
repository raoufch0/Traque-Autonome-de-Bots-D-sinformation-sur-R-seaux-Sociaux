"""Validation et creation des caracteristiques agreges par compte."""

from __future__ import annotations

import numpy as np
import pandas as pd


BASE_COLUMNS = {"account_id", "timestamp", "text"}
TARGET_COLUMNS = {"is_spam", "is_fake", "is_bot", "moderation_label"}


def _binary_labels(values: pd.Series, column: str) -> np.ndarray:
    mapping = {
        "1": 1,
        "true": 1,
        "yes": 1,
        "0": 0,
        "false": 0,
        "no": 0,
    }
    labels = values.astype(str).str.strip().str.lower().map(mapping)
    if labels.isna().any():
        raise ValueError("La colonne {0} doit contenir des etiquettes 0/1.".format(column))
    return labels.to_numpy(dtype=np.int64)


def _account_ids(values: pd.Series) -> pd.Series:
    account_ids = values.astype("string").str.strip()
    if account_ids.isna().any() or account_ids.eq("").any():
        raise ValueError("Chaque evenement doit avoir un account_id pseudonymise non vide.")
    return account_ids.astype(str)


def build_account_features(
    events: pd.DataFrame, bins: int = 24
) -> tuple[pd.DataFrame, np.ndarray]:
    """Agrege les textes et encode les dernieres 24 h en sequence horaire."""
    missing = BASE_COLUMNS.difference(events.columns)
    if missing:
        raise ValueError("Colonnes requises manquantes: {0}".format(", ".join(sorted(missing))))
    if bins != 24:
        raise ValueError("Le modele temporel attend exactement 24 cases horaires.")

    frame = events.copy()
    frame["account_id"] = _account_ids(frame["account_id"])
    frame["text"] = frame["text"].fillna("").astype(str)
    frame["normalized_text"] = frame["text"].str.strip().str.lower()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if frame["timestamp"].isna().any():
        raise ValueError("Chaque evenement doit avoir un timestamp ISO-8601 valide.")

    shared_text = (
        frame["normalized_text"].ne("")
        & frame.groupby("normalized_text")["account_id"].transform("nunique").gt(1)
    )
    coordination = shared_text.groupby(frame["account_id"]).mean()

    accounts: list[dict[str, object]] = []
    sequences: list[np.ndarray] = []
    for account_id, group in frame.groupby("account_id", sort=False):
        group = group.sort_values("timestamp")
        latest = group["timestamp"].max()
        recent = group[group["timestamp"] >= latest - pd.Timedelta(hours=24)]
        hour_index = (
            (recent["timestamp"] - (latest - pd.Timedelta(hours=24)))
            .dt.total_seconds()
            .floordiv(3600)
            .clip(0, 23)
            .astype(int)
            .to_numpy()
        )
        counts = np.zeros(bins, dtype=np.float32)
        repeats = np.zeros(bins, dtype=np.float32)
        night_share = np.zeros(bins, dtype=np.float32)
        recent_texts = recent["text"].to_numpy()
        recent_hours = recent["timestamp"].dt.hour.to_numpy()

        for hour in range(bins):
            indices = np.flatnonzero(hour_index == hour)
            if indices.size == 0:
                continue
            counts[hour] = float(indices.size)
            bin_texts = [recent_texts[position] for position in indices]
            repeats[hour] = 1.0 - (len(set(bin_texts)) / len(bin_texts))
            night_share[hour] = float(
                np.mean(
                    [
                        recent_hours[position] < 6 or recent_hours[position] >= 22
                        for position in indices
                    ]
                )
            )

        max_count = max(float(counts.max()), 1.0)
        counts = np.log1p(counts) / np.log1p(max_count)
        sequence = np.stack((counts, repeats, night_share), axis=1).astype(np.float32)
        accounts.append(
            {
                "account_id": account_id,
                "text": " ".join(group["text"].tolist()),
                "coordination_score": float(coordination.loc[account_id]),
            }
        )
        sequences.append(sequence)

    return pd.DataFrame(accounts), np.stack(sequences)


def account_targets(events: pd.DataFrame) -> pd.DataFrame:
    """Valide et ramene les etiquettes d'entrainement au niveau du compte."""
    missing = TARGET_COLUMNS.difference(events.columns)
    if missing:
        raise ValueError("Etiquettes d'entrainement manquantes: {0}".format(", ".join(sorted(missing))))
    frame = events.copy()
    if "account_id" not in frame.columns:
        raise ValueError("Colonne requise manquante: account_id.")
    frame["account_id"] = _account_ids(frame["account_id"])
    for column in ("is_spam", "is_fake", "is_bot"):
        frame[column] = _binary_labels(frame[column], column)
    frame["moderation_label"] = frame["moderation_label"].astype(str).str.strip()
    allowed = {"allow", "review", "restrict_candidate"}
    invalid = set(frame["moderation_label"].unique()).difference(allowed)
    if invalid:
        raise ValueError("Etiquettes de triage inconnues: {0}".format(", ".join(sorted(invalid))))
    inconsistent = (
        frame.groupby("account_id")["moderation_label"].nunique().gt(1)
    )
    if inconsistent.any():
        raise ValueError("Chaque compte doit avoir une etiquette moderation_label unique.")

    aggregations: dict[str, object] = {
        "is_spam": "max",
        "is_fake": "max",
        "is_bot": "max",
        "moderation_label": "first",
    }
    targets = frame.groupby("account_id", sort=False).agg(aggregations).reset_index()
    return targets
