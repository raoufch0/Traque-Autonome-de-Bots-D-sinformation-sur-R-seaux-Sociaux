"""Modeles de detection et de triage pour un usage de recherche hors ligne."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from torch import nn

from .features import account_targets, build_account_features


def _positive_probability(model: object, values: object) -> np.ndarray:
    probabilities = model.predict_proba(values)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(probabilities), dtype=np.float32)
    return probabilities[:, classes.index(1)]


class _TemporalNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.recurrent = nn.LSTM(input_size=3, hidden_size=16, batch_first=True)
        self.output = nn.Linear(16, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.recurrent(sequence)
        return self.output(hidden[-1]).squeeze(1)


class TemporalBotClassifier:
    """LSTM binaire qui apprend des rythmes de publication horaires."""

    def __init__(self, epochs: int = 35) -> None:
        self.epochs = epochs
        torch.manual_seed(17)
        self.network = _TemporalNetwork()

    def fit(self, sequences: np.ndarray, labels: np.ndarray) -> "TemporalBotClassifier":
        torch.manual_seed(17)
        x = torch.tensor(sequences, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=0.01)
        loss_function = nn.BCEWithLogitsLoss()
        self.network.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = loss_function(self.network(x), y)
            loss.backward()
            optimizer.step()
        return self

    def predict_proba(self, sequences: np.ndarray) -> np.ndarray:
        self.network.eval()
        with torch.no_grad():
            logits = self.network(torch.tensor(sequences, dtype=torch.float32))
            scores = torch.sigmoid(logits).cpu().numpy()
        return scores.astype(np.float32)


class DetectionPipeline:
    """Combine scores lexicaux et temporels pour proposer un triage humain."""

    def __init__(self) -> None:
        self.spam_model = make_pipeline(CountVectorizer(ngram_range=(1, 2)), MultinomialNB())
        self.fake_model = make_pipeline(CountVectorizer(ngram_range=(1, 2)), MultinomialNB())
        self.temporal_model = TemporalBotClassifier()
        self.moderation_model = LogisticRegression(max_iter=500, class_weight="balanced")
        self._fitted = False

    def fit(self, events: pd.DataFrame) -> "DetectionPipeline":
        accounts, sequences = build_account_features(events)
        targets = account_targets(events)
        accounts = accounts.merge(targets, on="account_id", validate="one_to_one")

        self.spam_model.fit(accounts["text"], accounts["is_spam"])
        self.fake_model.fit(accounts["text"], accounts["is_fake"])
        self.temporal_model.fit(sequences, accounts["is_bot"].to_numpy(dtype=np.int64))

        features = self._score_features(
            accounts["text"], sequences, accounts["coordination_score"].to_numpy()
        )
        self.moderation_model.fit(features, accounts["moderation_label"])
        self._fitted = True
        return self

    def _score_features(
        self, texts: pd.Series, sequences: np.ndarray, coordination: np.ndarray
    ) -> np.ndarray:
        spam = _positive_probability(self.spam_model, texts)
        fake = _positive_probability(self.fake_model, texts)
        bot = self.temporal_model.predict_proba(sequences)
        return np.column_stack((spam, fake, bot, coordination))

    def predict(self, events: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Entrainer DetectionPipeline avant de demander des predictions.")
        accounts, sequences = build_account_features(events)
        scores = self._score_features(
            accounts["text"], sequences, accounts["coordination_score"].to_numpy()
        )
        recommendations = self.moderation_model.predict(scores)
        return pd.DataFrame(
            {
                "account_id": accounts["account_id"],
                "spam_score": scores[:, 0],
                "fake_label_score": scores[:, 1],
                "bot_pattern_score": scores[:, 2],
                "coordination_score": scores[:, 3],
                "recommendation": recommendations,
                "human_review_required": True,
            }
        )
