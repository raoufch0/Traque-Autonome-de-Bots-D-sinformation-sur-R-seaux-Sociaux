"""Evaluation partagee pour la demonstration console et l'interface web."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import make_demo_events
from .features import BASE_COLUMNS, TARGET_COLUMNS, account_targets
from .models import DetectionPipeline
from .sources import (
    BOT_COLUMNS,
    BOT_SOURCE_COLUMNS,
    NEWS_COLUMNS,
    NEWS_SOURCE_COLUMNS,
    normalize_bot_dataset,
    normalize_news_dataset,
)


MODERATION_LABELS = ("allow", "review", "restrict_candidate")
MINIMUM_ACCOUNTS = 12
MAXIMUM_DISPLAYED_ACCOUNTS = 20_000
MINIMUM_SOURCE_ROWS = 12


def _binary_split(
    entity_ids: np.ndarray, labels: np.ndarray, entity_name: str
) -> tuple[np.ndarray, np.ndarray]:
    if len(entity_ids) < MINIMUM_SOURCE_ROWS:
        raise ValueError(
            "Il faut au moins {0} {1} pour separer entrainement et evaluation."
            .format(MINIMUM_SOURCE_ROWS, entity_name)
        )
    counts = np.bincount(labels, minlength=2)
    if np.any(counts < 2):
        raise ValueError(
            "Le jeu doit contenir au moins deux exemples de chaque classe pour "
            "entrainer et evaluer le modele."
        )
    train_ids, test_ids = train_test_split(
        entity_ids,
        test_size=0.25,
        random_state=17,
        stratify=labels,
    )
    return train_ids, test_ids


def _source_result(
    *,
    source: str,
    entity_ids: np.ndarray,
    expected: np.ndarray,
    predicted: np.ndarray,
    positive_scores: np.ndarray,
    train_count: int,
    entity_name: str,
    negative_label: str,
    positive_label: str,
    signal_key: str,
    signal_name: str,
    title: str,
    distribution_title: str,
    guard_title: str,
    guard_description: str,
    table_title: str,
    prediction_column: str,
    reference_column: str,
    metric_highlight_label: str,
    metric_highlight_note: str,
    completion_message: str,
) -> dict[str, object]:
    expected_labels = np.where(expected == 1, positive_label, negative_label)
    predicted_labels = np.where(predicted == 1, positive_label, negative_label)
    recommendation_counts = {
        negative_label: int(np.count_nonzero(predicted == 0)),
        positive_label: int(np.count_nonzero(predicted == 1)),
    }
    average_score = round(float(np.mean(positive_scores)), 4)
    rows = [
        {
            "entity_id": str(entity_id),
            signal_key: round(float(score), 4),
            "recommendation": str(prediction),
            "expected": str(reference),
            "human_review_required": True,
        }
        for entity_id, score, prediction, reference in zip(
            entity_ids, positive_scores, predicted_labels, expected_labels
        )
    ]
    rows = rows[:MAXIMUM_DISPLAYED_ACCOUNTS]
    return {
        "source": source,
        "analysis": {
            "title": title,
            "entity_name": entity_name,
            "unit_label": entity_name.lower(),
            "unit_plural": "comptes" if entity_name == "Compte" else "articles",
            "prediction_column": prediction_column,
            "reference_column": reference_column,
            "signal_title": "Signal principal",
            "distribution_title": distribution_title,
            "guard_title": guard_title,
            "guard_description": guard_description,
            "table_title": table_title,
            "metric_highlight_label": metric_highlight_label,
            "metric_highlight_note": metric_highlight_note,
            "completion_message": completion_message,
            "class_labels": {
                negative_label: negative_label.capitalize(),
                positive_label: positive_label.capitalize(),
            },
            "signal_definitions": [{"key": signal_key, "label": signal_name}],
        },
        "summary": {
            "accounts_total": int(len(entity_ids) + train_count),
            "posts_total": int(len(entity_ids) + train_count),
            "train_accounts": int(train_count),
            "evaluated_accounts": int(len(entity_ids)),
            "accuracy": round(float(accuracy_score(expected_labels, predicted_labels)), 4),
            "macro_f1": round(
                float(
                    f1_score(
                        expected_labels,
                        predicted_labels,
                        labels=[negative_label, positive_label],
                        average="macro",
                        zero_division=0,
                    )
                ),
                4,
            ),
            "recommendations": recommendation_counts,
            "average_scores": {signal_key: average_score},
            "displayed_accounts": len(rows),
            "highlighted_count": recommendation_counts[positive_label],
        },
        "predictions": rows,
    }


def analyze_events(events: pd.DataFrame) -> dict[str, object]:
    """Entraine sur des comptes distincts et evalue les comptes mis de cote."""
    missing_columns = (BASE_COLUMNS | TARGET_COLUMNS).difference(events.columns)
    if missing_columns:
        raise ValueError(
            "Colonnes requises manquantes: {0}".format(", ".join(sorted(missing_columns)))
        )
    if events.empty:
        raise ValueError("Le fichier ne contient aucun evenement.")
    events = events.copy()
    events["account_id"] = events["account_id"].astype("string").str.strip()

    targets = account_targets(events)
    if len(targets) < MINIMUM_ACCOUNTS:
        raise ValueError(
            "Il faut au moins {0} comptes distincts pour separer entrainement et evaluation."
            .format(MINIMUM_ACCOUNTS)
        )

    labels = targets["moderation_label"]
    test_size = math.ceil(len(targets) * 0.25)
    can_stratify = (
        labels.value_counts().ge(2).all()
        and test_size >= labels.nunique()
    )
    train_ids, test_ids = train_test_split(
        targets["account_id"].to_numpy(),
        test_size=0.25,
        random_state=17,
        stratify=labels if can_stratify else None,
    )
    train_events = events[events["account_id"].astype(str).isin(train_ids)]
    test_events = events[events["account_id"].astype(str).isin(test_ids)]

    train_targets = account_targets(train_events)
    if train_targets["moderation_label"].nunique() < 2:
        raise ValueError(
            "La partition d'entrainement ne contient qu'une classe de triage. "
            "Ajoutez des comptes etiquetes dans plusieurs classes."
        )

    pipeline = DetectionPipeline().fit(train_events)
    predictions = pipeline.predict(test_events)
    expected = account_targets(test_events)[["account_id", "moderation_label"]].rename(
        columns={"moderation_label": "expected"}
    )
    evaluated = predictions.merge(expected, on="account_id", validate="one_to_one")

    recommendation_counts = {
        label: int(evaluated["recommendation"].eq(label).sum())
        for label in MODERATION_LABELS
    }
    average_scores = {
        column: round(float(evaluated[column].mean()), 4)
        for column in (
            "spam_score",
            "fake_label_score",
            "bot_pattern_score",
            "coordination_score",
        )
    }
    rows = evaluated.head(MAXIMUM_DISPLAYED_ACCOUNTS).copy()
    for column in average_scores:
        rows[column] = rows[column].round(4)

    return {
        "source": "normalized",
        "analysis": {
            "title": "Triage comportemental",
            "entity_name": "Compte",
            "unit_label": "compte",
            "unit_plural": "comptes",
            "prediction_column": "RECOMMANDATION",
            "reference_column": "REFERENCE",
            "signal_title": "Signaux comportementaux",
            "distribution_title": "Recommandations de triage",
            "guard_title": "Decision humaine, toujours.",
            "guard_description": "Aucune restriction de compte n'est executee.",
            "table_title": "Comptes mis en evidence",
            "metric_highlight_label": "A examiner",
            "metric_highlight_note": "Revue humaine toujours requise",
            "completion_message": (
                "Analyse terminee. Les predictions portent sur des comptes gardes "
                "a part pour l'evaluation."
            ),
            "class_labels": {
                "allow": "Aucun signal fort",
                "review": "Revue recommandee",
                "restrict_candidate": "Priorite de revue",
            },
            "signal_definitions": [
                {"key": "spam_score", "label": "Spam lexical"},
                {"key": "fake_label_score", "label": "Etiquette fake"},
                {"key": "bot_pattern_score", "label": "Motif temporel (LSTM)"},
                {"key": "coordination_score", "label": "Coordination textuelle"},
            ],
        },
        "summary": {
            "accounts_total": int(len(targets)),
            "posts_total": int(len(events)),
            "train_accounts": int(len(train_ids)),
            "evaluated_accounts": int(len(evaluated)),
            "accuracy": round(
                float(evaluated["recommendation"].eq(evaluated["expected"]).mean()), 4
            ),
            "macro_f1": round(
                float(
                    f1_score(
                        evaluated["expected"],
                        evaluated["recommendation"],
                        labels=MODERATION_LABELS,
                        average="macro",
                        zero_division=0,
                    )
                ),
                4,
            ),
            "recommendations": recommendation_counts,
            "average_scores": average_scores,
            "displayed_accounts": int(len(rows)),
            "highlighted_count": int(
                recommendation_counts["review"] + recommendation_counts["restrict_candidate"]
            ),
        },
        "predictions": rows.assign(entity_id=rows["account_id"]).to_dict(orient="records"),
    }


def analyze_bot_dataset(events: pd.DataFrame) -> dict[str, object]:
    """Evalue les etiquettes bot avec une regression sur les metadonnees disponibles."""
    data = normalize_bot_dataset(events)
    labels = data["is_bot"].to_numpy(dtype=np.int64)
    train_ids, test_ids = _binary_split(
        data["account_id"].to_numpy(), labels, "comptes"
    )
    train_mask = data["account_id"].isin(train_ids)
    test_data = data[data["account_id"].isin(test_ids)].reset_index(drop=True)
    train_data = data[train_mask]
    feature_columns = (
        "retweet_count",
        "mention_count",
        "follower_count",
        "verified",
        "hashtag_count",
    )
    train_features = train_data[list(feature_columns)].copy()
    test_features = test_data[list(feature_columns)].copy()
    for column in ("retweet_count", "mention_count", "follower_count", "hashtag_count"):
        train_features[column] = np.log1p(train_features[column])
        test_features[column] = np.log1p(test_features[column])

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=17),
    )
    model.fit(train_features, train_data["is_bot"])
    predictions = model.predict(test_features).astype(np.int64)
    scores = model.predict_proba(test_features)[:, list(model.classes_).index(1)]
    result = _source_result(
        source="bot_dataset",
        entity_ids=test_data["account_id"].to_numpy(),
        expected=test_data["is_bot"].to_numpy(dtype=np.int64),
        predicted=predictions,
        positive_scores=scores,
        train_count=len(train_data),
        entity_name="Compte",
        negative_label="human",
        positive_label="bot",
        signal_key="bot_score",
        signal_name="Score bot · regression logistique",
        title="Detection de bots",
        distribution_title="Classes de comptes predites",
        guard_title="Une seule publication par compte",
        guard_description=(
            "Le jeu ne permet pas d'evaluer un rythme temporel (LSTM)."
        ),
        table_title="Comptes evalues",
        prediction_column="PREDICTION",
        reference_column="ETIQUETTE",
        metric_highlight_label="Comptes predits bots",
        metric_highlight_note="Etiquette du dataset, pas une preuve",
        completion_message=(
            "Evaluation terminee sur des comptes gardes a part. "
            "Le modele utilise les metadonnees du profil."
        ),
    )
    result["summary"]["accounts_total"] = len(data)
    result["summary"]["posts_total"] = len(data)
    result["analysis"]["class_labels"] = {"human": "Humain", "bot": "Bot"}
    return result


def analyze_news_dataset(events: pd.DataFrame) -> dict[str, object]:
    """Compare Naive Bayes et logistique sur le texte fake/real des articles."""
    data = normalize_news_dataset(events)
    labels = data["is_fake"].to_numpy(dtype=np.int64)
    train_ids, test_ids = _binary_split(
        data["article_id"].to_numpy(), labels, "articles"
    )
    train_mask = data["article_id"].isin(train_ids)
    train_data = data[train_mask].reset_index(drop=True)
    test_data = data[data["article_id"].isin(test_ids)].reset_index(drop=True)

    def naive_bayes_model():
        return make_pipeline(
            CountVectorizer(ngram_range=(1, 2), max_features=100_000, min_df=2),
            MultinomialNB(),
        )

    def logistic_model():
        return make_pipeline(
            TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=100_000,
                min_df=2,
                sublinear_tf=True,
                stop_words="english",
            ),
            LogisticRegression(
                max_iter=1000,
                C=1.0,
                class_weight="balanced",
                random_state=17,
            ),
        )

    subtrain_indices, validation_indices = train_test_split(
        np.arange(len(train_data)),
        test_size=0.2,
        random_state=29,
        stratify=train_data["is_fake"],
    )
    validation_text = train_data.iloc[validation_indices]["text"]
    validation_labels = train_data.iloc[validation_indices]["is_fake"].to_numpy()
    validation_scores: dict[str, tuple[float, float]] = {}
    for model_name, model_factory in (
        ("Naive Bayes", naive_bayes_model),
        ("Regression logistique TF-IDF", logistic_model),
    ):
        candidate = model_factory()
        candidate.fit(
            train_data.iloc[subtrain_indices]["text"],
            train_data.iloc[subtrain_indices]["is_fake"],
        )
        validation_predictions = candidate.predict(validation_text).astype(np.int64)
        validation_scores[model_name] = (
            float(f1_score(validation_labels, validation_predictions, average="macro")),
            float(accuracy_score(validation_labels, validation_predictions)),
        )
    selected_model_name = max(
        validation_scores,
        key=lambda name: (validation_scores[name][0], validation_scores[name][1]),
    )

    fitted_models = {
        "Naive Bayes": naive_bayes_model(),
        "Regression logistique TF-IDF": logistic_model(),
    }
    test_metrics: list[dict[str, object]] = []
    for model_name, model in fitted_models.items():
        model.fit(train_data["text"], train_data["is_fake"])
        test_predictions = model.predict(test_data["text"]).astype(np.int64)
        test_metrics.append(
            {
                "name": model_name,
                "accuracy": round(float(accuracy_score(test_data["is_fake"], test_predictions)), 4),
                "macro_f1": round(
                    float(
                        f1_score(
                            test_data["is_fake"],
                            test_predictions,
                            labels=[0, 1],
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    4,
                ),
            }
        )

    selected_model = fitted_models[selected_model_name]
    predictions = selected_model.predict(test_data["text"]).astype(np.int64)
    scores = selected_model.predict_proba(test_data["text"])[:, list(selected_model.classes_).index(1)]
    result = _source_result(
        source="fake_news_dataset",
        entity_ids=test_data["article_id"].to_numpy(),
        expected=test_data["is_fake"].to_numpy(dtype=np.int64),
        predicted=predictions,
        positive_scores=scores,
        train_count=len(train_data),
        entity_name="Article",
        negative_label="real",
        positive_label="fake",
        signal_key="fake_score",
        signal_name="Score fake · {0}".format(selected_model_name),
        title="Classification des articles",
        distribution_title="Etiquettes predites",
        guard_title="Etiquette source uniquement",
        guard_description="Ce score ne constitue pas une verification des faits.",
        table_title="Articles evalues",
        prediction_column="PREDICTION",
        reference_column="ETIQUETTE",
        metric_highlight_label="Articles etiquetes fake",
        metric_highlight_note="Etiquette du dataset, pas une verification",
        completion_message=(
            "Modele selectionne sur une validation interne ({0}); les metriques "
            "ci-dessus portent sur le lot de test garde a part."
        ).format(selected_model_name),
    )
    result["summary"]["accounts_total"] = len(data)
    result["summary"]["posts_total"] = len(data)
    result["summary"]["model_comparison"] = test_metrics
    result["analysis"]["selected_model"] = selected_model_name
    result["analysis"]["model_note"] = (
        "Le corpus contient des articles independants, pas des sequences d'activite "
        "de comptes. Le LSTM temporel n'est donc pas applicable a cette source."
    )
    result["analysis"]["class_labels"] = {"real": "Reel", "fake": "Fake"}
    return result


def analyze_uploaded_dataset(events: pd.DataFrame) -> dict[str, object]:
    """Detecte un schema pris en charge sans joindre des sources independantes."""
    columns = set(events.columns)
    if BASE_COLUMNS <= columns:
        return analyze_events(events)
    if BOT_COLUMNS <= columns or BOT_SOURCE_COLUMNS <= columns:
        return analyze_bot_dataset(events)
    if NEWS_COLUMNS <= columns or NEWS_SOURCE_COLUMNS <= columns:
        return analyze_news_dataset(events)
    raise ValueError(
        "Schema CSV non reconnu. Importez le schema normalise du README, le dataset "
        "bot (User ID/Tweet/Bot Label) ou le dataset fake news (title/text/date/label)."
    )


def analyze_demo() -> dict[str, object]:
    """Lance une evaluation reproductible sur des comptes synthetiques."""
    result = analyze_events(make_demo_events())
    result["source"] = "synthetic"
    return result
