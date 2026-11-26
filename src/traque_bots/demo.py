"""Demonstration console: entrainement et evaluation sur des comptes fictifs."""

from __future__ import annotations

import pandas as pd

from .analysis import analyze_demo


def main() -> None:
    result = analyze_demo()
    summary = result["summary"]
    predictions = pd.DataFrame(result["predictions"])

    print("Demo hors ligne - donnees synthetiques uniquement")
    print("Comptes d'entrainement: {0}".format(summary["train_accounts"]))
    print("Comptes de test independants: {0}".format(summary["evaluated_accounts"]))
    print("Exactitude indicative du triage: {0:.1%}".format(summary["accuracy"]))
    print("\nExemples (scores = signaux, pas des preuves):")
    print(
        predictions.head(8).to_string(
            index=False,
            formatters={
                "spam_score": "{:.2f}".format,
                "fake_label_score": "{:.2f}".format,
                "bot_pattern_score": "{:.2f}".format,
                "coordination_score": "{:.2f}".format,
            },
        )
    )
    print("\nToute recommandation exige une verification humaine.")
