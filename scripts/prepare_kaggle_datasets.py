"""Prepare les deux exports Kaggle sans jointure ni conservation des identifiants."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from traque_bots.sources import normalize_bot_dataset, normalize_news_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalise separement les datasets bot et fake news."
    )
    parser.add_argument("--bot-source", required=True, type=Path)
    parser.add_argument("--news-source", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("test"), type=Path)
    arguments = parser.parse_args()

    bot_source = pd.read_csv(arguments.bot_source, low_memory=False)
    news_source = pd.read_csv(arguments.news_source, low_memory=False)
    bot_data = normalize_bot_dataset(bot_source)
    news_data = normalize_news_dataset(news_source)

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    bot_path = arguments.output_dir / "bot_detection_normalized.csv"
    news_path = arguments.output_dir / "fake_news_normalized.csv"
    bot_data.to_csv(bot_path, index=False, encoding="utf-8")
    news_data.to_csv(news_path, index=False, encoding="utf-8")

    print("Bot dataset: {0} lignes, {1:,} octets".format(len(bot_data), bot_path.stat().st_size))
    print(
        "Fake news dataset: {0} lignes, {1:,} octets".format(
            len(news_data), news_path.stat().st_size
        )
    )
    print("Aucun identifiant utilisateur, nom d'auteur ou source n'est conserve.")


if __name__ == "__main__":
    main()
