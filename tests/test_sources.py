from __future__ import annotations

import unittest

import pandas as pd

from traque_bots.analysis import analyze_bot_dataset, analyze_news_dataset
from traque_bots.sources import normalize_bot_dataset, normalize_news_dataset


class SourceNormalizationTests(unittest.TestCase):
    def test_bot_export_is_pseudonymized_and_contact_details_are_redacted(self) -> None:
        source = pd.DataFrame(
            {
                "User ID": [987654321],
                "Username": ["private-handle"],
                "Tweet": ["Hi @private https://example.com mail@example.com"],
                "Retweet Count": [2],
                "Mention Count": [1],
                "Follower Count": [12],
                "Verified": [False],
                "Bot Label": [1],
                "Hashtags": ["#news"],
            }
        )

        normalized = normalize_bot_dataset(source)

        self.assertEqual(normalized.loc[0, "account_id"], "bot-000001")
        self.assertNotIn("private-handle", normalized.loc[0, "text"])
        self.assertNotIn("987654321", normalized.to_csv(index=False))
        self.assertIn("[MENTION]", normalized.loc[0, "text"])
        self.assertIn("[URL]", normalized.loc[0, "text"])
        self.assertNotIn("Username", normalized.columns)

    def test_news_export_keeps_source_labels_but_drops_author_metadata(self) -> None:
        source = pd.DataFrame(
            {
                "title": ["Report"],
                "text": ["A source article."],
                "date": ["2026-01-15"],
                "source": ["Example publication"],
                "author": ["Example Author"],
                "category": ["news"],
                "label": ["fake"],
            }
        )

        normalized = normalize_news_dataset(source)

        self.assertEqual(normalized.loc[0, "article_id"], "article-000001")
        self.assertEqual(normalized.loc[0, "is_fake"], 1)
        self.assertNotIn("author", normalized.columns)
        self.assertNotIn("source", normalized.columns)


class SourceAnalysisTests(unittest.TestCase):
    def test_bot_dataset_uses_profile_signals_without_claiming_temporal_analysis(self) -> None:
        events = pd.DataFrame(
            {
                "account_id": ["bot-{0:02d}".format(index) for index in range(20)],
                "text": ["post {0}".format(index) for index in range(20)],
                "is_bot": [index % 2 for index in range(20)],
                "retweet_count": [index * 2 for index in range(20)],
                "mention_count": [index % 3 for index in range(20)],
                "follower_count": [index * 10 for index in range(20)],
                "verified": [index % 2 for index in range(20)],
                "hashtag_count": [index % 4 for index in range(20)],
            }
        )

        result = analyze_bot_dataset(events)

        self.assertEqual(result["source"], "bot_dataset")
        self.assertEqual(result["summary"]["evaluated_accounts"], 5)
        self.assertIn("logistique", result["analysis"]["signal_definitions"][0]["label"])
        self.assertIn("pas d'evaluer un rythme temporel", result["analysis"]["guard_description"])

    def test_fake_news_dataset_trains_and_reports_article_labels(self) -> None:
        events = pd.DataFrame(
            {
                "article_id": ["article-{0:02d}".format(index) for index in range(20)],
                "timestamp": ["2026-01-15T00:00:00Z"] * 20,
                "text": [
                    ("false fabricated rumor " if index % 2 else "verified local report ")
                    + "shared article number {0}".format(index)
                    for index in range(20)
                ],
                "is_fake": [index % 2 for index in range(20)],
            }
        )

        result = analyze_news_dataset(events)

        self.assertEqual(result["source"], "fake_news_dataset")
        self.assertEqual(result["summary"]["evaluated_accounts"], 5)
        self.assertEqual(result["analysis"]["class_labels"], {"real": "Reel", "fake": "Fake"})
        self.assertIn("ne constitue pas une verification", result["analysis"]["guard_description"])
        comparisons = result["summary"]["model_comparison"]
        self.assertEqual(
            {model["name"] for model in comparisons},
            {"Naive Bayes", "Regression logistique TF-IDF"},
        )
        self.assertIn(result["analysis"]["selected_model"], {
            model["name"] for model in comparisons
        })
        self.assertIn("validation interne", result["analysis"]["completion_message"])
        self.assertIn("LSTM temporel", result["analysis"]["model_note"])


if __name__ == "__main__":
    unittest.main()
