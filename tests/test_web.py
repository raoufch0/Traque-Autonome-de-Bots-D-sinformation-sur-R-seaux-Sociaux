from __future__ import annotations

from io import BytesIO
import unittest

import pandas as pd

from traque_bots.web import app


class WebApplicationTests(unittest.TestCase):
    def setUp(self) -> None:
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_dashboard_is_served_locally(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Comprendre les signaux", response.data)

    def test_demo_endpoint_returns_held_out_account_metrics(self) -> None:
        response = self.client.post("/api/analyze")

        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(result["source"], "synthetic")
        self.assertEqual(result["summary"]["train_accounts"], 135)
        self.assertEqual(result["summary"]["evaluated_accounts"], 45)
        self.assertTrue(result["predictions"][0]["human_review_required"])

    def test_normalized_csv_upload_runs_the_evaluation(self) -> None:
        from traque_bots.data import make_demo_events

        csv_content = BytesIO()
        make_demo_events().to_csv(csv_content, index=False)
        csv_content.seek(0)
        response = self.client.post(
            "/api/analyze",
            data={"file": (csv_content, "normalized-events.csv")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(result["source"], "normalized")
        self.assertEqual(result["summary"]["evaluated_accounts"], 45)

    def test_bot_source_upload_uses_the_profile_model(self) -> None:
        source = pd.DataFrame(
            {
                "User ID": range(1000, 1020),
                "Username": ["private-{0}".format(index) for index in range(20)],
                "Tweet": ["post @private{0}".format(index) for index in range(20)],
                "Retweet Count": [index * 2 for index in range(20)],
                "Mention Count": [index % 3 for index in range(20)],
                "Follower Count": [index * 10 for index in range(20)],
                "Verified": [index % 2 == 0 for index in range(20)],
                "Bot Label": [index % 2 for index in range(20)],
                "Hashtags": ["#test"] * 20,
            }
        )
        upload = BytesIO()
        source.to_csv(upload, index=False)
        upload.seek(0)
        response = self.client.post(
            "/api/analyze",
            data={"file": (upload, "bot_detection_data.csv")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(result["source"], "bot_dataset")
        self.assertEqual(result["summary"]["evaluated_accounts"], 5)
        self.assertTrue(result["predictions"][0]["entity_id"].startswith("bot-"))

    def test_news_source_upload_uses_the_text_model(self) -> None:
        source = pd.DataFrame(
            {
                "title": ["Report {0}".format(index) for index in range(20)],
                "text": [
                    ("false fabricated rumor " if index % 2 else "verified local report ")
                    + "story {0}".format(index)
                    for index in range(20)
                ],
                "date": ["2026-01-15"] * 20,
                "source": ["Example publication"] * 20,
                "author": ["Example author"] * 20,
                "category": ["news"] * 20,
                "label": ["fake" if index % 2 else "real" for index in range(20)],
            }
        )
        upload = BytesIO()
        source.to_csv(upload, index=False)
        upload.seek(0)
        response = self.client.post(
            "/api/analyze",
            data={"file": (upload, "fake_news_dataset.csv")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(result["source"], "fake_news_dataset")
        self.assertEqual(result["summary"]["evaluated_accounts"], 5)
        self.assertTrue(result["predictions"][0]["entity_id"].startswith("article-"))

    def test_csv_endpoint_reports_missing_training_labels(self) -> None:
        rows = [
            "account-{0},2026-01-15T00:00:00Z,hello,0,0,allow".format(index)
            for index in range(12)
        ]
        csv_content = (
            "account_id,timestamp,text,is_spam,is_bot,moderation_label\n"
            + "\n".join(rows)
        ).encode()
        response = self.client.post(
            "/api/analyze",
            data={"file": (BytesIO(csv_content), "dataset.csv")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("is_fake", response.get_json()["error"])

    def test_non_csv_upload_is_rejected(self) -> None:
        response = self.client.post(
            "/api/analyze",
            data={"file": (BytesIO(b"not a dataset"), "dataset.txt")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn(".csv", response.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
