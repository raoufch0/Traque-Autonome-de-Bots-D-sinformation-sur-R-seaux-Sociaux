"""Generation de donnees synthetiques pour la demonstration hors ligne."""

from __future__ import annotations

import random

import pandas as pd


SPAM_TEXTS = [
    "Gagnez un cadeau maintenant cliquez ici",
    "Offre exclusive cliquez pour recevoir votre prix",
    "Rejoignez notre offre gratuite aujourd'hui",
]
FAKE_TEXTS = [
    "Un rapport choc affirme que cette nouvelle est prouvee",
    "Alerte partagez cette information avant sa suppression",
    "Une source anonyme revele un evenement incroyable",
]
NEUTRAL_TEXTS = [
    "Le marche ouvre en hausse apres les annonces du matin",
    "Voici quelques photos de notre promenade en ville",
    "Le conseil municipal publiera son compte rendu demain",
    "J'ai termine un livre interessant pendant le voyage",
    "Les previsions annoncent de la pluie pour le weekend",
    "Merci a toutes les personnes presentes a la reunion",
]


def make_demo_events(accounts: int = 180, seed: int = 17) -> pd.DataFrame:
    """Construit un jeu fictif deterministe, sans donnees de comptes reels."""
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    origin = pd.Timestamp("2026-01-15T00:00:00Z")

    for index in range(accounts):
        account_id = "demo-{0:04d}".format(index)
        is_bot = int(rng.random() < 0.48)
        is_spam = int(rng.random() < (0.72 if is_bot else 0.08))
        is_fake = int(rng.random() < (0.35 if is_bot else 0.18))
        count = rng.randint(14, 22)

        if is_spam and is_fake:
            texts = [
                "{0}. {1}".format(rng.choice(SPAM_TEXTS), rng.choice(FAKE_TEXTS))
                for _ in range(count)
            ]
        elif is_spam:
            texts = [rng.choice(SPAM_TEXTS) for _ in range(count)]
        elif is_fake:
            texts = [rng.choice(FAKE_TEXTS) for _ in range(count)]
        else:
            texts = [rng.choice(NEUTRAL_TEXTS) for _ in range(count)]

        if is_bot:
            offsets = sorted(rng.randint(0, 100) for _ in range(count))
        else:
            offsets = sorted(rng.randint(0, 23 * 60) for _ in range(count))

        if is_fake and is_spam:
            moderation_label = "restrict_candidate"
        elif is_fake or is_spam or is_bot:
            moderation_label = "review"
        else:
            moderation_label = "allow"

        for post_index, offset in enumerate(offsets):
            rows.append(
                {
                    "account_id": account_id,
                    "timestamp": origin + pd.Timedelta(minutes=offset),
                    "text": texts[post_index],
                    "is_spam": is_spam,
                    "is_fake": is_fake,
                    "is_bot": is_bot,
                    "moderation_label": moderation_label,
                }
            )

    return pd.DataFrame(rows)
