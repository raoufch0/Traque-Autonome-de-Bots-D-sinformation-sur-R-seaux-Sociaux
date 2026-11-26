# Traque autonome de bots et de desinformation

Prototype hors ligne pour reperer des signaux de spam, de contenu etiquete comme
fausse information et d'activite coordonnee. Le programme produit des scores et
des recommandations de triage; il ne contacte aucune plateforme et ne bannit
ni ne signale aucun compte.

## Contenu

- Naive Bayes multinomial pour les etiquettes lexicales `is_spam` et `is_fake`.
- Reseau recurrent LSTM sur les activites horaires d'un compte.
- Signal de coordination fonde sur la reutilisation du meme texte entre comptes.
- Regression logistique pour classer les dossiers en `allow`, `review` ou
  `restrict_candidate`.
- Demonstration reproductible avec des donnees synthetiques, sans identifiants
  reels ni acces reseau.
- [Rapport technique](docs/rapport_technique.md), avec schema d'entree, limites
  et recommandations d'evaluation.

`is_fake` represente une etiquette du jeu de donnees, pas une verification
factuelle. Les recommandations `restrict_candidate` doivent etre examinees par
une personne; elles ne constituent pas une decision de bannissement.

## Lancer le projet

Python 3.10 ou plus recent est requis. Ouvrir PowerShell dans le dossier du
projet (celui qui contient `pyproject.toml`), puis creer l'environnement et
installer le projet:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
```

Demarrer ensuite le serveur web:

```powershell
.\.venv\Scripts\python.exe -m traque_bots.web
```

Garder cette fenetre PowerShell ouverte et ouvrir http://127.0.0.1:5000 dans le
navigateur. Pour arreter le serveur, revenir dans PowerShell et appuyer sur
`Ctrl+C`. Lors des prochains lancements, il suffit de refaire la commande de
demarrage ci-dessus; l'installation ne se fait qu'une fois.

Le tableau de bord permet de lancer une demonstration synthetique ou d'importer
un export Bot Detection, un export Fake News ou un CSV complet normalise.
L'import est limite a 64 Mo; les donnees sont analysees en memoire et ne sont
pas sauvegardees par l'application. L'interface est locale et ne contacte
aucune plateforme.

La demo et les analyses separent les comptes ou articles avant l'entrainement
et l'evaluation. L'interface affiche les metriques sur les elements retenus et
les scores du modele adapte a la source; un export CSV des resultats affiches
est egalement disponible. Pour lancer la demonstration console a la place du
serveur web:

```powershell
.\.venv\Scripts\python.exe -m traque_bots
```

Les poids sont entraines a la volee et ne sont pas sauvegardes.

## Jeux Kaggle pris en charge

Le script `scripts/prepare_kaggle_datasets.py` convertit les deux exports
fournis en fichiers separes dans `test/`. Exemple depuis la racine du projet:

```powershell
python scripts/prepare_kaggle_datasets.py `
  --bot-source "C:\chemin\bot_detection_data.csv" `
  --news-source "C:\chemin\fake_news_dataset.csv" `
  --output-dir test
```

Il cree `test/bot_detection_normalized.csv` et
`test/fake_news_normalized.csv`. Les identifiants de comptes et d'articles sont
remplaces par des identifiants techniques, les champs username, location,
auteur et source sont exclus, et les mentions, URL, courriels et numeros de
telephone dans le texte sont masques. Le dataset fake news conserve le texte
des articles pour permettre l'entrainement lexical. Ces CSV sont generes
localement et ne sont pas inclus dans le depot; utilisez les exports Kaggle
locaux en entree du script pour les recreer.

Les deux sources ne sont pas jointes: le fichier bot a une ligne par compte et
une etiquette `Bot Label`; le fichier fake news a une ligne par article et une
etiquette `fake`/`real`. Aucune cle ne relie ces comptes aux articles. Le
tableau de bord detecte ces schemas et les evalue separement:

- Bot Detection: regression logistique sur les metadonnees de profil
  (`Retweet Count`, `Mention Count`, `Follower Count`, `Verified` et nombre de
  hashtags).
- Fake News: Naive Bayes et regression logistique TF-IDF sur le titre et le
  texte. Une validation interne selectionne le modele principal; les deux
  resultats sur le lot de test restent visibles dans le tableau de bord.

Le dataset bot fourni contient un seul tweet par compte: il ne permet donc pas
d'evaluer un rythme de publication ni le LSTM temporel. Les articles fake news
ne sont pas des sequences d'activite de comptes; le LSTM n'est donc pas
applique a cette source. Le label `fake` reste une etiquette de source, pas une
verification factuelle. Pour entrainer le pipeline complet Naive Bayes + LSTM
temporel + triage logistique, utiliser le schema complet ci-dessous avec
plusieurs publications par compte et des etiquettes verifiees.

### Schema complet du pipeline

Le pipeline complet attend une ligne par publication et les colonnes suivantes:

| Colonne | Contenu |
| --- | --- |
| `account_id` | Identifiant pseudonymise, stable dans le fichier |
| `timestamp` | Date ISO-8601 de publication |
| `text` | Texte de la publication |
| `is_spam` | 0/1, etiquette lexicale de spam |
| `is_fake` | 0/1, etiquette source associee au contenu |
| `is_bot` | 0/1, etiquette source du compte |
| `moderation_label` | `allow`, `review` ou `restrict_candidate` |

Ne pas fusionner des etiquettes provenant de definitions incompatibles sans
documenter leur provenance. Pour le pipeline complet, utiliser des
identifiants pseudonymises et exclure les donnees personnelles inutiles. Il faut
au moins 12 comptes distincts et plusieurs classes de triage dans la partition
d'entrainement. Si le jeu est trop petit ou invalide, l'interface affiche une
erreur de validation au lieu de produire des resultats partiels.

## Structure

```text
src/traque_bots/
  data.py       generation des evenements fictifs
  features.py   validation et agregation par compte
  models.py     Naive Bayes, LSTM et regression logistique
  analysis.py   separation des comptes, entrainement et metriques partagees
  sources.py    normalisation des deux exports Kaggle sans jointure
  demo.py       demonstration console
  web.py        serveur Flask local et import CSV
  templates/    tableau de bord web en francais
  static/       styles et interactions du tableau de bord
scripts/
  prepare_kaggle_datasets.py   preparation des fichiers locaux dans test/
test/             datasets normalises generes a partir des sources fournies
```

Le serveur ecoute uniquement sur `127.0.0.1` et utilise le serveur de
developpement Flask; ne pas l'exposer directement sur un reseau public.
