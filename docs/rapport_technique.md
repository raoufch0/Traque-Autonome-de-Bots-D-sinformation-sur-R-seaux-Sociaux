# Rapport technique

## Projet 9 — Traque de bots et analyse de désinformation

| Métadonnée | Valeur |
| --- | --- |
| Nature | Conception, implémentation et évaluation d’un prototype |
| Statut | Expérimental — usage hors ligne, avec revue humaine obligatoire |

> **Conclusion principale.** Les données fournies ont été évaluées sans
> fabriquer de lien entre comptes et articles. Les résultats restent proches
> du hasard : ce prototype démontre un parcours d’analyse, mais ne constitue
> pas un système fiable de détection ou de modération.

## Synthèse

Le projet fournit une application locale permettant d’analyser des signaux
associés au spam, aux étiquettes « fake », aux rythmes de publication et à la
réutilisation de textes. Pour le format complet, ces signaux alimentent un
modèle de triage qui propose une priorité de revue. L’application n’interagit
avec aucune plateforme et ne restreint, ne signale ni ne bannit de compte.

Les deux exports Kaggle fournis décrivent des objets différents : des comptes
avec une publication dans le jeu Bot Detection, et des articles dans le jeu
Fake News. Ils sont normalisés et analysés séparément. Sur le lot de test Fake
News, la régression logistique TF-IDF atteint 51,18 % d’exactitude, contre
50,42 % pour Naive Bayes. Le modèle principal reste sélectionné sur une
validation interne, et non en fonction du lot de test. Ces scores ne justifient
aucun usage opérationnel.

## 1. Objectif, périmètre et principes

### 1.1 Objectif

Le prototype vise à :

- produire des signaux interprétables à partir de textes et de métadonnées ;
- expérimenter Naive Bayes, un LSTM temporel et une régression logistique ;
- comparer des modèles sur des données étiquetées mises à part de
  l’entraînement ;
- orienter une revue humaine, sans transformer un score en preuve.

### 1.2 Limites fonctionnelles

Le terme « traque autonome » décrit le sujet du projet, et non le comportement
de l’application. Le logiciel n’effectue ni surveillance en temps réel,
ni collecte sur les réseaux sociaux, ni action de neutralisation. Les valeurs
`allow`, `review` et `restrict_candidate` sont des recommandations de triage.
La valeur `restrict_candidate` ne déclenche aucune restriction.

L’étiquette `is_fake` ou `fake` reproduit l’annotation de la source. Elle ne
constitue pas une vérification indépendante de la véracité d’un article.

## 2. Architecture et traitement

### 2.1 Parcours des données

1. L’utilisateur lance la démonstration synthétique ou importe un CSV.
2. Le schéma détermine le parcours adapté : pipeline complet, Bot Detection
   ou Fake News.
3. Les données sont validées et, pour les exports Kaggle, normalisées sans
   joindre les deux sources.
4. Les partitions d’entraînement, de validation et de test sont constituées
   avant l’évaluation.
5. Le tableau de bord présente les métriques et les prédictions du lot évalué ;
   les résultats filtrés peuvent être exportés en CSV.

### 2.2 Principaux modules

| Module | Responsabilité |
| --- | --- |
| [`analysis.py`](../src/traque_bots/analysis.py) | Validation, partitions, entraînement, sélection des modèles et métriques |
| [`features.py`](../src/traque_bots/features.py) | Validation du format complet et agrégation des événements par compte |
| [`models.py`](../src/traque_bots/models.py) | Classifieurs lexicaux, LSTM temporel et triage logistique |
| [`sources.py`](../src/traque_bots/sources.py) | Normalisation séparée des deux exports Kaggle |
| [`web.py`](../src/traque_bots/web.py) | Interface Flask locale et réception des fichiers |
| [`prepare_kaggle_datasets.py`](../scripts/prepare_kaggle_datasets.py) | Préparation reproductible des fichiers normalisés |

### 2.3 Traitement local et protection des données

Le serveur écoute sur `127.0.0.1`. La taille maximale d’un import est de
64 Mio ; les fichiers sont traités en mémoire et ne sont pas enregistrés par
l’application. Lors de la préparation des exports Kaggle, les identifiants
d’origine ne sont pas conservés : des identifiants techniques sont générés et
les champs username, location, author et source sont exclus des sorties.
Les motifs courants d’URL, d’adresse courriel, de mention et de numéro de
téléphone sont masqués dans le texte. Pour un CSV complet déjà normalisé,
l’utilisateur doit fournir lui-même des identifiants pseudonymisés.

Ces opérations réduisent certaines expositions, mais ne garantissent pas
l’anonymisation : les règles de masquage peuvent manquer des données
personnelles ou des identifiants indirects. Une revue des données et une
politique de conservation restent nécessaires avant tout usage hors
expérimentation locale.

## 3. Données et schémas

### 3.1 Format complet

Le pipeline complet attend une ligne par publication et les colonnes suivantes :

| Colonne | Description |
| --- | --- |
| `account_id` | Identifiant pseudonymisé stable au sein du fichier |
| `timestamp` | Horodatage ISO-8601 |
| `text` | Texte de la publication |
| `is_spam` | Étiquette binaire associée au spam |
| `is_fake` | Étiquette binaire fournie par la source |
| `is_bot` | Étiquette binaire associée au compte |
| `moderation_label` | `allow`, `review` ou `restrict_candidate` |

Les identifiants et horodatages sont validés. Pour chaque compte, les
étiquettes binaires sont agrégées par maximum ; une seule valeur
`moderation_label` est autorisée par compte. Le pipeline complet exige au
moins 12 comptes distincts et au moins deux classes de triage dans la partition
d’entraînement.

### 3.2 Exports fournis

| Source | Volume et unité | Étiquettes observées | Parcours d’analyse |
| --- | --- | --- | --- |
| Bot Detection | 50 000 lignes, une par compte | 24 982 humains / 25 018 bots | Régression logistique sur des caractéristiques de profil |
| Fake News | 20 000 articles | 9 944 « real » / 10 056 « fake » | Comparaison de Naive Bayes et de la régression logistique TF-IDF |

Le jeu Bot Detection contient une seule publication par compte. Il ne permet
donc pas d’évaluer une séquence de publication. Les articles Fake News ne sont
pas des historiques d’activité de comptes. Aucun identifiant commun ne relie
ces fichiers : les joindre par position, par ordre de ligne ou par étiquette
créerait des relations artificielles et n’est pas fait.

Les champs d’identité ou de provenance exclus ne sont pas utilisés comme
caractéristiques de prédiction. Le résultat demeure toutefois dépendant des
définitions et de la qualité des étiquettes sources.

## 4. Modèles et caractéristiques

### 4.1 Pipeline complet

#### Analyse lexicale

Deux pipelines indépendants combinent `CountVectorizer` en unigrammes et
bigrammes avec Naive Bayes multinomial. Ils estiment les étiquettes `is_spam`
et `is_fake` à partir des textes concaténés par compte. Les probabilités
produites sont des scores du modèle, et non des mesures de certitude ou de
véracité.

#### Motifs temporels

Le LSTM traite les 24 heures qui précèdent la publication la plus récente de
chaque compte. Chaque heure est représentée par trois caractéristiques :

1. le volume de publications, normalisé par compte ;
2. la proportion de textes répétés ;
3. la proportion de publications nocturnes, entre 22 h et 6 h UTC.

Le réseau a une entrée de trois caractéristiques par heure et une couche
récurrente de 16 unités cachées. L’implémentation utilise Adam, un taux
d’apprentissage de 0,01 et 35 époques. Un motif temporel atypique est un signal
à examiner ; il ne démontre ni automatisation ni intention malveillante.

#### Réutilisation de texte

Le score de coordination correspond à la part des événements d’un compte
associés à un texte non vide également observé sur au moins un autre compte
dans le même lot. Cette mesure simple ne distingue pas une campagne coordonnée
d’une citation, d’un contenu viral ou d’une autre cause légitime de répétition.

#### Triage

Une régression logistique classe les quatre signaux — spam, étiquette fake,
motif temporel et réutilisation textuelle — dans l’une des catégories
`allow`, `review` ou `restrict_candidate`. Toute sortie indique qu’une revue
humaine est requise ; aucune action sur un compte n’est effectuée.

### 4.2 Parcours adaptés aux exports

**Bot Detection.** Les valeurs numériques de retweets, mentions, abonnés et
hashtags sont transformées avec `log1p`, puis standardisées. Le statut
`verified` est également utilisé. Une régression logistique avec pondération
des classes (`max_iter=1000`) estime l’étiquette bot. Ce parcours n’utilise pas
le LSTM.

**Fake News.** Le texte d’entrée est la concaténation du titre et du corps de
l’article. Deux candidats sont comparés :

- **Naive Bayes multinomial :** `CountVectorizer` sur unigrammes et bigrammes,
  avec un vocabulaire plafonné à 100 000 termes et `min_df=2` ;
- **régression logistique TF-IDF :** unigrammes et bigrammes, pondération
  TF-IDF sous-linéaire, suppression des mots vides anglais et classes
  pondérées.

Le modèle principal est choisi à partir d’une validation interne. Les
métriques des deux candidats sont ensuite présentées sur le même lot de test.
Le LSTM temporel n’est pas appliqué aux articles. Un essai exploratoire distinct
d’un LSTM textuel a été réalisé, mais il n’est pas intégré au tableau de bord.

## 5. Protocole d’évaluation et résultats

### 5.1 Protocole

Les partitions externes sont stratifiées et utilisent la graine 17 : 75 % des
données pour l’entraînement et 25 % pour le test. Les modèles Bot Detection et
Fake News évaluent respectivement 12 500 comptes et 5 000 articles.

Pour Fake News, la partition d’entraînement de 15 000 articles est divisée une
seconde fois : 12 000 articles servent à entraîner les candidats de validation
et 3 000 à les comparer (graine 29). Le candidat retenu est ensuite réentraîné
sur les 15 000 articles avant l’évaluation sur les 5 000 articles de test.
La sélection privilégie le F1 macro, puis l’exactitude en cas d’égalité.
Cette séparation évite de choisir le modèle en fonction du résultat du test.

Les métriques rapportées sont l’exactitude et le F1 macro. Le F1 macro accorde
le même poids aux classes, ce qui le rend complémentaire à l’exactitude.

### 5.2 Résultats sur les fichiers fournis

| Jeu évalué | Modèle | Exactitude | F1 macro | Taille du test |
| --- | --- | ---: | ---: | ---: |
| Bot Detection | Régression logistique sur les profils | 49,60 % | 49,59 % | 12 500 comptes |
| Fake News | Naive Bayes | 50,42 % | 50,42 % | 5 000 articles |
| Fake News | Régression logistique TF-IDF | 51,18 % | 51,18 % | 5 000 articles |
| Fake News, essai exploratoire | LSTM textuel court | 50,04 % | Non rapporté | 5 000 articles |

Pour Fake News, Naive Bayes a été retenu par la validation interne, même si la
régression logistique obtient un score légèrement supérieur sur le lot de test.
Ce n’est pas une contradiction : le lot de test sert à mesurer le résultat
final, pas à sélectionner le modèle.

Les résultats proches de 50 % indiquent que les modèles et les caractéristiques
actuels n’extraient pas de signal prédictif fiable de ces fichiers. Le résultat
de l’essai LSTM textuel est exploratoire et ne démontre pas un avantage du LSTM.
Ces métriques ne doivent pas être présentées comme des performances sur
Twitter/X ni utilisées pour prendre une décision de modération.

### 5.3 Interprétation et validité

L’évaluation est une mesure sur les partitions aléatoires décrites ci-dessus.
Elle ne constitue pas une validation temporelle ou une validation par éditeur,
et ne mesure pas la généralisation à une autre plateforme, période, langue ou
population. Les métriques seules ne permettent pas non plus de quantifier
l’impact des faux positifs, les différences entre groupes ou la calibration
des scores.

La démonstration synthétique valide le chemin technique sur des données
générées ; elle ne constitue pas une évaluation indépendante de la qualité des
modèles sur des données réelles.

## 6. Limites et risques

- **Qualité des étiquettes :** les classes reprennent les annotations des
  fichiers ; leur exactitude et leur définition ne sont pas vérifiées par le
  prototype.
- **Faible pouvoir prédictif mesuré :** les scores observés sont proches du
  hasard et interdisent toute conclusion opérationnelle.
- **Données temporelles absentes :** une publication par compte ne suffit pas
  pour apprendre des rythmes de publication avec le LSTM.
- **Généralisation :** un découpage aléatoire ne remplace pas un test sur une
  période ou une source entièrement séparée.
- **Faux positifs :** satire, citation, événement viral, activité humaine
  intense ou automatisation légitime peuvent ressembler aux signaux observés.
- **Triage empilé :** dans le pipeline complet, le modèle de triage est ajusté
  sur des scores générés par les modèles de signal entraînés sur ces mêmes
  comptes. Une évaluation plus robuste devrait produire ces caractéristiques
  d’entraînement par validation croisée, hors pli.
- **Données personnelles :** le masquage par motifs courants n’est pas une
  garantie d’anonymisation ni un substitut à une analyse de confidentialité.
- **Coût humain :** une erreur de classification peut affecter injustement un
  utilisateur ; aucun score ne doit être diffusé comme une accusation.

## 7. Garde-fous et travaux recommandés

### Garde-fous actuels

- traitement local et absence d’appel à une plateforme ;
- conservation des deux sources séparées ;
- exclusion des champs d’identité et de provenance des caractéristiques ;
- affichage d’un avertissement indiquant que les scores ne sont pas des preuves ;
- revue humaine requise et absence d’API de signalement ou de bannissement.

### Étapes recommandées avant tout usage réel

1. Établir la provenance, la définition et la procédure d’annotation de chaque
   étiquette ; faire vérifier les exemples par des annotateurs qualifiés.
2. Collecter plusieurs publications par compte avec des horodatages cohérents
   avant d’évaluer le LSTM temporel.
3. Dédupliquer les contenus et compléter l’évaluation par des partitions
   temporelles et, lorsque pertinent, par source ou éditeur.
4. Rapporter précision, rappel, F1 par classe, matrices de confusion,
   calibration et intervalles d’incertitude ; examiner les faux positifs.
5. Évaluer les performances par groupes pertinents et documenter les risques
   de biais avant de tester une politique de triage.
6. Mettre en place une gouvernance des accès, une durée de conservation
   limitée, une procédure de contestation et une supervision humaine.
7. Utiliser une validation croisée hors pli pour entraîner le modèle de triage
   du pipeline complet, puis le réévaluer sur un lot indépendant.

## 8. Exécution et reproductibilité

Les instructions d’installation et de démarrage détaillées figurent dans le
[`README.md`](../README.md). Depuis la racine du projet, sous PowerShell :

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m traque_bots.web
```

Le tableau de bord est ensuite accessible à l’adresse
`http://127.0.0.1:5000`. La démonstration console peut être lancée avec :

```powershell
.\.venv\Scripts\python.exe -m traque_bots
```

Les graines des séparations et des modèles sont définies dans le code afin de
rendre ces exécutions reproductibles dans l’environnement utilisé. La
reproductibilité ne garantit pas des résultats identiques entre versions de
dépendances ou environnements matériels différents. Les modèles sont entraînés
à chaque analyse ; leurs poids ne sont pas sauvegardés.

## Conclusion

Le projet met en œuvre un prototype de recherche qui combine classification
lexicale, analyse temporelle et triage humain, tout en séparant correctement
les jeux Bot Detection et Fake News. L’évaluation des fichiers fournis ne met
toutefois pas en évidence de performance suffisante pour une utilisation
opérationnelle. La priorité est d’améliorer la qualité des données, de
construire une évaluation plus exigeante et de maintenir une supervision
humaine ; aucune neutralisation automatisée n’est justifiée par les résultats
actuels.
