---
title: "Exporter les données d’évaluation"
description: "Exportez les résultats d’évaluation par programmation à l’aide de l’API REST Evaluation."
---

Les Teams qui effectuent des évaluations dans W&amp;B Weave ont souvent besoin des résultats d’évaluation en dehors de l’interface Weave. Voici quelques cas d’usage courants :

* Extraire les métriques vers des feuilles de calcul ou des notebooks pour des analyses et visualisations personnalisées.
* Alimenter des pipelines CI/CD avec les résultats d’évaluation afin d’autoriser ou de bloquer les déploiements.
* Partager les résultats avec des parties prenantes qui n’ont pas de licences W&amp;B, via des outils de BI comme Looker ou des tableaux de bord internes.
* Mettre en place des pipelines de reporting automatisés qui agrègent les scores sur plusieurs projets.

L’[API REST Evaluation v2](https://trace.wandb.ai/docs) expose des concepts d’évaluation spécifiques : runs d’évaluation, prédictions, scores et scorers. Elle fournit ainsi une sortie plus riche et mieux structurée, avec des statistiques typées pour les scorers et des entrées de jeu de données résolues, par rapport à l’API Calls généraliste.

<div id="api-endpoints-used">
  ## Points de terminaison API utilisés
</div>

Les extraits de code de cette page utilisent les points de terminaison suivants de la [v2 Evaluation REST API](https://trace.wandb.ai/docs) :

* `GET /v2/{entity}/{project}/evaluation_runs` : Liste les runs d’évaluation d’un projet, avec des filtres facultatifs par référence d’évaluation, référence de modèle ou ID du run.
* `GET /v2/{entity}/{project}/evaluation_runs/{evaluation_run_id}` : Lit un run d’évaluation unique afin d’en récupérer le modèle, la référence d’évaluation, le statut, les horodatages et la synthèse.
* `POST /v2/{entity}/{project}/eval_results/query` : Récupère des lignes de résultats d’évaluation groupées pour une ou plusieurs évaluations. Renvoie, pour chaque ligne, des essais avec la sortie du modèle, les scores et, éventuellement, les entrées résolues de la ligne du jeu de données. Renvoie également des statistiques agrégées du scorer lorsqu’elles sont demandées.
* `GET /v2/{entity}/{project}/predictions/{prediction_id}` : Lit une prédiction individuelle avec ses entrées, sa sortie et sa référence de modèle.

L’authentification utilise HTTP Basic, avec `api` comme nom d’utilisateur et votre clé API W&amp;B comme mot de passe.

<div id="prerequisites">
  ## Prérequis
</div>

Les exemples de cette page utilisent Python, mais l&#39;API REST Evaluation est indépendante du langage : vous pouvez appeler les mêmes points de terminaison depuis TypeScript ou depuis n&#39;importe quel client HTTP.

* Python 3.7 ou version ultérieure.
* La bibliothèque `requests`. Installez-la avec `pip install requests`.
* Une clé API W&amp;B, définie dans la variable d&#39;environnement `WANDB_API_KEY`. Obtenez votre clé sur [wandb.ai/settings](https://wandb.ai/settings).

<div id="set-up-authentication">
  ## Configurer l’authentification
</div>

```python
import json
import os

import requests

TRACE_BASE = "https://trace.wandb.ai"
AUTH = ("api", os.environ["WANDB_API_KEY"])

entity = "my-team"
project = "my-project"
```


<div id="list-evaluation-runs">
  ## Lister les runs d’évaluation
</div>

Récupérez les runs d’évaluation récents d’un projet et affichez, pour chacun, des détails tels que l’ID et le statut.

```python
resp = requests.get(
    f"{TRACE_BASE}/v2/{entity}/{project}/evaluation_runs",
    auth=AUTH,
)
runs = [json.loads(line) for line in resp.text.strip().splitlines()]

for run in runs:
    print(run["evaluation_run_id"], run.get("status"))
```


<div id="read-a-single-evaluation-run">
  ## Lire un run d’évaluation spécifique
</div>

Récupérez les détails d’un run d’évaluation spécifique, notamment son modèle, sa référence d’évaluation, son statut et ses horodatages.

```python
eval_run_id = "<evaluation-run-id>"

resp = requests.get(
    f"{TRACE_BASE}/v2/{entity}/{project}/evaluation_runs/{eval_run_id}",
    auth=AUTH,
)
eval_run = resp.json()
print(eval_run["evaluation_run_id"], eval_run.get("status"), eval_run.get("model"))
```


<div id="get-predictions-and-scores">
  ## Obtenir les prédictions et les scores
</div>

Utilisez le point de terminaison `eval_results/query` pour récupérer les résultats ligne par ligne d’un run d’Évaluation. Chaque ligne inclut les entrées résolues du jeu de données, la sortie du modèle et les résultats individuels du scorer. Définissez `include_rows`, `include_raw_data_rows` et `resolve_row_refs` pour obtenir le niveau de détail complet pour chaque ligne.

```python
eval_run_id = "<evaluation-run-id>"

resp = requests.post(
    f"{TRACE_BASE}/v2/{entity}/{project}/eval_results/query",
    json={
        "evaluation_run_ids": [eval_run_id],
        "include_rows": True,
        "include_raw_data_rows": True,
        "resolve_row_refs": True,
    },
    auth=AUTH,
)
results = resp.json()

for row in results["rows"]:
    inputs = row.get("raw_data_row")
    for ev in row.get("evaluations", []):
        for trial in ev.get("trials", []):
            output = trial.get("model_output")
            scores = trial.get("scores", {})
            print("Input:", inputs)
            print("Output:", output)
            print("Scores:", scores)
```


<div id="get-aggregated-scores">
  ## Obtenir des scores agrégés
</div>

Le même point de terminaison `eval_results/query` peut également renvoyer des statistiques agrégées sur les évaluateurs au lieu de données ligne par ligne. Définissez `include_summary` pour obtenir des métriques de synthèse, comme les taux de réussite pour les évaluateurs binaires et les moyennes pour les évaluateurs continus.

```python
resp = requests.post(
    f"{TRACE_BASE}/v2/{entity}/{project}/eval_results/query",
    json={
        "evaluation_run_ids": [eval_run_id],
        "include_summary": True,
        "include_rows": False,
    },
    auth=AUTH,
)
results = resp.json()

for ev in results["summary"]["evaluations"]:
    for stat in ev["scorer_stats"]:
        print(stat["scorer_key"], stat.get("value_type"), stat.get("pass_rate") or stat.get("numeric_mean"))
```


<div id="read-a-single-prediction">
  ## Lire une seule prédiction
</div>

Récupérez les informations complètes d’une prédiction donnée, y compris ses entrées, sa sortie et la référence du modèle.

```python
prediction_id = "<predict-call-id>"

resp = requests.get(
    f"{TRACE_BASE}/v2/{entity}/{project}/predictions/{prediction_id}",
    auth=AUTH,
)
prediction = resp.json()
print(prediction)
```


<div id="how-to-use-row-digests">
  ## Comment utiliser les empreintes de ligne
</div>

Chaque ligne de résultat du point de terminaison `eval_results/query` inclut un `row_digest`, un hachage de contenu qui identifie de manière unique une entrée spécifique dans le jeu de données d&#39;évaluation en fonction de son contenu, et non de sa position. Les empreintes de ligne sont utiles pour :

* **Comparaison entre évaluations** : lorsque vous exécutez deux modèles différents sur le même jeu de données, les lignes ayant la même empreinte correspondent à la même entrée. Vous pouvez effectuer une jointure sur `row_digest` pour comparer les performances de différents modèles sur exactement la même tâche.
* **Déduplication** : si la même tâche apparaît dans plusieurs suites d&#39;évaluation, l&#39;empreinte vous permet de l&#39;identifier.
* **Reproductibilité** : l&#39;empreinte est déterminée par le contenu. Ainsi, si quelqu&#39;un modifie une ligne du jeu de données (en changeant le texte de l&#39;instruction, le barème ou d&#39;autres champs), elle obtient une nouvelle empreinte. Vous pouvez vérifier si deux runs d&#39;évaluation ont utilisé des entrées identiques ou des versions légèrement différentes.