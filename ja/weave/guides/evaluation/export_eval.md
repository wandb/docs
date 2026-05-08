---
title: "評価データをエクスポートする"
description: "Evaluation REST API を使用して、評価結果をプログラムからエクスポートします。"
---

W&amp;B Weave で評価を実行する Teams では、Weave UI の外で評価結果が必要になることがよくあります。一般的なユースケースは次のとおりです。

* カスタム分析や可視化のために、メトリクスをスプレッドシートやノートブックに取り込みます。
* 評価結果を CI/CD パイプラインに組み込み、デプロイの可否判定に使用します。
* W&amp;B のシートを持たない関係者と、Looker などの BI ツールや社内ダッシュボードを通じて結果を共有します。
* 複数のプロジェクトにまたがるスコアを集約する自動レポート パイプラインを構築します。

[v2 Evaluation REST API](https://trace.wandb.ai/docs) は、評価に特化した概念である評価 run、予測、スコア、Scorer を提供します。その結果、汎用的な Calls API と比べて、型付きの Scorer 統計情報や解決済みのデータセット入力を含む、よりリッチで構造化された出力を取得できます。

<div id="api-endpoints-used">
  ## 使用する API エンドポイント
</div>

このページのスニペットでは、[v2 Evaluation REST API](https://trace.wandb.ai/docs) の以下のエンドポイントを使用します。

* `GET /v2/{entity}/{project}/evaluation_runs`: プロジェクト内の評価 run を一覧表示します。評価参照、モデル参照、または run ID によるフィルターを任意で指定できます。
* `GET /v2/{entity}/{project}/evaluation_runs/{evaluation_run_id}`: 単一の評価 run を取得し、そのモデル、評価参照、ステータス、タイムスタンプ、summary を取得します。
* `POST /v2/{entity}/{project}/eval_results/query`: 1 つ以上の評価について、グループ化された評価結果の行を取得します。各行について、モデル出力、スコア、必要に応じて解決済みのデータセット行 inputs を含む trial を返します。要求した場合は、集計された Scorer の統計も返します。
* `GET /v2/{entity}/{project}/predictions/{prediction_id}`: 個別の予測を取得し、その inputs、出力、およびモデル参照を返します。

認証には HTTP Basic を使用し、ユーザー名には `api`、パスワードには W&amp;B APIキーを使用します。

<div id="prerequisites">
  ## 前提条件
</div>

このページの例では Python を使用していますが、評価 REST API は言語に依存しません。TypeScript や任意の HTTP クライアントから、同じエンドポイントを呼び出せます。

* Python 3.7 以降。
* `requests` ライブラリ。`pip install requests` でインストールしてください。
* `WANDB_API_KEY` 環境変数に設定した W&amp;B APIキー。[wandb.ai/settings](https://wandb.ai/settings) でキーを取得してください。

<div id="set-up-authentication">
  ## 認証を設定する
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
  ## 評価 run の一覧表示
</div>

プロジェクト内の最近の評価 run を取得し、ID やステータスなど、各 run の詳細を一覧表示します。

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
  ## 単一の評価 run を取得する
</div>

特定の評価 run の詳細 (モデル、評価参照、ステータス、タイムスタンプなど) を取得します。

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
  ## 予測とスコアを取得する
</div>

評価 run の行ごとの結果を取得するには、`eval_results/query` エンドポイントを使用します。各行には、解決されたデータセットの入力、モデルの出力、個々の Scorer の結果が含まれます。行ごとの完全な詳細を取得するには、`include_rows`、`include_raw_data_rows`、`resolve_row_refs` を設定します。

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
  ## 集計されたスコアを取得する
</div>

同じ `eval_results/query` エンドポイントでは、行ごとのデータではなく、集計した Scorer の統計情報を返すこともできます。`include_summary` を設定すると、バイナリ Scorer の合格率や連続値 Scorer の平均など、サマリーレベルのメトリクスを取得できます。

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
  ## 単一の予測を取得する
</div>

個々の予測について、inputs、出力、モデル参照を含むすべての詳細情報を取得します。

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
  ## 行ダイジェストの使い方
</div>

`eval_results/query` エンドポイント の各結果行には `row_digest` が含まれます。これは、位置ではなく内容に基づいて、評価データセット内の特定の入力を一意に識別するコンテンツハッシュです。行ダイジェストは、次のような用途で役立ちます。

* **評価間の比較**: 同じデータセットに対して 2 つの異なるモデルを実行すると、同じダイジェストを持つ行は同じ入力を表します。`row_digest` で結合すれば、異なるモデルがまったく同じタスクでどのような性能を示したかを比較できます。
* **重複排除**: 同じタスクが複数の評価スイートに含まれている場合、ダイジェストを使ってそれを識別できます。
* **再現性**: ダイジェストは内容に基づいて決まるため、誰かがデータセットの行を変更すると (指示テキスト、ルーブリック、その他のフィールドを変更すると) 、新しいダイジェストになります。2 つの評価 run で同一の入力が使用されたのか、それとも少し異なるバージョンが使用されたのかを確認できます。