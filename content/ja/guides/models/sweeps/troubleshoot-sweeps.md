---
title: Sweeps troubleshooting
description: W&B Sweep でよくある問題のトラブルシューティング。
menu:
  default:
    identifier: ja-guides-models-sweeps-troubleshoot-sweeps
    parent: sweeps
---

提案されたガイダンスに従って、一般的なエラーメッセージのトラブルシューティングを行います。

### `CommError, Run does not exist` および `ERROR Error uploading`

これらの 2 つのエラーメッセージが両方とも返された場合、W&B の Run ID が定義されている可能性があります。例として、Jupyter Notebooks または Python スクリプトのどこかに、次のようなコードスニペットが定義されている場合があります。

```python
wandb.init(id="some-string")
```

W&B は W&B Sweeps によって作成された Runs に対して、ランダムで一意な ID を自動的に生成するため、W&B Sweeps の Run ID を設定することはできません。

W&B の Run ID は、project 内で一意である必要があります。

テーブル やグラフに表示されるカスタム名を付けたい場合は、W&B を初期化する際に、name パラメータに名前を渡すことをお勧めします。例：

```python
wandb.init(name="a helpful readable run name")
```

### `Cuda out of memory`

このエラーメッセージが表示された場合は、プロセスベースの実行を使用するようにコードをリファクタリングしてください。より具体的には、コードを Python スクリプトに書き換えます。さらに、W&B Python SDK ではなく、CLI から W&B Sweep Agent を呼び出します。

例として、コードを `train.py` という Python スクリプトに書き換えたとします。トレーニング スクリプト (`train.py`) の名前を YAML Sweep 設定ファイル (`config.yaml` （この例の場合）) に追加します。

```yaml
program: train.py
method: bayes
metric:
  name: validation_loss
  goal: maximize
parameters:
  learning_rate:
    min: 0.0001
    max: 0.1
  optimizer:
    values: ["adam", "sgd"]
```

次に、次の内容を `train.py` Python スクリプトに追加します。

```python
if _name_ == "_main_":
    train()
```

CLI に移動し、wandb sweep で W&B Sweep を初期化します。

```shell
wandb sweep config.yaml
```

返された W&B Sweep ID をメモしておきます。次に、Python SDK ([`wandb.agent`]({{< relref path="/ref/python/agent.md" lang="ja" >}})) ではなく、CLI で [`wandb agent`]({{< relref path="/ref/cli/wandb-agent.md" lang="ja" >}}) を使用して Sweep ジョブを開始します。以下のコードスニペットの `sweep_ID` を、前の手順で返された Sweep ID に置き換えます。

```shell
wandb agent sweep_ID
```

### `anaconda 400 error`

通常、次のエラーは、最適化しているメトリックをログに記録していない場合に発生します。

```shell
wandb: ERROR Error while calling W&B API: anaconda 400 error: 
{"code": 400, "message": "TypeError: bad operand type for unary -: 'NoneType'"}
```

YAML ファイルまたはネストされた 辞書 内で、最適化する "metric" という名前の キー を指定します。このメトリックを必ずログ (`wandb.log`) に記録してください。さらに、Python スクリプトまたは Jupyter Notebook 内で sweep を最適化するために定義した _exact_ なメトリック名を使用していることを確認してください。設定ファイルの詳細については、[Sweep 設定の定義]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ja" >}})を参照してください。
