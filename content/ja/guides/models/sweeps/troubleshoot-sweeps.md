---
title: スイープのトラブルシューティング
description: 一般的な W&B Sweep の問題をトラブルシュートする。
menu:
  default:
    identifier: ja-guides-models-sweeps-troubleshoot-sweeps
    parent: sweeps
---

一般的なエラーメッセージのトラブルシューティングには、提案されたガイダンスを参照してください。

### `CommError, Run does not exist` および `ERROR Error uploading`

これら2つのエラーメッセージが返される場合、W&B Run ID が定義されている可能性があります。例えば、Jupyter Notebooks や Python スクリプトのどこかに類似のコードスニペットが定義されているかもしれません。

```python
wandb.init(id="some-string")
```

W&B Sweeps では Run ID を設定することはできません。なぜなら、W&B が作成する Runs には、W&B が自動的にランダムで一意の ID を生成するからです。

W&B Run IDs は、プロジェクト内で一意である必要があります。

テーブルやグラフに表示するカスタム名を設定したい場合は、W&B を初期化するときに name パラメータに名前を渡すことをお勧めします。例えば：

```python
wandb.init(name="a helpful readable run name")
```

### `Cuda out of memory`

このエラーメッセージが表示される場合は、プロセスベースの実行を使用するようにコードをリファクタリングしてください。具体的には、コードを Python スクリプトに書き換えてください。また、W&B Python SDK ではなく CLI から W&B Sweep Agent を呼び出してください。

例として、コードを `train.py` という名の Python スクリプトに書き直すとします。その際、トレーニングスクリプト (`train.py`) の名前を YAML Sweep 設定ファイル (`config.yaml` の例) に追加します。

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

次に、Python スクリプト `train.py` に以下を追加します。

```python
if _name_ == "_main_":
    train()
```

CLI に移動して、wandb sweep を使用して W&B Sweep を初期化します。

```shell
wandb sweep config.yaml
```

返された W&B Sweep ID をメモします。次に、 Sweep のジョブを CLI で [`wandb agent`]({{< relref path="/ref/cli/wandb-agent.md" lang="ja" >}}) を使用して開始します。Python SDK ([`wandb.agent`]({{< relref path="/ref/python/agent.md" lang="ja" >}})) ではなく、CLI を使用します。次のコードスニペットでは、`sweep_ID` を前のステップで返された Sweep ID に置き換えてください。

```shell
wandb agent sweep_ID
```

### `anaconda 400 error`

このエラーは通常、最適化しているメトリックをログしていない場合に発生します。

```shell
wandb: ERROR Error while calling W&B API: anaconda 400 error: 
{"code": 400, "message": "TypeError: bad operand type for unary -: 'NoneType'"}
```

YAML ファイルやネストされた辞書内で、最適化する「metric」 というキーを指定します。このメトリックをログ (`wandb.log`) することを確認してください。また、Python スクリプトや Jupyter Notebook 内で最適化するように定義した _exact_ なメトリック名を必ず使用してください。設定ファイルについての詳細は、[Define sweep configuration]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ja" >}}) を参照してください。