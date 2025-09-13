---
title: Sweeps トラブルシューティング
description: よくある W&B Sweep の問題のトラブルシューティング。
menu:
  default:
    identifier: ja-guides-models-sweeps-troubleshoot-sweeps
    parent: sweeps
---

提案されたガイダンスに従って、よくあるエラーメッセージをトラブルシュートしてください。

### `CommError, Run does not exist` と `ERROR Error uploading`

これら 2 つのエラーメッセージが同時に返される場合、W&B の Run ID を手動で設定していることが原因かもしれません。たとえば、Jupyter Notebook や Python スクリプトのどこかに次のようなコードスニペットを書いている可能性があります:

```python
wandb.init(id="some-string")
```

W&B Sweeps によって作成される Runs には、W&B がランダムで一意の ID を自動生成するため、Run ID を手動で設定することはできません。

W&B の Run ID は同一 Project 内で一意である必要があります。

テーブルやグラフに表示されるカスタム名を付けたい場合は、W&B を初期化する際に name パラメータへ名前を渡すことをおすすめします。例:

```python
wandb.init(name="a helpful readable run name")
```

### `Cuda out of memory`

このエラーメッセージが表示される場合は、プロセスベースの実行に切り替えるように code をリファクタリングしてください。より具体的には、code を Python スクリプトに書き換え、W&B Python SDK ではなく CLI から W&B Sweep Agent を呼び出してください。

例として、code を `train.py` という Python スクリプトに書き換えたとします。YAML の sweep configuration ファイル（この例では `config.yaml`）に、トレーニングスクリプト（`train.py`）の名前を追加します:

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

次に、`train.py` の Python スクリプトに以下を追加します:

```python
if _name_ == "_main_":
    train()
```

CLI に移動し、wandb sweep を使って W&B Sweep を初期化します:

```shell
wandb sweep config.yaml
```

返ってきた W&B の Sweep ID を控えておきます。続いて、Python SDK（[`wandb.agent`]({{< relref path="/ref/python/sdk/functions/agent.md" lang="ja" >}})）ではなく CLI から、[`wandb agent`]({{< relref path="/ref/cli/wandb-agent.md" lang="ja" >}}) を使って Sweep ジョブを開始します。以下のコードスニペット内の `sweep_ID` を、先ほど控えた Sweep ID に置き換えてください:

```shell
wandb agent sweep_ID
```

### `anaconda 400 error`

最適化対象のメトリクスをログしていない場合に、次のエラーが発生することがあります:

```shell
wandb: ERROR Error while calling W&B API: anaconda 400 error: 
{"code": 400, "message": "TypeError: bad operand type for unary -: 'NoneType'"}
```

最適化対象を指定するために、YAML ファイルまたはネストされた 辞書 に "metric" というキーを定義しているはずです。このメトリクスを `wandb.log` で必ずログしてください。さらに、Python スクリプトや Jupyter Notebook の中で、sweep を最適化するように定義したメトリクス名と _厳密に_ 同一の名前を使用していることを確認してください。設定ファイルの詳細は、[sweep configuration を定義する]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ja" >}}) を参照してください。