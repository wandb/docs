---
title: スイープのトラブルシューティング
description: 一般的な W&B Sweep の問題をトラブルシュートする
menu:
  default:
    identifier: ja-guides-models-sweeps-troubleshoot-sweeps
    parent: sweeps
---

よくあるエラーメッセージとその対処法についてご案内します。

### `CommError, Run does not exist` および `ERROR Error uploading`

この2つのエラーメッセージが同時に表示される場合、W&B の Run ID を設定している可能性があります。例えば、Jupyter Notebook や Python スクリプトのどこかに、次のようなコードスニペットが記載されていることがあります。

```python
wandb.init(id="some-string")
```

W&B Sweeps で Run ID を自分で設定することはできません。Sweeps で作成される Run には、W&B が自動的にランダムでユニークな ID を割り当てます。

また、W&B の Run ID は 1 つのプロジェクト内でユニークである必要があります。

テーブルやグラフ上で任意の名前を表示したい場合は、W&B 初期化時に `name` パラメータで名前を渡すことをおすすめします。例:

```python
wandb.init(name="a helpful readable run name")
```

### `Cuda out of memory`

このエラーメッセージが表示された場合は、コードをプロセスベースで実行する形にリファクタリングしてください。より具体的には、Python スクリプトに書き換えてください。また、W&B Sweep Agent は W&B Python SDK から呼び出すのではなく、CLI から実行してください。

たとえば、`train.py` という Python スクリプトに書き換えるとします。YAML のスイープ設定ファイル（この例では `config.yaml`）には、`train.py` を記載します。

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

続いて、`train.py` の Python スクリプトには次のように記述します。

```python
if _name_ == "_main_":
    train()
```

次に CLI で移動し、W&B Sweep を wandb sweep で初期化します。

```shell
wandb sweep config.yaml
```

返ってきた W&B Sweep ID をメモしてください。次に CLI から、Python SDK ではなく [`wandb agent`]({{< relref path="/ref/cli/wandb-agent.md" lang="ja" >}}) で Sweep ジョブを開始します（Python SDK の [`wandb.agent`]({{< relref path="/ref/python/sdk/functions/agent.md" lang="ja" >}}) ではありません）。下記のコードスニペットの `sweep_ID` 部分を、先ほど返された Sweep ID に置き換えてください。

```shell
wandb agent sweep_ID
```

### `anaconda 400 error`

このエラーは、最適化しようとしているメトリクスをログしていない場合によく発生します。

```shell
wandb: ERROR Error while calling W&B API: anaconda 400 error: 
{"code": 400, "message": "TypeError: bad operand type for unary -: 'NoneType'"}
```

YAML ファイルやネストされた辞書内で、最適化する "metric" というキーを指定している場合、そのメトリクスを必ず `wandb.log` でログしてください。また、Python スクリプトや Jupyter Notebook に記載する際は、スイープで最適化対象として定義したメトリック名と _まったく同じ_ 名前を使ってください。設定ファイルの詳細については、[スイープ設定の定義方法]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ja" >}}) を参照してください。