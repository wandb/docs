---
title: スイープのトラブルシューティング
description: よくある W&B Sweep の問題をトラブルシュートする
menu:
  default:
    identifier: troubleshoot-sweeps
    parent: sweeps
---

よくあるエラーメッセージと、その対処方法をまとめています。

### `CommError, Run does not exist` および `ERROR Error uploading`

これら 2 つのエラーメッセージが返された場合、W&B Run の ID が設定されている可能性があります。例えば、Jupyter Notebook や Python スクリプトのどこかに、以下のようなコードスニペットが書かれている場合です。

```python
wandb.init(id="some-string")
```

W&B Sweeps で Run ID を自分で設定することはできません。なぜなら、W&B Sweeps で作成される Run には、W&B 側で自動的にランダムかつユニークな ID が付与されるためです。

W&B Run の ID は、1 つの Project の中でユニークである必要があります。

テーブルやグラフ上でわかりやすい名前を表示したい場合は、W&B の初期化時に `name` パラメータを使って名前を付けることをおすすめします。例えば以下のようにします。

```python
wandb.init(name="a helpful readable run name")
```

### `Cuda out of memory`

このエラーメッセージが表示された場合は、コードをプロセスベースの実行形式へリファクタリングしてください。具体的には、コードを Python スクリプトに書き換え、W&B Sweep agent を CLI から呼び出して実行する方法に切り替えます（W&B Python SDK からの直接呼び出しは使いません）。

例えば、コードを書き換えた Python スクリプトのファイル名が `train.py` だとします。YAML の Sweep 設定ファイル（この例では `config.yaml`）内の `program` に、トレーニングスクリプト（`train.py`）の名前を書きます。

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

次に、`train.py` の Python スクリプトには以下を追加してください。

```python
if _name_ == "_main_":
    train()
```

その後、CLI で W&B Sweep を初期化します。

```shell
wandb sweep config.yaml
```

返ってきた W&B Sweep ID を控えておきます。次に、Sweep のジョブを開始するには CLI から [`wandb agent`]({{< relref "/ref/cli/wandb-agent.md" >}}) を実行してください（Python SDK の [`wandb.agent`]({{< relref "/ref/python/sdk/functions/agent.md" >}}) ではなく、CLI で実行します）。下記コードスニペット内の `sweep_ID` を、先ほど控えた Sweep ID に置き換えてください。

```shell
wandb agent sweep_ID
```

### `anaconda 400 error`

このエラーは、最適化対象の metric をログしていないときに発生するのが一般的です。

```shell
wandb: ERROR Error while calling W&B API: anaconda 400 error: 
{"code": 400, "message": "TypeError: bad operand type for unary -: 'NoneType'"}
```

YAML ファイルやネストされた辞書内で「metric」という key で最適化する metric を指定しています。必ずこの metric を `wandb.log` で記録（ログ）してください。また、Python スクリプトや Jupyter Notebook でその sweep で最適化する、と指定した metric 名と**全く同じ**名前でログを出す必要があります。設定ファイルの詳細については [Define sweep configuration]({{< relref "/guides/models/sweeps/define-sweep-configuration/" >}}) をご覧ください。