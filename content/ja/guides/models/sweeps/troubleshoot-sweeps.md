---
title: Sweeps troubleshooting
description: W&B Sweep でよくある問題をトラブルシューティングします。
menu:
  default:
    identifier: ja-guides-models-sweeps-troubleshoot-sweeps
    parent: sweeps
---

提案されたガイダンスに従って、一般的なエラーメッセージのトラブルシューティングを行います。

### `CommError, Run does not exist` および `ERROR Error uploading`

これらの 2 つのエラーメッセージが両方とも返された場合、W&B の Run ID が定義されている可能性があります。例として、Jupyter Notebook または Python スクリプトのどこかに、次のようなコードスニペットが定義されている可能性があります。

```python
wandb.init(id="some-string")
```

W&B は W&B Sweeps によって作成された Runs に対して、ランダムでユニークな ID を自動的に生成するため、W&B Sweeps の Run ID を設定することはできません。

W&B の Run ID は、1 つの project 内でユニークである必要があります。

W&B を初期化する際に、テーブルやグラフに表示されるカスタム名を付けたい場合は、name パラメータに名前を渡すことをお勧めします。例：

```python
wandb.init(name="a helpful readable run name")
```

### `Cuda out of memory`

このエラーメッセージが表示された場合は、コードをリファクタリングして、プロセスベースの実行を使用するようにします。より具体的には、コードを Python スクリプトに書き換えます。さらに、W&B Python SDK ではなく、CLI から W&B Sweep Agent を呼び出します。

例として、コードを `train.py` という Python スクリプトに書き換えたとします。トレーニングスクリプトの名前 (`train.py`) を YAML Sweep configuration ファイル (`config.yaml`（この例）) に追加します。

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

次に、次のコードを `train.py` Python スクリプトに追加します。

```python
if _name_ == "_main_":
    train()
```

CLI に移動し、`wandb sweep` で W&B Sweep を初期化します。

```shell
wandb sweep config.yaml
```

返された W&B Sweep ID をメモします。次に、Python SDK ([`wandb.agent`]({{< relref path="/ref/python/agent.md" lang="ja" >}})) ではなく、CLI で [`wandb agent`]({{< relref path="/ref/cli/wandb-agent.md" lang="ja" >}}) を使用して Sweep ジョブを開始します。次のコードスニペットの `sweep_ID` を、前の手順で返された Sweep ID に置き換えます。

```shell
wandb agent sweep_ID
```

### `anaconda 400 error`

次のエラーは通常、最適化しているメトリックをログに記録しない場合に発生します。

```shell
wandb: ERROR Error while calling W&B API: anaconda 400 error: 
{"code": 400, "message": "TypeError: bad operand type for unary -: 'NoneType'"}
```

YAML ファイルまたはネストされた辞書の中で、最適化する「metric」という名前のキーを指定します。このメトリックを必ずログに記録 (`wandb.log`) してください。さらに、Python スクリプトまたは Jupyter Notebook 内で Sweep を最適化するために定義した _正確な_ メトリック名を使用してください。設定ファイルの詳細については、[Sweep configuration の定義]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ja" >}}) を参照してください。
