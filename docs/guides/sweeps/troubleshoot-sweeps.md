---
description: 一般的な W&B Sweep の問題のトラブルシューティング。
displayed_sidebar: default
---


# Troubleshoot Sweeps

<head>
  <title>Troubleshoot W&B Sweeps</title>
</head>

Troubleshoot common error messages with the guidance suggested.

### `CommError, Run does not exist` and `ERROR Error uploading`

この2つのエラーメッセージが両方とも返される場合、W&B Run ID が定義されている可能性があります。例えば、Jupyter ノートブックや Python スクリプトのどこかに次のようなコードスニペットが定義されているかもしれません:

```python
wandb.init(id="some-string")
```

W&B Sweeps では Run ID を設定することはできません。W&B が自動的にランダムでユニークな ID を生成するからです。

W&B Run ID はプロジェクト内でユニークである必要があります。

カスタムの名前を設定したい場合は、W&B を初期化する際に `name` パラメータに名前を渡すことをお勧めします。例えば、テーブルやグラフに表示される読みやすい名前を設定できます:

```python
wandb.init(name="a helpful readable run name")
```

### `Cuda out of memory`

このエラーメッセージが表示された場合は、コードをプロセスベースの実行にリファクタリングしてください。具体的には、コードを Python スクリプトに書き直し、W&B Sweep Agent を W&B Python SDK ではなく CLI から呼び出します。

例えば、コードを書き直して `train.py` という名前の Python スクリプトにすると仮定します。トレーニングスクリプト (`train.py`) の名前を YAML Sweep configuration ファイル (`config.yaml`) に追加します:

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

次に、`train.py` Python スクリプトに次のコードを追加します:

```python
if _name_ == "_main_":
    train()
```

CLI に移動して、wandb sweep を使用して W&B Sweep を初期化します:

```shell
wandb sweep config.yaml
```

返された W&B Sweep ID をメモしておきます。次に、Sweep ジョブを CLI から [`wandb agent`](../../ref/cli/wandb-agent.md) を使用して開始します。Python SDK ([`wandb.agent`](../../ref/python/agent.md)) ではなく CLI を使用します。以下のコードスニペットの `sweep_ID` を前のステップで返された Sweep ID に置き換えます:

```shell
wandb agent sweep_ID
```

### `anaconda 400 error`

このエラーは、最適化するメトリックをログしていないときに通常発生します:

```shell
wandb: ERROR Error while calling W&B API: anaconda 400 error: 
{"code": 400, "message": "TypeError: bad operand type for unary -: 'NoneType'"}
```

YAML ファイルまたはネストされた辞書内で、最適化するキー名 "metric" を指定します。このメトリックを `wandb.log` でログすることを確認してください。さらに、Python スクリプトや Jupyter ノートブックで、sweep の最適化に使用する _正確な_ メトリック名を使用してください。設定ファイルの詳細については、[Define sweep configuration](./define-sweep-configuration.md) を参照してください。