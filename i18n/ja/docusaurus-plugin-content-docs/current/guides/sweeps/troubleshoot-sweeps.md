---
description: Troubleshoot common W&B Sweep issues.
displayed_sidebar: ja
---

# スイープのトラブルシューティング

<head>
  <title>W&Bスイープのトラブルシューティング</title>
</head>

一般的なエラーメッセージに対処するための指針を提案します。

### `CommError, Run does not exist` および `ERROR Error uploading`

これら2つのエラーメッセージが両方とも返される場合、W&B Run IDが定義されている可能性があります。例として、JupyterノートブックやPythonスクリプトのどこかに以下のようなコードスニペットが定義されているかもしれません。

```python
wandb.init(id="some-string")
```

W&Bスイープでは、Run IDを設定することはできません。なぜなら、Weights & Biasesは、W&Bスイープによって作成されたRunに対してランダムで一意のIDを自動的に生成するからです。

W&B Run IDは、プロジェクト内で一意である必要があります。

名前パラメータにカスタム名を設定して、テーブルやグラフに表示される名前を指定する場合は、Weights & Biasesを初期化する際に名前を渡すことをお勧めします。例えば、

```python
wandb.init(name="a helpful readable run name")
```
### `Cuda out of memory`

このエラーメッセージが表示された場合、プロセスベースの実行を使用するようにコードをリファクタリングしてください。具体的には、コードをPythonスクリプトに書き換え、W&B Python SDKの代わりにCLIからW&B スイープエージェントを呼び出します。

例として、コードを`train.py`という名前のPythonスクリプトに書き換えたとします。トレーニングスクリプト（`train.py`）の名前を、YAMLスイープ構成ファイル（この例では`config.yaml`）に追加します。

```
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

次に、`train.py` Pythonスクリプトに以下を追加します。

```python
if _name_ == "_main_":
    train()
```

CLIに移動し、wandb sweepを使用してW&Bスイープを初期化します。

```
wandb sweep config.yaml
```
W&B スイープ ID をメモしておいてください。次に、Python SDK の [`wandb.agent`](https://docs.wandb.ai/ref/python/agent) ではなく、CLI の [`wandb agent`](https://docs.wandb.ai/ref/cli/wandb-agent) を使ってスイープジョブを開始します。以下のコードスニペットで `sweep_ID` を前のステップで返されたスイープ ID に置き換えてください。

```
wandb agent sweep_ID
```

### `anaconda 400 error`

以下のエラーは、最適化するメトリックをログに記録しない場合に通常発生します。

```python
wandb: ERROR Error while calling W&B API: anaconda 400 error: 
{"code": 400, "message": "TypeError: bad operand type for unary -: 'NoneType'"}
```

YAML ファイルやネストされたディクショナリ内で、最適化する "metric" という名前のキーを指定します。このメトリックをログ（`wandb.log`）に記録することを確認してください。さらに、Python スクリプトや Jupyter ノートブック内でスイープの最適化に定義した _正確な_ メトリック名を使用してください。設定ファイルの詳細については、[スイープ設定の定義](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration)を参照してください。