---
title: Sweeps troubleshooting
description: W&B Sweep の一般的な問題をトラブルシュートする方法。
menu:
  default:
    identifier: ja-guides-models-sweeps-troubleshoot-sweeps
    parent: sweeps
---

一般的なエラーメッセージのトラブルシューティングは、提案されたガイダンスに従って行ってください。

### `CommError, Run does not exist` と `ERROR Error uploading`

これら2つのエラーメッセージが返される場合、W&B Run ID が既に定義されている可能性があります。たとえば、次のようなコードスニペットが Jupyter ノートブックや Python スクリプトのどこかに定義されていることがあります。

```python
wandb.init(id="some-string")
```

W&B Sweeps の場合、Run ID を設定することはできません。W&B は、W&B Sweeps によって作成される Runs に対してランダムでユニークな ID を自動生成します。

W&B Run ID は、プロジェクト内でユニークである必要があります。

テーブルやグラフに表示されるカスタム名を設定したい場合は、W&B を初期化するときに name パラメータに名前を渡すことをお勧めします。例えば：

```python
wandb.init(name="a helpful readable run name")
```

### `Cuda out of memory`

このエラーメッセージが表示された場合、コードをプロセスベースの実行にリファクタリングしてください。具体的には、コードを Python スクリプトに書き直してください。加えて、W&B Python SDK の代わりに CLI から W&B Sweep Agent を呼び出してください。

たとえば、コードを `train.py` という Python スクリプトに書き直しましょう。トレーニングスクリプトの名前 (`train.py`) を YAML スイープ設定ファイル (`config.yaml` の例) に追加してください。

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

次に、`train.py` の Python スクリプトに以下を追加してください。

```python
if _name_ == "_main_":
    train()
```

CLI に移動して、wandb sweep で W&B Sweep を初期化してください。

```shell
wandb sweep config.yaml
```

返された W&B Sweep ID をメモしてください。次に、CLI を使って Python SDK の代わりに [`wandb agent`]({{< relref path="/ref/cli/wandb-agent.md" lang="ja" >}}) で Sweep ジョブを開始してください ([`wandb.agent`]({{< relref path="/ref/python/agent.md" lang="ja" >}}))。以下のコードスニペットの `sweep_ID` を前のステップで返された Sweep ID に置き換えてください。

```shell
wandb agent sweep_ID
```

### `anaconda 400 error`

次のエラーは通常、最適化するメトリックをログしない場合に発生します。

```shell
wandb: ERROR Error while calling W&B API: anaconda 400 error: 
{"code": 400, "message": "TypeError: bad operand type for unary -: 'NoneType'"}
```

YAML ファイルまたはネストされた辞書内で、"metric" という名前のキーを最適化するよう指定します。このメトリックをログ（`wandb.log`）することを確認してください。さらに、Python スクリプトや Jupyter ノートブック内でスイープを最適化するように定義した _正確な_ メトリック名を使用していることを確認してください。設定ファイルの詳細については、[Define sweep configuration]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ja" >}}) を参照してください。