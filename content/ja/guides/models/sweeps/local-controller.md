---
title: Manage algorithms locally
description: W&B の クラウド でホストされているサービスを使用する代わりに、ローカルでアルゴリズムを検索して停止します。
menu:
  default:
    identifier: ja-guides-models-sweeps-local-controller
    parent: sweeps
---

ハイパーパラメータコントローラは、デフォルトで Weights & Biases によってクラウドサービスとしてホストされています。W&B エージェント はコントローラと通信して、トレーニングに使用する次のパラメータセットを決定します。コントローラは、どの run を停止できるかを判断するために、早期停止アルゴリズムを実行する役割も担います。

ローカルコントローラの機能により、ユーザーはローカルで検索と停止アルゴリズムを開始できます。ローカルコントローラは、ユーザーがコードを検査および分析して問題をデバッグしたり、クラウドサービスに組み込むことができる新機能を開発したりする機能を提供します。

{{% alert color="secondary" %}}
この機能は、 Sweeps ツール用の新しいアルゴリズムのより迅速な開発とデバッグをサポートするために提供されています。実際のハイパーパラメータ最適化のワークロードを目的としたものではありません。
{{% /alert %}}

開始する前に、W&B SDK（`wandb`）をインストールする必要があります。次のコードスニペットをコマンドラインに入力します。

```
pip install wandb sweeps
```

以下の例では、設定ファイルとトレーニングループが Python スクリプトまたは Jupyter Notebook で定義されていることを前提としています。設定ファイルを定義する方法の詳細については、[sweep configuration の定義]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ja" >}})を参照してください。

### コマンドラインからローカルコントローラを実行する

W&B がクラウドサービスとしてホストするハイパーパラメータコントローラを使用する場合と同様に、 sweep を初期化します。コントローラフラグ（`controller`）を指定して、W&B sweep ジョブにローカルコントローラを使用することを示します。

```bash
wandb sweep --controller config.yaml
```

または、 sweep の初期化と、ローカルコントローラを使用することの指定を2つのステップに分けることもできます。

ステップを分離するには、まず次のキーと 値 を sweep の YAML 設定ファイルに追加します。

```yaml
controller:
  type: local
```

次に、 sweep を初期化します。

```bash
wandb sweep config.yaml
```

sweep を初期化したら、[`wandb controller`]({{< relref path="/ref/python/controller.md" lang="ja" >}}) でコントローラを開始します。

```bash
# wandb sweep コマンドは sweep_id を出力します
wandb controller {user}/{entity}/{sweep_id}
```

ローカルコントローラを使用することを指定したら、1つ以上の Sweep agent を起動して sweep を実行します。通常と同様に、W&B Sweep を開始します。詳細については、[Sweep agent の開始]({{< relref path="/guides/models/sweeps/start-sweep-agents.md" lang="ja" >}})を参照してください。

```bash
wandb sweep sweep_ID
```

### W&B Python SDK でローカルコントローラを実行する

次のコードスニペットは、W&B Python SDK でローカルコントローラを指定および使用する方法を示しています。

Python SDK でコントローラを使用する最も簡単な方法は、 sweep ID を [`wandb.controller`]({{< relref path="/ref/python/controller.md" lang="ja" >}}) メソッドに渡すことです。次に、戻り オブジェクト の `run` メソッドを使用して、 sweep ジョブを開始します。

```python
sweep = wandb.controller(sweep_id)
sweep.run()
```

コントローラ ループ をより詳細に制御したい場合:

```python
import wandb

sweep = wandb.controller(sweep_id)
while not sweep.done():
    sweep.print_status()
    sweep.step()
    time.sleep(5)
```

または、提供されるパラメータをさらに制御する場合:

```python
import wandb

sweep = wandb.controller(sweep_id)
while not sweep.done():
    params = sweep.search()
    sweep.schedule(params)
    sweep.print_status()
```

コードですべての sweep を指定する場合は、次のようになります。

```python
import wandb

sweep = wandb.controller()
sweep.configure_search("grid")
sweep.configure_program("train-dummy.py")
sweep.configure_controller(type="local")
sweep.configure_parameter("param1", value=3)
sweep.create()
sweep.run()
```
