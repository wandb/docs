---
description: W&B クラウドホストサービスを使用せずに、ローカルで検索および停止アルゴリズムを実行します。
displayed_sidebar: default
---


# Search and stop algorithms locally

<head>
  <title>Search and stop algorithms locally with W&B agents</title>
</head>

ハイパーパラメーターコントローラはデフォルトでWeights & Biasedによりクラウドサービスとしてホストされています。W&Bエージェントはコントローラと通信して、トレーニングに使用する次の一連のパラメーターを決定します。コントローラは、早期停止アルゴリズムの実行についても責任を持ち、どのrunを停止するかを決定します。

ローカルコントローラ機能を使用すると、ユーザーはローカルで検索および停止アルゴリズムを開始できます。ローカルコントローラにより、ユーザーはコードを検査および操作して、問題をデバッグしたり、新しい機能を開発してクラウドサービスに統合することができます。

:::caution
この機能は、Sweepsツールの新しいアルゴリズムの開発とデバッグを迅速に行うために提供されています。実際のハイパーパラメーター最適化のワークロードには意図されていません。
:::

始める前に、W&B SDK（`wandb`）をインストールする必要があります。以下のコードスニペットをコマンドラインに入力してください:

```
pip install wandb sweeps 
```

以下の例では、すでに設定ファイルとPythonスクリプトまたはJupyter Notebookで定義されたトレーニングループがあることを前提としています。設定ファイルの定義方法について詳しくは、[Define sweep configuration](./define-sweep-configuration.md)を参照してください。

### コマンドラインからローカルコントローラを実行

通常、W&Bがクラウドサービスとしてホストするハイパーパラメーターコントローラを使用する場合と同様に、sweepを初期化します。コントローラフラグ（`controller`）を指定して、W&B sweepジョブにローカルコントローラを使用することを示します。

```bash
wandb sweep --controller config.yaml
```

または、sweepの初期化とローカルコントローラを使用することの指定を2つのステップに分けることもできます。

ステップを分けるには、まず次のキー/値をsweepのYAML設定ファイルに追加します：

```yaml
controller:
  type: local
```

次に、sweepを初期化します：

```bash
wandb sweep config.yaml
```

sweepを初期化したら、[`wandb controller`](../../ref/python/controller.md)を使用してコントローラを起動します：

```bash
# wandb sweepコマンドはsweep_idを表示します
wandb controller {user}/{entity}/{sweep_id}
```

ローカルコントローラを使用することを指定したら、1つ以上のSweep agentを開始してsweepを実行します。通常の方法でW&B Sweepを開始します。[Start sweep agents](../../guides/sweeps/start-sweep-agents.md)をご覧ください。

```bash
wandb sweep sweep_ID
```

### W&B Python SDKでローカルコントローラを実行

次のコードスニペットは、W&B Python SDKを使用してローカルコントローラを指定し、利用する方法を示しています。

Python SDKでコントローラを使用する最も簡単な方法は、[`wandb.controller`](../../ref/python/controller.md)メソッドにsweep IDを渡すことです。次に、返されたオブジェクトの`run`メソッドを使用してsweepジョブを開始します：

```python
sweep = wandb.controller(sweep_id)
sweep.run()
```

コントローラーループをさらに制御したい場合：

```python
import wandb

sweep = wandb.controller(sweep_id)
while not sweep.done():
    sweep.print_status()
    sweep.step()
    time.sleep(5)
```

提供されるパラメーターをさらに制御したい場合：

```python
import wandb

sweep = wandb.controller(sweep_id)
while not sweep.done():
    params = sweep.search()
    sweep.schedule(params)
    sweep.print_status()
```

全てのsweepをコードで指定したい場合、以下のようにできます：

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