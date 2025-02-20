---
title: Manage algorithms locally
description: W&B クラウドホストサービスを使用するのではなく、ローカルで検索と停止アルゴリズムを使用します。
menu:
  default:
    identifier: ja-guides-models-sweeps-local-controller
    parent: sweeps
---

ハイパーパラメーターコントローラは、デフォルトで Weights & Biases によってクラウドサービスとしてホストされています。W&B エージェントは、トレーニングに使用する次のパラメーターセットを決定するためにコントローラと通信します。コントローラは、早期停止アルゴリズムを実行して、どの run を停止できるかを決定する責任も負っています。

ローカルコントローラ機能は、ユーザーがローカルで探索とアルゴリズムの停止を開始できるようにします。ローカルコントローラを使用すると、コードを検査してインスツルメント化することでデバッグを行うと同時に、クラウドサービスに組み込むことができる新機能を開発することができます。

{{% alert color="secondary" %}}
この機能は、Sweeps ツール用の新しいアルゴリズムのより迅速な開発とデバッグをサポートするために提供されています。実際のハイパーパラメータ最適化の作業には意図されていません。
{{% /alert %}}

始める前に、W&B SDK（`wandb`）をインストールする必要があります。以下のコードスニペットをコマンドラインに入力してください:

```
pip install wandb sweeps 
```

以下の例は、Python スクリプトまたは Jupyter ノートブック内に設定ファイルとトレーニングループがすでに定義されていることを前提としています。設定ファイルの定義方法についての詳細は、[Define sweep configuration]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ja" >}}) を参照してください。

### コマンドラインからローカルコントローラを実行する

通常 W&B がクラウドサービスとしてホストするハイパーパラメータコントローラを使用する場合と同様に、sweep を初期化します。 ローカルコントローラを W&B のスイープジョブに使用したいことを示すために、コントローラフラグ（`controller`）を指定してください:

```bash
wandb sweep --controller config.yaml
```

または、sweep の初期化とローカルコントローラを使用したいという指定を 2 つのステップに分けることができます。

ステップを分けるには、まず次のキーと値を sweep の YAML 設定ファイルに追加します:

```yaml
controller:
  type: local
```

次に、スイープを初期化します:

```bash
wandb sweep config.yaml
```

スイープを初期化した後、[`wandb controller`]({{< relref path="/ref/python/controller.md" lang="ja" >}}) を使ってコントローラを開始します:

```bash
# wandb sweep コマンドは sweep_id を出力します
wandb controller {user}/{entity}/{sweep_id}
```

ローカルコントローラを使用したいことを指定したら、1 つ以上の Sweep エージェントを開始して sweep を実行します。通常と同様に W&B Sweep を開始してください。詳細については [Start sweep agents]({{< relref path="/guides/models/sweeps/start-sweep-agents.md" lang="ja" >}}) を参照してください。

```bash
wandb sweep sweep_ID
```

### W&B Python SDK でローカルコントローラを実行する

以下のコードスニペットは、W&B Python SDK でローカルコントローラを指定して使用する方法を示しています。

Python SDK でコントローラを使用する最も簡単な方法は、sweep ID を [`wandb.controller`]({{< relref path="/ref/python/controller.md" lang="ja" >}}) メソッドに渡すことです。次に、返されたオブジェクトの `run` メソッドを使用して sweep ジョブを開始します:

```python
sweep = wandb.controller(sweep_id)
sweep.run()
```

コントローラループをもっと制御したい場合:

```python
import wandb

sweep = wandb.controller(sweep_id)
while not sweep.done():
    sweep.print_status()
    sweep.step()
    time.sleep(5)
```

さらにパラメータの提供を詳細に制御したい場合:

```python
import wandb

sweep = wandb.controller(sweep_id)
while not sweep.done():
    params = sweep.search()
    sweep.schedule(params)
    sweep.print_status()
```

スイープをすべてコードで指定したい場合は、次のようにできます:

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