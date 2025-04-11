---
title: アルゴリズムをローカルで管理する
description: W&B のクラウドホスティッドサービスを使用せずに、ローカルで検索およびストップアルゴリズムを実行します。
menu:
  default:
    identifier: ja-guides-models-sweeps-local-controller
    parent: sweeps
---

ハイパーパラメータコントローラは、デフォルトでウェイト＆バイアスによってクラウドサービスとしてホストされています。W&Bエージェントはコントローラと通信して、トレーニングに使用する次のパラメータセットを決定します。コントローラは、どのrunを停止するかを決定するための早期停止アルゴリズムの実行も担当しています。

ローカルコントローラ機能により、ユーザーはアルゴリズムをローカルで検索および停止することができます。ローカルコントローラは、ユーザーにコードを検査および操作する能力を与え、問題のデバッグやクラウドサービスに組み込むことができる新機能の開発を可能にします。

{{% alert color="secondary" %}}
この機能は、Sweepsツールの新しいアルゴリズムの迅速な開発とデバッグをサポートするために提供されています。本来のハイパーパラメータ最適化のワークロードを目的としていません。
{{% /alert %}}

始める前に、W&B SDK (`wandb`) をインストールする必要があります。次のコマンドをコマンドラインに入力してください：

```
pip install wandb sweeps 
```

以下の例では、既に設定ファイルとPythonスクリプトまたはJupyterノートブックで定義されたトレーニングループがあることを前提としています。設定ファイルの定義方法についての詳細は、[Define sweep configuration]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ja" >}}) を参照してください。

### コマンドラインからローカルコントローラを実行する

通常、W&Bがホストするクラウドサービスのハイパーパラメータコントローラを使用する際と同様に、スイープを初期化します。ローカルコントローラをW&Bスイープジョブ用に使用することを示すために、コントローラフラグ (`controller`) を指定します：

```bash
wandb sweep --controller config.yaml
```

または、スイープを初期化することと、ローカルコントローラを使用することを指定することを2つのステップに分けることもできます。

ステップを分けるには、まず次のキーと値をスイープのYAML設定ファイルに追加します：

```yaml
controller:
  type: local
```

次に、スイープを初期化します：

```bash
wandb sweep config.yaml
```

スイープを初期化した後、[`wandb controller`]({{< relref path="/ref/python/controller.md" lang="ja" >}}) を使用してコントローラを開始します：

```bash
# wandb sweep コマンドが sweep_id を出力します
wandb controller {user}/{entity}/{sweep_id}
```

ローカルコントローラを使用することを指定した後、スイープを実行するために1つ以上のSweepエージェントを開始します。通常のW&Bスイープを開始するのと同じようにしてください。[Start sweep agents]({{< relref path="/guides/models/sweeps/start-sweep-agents.md" lang="ja" >}}) を参照してください。

```bash
wandb sweep sweep_ID
```

### W&B Python SDKを使用してローカルコントローラを実行する

以下のコードスニペットは、W&B Python SDKを使用してローカルコントローラを指定し、使用する方法を示しています。

Python SDKでコントローラを使用する最も簡単な方法は、[`wandb.controller`]({{< relref path="/ref/python/controller.md" lang="ja" >}}) メソッドにスイープIDを渡すことです。次に、戻りオブジェクトの `run` メソッドを使用してスイープジョブを開始します：

```python
sweep = wandb.controller(sweep_id)
sweep.run()
```

コントローラループをより詳細に制御したい場合：

```python
import wandb

sweep = wandb.controller(sweep_id)
while not sweep.done():
    sweep.print_status()
    sweep.step()
    time.sleep(5)
```

パラメータの提供に関してさらに詳細なコントロールが必要な場合：

```python
import wandb

sweep = wandb.controller(sweep_id)
while not sweep.done():
    params = sweep.search()
    sweep.schedule(params)
    sweep.print_status()
```

コードですべてのスイープを指定したい場合は、次のように実装できます：

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