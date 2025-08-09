---
title: アルゴリズムをローカルで管理する
description: W&B のクラウドホスト型サービスを使わずに、検索とストップのアルゴリズムをローカルで実行します。
menu:
  default:
    identifier: ja-guides-models-sweeps-local-controller
    parent: sweeps
---

ハイパーパラメーターコントローラは、デフォルトで Weights & Biases がクラウドサービスとしてホストしています。W&B エージェントはコントローラと通信し、トレーニングに使用する次のパラメータセットを決定します。コントローラは、どの run を停止できるかを判断するためのアーリーストッピングアルゴリズムの実行も担当します。

ローカルコントローラ機能を使うと、ユーザーがアルゴリズムの探索や停止をローカルで実行できます。ローカルコントローラでは、コードを調査したりインストゥルメント化したりしてデバッグや新しい機能の開発が可能です。開発した機能はクラウドサービスにも組み込むことができます。

{{% alert color="secondary" %}}
この機能は、Sweeps ツール向けの新しいアルゴリズムの開発やデバッグを素早く行うためのサポートとして提供されています。実際のハイパーパラメーター最適化用途では推奨されません。
{{% /alert %}}

始める前に、W&B SDK（`wandb`）をインストールしてください。以下のコードスニペットをコマンドラインに入力します。

```
pip install wandb sweeps 
```

以下の例では、設定ファイルとトレーニングループが python スクリプトまたは Jupyter Notebook で定義されていることを前提としています。設定ファイルの定義方法については [スイープ設定ファイルの定義]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ja" >}}) を参照してください。

### コマンドラインからローカルコントローラを実行する

W&B のクラウドサービスとしてホストされているハイパーパラメーターコントローラを使う場合と同様に、スイープを初期化します。W&B sweep のジョブでローカルコントローラを利用するには、コントローラフラグ（`controller`）を指定します。

```bash
wandb sweep --controller config.yaml
```

もしくは、スイープの初期化とローカルコントローラの指定を2つのステップに分けることもできます。

ステップを分ける場合は、まず以下のキーと値を sweep の YAML 設定ファイルに追加します。

```yaml
controller:
  type: local
```

次に、スイープを初期化します。

```bash
wandb sweep config.yaml
```

`wandb sweep` を実行すると、スイープIDが生成されます。スイープを初期化した後、[`wandb controller`]({{< relref path="/ref/python/sdk/functions/controller.md" lang="ja" >}}) を利用してコントローラを起動します。

```bash
wandb controller {user}/{entity}/{sweep_id}
```

ローカルコントローラの利用を指定したら、1つ以上の Sweep agent を起動して sweep を実行します。W&B Sweep の開始方法は通常と同じです。詳細は [スイープエージェントの開始]({{< relref path="/guides/models/sweeps/start-sweep-agents.md" lang="ja" >}}) を参照してください。

```bash
wandb sweep sweep_ID
```

### W&B Python SDK でローカルコントローラを実行する

以下のコードスニペットは、W&B Python SDK でローカルコントローラを指定・利用する方法を示します。

Python SDK で簡単にコントローラを使うには、[`wandb.controller`]({{< relref path="/ref/python/sdk/functions/controller.md" lang="ja" >}}) メソッドにスイープIDを渡します。次に、返されたオブジェクトの `run` メソッドを使い sweep ジョブを起動します。

```python
sweep = wandb.controller(sweep_id)
sweep.run()
```

コントローラループをより細かく制御したい場合：

```python
import wandb

sweep = wandb.controller(sweep_id)
while not sweep.done():
    sweep.print_status()
    sweep.step()
    time.sleep(5)
```

さらに提供されるパラメータに対して詳細な制御を行う場合：

```python
import wandb

sweep = wandb.controller(sweep_id)
while not sweep.done():
    params = sweep.search()
    sweep.schedule(params)
    sweep.print_status()
```

すべてコードで sweep を指定したい場合は、以下のように記述できます。

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