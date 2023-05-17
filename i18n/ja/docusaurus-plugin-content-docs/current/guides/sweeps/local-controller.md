---
description: >-
  Search and stop algorithms locally instead of using the Weights & Biases
  cloud-hosted service.
displayed_sidebar: default
---

# ローカルで探索と停止アルゴリズムを実行する

<head>
  <title>W&Bエージェントでローカルに探索と停止アルゴリズムを実行する</title>
</head>

ハイパーパラメータコントローラは、デフォルトでWeights & Biasesによってクラウドサービスとしてホストされています。W&Bエージェントは、コントローラと通信して、トレーニングに使用する次のパラメータセットを決定します。また、コントローラは、どのrunを停止できるかを判断するために、早期停止アルゴリズムを実行する責任があります。

ローカルコントローラ機能は、ユーザーがローカルで探索と停止アルゴリズムを開始することを可能にします。ローカルコントローラは、ユーザーが問題をデバッグしたり、クラウドサービスに組み込むことができる新しい機能を開発するために、コードを調べたり操作できる機能を提供します。

:::caution
この機能は、Sweepsツール用の新しいアルゴリズムの開発とデバッグを高速化するために提供されています。ハイパーパラメータ最適化の実際のワークロードには、使用を想定していません。
:::

始める前に、Weights & Biases SDK(`wandb`)をインストールする必要があります。以下のコードスニペットをコマンドラインに入力してください。

```
pip install wandb sweeps
```

次の例では、設定ファイルとトレーニングループがPythonスクリプトまたはJupyterノートブックで定義されていることを前提としています。設定ファイルの定義方法についての詳細は、[スイープ構成の定義](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration)をご覧ください。

### コマンドラインからローカルコントローラを実行する

W&Bクラウドサービスでホストされたハイパーパラメータコントローラを使用する場合と同様に、スイープを初期化します。コントローラーフラグ（`controller`）を指定して、W&Bスイープジョブのローカルコントローラを使用することを示します。
```python
wandb sweep --controller config.yaml
```

または、スイープの初期化とローカルコントローラーの使用を指定するのを2つのステップに分けることができます。

ステップを分けるには、まず以下のキー-値をスイープのYAML設定ファイルに追加してください:

```yaml
controller:
  type: local
```

次に、スイープを初期化します:

```
wandb sweep config.yaml
```

スイープを初期化した後、[`wandb controller`](https://docs.wandb.ai/ref/python/controller) でコントローラーを起動します:

```python
# wandb sweep コマンドは sweep_id を表示します
wandb controller {user}/{entity}/{sweep_id}
```

ローカルコントローラーの使用を指定したら、スイープを実行するために1つ以上のスイープエージェントを起動します。通常と同様に、W&B スイープを開始します。詳細については、[スイープエージェントの開始](https://docs.wandb.ai/guides/sweeps/start-sweep-agents) を参照してください。

```
wandb sweep sweep_ID
```
### W&B Python SDK を使用してローカルコントローラーを実行する

以下のコードスニペットは、Weights & Biases Python SDK でローカルコントローラーを指定し、使用する方法を示しています。

Python SDK でコントローラを使用する最も簡単な方法は、スイープID を [`wandb.controller`](https://docs.wandb.ai/ref/python/controller) メソッドに渡すことです。次に、返されるオブジェクトの `run` メソッドを使用して、スイープジョブを開始します。

```python
sweep = wandb.controller(sweep_id)
sweep.run()
```

コントローラループの制御をより詳細に行いたい場合：

```python
import wandb

sweep = wandb.controller(sweep_id)
while not sweep.done():
    sweep.print_status()
    sweep.step()
    time.sleep(5)
```

または、提供されるパラメータに対してさらに制御を行いたい場合:

```python
import wandb

sweep = wandb.controller(sweep_id)
while not sweep.done():
    params = sweep.search()
    sweep.schedule(params)
    sweep.print_status()
```
もし、スイープを完全にコードで指定したい場合は、以下のようなことができます。

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