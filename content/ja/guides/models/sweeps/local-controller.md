---
title: アルゴリズムをローカルで管理する
description: W&B のクラウドホストサービスを使わずに、検索や停止アルゴリズムをローカルで実行します。
menu:
  default:
    identifier: local-controller
    parent: sweeps
---

ハイパーパラメータコントローラは、デフォルトで Weights & Biases によりクラウドサービスとしてホストされています。W&B エージェントはコントローラと通信し、トレーニングに使用する次のパラメータセットを決定します。コントローラはまた、どの run を停止できるかを判断する早期停止アルゴリズムも実行します。

ローカルコントローラ機能を使うと、ユーザーはローカルでサーチや停止アルゴリズムを実行できます。ローカルコントローラを使うことで、コードを確認したり、問題のデバッグや新機能の開発をクラウドサービスに組み込む前に行えます。

{{% alert color="secondary" %}}
この機能は、Sweeps ツール用の新しいアルゴリズムの開発やデバッグを迅速に行うために用意されています。実際のハイパーパラメータ最適化ワークロードには意図されていません。
{{% /alert %}}

始める前に、W&B SDK（`wandb`）をインストールしてください。以下のコードスニペットをコマンドラインで入力します。

```
pip install wandb sweeps 
```

以下の例では、設定ファイルとトレーニングループが Python スクリプトまたは Jupyter Notebook で既に定義されていることを前提としています。設定ファイルの定義方法については、[Define sweep configuration]({{< relref "/guides/models/sweeps/define-sweep-configuration/" >}}) をご覧ください。

### コマンドラインからローカルコントローラを実行

W&B のクラウドサービスでホストされるハイパーパラメータコントローラを使う場合と同様に、sweep を初期化します。sweep job にローカルコントローラを利用したい場合は、`controller` フラグを指定してください。

```bash
wandb sweep --controller config.yaml
```

また、sweep の初期化とローカルコントローラの指定を2ステップに分けることもできます。

この方法では、まず sweep の YAML 設定ファイルに以下のキーと値を追加します。

```yaml
controller:
  type: local
```

次に、sweep を初期化します。

```bash
wandb sweep config.yaml
```

`wandb sweep` によって sweep ID が生成されます。sweep を初期化した後、[`wandb controller`]({{< relref "/ref/python/sdk/functions/controller.md" >}}) を使ってコントローラを起動します。

```bash
wandb controller {user}/{entity}/{sweep_id}
```

ローカルコントローラの利用を指定したら、sweep を実行するために1つ以上の Sweep agent を起動します。通常どおり W&B Sweep を開始してください。詳細は [Start sweep agents]({{< relref "/guides/models/sweeps/start-sweep-agents.md" >}}) も参照してください。

```bash
wandb sweep sweep_ID
```

### W&B Python SDK でローカルコントローラを実行

以下のコードスニペットでは、W&B Python SDK でローカルコントローラを指定して利用する方法を示します。

Python SDK でコントローラを使う最も簡単な方法は、[`wandb.controller`]({{< relref "/ref/python/sdk/functions/controller.md" >}}) メソッドに sweep ID を渡すことです。次に、返されたオブジェクトの `run` メソッドで sweep job を開始します。

```python
sweep = wandb.controller(sweep_id)
sweep.run()
```

コントローラループをより細かく制御したい場合は次のようにします。

```python
import wandb

sweep = wandb.controller(sweep_id)
while not sweep.done():
    sweep.print_status()
    sweep.step()
    time.sleep(5)
```

パラメータの取得方法をさらに細かく制御したいなら以下のようにします。

```python
import wandb

sweep = wandb.controller(sweep_id)
while not sweep.done():
    params = sweep.search()
    sweep.schedule(params)
    sweep.print_status()
```

設定をすべてコードで指定したい場合、以下のように実装できます。

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