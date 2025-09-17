---
title: アルゴリズムをローカルで管理する
description: W&B のクラウドホスト型サービスを使用する代わりに、探索・停止アルゴリズムをローカルで実行します。
menu:
  default:
    identifier: ja-guides-models-sweeps-local-controller
    parent: sweeps
---

デフォルトでは、ハイパーパラメーター コントローラは Weights & Biases によってクラウド サービスとしてホストされています。W&B のエージェントは、トレーニングに使用する次のパラメータ セットを決定するためにコントローラと通信します。コントローラは、どの run を停止できるかを判定するための早期停止アルゴリズムの実行も担当します。

ローカル コントローラ機能を使うと、ユーザーは探索や停止のアルゴリズムをローカルで開始・実行できます。ローカル コントローラにより、問題のデバッグや、クラウド サービスに取り込める新機能の開発のために、コードを検査・計測することが可能になります。

{{% alert color="secondary" %}}
この機能は、Sweeps ツール向けの新しいアルゴリズムの高速な開発とデバッグを支援する目的で提供されています。実際のハイパーパラメーター最適化のワークロードを想定したものではありません。
{{% /alert %}}

開始する前に、W&B SDK（`wandb`）をインストールする必要があります。次のコードスニペットをコマンドラインに入力してください:

```
pip install wandb sweeps 
```

以下の例では、すでに設定ファイルとトレーニング ループが Python スクリプトまたは Jupyter Notebook 内で定義されているものとします。設定ファイルの定義方法については、[sweep configuration を定義する]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ja" >}}) を参照してください。

### コマンドラインからローカル コントローラを実行する

W&B のクラウド サービスでホストされるハイパーパラメーター コントローラを使うときと同様に、sweep を初期化します。W&B の sweep ジョブにローカル コントローラを使いたいことを示すため、コントローラ フラグ（`controller`）を指定します:

```bash
wandb sweep --controller config.yaml
```

あるいは、sweep の初期化とローカル コントローラを使用する指定を 2 段階に分けることもできます。

手順を分けるには、まず sweep の YAML 設定ファイルに次のキーと値を追加します:

```yaml
controller:
  type: local
```

次に、sweep を初期化します:

```bash
wandb sweep config.yaml
```

`wandb sweep` は sweep ID を生成します。sweep を初期化したら、[`wandb controller`]({{< relref path="/ref/python/sdk/functions/controller.md" lang="ja" >}}) でコントローラを起動します:

```bash
wandb controller {user}/{entity}/{sweep_id}
```

ローカル コントローラを使うことを指定したら、sweep を実行するために 1 つ以上の sweep agent を起動します。通常どおりに W&B の Sweep を開始してください。詳しくは、[sweep agent を開始する]({{< relref path="/guides/models/sweeps/start-sweep-agents.md" lang="ja" >}}) を参照してください。

```bash
wandb sweep sweep_ID
```

### W&B Python SDK でローカル コントローラを実行する

以下のコードスニペットは、W&B Python SDK でローカル コントローラを指定して使用する方法を示します。

Python SDK でコントローラを使う最も簡単な方法は、[`wandb.controller`]({{< relref path="/ref/python/sdk/functions/controller.md" lang="ja" >}}) メソッドに sweep ID を渡すことです。次に、戻り値のオブジェクトの `run` メソッドを使って sweep ジョブを開始します:

```python
sweep = wandb.controller(sweep_id)
sweep.run()
```

コントローラ ループをより細かく制御したい場合:

```python
import wandb

sweep = wandb.controller(sweep_id)
while not sweep.done():
    sweep.print_status()
    sweep.step()
    time.sleep(5)
```

提供されるパラメータをさらに細かく制御したい場合:

```python
import wandb

sweep = wandb.controller(sweep_id)
while not sweep.done():
    params = sweep.search()
    sweep.schedule(params)
    sweep.print_status()
```

sweep を完全にコードで指定したい場合は、次のようにします:

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