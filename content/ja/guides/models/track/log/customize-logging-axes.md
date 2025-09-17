---
title: 対数軸をカスタマイズする
menu:
  default:
    identifier: ja-guides-models-track-log-customize-logging-axes
    parent: log-objects-and-media
---

W&B にメトリクスをログするとき、カスタムの x 軸を設定できます。デフォルトでは、W&B はメトリクスを *ステップ* としてログします。各ステップは `wandb.Run.log()` API 呼び出し 1 回に対応します。

例えば、次のスクリプトには 10 回反復する `for` ループがあります。各反復で、`validation_loss` というメトリクスをログし、ステップ番号を 1 ずつ増やします。

```python
import wandb

with wandb.init() as run:
  # range 関数は 0 から 9 までの数列を作成します
  for i in range(10):
    log_dict = {
        "validation_loss": 1/(i+1)   
    }
    run.log(log_dict)
```

プロジェクトの Workspace では、`validation_loss` メトリクスが `step` の x 軸に対してプロットされます。`wandb.Run.log()` が呼ばれるたびに x 軸は 1 ずつ増加します。上のコードでは、x 軸には 0, 1, 2, ..., 9 のステップ番号が表示されます。

{{< img src="/images/experiments/standard_axes.png" alt="x 軸に `step` を使った折れ線パネル。" >}}

状況によっては、対数スケールの x 軸など、別の x 軸でメトリクスをログした方が適切なことがあります。[`define_metric()`]({{< relref path="/ref/python/sdk/classes/run/#define_metric" lang="ja" >}}) メソッドを使うと、ログした任意のメトリクスをカスタム x 軸として利用できます。

y 軸に表示したいメトリクスは `name` パラメータで指定します。`step_metric` パラメータには x 軸として使いたいメトリクスを指定します。カスタムメトリクスをログするときは、辞書のキーと値として x 軸と y 軸の両方の値を指定します。

次のコードスニペットをコピー & ペーストして、カスタム x 軸メトリクスを設定してください。`<>` 内の値は自身の値に置き換えてください:

```python
import wandb

custom_step = "<custom_step>"  # カスタム x 軸の名前
metric_name = "<metric>"  # y 軸メトリクスの名前

with wandb.init() as run:
    # ステップメトリクス（x 軸）と、それに対してログするメトリクス（y 軸）を指定
    run.define_metric(step_metric = custom_step, name = metric_name)

    for i in range(10):
        log_dict = {
            custom_step : int,  # x 軸の値
            metric_name : int,  # y 軸の値
        }
        run.log(log_dict)
```

例として、次のコードスニペットは `x_axis_squared` というカスタム x 軸を作成します。カスタム x 軸の値は、for ループのインデックス `i` の二乗（`i**2`）です。y 軸は、Python の組み込み `random` モジュールを使って生成した検証損失（`"validation_loss"`）のダミー値です:

```python
import wandb
import random

with wandb.init() as run:
    run.define_metric(step_metric = "x_axis_squared", name = "validation_loss")

    for i in range(10):
        log_dict = {
            "x_axis_squared": i**2,
            "validation_loss": random.random(),
        }
        run.log(log_dict)
```

次の画像は、W&B App の UI に表示されるプロットです。`validation_loss` メトリクスは、for ループのインデックス `i` の二乗であるカスタム x 軸 `x_axis_squared` に対してプロットされています。x 軸の値は `0, 1, 4, 9, 16, 25, 36, 49, 64, 81` で、`0, 1, 2, ..., 9` の各二乗に対応しています。

{{< img src="/images/experiments/custom_x_axes.png" alt="カスタム x 軸を使った折れ線パネル。ループ回数の二乗を W&B にログしています。" >}}

文字列のプレフィックスを使った `globs` を利用すると、複数のメトリクスに対して一括でカスタム x 軸を設定できます。例えば、次のコードスニペットでは、`train/*` というプレフィックスを持つメトリクスを x 軸 `train/step` に対してプロットします:

```python
import wandb

with wandb.init() as run:

    # すべての train/ メトリクスにこのステップを適用
    run.define_metric("train/*", step_metric="train/step")

    for i in range(10):
        log_dict = {
            "train/step": 2**i,  # 内部の W&B ステップに対する指数的増加
            "train/loss": 1 / (i + 1),  # x 軸は train/step
            "train/accuracy": 1 - (1 / (1 + i)),  # x 軸は train/step
            "val/loss": 1 / (1 + i),  # x 軸は内部の wandb ステップ
        }
        run.log(log_dict)
```