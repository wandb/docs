---
title: ログ軸をカスタマイズする
menu:
  default:
    identifier: ja-guides-models-track-log-customize-logging-axes
    parent: log-objects-and-media
---

W&B にメトリクスをログする際、カスタムの x 軸を設定できます。デフォルトでは、W&B はメトリクスを *ステップ* ごとにログします。各ステップは `wandb.Run.log()` API コールに対応しています。

例えば、以下のスクリプトでは `for` ループが 10 回繰り返されます。各イテレーションで、スクリプトは `validation_loss` というメトリクスをログし、ステップ数を 1 ずつ増やします。

```python
import wandb

with wandb.init() as run:
  # range 関数は 0 から 9 の数列を生成します
  for i in range(10):
    log_dict = {
        "validation_loss": 1/(i+1)   
    }
    run.log(log_dict)
```

Project の Workspace では、`validation_loss` メトリクスが `step` x 軸に対してプロットされます。この x 軸は `wandb.Run.log()` が呼び出されるたびに 1 ずつ増加します。前述のコードの場合、x 軸には 0, 1, 2, ..., 9 というステップ番号が表示されます。

{{< img src="/images/experiments/standard_axes.png" alt="`step` を x 軸にした折れ線グラフパネル。" >}}

特定の状況では、対数スケールの x 軸など、異なる x 軸にメトリクスをログしたほうがわかりやすい場合もあります。[`define_metric()`]({{< relref path="/ref/python/sdk/classes/run/#define_metric" lang="ja" >}}) メソッドを使うと、ログ中の任意のメトリクスをカスタム x 軸として利用できます。

y 軸として表示したいメトリクスを `name` パラメータで指定します。`step_metric` パラメータには x 軸として使いたいメトリクスを指定します。カスタムメトリクスをログする際は、x 軸と y 軸の値をキーと値のペアとして辞書にして渡してください。

カスタム x 軸メトリクスを設定するコードスニペットを下記に示します。`<>` 内の値は自分の値に置き換えてください。

```python
import wandb

custom_step = "<custom_step>"  # カスタム x 軸の名前
metric_name = "<metric>"  # y 軸メトリクスの名前

with wandb.init() as run:
    # ステップメトリクス (x 軸) と、そのメトリクスに対してログする y 軸を指定
    run.define_metric(step_metric = custom_step, name = metric_name)

    for i in range(10):
        log_dict = {
            custom_step : int,  # x 軸の値
            metric_name : int,  # y 軸の値
        }
        run.log(log_dict)
```

例として、以下のコードスニペットは `x_axis_squared` というカスタム x 軸を作成しています。カスタム x 軸の値は for ループのインデックス `i` を 2 乗したもの (`i**2`) です。y 軸は Python の組み込み `random` モジュールを使った `validation_loss` (バリデーションロス) のダミー値です。

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

下記の画像は、W&B App UI での結果のプロットを表しています。`validation_loss` メトリクスがカスタム x 軸 `x_axis_squared` に対してプロットされています。この x 軸は for ループのインデックス `i` の 2 乗 (0, 1, 4, 9, 16, 25, 36, 49, 64, 81) になっていることが確認できます。

{{< img src="/images/experiments/custom_x_axes.png" alt="ループ番号を 2 乗して W&B にログし、カスタム x 軸を使っている折れ線グラフパネル。" >}}

複数のメトリクスに対して `globs`（文字列プレフィックス）を使ってカスタム x 軸を指定することも可能です。例えば、下記のコードスニペットでは `train/*` で始まるログ済みメトリクスをまとめて、`train/step` を x 軸としてプロットします。

```python
import wandb

with wandb.init() as run:

    # 他の全ての train/ メトリクスにこの step を使用
    run.define_metric("train/*", step_metric="train/step")

    for i in range(10):
        log_dict = {
            "train/step": 2**i,  # W&B 内部ステップを用いた指数関数的増加
            "train/loss": 1 / (i + 1),  # x 軸は train/step
            "train/accuracy": 1 - (1 / (1 + i)),  # x 軸は train/step
            "val/loss": 1 / (1 + i),  # x 軸は内部 wandb step
        }
        run.log(log_dict)
```