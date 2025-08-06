---
title: ログ軸をカスタマイズする
menu:
  default:
    identifier: customize-logging-axes
    parent: log-objects-and-media
---

W&B にメトリクスをログする際、カスタムの x 軸を設定できます。デフォルトでは、W&B はメトリクスを *step* としてログします。各 step は `wandb.Run.log()` API の呼び出しに対応します。

例えば、以下のスクリプトは 10 回ループする `for` ループがあります。各イテレーションで `validation_loss` というメトリクスをログし、step 番号を 1 ずつ増やしています。

```python
import wandb

with wandb.init() as run:
  # range 関数は 0 から 9 までの数列を生成します
  for i in range(10):
    log_dict = {
        "validation_loss": 1/(i+1)   
    }
    run.log(log_dict)
```

Project の Workspace では、`validation_loss` メトリクスが `step` を x 軸としてプロットされます。`wandb.Run.log()` が呼ばれるたびに step が 1 増加します。上記のコードでは、x 軸には 0, 1, 2, ..., 9 の step 番号が表示されます。

{{< img src="/images/experiments/standard_axes.png" alt="`step` を x 軸に使った折れ線グラフパネル。" >}}

場合によっては、x 軸として他の指標（例えば対数スケールの x 軸など）を使った方が分かりやすいことがあります。ログしたどんなメトリクスでもカスタム x 軸として利用するには、[`define_metric()`]({{< relref "/ref/python/sdk/classes/run/#define_metric" >}}) メソッドを使ってください。

y 軸として表示したいメトリクスは `name` パラメータで指定します。`step_metric` パラメータには x 軸に使いたいメトリクスを指定します。カスタムメトリクスをログする場合、x 軸と y 軸それぞれに対応する値を辞書のキー・バリューの形で指定します。

以下のコードスニペットをコピー＆ペーストして、カスタム x 軸を設定できます。`<>` 内の値は自分の値に置き換えてください。

```python
import wandb

custom_step = "<custom_step>"  # カスタム x 軸の名前
metric_name = "<metric>"  # y 軸メトリクスの名前

with wandb.init() as run:
    # x 軸メトリクス（step_metric）と、その上に表示するメトリクス（name）を指定
    run.define_metric(step_metric = custom_step, name = metric_name)

    for i in range(10):
        log_dict = {
            custom_step : int,  # x 軸の値
            metric_name : int,  # y 軸の値
        }
        run.log(log_dict)
```

例として、次のコードスニペットは `x_axis_squared` というカスタム x 軸を作成します。カスタム x 軸の値は、for ループ内のインデックス `i` の2乗（`i**2`）です。y 軸には Python の組み込み `random` モジュールを使って作成した `validation_loss` のダミー値を使っています。

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

以下の画像は、W&B App UI でのプロット結果を示しています。`validation_loss` メトリクスがカスタム x 軸 `x_axis_squared`（for ループのインデックス `i` の2乗）でプロットされています。x 軸の値は `0, 1, 4, 9, 16, 25, 36, 49, 64, 81` となり、それぞれ `0, 1, 2, ..., 9` の2乗に対応しています。

{{< img src="/images/experiments/custom_x_axes.png" alt="カスタム x 軸を使った折れ線グラフパネル。W&B にはループ番号の二乗が x 軸値としてログされています。" >}}

複数のメトリクスに対しても、文字列プレフィックス付きの `globs` を使ってカスタム x 軸を設定できます。例えば、次のコードスニペットでは、`train/*` プレフィックスのついた全てのメトリクスを x 軸 `train/step` でプロットします。

```python
import wandb

with wandb.init() as run:

    # train/ で始まる全てのメトリクスをこの step に揃えて記録
    run.define_metric("train/*", step_metric="train/step")

    for i in range(10):
        log_dict = {
            "train/step": 2**i,  # 指数的増加（内部W&B step）
            "train/loss": 1 / (i + 1),  # x 軸は train/step
            "train/accuracy": 1 - (1 / (1 + i)),  # x 軸は train/step
            "val/loss": 1 / (1 + i),  # x 軸は内部 wandb step
        }
        run.log(log_dict)
```