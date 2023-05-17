---
displayed_sidebar: default
---
# ログ軸のカスタマイズ

`define_metric`を使用して、**カスタムx軸**を設定します。カスタムx軸は、トレーニング中に過去の異なる時間ステップに非同期でログを記録する必要がある状況で便利です。例えば、エピソードごとの報酬とステップごとの報酬をトラッキングするRLで役立ちます。

[Google Colabで`define_metric`を試す →](http://wandb.me/define-metric-colab)

### 軸のカスタマイズ

デフォルトでは、すべてのメトリクスはW&B内部の`step`という同じx軸に対して記録されます。しかし、場合によっては、以前のステップにログを記録するか、異なるx軸を使用したいことがあります。

以下に、デフォルトのステップではなく、カスタムx軸メトリックを設定する例を示します。

```python
import wandb

wandb.init()
# カスタムx軸メトリック定義
wandb.define_metric("custom_step")
# どのメトリクスを対応させるか定義
wandb.define_metric(
  "validation_loss", step_metric="custom_step")

for i in range(10):
  log_dict = {
      "train_loss": 1/(i+1),
      "custom_step": i**2,
      "validation_loss": 1/(i+1)   
  }
  wandb.log(log_dict)
```

x軸は、グロブを使って設定することもできます。現在、文字列プレフィックスを持つグロブのみが利用可能です。次の例では、プレフィックス`"train/"`でログされたすべてのメトリクスをx軸`"train/step"`にプロットします。



```python

import wandb



wandb.init()

# カスタムx軸メトリックを定義する

wandb.define_metric("train/step")

# 他のすべてのtrain/メトリックをこのステップで使用するように設定する

wandb.define_metric("train/*", step_metric="train/step")



for i in range(10):

  log_dict = {

      "train/step": 2 ** i # W&Bステップとともに指数関数的に増加

      "train/loss": 1/(i+1), # x軸はtrain/step

      "train/accuracy": 1 - (1/(1+i)), # x軸はtrain/step

      "val/loss": 1/(1+i), # x軸は内部のwandbステップ

      

  }

  wandb.log(log_dict)

```