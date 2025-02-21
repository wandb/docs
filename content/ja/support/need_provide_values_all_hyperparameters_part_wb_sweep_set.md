---
title: Do I need to provide values for all hyperparameters as part of the W&B Sweep.
  Can I set defaults?
menu:
  support:
    identifier: ja-support-need_provide_values_all_hyperparameters_part_wb_sweep_set
tags:
- sweeps
toc_hide: true
type: docs
---

ハイパーパラメーター名と値には、辞書のように機能する `wandb.config` を使用して、sweep configuration からアクセスします。

sweep 外での run の場合、`config` 引数に辞書を渡すことで `wandb.init` で `wandb.config` の値を設定します。sweep 内では、`wandb.init` に提供された設定はデフォルト値として機能し、sweep がそれを上書きできます。

明示的な振る舞いには `config.setdefaults` を使用します。以下のコードスニペットは両方のメソッドを示しています：

{{< tabpane text=true >}}
{{% tab "wandb.init()" %}}
```python
# ハイパーパラメーターのデフォルト値を設定
config_defaults = {"lr": 0.1, "batch_size": 256}

# run を開始し、sweep が上書きできるデフォルト値を提供
with wandb.init(config=config_defaults) as run:
    # トレーニングコードをここに追加
    ...
```
{{% /tab %}}
{{% tab "config.setdefaults()" %}}
```python
# ハイパーパラメーターのデフォルト値を設定
config_defaults = {"lr": 0.1, "batch_size": 256}

# run を開始
with wandb.init() as run:
    # sweep によって設定されていない値を更新
    run.config.setdefaults(config_defaults)

    # トレーニングコードをここに追加
```
{{% /tab %}}
{{< /tabpane >}}