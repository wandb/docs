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

`wandb.config` を使用して、 sweep configuration からハイパーパラメータの名前と値にアクセスします。これは辞書のように機能します。

sweep 外の run の場合、 `wandb.init` の `config` 引数に辞書を渡して `wandb.config` の値を設定します。 sweep では、 `wandb.init` に指定された設定はすべてデフォルト値として機能し、 sweep はそれを上書きできます。

明示的な振る舞いには `config.setdefaults` を使用します。次のコードスニペットは、両方のメソッドを示しています。

{{< tabpane text=true >}}
{{% tab "wandb.init()" %}}
```python
# ハイパーパラメータのデフォルト値を設定
config_defaults = {"lr": 0.1, "batch_size": 256}

# run を開始し、 sweep が上書きできる
# デフォルト値を提供します
with wandb.init(config=config_defaults) as run:
    # ここにトレーニング コードを追加
    ...
```
{{% /tab %}}
{{% tab "config.setdefaults()" %}}
```python
# ハイパーパラメータのデフォルト値を設定
config_defaults = {"lr": 0.1, "batch_size": 256}

# run を開始
with wandb.init() as run:
    # sweep で設定されていない値を更新します
    run.config.setdefaults(config_defaults)

    # ここにトレーニング コードを追加
```
{{% /tab %}}
{{< /tabpane >}}
