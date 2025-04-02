---
title: Do I need to provide values for all hyperparameters as part of the W&B Sweep.
  Can I set defaults?
menu:
  support:
    identifier: ja-support-kb-articles-need_provide_values_all_hyperparameters_part_wb_sweep_set
support:
- sweeps
toc_hide: true
type: docs
url: /support/:filename
---

`wandb.config` を使用して、sweep configuration からハイパーパラメータの名前と値にアクセスします。これは辞書のように機能します。

sweep 外の run の場合、`wandb.init` の `config` 引数に辞書を渡して、`wandb.config` の値を設定します。sweep では、`wandb.init` に指定された設定はすべてデフォルト値として機能し、sweep はそれを上書きできます。

明示的な振る舞いには `config.setdefaults` を使用します。次のコードスニペットは、両方の方法を示しています。

{{< tabpane text=true >}}
{{% tab "wandb.init()" %}}
```python
# Set default values for hyperparameters
config_defaults = {"lr": 0.1, "batch_size": 256}

# Start a run and provide defaults
# that a sweep can override
with wandb.init(config=config_defaults) as run:
    # Add training code here
    ...
```
{{% /tab %}}
{{% tab "config.setdefaults()" %}}
```python
# Set default values for hyperparameters
config_defaults = {"lr": 0.1, "batch_size": 256}

# Start a run
with wandb.init() as run:
    # Update any values not set by the sweep
    run.config.setdefaults(config_defaults)

    # Add training code here
```
{{% /tab %}}
{{< /tabpane >}}
