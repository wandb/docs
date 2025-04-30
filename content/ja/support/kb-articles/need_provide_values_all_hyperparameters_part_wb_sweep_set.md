---
title: すべてのハイパーパラメーターの値を W&B Sweep の一部として提供する必要がありますか。デフォルト値を設定できますか？
menu:
  support:
    identifier: >-
      ja-support-kb-articles-need_provide_values_all_hyperparameters_part_wb_sweep_set
support:
  - sweeps
toc_hide: true
type: docs
url: /ja/support/:filename
---
ハイパーパラメーターの名前と値にアクセスするには、辞書のように振る舞う `wandb.config` を使って、sweep configuration から取得します。

sweep の外で run を行う場合、`wandb.config` の値を設定するには、辞書を `wandb.init` の `config` 引数に渡します。sweep では、`wandb.init` に提供される任意の設定がデフォルト値として機能し、sweep がそれを上書きできます。

明示的な振る舞いには `config.setdefaults` を使用します。以下のコードスニペットは両方のメソッドを示しています：

{{< tabpane text=true >}}
{{% tab "wandb.init()" %}}
```python
# ハイパーパラメーターのデフォルト値を設定
config_defaults = {"lr": 0.1, "batch_size": 256}

# run を開始し、デフォルトを指定
# sweep がこれを上書きできます
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