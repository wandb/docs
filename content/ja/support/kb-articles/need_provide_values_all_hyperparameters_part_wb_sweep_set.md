---
title: W&B Sweep で全てのハイパーパラメーターに値を指定する必要がありますか？デフォルト値を設定できますか？
menu:
  support:
    identifier: ja-support-kb-articles-need_provide_values_all_hyperparameters_part_wb_sweep_set
support:
- スイープ
toc_hide: true
type: docs
url: /support/:filename
---

(run.config()) を使って sweep configuration からハイパーパラメーターの名前と値にアクセスできます。これは辞書のように振る舞います。

sweep 以外の run では、`wandb.init()` の `config` 引数に辞書を渡すことで `wandb.Run.config()` の値を設定します。sweep 内では、`wandb.init()` に渡した設定がデフォルト値となり、sweep がそれを上書きできます。

明示的な振る舞いを指定したい場合は、`wandb.Run.config.setdefaults()` を使用してください。以下のコードスニペットで両方の使い方を示します。

{{< tabpane text=true >}}
{{% tab "wandb.init()" %}}
```python
# ハイパーパラメーターのデフォルト値を設定
config_defaults = {"lr": 0.1, "batch_size": 256}

# run を開始し、デフォルト値を設定
# sweep 側で上書き可能
with wandb.init(config=config_defaults) as run:
    # ここにトレーニングコードを追加
    ...
```
{{% /tab %}}
{{% tab "config.setdefaults()" %}}
```python
# ハイパーパラメーターのデフォルト値を設定
config_defaults = {"lr": 0.1, "batch_size": 256}

# run を開始
with wandb.init() as run:
    # sweep で設定されていない値のみ更新
    run.config.setdefaults(config_defaults)

    # ここにトレーニングコードを追加
```
{{% /tab %}}
{{< /tabpane >}}