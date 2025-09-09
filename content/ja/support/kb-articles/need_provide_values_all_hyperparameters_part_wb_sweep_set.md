---
title: W&B Sweep で、すべてのハイパーパラメーターの値を指定する必要がありますか？デフォルト値を設定できますか？
menu:
  support:
    identifier: ja-support-kb-articles-need_provide_values_all_hyperparameters_part_wb_sweep_set
support:
- sweeps
toc_hide: true
type: docs
url: /support/:filename
---

sweep configuration に含まれるハイパーパラメーター名と値には、辞書のように振る舞う `(run.config())` を使ってアクセスできます。

sweep の外側の run では、`wandb.Run.config()` の値を `wandb.init()` の `config` 引数に辞書を渡して設定します。sweep の中では、`wandb.init()` に渡した任意の設定はデフォルト値として扱われ、sweep がそれを上書きできます。

明示的な振る舞いが必要な場合は `rwandb.Run.config.setdefaults()` を使用します。次のコードスニペットは両方の方法を示します:

{{< tabpane text=true >}}
{{% tab "wandb.init()" %}}
```python
# ハイパーパラメーターのデフォルト値を設定します
config_defaults = {"lr": 0.1, "batch_size": 256}

# run を開始し、デフォルトを指定します
# sweep が上書き可能
with wandb.init(config=config_defaults) as run:
    # ここにトレーニング コードを追加します
    ...
```
{{% /tab %}}
{{% tab "config.setdefaults()" %}}
```python
# ハイパーパラメーターのデフォルト値を設定します
config_defaults = {"lr": 0.1, "batch_size": 256}

# run を開始します
with wandb.init() as run:
    # sweep で設定されていない値を更新します
    run.config.setdefaults(config_defaults)

    # ここにトレーニング コードを追加します
```
{{% /tab %}}
{{< /tabpane >}}