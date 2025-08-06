---
title: W&B Sweep で全てのハイパーパラメーターの値を指定する必要がありますか？デフォルト値を設定できますか？
url: /support/:filename
toc_hide: true
type: docs
support:
- スイープ
---

(run.config()) を使って sweep configuration からハイパーパラメーター名や値に アクセス できます。これは辞書のように動作します。

sweep 外の run では、`wandb.Run.config()` の値を `wandb.init()` の `config` 引数に辞書を渡して設定します。sweep 内では、`wandb.init()` に渡された設定がデフォルト値となり、sweep によって上書きされます。

明示的な振る舞い をしたい場合は、`wandb.Run.config.setdefaults()` を使用してください。以下のコードスニペットは両方の方法を示しています。

{{< tabpane text=true >}}
{{% tab "wandb.init()" %}}
```python
# ハイパーパラメーターのデフォルト値を設定
config_defaults = {"lr": 0.1, "batch_size": 256}

# run を開始し、デフォルト値を指定
# sweep で上書き可能
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