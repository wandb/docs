---
description: W&B run をフォークする
displayed_sidebar: default
---


# Fork from a Run
:::caution
run をフォークする機能はプライベートプレビュー中です。この機能のア​​クセスをリクエストするには、W&Bサポート（support@wandb.com）に連絡してください。
:::

既存のW&B run から「フォーク」するには、[`wandb.init()`](../../ref/python/init.md) で run を初期化するときに `fork_from` を使用します。run からフォークすると、W&B はソース run の `run ID` と `step` を使用して新しい run を作成します。

run をフォークすることで、元の run に影響を与えることなく、特定の実験ポイントから異なるパラメータやモデルを探索することができます。

:::info
run をフォークするには、連続するステップが必要です。run 履歴とシステムメトリクスの基本的な時系列順序を乱さないように、フォークポイントを設定するために [`define_metric()`](https://docs.wandb.ai/ref/python/run#define_metric) で定義される非連続ステップを使用することはできません。
:::

:::info
run をフォークするには、[`wandb`](https://pypi.org/project/wandb/) SDK バージョン >= 0.16.5 が必要です。
:::

## フォークした run を開始する

run をフォークするには、[`wandb.init()`](../../ref/python/init.md) の `fork_from` 引数を使用し、フォーク元の run の `run ID` と `step` を指定します：

```python
import wandb

# 後でフォークする run を初期化します
original_run = wandb.init(project="your_project_name", entity="your_entity_name")
# ... トレーニングやログ ...
original_run.finish()

# 特定のステップから run をフォークする
forked_run = wandb.init(
    project="your_project_name",
    entity="your_entity_name",
    fork_from=f"{original_run.id}?_step=200",
)
```

## フォークした run から続行する
フォークされた run を初期化した後、新しい run にログを継続することができます。同じメトリクスをログして継続性を保ち、新しいメトリクスも追加することができます：

```python
import wandb
import math

# 最初の run を初期化し、いくつかのメトリクスをログ
run1 = wandb.init("your_project_name", entity="your_entity_name")
for i in range(300):
    run1.log({"metric": i})
run1.finish()

# 特定のステップで最初の run からフォークし、ステップ200からメトリクスをログ
run2 = wandb.init(
    "your_project_name", entity="your_entity_name", fork_from=f"{run1.id}?_step=200"
)

# 新しい run でログを続ける
# 最初のいくつかのステップでは、run1 からそのままメトリクスをログ
# ステップ250以降では、急激なパターンをログ
for i in range(200, 300):
    if i < 250:
        run2.log({"metric": i})  # スパイクなしでrun1からログを続ける
    else:
        # ステップ250から急激な振る舞いを導入
        subtle_spike = i + (2 * math.sin(i / 3.0))  # さりげないスパイクパターンを適用
        run2.log({"metric": subtle_spike})
    # すべてのステップで新しいメトリクスも追加でログ
    run2.log({"additional_metric": i * 1.1})
run2.finish()
```