---
title: run をフォークする
description: W&B run をフォークする
menu:
  default:
    identifier: ja-guides-models-track-runs-forking
    parent: what-are-runs
---

{{% alert color="secondary" %}}
run をフォークする機能はプライベートプレビューです。この機能へのアクセスをリクエストするには、support@wandb.com まで W&B サポートにお問い合わせください。
{{% /alert %}}

run を初期化する際に `fork_from` を使用して、既存の W&B run から"フォーク"します。run をフォークすると、W&B はソース run の `run ID` と `step` を使用して新しい run を作成します。

run をフォークすることで、元の run に影響を与えずに、実験の特定のポイントから異なるパラメータやモデルを探索することができます。

{{% alert %}}
* run のフォークには [`wandb`](https://pypi.org/project/wandb/) SDK バージョン >= 0.16.5 が必要です
* run のフォークには、単調に増加するステップが必要です。[`define_metric()`]({{< relref path="/ref/python/run#define_metric" lang="ja" >}})で定義された非単調なステップを使用してフォークポイントを設定することはできません。これは、run 履歴およびシステムメトリクスの重要な時間的順序を乱すためです。
{{% /alert %}}

## フォークされた run を開始する

run をフォークするには、[`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ja" >}})で `fork_from` 引数を使用し、フォーク元としてのソース `run ID` と `step` を指定します:

```python
import wandb

# 後でフォークするための run を初期化
original_run = wandb.init(project="your_project_name", entity="your_entity_name")
# ... トレーニングやログを実行 ...
original_run.finish()

# 特定のステップから run をフォーク
forked_run = wandb.init(
    project="your_project_name",
    entity="your_entity_name",
    fork_from=f"{original_run.id}?_step=200",
)
```

### 不変の run ID を使用する

不変の run ID を使用して、特定の run への一貫性のある変更不可能な参照を保証します。ユーザーインターフェースから不変の run ID を取得するには、次の手順に従います:

1. **Overview タブにアクセスする:** ソース run のページで [**Overview タブ**]({{< relref path="./#overview-tab" lang="ja" >}}) に移動します。

2. **不変の Run ID をコピーする:** **Overview** タブの右上隅にある `...` メニュー（三点ドット）をクリックします。ドロップダウンメニューから `Copy Immutable Run ID` オプションを選択します。

これらの手順を追うことで、フォークされた run に使用できる安定した変更不可能な run への参照を得ることができます。

## フォークされた run から続行する
フォークされた run を初期化した後、新しい run にログを続行することができます。同じメトリクスをログすることで連続性を持たせ、新しいメトリクスを導入することも可能です。

例えば、次のコード例では、最初に run をフォークし、次にトレーニングステップ 200 からフォークされた run にメトリクスをログする方法を示しています:

```python
import wandb
import math

# 最初の run を初期化し、いくつかのメトリクスをログ
run1 = wandb.init("your_project_name", entity="your_entity_name")
for i in range(300):
    run1.log({"metric": i})
run1.finish()

# 特定のステップから最初の run をフォークし、ステップ 200 からメトリクスをログ
run2 = wandb.init(
    "your_project_name", entity="your_entity_name", fork_from=f"{run1.id}?_step=200"
)

# 新しい run でログを続行
# 最初のいくつかのステップで、run1 からそのままメトリクスをログ
# ステップ 250 以降、スパイキーなパターンをログする
for i in range(200, 300):
    if i < 250:
        run2.log({"metric": i})  # スパイクなしで run1 からログを続行
    else:
        # ステップ 250 からスパイキーな振る舞いを導入
        subtle_spike = i + (2 * math.sin(i / 3.0))  # 微細なスパイキーパターンを適用
        run2.log({"metric": subtle_spike})
    # さらに、すべてのステップで新しいメトリクスをログ
    run2.log({"additional_metric": i * 1.1})
run2.finish()
```

{{% alert title="巻き戻しとフォークの互換性" %}}
フォークは、あなたの run を管理し実験する上でより多くの柔軟性を提供することにより、[`巻き戻し`]({{< relref path="/guides/models/track/runs/rewind" lang="ja" >}})を補完します。

run をフォークする際、W&B は特定のポイントで run から新しいブランチを作成し、異なるパラメータやモデルを試みることができます。

run を巻き戻す際、W&B は run 履歴自体を修正または変更することができます。
{{% /alert %}}