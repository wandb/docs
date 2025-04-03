---
title: Fork a run
description: W&B の run をフォークする
menu:
  default:
    identifier: ja-guides-models-track-runs-forking
    parent: what-are-runs
---

{{% alert color="secondary" %}}
run をフォークする機能はプライベートプレビュー版です。この機能へのアクセスをご希望の場合は、W&B Support (support@wandb.com) までご連絡ください。
{{% /alert %}}

既存の W&B の run から「フォーク」するには、[`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ja" >}}) で run を初期化する際に `fork_from` を使用します。run からフォークすると、W&B はソース run の `run ID` と `step` を使用して新しい run を作成します。

run をフォークすると、元の run に影響を与えることなく、実験の特定の時点から異なるパラメータまたは model を調べることができます。

{{% alert %}}
* run をフォークするには、[`wandb`](https://pypi.org/project/wandb/) SDK バージョン 0.16.5 以上が必要です。
* run をフォークするには、単調増加する step が必要です。[`define_metric()`]({{< relref path="/ref/python/run#define_metric" lang="ja" >}}) で定義された非単調な step を使用してフォークポイントを設定することはできません。run の履歴とシステム メトリクスの本質的な時間順序が崩れるためです。
{{% /alert %}}

## フォークされた run を開始する

run をフォークするには、[`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ja" >}}) で `fork_from` 引数を使用し、フォーク元のソース `run ID` とソース run からの `step` を指定します。

```python
import wandb

# 後でフォークされる run を初期化する
original_run = wandb.init(project="your_project_name", entity="your_entity_name")
# ... トレーニングまたはログの記録を実行 ...
original_run.finish()

# 特定の step から run をフォークする
forked_run = wandb.init(
    project="your_project_name",
    entity="your_entity_name",
    fork_from=f"{original_run.id}?_step=200",
)
```

### イミュータブルな run ID を使用する

イミュータブルな run ID を使用して、特定の run への一貫性があり、変更されない参照を確保します。ユーザーインターフェースからイミュータブルな run ID を取得するには、次の手順に従います。

1. **Overviewタブにアクセスする:** ソース run のページの [**Overviewタブ**]({{< relref path="./#overview-tab" lang="ja" >}}) に移動します。

2. **イミュータブルな Run ID をコピーする:** **Overview** タブの右上隅にある `...` メニュー (3 つのドット) をクリックします。ドロップダウンメニューから [イミュータブルな Run ID をコピー] オプションを選択します。

これらの手順に従うことで、run への安定した変更されない参照が得られ、run のフォークに使用できます。

## フォークされた run から続行する
フォークされた run を初期化したら、新しい run へのログ記録を続行できます。継続性のために同じメトリクスをログに記録し、新しいメトリクスを導入できます。

たとえば、次のコード例は、最初に run をフォークし、次に 200 のトレーニング step から始まるフォークされた run にメトリクスをログに記録する方法を示しています。

```python
import wandb
import math

# 最初の run を初期化し、いくつかのメトリクスをログに記録する
run1 = wandb.init("your_project_name", entity="your_entity_name")
for i in range(300):
    run1.log({"metric": i})
run1.finish()

# 特定の step で最初の run からフォークし、step 200 から始まるメトリクスをログに記録する
run2 = wandb.init(
    "your_project_name", entity="your_entity_name", fork_from=f"{run1.id}?_step=200"
)

# 新しい run でログ記録を続行する
# 最初のいくつかの step では、run1 からメトリクスをそのままログに記録する
# Step 250 以降は、スパイキーパターンをログに記録し始める
for i in range(200, 300):
    if i < 250:
        run2.log({"metric": i})  # スパイクなしで run1 からログ記録を続行する
    else:
        # Step 250 からスパイキーな振る舞いを導入する
        subtle_spike = i + (2 * math.sin(i / 3.0))  # 微妙なスパイキーパターンを適用する
        run2.log({"metric": subtle_spike})
    # さらに、すべての step で新しいメトリクスをログに記録する
    run2.log({"additional_metric": i * 1.1})
run2.finish()
```

{{% alert title="巻き戻しとフォークの互換性" %}}
フォークは、run の管理と実験においてより柔軟性を提供することで、[`rewind`]({{< relref path="/guides/models/track/runs/rewind" lang="ja" >}}) を補完します。

run からフォークすると、W&B は特定のポイントで run から新しいブランチを作成し、異なるパラメータまたは model を試すことができます。

run を巻き戻すと、W&B は run の履歴自体を修正または変更できます。
{{% /alert %}}
