---
title: Fork a run
description: W&B の run をフォークする
menu:
  default:
    identifier: ja-guides-models-track-runs-forking
    parent: what-are-runs
---

{{% alert color="secondary" %}}
run をフォークする機能は、プライベートプレビュー版です。この機能へのアクセスをリクエストするには、W&B Support (support@wandb.com) までご連絡ください。
{{% /alert %}}

既存の W&B の run から「フォーク」するには、[`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ja" >}}) で run を初期化する際に `fork_from` を使用します。run からフォークすると、W&B はソース run の `run ID` と `step` を使用して新しい run を作成します。

run をフォークすると、元の run に影響を与えることなく、実験の特定の時点から異なるパラメータまたは model を探索できます。

{{% alert %}}
* run のフォークには、[`wandb`](https://pypi.org/project/wandb/) SDK バージョン >= 0.16.5 が必要です。
* run のフォークには、単調増加する step が必要です。[`define_metric()`]({{< relref path="/ref/python/run#define_metric" lang="ja" >}}) で定義された非単調な step を使用してフォークポイントを設定することはできません。これは、run の履歴とシステムメトリクスの本質的な時間的順序が崩れるためです。
{{% /alert %}}

## フォークされた run を開始する

run をフォークするには、[`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ja" >}}) で `fork_from` 引数を使用し、フォーク元のソース `run ID` とソース run の `step` を指定します。

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

### イミュータブル run ID を使用する

イミュータブル run ID を使用すると、特定の run への一貫性のある不変の参照を確保できます。ユーザーインターフェースからイミュータブル run ID を取得するには、次の手順に従います。

1.  **Overview タブにアクセス:** ソース run のページの [**Overview タブ**]({{< relref path="./#overview-tab" lang="ja" >}}) に移動します。

2.  **イミュータブル Run ID をコピー:** **Overview** タブの右上隅にある `...` メニュー (三点リーダー) をクリックします。ドロップダウンメニューから `イミュータブル Run ID をコピー` オプションを選択します。

これらの手順に従うことで、run への安定した不変の参照が得られ、run のフォークに使用できます。

## フォークされた run から続行する
フォークされた run を初期化した後、新しい run へのログ記録を続行できます。継続性のために同じメトリクスをログに記録し、新しいメトリクスを導入できます。

たとえば、次のコード例は、最初に run をフォークする方法と、次に 200 のトレーニング step から始まるフォークされた run にメトリクスをログに記録する方法を示しています。

```python
import wandb
import math

# 最初の run を初期化し、いくつかのメトリクスをログに記録する
run1 = wandb.init("your_project_name", entity="your_entity_name")
for i in range(300):
    run1.log({"metric": i})
run1.finish()

# 特定の step で最初の run からフォークし、step 200 からメトリックをログに記録する
run2 = wandb.init(
    "your_project_name", entity="your_entity_name", fork_from=f"{run1.id}?_step=200"
)

# 新しい run でログ記録を続行する
# 最初の数 step では、run1 からのメトリックをそのままログに記録する
# step 250 以降は、スパイキーパターンをログに記録し始める
for i in range(200, 300):
    if i < 250:
        run2.log({"metric": i})  # スパイクなしで run1 からのログ記録を続行する
    else:
        # step 250 からスパイキーな振る舞いを導入する
        subtle_spike = i + (2 * math.sin(i / 3.0))  # 微妙なスパイキーパターンを適用する
        run2.log({"metric": subtle_spike})
    # さらに、すべての step で新しいメトリックをログに記録する
    run2.log({"additional_metric": i * 1.1})
run2.finish()
```

{{% alert title="巻き戻しとフォークの互換性" %}}
フォークは、run の管理と実験においてより柔軟性を提供することにより、[`rewind`]({{< relref path="/guides/models/track/runs/rewind" lang="ja" >}}) を補完します。

run からフォークすると、W&B は特定の時点で run から新しいブランチを作成し、異なるパラメータまたは model を試します。

run を巻き戻すと、W&B では run の履歴自体を修正または変更できます。
{{% /alert %}}
