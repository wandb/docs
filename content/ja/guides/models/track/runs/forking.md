---
title: run をフォークする
description: W&B の run をフォークする
menu:
  default:
    identifier: ja-guides-models-track-runs-forking
    parent: what-are-runs
---

{{% alert color="secondary" %}}
run をフォークする機能はプライベートプレビュー中です。この機能への アクセス を希望する場合は W&B サポート（support@wandb.com）までご連絡ください。
{{% /alert %}}

[`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init" lang="ja" >}}) で run を初期化するときに `fork_from` を使うと、既存の W&B の run から「フォーク」できます。run からフォークすると、W&B は元の run の `run ID` と `step` を使って新しい run を作成します。

run をフォークすると、元の run に影響を与えずに、実験の特定の時点から異なるパラメータやモデルを試すことができます。

{{% alert %}}
* run のフォークには [`wandb`](https://pypi.org/project/wandb/) SDK バージョン >= 0.16.5 が必要です
* run のフォークには単調増加する step が必要です。[`define_metric()`]({{< relref path="/ref/python/sdk/classes/run#define_metric" lang="ja" >}}) で定義した非単調な step をフォークポイントの設定には使えません。run の履歴やシステムメトリクスの本質的な時系列順序が崩れてしまうためです。
{{% /alert %}}


## フォークした run を開始する

run をフォークするには、[`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init" lang="ja" >}}) の `fork_from` 引数を使い、元の run の `run ID` とフォーク元となる `step` を指定します:

```python
import wandb

# 後でフォークするための run を初期化
original_run = wandb.init(project="your_project_name", entity="your_entity_name")
# ... トレーニングやログを実行 ...
original_run.finish()

# 特定の step から run をフォーク
forked_run = wandb.init(
    project="your_project_name",
    entity="your_entity_name",
    fork_from=f"{original_run.id}?_step=200",
)
```

### 不変の run ID を使う

不変の run ID を使うと、特定の run への参照を一貫して変更不能な形で保持できます。ユーザーインターフェースから不変の run ID を取得するには次の手順に従ってください:

1. **Overview タブに アクセス:** 元の run のページで [**Overview** タブ]({{< relref path="./#overview-tab" lang="ja" >}}) に移動します。

2. **Immutable Run ID をコピー:** **Overview** タブ右上の `...` メニュー（三点リーダー）をクリックし、ドロップダウンから `Copy Immutable Run ID` を選択します。

これらの手順に従うことで、run をフォークする際に使用できる、安定して変わらない参照を得られます。

## フォークした run から続ける
フォークした run を初期化したら、新しい run へログを継続できます。継続性のために同じメトリクスをログしても、新しいメトリクスを追加しても構いません。 

例えば次のコード例は、まず run をフォークし、その後 step 200 からフォークした run にメトリクスをログする方法を示しています:

```python
import wandb
import math

# 最初の run を初期化し、いくつかのメトリクスをログする
run1 = wandb.init("your_project_name", entity="your_entity_name")
for i in range(300):
    run1.log({"metric": i})
run1.finish()

# 最初の run の特定の step からフォークし、step 200 からメトリクスのログを開始
run2 = wandb.init(
    "your_project_name", entity="your_entity_name", fork_from=f"{run1.id}?_step=200"
)

# 新しい run でログを継続
# 最初の数 step は run1 の値をそのままログ
# step 250 以降はスパイク状のパターンをログ
for i in range(200, 300):
    if i < 250:
        run2.log({"metric": i})  # スパイクなしで run1 から継続してログ
    else:
        # step 250 からスパイク状の振る舞いを導入
        subtle_spike = i + (2 * math.sin(i / 3.0))  # 控えめなスパイク状パターンを適用
        run2.log({"metric": subtle_spike})
    # さらに全ての step で新しいメトリクスもログ
    run2.log({"additional_metric": i * 1.1})
run2.finish()
```

{{% alert title="Rewind と フォークの互換性" %}}
フォークは [`rewind`]({{< relref path="/guides/models/track/runs/rewind" lang="ja" >}}) を補完し、run の管理や実験をより柔軟にします。 

run からフォークすると、W&B は特定の時点で run から分岐（ブランチ）を作成し、異なるパラメータやモデルを試せるようにします。 

run を rewind すると、W&B は run の履歴そのものを修正または変更できるようにします。
{{% /alert %}}