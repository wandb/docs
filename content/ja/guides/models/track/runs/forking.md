---
title: run をフォークする
description: W&B run をフォークする
menu:
  default:
    identifier: ja-guides-models-track-runs-forking
    parent: what-are-runs
---

{{% alert color="secondary" %}}
run のフォーク機能はプライベートプレビュー中です。この機能へのアクセスを希望される場合は、support@wandb.com まで W&B サポートにご連絡ください。
{{% /alert %}}

[`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init" lang="ja" >}}) で run を初期化する際に `fork_from` を使うと、既存の W&B run から「フォーク」できます。run をフォークすると、W&B は元の run の `run ID` と `step` を使って新しい run を作成します。

run をフォークすることで、オリジナルの run へ影響を与えることなく、実験の特定のポイントからパラメータやモデルの異なるバリエーションを試すことができます。

{{% alert %}}
* run のフォークには [`wandb`](https://pypi.org/project/wandb/) SDK バージョン 0.16.5 以上が必要です
* run のフォークには、step が単調増加であることが求められます。[`define_metric()`]({{< relref path="/ref/python/sdk/classes/run#define_metric" lang="ja" >}}) で定義した非単調な step をフォークポイントとして使うことはできません。これは、run の履歴およびシステムメトリクスの本質的な時系列順序が乱れるためです。
{{% /alert %}}


## フォークした run を開始する

run をフォークするには、[`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init" lang="ja" >}}) の `fork_from` 引数を利用して、フォーク元の `run ID` および `step` を指定します:

```python
import wandb

# 後でフォークするための run を初期化
original_run = wandb.init(project="your_project_name", entity="your_entity_name")
# ... トレーニングやログの処理 ...
original_run.finish()

# 特定の step から run をフォーク
forked_run = wandb.init(
    project="your_project_name",
    entity="your_entity_name",
    fork_from=f"{original_run.id}?_step=200",
)
```

### 不変の run ID を使用する

特定の run への参照が一貫して変化しないように、不変の run ID（immutable run ID） を使用してください。ユーザーインターフェースから不変の run ID を取得する手順は以下の通りです：

1. **Overview タブへアクセス:** フォーク元の run のページで、[**Overview**タブ]({{< relref path="./#overview-tab" lang="ja" >}}) に移動します。

2. **不変の run ID をコピー:** **Overview**タブの右上にある `...` メニュー（三点リーダー）をクリックし、ドロップダウンメニューから `Copy Immutable Run ID` オプションを選択します。

これらの手順により、フォークした run のために安定して変化しない参照を取得できます。

## フォークした run から継続する
フォークした run を初期化した後は、新しい run へのログ記録を続けることができます。継続性のために同じメトリクスを記録したり、新しいメトリクスを追加することも可能です。

たとえば、次のコード例では、まず run をフォークし、step 200 から forked run にメトリクスをログする方法を示しています:

```python
import wandb
import math

# 最初の run を初期化し、一部のメトリクスをログ
run1 = wandb.init("your_project_name", entity="your_entity_name")
for i in range(300):
    run1.log({"metric": i})
run1.finish()

# 最初の run の特定の step からフォークし、その step からメトリクスをログ
run2 = wandb.init(
    "your_project_name", entity="your_entity_name", fork_from=f"{run1.id}?_step=200"
)

# 新しい run でログを継続
# 最初のいくつかのステップでは、run1 の値をそのままログ
# step 250 より後はスパイキーなパターンを記録
for i in range(200, 300):
    if i < 250:
        run2.log({"metric": i})  # run1 からスパイクなしのメトリクスを継続記録
    else:
        # step 250 以降でスパイキーな振る舞いを追加
        subtle_spike = i + (2 * math.sin(i / 3.0))  # 微妙なスパイキーなパターンを反映
        run2.log({"metric": subtle_spike})
    # すべてのステップで新しいメトリクスも追加で記録
    run2.log({"additional_metric": i * 1.1})
run2.finish()
```

{{% alert title="Rewind とフォークの互換性" %}}
フォーク機能は、[`rewind`]({{< relref path="/guides/models/track/runs/rewind" lang="ja" >}}) 機能と組み合わせることで、run の管理や様々な実験を柔軟に行えるようにします。

run をフォークする場合、W&B は特定のポイントで run から新しいブランチを作り、異なるパラメータやモデルを試せるようにします。

run を rewind する場合、W&B は run の履歴自体を修正・訂正できるようにします。
{{% /alert %}}