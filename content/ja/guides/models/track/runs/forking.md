---
title: run をフォークする
description: W&B run をフォークする
menu:
  default:
    identifier: forking
    parent: what-are-runs
---

{{% alert color="secondary" %}}
run のフォーク機能は現在プライベートプレビューです。この機能の利用をご希望の場合は、support@wandb.com まで W&B サポートへご連絡ください。
{{% /alert %}}

[`wandb.init()`]({{< relref "/ref/python/sdk/functions/init" >}}) を使って run を初期化する際に `fork_from` を指定すると、既存の W&B run から「フォーク」できます。run をフォークすると、W&B は元の run の `run ID` と `step` を使って新しい run を作成します。

run をフォークすることで、実験の特定のポイントから元の run に影響を与えずに異なるパラメータやモデルを試行できるようになります。

{{% alert %}}
* run フォークには [`wandb`](https://pypi.org/project/wandb/) SDK バージョン >= 0.16.5 が必要です。
* run のフォークは step が単調増加していることが必要です。[`define_metric()`]({{< relref "/ref/python/sdk/classes/run#define_metric" >}}) で非単調な step を定義した場合、そのポイントでフォークすることはできません。これは run 履歴とシステムメトリクスの本来の時系列順序が失われてしまうためです。
{{% /alert %}}


## フォークした run を開始する

run をフォークするには、[`wandb.init()`]({{< relref "/ref/python/sdk/functions/init" >}}) の `fork_from` 引数を使い、 フォーク元 run の `run ID` とそこから始めたい `step` を指定します。

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

### 変更不可の run ID を使用する

特定の run を一意に、かつ変更されない状態で参照するには変更不可（immutable）な run ID を使います。ユーザーインターフェースから変更不可の run ID を取得する手順は以下の通りです。

1. **Overviewタブにアクセス:** フォーク元 run のページの [**Overview** タブ]({{< relref "./#overview-tab" >}}) に移動します。

2. **変更不可の Run ID をコピー:** **Overview** タブの右上にある `...` メニュー（三点リーダー）をクリックし、ドロップダウンメニューから `Copy Immutable Run ID` を選択します。

この操作により、その run を特定するための安定した参照（変更されない ID）を取得できます。これをフォークの際に利用します。

## フォークした run から継続する

フォークした run を初期化した後は、新しい run へのログ記録を継続できます。継続性のために同じメトリクスを記録しつつ、新たなメトリクスも追加可能です。

以下の例は、まず run をフォークし、その後トレーニングステップ 200 からフォーク先 run にメトリクスを記録する方法を示しています。

```python
import wandb
import math

# 最初の run を初期化し、一部メトリクスを記録
run1 = wandb.init("your_project_name", entity="your_entity_name")
for i in range(300):
    run1.log({"metric": i})
run1.finish()

# 最初の run を特定の step でフォークし、step 200 からメトリクスを記録
run2 = wandb.init(
    "your_project_name", entity="your_entity_name", fork_from=f"{run1.id}?_step=200"
)

# 新しい run でログを継続
# 最初の数ステップは run1 と同じ値を記録
# 250 ステップ以降でスパイクパターンを導入
for i in range(200, 300):
    if i < 250:
        run2.log({"metric": i})  # 250 までは run1 の値をそのまま記録
    else:
        # 250 ステップ以降でスパイクした振る舞いを導入
        subtle_spike = i + (2 * math.sin(i / 3.0))  # 微妙なスパイクパターン
        run2.log({"metric": subtle_spike})
    # すべてのステップで新たなメトリクスも追加で記録
    run2.log({"additional_metric": i * 1.1})
run2.finish()
```

{{% alert title="Rewindとフォークの互換性" %}}
フォークは [`rewind`]({{< relref "/guides/models/track/runs/rewind" >}}) と組み合わせることで、run の管理や実験により柔軟性をもたらします。

run をフォークすれば、特定の時点から異なるパラメータやモデルを試す新しいブランチを作ることができます。

一方で rewind では、run の履歴そのものを修正または訂正できます。
{{% /alert %}}