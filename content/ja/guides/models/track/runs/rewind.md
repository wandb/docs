---
title: run を巻き戻す
description: 巻き戻し
menu:
  default:
    identifier: ja-guides-models-track-runs-rewind
    parent: what-are-runs
---

# run をリワインドする
{{% alert color="secondary" %}}
run のリワインド機能はプライベートプレビュー中です。この機能への アクセス を希望する場合は、support@wandb.com まで W&B サポートにご連絡ください。

W&B では現在、以下はサポートされていません:
* **ログのリワインド**: 新しい run セグメントではログがリセットされます。
* **システムメトリクスのリワインド**: リワインドポイント以降の新しいシステムメトリクスのみを W&B は記録します。
* **Artifact の関連付け**: Artifact は作成元の run に関連付けられます。
{{% /alert %}}

{{% alert %}}
* run をリワインドするには、[W&B Python SDK](https://pypi.org/project/wandb/) バージョン `0.17.1` 以上が必要です。
* 単調増加するステップを使用する必要があります。[`define_metric()`]({{< relref path="/ref/python/sdk/classes/run#define_metric" lang="ja" >}}) で定義された非単調ステップには対応していません。これは run の履歴やシステムメトリクスの時系列順序を崩すためです。
{{% /alert %}}

run の履歴を修正したり変更したりする際、元のデータを失うことなく run をリワインドできます。さらに、リワインドした時点から新しいデータのログも可能です。W&B は、新たに記録された履歴に基づいてリワインドした run のサマリーメトリクスを再計算します。これにより、以下のような挙動になります:
- **履歴の切り捨て**: W&B はリワインドポイントまで履歴を切り捨て、その後の新しいデータログを可能にします。
- **サマリーメトリクス**: 新たに記録した履歴に基づき再計算されます。
- **設定の保持**: 元の設定はそのまま保持され、新たな設定をマージすることも可能です。

run をリワインドすると、指定したステップの状態まで run をリセットし、元のデータを保持したまま一貫性のある run ID を維持します。つまり次のようになります:

- **run のアーカイブ**: W&B は元の run をアーカイブします。run へは [Run Overview]({{< relref path="./#overview-tab" lang="ja" >}}) タブから アクセス できます。
- **Artifact の関連付け**: Artifact はそれらを生成した run に関連付けられます。
- **不変の run ID**: 正確な状態での fork を可能にするため、run ID は不変になります。
- **run ID のコピー**: 不変の run ID をコピーできるボタンで run 管理が容易になります。

{{% alert title="リワインドと fork の互換性" %}}
フォーク機能はリワインド機能を補完します。

run から fork すると、特定の時点から run を分岐させて別のパラメータやモデルを試せます。

run をリワインドすると、run の履歴自体を修正・変更することができます。
{{% /alert %}}



## run をリワインドする

`resume_from` を [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init" lang="ja" >}}) で使用し、特定のステップまで run の履歴を「リワインド」します。run の名前と、リワインドしたいステップを指定してください:

```python
import wandb
import math

# 最初の run を初期化してメトリクスを記録
# your_project_name と your_entity_name を置き換えてください！
run1 = wandb.init(project="your_project_name", entity="your_entity_name")
for i in range(300):
    run1.log({"metric": i})
run1.finish()

# 最初の run の特定ステップからリワインドし、200 ステップ目から metric のログを再開
run2 = wandb.init(project="your_project_name", entity="your_entity_name", resume_from=f"{run1.id}?_step=200")

# 新しい run でログを継続
# 最初の数ステップは run1 の metric をそのまま記録
# 250 ステップ以降はスパイキーなパターンでログ
for i in range(200, 300):
    if i < 250:
        run2.log({"metric": i, "step": i})  # run1 からスパイクなしでログを継続
    else:
        # 250 ステップ目からスパイキーな振る舞いを導入
        subtle_spike = i + (2 * math.sin(i / 3.0))  # さりげないスパイクパターンを適用
        run2.log({"metric": subtle_spike, "step": i})
    # すべてのステップで追加のメトリクスもログ
    run2.log({"additional_metric": i * 1.1, "step": i})
run2.finish()
```

## アーカイブされた run を閲覧する

run をリワインドした後は、W&B App UI でアーカイブ済みの run を確認できます。以下の手順でアーカイブ済み run を閲覧してください:

1. **Overview タブに アクセス する:** run のページにある [**Overview** タブ]({{< relref path="./#overview-tab" lang="ja" >}}) に移動します。このタブで run の詳細や履歴を確認できます。
2. **Forked From フィールドを探す:** **Overview** タブ内に `Forked From` フィールドが表示されます。このフィールドには再開の履歴が記録されており、**Forked From** からソースとなった run へのリンクが含まれています。これにより元の run への遡りや、リワインド履歴の全体把握が可能です。

`Forked From` フィールドを活用することで、アーカイブされた再開のツリー構造を簡単に辿り、それぞれのリワインドの順序や由来を把握できます。

## リワインドした run から fork する

リワインド済みの run から fork するには、[`fork_from`]({{< relref path="/guides/models/track/runs/forking" lang="ja" >}}) 引数を `wandb.init()` で指定し、ソースとなる run の ID と fork したいステップを指定します:

```python 
import wandb

# run を特定ステップから fork
forked_run = wandb.init(
    project="your_project_name",
    entity="your_entity_name",
    fork_from=f"{rewind_run.id}?_step=500",
)

# 新しい run でログを継続
for i in range(500, 1000):
    forked_run.log({"metric": i*3})
forked_run.finish()
```