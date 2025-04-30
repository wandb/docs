---
title: run を巻き戻す
description: 巻き戻す
menu:
  default:
    identifier: ja-guides-models-track-runs-rewind
    parent: what-are-runs
---

# runを巻き戻す
{{% alert color="secondary" %}}
runを巻き戻すオプションはプライベートプレビューです。この機能へのアクセスをリクエストするには、W&Bサポート(support@wandb.com)までお問い合わせください。

現在、W&Bがサポートしていないもの:
* **ログの巻き戻し**: 新しいrunセグメントでログがリセットされます。
* **システムメトリクスの巻き戻し**: W&Bは巻き戻しポイントの後にのみ新しいシステムメトリクスをログします。
* **アーティファクトの関連付け**: W&Bは生成されたアーティファクトをそのソースrunに関連付けます。
{{% /alert %}}

{{% alert %}}
* runを巻き戻すには、[W&B Python SDK](https://pypi.org/project/wandb/) バージョン >= `0.17.1` が必要です。
* 単調増加するステップを使用する必要があります。run履歴とシステムメトリクスの必要な時間順序を乱すため、[`define_metric()`]({{< relref path="/ref/python/run#define_metric" lang="ja" >}})で定義された非単調ステップを使用することはできません。
{{% /alert %}}

runを巻き戻して、元のデータを失うことなくrunの履歴を修正または変更します。さらに、runを巻き戻すと、その時点から新しいデータをログすることができます。W&Bは、新しい履歴に基づく巻き戻し対象のrunのサマリーメトリクスを再計算します。これは以下の振る舞いを意味します:
- **履歴の切断**: W&Bは巻き戻しポイントまで履歴を切断し、新しいデータのログを可能にします。
- **サマリーメトリクス**: 新しい履歴に基づいて再計算されます。
- **設定の保持**: W&Bは元の設定を保持し、新しい設定をマージすることができます。

runを巻き戻すと、W&Bは指定されたステップにrunの状態をリセットし、元のデータを保持し、一貫したrun IDを維持します。これは次のことを意味します:

- **runのアーカイブ**: W&Bは元のrunをアーカイブします。runは[**Run Overview**]({{< relref path="./#overview-tab" lang="ja" >}}) タブからアクセス可能です。
- **アーティファクトの関連付け**: アーティファクトを生成するrunと関連付けます。
- **不変のrun ID**: 正確な状態からフォークするための一貫性が導入されます。
- **不変のrun IDをコピー**: run管理を改善するために不変のrun IDをコピーするボタンがあります。

{{% alert title="巻き戻しとフォークの互換性" %}}
フォークは巻き戻しと補完し合います。

runからフォークすると、W&Bは特定のポイントでrunを分岐させて異なるパラメータやモデルを試します。

runを巻き戻すと、W&Bはrun履歴そのものを修正または変更することを可能にします。
{{% /alert %}}



## runを巻き戻す

`resume_from`を[`wandb.init()`]({{< relref path="/ref/python/init" lang="ja" >}})と共に使用して、runの履歴を特定のステップまで「巻き戻し」ます。runの名前と巻き戻すステップを指定します:

```python
import wandb
import math

# 最初のrunを初期化していくつかのメトリクスをログする
# your_project_nameとyour_entity_nameを置き換えてください!
run1 = wandb.init(project="your_project_name", entity="your_entity_name")
for i in range(300):
    run1.log({"metric": i})
run1.finish()

# 最初のrunの特定のステップから巻き戻してステップ200からメトリクスをログする
run2 = wandb.init(project="your_project_name", entity="your_entity_name", resume_from=f"{run1.id}?_step=200")

# 新しいrunでログを続ける
# 最初のいくつかのステップでは、run1からメトリクスをそのままログする
# ステップ250以降、尖ったパターンのログを開始する
for i in range(200, 300):
    if i < 250:
        run2.log({"metric": i, "step": i})  # スパイクなしでrun1からログを続行
    else:
        # ステップ250から尖った振る舞いを導入
        subtle_spike = i + (2 * math.sin(i / 3.0))  # subtleなスパイクパターンを適用
        run2.log({"metric": subtle_spike, "step": i})
    # さらに新しいメトリクスをすべてのステップでログ
    run2.log({"additional_metric": i * 1.1, "step": i})
run2.finish()
```

## アーカイブされたrunを見る

runを巻き戻した後、W&B App UIでアーカイブされたrunを探索できます。以下のステップに従ってアーカイブされたrunを表示します:

1. **Overviewタブにアクセスする**: runのページで[**Overviewタブ**]({{< relref path="./#overview-tab" lang="ja" >}})に移動します。このタブはrunの詳細と履歴を包括的に表示します。
2. **Forked Fromフィールドを見つける**: **Overview**タブ内で、`Forked From`フィールドを見つけます。このフィールドは再開の履歴を記録します。**Forked From**フィールドにはソースrunへのリンクが含まれており、オリジナルのrunに遡り、全体の巻き戻し履歴を理解することができます。

`Forked From`フィールドを使用することで、アーカイブされた再開のツリーを簡単にナビゲートし、それぞれの巻き戻しの順序と起源についての洞察を得ることができます。

## 巻き戻されたrunからフォークする

巻き戻されたrunからフォークするには、`wandb.init()`の[**`fork_from`**]({{< relref path="/guides/models/track/runs/forking" lang="ja" >}})引数を使用し、ソースrun IDとフォークするソースrunのステップを指定します:

```python 
import wandb

# 特定のステップからrunをフォークする
forked_run = wandb.init(
    project="your_project_name",
    entity="your_entity_name",
    fork_from=f"{rewind_run.id}?_step=500",
)

# 新しいrunでログを続ける
for i in range(500, 1000):
    forked_run.log({"metric": i*3})
forked_run.finish()
```