---
title: Rewind a run
description: 巻き戻し
menu:
  default:
    identifier: ja-guides-models-track-runs-rewind
    parent: what-are-runs
---

# run の巻き戻し
{{% alert color="secondary" %}}
run を巻き戻すオプションは、プライベートプレビュー版です。この機能へのアクセスをご希望の場合は、W&B Support（support@wandb.com）までご連絡ください。

W&B は現在、以下をサポートしていません。
* **ログの巻き戻し**: ログは新しい run セグメントでリセットされます。
* **システムメトリクスの巻き戻し**: W&B は、巻き戻しポイントより後の新しいシステムメトリクスのみを記録します。
* **Artifact の関連付け**: W&B は、Artifact をそれを生成するソース run に関連付けます。
{{% /alert %}}

{{% alert %}}
* run を巻き戻すには、[W&B Python SDK](https://pypi.org/project/wandb/) バージョン >= `0.17.1` が必要です。
* 単調増加するステップを使用する必要があります。[`define_metric()`]({{< relref path="/ref/python/run#define_metric" lang="ja" >}}) で定義された非単調なステップは、run の履歴とシステムメトリクスの必要な時系列順序を混乱させるため使用できません。
{{% /alert %}}

run を巻き戻して、元のデータを失うことなく、run の履歴を修正または変更します。さらに、run を巻き戻すと、その時点から新しいデータを記録できます。W&B は、新たに記録された履歴に基づいて、巻き戻した run のサマリーメトリクスを再計算します。これは、次の振る舞いを意味します。
- **履歴の切り捨て**: W&B は履歴を巻き戻しポイントまで切り捨て、新しいデータロギングを可能にします。
- **サマリーメトリクス**: 新たに記録された履歴に基づいて再計算されます。
- **設定の保持**: W&B は元の設定を保持し、新しい設定をマージできます。

run を巻き戻すと、W&B は run の状態を指定されたステップにリセットし、元のデータを保持し、一貫した run ID を維持します。これは次のことを意味します。

- **run のアーカイブ**: W&B は元の run をアーカイブします。アーカイブされた run は、[**Run Overview**]({{< relref path="./#overview-tab" lang="ja" >}}) タブからアクセスできます。
- **Artifact の関連付け**: Artifact をそれを生成する run に関連付けます。
- **不変の run ID**: 正確な状態からのフォークの一貫性のために導入されました。
- **不変の run ID のコピー**: run 管理を改善するために、不変の run ID をコピーするボタン。

{{% alert title="巻き戻しとフォークの互換性" %}}
フォークは巻き戻しを補完します。

run からフォークすると、W&B は特定の時点で run から新しいブランチを作成し、さまざまなパラメータや Models を試すことができます。

run を巻き戻すと、W&B を使用して run の履歴自体を修正または変更できます。
{{% /alert %}}

## run の巻き戻し

`resume_from` を使用して [`wandb.init()`]({{< relref path="/ref/python/init" lang="ja" >}}) を使用して、run の履歴を特定のステップに「巻き戻し」ます。巻き戻す run の名前とステップを指定します。

```python
import wandb
import math

# Initialize the first run and log some metrics
# Replace with your_project_name and your_entity_name!
run1 = wandb.init(project="your_project_name", entity="your_entity_name")
for i in range(300):
    run1.log({"metric": i})
run1.finish()

# Rewind from the first run at a specific step and log the metric starting from step 200
run2 = wandb.init(project="your_project_name", entity="your_entity_name", resume_from=f"{run1.id}?_step=200")

# Continue logging in the new run
# For the first few steps, log the metric as is from run1
# After step 250, start logging the spikey pattern
for i in range(200, 300):
    if i < 250:
        run2.log({"metric": i, "step": i})  # Continue logging from run1 without spikes
    else:
        # Introduce the spikey behavior starting from step 250
        subtle_spike = i + (2 * math.sin(i / 3.0))  # Apply a subtle spikey pattern
        run2.log({"metric": subtle_spike, "step": i})
    # Additionally log the new metric at all steps
    run2.log({"additional_metric": i * 1.1, "step": i})
run2.finish()
```

## アーカイブされた run の表示

run を巻き戻した後、W&B App UI でアーカイブされた run を調べることができます。アーカイブされた run を表示するには、次の手順に従います。

1. **Overview タブにアクセスする**: run のページの [**Overview タブ**]({{< relref path="./#overview-tab" lang="ja" >}}) に移動します。このタブには、run の詳細と履歴の包括的なビューが表示されます。
2. **Forked From フィールドを見つける**: **Overview** タブ内で、`Forked From` フィールドを見つけます。このフィールドは、再開の履歴をキャプチャします。**Forked From** フィールドには、ソース run へのリンクが含まれており、元の run にトレースバックして、巻き戻し履歴全体を理解できます。

`Forked From` フィールドを使用すると、アーカイブされた再開の ツリー を簡単にナビゲートし、各巻き戻しのシーケンスとオリジンに関する洞察を得ることができます。

## 巻き戻した run からフォークする

巻き戻した run からフォークするには、`wandb.init()` の [**`fork_from`**]({{< relref path="/guides/models/track/runs/forking" lang="ja" >}}) 引数を使用し、ソース run ID と、フォーク元のソース run からのステップを指定します。

```python
import wandb

# Fork the run from a specific step
forked_run = wandb.init(
    project="your_project_name",
    entity="your_entity_name",
    fork_from=f"{rewind_run.id}?_step=500",
)

# Continue logging in the new run
for i in range(500, 1000):
    forked_run.log({"metric": i*3})
forked_run.finish()
```
