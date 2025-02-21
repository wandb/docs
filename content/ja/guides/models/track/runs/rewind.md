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
run を巻き戻すオプションは、プライベートプレビュー版です。この機能へのアクセスをリクエストするには、W&B Support（support@wandb.com）までご連絡ください。

W&B は現在、以下をサポートしていません。
* **ログの巻き戻し**：ログは新しい run セグメントでリセットされます。
* **システムメトリクスの巻き戻し**：W&B は巻き戻しポイント後の新しいシステムメトリクスのみをログに記録します。
* **Artifact の関連付け**：W&B は Artifact を、それらを生成するソース run に関連付けます。
{{% /alert %}}

{{% alert %}}
* run を巻き戻すには、[W&B Python SDK](https://pypi.org/project/wandb/) バージョン >= `0.17.1` が必要です。
* 単調増加するステップを使用する必要があります。[`define_metric()`]({{< relref path="/ref/python/run#define_metric" lang="ja" >}}) で定義された非単調なステップは、run の履歴とシステムメトリクスの必要な時系列順序を妨げるため、使用できません。
{{% /alert %}}

run を巻き戻して、元のデータを失うことなく run の履歴を修正または変更します。さらに、run を巻き戻すと、その時点から新しいデータをログに記録できます。W&B は、新しくログに記録された履歴に基づいて、巻き戻した run のサマリーメトリクスを再計算します。これは、次の振る舞いを意味します。
- **履歴の切り捨て**：W&B は履歴を巻き戻しポイントまで切り捨て、新しいデータロギングを可能にします。
- **サマリーメトリクス**：新しくログに記録された履歴に基づいて再計算されます。
- **設定の保持**：W&B は元の設定を保持し、新しい設定をマージできます。

run を巻き戻すと、W&B は run の状態を指定されたステップにリセットし、元のデータを保持し、一貫した run ID を維持します。これは次のことを意味します。

- **run のアーカイブ**：W&B は元の run をアーカイブします。アーカイブされた run は、[**Run Overview**]({{< relref path="./#overview-tab" lang="ja" >}}) タブからアクセスできます。
- **Artifact の関連付け**：Artifact を、それらを生成する run に関連付けます。
- **不変の run ID**：正確な状態からのフォークの一貫性のために導入されました。
- **不変の run ID のコピー**：run 管理を改善するための不変の run ID をコピーするボタン。

{{% alert title="巻き戻しとフォークの互換性" %}}
フォークは巻き戻しを補完します。

run からフォークすると、W&B は特定のポイントで run から新しいブランチを作成し、異なるパラメータまたはモデルを試します。

run を巻き戻すと、W&B は run の履歴自体を修正または変更できます。
{{% /alert %}}

## run の巻き戻し

[`wandb.init()`]({{< relref path="/ref/python/init" lang="ja" >}}) で `resume_from` を使用して、run の履歴を特定のステップに「巻き戻し」ます。run の名前と、巻き戻すステップを指定します。

```python
import wandb
import math

# 最初の run を初期化し、いくつかのメトリクスをログに記録します
# your_project_name と your_entity_name に置き換えてください!
run1 = wandb.init(project="your_project_name", entity="your_entity_name")
for i in range(300):
    run1.log({"metric": i})
run1.finish()

# 特定のステップで最初の run から巻き戻し、ステップ 200 からメトリクスをログに記録します
run2 = wandb.init(project="your_project_name", entity="your_entity_name", resume_from=f"{run1.id}?_step=200")

# 新しい run でロギングを続行します
# 最初のいくつかのステップでは、run1 からメトリクスをそのままログに記録します
# ステップ 250 の後、スパイキーパターンをログに記録し始めます
for i in range(200, 300):
    if i < 250:
        run2.log({"metric": i, "step": i})  # スパイクなしで run1 からロギングを続行します
    else:
        # ステップ 250 からスパイキーな振る舞いを導入します
        subtle_spike = i + (2 * math.sin(i / 3.0))  # 微妙なスパイキーパターンを適用します
        run2.log({"metric": subtle_spike, "step": i})
    # さらに、すべてのステップで新しいメトリクスをログに記録します
    run2.log({"additional_metric": i * 1.1, "step": i})
run2.finish()
```

## アーカイブされた run の表示

run を巻き戻した後、W&B App UI でアーカイブされた run を調べることができます。アーカイブされた run を表示するには、次の手順に従います。

1. **Overview タブへのアクセス**：run のページの [**Overview タブ**]({{< relref path="./#overview-tab" lang="ja" >}}) に移動します。このタブは、run の詳細と履歴の包括的なビューを提供します。
2. **Forked From フィールドの検索**：**Overview** タブ内で、`Forked From` フィールドを見つけます。このフィールドは、再開の履歴をキャプチャします。**Forked From** フィールドには、ソース run へのリンクが含まれており、元の run にトレースバックし、巻き戻し履歴全体を理解できます。

`Forked From` フィールドを使用すると、アーカイブされた再開の ツリー を簡単にナビゲートし、各巻き戻しのシーケンスと起源に関する洞察を得ることができます。

## 巻き戻した run からのフォーク

巻き戻された run からフォークするには、`wandb.init()` の [**`fork_from`**]({{< relref path="/guides/models/track/runs/forking" lang="ja" >}}) 引数を使用し、ソース run ID と、フォーク元のソース run からのステップを指定します。

```python
import wandb

# 特定のステップから run をフォークします
forked_run = wandb.init(
    project="your_project_name",
    entity="your_entity_name",
    fork_from=f"{rewind_run.id}?_step=500",
)

# 新しい run でロギングを続行します
for i in range(500, 1000):
    forked_run.log({"metric": i*3})
forked_run.finish()
```