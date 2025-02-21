---
title: Add labels to runs with tags
menu:
  default:
    identifier: ja-guides-models-track-runs-tags
    parent: what-are-runs
---

特定の機能をラベル付けするためにタグを追加して、ログに記録されたメトリクスやアーティファクトのデータからは明らかでないrunを識別できます。

たとえば、runのモデルが `in_production` であること、runが `preemptible` であること、このrunが `baseline` を表すことなどを示すために、runにタグを追加できます。

## 1つまたは複数のrunにタグを追加する

プログラムで、またはインタラクティブにrunにタグを追加します。

ユースケースに基づいて、ニーズに最適なタブを以下から選択してください。

{{< tabpane text=true >}}
    {{% tab header="W&B Python SDK" %}}
runの作成時にタグを追加できます。

```python
import wandb

run = wandb.init(
  entity="entity",
  project="<project-name>",
  tags=["tag1", "tag2"]
)
```

runを初期化した後にタグを更新することもできます。たとえば、次のコードスニペットは、特定のメトリクスがあらかじめ定義された閾値を超えた場合にタグを更新する方法を示しています。

```python
import wandb

run = wandb.init(
  entity="entity", 
  project="capsules", 
  tags=["debug"]
  )

# python logic to train model

if current_loss < threshold:
    run.tags = run.tags + ("release_candidate",)
```
    {{% /tab %}}
    {{% tab header="Public API" %}}
runを作成した後、[Public API]({{< relref path="/guides/models/track/public-api-guide.md" lang="ja" >}})を使用してタグを更新できます。例：

```python
run = wandb.Api().run("{entity}/{project}/{run-id}")
run.tags.append("tag1")  # you can choose tags based on run data here
run.update()
```
    {{% /tab %}}
    {{% tab header="Project page" %}}
この方法は、同じタグを多数のRunsに適用する場合に最適です。

1. プロジェクト の Workspace に移動します。
2. プロジェクトのサイドバーから **Runs** を選択します。
3. テーブルから1つまたは複数のRunsを選択します。
4. 1つまたは複数のRunsを選択したら、テーブルの上の **Tag** ボタンを選択します。
5. 追加するタグを入力し、**Create new tag** チェックボックスを選択して、タグを追加します。
    {{% /tab %}}
    {{% tab header="Run page" %}}
この方法は、単一のRunに手動でタグを適用する場合に最適です。

1. プロジェクト の Workspace に移動します。
2. プロジェクトの Workspace 内のRunsのリストからRunを選択します。
3. プロジェクトのサイドバーから **Overview** を選択します。
4. **Tags** の横にある灰色のプラスアイコン（**+**）ボタンを選択します。
5. 追加するタグを入力し、テキストボックスの下にある **Add** を選択して、新しいタグを追加します。
    {{% /tab %}}
{{< /tabpane >}}

## 1つまたは複数のRunsからタグを削除する

タグは、W&B App UIを使用してRunsから削除することもできます。

{{< tabpane text=true >}}
{{% tab header="Project page"%}}
この方法は、多数のRunsからタグを削除する場合に最適です。

1. プロジェクトのRunサイドバーで、右上にあるテーブルアイコンを選択します。これにより、サイドバーが展開されて完全なRunテーブルになります。
2. テーブル内のRunの上にマウスを置くと、左側にチェックボックスが表示されるか、ヘッダー行でチェックボックスを探して、すべてのRunsを選択します。
3. チェックボックスを選択して、一括アクションを有効にします。
4. タグを削除するRunsを選択します。
5. Runsの行の上にある **Tag** ボタンを選択します。
6. タグの横にあるチェックボックスを選択して、Runから削除します。

{{% /tab %}}
{{% tab header="Run page"%}}

1. Runページの左側のサイドバーで、一番上の **Overview** タブを選択します。Runのタグがここに表示されます。
2. タグの上にマウスを置き、「x」を選択してRunから削除します。

{{% /tab %}}
{{< /tabpane >}}
