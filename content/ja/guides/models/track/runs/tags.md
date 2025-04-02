---
title: Add labels to runs with tags
menu:
  default:
    identifier: ja-guides-models-track-runs-tags
    parent: what-are-runs
---

ログに記録されたメトリクスや Artifact データからは明らかでない、特定の機能を持つ run にラベルを付けるために、タグを追加します。

例えば、run の model が `in_production` であること、run が `preemptible` であること、この run が `baseline` を表していることなどを表すタグを、run に追加できます。

## 1つまたは複数の run にタグを追加する

プログラムで、またはインタラクティブに、run にタグを追加します。

ユースケースに応じて、ニーズに最も適した以下のタブを選択してください。

{{< tabpane text=true >}}
    {{% tab header="W&B Python SDK" %}}
run の作成時にタグを追加できます。

```python
import wandb

run = wandb.init(
  entity="entity",
  project="<project-name>",
  tags=["tag1", "tag2"]
)
```

run を初期化した後にタグを更新することもできます。例えば、以下のコードスニペットは、特定のメトリクスが事前定義された閾値を超えた場合にタグを更新する方法を示しています。

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
run の作成後、[Public API]({{< relref path="/guides/models/track/public-api-guide.md" lang="ja" >}}) を使用してタグを更新できます。例：

```python
run = wandb.Api().run("{entity}/{project}/{run-id}")
run.tags.append("tag1")  # you can choose tags based on run data here
run.update()
```    
    {{% /tab %}}
    {{% tab header="Project page" %}}
この方法は、多数の run に同じタグを付けるのに最適です。

1. プロジェクトの Workspace に移動します。
2. プロジェクトのサイドバーから **Runs** を選択します。
3. テーブルから1つまたは複数の run を選択します。
4. 1つまたは複数の run を選択したら、テーブルの上の **Tag** ボタンを選択します。
5. 追加するタグを入力し、**Create new tag** チェックボックスを選択してタグを追加します。
    {{% /tab %}}
    {{% tab header="Run page" %}}
この方法は、1つの run に手動でタグを適用するのに最適です。

1. プロジェクトの Workspace に移動します。
2. プロジェクトの Workspace 内の run のリストから run を選択します。
3. プロジェクトのサイドバーから **Overview** を選択します。
4. **Tags** の横にある灰色のプラスアイコン（**+**）ボタンを選択します。
5. 追加するタグを入力し、テキストボックスの下にある **Add** を選択して新しいタグを追加します。
    {{% /tab %}}
{{< /tabpane >}}

## 1つまたは複数の run からタグを削除する

タグは、W&B App UI を使用して run から削除することもできます。

{{< tabpane text=true >}}
{{% tab header="Project page"%}}
この方法は、多数の run からタグを削除するのに最適です。

1. プロジェクトの Run サイドバーで、右上にあるテーブルアイコンを選択します。これにより、サイドバーが展開されて Runs テーブル全体が表示されます。
2. テーブル内の run にカーソルを合わせると、左側にチェックボックスが表示されます。または、ヘッダー行にすべての run を選択するためのチェックボックスがあります。
3. チェックボックスを選択して、一括操作を有効にします。
4. タグを削除する run を選択します。
5. run の行の上にある **Tag** ボタンを選択します。
6. タグの横にあるチェックボックスを選択して、run からタグを削除します。

{{% /tab %}}
{{% tab header="Run page"%}}

1. Run ページの左側のサイドバーで、一番上の **Overview** タブを選択します。run のタグがここに表示されます。
2. タグにカーソルを合わせ、"x" を選択して run から削除します。

{{% /tab %}}
{{< /tabpane >}}
