---
title: Add labels to runs with tags
menu:
  default:
    identifier: ja-guides-models-track-runs-tags
    parent: what-are-runs
---

タグを追加して、記録されたメトリクスやアーティファクトデータからは明らかでないかもしれない特定の特徴で run をラベル付けします。

例えば、ある run のモデルが `in_production` であることを示すタグ、run が `preemptible` であることを示すタグ、この run が `baseline` を表していることを示すタグなどを追加できます。

## 一つまたは複数の run にタグを追加する

プログラム的にまたは対話的にあなたの run にタグを追加します。

ユースケースに基づいて、以下のタブからニーズに最も適したものを選択してください。

{{< tabpane text=true >}}
    {{% tab header="W&B Python SDK" %}}
run が作成されるときにタグを追加できます。

```python
import wandb

run = wandb.init(
  entity="entity",
  project="<project-name>",
  tags=["tag1", "tag2"]
)
```

run を初期化した後でもタグを更新することができます。例えば、特定のメトリクスが事前に定義された閾値を下回った場合にタグを更新する方法が以下のコードスニペットに示されています。

```python
import wandb

run = wandb.init(
  entity="entity", 
  project="capsules", 
  tags=["debug"]
  )

# python ロジックでモデルをトレーニングする

if current_loss < threshold:
    run.tags = run.tags + ("release_candidate",)
```    
    {{% /tab %}}
    {{% tab header="Public API" %}}
run が作成された後に、[Public API]({{< relref path="/guides/models/track/public-api-guide.md" lang="ja" >}})を使用してタグを更新できます。例えば：

```python
run = wandb.Api().run("{entity}/{project}/{run-id}")
run.tags.append("tag1")  # run データに基づいてタグを選択できます
run.update()
```    
    {{% /tab %}}
    {{% tab header="Project page" %}}
この方法は、同じタグまたはタグで大量の run にタグ付けするのに最適です。

1. プロジェクトのワークスペースに移動します。
2. プロジェクトサイドバーから **Runs** を選択します。
3. テーブルから一つまたは複数の run を選択します。
4. 一つまたは複数の run を選択すると、テーブルの上の **Tag** ボタンを選択します。
5. 追加したいタグを入力し、**Create new tag** チェックボックスを選択してタグを追加します。    
    {{% /tab %}}
    {{% tab header="Run page" %}}
この方法は、単一の run に手動でタグを適用するのに最適です。

1. プロジェクトのワークスペースに移動します。
2. プロジェクトのワークスペース内の run リストから run を選択します。
1. プロジェクトサイドバーから **Overview** を選択します。
2. **Tags** ボタンの横にあるグレーのプラスアイコン（**+**）を選択します。
3. 追加したいタグを入力し、テキストボックスの下の **Add** を選択して新しいタグを追加します。    
    {{% /tab %}}
{{< /tabpane >}}



## 一つまたは複数の run からタグを削除する

W&B アプリ UI で run からタグを削除することもできます。

{{< tabpane text=true >}}
{{% tab header="Project page"%}}
この方法は、大量の run からタグを削除するのに最適です。

1. プロジェクトの Run サイドバーで、右上のテーブルアイコンを選択します。これにより、サイドバーは完全な run テーブルに展開されます。
2. テーブル内の run 上にカーソルを合わせると、左側にチェックボックスが表示されるか、ヘッダ行にすべての run を選択するチェックボックスを確認します。
3. チェックボックスを選択して一括操作を有効にします。
4. タグを削除したい run を選択します。
5. run の行の上の **Tag** ボタンを選択します。
6. run からタグを削除するには、タグの横のチェックボックスを選択します。

{{% /tab %}}
{{% tab header="Run page"%}}

1. Run ページの左サイドバーで、上部の **Overview** タブを選択します。run のタグはここに表示されます。
2. タグにカーソルを合わせて、"x" を選択して run から削除します。

{{% /tab %}}
{{< /tabpane >}}