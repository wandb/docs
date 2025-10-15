---
title: runs にタグでラベルを追加する
menu:
  default:
    identifier: ja-guides-models-track-runs-tags
    parent: what-are-runs
---

特定の特徴を持つ runs にタグを追加して、ログされたメトリクスやアーティファクトデータからは見えない情報をラベル付けできます。

たとえば、ある run のモデルが `in_production` であること、run が `preemptible` であること、この run が `baseline` を表していることなどを示すためにタグを追加できます。

## 1つまたは複数の runs にタグを追加する

プログラムによって、または対話的に runs にタグを追加します。

あなたのユースケースに基づいて、以下のタブからニーズに最も合ったものを選択してください。

{{< tabpane text=true >}}
    {{% tab header="W&B Python SDK" %}}
run が作成されるときにタグを追加できます:

```python
import wandb

run = wandb.init(
  entity="entity",
  project="<project-name>",
  tags=["tag1", "tag2"]
)
```

run を初期化した後でもタグを更新できます。たとえば、特定のメトリクスが事前に定義されたしきい値を超えた場合、タグを更新する方法を示すコードスニペットです:

```python
import wandb

run = wandb.init(
  entity="entity", 
  project="capsules", 
  tags=["debug"]
  )

# モデルをトレーニングするための Python ロジック

if current_loss < threshold:
    run.tags = run.tags + ("release_candidate",)
```    
    {{% /tab %}}
    {{% tab header="Public API" %}}
run を作成した後、[Public API]({{< relref path="/guides/models/track/public-api-guide.md" lang="ja" >}})を使用してタグを更新することができます。例:

```python
run = wandb.Api().run("{entity}/{project}/{run-id}")
run.tags.append("tag1")  # ここで run データに基づいてタグを選択できます
run.update()
```    
    {{% /tab %}}
    {{% tab header="Project page" %}}
このメソッドは、同じタグまたは複数のタグを大量の runs にタグ付けするのに最も適しています。

1. プロジェクトワークスペースに移動します。
2. プロジェクトのサイドバーから **Runs** を選択します。
3. テーブルから1つまたは複数の runs を選択します。
4. 1つまたは複数の runs を選択すると、テーブルの上にある **Tag** ボタンを選択します。
5. 追加したいタグを入力し、そのタグを追加するために **Create new tag** チェックボックスを選択します。    
    {{% /tab %}}
    {{% tab header="Run page" %}}
このメソッドは、1つの run に手動でタグを適用するのに最も適しています。

1. プロジェクトワークスペースに移動します。
2. プロジェクトのワークスペース内の runs リストから run を1つ選択します。
1. プロジェクトサイドバーから **Overview** を選択します。
2. **Tags** の横にある灰色のプラスアイコン (**+**) ボタンを選択します。
3. 追加したいタグを入力し、新しいタグを追加するためにテキストボックスの下にある **Add** を選択します。    
    {{% /tab %}}
{{< /tabpane >}}

## 1つまたは複数の runs からタグを削除する

W&B App の UI を使って、runs からタグを削除することもできます。

{{< tabpane text=true >}}
{{% tab header="Project page"%}}
このメソッドは、大量の runs からタグを削除するのに最も適しています。

1. プロジェクトの Run サイドバーで、右上のテーブルアイコンを選択します。これにより、サイドバーがフルランズテーブルに展開されます。
2. テーブル内の run にカーソルを合わせると、左側にチェックボックスが表示されるか、すべての runs を選択するためのチェックボックスがヘッダー行に表示されます。
3. チェックボックスを選択して一括操作を有効にします。
4. タグを削除したい runs を選択します。
5. run の行の上にある **Tag** ボタンを選択します。
6. run から削除するために、タグの横にあるチェックボックスを選択します。

{{% /tab %}}
{{% tab header="Run page"%}}

1. Run ページの左サイドバーで、上部の **Overview** タブを選択します。ここで run のタグが表示されます。
2. タグにカーソルを合せ、「x」を選択して run から削除します。

{{% /tab %}}
{{< /tabpane >}}