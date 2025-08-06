---
title: タグを使って run にラベルを追加する
menu:
  default:
    identifier: ja-guides-models-track-runs-tags
    parent: what-are-runs
---

特定の特徴を持つ run にタグを追加して、ログしたメトリクスやアーティファクト データからは分かりにくい run をラベル付けできます。

例えば、run の model が `in_production` であること、run が `preemptible` であること、この run が `baseline` を表していることなどをタグとして追加できます。

## 1つまたは複数の run にタグを追加する

プログラムまたはインタラクティブに run にタグを追加できます。

ユースケースに応じて、以下のタブから最適な方法を選択してください。

{{< tabpane text=true >}}
    {{% tab header="W&B Python SDK" %}}
run 作成時にタグを追加できます。

```python
import wandb

run = wandb.init(
  entity="entity",
  project="<project-name>",
  tags=["tag1", "tag2"]
)
```

run の初期化後にもタグを更新できます。例えば、以下のコードスニペットは、特定のメトリクスがあらかじめ設定したしきい値を下回った場合に、タグを更新する方法を示しています。

```python
import wandb

run = wandb.init(
  entity="entity", 
  project="capsules", 
  tags=["debug"]
  )

# モデルをトレーニングする Python ロジック

if current_loss < threshold:
    run.tags = run.tags + ("release_candidate",)
```
    {{% /tab %}}
    {{% tab header="Public API" %}}
run 作成後、[Public API]({{< relref path="/guides/models/track/public-api-guide.md" lang="ja" >}}) を利用してタグを追加・更新できます。例えば:

```python
run = wandb.Api().run("{entity}/{project}/{run-id}")
run.tags.append("tag1")  # ここで run のデータに基づいてタグを選択できます
run.update()
```
    {{% /tab %}}
    {{% tab header="Project page" %}}
この方法は、多くの run に同じタグを一括で付けたい場合に適しています。

1. プロジェクトの workspace に移動します。
2. プロジェクトのサイドバーから **Runs** を選択します。
3. テーブルから 1 つまたは複数の run を選択します。
4. 選択ができたら、テーブル上部の **Tag** ボタンをクリックします。
5. 追加したいタグを入力し、**Create new tag** チェックボックスを選択すると新しいタグが追加されます。
    {{% /tab %}}
    {{% tab header="Run page" %}}
この方法は、1 つの run に手動でタグを追加したい場合に便利です。

1. プロジェクトの workspace に移動します。
2. プロジェクト workspace 内の run 一覧から任意の run を選択します。
1. プロジェクトサイドバーから **Overview** を選択します。
2. **Tags** となりのグレーのプラスアイコン（**+**）ボタンを選びます。
3. 追加したいタグを入力し、テキストボックス下の **Add** を選択すると新規タグが追加されます。
    {{% /tab %}}
{{< /tabpane >}}



## 1つまたは複数の run からタグを削除する

タグは W&B アプリ UI からも run から削除できます。

{{< tabpane text=true >}}
{{% tab header="Project page"%}}
この方法は、多くの run からタグを一括で削除したい場合に適しています。

1. プロジェクトの Run サイドバーで、右上のテーブルアイコンを選択します。これでサイドバーが展開し、全 run のテーブルが表示されます。
2. テーブルの run 上にマウスを重ねると左側にチェックボックスが表示されます。すべての run を選択したい場合はヘッダー行のチェックボックスを使用します。
3. チェックボックスを選択して一括操作を有効にします。
4. タグを削除したい run を選択します。
5. run の列の上にある **Tag** ボタンを選択します。
6. 削除したいタグの隣にあるチェックボックスを選択して、run からタグを削除します。

{{% /tab %}}
{{% tab header="Run page"%}}

1. Run ページの左サイドバーで一番上の **Overview** タブを選択します。run に付いているタグが表示されます。
2. タグの上にマウスを重ねて「x」マークをクリックすると、そのタグを run から削除できます。

{{% /tab %}}
{{< /tabpane >}}