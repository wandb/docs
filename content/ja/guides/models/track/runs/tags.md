---
title: タグで run にラベルを追加する
menu:
  default:
    identifier: tags
    parent: what-are-runs
---

特定の特徴で runs をラベル付けするタグを追加できます。これにより、ログされたメトリクスやアーティファクト データでは分かりづらい情報も明示できます。

たとえば、モデルが `in_production` な run、`preemptible` な run、この run が `baseline` を表している場合など、run にタグを追加できます。

## 1つまたは複数の run にタグを追加する

プログラム的にもインタラクティブにも、runs にタグを付けることができます。

ユースケースに合わせて、以下のタブから最適な方法を選択してください。

{{< tabpane text=true >}}
    {{% tab header="W&B Python SDK" %}}
run 作成時にタグを付けることができます：

```python
import wandb

run = wandb.init(
  entity="entity",
  project="<project-name>",
  tags=["tag1", "tag2"]
)
```

また、run を初期化した後にタグを更新することも可能です。たとえば、特定のメトリクスがあらかじめ設定したしきい値を下回った場合にタグを追加したい場合は、次のコードスニペットのように対応できます。

```python
import wandb

run = wandb.init(
  entity="entity", 
  project="capsules", 
  tags=["debug"]
  )

# モデルの訓練に関する python ロジック

if current_loss < threshold:
    run.tags = run.tags + ("release_candidate",)
```    
    {{% /tab %}}
    {{% tab header="Public API" %}}
run 作成後に [Public API]({{< relref "/guides/models/track/public-api-guide.md" >}}) を使ってタグを更新できます。例：

```python
run = wandb.Api().run("{entity}/{project}/{run-id}")
run.tags.append("tag1")  # run データにあわせて任意のタグを追加できます
run.update()
```    
    {{% /tab %}}
    {{% tab header="Project page" %}}
この方法は、多数の runs に同じタグをまとめて付与する場合に最適です。

1. 対象のプロジェクト workspace に移動します。
2. プロジェクトのサイドバーから **Runs** を選択します。
3. テーブルから 1 つまたは複数の run を選択します。
4. いずれかの run を選択するとテーブルの上部に現れる **Tag** ボタンをクリックします。
5. 追加したいタグを入力し、**Create new tag** チェックボックスで新規タグとして追加します。    
    {{% /tab %}}
    {{% tab header="Run page" %}}
この方法は、1 つの run に手動でタグを追加したい場合に最適です。

1. 対象のプロジェクト workspace に移動します。
2. プロジェクト workspace 内の runs リストから run を 1 つ選択します。
1. プロジェクトサイドバーの **Overview** を選択します。
2. **Tags** の隣にあるグレーのプラスアイコン（**+**）ボタンをクリックします。
3. 追加したいタグ名を入力し、テキストボックス下の **Add** を選択して新規タグを追加します。    
    {{% /tab %}}
{{< /tabpane >}}



## 1つまたは複数の run からタグを削除する

タグは W&B のアプリケーション UI から runs から削除できます。

{{< tabpane text=true >}}
{{% tab header="Project page"%}}
この方法は、多数の runs からタグをまとめて削除する場合に最適です。

1. プロジェクトの Run サイドバーで、右上のテーブルアイコンをクリックします。するとサイドバーが展開されて、全 runs テーブルが表示されます。
2. テーブル内で run の上にカーソルを合わせると左側にチェックボックスが表示されます。または、ヘッダー行のチェックボックスで全選択も可能です。
3. チェックボックスをオンにし、一括操作を有効化します。
4. タグ削除対象の run を選択します。
5. runs の行の上にある **Tag** ボタンをクリックします。
6. 削除したいタグの横のチェックボックスを外して、そのタグを run から削除します。

{{% /tab %}}
{{% tab header="Run page"%}}

1. Run ページの左サイドバーから、最上部の **Overview** タブを選択します。ここで run に付与されているタグが確認できます。
2. 削除したいタグにカーソルを合わせ、「x」をクリックするとそのタグが run から削除されます。

{{% /tab %}}
{{< /tabpane >}}