---
title: タグを使って runs にラベルを追加する
menu:
  default:
    identifier: ja-guides-models-track-runs-tags
    parent: what-are-runs
---

ログした メトリクス や Artifacts の データ からは分かりにくい特性で run をラベル付けするために、タグを追加します。
例えば、その run の モデル が `in_production` である、run が `preemptible` である、その run が `baseline` を表す、などを示すタグを追加できます。

## 1 つ以上の run に タグを追加する

プログラムから、または対話的に run に タグを追加できます。
ユースケース に応じて、適切なタブを選択してください:

{{< tabpane text=true >}}
    {{% tab header="W&B Python SDK" %}}
run 作成時に タグを追加できます:

```python
import wandb

run = wandb.init(
  entity="entity",
  project="<project-name>",
  tags=["tag1", "tag2"]
)
```

run を初期化した後に タグを更新することもできます。例えば、次の コードスニペット は、特定の メトリクス があらかじめ定義したしきい値を超えた場合に タグを更新する方法を示します:

```python
import wandb

run = wandb.init(
  entity="entity", 
  project="capsules", 
  tags=["debug"]
  )

# モデル学習のための Python ロジック

if current_loss < threshold:
    run.tags = run.tags + ("release_candidate",)
```    
    {{% /tab %}}
    {{% tab header="Public API" %}}
run を作成した後、[the Public API]({{< relref path="/guides/models/track/public-api-guide.md" lang="ja" >}}) を使って タグを更新できます。例えば:

```python
run = wandb.Api().run("{entity}/{project}/{run-id}")
run.tags.append("tag1")  # ここで run の データ に基づいて タグを選べます
run.update()
```    
    {{% /tab %}}
    {{% tab header="Project page" %}}
この方法は、多数の run に 同じ タグを付けるのに適しています。

1. プロジェクトの Workspace に移動します。
2. プロジェクトのサイドバーで **Runs** を選択します。
3. テーブルから 1 つ以上の run を選択します。
4. run を選択したら、テーブル上部の **Tag** ボタンを選択します。
5. 追加したい タグ を入力し、**Create new tag** チェックボックスを選択して タグ を追加します。    
    {{% /tab %}}
    {{% tab header="Run page" %}}
この方法は、単一の run に 手動で タグを付与するのに適しています。

1. プロジェクトの Workspace に移動します。
2. プロジェクトの Workspace 内の run 一覧から run を 1 つ選択します。
1. プロジェクトのサイドバーで **Overview** を選択します。
2. **Tags** の横にある灰色のプラス アイコン（**+**）ボタンを選択します。
3. 追加したい タグ を入力し、テキストボックスの下にある **Add** を選択して新しい タグ を追加します。    
    {{% /tab %}}
{{< /tabpane >}}

## 1 つ以上の run から タグ を削除する

W&B アプリの UI からも、run から タグ を削除できます。

{{< tabpane text=true >}}
{{% tab header="Project page"%}}
この方法は、多数の run から タグ を一括で削除するのに適しています。

1. プロジェクトの Run サイドバーで、右上のテーブル アイコンを選択します。サイドバーが展開され、完全な **Runs** テーブルが表示されます。
2. テーブル内の run にカーソルを合わせると左側にチェックボックスが表示されます。すべての run を選択するには、ヘッダー行のチェックボックスを使用します。
3. チェックボックスを選択して一括操作を有効にします。 
4. タグ を削除したい run を選択します。
5. run の行の上にある **Tag** ボタンを選択します。
6. 削除したい タグ の横にあるチェックボックスを選択して、run から削除します。

{{% /tab %}}
{{% tab header="Run page"%}}

1. Run ページ左側のサイドバーで、一番上の **Overview** タブを選択します。ここに、その run の タグ が表示されます。
2. タグ にカーソルを合わせ、"x" を選択して run から削除します。

{{% /tab %}}
{{< /tabpane >}}