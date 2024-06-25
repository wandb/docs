---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# タグ

タグは、ログされたメトリクスやArtifactデータからは明らかでない特定の特徴でrunをラベル付けするのに使用できます。例えば、このrunのモデルは`in_production`、あのrunは`preemptible`、このrunは`baseline`を表しています。

## タグの追加方法

タグはrunが作成されるときに追加できます：`wandb.init(tags=["tag1", "tag2"])` 。

また、トレーニング中にrunのタグを更新することもできます（例えば、特定のメトリクスが事前に定義された閾値を超えた場合）：

```python
run = wandb.init(entity="entity", project="capsules", tags=["debug"])

...

if current_loss < threshold:
    run.tags = run.tags + ("release_candidate",)
```

また、runがW&Bにログされた後にも、いくつかの方法でタグを追加することができます。

<Tabs
  defaultValue="publicapi"
  values={[
    {label: 'Using the Public API', value: 'publicapi'},
    {label: 'Project Page', value: 'projectpage'},
    {label: 'Run Page', value: 'runpage'},
  ]}>
  <TabItem value="publicapi">

runが作成された後、[our public API](../../../guides/track/public-api-guide.md)を使用してタグを更新できます：

```python
run = wandb.Api().run("{entity}/{project}/{run-id}")
run.tags.append("tag1")  # runデータに基づいてタグを選択できます
run.update()
```

Public APIの使い方の詳細は、[reference documentation](../../../ref/README.md) または [guide](../../../guides/track/public-api-guide.md) を参照してください。

  </TabItem>
  <TabItem value="projectpage">

この方法は、大量のrunに同じタグを付けるのに最適です。

[Project Page](../pages/project-page.md) の [runs sidebar](../pages/project-page.md#search-for-runs) で、右上のテーブルアイコンをクリックします。これでサイドバーが全画面の[runs table](runs-table.md)に拡張されます。

テーブルのrunの上にマウスを置くと左側にチェックボックスが表示されます。または、ヘッダー行のチェックボックスを見つけてすべてのrunを選択することもできます。

チェックボックスをクリックすると一括操作が有効になります。タグを適用したいrunを選択します。

行の上の「Tag」ボタンをクリックします。

追加したいタグを入力し、テキストボックスの下の「Add」をクリックして新しいタグを追加します。

  </TabItem>
  <TabItem value="runpage">

この方法は、手動で単一のrunにタグを適用するのに最適です。

[Run Page](../pages/run-page.md) の左サイドバーで、トップの [Overviewタブ](../pages/run-page.md#overview-tab) をクリックします。

「Tags」の横にある灰色の ➕ ボタンをクリックしてタグを追加します。

追加したいタグを入力し、テキストボックスの下の「Add」をクリックして新しいタグを追加します。

  </TabItem>
</Tabs>

## タグの削除方法

タグはUIを通じてrunから削除することもできます。

<Tabs
  defaultValue="projectpage"
  values={[
    {label: 'Project Page', value: 'projectpage'},
    {label: 'Run Page', value: 'runpage'},
  ]}>
  <TabItem value="projectpage">

この方法は、大量のrunからタグを削除するのに最適です。

[Project Page](../pages/project-page.md) の [runs sidebar](../pages/project-page.md#search-for-runs) で、右上のテーブルアイコンをクリックします。これでサイドバーが全画面の[runs table](runs-table.md)に拡張されます。

テーブルのrunの上にマウスを置くと左側にチェックボックスが表示されます。または、ヘッダー行のチェックボックスを見つけてすべてのrunを選択することもできます。

いずれかのチェックボックスをクリックすると一括操作が有効になります。タグを削除したいrunを選択します。

行の上の「Tag」ボタンをクリックします。

タグの横にあるチェックボックスをクリックしてrunから削除します。

  </TabItem>
  <TabItem value="runpage">

[Run Page](../pages/run-page.md) の左サイドバーで、トップの [Overviewタブ](../pages/run-page.md#overview-tab) をクリックします。ここにrunのタグが表示されます。

タグの上にマウスを置いて「x」をクリックし、runから削除します。

  </TabItem>
</Tabs>