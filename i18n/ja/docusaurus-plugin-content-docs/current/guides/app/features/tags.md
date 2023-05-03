import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# タグ

タグは、記録されたメトリックスやアーティファクトデータからは明らかでない特定の機能を持つrunsをラベル付けするために使用できます。例えば、このrunのモデルは `in_production` 、その他のrunは `preemptible` 、このrunは `baseline` を表しています。

## タグの追加方法

runが作成されるときにタグを追加できます: `wandb.init(tags=["tag1", "tag2"])` .

トレーニング中にrunのタグを更新することもできます（例えば、特定のメトリクスが事前に定義された閾値を超えた場合）:

```python
run = wandb.init(entity="entity", project="capsules", tags=["debug"])

...

if current_loss < threshold:
    run.tags = run.tags + ("release_candidate",)
```

また、Weights & Biasesにログインされたrunsに対してタグを追加するいくつかの方法があります。

<Tabs
  defaultValue="publicapi"
  values={[
    {label: '公開APIを使用して', value: 'publicapi'},
    {label: 'プロジェクトページ', value: 'projectpage'},
    {label: 'Runページ', value: 'runpage'},
  ]}>
  <TabItem value="publicapi">
runが作成された後、次のように[弊社の公開API](../../../guides/track/public-api-guide.md)を使用してタグを更新できます。

```python
run = wandb.Api().run("{entity}/{project}/{run-id}"})
run.tags.append("tag1")  # ここでrunデータに基づいてタグを選択できます
run.update()
```

Public APIの使い方については、[リファレンスドキュメント](../../../ref/README.md)や[ガイド](../../../guides/track/public-api-guide.md)で詳しくご紹介しています。

  </TabItem>
  <TabItem value="projectpage">

この方法は、同じタグやタグを大量のrunsに付けるのに最適です。

[プロジェクトページ](../pages/project-page.md)の[runsサイドバー](../pages/project-page.md#search-for-runs)で、右上のテーブルアイコンをクリックします。これにより、サイドバーが[runsテーブル](runs-table.md)に拡張されます。

テーブルの中のrunにカーソルを置くと左側にチェックボックスが表示されるか、すべてのrunsを選択できるヘッダー行にチェックボックスが表示されます。

チェックボックスをクリックして一括操作を有効にします。タグを適用したいrunsを選択します。

runsの行の上にあるタグボタンをクリックします。

追加したいタグを入力し、テキストボックスの下にある「追加」をクリックして新しいタグを追加します。

  </TabItem>
  <TabItem value="runpage">

この方法は、手動で1つのrunにタグまたはタグを適用するのに最適です。
[Run Page](../pages/run-page.md)の左サイドバーにある、一番上の[概要タブ](../pages/run-page.md#overview-tab)をクリックしてください。

"Tags"の隣に、グレーの➕ボタンがあります。そのプラスをクリックしてタグを追加します。

追加したいタグを入力し、テキストボックスの下の"Add"をクリックして新しいタグを追加します。

  </TabItem>
</Tabs>

## タグの削除方法

UIを通じて、タグはrunsから削除することもできます。

<Tabs
  defaultValue="projectpage"
  values={[
    {label: 'プロジェクトページ', value: 'projectpage'},
    {label: 'Runページ', value: 'runpage'},
  ]}>
  <TabItem value="projectpage">

この方法は、多くのrunsからタグを削除するのに最適です。

[プロジェクトページ](../pages/project-page.md) の[runsサイドバー](../pages/project-page.md#search-for-runs)で、右上にあるテーブルアイコンをクリックします。これにより、サイドバーが完全な[runsテーブル](runs-table.md)に展開されます。

テーブル内のrunの上にマウスを置くと、左側にチェックボックスが表示されます。また、すべてのrunsを選択できるヘッダー行にチェックボックスが表示されます。

どちらかのチェックボックスをクリックして一括操作を有効にします。タグを削除したいrunsを選択してください。
上のrunsの列の上にあるタグボタンをクリックしてください。

ランのタグを削除するには、そのタグの横にあるチェックボックスをクリックします。

  </TabItem>

  <TabItem value="runpage">

[実行ページ](../pages/run-page.md)の左側のサイドバーで、一番上の[概要タブ](../pages/run-page.md#overview-tab)をクリックしてください。ここで、ランのタグが表示されます。

タグの上にマウスを置いて、その "x" をクリックして実行からタグを削除します。

  </TabItem>

</Tabs>