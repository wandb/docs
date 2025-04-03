---
title: Query panels
description: このページのいくつかの機能はベータ版で、機能フラグの背後に隠されています。関連するすべての機能をアンロックするには、プロフィールページの自己紹介に
  `weave-plot` を追加してください。
cascade:
- url: guides/app/features/panels/query-panels/:filename
menu:
  default:
    identifier: ja-guides-models-app-features-panels-query-panels-_index
    parent: panels
url: guides/app/features/panels/query-panels
---

{{% alert %}}
W&B Weave をお探しですか? 生成 AI アプリケーション構築のための W&B のツールスイートですか? Weave のドキュメントはこちらにあります: [wandb.me/weave](https://wandb.github.io/weave/?utm_source=wandb_docs&utm_medium=docs&utm_campaign=weave-nudge)。
{{% /alert %}}

クエリー パネルを使用して、データをクエリーし、インタラクティブに視覚化します。

{{< img src="/images/weave/pretty_panel.png" alt="" >}}

## クエリー パネルの作成

ワークスペースまたは レポート 内にクエリーを追加します。

{{< tabpane text=true >}}
{{% tab header="プロジェクト ワークスペース" value="workspace" %}}

  1. プロジェクト の ワークスペース に移動します。
  2. 右上隅にある「Add panel（ パネル を追加）」をクリックします。
  3. ドロップダウンから「Query panel（クエリー パネル ）」を選択します。
  {{< img src="/images/weave/add_weave_panel_workspace.png" alt="" >}}

{{% /tab %}}

{{% tab header="W&B Report" value="report" %}}

「/Query panel（/クエリー パネル ）」と入力して選択します。

{{< img src="/images/weave/add_weave_panel_report_1.png" alt="" >}}

または、クエリーを 一連の Runs に関連付けることもできます。
1. レポート 内で、「/Panel grid（/ パネル グリッド）」と入力して選択します。
2. 「Add panel（ パネル を追加）」ボタンをクリックします。
3. ドロップダウンから「Query panel（クエリー パネル ）」を選択します。

{{% /tab %}}
{{< /tabpane >}}
  

## クエリーのコンポーネント

### 式

クエリー式を使用して、Runs、Artifacts、Models、Tables など、W&B に保存されているデータをクエリーします。

#### 例: テーブル のクエリー
W&B Table をクエリーするとします。 トレーニング コード で、`"cifar10_sample_table"`というテーブルを ログ に記録します。

```python
import wandb
wandb.log({"cifar10_sample_table":<MY_TABLE>})
```

クエリー パネル 内では、次のコードでテーブルをクエリーできます。
```python
runs.summary["cifar10_sample_table"]
```
{{< img src="/images/weave/basic_weave_expression.png" alt="" >}}

これを分解すると次のようになります。

* `runs` は、クエリー パネル が ワークスペース にある場合、クエリー パネル 式に自動的に挿入される変数です。 その「値」は、特定の ワークスペース で表示できる Runs のリストです。 [run で使用できるさまざまな属性については、こちらをお読みください]({{< relref path="../../../../track/public-api-guide.md#understanding-the-different-attributes" lang="ja" >}})。
* `summary` は、Run の Summary オブジェクトを返す op です。 Op は _mapped_ です。つまり、この op はリスト内の各 Run に適用され、Summary オブジェクトのリストが生成されます。
* `["cifar10_sample_table"]` は、`predictions` という パラメータ を持つ Pick op（角かっこで示されます）です。 Summary オブジェクトは ディクショナリー または マップ のように動作するため、この操作は各 Summary オブジェクトから `predictions` フィールドを選択します。

独自のクエリーをインタラクティブに作成する方法については、[この レポート ](https://wandb.ai/luis_team_test/weave_example_queries/reports/Weave-queries---Vmlldzo1NzIxOTY2?accessToken=bvzq5hwooare9zy790yfl3oitutbvno2i6c2s81gk91750m53m2hdclj0jvryhcr)を参照してください。

### 設定

パネル の左上隅にある歯車アイコンを選択して、クエリー 設定 を展開します。 これにより、 ユーザー は パネル のタイプと、結果 パネル の パラメータ を 設定 できます。

{{< img src="/images/weave/weave_panel_config.png" alt="" >}}

### 結果 パネル

最後に、クエリー結果 パネル は、選択したクエリー パネル を使用して、クエリー式の結果をレンダリングし、データ をインタラクティブな形式で表示するように 設定 によって 設定 されます。 次の画像は、同じデータの Table とプロットを示しています。

{{< img src="/images/weave/result_panel_table.png" alt="" >}}

{{< img src="/images/weave/result_panel_plot.png" alt="" >}}

## 基本操作
クエリー パネル 内で実行できる一般的な操作を次に示します。
### 並べ替え
列オプションから並べ替えを行います。
{{< img src="/images/weave/weave_sort.png" alt="" >}}

### フィルター
クエリーで直接 フィルター するか、左上隅にある フィルター ボタンを使用できます（2 番目の画像）。
{{< img src="/images/weave/weave_filter_1.png" alt="" >}}
{{< img src="/images/weave/weave_filter_2.png" alt="" >}}

### マップ
マップ 操作はリストを反復処理し、データ 内の各要素に関数を適用します。 これは、 パネル クエリー で直接行うか、列オプションから新しい列を挿入して行うことができます。
{{< img src="/images/weave/weave_map.png" alt="" >}}
{{< img src="/images/weave/weave_map.gif" alt="" >}}

### グループ化
クエリーまたは列オプションを使用してグループ化できます。
{{< img src="/images/weave/weave_groupby.png" alt="" >}}
{{< img src="/images/weave/weave_groupby.gif" alt="" >}}

### 連結
連結操作を使用すると、2 つのテーブルを連結し、 パネル 設定 から連結または結合できます。
{{< img src="/images/weave/weave_concat.gif" alt="" >}}

### 結合
クエリーでテーブルを直接結合することも可能です。 次のクエリー式を検討してください。
```python
project("luis_team_test", "weave_example_queries").runs.summary["short_table_0"].table.rows.concat.join(\
project("luis_team_test", "weave_example_queries").runs.summary["short_table_1"].table.rows.concat,\
(row) => row["Label"],(row) => row["Label"], "Table1", "Table2",\
"false", "false")
```
{{< img src="/images/weave/weave_join.png" alt="" >}}

左側のテーブルは、次のように生成されます。
```python
project("luis_team_test", "weave_example_queries").\
runs.summary["short_table_0"].table.rows.concat.join
```
右側のテーブルは、次のように生成されます。
```python
project("luis_team_test", "weave_example_queries").\
runs.summary["short_table_1"].table.rows.concat
```
説明:
* `(row) => row["Label"]` は各テーブルのセレクターであり、結合する列を決定します
* `"Table1"` と `"Table2"` は、結合時の各テーブルの名前です
* `true` と `false` は、左側の内部/外部結合 設定 用です


## Runs オブジェクト
クエリー パネル を使用して `runs` オブジェクトにアクセスします。 Run オブジェクトは、 Experiments のレコードを保存します。 詳細については、 レポート の[このセクション](https://wandb.ai/luis_team_test/weave_example_queries/reports/Weave-queries---Vmlldzo1NzIxOTY2?accessToken=bvzq5hwooare9zy790yfl3oitutbvno2i6c2s81gk91750m53m2hdclj0jvryhcr#3.-accessing-runs-object)を参照してください。簡単にまとめると、`runs` オブジェクトには次のものが含まれます。
* `summary`: Run の結果をまとめた情報の ディクショナリー 。 これは、精度や損失などのスカラー、または大きなファイルにすることができます。 デフォルトでは、`wandb.log()` は Summary を ログ に記録された 時系列 の最終値に 設定 します。 Summary の内容を直接 設定 できます。 Summary を Run の出力として考えてください。
* `history`: 損失など、モデル の トレーニング 中に変化する値を保存するための ディクショナリー のリスト。 コマンド `wandb.log()` はこの オブジェクト に追加されます。
* `config`: トレーニング Run の ハイパー パラメータ ーや、データセット Artifact を作成する Run の 前処理 メソッド など、Run の 設定 情報の ディクショナリー 。 これらを Run の「入力」と考えてください。
{{< img src="/images/weave/weave_runs_object.png" alt="" >}}

## Artifacts へのアクセス

Artifacts は W&B の 中核となる概念です。 これらは、 バージョン 管理された名前付きのファイルと ディレクトリー のコレクションです。 Artifacts を使用して、モデル の重み、データセット 、およびその他のファイルまたは ディレクトリー を追跡します。 Artifacts は W&B に保存され、ダウンロードしたり、他の Runs で使用したりできます。 詳細と例については、 レポート の[このセクション](https://wandb.ai/luis_team_test/weave_example_queries/reports/Weave-queries---Vmlldzo1NzIxOTY2?accessToken=bvzq5hwooare9zy790yfl3oitutbvno2i6c2s81gk91750m53m2hdclj0jvryhcr#4.-accessing-artifacts)を参照してください。 Artifacts には通常、`project` オブジェクト からアクセスします。
* `project.artifactVersion()`: プロジェクト 内の指定された名前と バージョン の特定の Artifact バージョン を返します。
* `project.artifact("")`: プロジェクト 内の指定された名前の Artifact を返します。 次に、`.versions` を使用して、この Artifact のすべての バージョン のリストを取得できます。
* `project.artifactType()`: プロジェクト 内の指定された名前の `artifactType` を返します。 次に、`.artifacts` を使用して、このタイプのすべての Artifacts のリストを取得できます。
* `project.artifactTypes`: プロジェクト 下にあるすべての Artifact タイプ のリストを返します。
{{< img src="/images/weave/weave_artifacts.png" alt="" >}}
