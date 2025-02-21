---
title: Query panels
description: このページの一部の機能はベータ版であり、機能フラグの背後に隠されています。関連するすべての機能をアンロックするには、プロファイルページの自己紹介に
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
W&B Weave をお探しですか？生成 AI アプリケーション構築のための W&B のツールスイートですか？weave のドキュメントはこちらにあります: [wandb.me/weave](https://wandb.github.io/weave/?utm_source=wandb_docs&utm_medium=docs&utm_campaign=weave-nudge)。
{{% /alert %}}

クエリー パネルを使用して、データをクエリーし、インタラクティブに可視化します。

{{< img src="/images/weave/pretty_panel.png" alt="" >}}

## クエリー パネルを作成する

ワークスペースまたは レポート 内にクエリーを追加します。

{{< tabpane text=true >}}
{{% tab header="プロジェクト ワークスペース" value="workspace" %}}

  1. プロジェクト のワークスペースに移動します。
  2. 右上隅にある「Add panel (パネルを追加)」をクリックします。
  3. ドロップダウンから「Query panel (クエリー パネル)」を選択します。
  {{< img src="/images/weave/add_weave_panel_workspace.png" alt="" >}}

{{% /tab %}}

{{% tab header="W&B Report" value="report" %}}

「/Query panel (/クエリー パネル)」と入力して選択します。

{{< img src="/images/weave/add_weave_panel_report_1.png" alt="" >}}

あるいは、クエリーを run のセットに関連付けることもできます。
1. レポート 内で、「/Panel grid (/パネル グリッド)」と入力して選択します。
2. 「Add panel (パネルを追加)」ボタンをクリックします。
3. ドロップダウンから「Query panel (クエリー パネル)」を選択します。

{{% /tab %}}
{{< /tabpane >}}
  

## クエリーの構成要素

### 式

クエリー式を使用して、run、 Artifacts 、 Models 、 Tables など、W&B に保存されているデータをクエリーします。

#### 例: テーブル のクエリー
W&B Table をクエリーするとします。トレーニング コードで、`"cifar10_sample_table"` という名前のテーブルをログに記録します。

```python
import wandb
wandb.log({"cifar10_sample_table":<MY_TABLE>})
```

クエリー パネル内で、次のクエリーを使用してテーブルをクエリーできます。
```python
runs.summary["cifar10_sample_table"]
```
{{< img src="/images/weave/basic_weave_expression.png" alt="" >}}

これを分解すると、次のようになります。

* `runs` は、クエリー パネルが ワークスペース 内にある場合に、クエリー パネル式に自動的に注入される変数です。その「値」は、その特定の ワークスペース に表示される run のリストです。[run 内で使用できるさまざまな属性については、こちらをお読みください]({{< relref path="../../../../track/public-api-guide.md#understanding-the-different-attributes" lang="ja" >}})。
* `summary` は、Run の Summary オブジェクトを返す op です。Ops は _mapped (マップ)_ されます。つまり、この op はリスト内の各 Run に適用され、Summary オブジェクトのリストが生成されます。
* `["cifar10_sample_table"]` は、Pick op（角括弧で示されます）で、 パラメータ は `predictions` です。Summary オブジェクトは辞書またはマップのように機能するため、この操作は各 Summary オブジェクトから `predictions` フィールドを選択します。

インタラクティブに独自のクエリーを作成する方法については、[この レポート ](https://wandb.ai/luis_team_test/weave_example_queries/reports/Weave-queries---Vmlldzo1NzIxOTY2?accessToken=bvzq5hwooare9zy790yfl3oitutbvno2i6c2s81gk91750m53m2hdclj0jvryhcr)をご覧ください。

### 設定

パネルの左上隅にある歯車アイコンを選択して、クエリー 設定 を展開します。これにより、ユーザー はパネルのタイプと結果パネルの パラメータ を構成できます。

{{< img src="/images/weave/weave_panel_config.png" alt="" >}}

### 結果パネル

最後に、クエリー結果パネルは、選択したクエリー パネルを使用して、クエリー式の結果をレンダリングし、 設定 によって構成されて、データをインタラクティブな形式で表示します。次の画像は、同じデータの Table と Plot を示しています。

{{< img src="/images/weave/result_panel_table.png" alt="" >}}

{{< img src="/images/weave/result_panel_plot.png" alt="" >}}

## 基本的な操作
クエリー パネル内で実行できる一般的な操作を次に示します。
### 並べ替え
列オプションから並べ替えます。
{{< img src="/images/weave/weave_sort.png" alt="" >}}

### フィルター
クエリーで直接フィルターするか、左上隅にあるフィルター ボタン（2 番目の画像）を使用できます。
{{< img src="/images/weave/weave_filter_1.png" alt="" >}}
{{< img src="/images/weave/weave_filter_2.png" alt="" >}}

### マップ
マップ操作は、リストを反復処理し、データ内の各要素に関数を適用します。これは、パネル クエリー で直接行うか、列オプションから新しい列を挿入して行うことができます。
{{< img src="/images/weave/weave_map.png" alt="" >}}
{{< img src="/images/weave/weave_map.gif" alt="" >}}

### グループ化
クエリーを使用するか、列オプションからグループ化できます。
{{< img src="/images/weave/weave_groupby.png" alt="" >}}
{{< img src="/images/weave/weave_groupby.gif" alt="" >}}

### 連結
concat 操作を使用すると、2 つのテーブルを連結し、パネル 設定 から連結または結合できます。
{{< img src="/images/weave/weave_concat.gif" alt="" >}}

### 結合
クエリーでテーブルを直接結合することもできます。次のクエリー式を検討してください。
```python
project("luis_team_test", "weave_example_queries").runs.summary["short_table_0"].table.rows.concat.join(\
project("luis_team_test", "weave_example_queries").runs.summary["short_table_1"].table.rows.concat,\
(row) => row["Label"],(row) => row["Label"], "Table1", "Table2",\
"false", "false")
```
{{< img src="/images/weave/weave_join.png" alt="" >}}

左側のテーブルは、次から生成されます。
```python
project("luis_team_test", "weave_example_queries").\
runs.summary["short_table_0"].table.rows.concat.join
```
右側のテーブルは、次から生成されます。
```python
project("luis_team_test", "weave_example_queries").\
runs.summary["short_table_1"].table.rows.concat
```
ここで、
* `(row) => row["Label"]` は、各テーブルのセレクターであり、結合する列を決定します。
* `"Table1"` と `"Table2"` は、結合されたときの各テーブルの名前です。
* `true` と `false` は、左側と右側の内部/外部結合 設定 用です。

## Runs オブジェクト
クエリー パネルを使用して、`runs` オブジェクトにアクセスします。Run オブジェクトには、 Experiments の記録が保存されます。詳細については、 レポート の[このセクション](https://wandb.ai/luis_team_test/weave_example_queries/reports/Weave-queries---Vmlldzo1NzIxOTY2?accessToken=bvzq5hwooare9zy790yfl3oitutbvno2i6c2s81gk91750m53m2hdclj0jvryhcr#3.-accessing-runs-object)を参照してください。ただし、簡単な概要として、`runs` オブジェクトには次のものが用意されています。
* `summary`: run の結果をまとめた情報の辞書。これには、精度や損失などのスカラー、または大きなファイルを含めることができます。デフォルトでは、`wandb.log()` は、ログに記録された 時系列 の最終値を summary に 設定 します。summary の内容は直接 設定 できます。summary は、run の出力と考えてください。
* `history`: 損失など、モデル のトレーニング 中に変化する値を保存するための辞書のリスト。コマンド `wandb.log()` は、このオブジェクトに追加されます。
* `config`: トレーニング run の ハイパーパラメータ や、データセット Artifact を作成する run の 前処理 メソッドなど、run の 設定 情報の辞書。これらは、run の「入力」と考えてください。
{{< img src="/images/weave/weave_runs_object.png" alt="" >}}

## Artifacts にアクセスする

Artifacts は、W&B の中心的な概念です。これらは、バージョン管理された、名前付きのファイルとディレクトリーのコレクションです。Artifacts を使用して、モデル の重み、データセット 、およびその他のファイルまたはディレクトリーを追跡します。Artifacts は W&B に保存され、ダウンロードしたり、他の run で使用したりできます。詳細と例については、 レポート の[このセクション](https://wandb.ai/luis_team_test/weave_example_queries/reports/Weave-queries---Vmlldzo1NzIxOTY2?accessToken=bvzq5hwooare9zy790yfl3oitutbvno2i6c2s81gk91750m53m2hdclj0jvryhcr#4.-accessing-artifacts)を参照してください。Artifacts には通常、`project` オブジェクトからアクセスします。
* `project.artifactVersion()`: プロジェクト 内の特定の名前とバージョンの Artifact バージョンを返します。
* `project.artifact("")`: プロジェクト 内の特定 の名前の Artifact を返します。次に、`.versions` を使用して、この Artifact のすべてのバージョンのリストを取得できます。
* `project.artifactType()`: プロジェクト 内の特定 の名前の `artifactType` を返します。次に、`.artifacts` を使用して、このタイプのすべての Artifact のリストを取得できます。
* `project.artifactTypes`: プロジェクト 下のすべての Artifact タイプのリストを返します。
{{< img src="/images/weave/weave_artifacts.png" alt="" >}}
