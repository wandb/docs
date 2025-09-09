---
title: クエリ パネル
description: このページの一部の機能はベータ版で、機能フラグの有効化が必要です。プロフィールページの bio に `weave-plot` を追加すると、関連するすべての機能が利用可能になります。
cascade:
- url: guides/app/features/panels/query-panels/:filename
menu:
  default:
    identifier: ja-guides-models-app-features-panels-query-panels-_index
    parent: panels
url: guides/app/features/panels/query-panels
---

{{% alert %}}
W&B の Weave をお探しですか？ Generative AI アプリケーション構築のための W&B のツール群です。ドキュメントはこちら: [wandb.me/weave](https://wandb.github.io/weave/?utm_source=wandb_docs&utm_medium=docs&utm_campaign=weave-nudge).
{{% /alert %}}

Query パネルを使って、データをクエリし、インタラクティブに可視化します。

{{< img src="/images/weave/pretty_panel.png" alt="Query パネル" >}}




## Query パネルを作成する

Workspace や Report にクエリを追加します。

{{< tabpane text=true >}}
{{% tab header="Project Workspace" value="workspace" %}}

  1. 対象の Project の Workspace に移動します。 
  2. 右上の `Add panel` をクリックします。
  3. ドロップダウンから `Query panel` を選択します。
  {{< img src="/images/weave/add_weave_panel_workspace.png" alt="Add panel のドロップダウン" >}}

{{% /tab %}}

{{% tab header="W&B Report" value="report" %}}

「/Query panel」と入力して選択します。

{{< img src="/images/weave/add_weave_panel_report_1.png" alt="Query panel のオプション" >}}

別の方法として、クエリを複数の run に関連付けることもできます:
1. Report 内で「/Panel grid」と入力して選択します。
2. `Add panel` ボタンをクリックします。
3. ドロップダウンから `Query panel` を選択します。

{{% /tab %}}
{{< /tabpane >}}
  

## クエリ コンポーネント

### 式

クエリ式を使って、W&B に保存されているデータ（run、Artifacts、モデル、テーブル など）をクエリできます。 

#### 例: テーブルをクエリする
W&B Table をクエリしたいとします。トレーニング コードで "cifar10_sample_table" というテーブルをログしています:

```python
import wandb
with wandb.init() as run:
  run.log({"cifar10_sample_table":<MY_TABLE>})
```

Query パネル内では、次のようにテーブルをクエリできます:
```python
runs.summary["cifar10_sample_table"]
```
{{< img src="/images/weave/basic_weave_expression.png" alt="テーブルのクエリ式" >}}

内訳:

* `runs` は、Workspace 上の Query パネルでクエリ式に自動注入される変数です。その「値」は、その Workspace で表示可能な run のリストです。[Run 内で利用可能なさまざまな属性についてはこちらを参照]({{< relref path="../../../../track/public-api-guide.md#understanding-the-different-attributes" lang="ja" >}})。
* `summary` は、ある Run の Summary オブジェクトを返す op です。Ops は _mapped_ であり、この op はリスト内の各 Run に適用され、その結果 Summary オブジェクトのリストが得られます。
* `["cifar10_sample_table"]` は Pick op（角括弧で表記）で、引数は `predictions` です。Summary オブジェクトは辞書（map）のように扱えるため、この操作は各 Summary オブジェクトから `predictions` フィールドを取り出します。

インタラクティブに独自のクエリを書く方法は、[Query panel のデモ](https://wandb.ai/luis_team_test/weave_example_queries/reports/Weave-queries---Vmlldzo1NzIxOTY2?accessToken=bvzq5hwooare9zy790yfl3oitutbvno2i6c2s81gk91750m53m2hdclj0jvryhcr)をご覧ください。

### 設定

パネル左上の歯車アイコンを選ぶと、クエリの設定を展開できます。これにより、パネルの種類や結果パネルのパラメータを設定できます。

{{< img src="/images/weave/weave_panel_config.png" alt="パネルの設定メニュー" >}}

### 結果パネル

最後に、クエリ結果パネルは、選択した Query パネルと設定内容にもとづいてクエリ式の結果をレンダリングし、データをインタラクティブに表示します。以下は同じデータを Table と Plot で表示した例です。

{{< img src="/images/weave/result_panel_table.png" alt="テーブルの結果パネル" >}}

{{< img src="/images/weave/result_panel_plot.png" alt="プロットの結果パネル" >}}

## 基本操作
Query パネル内でよく使う操作は次のとおりです。
### Sort
列のオプションからソートできます:
{{< img src="/images/weave/weave_sort.png" alt="列のソート オプション" >}}

### Filter
クエリ内で直接フィルターするか、左上のフィルター ボタン（2 枚目の画像）を使っても構いません。
{{< img src="/images/weave/weave_filter_1.png" alt="クエリのフィルター構文" >}}
{{< img src="/images/weave/weave_filter_2.png" alt="フィルター ボタン" >}}

### Map
Map 操作は、リストを走査して各要素に関数を適用します。パネルのクエリで直接実行することも、列のオプションから新しい列を挿入して実行することもできます。
{{< img src="/images/weave/weave_map.png" alt="Map 操作のクエリ" >}}
{{< img src="/images/weave/weave_map.gif" alt="Map 列の挿入" >}}

### Groupby
groupby は、クエリで行うことも、列のオプションから行うこともできます。
{{< img src="/images/weave/weave_groupby.png" alt="Group by のクエリ" >}}
{{< img src="/images/weave/weave_groupby.gif" alt="Group by の列オプション" >}}

### Concat
concat 操作では 2 つのテーブルを連結できます。連結や結合はパネルの設定からも実行できます。
{{< img src="/images/weave/weave_concat.gif" alt="テーブルの連結" >}}

### Join
クエリ内でテーブルを直接結合することも可能です。次のクエリ式を考えてみましょう:
```python
project("luis_team_test", "weave_example_queries").runs.summary["short_table_0"].table.rows.concat.join(\
project("luis_team_test", "weave_example_queries").runs.summary["short_table_1"].table.rows.concat,\
(row) => row["Label"],(row) => row["Label"], "Table1", "Table2",\
"false", "false")
```
{{< img src="/images/weave/weave_join.png" alt="テーブルの結合操作" >}}

左側のテーブルは次の式で生成しています:
```python
project("luis_team_test", "weave_example_queries").\
runs.summary["short_table_0"].table.rows.concat.join
```
右側のテーブルは次の式で生成しています:
```python
project("luis_team_test", "weave_example_queries").\
runs.summary["short_table_1"].table.rows.concat
```
Where:
* `(row) => row["Label"]` は各テーブルのセレクターで、どの列を結合キーにするかを指定します
* `"Table1"` と `"Table2"` は、結合後に各テーブルを表す名前です
* `true` と `false` は、left / right の inner / outer 結合設定に対応します


## Runs オブジェクト
Query パネルを使って `runs` オブジェクトにアクセスできます。Run オブジェクトは実験の記録を保存します。詳細は [Accessing Runs object](https://wandb.ai/luis_team_test/weave_example_queries/reports/Weave-queries---Vmlldzo1NzIxOTY2?accessToken=bvzq5hwooare9zy790yfl3oitutbvno2i6c2s81gk91750m53m2hdclj0jvryhcr#3.-accessing-runs-object) を参照してください。概要として、`runs` オブジェクトには次が利用できます:
* `summary`: run の結果を要約する情報の辞書。精度や損失のようなスカラーや、大きなファイルを含むこともあります。デフォルトでは、`wandb.Run.log()` はログされた時系列の最終値を summary に設定します。summary の内容は直接設定することもできます。summary は run の出力だと考えてください。
* `history`: モデルのトレーニング中に変化する値（損失など）を保存するための辞書のリスト。`wandb.Run.log()` はこのオブジェクトに追記します。
* `config`: run の設定情報（トレーニング run のハイパーパラメーターや、データセット Artifact を作成する run の前処理メソッドなど）の辞書。run の「入力」に相当します。
{{< img src="/images/weave/weave_runs_object.png" alt="Runs オブジェクトの構造" >}}

## Artifacts にアクセス

Artifacts は W&B の中核概念です。バージョン管理された、名前付きのファイルやディレクトリーのコレクションです。Artifacts を使って、モデルの重み、データセット、その他のファイルやディレクトリーを追跡できます。Artifacts は W&B に保存され、ダウンロードしたり、他の run で利用したりできます。詳細と例は [Accessing Artifacts](https://wandb.ai/luis_team_test/weave_example_queries/reports/Weave-queries---Vmlldzo1NzIxOTY2?accessToken=bvzq5hwooare9zy790yfl3oitutbvno2i6c2s81gk91750m53m2hdclj0jvryhcr#4.-accessing-artifacts) を参照してください。Artifacts は通常 `project` オブジェクトからアクセスします:
* `project.artifactVersion()`: Project 内で、指定した名前とバージョンに対応する特定の Artifact バージョンを返します
* `project.artifact("")`: Project 内で、指定した名前の Artifact を返します。続いて `.versions` を使うと、その Artifact のすべてのバージョン一覧を取得できます
* `project.artifactType()`: Project 内で、指定した名前の `artifactType` を返します。続いて `.artifacts` を使うと、そのタイプの Artifact 一覧を取得できます
* `project.artifactTypes`: その Project 配下のすべての Artifact タイプの一覧を返します
{{< img src="/images/weave/weave_artifacts.png" alt="Artifact へのアクセス方法" >}}