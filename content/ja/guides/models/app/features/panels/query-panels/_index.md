---
title: クエリパネル
description: このページの一部の機能はベータ版で、フィーチャーフラグによって非表示になっています。関連するすべての機能を利用するには、プロフィールページの自己紹介欄に
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
W&B Weave（生成 AI アプリケーション構築のためのツール群）をお探しですか？Weave のドキュメントはこちら: [wandb.me/weave](https://wandb.github.io/weave/?utm_source=wandb_docs&utm_medium=docs&utm_campaign=weave-nudge)
{{% /alert %}}

クエリパネルを使って、データのクエリやインタラクティブな可視化ができます。

{{< img src="/images/weave/pretty_panel.png" alt="クエリパネル" >}}



## クエリパネルの作成

ワークスペースまたはレポート内でクエリを追加できます。

{{< tabpane text=true >}}
{{% tab header="Project workspace" value="workspace" %}}

  1. 対象の Project の workspace に移動します。
  2. 画面右上の `Add panel` をクリックします。
  3. 表示されるドロップダウンから `Query panel` を選択します。
  {{< img src="/images/weave/add_weave_panel_workspace.png" alt="Add panel ドロップダウン" >}}

{{% /tab %}}

{{% tab header="W&B Report" value="report" %}}

`/Query panel` と入力して選択します。

{{< img src="/images/weave/add_weave_panel_report_1.png" alt="Query panel オプション" >}}

また、クエリを特定の runs セットに紐付けることもできます:
1. レポート内で `/Panel grid` と入力し、選択します。
2. `Add panel` ボタンをクリックします。
3. ドロップダウンから `Query panel` を選択します。

{{% /tab %}}
{{< /tabpane >}}
  

## クエリの構成要素

### Expressions（式）

Query expression を使って、W&B に保存されている runs、artifacts、models、tables など多様なデータにクエリを実行できます。

#### 例：テーブルにクエリする
W&B Table にクエリしたいとします。トレーニングコード内で "cifar10_sample_table" というテーブルをログした場合：

```python
import wandb
with wandb.init() as run:
  run.log({"cifar10_sample_table":<MY_TABLE>})
```

クエリパネル内で以下のようにクエリできます：
```python
runs.summary["cifar10_sample_table"]
```
{{< img src="/images/weave/basic_weave_expression.png" alt="Table クエリ式" >}}

この表現の内訳は以下のとおりです：

* `runs` は Workspace 内の Query Panel Expressions で自動的に注入される変数です。これはその workspace で見える全 runs のリストを表します。[run 内で利用可能な属性一覧はこちら]({{< relref path="../../../../track/public-api-guide.md#understanding-the-different-attributes" lang="ja" >}}) をご覧ください。
* `summary` は Run の Summary オブジェクトを返す op です。op は _マッピング_ されるため、この op は runs リスト内の各 Run に適用され、Summary オブジェクトのリストになります。
* `["cifar10_sample_table"]` はブラケットで表す Pick op で、ここでは `predictions` フィールドを各 Summary オブジェクトから取得します。Summary オブジェクトは辞書や map のように扱えます。

自分でクエリを書いてインタラクティブに学びたい場合は [Query panel demo](https://wandb.ai/luis_team_test/weave_example_queries/reports/Weave-queries---Vmlldzo1NzIxOTY2?accessToken=bvzq5hwooare9zy790yfl3oitutbvno2i6c2s81gk91750m53m2hdclj0jvryhcr) をご覧ください。

### 設定（Configurations）

パネル左上の歯車アイコンをクリックするとクエリ設定メニューが開きます。ここではパネルタイプや、表示する結果パネルのパラメータを設定できます。

{{< img src="/images/weave/weave_panel_config.png" alt="パネル設定メニュー" >}}

### 結果パネル

クエリの結果パネルでは、指定したクエリ expression の結果が選択したパネル形式でインタラクティブに表示されます。下記は同じデータの Table と Plot の例です。

{{< img src="/images/weave/result_panel_table.png" alt="Table 結果パネル" >}}

{{< img src="/images/weave/result_panel_plot.png" alt="Plot 結果パネル" >}}

## 基本操作
クエリパネル内で利用できる主な操作は以下のとおりです。

### ソート（Sort）
カラムのオプションからソートできます：
{{< img src="/images/weave/weave_sort.png" alt="カラムのソートオプション" >}}

### フィルター（Filter）
クエリ内で直接フィルターするか、左上のフィルターボタン（2枚目画像）からも可能です。
{{< img src="/images/weave/weave_filter_1.png" alt="クエリフィルター構文" >}}
{{< img src="/images/weave/weave_filter_2.png" alt="フィルターボタン" >}}

### マップ（Map）
Map 操作はリストをイテレートし、各要素に関数を適用します。パネルクエリで直接行うか、カラムオプションから新しい列を追加することもできます。
{{< img src="/images/weave/weave_map.png" alt="Map 操作クエリ" >}}
{{< img src="/images/weave/weave_map.gif" alt="Map 列の追加" >}}

### グループ化（Groupby）
グループ化もクエリまたはカラムオプションから実行可能です。
{{< img src="/images/weave/weave_groupby.png" alt="Group by クエリ" >}}
{{< img src="/images/weave/weave_groupby.gif" alt="Group by カラムオプション" >}}

### 結合（Concat）
Concat 操作で２つのテーブルを連結できます。パネル設定から連結や join を行うこともできます。
{{< img src="/images/weave/weave_concat.gif" alt="テーブル結合" >}}

### ジョイン（Join）
クエリ内でテーブルの join も可能です。例えば以下の expression をご覧ください：
```python
project("luis_team_test", "weave_example_queries").runs.summary["short_table_0"].table.rows.concat.join(\
project("luis_team_test", "weave_example_queries").runs.summary["short_table_1"].table.rows.concat,\
(row) => row["Label"],(row) => row["Label"], "Table1", "Table2",\
"false", "false")
```
{{< img src="/images/weave/weave_join.png" alt="テーブル join 操作" >}}

左側のテーブルは以下から生成しています：
```python
project("luis_team_test", "weave_example_queries").\
runs.summary["short_table_0"].table.rows.concat.join
```
右側のテーブルは以下から生成しています：
```python
project("luis_team_test", "weave_example_queries").\
runs.summary["short_table_1"].table.rows.concat
```
ポイントは以下の通りです：
* `(row) => row["Label"]` は各テーブルのセレクターで、どのカラムで join するか指定します。
* `"Table1"` `"Table2"` は join 後のテーブル名です。
* `true`, `false` は left/right の inner/outer join 設定用です。


## Runs オブジェクト
クエリパネルから `runs` オブジェクトにアクセスできます。Run オブジェクトは、実験管理の記録を保存します。詳細は [Accessing runs object](https://wandb.ai/luis_team_test/weave_example_queries/reports/Weave-queries---Vmlldzo1NzIxOTY2?accessToken=bvzq5hwooare9zy790yfl3oitutbvno2i6c2s81gk91750m53m2hdclj0jvryhcr#3.-accessing-runs-object) をご覧くださいが、おおまかな構成は次の通りです：

* `summary`: run の結果を要約する情報の辞書です。精度や損失などのスカラーや、大きなファイルも含められます。`wandb.Run.log()` はデフォルトで時系列データの最終値を summary にセットします。summary の内容は直接上書き可能です。まとめると summary ＝ run のアウトプットです。
* `history`: 各イテレーションごとに変化する値（トレーニング中の損失など）を格納するための辞書リストです。`wandb.Run.log()` コマンドで追記されます。
* `config`: run の設定情報（ハイパーパラメータやデータセット生成時の前処理方法など）。これが、run の "入力" と捉えられます。
{{< img src="/images/weave/weave_runs_object.png" alt="Runs オブジェクト構造" >}}

## Artifacts へのアクセス

Artifacts は W&B の重要な概念で、バージョン管理された名前付きのファイル・ディレクトリコレクションです。モデルの重み、データセット、その他のファイル・ディレクトリ追跡に利用します。Artifacts は W&B 上に保存され、ダウンロードしたり、他の runs で利用することもできます。より詳しい利用例やアクセス方法は [Accessing artifacts](https://wandb.ai/luis_team_test/weave_example_queries/reports/Weave-queries---Vmlldzo1NzIxOTY2?accessToken=bvzq5hwooare9zy790yfl3oitutbvno2i6c2s81gk91750m53m2hdclj0jvryhcr#4.-accessing-artifacts) をご参照ください。Artifacts は通常 `project` オブジェクトからアクセスします：

* `project.artifactVersion()`: 指定した名前・バージョンの特定の artifact version を返します
* `project.artifact("")`: 指定した名前の artifact を返します。`.versions` でこの artifact のすべてのバージョン一覧が取得可能です
* `project.artifactType()`: 指定した名前の `artifactType` を返します。`.artifacts` でこのタイプの全 artifacts のリストが取得できます
* `project.artifactTypes`: その Project 内の全ての artifact type のリストを返します
{{< img src="/images/weave/weave_artifacts.png" alt="Artifact アクセス方法" >}}