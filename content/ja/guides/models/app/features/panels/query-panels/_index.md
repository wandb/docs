---
title: クエリパネル
description: このページのいくつかの機能はベータ版であり、フィーチャーフラグによって非表示になっています。すべての関連機能を有効にするには、プロフィールページの自己紹介に
  `weave-plot` を追加してください。
url: guides/app/features/panels/query-panels
menu:
  default:
    identifier: intro_query_panel
    parent: panels
cascade:
- url: guides/app/features/panels/query-panels/:filename
---

{{% alert %}}
W&B Weave（W&B の生成 AI アプリケーション構築ツール群）をお探しですか？Weave のドキュメントはこちら: [wandb.me/weave](https://wandb.github.io/weave/?utm_source=wandb_docs&utm_medium=docs&utm_campaign=weave-nudge)
{{% /alert %}}

クエリパネルを使ってデータのクエリや対話的な可視化ができます。

{{< img src="/images/weave/pretty_panel.png" alt="クエリパネル" >}}




## クエリパネルの作成

クエリをワークスペースまたはレポート内に追加します。

{{< tabpane text=true >}}
{{% tab header="Project workspace" value="workspace" %}}

  1. 対象 Project の Workspace に移動します。
  2. 右上の `Add panel` をクリックします。
  3. ドロップダウンから `Query panel` を選択します。
  {{< img src="/images/weave/add_weave_panel_workspace.png" alt="Add panel のドロップダウン" >}}

{{% /tab %}}

{{% tab header="W&B Report" value="report" %}}

`/Query panel` と入力して選択します。

{{< img src="/images/weave/add_weave_panel_report_1.png" alt="Query panel オプション" >}}

または、特定の Run セットにクエリを紐づけることもできます。
1. レポート内で `/Panel grid` と入力して選択します。
2. `Add panel` ボタンをクリックします。
3. ドロップダウンから `Query panel` を選択します。

{{% /tab %}}
{{< /tabpane >}}
  

## クエリの構成要素

### 式（Expressions）

クエリ式を使って W&B に保存されたデータ（Runs, Artifacts, Models, Tables など）をクエリできます。

#### 例：テーブルをクエリする
W&B Table をクエリするとします。トレーニングコード内で `"cifar10_sample_table"` というテーブルを記録しています：

```python
import wandb
with wandb.init() as run:
  run.log({"cifar10_sample_table":<MY_TABLE>})
```

クエリパネル内で、次のようにテーブルをクエリできます：
```python
runs.summary["cifar10_sample_table"]
```
{{< img src="/images/weave/basic_weave_expression.png" alt="テーブルクエリ式" >}}

この例を分解すると：

* `runs`：Workspace 内の Query Panel Expressions で自動的に挿入される変数です。その Workspace で表示可能な Runs のリストを表します。[Run 内で利用できる各属性についてはこちらをご覧ください]({{< relref "../../../../track/public-api-guide.md#understanding-the-different-attributes" >}})。
* `summary`：Run の Summary オブジェクトを返す op です。op は _map 処理_ され、リスト内の各 Run に適用されるため、Summary オブジェクトのリストになります。
* `["cifar10_sample_table"]`：ブラケットで表す Pick op で、パラメータは `predictions`。Summary オブジェクトは辞書のように振る舞うため、この操作で各 Summary オブジェクトから `predictions` フィールドを抽出します。

対話的に独自のクエリを書く方法は [Query panel demo](https://wandb.ai/luis_team_test/weave_example_queries/reports/Weave-queries---Vmlldzo1NzIxOTY2?accessToken=bvzq5hwooare9zy790yfl3oitutbvno2i6c2s81gk91750m53m2hdclj0jvryhcr) をご参照ください。

### コンフィギュレーション

パネル左上のギアアイコンを選択して、クエリの設定パネルを展開できます。ここで、パネル種別や結果パネルのパラメータを設定できます。

{{< img src="/images/weave/weave_panel_config.png" alt="パネル設定メニュー" >}}

### 結果パネル

最後に、クエリ結果パネルはクエリ式の結果を、選択されたクエリパネル種別と設定に応じて対話的にデータを表示します。次の画像は、同じデータの Table 表示と Plot 表示例です。

{{< img src="/images/weave/result_panel_table.png" alt="Table 結果パネル" >}}

{{< img src="/images/weave/result_panel_plot.png" alt="Plot 結果パネル" >}}

## 基本操作
クエリパネル内でよく使う操作は以下の通りです。
### ソート
カラムオプションからソートできます：
{{< img src="/images/weave/weave_sort.png" alt="カラム並び替えオプション" >}}

### フィルター
クエリ内で直接フィルターするか、左上（2枚目写真）のフィルターボタンからも操作できます。
{{< img src="/images/weave/weave_filter_1.png" alt="クエリ内フィルター構文" >}}
{{< img src="/images/weave/weave_filter_2.png" alt="フィルターボタン" >}}

### マップ
Map 操作はリスト内の各要素に関数を適用します。パネルクエリで直接実行、またはカラムオプションから新しいカラムを挿入して利用できます。
{{< img src="/images/weave/weave_map.png" alt="Map 操作クエリ" >}}
{{< img src="/images/weave/weave_map.gif" alt="Map カラム挿入" >}}

### グループ化（Groupby）
クエリまたはカラムオプションから groupby できます。
{{< img src="/images/weave/weave_groupby.png" alt="Groupby クエリ" >}}
{{< img src="/images/weave/weave_groupby.gif" alt="Groupby カラムオプション" >}}

### 結合（Concat）
concat 操作は2つのテーブルを連結します。パネル設定からも結合・連結できます。
{{< img src="/images/weave/weave_concat.gif" alt="テーブル結合" >}}

### ジョイン（Join）
クエリ内で直接テーブル同士を join することも可能です。次のようなクエリ式を考えてみます：
```python
project("luis_team_test", "weave_example_queries").runs.summary["short_table_0"].table.rows.concat.join(\
project("luis_team_test", "weave_example_queries").runs.summary["short_table_1"].table.rows.concat,\
(row) => row["Label"],(row) => row["Label"], "Table1", "Table2",\
"false", "false")
```
{{< img src="/images/weave/weave_join.png" alt="テーブル結合操作" >}}

左側のテーブルは次で作成されています：
```python
project("luis_team_test", "weave_example_queries").\
runs.summary["short_table_0"].table.rows.concat.join
```
右側のテーブルは次で作成されています：
```python
project("luis_team_test", "weave_example_queries").\
runs.summary["short_table_1"].table.rows.concat
```
ポイントは以下の通りです：
* `(row) => row["Label"]`：それぞれのテーブルで、どのカラムで join するかを指定するセレクタ
* `"Table1"`、`"Table2"`： join 後の各テーブルの名前
* `true` / `false`：左側または右側の内部/外部 join 設定

## Runs オブジェクト
クエリパネルを使用して `runs` オブジェクトへアクセスします。Run オブジェクトは実験の記録を保持します。詳細は [Accessing runs object](https://wandb.ai/luis_team_test/weave_example_queries/reports/Weave-queries---Vmlldzo1NzIxOTY2?accessToken=bvzq5hwooare9zy790yfl3oitutbvno2i6c2s81gk91750m53m2hdclj0jvryhcr#3.-accessing-runs-object) でもご覧いただけます。概要として、`runs` オブジェクトには以下があります：
* `summary`：Run の結果の概要情報を持つ辞書。精度や損失などのスカラー値や、大きなファイルも含められます。デフォルトでは `wandb.Run.log()` で記録した時系列の最終値が summary になります。直接 summary の内容をセットすることも可能です。summary は Run のアウトプット的なイメージです。
* `history`：トレーニング中に変化する値（例：損失値など）を記録する辞書のリスト。`wandb.Run.log()` コマンドでこのオブジェクトに追記されます。
* `config`：Run の設定情報（例：ハイパーパラメータやデータセット作成時の前処理手法など）が格納された辞書。インプット情報と考えてください。
{{< img src="/images/weave/weave_runs_object.png" alt="Runs オブジェクト構造" >}}

## Artifacts へのアクセス

Artifacts は W&B のコアコンセプトです。ファイルやディレクトリーの名前付きバージョン管理コレクションです。Artifacts を使って、モデルの重み、データセット、その他あらゆるファイルやディレクトリを管理できます。Artifacts は W&B に保存され、他の Run でもダウンロード・利用可能です。詳細やサンプルは [Accessing artifacts](https://wandb.ai/luis_team_test/weave_example_queries/reports/Weave-queries---Vmlldzo1NzIxOTY2?accessToken=bvzq5hwooare9zy790yfl3oitutbvno2i6c2s81gk91750m53m2hdclj0jvryhcr#4.-accessing-artifacts) をご参照ください。Artifacts へは `project` オブジェクトからアクセスします：
* `project.artifactVersion()`：プロジェクト内で指定した名前＆バージョンのアーティファクトバージョンを返します
* `project.artifact("")`：プロジェクト内で指定した名前のアーティファクトを返します。`.versions` を使うとこのアーティファクトの全バージョンリストを取得できます
* `project.artifactType()`：プロジェクト内で指定した名前の `artifactType` を返します。その後 `.artifacts` でこのタイプの全アーティファクト一覧を取得可能です
* `project.artifactTypes`：プロジェクト配下の全アーティファクトタイプのリストを返します
{{< img src="/images/weave/weave_artifacts.png" alt="Artifacts アクセス方法" >}}