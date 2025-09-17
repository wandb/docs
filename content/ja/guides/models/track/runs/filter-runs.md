---
title: runs のフィルタリングと検索
description: Project ページのサイドバーとテーブルの使い方
menu:
  default:
    identifier: ja-guides-models-track-runs-filter-runs
    parent: what-are-runs
---

W&B にログされた run から洞察を得るには、Project ページを活用しましょう。**Workspace** ページと **Runs** ページのどちらからでも run のフィルタや検索ができます。

## run をフィルターする

ステータス、[タグ]({{< relref path="#filter-runs-with-tags" lang="ja" >}})、[正規表現 (RegEx)]({{< relref path="#filter-runs-with-regular-expressions-regex" lang="ja" >}})、その他のプロパティに基づいて、Filter ボタンで run をフィルターします。

run の色のカスタマイズについては、[run の色を編集・ランダム化・リセットする方法]({{< relref path="guides/models/track/runs/run-colors" lang="ja" >}})を参照してください。

### タグで run をフィルターする

Filter ボタンを使って、タグに基づいて run をフィルターします。

1. Project サイドバーから **Runs** タブをクリックします。
2. Runs テーブル上部にある、じょうごのアイコンの **Filter** ボタンを選択します。
3. 左から順に、ドロップダウンで `Tags` を選び、論理演算子を選び、フィルターの検索値を選択します。

### regex で run をフィルターする

regex で期待どおりの結果が得られない場合は、Runs Table の run を絞り込むために [tags]({{< relref path="tags.md" lang="ja" >}}) を利用できます。タグは run の作成時または完了後に追加できます。run にタグを追加したら、以下の GIF のようにタグフィルターを追加できます。

{{< img src="/images/app_ui/filter_runs.gif" alt="タグで run をフィルターする" >}}

1. Project サイドバーから **Runs** タブをクリックします。
2. Runs テーブル上部の検索ボックスをクリックします。
3. トグル **RegEx** (.*) が有効になっていることを確認します (トグルは青色になります)。
4. 検索ボックスに正規表現を入力します。

## run を検索する

正規表現 (RegEx) を使って、指定したパターンに一致する run を見つけます。検索ボックスにクエリを入力すると、Workspace 上のグラフに表示される run とテーブルの行の両方が絞り込まれます。

## run をグループ化する

1 つ以上の列 (非表示列を含む) で run をグループ化するには:

1. 検索ボックスの下にある、罫線の入った紙のアイコンの **Group** ボタンをクリックします。
1. グループ化の基準とする列を 1 つ以上選択します。
1. 各グループ化された run のセットは既定で折りたたまれています。展開するには、グループ名の横の矢印をクリックします。

## 最小値と最大値で run をソートする
ログされたメトリクスの最小値または最大値で Runs テーブルを並べ替えます。これは、記録された最良 (または最悪) の値を見たい場合に便利です。

次の手順は、記録された最小値または最大値に基づいて、特定のメトリクスで Runs テーブルを並べ替える方法を説明します。

1. 並べ替えたいメトリクスの列にマウスカーソルを重ねます。
2. ケバブメニュー (縦に並んだ 3 本線) を選択します。
3. ドロップダウンから **Show min** または **Show max** を選択します。
4. 同じドロップダウンから、昇順に並べ替えるには **Sort by asc**、降順に並べ替えるには **Sort by desc** を選択します。 

{{< img src="/images/app_ui/runs_min_max.gif" alt="最小/最大の値でソート" >}}

## run の End Time を検索する

クライアント プロセスからの最後のハートビートを記録する `End Time` という列を提供しています。フィールドはデフォルトで非表示です。

{{< img src="/images/app_ui/search_run_endtime.png" alt="End Time 列" >}}

## Runs テーブルを CSV にエクスポートする

ダウンロードボタンで、すべての run、ハイパーパラメーター、サマリー メトリクスのテーブルを CSV にエクスポートします。

{{< img src="/images/app_ui/export_to_csv.gif" alt="CSV へのエクスポートのプレビューを含むモーダル" >}}