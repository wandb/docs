---
title: run のフィルタと検索
description: プロジェクトページでサイドバーとテーブルを使う方法
menu:
  default:
    identifier: ja-guides-models-track-runs-filter-runs
    parent: what-are-runs
---

プロジェクトページを使って、W&B にログされた Run からインサイトを得ましょう。**Workspace** ページや **Runs** ページの両方から Run をフィルタ・検索できます。

## Run のフィルタ

Run のステータス、[タグ]({{< relref path="#filter-runs-with-tags" lang="ja" >}})、[正規表現 (RegEx)]({{< relref path="#filter-runs-with-regular-expressions-regex" lang="ja" >}})、その他のプロパティで Run をフィルタできます。フィルタボタンをご利用ください。

Run の色編集・ランダム化・リセット方法については [Run の色をカスタマイズする]({{< relref path="guides/models/track/runs/run-colors" lang="ja" >}})をご覧ください。

### タグで Run をフィルタ

フィルタボタンを使って、Run をタグでフィルタできます。

1. プロジェクトサイドバーから **Runs** タブをクリックします。
2. Runs テーブル上部のじょうご型アイコン **Filter** ボタンを選択します。
3. 左から右へドロップダウンメニューで `"Tags"` を選び、論理演算子を選んで、検索値を入力します。

### 正規表現で Run をフィルタ

正規表現で期待通りの結果が得られない場合は、[タグ]({{< relref path="tags.md" lang="ja" >}})を利用して Runs Table 内の Run を絞り込むこともできます。Run 作成時、または完了後にタグを追加できます。タグが Run に追加された後、以下の GIF のようにタグフィルタを追加できます。

{{< img src="/images/app_ui/filter_runs.gif" alt="Filter runs by tags" >}}

1. プロジェクトサイドバーから **Runs** タブをクリックします。
2. Runs テーブル上部の検索ボックスをクリックします。
3. **RegEx** トグル（.*）が有効（青色）になっていることを確認します。
4. 検索ボックスに正規表現を入力します。

## Run の検索

正規表現 (RegEx) を使って、指定したパターンに合致する Run を検索できます。検索ボックスにクエリを入力すると、ワークスペース内のグラフとテーブルの行がその条件で絞り込まれます。

## Run のグループ化

1 つ以上の列（非表示の列も含む）によって Run をグループ化できます。

1. 検索ボックスの下にある、ノートシート型アイコンの **Group** ボタンをクリックします。
2. グループ化したい列を 1 つ以上選択します。
3. グループ化された Run セットはデフォルトで折りたたまれています。展開するにはグループ名の横矢印をクリックします。

## 最小値・最大値で Run をソート

記録されたメトリクスの最小値または最大値で Runs テーブルをソートします。最良（または最悪）の記録値を確認したい場合に便利です。

以下の手順で、特定メトリクスの最小または最大の値で Run テーブルを並び替えます。

1. ソートしたいメトリクスがある列にマウスカーソルを合わせます。
2. ケバブメニュー（三本線）を選択します。
3. ドロップダウンから **Show min** または **Show max** を選びます。
4. 同じドロップダウンで **Sort by asc** または **Sort by desc** を選択で、昇順・降順ソートを切り替えられます。

{{< img src="/images/app_ui/runs_min_max.gif" alt="Sort by min/max values" >}}

## Run の End Time 検索

クライアントプロセスから最後にハートビートを受信した時刻を記録する `End Time` という列を用意しています。このフィールドはデフォルトでは非表示です。

{{< img src="/images/app_ui/search_run_endtime.png" alt="End Time column" >}}

## Runs テーブルを CSV にエクスポート

すべての Run、ハイパーパラメーター、およびサマリーメトリクスのテーブルを、ダウンロードボタンで CSV にエクスポートできます。

{{< img src="/images/app_ui/export_to_csv.gif" alt="Modal with preview of export to CSV" >}}