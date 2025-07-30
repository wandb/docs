---
title: run をフィルタリングし検索する
description: プロジェクトページのサイドバーとテーブルの使い方
menu:
  default:
    identifier: ja-guides-models-track-runs-filter-runs
    parent: what-are-runs
---

プロジェクトページを使用して、W&B にログされた run からインサイトを得ることができます。**Workspace** ページと **Runs** ページの両方で、run をフィルタリングおよび検索できます。

## run をフィルタリングする

ステータス、タグ、またはその他のプロパティに基づいて、フィルター ボタンを使用して run をフィルタリングします。

### タグで run をフィルタリングする

フィルター ボタンを使用して、タグに基づいて run をフィルタリングします。

{{< img src="/images/app_ui/filter_runs.gif" alt="" >}}

### 正規表現で run をフィルタリングする

正規表現で望む結果が得られない場合は、[タグ]({{< relref path="tags.md" lang="ja" >}})を使用して Runs Table で run をフィルタリングすることができます。タグは run 作成時、または完了後に追加できます。一旦タグが run に追加されると、以下の gif に示されているように、タグ フィルターを追加できます。

{{< img src="/images/app_ui/tags.gif" alt="If regex doesn't provide you the desired results, you can make use of tags to filter out the runs in Runs Table" >}}

## run を検索する

指定した正規表現を使用して run を見つけるには、regex を使用します。検索ボックスにクエリを入力すると、ワークスペースのグラフで表示される run がフィルタリングされ、テーブルの行もフィルタリングされます。

## run をグループ化する

1 つまたは複数の列（隠し列を含む）で run をグループ化するには:

1. 検索ボックスの下にある、罫線のついた紙のように見える **Group** ボタンをクリックします。
2. 結果をグループ化する 1 つ以上の列を選択します。
3. グループ化された各 run セットはデフォルトで折りたたまれています。展開するには、グループ名の横にある矢印をクリックします。

## 最小値と最大値で run を並べ替える

ログされたメトリクスの最小値または最大値で run テーブルを並べ替えます。これは、記録された最良または最悪の値を表示したい場合に特に便利です。

次の手順は、記録された最小値または最大値に基づいて特定のメトリクスで run テーブルを並べ替える方法を説明します：

1. 並べ替えたいメトリクスを含む列にマウスを合わせます。
2. ケバブ メニュー（三本の縦線）を選択します。
3. ドロップダウンから、**Show min** または **Show max** を選択します。
4. 同じドロップダウンから、**Sort by asc** または **Sort by desc** を選択して、それぞれ昇順または降順で並べ替えます。

{{< img src="/images/app_ui/runs_min_max.gif" alt="" >}}

## run の終了時間を検索する

クライアントプロセスからの最後のハートビートをログする `End Time` という名前の列を提供します。このフィールドはデフォルトでは非表示になっています。

{{< img src="/images/app_ui/search_run_endtime.png" alt="" >}}

## run テーブルを CSV にエクスポートする

すべての run、ハイパーパラメーター、およびサマリーメトリクスのテーブルを、ダウンロード ボタンを使用して CSV にエクスポートします。

{{< img src="/images/app_ui/export_to_csv.gif" alt="" >}}