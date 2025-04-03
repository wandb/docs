---
title: Filter and search runs
description: プロジェクトページでサイドバーとテーブルを使用する方法
menu:
  default:
    identifier: ja-guides-models-track-runs-filter-runs
    parent: what-are-runs
---

WandB にログされた run からの洞察を得るには、プロジェクトページを使用してください。**Workspace** ページと **Runs** ページの両方から、run をフィルタリングおよび検索できます。

## run をフィルタリングする

フィルターボタンを使用して、ステータス、タグ、またはその他のプロパティに基づいて run をフィルタリングします。

### タグで run をフィルタリングする

フィルターボタンを使用して、タグに基づいて run をフィルタリングします。

{{< img src="/images/app_ui/filter_runs.gif" alt="" >}}

### 正規表現で run をフィルタリングする

正規表現で目的の検索結果が得られない場合は、[タグ]({{< relref path="tags.md" lang="ja" >}})を使用して、Runs Table で run をフィルタリングできます。タグは、run の作成時または完了後に追加できます。タグが run に追加されると、以下の gif に示すようにタグフィルターを追加できます。

{{< img src="/images/app_ui/tags.gif" alt="If regex doesn't provide you the desired results, you can make use of tags to filter out the runs in Runs Table" >}}

## run を検索する

[正規表現](https://dev.mysql.com/doc/refman/8.0/en/regexp.html)を使用して、指定した正規表現に一致する run を検索します。検索ボックスにクエリを入力すると、**Workspace** 上のグラフに表示される run が絞り込まれるとともに、テーブルの行もフィルタリングされます。

## run をグループ化する

1 つまたは複数の列 (非表示の列を含む) で run をグループ化するには:

1. 検索ボックスの下にある、罫線が引かれた用紙のような **Group** ボタンをクリックします。
2. 結果をグループ化する 1 つまたは複数の列を選択します。
3. グループ化された run の各セットは、デフォルトで折りたたまれています。展開するには、グループ名の横にある矢印をクリックします。

## 最小値と最大値で run を並べ替える
ログに記録されたメトリクスの最小値または最大値で、run テーブルを並べ替えます。これは、記録された最高 (または最低) の値を表示する場合に特に役立ちます。

次のステップでは、記録された最小値または最大値に基づいて、特定のメトリクスで run テーブルを並べ替える方法について説明します。

1. 並べ替えに使用するメトリクスを含む列にマウスを合わせます。
2. ケバブメニュー (縦の 3 本線) を選択します。
3. ドロップダウンから、**Show min** または **Show max** を選択します。
4. 同じドロップダウンから、**Sort by asc** または **Sort by desc** を選択して、それぞれ昇順または降順で並べ替えます。

{{< img src="/images/app_ui/runs_min_max.gif" alt="" >}}

## run の検索終了時間

クライアント プロセスからの最後のハートビートをログに記録する `End Time` という名前の列を提供します。このフィールドはデフォルトで非表示になっています。

{{< img src="/images/app_ui/search_run_endtime.png" alt="" >}}

## Runs Table を CSV にエクスポート

ダウンロードボタンを使用して、すべての run、ハイパーパラメーター 、およびサマリー メトリクスのテーブルを CSV にエクスポートします。

{{< img src="/images/app_ui/export_to_csv.gif" alt="" >}}