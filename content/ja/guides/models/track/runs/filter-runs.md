---
title: Filter and search runs
description: プロジェクトページでサイドバーとテーブルを使用する方法
menu:
  default:
    identifier: ja-guides-models-track-runs-filter-runs
    parent: what-are-runs
---

W&B にログされた run から得られた知見を、 プロジェクト ページで活用しましょう。

## run のフィルタリング

フィルタ ボタンを使って、ステータス、タグ、その他のプロパティに基づいて run をフィルタリングします。

### タグによる run のフィルタリング

フィルタ ボタンを使って、タグに基づいて run をフィルタリングします。

{{< img src="/images/app_ui/filter_runs.gif" alt="" >}}

### 正規表現による run のフィルタリング

正規表現で期待する結果が得られない場合は、[タグ]({{< relref path="tags.md" lang="ja" >}}) を使用して Runs Table 内の run をフィルタリングできます。タグは、run の作成時または完了後に追加できます。タグが run に追加されると、以下の gif に示すようにタグ フィルタを追加できます。

{{< img src="/images/app_ui/tags.gif" alt="If regex doesn't provide you the desired results, you can make use of tags to filter out the runs in Runs Table" >}}

## run 名の検索

[regex](https://dev.mysql.com/doc/refman/8.0/en/regexp.html) を使用して、指定した正規表現に一致する run を検索します。検索ボックスにクエリを入力すると、 ワークスペース 上のグラフに表示される run が絞り込まれるだけでなく、テーブルの行もフィルタリングされます。

## 最小値と最大値で run をソートする
ログされた メトリクス の最小値または最大値で、runs table をソートします。これは、記録された最高 (または最低) の 値 を表示する場合に特に便利です。

以下の手順では、記録された最小値または最大値に基づいて、特定の メトリクス で run テーブルをソートする方法について説明します。

1. ソートする メトリクス がある列にマウスを合わせます。
2. ケバブ メニュー (縦の三本線) を選択します。
3. ドロップダウンから、[**最小値の表示**] または [**最大値の表示**] を選択します。
4. 同じドロップダウンから、[**昇順でソート**] または [**降順でソート**] を選択して、それぞれ昇順または降順でソートします。

{{< img src="/images/app_ui/runs_min_max.gif" alt="" >}}

## run の終了時間の検索

クライアント プロセス からの最後のハートビートをログする `End Time` という名前の列があります。このフィールドはデフォルトで非表示になっています。

{{< img src="/images/app_ui/search_run_endtime.png" alt="" >}}

## runs table を CSV にエクスポートする

すべての run 、 ハイパーパラメーター 、およびサマリー メトリクス のテーブルを、ダウンロード ボタンを使用して CSV にエクスポートします。

{{< img src="/images/app_ui/export_to_csv.gif" alt="" >}}
