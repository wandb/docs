---
title: プロジェクト間で run を比較する
description: 異なる 2 つの Projects の run を、クロスプロジェクトレポートで比較しましょう。
menu:
  default:
    identifier: cross-project-reports
    parent: reports
weight: 60
---

{{% alert %}}
[異なる Projects 間での Runs 比較をデモした動画](https://www.youtube.com/watch?v=uD4if_nGrs4)（2分）をご覧ください。
{{% /alert %}}

異なる 2 つの Projects の Runs をクロス Project レポートで比較できます。Run セットテーブルの Project セレクターから比較したい Project を選択してください。

{{< img src="/images/reports/howto_pick_a_different_project_to_draw_runs_from.gif" alt="異なる Projects 間での Runs 比較" >}}

このセクションの可視化は、最初にアクティブな Runset からカラムを取得します。目的のメトリクスがラインプロットに表示されない場合、該当のカラムが最初にチェックされた Run セットに含まれているか確認してください。

この機能では時系列ライン上の履歴データに対応していますが、異なる Projects から異なるサマリーメトリクスを取得することはできません。つまり、他の Project でのみ記録されているカラムから散布図を作成することはできません。

もし 2 つの Projects の Runs を比較したいのにカラムがうまく表示されない場合は、1 つの Project の Runs にタグを付けてから、その Runs をもう一方の Project に移動してください。各 Project の Runs だけをフィルタリングすることは可能ですが、レポートには両方の Runs に対する全カラムが含まれます。

## 閲覧専用レポートリンク

プライベートな Project や Team Project にあるレポートの閲覧専用リンクを共有できます。

{{< img src="/images/reports/magic-links.gif" alt="閲覧専用レポートリンク" >}}

閲覧専用レポートリンクは URL にシークレットアクセストークンを追加するため、リンクを開いた誰でもページを閲覧できます。このマジックリンクを使えば、最初にログインしなくてもレポートを見ることができます。[W&B Local]({{< relref "/guides/hosting/" >}}) のプライベートクラウド環境のお客様の場合、これらのリンクはファイアウォール内に留まるため、自チームでプライベートインスタンスへのアクセス権 _かつ_ 閲覧専用リンクへのアクセス権があるメンバーだけがレポートを閲覧できます。

**閲覧専用モード** では、未ログインの方でもグラフの内容や値のツールチップを確認でき、グラフのズームイン・アウトやテーブルのカラムスクロールも可能です。ただし、このモードでは新しいグラフやテーブルクエリを作成してデータ探索することはできません。また、閲覧専用リンクからレポートを見る場合、Runs をクリックして Run ページに移動することはできません。さらに、共有モーダルも利用不可となり、ホバー時には「Sharing not available for view only access」（閲覧専用アクセスでは共有できません）というツールチップが表示されます。

{{% alert color="info" %}}
マジックリンクは「Private」および「Team」projects のみに利用できます。「Public」（誰でも閲覧可）や「Open」（誰でも閲覧・Runs 投稿可）の projects では、このリンクでアクセス ON/OFF の切り替えはできません。これらの projects はすでにリンクがあれば誰でも閲覧できるためです。
{{% /alert %}}

## グラフをレポートに送信

自分の Workspace からグラフをレポートに送って進捗を記録しましょう。コピーしたいグラフやパネルのドロップダウンメニューをクリックし、**Add to report** を選択して送り先のレポートを選びます。