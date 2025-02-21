---
title: Compare runs across projects
description: クロス プロジェクト レポート を使用して、2 つの異なる プロジェクト の run を比較します。
menu:
  default:
    identifier: ja-guides-core-reports-cross-project-reports
    parent: reports
weight: 60
---

クロスプロジェクト レポート を使用して、2 つの異なる project からの run を比較します。run セット テーブルの project セレクターを使用して、project を選択します。

{{< img src="/images/reports/howto_pick_a_different_project_to_draw_runs_from.gif" alt="異なる project 間で run を比較する" >}}

セクション内の 可視化 は、最初のアクティブな runset から列をプルします。折れ線グラフで探している メトリクス が表示されない場合は、セクションで最初にチェックされた run セットにその列があることを確認してください。

この機能は、 時系列 の履歴 データをサポートしていますが、異なる project から異なるサマリー メトリクス をプルすることはサポートしていません。つまり、別の project にのみ ログ 記録されている列から散布図を作成することはできません。

2 つの project から run を比較する必要があり、列が機能しない場合は、一方の project の run にタグを追加し、それらの run をもう一方の project に移動します。各 project からの run のみをフィルタリングできますが、 report には両方の run セットのすべての列が含まれます。

## 閲覧専用 report リンク

private project または Team project 内の report への閲覧専用リンクを共有します。

{{< img src="/images/reports/magic-links.gif" alt="" >}}

閲覧専用 report リンク は、秘密の アクセス トークン を URL に追加するため、リンクを開いた人は誰でもページを閲覧できます。誰でもマジックリンクを使用して、最初に ログ インしなくても report を表示できます。[W&B Local]({{< relref path="/guides/hosting/" lang="ja" >}}) プライベート クラウド インストールをご利用のお客様の場合、これらのリンクはファイアウォールの内側に保持されるため、プライベート インスタンスへの アクセス 権と閲覧専用リンクへの アクセス 権を持つ チーム のメンバーのみが report を閲覧できます。

**閲覧専用モード** では、 ログ インしていない人は、チャートを表示したり、マウスオーバーして 値 のツールチップを表示したり、チャートを拡大/縮小したり、テーブル内の列をスクロールしたりできます。閲覧モードでは、新しいチャートや新しいテーブル クエリを作成して data を探索することはできません。report リンク への閲覧専用の訪問者は、run をクリックして run ページに移動することはできません。また、閲覧専用の訪問者は共有モーダルを見ることはできず、代わりにホバーすると `Sharing not available for view only access` というツールチップが表示されます。

{{% alert color="info" %}}
マジックリンクは、「Private」および「Team」 project でのみ利用できます。「Public」（誰でも閲覧可能）または「Open」（誰でも run を閲覧および投稿可能） project の場合、この project は公開されており、リンクを持っている人なら誰でも利用できることを意味するため、リンクをオン/オフにすることはできません。
{{% /alert %}}

## グラフを report に送信する

workspace から report にグラフを送信して、進捗状況を追跡します。report にコピーするチャートまたは パネル のドロップダウン メニューをクリックし、**Add to report** をクリックして、宛先の report を選択します。
