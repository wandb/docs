---
title: プロジェクト間での run の比較
description: クロスプロジェクト Reports を使用して、異なる 2 つの Projects 間の run を比較します。
menu:
  default:
    identifier: ja-guides-core-reports-cross-project-reports
    parent: reports
weight: 60
---

{{% alert %}}
[Projects にまたがる run の比較をデモするビデオ](https://www.youtube.com/watch?v=uD4if_nGrs4) (2 分) をご覧ください。
{{% /alert %}}

異なる 2 つの Projects の run を、クロスプロジェクト Reports で比較します。run セットテーブルの Project セレクターを使って、Project を選択します。

{{< img src="/images/reports/howto_pick_a_different_project_to_draw_runs_from.gif" alt="異なる Projects 間で run を比較する" >}}

このセクションの可視化は、最初のアクティブな run セットから列を取得します。折れ線グラフで探しているメトリクスが表示されない場合は、セクションで最初にチェックされた run セットにその列が用意されていることを確認してください。

この機能は時系列の履歴データをサポートしていますが、異なる Projects から異なるサマリーメトリクスを取得することはサポートしていません。つまり、別の Project にのみログされた列から散布図を作成することはできません。

2 つの Projects の run を比較する必要があり、列がうまく使えない場合は、一方の Project の run にタグを追加し、それらの run をもう一方の Project に移動してください。各 Project からの run のみをフィルタリングすることはできますが、Report には両方の run セットのすべての列が含まれます。

## 閲覧専用の Report リンク

プライベート Projects または Team Projects 内の Report への閲覧専用リンクを共有します。

{{< img src="/images/reports/magic-links.gif" alt="閲覧専用の Report リンク" >}}

閲覧専用の Report リンクは、URL に秘密のアクセストークンが追加されるため、リンクを開いた人は誰でもページを閲覧できます。誰でもマジックリンクを使って、最初にログインすることなく Report を閲覧できます。[W&B Local]({{< relref path="/guides/hosting/" lang="ja" >}}) プライベートクラウドのインスタレーションを利用しているお客様の場合、これらのリンクはファイアウォールの内側に留まるため、プライベートインスタンスへのアクセス _と_ 閲覧専用リンクへのアクセスがあるチームのメンバーのみが Report を閲覧できます。

{{% alert color="info" %}}
マジックリンクは、「Private」と「Team」Projects でのみ利用可能です。「Public」（誰でも閲覧可能）または「Open」（誰でも閲覧および run の貢献が可能）Projects の場合、この Project は公開されているため、リンクを持つ人なら誰でもすでに利用可能であることを意味するため、リンクをオン/オフにすることはできません。
{{% /alert %}}

## グラフを Report に送信する

進捗を追跡するために、Workspace からグラフを Report に送信します。Report にコピーしたいチャートまたはパネルのドロップダウンメニューをクリックし、**Add to report** をクリックして送信先の Report を選択します。