---
title: プロジェクト
description: モデルのバージョンを比較し、スクラッチワークスペースで結果を探索し、ノートや可視化を保存するために学びをレポートにエクスポートする
menu:
  default:
    identifier: ja-guides-models-track-project-page
    parent: experiments
weight: 3
---

A *プロジェクト* は、結果の可視化、実験の比較、アーティファクトの閲覧とダウンロード、オートメーションの作成などを行うための中央の場所です。

{{% alert %}}
各プロジェクトには、その公開範囲を決定する公開設定があります。公開範囲についての詳細は、[プロジェクトの公開範囲]({{< relref path="/guides/hosting/iam/access-management/restricted-projects.md" lang="ja" >}})を参照してください。
{{% /alert %}}

各プロジェクトには、サイドバーからアクセスできる次の項目が含まれています:

* [**Overview**]({{< relref path="project-page.md#overview-tab" lang="ja" >}}): プロジェクトのスナップショット
* [**Workspace**]({{< relref path="project-page.md#workspace-tab" lang="ja" >}}): 個人の可視化サンドボックス
* [**Runs**]({{< relref path="#runs-tab" lang="ja" >}}): プロジェクト内のすべての run を一覧表示するテーブル
* **Automations**: プロジェクトで設定されたオートメーション
* [**Sweeps**]({{< relref path="project-page.md#sweeps-tab" lang="ja" >}}): 自動探索と最適化
* [**Reports**]({{< relref path="project-page.md#reports-tab" lang="ja" >}}): メモ、run、およびグラフの保存されたスナップショット
* [**Artifacts**]({{< relref path="#artifacts-tab" lang="ja" >}}): すべての run およびその run に関連するアーティファクトを含む

## Overview タブ

* **Project name**: プロジェクトの名前。W&B は、指定したプロジェクトフィールド名で run を初期化するときにプロジェクトを作成します。右上隅の **Edit** ボタンを選択することで、いつでもプロジェクト名を変更できます。
* **Description**: プロジェクトの説明。
* **Project visibility**: プロジェクトの公開範囲。それにアクセスできる人を決定する公開設定です。詳細は[プロジェクトの公開範囲]({{< relref path="/guides/hosting/iam/access-management/restricted-projects.md" lang="ja" >}})を参照してください。
* **Last active**: このプロジェクトにデータが最後にログ記録された日時のタイムスタンプ
* **Owner**: このプロジェクトを所有するエンティティ
* **Contributors**: このプロジェクトに貢献するユーザーの数
* **Total runs**: このプロジェクトの総 run 数
* **Total compute**: プロジェクト内のすべての run 時間を合計して得られるトータル
* **Undelete runs**: ドロップダウンメニューをクリックし、「Undelete all runs」をクリックしてプロジェクト内の削除された run を復元します。
* **Delete project**: 右上隅のドットメニューをクリックしてプロジェクトを削除

[ライブ例を見る](https://app.wandb.ai/example-team/sweep-demo/overview)

{{< img src="/images/track/overview_tab_image.png" alt="" >}}

## Workspace タブ

プロジェクトの*ワークスペース*は、実験を比較するための個人的なサンドボックスを提供します。プロジェクトを使用して、比較できるモデルを整理し、異なるアーキテクチャー、ハイパーパラメーター、データセット、プロセッシングなどで同じ問題に取り組みます。

**Runs Sidebar**: プロジェクト内のすべての run のリスト。

* **Dot menu**: サイドバーの行にカーソルを合わせると、左側にメニューが表示されます。このメニューを使用して、run の名前を変更、run の削除、または active run の停止を行います。
* **Visibility icon**: グラフで run をオンまたはオフにするために目のアイコンをクリックします。
* **Color**: run の色を私たちのプリセットの1つまたはカスタムカラーに変更します。
* **Search**: 名前で run を検索します。これにより、プロットで表示される run もフィルタリングされます。
* **Filter**: サイドバーフィルターを使用して、表示される run のセットを絞り込みます。
* **Group**: 設定した列を選択して run を動的にグループ化します。例えば、アーキテクチャーでグループ化することができます。グループ化すると、プロットに平均値に沿った線が表示され、グラフ上のポイントの分散の地域を示す影が表示されます。
* **Sort**: 最小の損失や最大の精度を持つ run など、value で run をソートします。ソートは、どの run がグラフに表示されるかに影響します。
* **Expand button**: サイドバーを完全なテーブルに拡張します。
* **Run count**: 上部のかっこ内の数字は、プロジェクト内のすべての run 数です。数字 (N visualized) は、目のアイコンがオンになっていて、各プロットで可視化可能な run 数です。以下の例では、グラフは183 runのうち最初の10 runのみを表示しています。表示される run の最大数を増やすには、グラフを編集します。

[Runs タブ](#runs-tab)で列をピン止め、非表示、または順序を変更すると、Runs サイドバーにもこれらのカスタマイズが反映されます。

**Panels layout**: このスクラッチスペースを使用して、結果を探索し、チャートを追加および削除し、異なるメトリクスに基づいてモデルのバージョンを比較します。

[ライブ例を見る](https://app.wandb.ai/example-team/sweep-demo)

{{< img src="/images/app_ui/workspace_tab_example.png" alt="" >}}

### Add a section of panels

セクションのドロップダウンメニューをクリックし、「Add section」をクリックして、セクションを作成します。セクションの名前を変更したり、ドラッグして順序を再編成したり、セクションを展開または折りたたむことができます。

各セクションには右上隅にオプションがあります:

* **Switch to custom layout**: カスタムレイアウトでは、個々のパネルのサイズを変更できます。
* **Switch to standard layout**: 標準レイアウトでは、セクション内のすべてのパネルのサイズを一括で変更でき、ページネーションが利用できます。
* **Add section**: ドロップダウンメニューから上または下にセクションを追加するか、ページの下部のボタンをクリックして新しいセクションを追加します。
* **Rename section**: セクションのタイトルを変更します。
* **Export section to report**: このパネルのセクションを新しいレポートに保存します。
* **Delete section**: セクション全体とすべてのチャートを削除します。これは、ワークスペースバーのページ下部にあるアンドゥボタンで取り消すことができます。
* **Add panel**: プラスボタンをクリックしてセクションにパネルを追加します。

{{< img src="/images/app_ui/add-section.gif" alt="" >}}

### Move panels between sections

ドラッグアンドドロップしてセクションに整理します。また、パネルの右上隅にある「Move」ボタンをクリックしてセクションを選択し、パネルを移動します。

{{< img src="/images/app_ui/move-panel.gif" alt="" >}}

### Resize panels

* **Standard layout**: すべてのパネルは同じサイズに保たれ、ページがあります。セクションの右下隅をクリックアンドドラッグして、セクションのサイズを変更します。
* **Custom layout**: 各パネルは個別にサイズが設定されており、ページはありません。

{{< img src="/images/app_ui/resize-panel.gif" alt="" >}}

### Search for metrics

ワークスペースの検索ボックスを使用してパネルを絞り込みます。この検索はパネルのタイトルに一致し、デフォルトでは表示されているメトリクスの名前となります。

{{< img src="/images/app_ui/search_in_the_workspace.png" alt="" >}}

## Runs タブ

Runs タブを使用して、run をフィルタリング、グループ化、およびソートします。

{{< img src="/images/runs/run-table-example.png" alt="" >}}

次のタブには、Runs タブで実行できる一般的な操作がいくつか示されています。

{{< tabpane text=true >}}
   {{% tab header="Customize columns" %}}
Runs タブは、プロジェクト内の run の詳細を表示します。デフォルトで多数の列が表示されます。

- すべての列を表示するには、ページを水平にスクロールします。
- 列の順序を変更するには、列を左右にドラッグします。
- 列をピン止めするには、列名にカーソルを合わせて、表示されるアクションメニュー `...` をクリックし、次に **Pin column** をクリックします。ピン止めされた列は、ページの左側に、**Name** 列の後に表示されます。ピン止めされた列を取り除くには、**Unpin column** を選択します。
- 列を非表示にするには、列名にカーソルを合わせて、表示されるアクションメニュー `...` をクリックし、次に **Hide column** をクリックします。現在非表示のすべての列を表示するには、**Columns** をクリックします。
- 複数の列を一度に表示、非表示、ピン止め、およびピン解除するには、**Columns** をクリックします。
  - 非表示の列の名前をクリックして表示します。
  - 表示列の名前をクリックして非表示にします。
  - 表示列の横のピンアイコンをクリックしてピン止めします。

Runs タブをカスタマイズすると、そのカスタマイズは[Workspace タブ]({{< relref path="#workspace-tab" lang="ja" >}})の **Runs** セレクタにも反映されます。
   {{% /tab %}}

   {{% tab header="Sort" %}}
指定した列の値でテーブル内のすべての行をソートします。

1. 列タイトルにマウスをホバーします。ケバブメニューが表示されます（三つの垂直ドット）。
2. ケバブメニュー（三つの垂直ドット）を選択します。
3. **Sort Asc** または **Sort Desc** を選択して、行を昇順または降順にソートします。

{{< img src="/images/data_vis/data_vis_sort_kebab.png" alt="See the digits for which the model most confidently guessed '0'." >}}

上の画像は、`val_acc` というテーブル列のソートオプションを表示する方法を示しています。
   {{% /tab %}}
   {{% tab header="Filter" %}}
ダッシュボードの左上の **Filter** ボタンを使って、式に基づいてすべての行をフィルタリングします。

{{< img src="/images/data_vis/filter.png" alt="See only examples which the model gets wrong." >}}

**Add filter** を選択して、行に1つ以上のフィルターを追加します。3つのドロップダウンメニューが表示され、左から右にかけて、フィルターのタイプは次のようになります: 列名、演算子、および値

|                   | 列名 | バイナリ関係    | 値       |
| -----------       | ----------- | ----------- | ----------- |
| 許可される値   | String       |  &equals;, &ne;, &le;, &ge;, IN, NOT IN,  | 整数, 浮動小数点数, 文字列, タイムスタンプ, null |

式エディタは、オートコンプリートを使用して、列名と論理述語構造などの用語ごとのオプションのリストを表示します。複数の論理述部を使用して「ands」または「or」で式を組み合わせることができます（場合によっては括弧を使用します）。

{{< img src="/images/data_vis/filter_example.png" alt="" >}}
上の画像は、`val_loss` 列に基づいたフィルターを示しています。このフィルターは、1以下の検証損失を持つ run を表示します。
   {{% /tab %}}
   {{% tab header="Group" %}}
特定の列の値で **Group by** ボタンを使い、すべての行をグループ化します。

{{< img src="/images/data_vis/group.png" alt="The truth distribution shows small errors: 8s and 2s are confused for 7s and 9s for 2s." >}}

デフォルトでは、これにより他の数値列がヒストグラムに変わり、グループ全体のその列の値の分布を示します。グループ化は、データ内の高レベルのパターンを理解するのに便利です。
   {{% /tab %}}
{{< /tabpane >}}

## Reports タブ

結果のスナップショットを一か所で確認し、チームと学びを共有します。

{{< img src="/images/app_ui/reports-tab.png" alt="" >}}

## Sweeps タブ

[スイープ]({{< relref path="/guides/models/sweeps/" lang="ja" >}})をプロジェクトから新たに開始します。

{{< img src="/images/app_ui/sweeps-tab.png" alt="" >}}

## Artifacts タブ

プロジェクトに関連付けられたすべての[Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}})をトレーニングデータセットや [ファインチューンされたモデル]({{< relref path="/guides/core/registry/" lang="ja" >}})から[メトリクスやメディアのテーブル]({{< relref path="/guides/models/tables/tables-walkthrough.md" lang="ja" >}})まで表示します。

### Overview パネル

{{< img src="/images/app_ui/overview_panel.png" alt="" >}}

概要パネルでは、アーティファクトの名前やバージョン、変更を検出し重複を防止するために使用されるハッシュダイジェスト、作成日、およびエイリアスなど、アーティファクトに関するさまざまな高レベル情報を見つけることができます。ここでエイリアスを追加または削除し、バージョンおよびアーティファクト全体に対してメモを取ることができます。

### Metadata パネル

{{< img src="/images/app_ui/metadata_panel.png" alt="" >}}

メタデータパネルは、アーティファクトが構築されたときに提供されるメタデータへのアクセスを提供します。このメタデータには、アーティファクトを再構築するために必要な設定引数、詳細情報を見つけるためのURL、またはアーティファクトをログする際に生成されたメトリクスが含まれている場合があります。さらに、アーティファクトを生成する run の設定や、アーティファクトをログする際の履歴メトリクスを見ることができます。

### Usage パネル

{{< img src="/images/app_ui/usage_panel.png" alt="" >}}

Usage パネルは、ウェブアプリの外部で、例えばローカルマシン上で使用するためにアーティファクトをダウンロードするためのコードスニペットを提供します。このセクションはまた、アーティファクトを出力した run 及びアーティファクトを入力として使用する run へのリンクも示しています。

### Files パネル

{{< img src="/images/app_ui/files_panel.png" alt="" >}}

ファイルパネルは、アーティファクトに関連付けられたファイルとフォルダを一覧表示します。W&B は特定のファイルを run 用に自動的にアップロードします。例えば、`requirements.txt` は run が使用した各ライブラリのバージョンを示し、`wandb-metadata.json` および `wandb-summary.json` は run に関する情報を含みます。他のファイルもアーティファクトやメディアなど、run の設定に応じてアップロードされる場合があります。このファイルツリーをナビゲートし、W&B ウェブアプリで直接内容を確認することができます。

Artifacts に関連付けられた[テーブル]({{< relref path="/guides/models/tables//tables-walkthrough.md" lang="ja" >}})は、このコンテキストでは特にリッチでインタラクティブです。Artifacts で Tables を使用する方法についての詳細は[こちら]({{< relref path="/guides/models/tables//visualize-tables.md" lang="ja" >}})から学べます。

{{< img src="/images/app_ui/files_panel_table.png" alt="" >}}

### Lineage パネル

{{< img src="/images/app_ui/lineage_panel.png" alt="" >}}

リネージパネルは、プロジェクトに関連付けられたすべてのアーティファクトとそれらをお互いに接続する run を表示します。run タイプはブロックとして、アーティファクトは円として表示され、与えられたタイプの run が与えられたタイプのアーティファクトを消費または生成するときに矢印で示されます。左側の列で選択された特定のアーティファクトのタイプが強調表示されます。

個々のアーティファクトバージョンとそれらを接続する特定の run を表示するには、爆発切り替えをクリックしてください。

### Action History Audit タブ

{{< img src="/images/app_ui/action_history_audit_tab_1.png" alt="" >}}

{{< img src="/images/app_ui/action_history_audit_tab_2.png" alt="" >}}

アクションヒストリー監査タブは、コレクションのすべてのエイリアスアクションとメンバーシップの変更を示しており、リソースの進化全体を監査できます。

### Versions タブ

{{< img src="/images/app_ui/versions_tab.png" alt="" >}}

バージョン タブは、アーティファクトのすべてのバージョンと、バージョンがログ記録された時点での Run History の各数値のカラムを示しています。これにより、パフォーマンスを比較し、興味のあるバージョンを迅速に特定することができます。

## プロジェクトにスターを付ける

プロジェクトにスターを付けると、そのプロジェクトを重要としてマークします。あなたとあなたのチームがスター付きで重要としてマークしたプロジェクトは、組織のホームページの上部に表示されます。

たとえば、次の画像では、`zoo_experiment` と `registry_demo` の 2 つのプロジェクトが重要としてマークされています。これらのプロジェクトは、組織のホームページの上部の **スター付きプロジェクト** セクション内に表示されています。
{{< img src="/images/track/star-projects.png" alt="" >}}

プロジェクトを重要としてマークする方法は2つあります: プロジェクトのオーバービュータブ内またはチームのプロファイルページ内です。

{{< tabpane text=true >}}
    {{% tab header="Project overview" %}}
1. W&B App の `https://wandb.ai/<team>/<project-name>` で W&B プロジェクトに移動します。
2. プロジェクトサイドバーから **Overview** タブを選択します。
3. 右上隅の **Edit** ボタンの横にある星アイコンを選択します。

{{< img src="/images/track/star-project-overview-tab.png" alt="" >}}
    {{% /tab %}}
    {{% tab header="Team profile" %}}
1. チームのプロファイルページにある `https://wandb.ai/<team>/projects` に移動します。
2. **Projects** タブを選択します。
3. スターを付けたいプロジェクトの横にマウスをホバーし、表示された星アイコンをクリックします。

たとえば、次の画像は "Compare_Zoo_Models" プロジェクトの横にある星アイコンを示しています。
{{< img src="/images/track/star-project-team-profile-page.png" alt="" >}}
    {{% /tab %}}
{{< /tabpane >}}

組織名を左上隅のアプリ内でクリックし、プロジェクトが組織のランディングページに表示されることを確認します。

## プロジェクトを削除する

プロジェクトを削除するには、オーバービュータブの右の三点リーダーをクリックします。

{{< img src="/images/app_ui/howto_delete_project.gif" alt="" >}}

プロジェクトが空の場合は、右上のドロップダウンメニューをクリックして **Delete project** を選択し、削除できます。

{{< img src="/images/app_ui/howto_delete_project_2.png" alt="" >}}

## プロジェクトにメモを追加する

プロジェクトにメモを追加するには、説明オーバービューまたはワークスペース内のマークダウンパネルとして行います。

### プロジェクトに説明オーバービューを追加

ページに追加した説明は、プロファイルの **Overview** タブに表示されます。

1. W&B プロジェクトに移動
2. プロジェクトサイドバーから **Overview** タブを選択
3. 右上隅の **Edit** を選択
4. **Description** フィールドにメモを追加
5. **Save** ボタンを選択

{{% alert title="Create reports to create descriptive notes comparing runs" %}}
W&B レポートを作成して、プロットやマークダウンを並べて追加することもできます。異なる run を示すために異なるセクションを使用し、取り組んだことについてのストーリーを語ります。
{{% /alert %}}

### ワークスペースに run のメモを追加

1. W&B プロジェクトに移動
2. プロジェクトサイドバーから **Workspace** タブを選択
3. 右上のコーナーから **Add panels** ボタンを選択
4. 表示されるモーダルから **TEXT AND CODE** ドロップダウンを選択
5. **Markdown** を選択
6. ワークスペース内に表示されるマークダウンパネルにメモを追加します。