---
title: Projects
description: モデルのバージョンを比較し、スクラッチ ワークスペース で結果を調査し、 学び をレポートにエクスポートして、メモと 可視化 を保存します。
menu:
  default:
    identifier: ja-guides-models-track-project-page
    parent: experiments
weight: 3
---

*project* は、結果の可視化、実験の比較、Artifactsの表示とダウンロード、オートメーションの作成などを行う中心的な場所です。

{{% alert %}}
各projectには、誰がそれにアクセスできるかを決定する可視性の設定があります。誰が project にアクセスできるかの詳細については、[Project visibility]({{< relref path="/guides/hosting/iam/access-management/restricted-projects.md" lang="ja" >}}) を参照してください。
{{% /alert %}}

各 project には、サイドバーからアクセスできる次のものが含まれています。

* [**Overview**]({{< relref path="project-page.md#overview-tab" lang="ja" >}}): project のスナップショット
* [**Workspace**]({{< relref path="project-page.md#workspace-tab" lang="ja" >}}): 個人的な可視化サンドボックス
* [**Runs**]({{< relref path="#runs-tab" lang="ja" >}}): project 内のすべての run をリストしたテーブル
* **Automations**: project で構成されたオートメーション
* [**Sweeps**]({{< relref path="project-page.md#sweeps-tab" lang="ja" >}}): 自動化された探索と最適化
* [**Reports**]({{< relref path="project-page.md#reports-tab" lang="ja" >}}): ノート、run、グラフの保存されたスナップショット
* [**Artifacts**]({{< relref path="#artifacts-tab" lang="ja" >}}): すべての run とその run に関連付けられた Artifacts が含まれています

## Overview タブ

* **Project name**: project の名前。W&B は、project フィールドに指定した名前で run を初期化すると、project を作成します。project の名前は、右上隅にある **Edit** ボタンを選択すると、いつでも変更できます。
* **Description**: project の説明。
* **Project visibility**: project の可視性。誰がアクセスできるかを決定する可視性設定。詳細については、[Project visibility]({{< relref path="/guides/hosting/iam/access-management/restricted-projects.md" lang="ja" >}}) を参照してください。
* **Last active**: この project に最後にデータが記録されたときのタイムスタンプ
* **Owner**: この project のエンティティ
* **Contributors**: この project に貢献する ユーザー の数
* **Total runs**: この project 内の run の総数
* **Total compute**: この合計を取得するために、project 内のすべての run 時間を合計します。
* **Undelete runs**: ドロップダウン メニューをクリックし、[Undelete all runs] をクリックして、project 内で削除された run を復元します。
* **Delete project**: 右隅にあるドット メニューをクリックして、project を削除します。

[ライブの例を見る](https://app.wandb.ai/example-team/sweep-demo/overview)

{{< img src="/images/track/overview_tab_image.png" alt="" >}}

## Workspace タブ

project の *workspace* は、実験を比較するための個人的なサンドボックスを提供します。異なるアーキテクチャ、ハイパーパラメーター、データセット、プロセッシングなどで同じ問題に取り組んでいる、比較できる Models を整理するために project を使用します。

**Runs Sidebar**: project 内のすべての run のリスト。

* **Dot menu**: サイドバーの行にカーソルを合わせると、左側にメニューが表示されます。このメニューを使用して、run の名前を変更したり、run を削除したり、アクティブな run を停止したりできます。
* **Visibility icon**: グラフ上の run のオンとオフを切り替えるには、目のアイコンをクリックします。
* **Color**: run の色をプリセットのいずれか、またはカスタムの色に変更します。
* **Search**: 名前で run を検索します。これにより、プロットで表示される run もフィルター処理されます。
* **Filter**: サイドバー フィルターを使用して、表示される run のセットを絞り込みます。
* **Group**: 設定列を選択して、たとえばアーキテクチャごとに run を動的にグループ化します。グループ化すると、プロットには平均値に沿った線と、グラフ上のポイントの分散の影付き領域が表示されます。
* **Sort**: 損失が最小または精度が最大の run など、run の並べ替えに使用する値を選択します。並べ替えは、グラフに表示される run に影響します。
* **Expand button**: サイドバーをテーブル全体に展開します。
* **Run count**: 上部のかっこ内の数値は、project 内の run の総数です。数値 (N visualized) は、目のアイコンがオンになっており、各プロットで可視化できる run の数です。以下の例では、グラフには 183 の run のうち最初の 10 個のみが表示されています。グラフを編集して、表示される run の最大数を増やします。

[Runs tab](#runs-tab) で列をピン留め、非表示、または順序を変更すると、Runs サイドバーにこれらのカスタマイズが反映されます。

**Panels layout**: このスクラッチ スペースを使用して、結果を調べたり、チャートを追加および削除したり、さまざまなメトリクスに基づいて Models のバージョンを比較したりできます。

[ライブの例を見る](https://app.wandb.ai/example-team/sweep-demo)

{{< img src="/images/app_ui/workspace_tab_example.png" alt="" >}}

### パネルのセクションを追加する

セクション ドロップダウン メニューをクリックし、[Add section] をクリックして、パネルの新しいセクションを作成します。セクションの名前を変更したり、ドラッグして再編成したり、セクションを展開および折りたたんだりできます。

各セクションの右上隅にはオプションがあります。

* **Switch to custom layout**: カスタム レイアウトでは、パネルのサイズを個別に変更できます。
* **Switch to standard layout**: 標準レイアウトでは、セクション内のすべてのパネルのサイズを一度に変更でき、ページネーションが提供されます。
* **Add section**: ドロップダウン メニューから上下にセクションを追加するか、ページの下部にあるボタンをクリックして新しいセクションを追加します。
* **Rename section**: セクションのタイトルを変更します。
* **Export section to report**: このパネルのセクションを新しい Report に保存します。
* **Delete section**: セクション全体とすべてのチャートを削除します。これは、ワークスペース バーのページの下部にある元に戻すボタンで元に戻すことができます。
* **Add panel**: プラス ボタンをクリックして、セクションにパネルを追加します。

{{< img src="/images/app_ui/add-section.gif" alt="" >}}

### セクション間でパネルを移動する

パネルをドラッグ アンド ドロップして、セクションに再配置および整理します。パネルの右上隅にある [Move] ボタンをクリックして、パネルの移動先のセクションを選択することもできます。

{{< img src="/images/app_ui/move-panel.gif" alt="" >}}

### パネルのサイズを変更する

* **Standard layout**: すべてのパネルのサイズは同じに維持され、パネルのページがあります。右下隅をクリックしてドラッグすると、パネルのサイズを変更できます。セクションの右下隅をクリックしてドラッグすると、セクションのサイズを変更できます。
* **Custom layout**: すべてのパネルのサイズは個別に設定され、ページはありません。

{{< img src="/images/app_ui/resize-panel.gif" alt="" >}}

### メトリクスを検索する

ワークスペースの検索ボックスを使用して、パネルを絞り込みます。この検索は、パネルのタイトル (デフォルトでは可視化されたメトリクスの名前) と一致します。

{{< img src="/images/app_ui/search_in_the_workspace.png" alt="" >}}

## Runs タブ

Runs タブを使用して、run をフィルター処理、グループ化、および並べ替えます。

{{< img src="/images/runs/run-table-example.png" alt="" >}}

次のタブは、Runs タブで実行できる一般的なアクションを示しています。

{{< tabpane text=true >}}
   {{% tab header="Customize columns" %}}
Runs タブには、project 内の run に関する詳細が表示されます。デフォルトでは、多数の列が表示されます。

- 表示されているすべての列を表示するには、ページを水平方向にスクロールします。
- 列の順序を変更するには、列を左右にドラッグします。
- 列を固定するには、列名にカーソルを合わせ、表示されるアクション メニュー `...` をクリックして、**Pin column** をクリックします。固定された列は、**Name** 列の後のページの左側に表示されます。固定された列を固定解除するには、**Unpin column** を選択します。
- 列を非表示にするには、列名にカーソルを合わせ、表示されるアクション メニュー `...` をクリックして、**Hide column** をクリックします。現在非表示になっているすべての列を表示するには、**Columns** をクリックします。
  - 複数の列を一度に表示、非表示、固定、および固定解除するには、**Columns** をクリックします。
  - 非表示の列の名前をクリックして、非表示を解除します。
  - 表示されている列の名前をクリックして、非表示にします。
  - 表示されている列の横にあるピン アイコンをクリックして、固定します。

Runs タブをカスタマイズすると、カスタマイズは [Workspace タブ]({{< relref path="#workspace-tab" lang="ja" >}}) の **Runs** セレクターにも反映されます。
   {{% /tab %}}

   {{% tab header="Sort" %}}
テーブル内のすべての行を、指定された列の値で並べ替えます。

1. マウスを列タイトルに合わせます。ケバブ メニュー (3 つの縦のドキュメント) が表示されます。
2. ケバブ メニュー (3 つの縦のドット) で選択します。
3. **Sort Asc** または **Sort Desc** を選択して、行をそれぞれ昇順または降順に並べ替えます。

{{< img src="/images/data_vis/data_vis_sort_kebob.png" alt="See the digits for which the model most confidently guessed '0'." >}}

上記の画像は、`val_acc` という名前のテーブル列の並べ替えオプションを表示する方法を示しています。
   {{% /tab %}}
   {{% tab header="Filter" %}}
ダッシュボードの左上にある **Filter** ボタンを使用して、式で行全体をフィルター処理します。

{{< img src="/images/data_vis/filter.png" alt="See only examples which the model gets wrong." >}}

**Add filter** を選択して、行に 1 つ以上のフィルターを追加します。3 つのドロップダウン メニューが表示されます。左から右へ、フィルター タイプは、列名、演算子、および値に基づいています。

|                   | 列名  | 二項関係  | 値  |
| -----------       | ----------- | ----------- | ----------- |
| 受け入れられる値  | 文字列  | &equals;, &ne;, &le;, &ge;, IN, NOT IN,  | 整数、浮動小数点数、文字列、タイムスタンプ、null |

式エディターには、列名と論理述語構造のオートコンプリートを使用して、各項のオプションのリストが表示されます。「and」または「or」(および場合によっては括弧) を使用して、複数の論理述語を 1 つの式に接続できます。

{{< img src="/images/data_vis/filter_example.png" alt="" >}}
上記の画像は、`val_loss` 列に基づいたフィルターを示しています。フィルターは、検証損失が 1 以下の run を表示します。
   {{% /tab %}}
   {{% tab header="Group" %}}
列ヘッダーの **Group by** ボタンを使用して、特定の列の値で行全体をグループ化します。

{{< img src="/images/data_vis/group.png" alt="The truth distribution shows small errors: 8s and 2s are confused for 7s and 9s for 2s." >}}

デフォルトでは、これにより、他の数値列が、グループ全体のその列の値の分布を示すヒストグラムに変わります。グループ化は、データ内のより高レベルのパターンを理解するのに役立ちます。
   {{% /tab %}}
{{< /tabpane >}}

## Reports タブ

1 か所で結果のすべてのスナップショットを確認し、チームと学びを共有します。

{{< img src="/images/app_ui/reports-tab.png" alt="" >}}

## Sweeps タブ

project から新しい [sweep]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) を開始します。

{{< img src="/images/app_ui/sweeps-tab.png" alt="" >}}

## Artifacts タブ

トレーニングデータセットや [fine-tuned models]({{< relref path="/guides/core/registry/" lang="ja" >}}) から、[メトリクスとメディアのテーブル]({{< relref path="/guides/models/tables/tables-walkthrough.md" lang="ja" >}}) まで、project に関連付けられているすべての [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) を表示します。

### Overview パネル

{{< img src="/images/app_ui/overview_panel.png" alt="" >}}

Overview パネルには、Artifacts の名前とバージョン、変更を検出して重複を防ぐために使用されるハッシュ ダイジェスト、作成日、エイリアスなど、Artifacts に関するさまざまな高度な情報が表示されます。ここでエイリアスを追加または削除したり、バージョンと Artifacts 全体の両方に関するメモを取ることができます。

### Metadata パネル

{{< img src="/images/app_ui/metadata_panel.png" alt="" >}}

Metadata パネルは、Artifacts のメタデータへのアクセスを提供します。このメタデータは、Artifacts の構築時に提供されます。このメタデータには、Artifacts を再構築するために必要な構成 引数 、詳細情報が見つかる URL、または Artifacts を記録した run 中に生成されたメトリクスが含まれる場合があります。さらに、Artifacts を生成した run の構成と、Artifacts のロギング時の履歴メトリクスを確認できます。

### Usage パネル

{{< img src="/images/app_ui/usage_panel.png" alt="" >}}

Usage パネルは、ウェブ アプリの外 (たとえば、ローカル マシン上) で使用するために Artifacts をダウンロードするためのコード スニペットを提供します。このセクションでは、Artifacts を出力した run と、Artifacts を入力として使用する run も示し、リンクします。

### Files パネル

{{< img src="/images/app_ui/files_panel.png" alt="" >}}

Files パネルには、Artifacts に関連付けられているファイルとフォルダーがリストされます。W&B は、run の特定のファイルを自動的にアップロードします。たとえば、`requirements.txt` は run が使用した各ライブラリのバージョンを示し、`wandb-metadata.json` および `wandb-summary.json` には run に関する情報が含まれています。run の構成に応じて、Artifacts やメディアなど、他のファイルがアップロードされる場合があります。このファイル ツリーをナビゲートして、W&B ウェブ アプリでコンテンツを直接表示できます。

Artifacts に関連付けられた [Tables]({{< relref path="/guides/models/tables//tables-walkthrough.md" lang="ja" >}}) は、このコンテキストで特に豊富でインタラクティブです。Artifacts で Tables を使用する方法について詳しくは、[こちら]({{< relref path="/guides/models/tables//visualize-tables.md" lang="ja" >}}) をご覧ください。

{{< img src="/images/app_ui/files_panel_table.png" alt="" >}}

### Lineage パネル

{{< img src="/images/app_ui/lineage_panel.png" alt="" >}}

Lineage パネルは、project に関連付けられているすべての Artifacts と、それらを相互に接続する run のビューを提供します。これは、run タイプをブロックとして、Artifacts を円として表示し、特定のタイプの run が特定のタイプの Artifacts を消費または生成するときを示す矢印を表示します。左側の列で選択された特定の Artifacts のタイプが強調表示されます。

[Explode] トグルをクリックすると、個々の Artifacts バージョンと、それらを接続する特定の run がすべて表示されます。

### Action History Audit タブ

{{< img src="/images/app_ui/action_history_audit_tab_1.png" alt="" >}}

{{< img src="/images/app_ui/action_history_audit_tab_2.png" alt="" >}}

アクション履歴監査タブには、リソースの進化全体を監査できるように、Collection のすべてのエイリアス アクションとメンバーシップの変更が表示されます。

### Versions タブ

{{< img src="/images/app_ui/versions_tab.png" alt="" >}}

Versions タブには、Artifacts のすべてのバージョンと、バージョンのロギング時の Run History の各数値の値の列が表示されます。これにより、パフォーマンスを比較し、関心のあるバージョンをすばやく特定できます。

## project にスターを付ける

project にスターを追加して、その project を重要としてマークします。あなたとあなたのチームがスターで重要としてマークした project は、組織のホームページの上部に表示されます。

たとえば、次の画像は、重要としてマークされている 2 つの project、`zoo_experiment` と `registry_demo` を示しています。両方の project は、組織のホームページの **Starred projects** セクションの上部に表示されます。
{{< img src="/images/track/star-projects.png" alt="" >}}

project を重要としてマークするには、project の Overview タブ内またはチームのプロファイル ページの 2 つの方法があります。

{{< tabpane text=true >}}
    {{% tab header="Project overview" %}}
1. W&B アプリ ( `https://wandb.ai/<team>/<project-name>` ) で W&B project に移動します。
2. project サイドバーから **Overview** タブを選択します。
3. 右上隅にある **Edit** ボタンの横にあるスター アイコンを選択します。

{{< img src="/images/track/star-project-overview-tab.png" alt="" >}}
    {{% /tab %}}
    {{% tab header="Team profile" %}}
1. チームのプロファイル ページ ( `https://wandb.ai/<team>/projects` ) に移動します。
2. **Projects** タブを選択します。
3. スターを付ける project の横にマウスを合わせます。表示されるスター アイコンをクリックします。

たとえば、次の画像は、"Compare_Zoo_Models" project の横にあるスター アイコンを示しています。
{{< img src="/images/track/star-project-team-profile-page.png" alt="" >}}
    {{% /tab %}}
{{< /tabpane >}}

アプリの左上隅にある組織名をクリックして、project が組織のランディング ページに表示されることを確認します。

## project を削除する

Overview タブの右側にある 3 つのドットをクリックして、project を削除できます。

{{< img src="/images/app_ui/howto_delete_project.gif" alt="" >}}

project が空の場合、右上にあるドロップダウン メニューをクリックし、**Delete project** を選択して削除できます。

{{< img src="/images/app_ui/howto_delete_project_2.png" alt="" >}}

## project にメモを追加する

説明の概要として、またはワークスペース内のマークダウン パネルとして、project にメモを追加します。

### 説明の概要を project に追加する

ページに追加する説明は、プロファイルの **Overview** タブに表示されます。

1. W&B project に移動します。
2. project サイドバーから **Overview** タブを選択します。
3. 右上隅にある [Edit] を選択します。
4. **Description** フィールドにメモを追加します。
5. **Save** ボタンを選択します。

{{% alert title="Create reports to create descriptive notes comparing runs" %}}
W&B Report を作成して、プロットとマークダウンを並べて追加することもできます。異なるセクションを使用して異なる run を表示し、作業内容に関するストーリーを伝えます。
{{% /alert %}}

### run ワークスペースにメモを追加する

1. W&B project に移動します。
2. project サイドバーから **Workspace** タブを選択します。
3. 右上隅にある **Add panels** ボタンを選択します。
4. 表示されるモーダルから **TEXT AND CODE** ドロップダウンを選択します。
5. **Markdown** を選択します。
6. ワークスペースに表示されるマークダウン パネルにメモを追加します。
