---
title: Projects
description: モデルの バージョン を比較し、スクラッチ ワークスペース で 結果 を探索し、 学び を レポート にエクスポートして、メモと 可視化
  を保存します。
menu:
  default:
    identifier: ja-guides-models-track-project-page
    parent: experiments
weight: 3
---

*project* は、結果の可視化、実験の比較、Artifacts の表示とダウンロード、オートメーションの作成などを行う中心的な場所です。

{{% alert %}}
各 project には、誰がアクセスできるかを決定する可視性設定があります。誰が project にアクセスできるかについての詳細は、[Project visibility]({{< relref path="/guides/hosting/iam/access-management/restricted-projects.md" lang="ja" >}}) を参照してください。
{{% /alert %}}

各 project には、サイドバーからアクセスできる次のものが含まれています。

* [**Overview**]({{< relref path="project-page.md#overview-tab" lang="ja" >}}): project のスナップショット
* [**Workspace**]({{< relref path="project-page.md#workspace-tab" lang="ja" >}}): 個人的な可視化サンドボックス
* [**Runs**]({{< relref path="#runs-tab" lang="ja" >}}): project 内のすべての run をリストするテーブル
* **Automations**: project で設定された Automations
* [**Sweeps**]({{< relref path="project-page.md#sweeps-tab" lang="ja" >}}): 自動化された探索と最適化
* [**Reports**]({{< relref path="project-page.md#reports-tab" lang="ja" >}}): ノート、run、グラフの保存されたスナップショット
* [**Artifacts**]({{< relref path="#artifacts-tab" lang="ja" >}}): すべての run と、その run に関連付けられた Artifacts が含まれています

## Overviewタブ

* **Project name**: project の名前。W&B は、run を初期化するときに、project フィールドに指定した名前で project を作成します。project の名前は、右上隅にある **Edit** ボタンを選択することで、いつでも変更できます。
* **Description**: project の説明。
* **Project visibility**: project の可視性。誰がアクセスできるかを決定する可視性設定。詳細については、[Project visibility]({{< relref path="/guides/hosting/iam/access-management/restricted-projects.md" lang="ja" >}}) を参照してください。
* **Last active**: この project に最後にデータが記録されたときのタイムスタンプ
* **Owner**: この project の所有 Entity
* **Contributors**: この project に貢献している Users の数
* **Total runs**: この project 内の run の合計数
* **Total compute**: project 内のすべての run 時間を合計して、この合計を取得します
* **Undelete runs**: ドロップダウンメニューをクリックし、"Undelete all runs" をクリックして、project 内で削除された run を復元します。
* **Delete project**: 右隅にあるドットメニューをクリックして、project を削除します

[ライブ例を見る](https://app.wandb.ai/example-team/sweep-demo/overview)

{{< img src="/images/track/overview_tab_image.png" alt="" >}}

## Workspaceタブ

project の *workspace* は、実験を比較するための個人的なサンドボックスを提供します。異なるアーキテクチャ、ハイパーパラメータ、データセット、プリプロセッシングなどで同じ問題に取り組んでいる、比較できる Models を整理するために project を使用します。

**Runs Sidebar**: project 内のすべての run のリスト。

* **Dot menu**: サイドバーの行にマウスを合わせると、左側にメニューが表示されます。このメニューを使用して、run の名前を変更したり、run を削除したり、アクティブな run を停止したりできます。
* **Visibility icon**: グラフ上の run のオン/オフを切り替えるには、目のアイコンをクリックします
* **Color**: run の色を、プリセットのいずれかまたはカスタムカラーに変更します
* **Search**: 名前で run を検索します。これにより、プロットに表示される run もフィルタリングされます。
* **Filter**: サイドバーフィルターを使用して、表示される run のセットを絞り込みます
* **Group**: 設定列を選択して、たとえばアーキテクチャごとに run を動的にグループ化します。グループ化により、プロットは平均値に沿った線と、グラフ上の点の分散の影付き領域で表示されます。
* **Sort**: たとえば、損失が最も低い run や精度が最も高い run など、run のソートに使用する値を選択します。ソートは、グラフに表示される run に影響します。
* **Expand button**: サイドバーをテーブル全体に展開します
* **Run count**: 上部の括弧内の数字は、project 内の run の合計数です。数字 (N visualized) は、目のアイコンがオンになっており、各プロットで可視化できる run の数です。以下の例では、グラフは 183 件の run のうち最初の 10 件のみを表示しています。グラフを編集して、表示可能な run の最大数を増やします。

[Runs tab](#runs-tab) で列を固定、非表示、または順序を変更すると、Runs サイドバーにこれらのカスタマイズが反映されます。

**Panels layout**: このスクラッチスペースを使用して、結果を探索し、チャートを追加および削除し、異なるメトリクスに基づいて Models のバージョンを比較します

[ライブ例を見る](https://app.wandb.ai/example-team/sweep-demo)

{{< img src="/images/app_ui/workspace_tab_example.png" alt="" >}}

### パネルのセクションを追加する

セクションのドロップダウンメニューをクリックし、"Add section" をクリックして、パネルの新しいセクションを作成します。セクションの名前を変更したり、ドラッグして再編成したり、セクションを展開したり折りたたんだりできます。

各セクションの右上隅にはオプションがあります。

* **Switch to custom layout**: カスタムレイアウトでは、パネルのサイズを個別に変更できます。
* **Switch to standard layout**: 標準レイアウトでは、セクション内のすべてのパネルのサイズを一度に変更でき、ページネーションが提供されます。
* **Add section**: ドロップダウンメニューから上下にセクションを追加するか、ページの下部にあるボタンをクリックして新しいセクションを追加します。
* **Rename section**: セクションのタイトルを変更します。
* **Export section to report**: このパネルのセクションを新しい report に保存します。
* **Delete section**: セクション全体とすべてのチャートを削除します。これは、ワークスペースバーのページ下部にある元に戻すボタンで元に戻すことができます。
* **Add panel**: プラスボタンをクリックして、セクションにパネルを追加します。

{{< img src="/images/app_ui/add-section.gif" alt="" >}}

### セクション間でパネルを移動する

パネルをドラッグアンドドロップして、セクションに再配置および整理します。パネルの右上隅にある "Move" ボタンをクリックして、パネルの移動先のセクションを選択することもできます。

{{< img src="/images/app_ui/move-panel.gif" alt="" >}}

### パネルのサイズを変更する

* **Standard layout**: すべてのパネルのサイズは同じに維持され、パネルのページがあります。右下隅をクリックしてドラッグすると、パネルのサイズを変更できます。セクションの右下隅をクリックしてドラッグすると、セクションのサイズを変更できます。
* **Custom layout**: すべてのパネルのサイズは個別に設定され、ページはありません。

{{< img src="/images/app_ui/resize-panel.gif" alt="" >}}

### メトリクスを検索する

ワークスペースの検索ボックスを使用して、パネルを絞り込みます。この検索は、パネルのタイトル (デフォルトでは可視化されたメトリクスの名前) と一致します。

{{< img src="/images/app_ui/search_in_the_workspace.png" alt="" >}}

## Runsタブ

Runs タブを使用して、run をフィルタリング、グループ化、およびソートします。

{{< img src="/images/runs/run-table-example.png" alt="" >}}

次のタブは、Runs タブで実行できる一般的なアクションの一部を示しています。

{{< tabpane text=true >}}
   {{% tab header="列のカスタマイズ" %}}
Runs タブには、project 内の run に関する詳細が表示されます。デフォルトでは、多数の列が表示されます。

- 表示されているすべての列を表示するには、ページを水平方向にスクロールします。
- 列の順序を変更するには、列を左または右にドラッグします。
- 列を固定するには、列名にカーソルを合わせ、表示されるアクションメニュー `...` をクリックして、**Pin column** をクリックします。固定された列は、**Name** 列の後に、ページの左側の近くに表示されます。固定された列を固定解除するには、**Unpin column** を選択します。
- 列を非表示にするには、列名にカーソルを合わせ、表示されるアクションメニュー `...` をクリックして、**Hide column** をクリックします。現在非表示になっているすべての列を表示するには、**Columns** をクリックします。
  - 複数の列を一度に表示、非表示、固定、および固定解除するには、**Columns** をクリックします。
  - 非表示の列の名前をクリックして、非表示を解除します。
  - 表示されている列の名前をクリックして、非表示にします。
  - 表示されている列の横にあるピンアイコンをクリックして、固定します。

Runs タブをカスタマイズすると、カスタマイズは [Workspace タブ]({{< relref path="#workspace-tab" lang="ja" >}}) の **Runs** セレクターにも反映されます。
   {{% /tab %}}

   {{% tab header="ソート" %}}
テーブル内のすべての行を、指定された列の値でソートします。

1. マウスを列のタイトルに合わせます。ケバブメニュー (縦に 3 つのドット) が表示されます。
2. ケバブメニュー (縦に 3 つのドット) を選択します。
3. **Sort Asc** または **Sort Desc** を選択して、それぞれ昇順または降順で行をソートします。

{{< img src="/images/data_vis/data_vis_sort_kebob.png" alt="モデルが最も自信を持って「0」と推測した数字を確認してください。" >}}

上記の画像は、`val_acc` というテーブル列のソートオプションを表示する方法を示しています。
   {{% /tab %}}
   {{% tab header="フィルター" %}}
ダッシュボードの左上にある **Filter** ボタンを使用して、式で行をフィルタリングします。

{{< img src="/images/data_vis/filter.png" alt="モデルが間違っている例のみを表示します。" >}}

**Add filter** を選択して、1 つ以上のフィルターを行に追加します。3 つのドロップダウンメニューが表示されます。左から右に、フィルターの種類は、列名、演算子、値に基づいています。

|                   | 列名 | 二項関係    | 値       |
| -----------       | ----------- | ----------- | ----------- |
| 使用可能な値   | 文字列       |  &equals;, &ne;, &le;, &ge;, IN, NOT IN,  | 整数、浮動小数点数、文字列、タイムスタンプ、null |

式エディターには、列名と論理述語構造のオートコンプリートを使用して、各項のオプションのリストが表示されます。「and」または「or」(場合によっては括弧) を使用して、複数の論理述語を 1 つの式に接続できます。

{{< img src="/images/data_vis/filter_example.png" alt="" >}}
上記の画像は、`val_loss` 列に基づくフィルターを示しています。フィルターは、検証損失が 1 以下の run を表示します。
   {{% /tab %}}
   {{% tab header="グループ" %}}
列ヘッダーにある **Group by** ボタンを使用して、特定の列の値で行をグループ化します。

{{< img src="/images/data_vis/group.png" alt="真実の分布は小さなエラーを示しています: 8 と 2 は 7 と 9、2 は 2 と混同されています。" >}}

デフォルトでは、これにより、他の数値列が、グループ全体のその列の値の分布を示すヒストグラムに変わります。グループ化は、データ内のより高レベルのパターンを理解するのに役立ちます。
   {{% /tab %}}
{{< /tabpane >}}

## Reportsタブ

結果のすべてのスナップショットを 1 か所で確認し、チームと学びを共有します。

{{< img src="/images/app_ui/reports-tab.png" alt="" >}}

## Sweepsタブ

project から新しい [sweep]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) を開始します。

{{< img src="/images/app_ui/sweeps-tab.png" alt="" >}}

## Artifactsタブ

トレーニングデータセットや [fine-tuned models]({{< relref path="/guides/models/registry/model_registry/" lang="ja" >}}) から [metrics とメディアのテーブル]({{< relref path="/guides/core/tables/tables-walkthrough.md" lang="ja" >}}) まで、project に関連付けられたすべての [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) を表示します。

### Overviewパネル

{{< img src="/images/app_ui/overview_panel.png" alt="" >}}

Overview パネルには、名前とバージョン、変更を検出して重複を防ぐために使用されるハッシュダイジェスト、作成日、エイリアスなど、Artifacts に関するさまざまな高レベルの情報が表示されます。ここでエイリアスを追加または削除したり、バージョンと Artifacts 全体の両方についてメモを取ることができます。

### Metadataパネル

{{< img src="/images/app_ui/metadata_panel.png" alt="" >}}

Metadata パネルは、Artifacts の Metadata へのアクセスを提供します。この Metadata は、Artifacts の構築時に提供されます。この Metadata には、Artifacts を再構築するために必要な設定引数、詳細情報を見つけることができる URL、または Artifacts をログに記録した run 中に生成されたメトリクスが含まれる場合があります。さらに、Artifacts を生成した run の設定と、Artifacts のログ記録時の履歴メトリクスを確認できます。

### Usageパネル

{{< img src="/images/app_ui/usage_panel.png" alt="" >}}

Usage パネルは、たとえばローカルマシンなど、Web アプリの外で使用するために Artifacts をダウンロードするためのコードスニペットを提供します。このセクションでは、Artifacts を出力した run と、Artifacts を入力として使用する run を示し、リンクも提供します。

### Filesパネル

{{< img src="/images/app_ui/files_panel.png" alt="" >}}

Files パネルには、Artifacts に関連付けられたファイルとフォルダーがリストされます。W&B は、run の特定のファイルを自動的にアップロードします。たとえば、`requirements.txt` は run が使用した各ライブラリのバージョンを示し、`wandb-metadata.json` と `wandb-summary.json` には run に関する情報が含まれます。run の設定に応じて、Artifacts やメディアなどの他のファイルがアップロードされる場合があります。このファイルツリーをナビゲートし、W&B Web アプリでコンテンツを直接表示できます。

Artifacts に関連付けられた [Tables]({{< relref path="/guides/core/tables/tables-walkthrough.md" lang="ja" >}}) は、特にこのコンテキストでリッチでインタラクティブです。Artifacts での Tables の使用の詳細については、[こちら]({{< relref path="/guides/core/tables/visualize-tables.md" lang="ja" >}}) を参照してください。

{{< img src="/images/app_ui/files_panel_table.png" alt="" >}}

### Lineageパネル

{{< img src="/images/app_ui/lineage_panel.png" alt="" >}}

Lineage パネルは、project に関連付けられたすべての Artifacts と、それらを相互に接続する run のビューを提供します。run の種類をブロックとして、Artifacts を円として表示し、特定の種類の run が特定の種類の Artifacts を消費または生成する場合を示す矢印を表示します。左側の列で選択された特定の Artifacts の種類が強調表示されます。

Explode トグルをクリックして、すべての個々の Artifacts バージョンと、それらを接続する特定の run を表示します。

### Action History Auditタブ

{{< img src="/images/app_ui/action_history_audit_tab_1.png" alt="" >}}

{{< img src="/images/app_ui/action_history_audit_tab_2.png" alt="" >}}

アクション履歴監査タブには、Collection のすべてのエイリアスアクションとメンバーシップの変更が表示されるため、リソースの進化全体を監査できます。

### Versionsタブ

{{< img src="/images/app_ui/versions_tab.png" alt="" >}}

バージョンタブには、Artifacts のすべてのバージョンと、バージョンをログに記録した時点での Run History の各数値の列が表示されます。これにより、パフォーマンスを比較し、関心のあるバージョンをすばやく識別できます。

## project にスターを付ける

project にスターを追加して、その project を重要としてマークします。あなたとあなたの Team がスターで重要としてマークした project は、組織のホームページの上部に表示されます。

たとえば、次の画像は、重要としてマークされている 2 つの project、`zoo_experiment` と `registry_demo` を示しています。どちらの project も、組織のホームページの上部にある **Starred projects** セクション内に表示されます。
{{< img src="/images/track/star-projects.png" alt="" >}}

project を重要としてマークするには、project の Overview タブ内または Team のプロファイルページ内の 2 つの方法があります。

{{< tabpane text=true >}}
    {{% tab header="Project overview" %}}
1. W&B アプリの `https://wandb.ai/<team>/<project-name>` で W&B project に移動します。
2. project サイドバーから **Overview** タブを選択します。
3. 右上隅の **Edit** ボタンの横にあるスターアイコンを選択します。

{{< img src="/images/track/star-project-overview-tab.png" alt="" >}}
    {{% /tab %}}
    {{% tab header="Team profile" %}}
1. `https://wandb.ai/<team>/projects` で Team のプロファイルページに移動します。
2. **Projects** タブを選択します。
3. スターを付ける project の横にマウスを合わせます。表示されるスターアイコンをクリックします。

たとえば、次の画像は、「Compare_Zoo_Models」project の横にあるスターアイコンを示しています。
{{< img src="/images/track/star-project-team-profile-page.png" alt="" >}}
    {{% /tab %}}
{{< /tabpane >}}

アプリの左上隅にある組織名をクリックして、project が組織のランディングページに表示されることを確認します。

## project を削除する

Overview タブの右側にある 3 つのドットをクリックして、project を削除できます。

{{< img src="/images/app_ui/howto_delete_project.gif" alt="" >}}

project が空の場合、右上にあるドロップダウンメニューをクリックし、**Delete project** を選択して削除できます。

{{< img src="/images/app_ui/howto_delete_project_2.png" alt="" >}}

## project にメモを追加する

説明の概要として、または workspace 内の markdown パネルとして、project にメモを追加します。

### project に説明の概要を追加する

ページに追加した説明は、プロファイルの **Overview** タブに表示されます。

1. W&B project に移動します
2. project サイドバーから **Overview** タブを選択します
3. 右上隅にある Edit を選択します
4. **Description** フィールドにメモを追加します
5. **Save** ボタンを選択します

{{% alert title="run を比較する説明的なメモを作成するために reports を作成する" %}}
W&B Report を作成して、プロットと markdown を並べて追加することもできます。異なるセクションを使用して、異なる run を表示し、作業内容に関するストーリーを伝えます。
{{% /alert %}}

### run workspace にメモを追加する

1. W&B project に移動します
2. project サイドバーから **Workspace** タブを選択します
3. 右上隅にある **Add panels** ボタンを選択します
4. 表示されるモーダルから **TEXT AND CODE** ドロップダウンを選択します
5. **Markdown** を選択します
6. workspace に表示される markdown パネルにメモを追加します
