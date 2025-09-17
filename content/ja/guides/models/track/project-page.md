---
title: Projects
description: モデルのバージョンを比較し、スクラッチ Workspace で結果を探索し、学びを Report にエクスポートしてノートや可視化を保存する
menu:
  default:
    identifier: ja-guides-models-track-project-page
    parent: experiments
weight: 3
---

Project は、結果の可視化、experiments の比較、Artifacts の表示とダウンロード、Automation の作成などを行う中心的な場所です。

{{% alert %}}
各 Project には公開範囲の設定があり、誰がアクセスできるかを制御します。Project にアクセスできるユーザーについては、[Project visibility]({{< relref path="/guides/hosting/iam/access-management/restricted-projects.md" lang="ja" >}}) を参照してください。
{{% /alert %}}

各 Project には次のタブがあります:

* [Overview]({{< relref path="project-page.md#overview-tab" lang="ja" >}}): Project のスナップショット
* [Workspace]({{< relref path="project-page.md#workspace-tab" lang="ja" >}}): 個人用の可視化サンドボックス
* [Runs]({{< relref path="#runs-tab" lang="ja" >}}): Project 内のすべての Runs を一覧するテーブル
* [Automations]({{< relref path="#automations-tab" lang="ja" >}}): Project で設定された Automations
* [Sweeps]({{< relref path="project-page.md#sweeps-tab" lang="ja" >}}): 自動探索と最適化
* [Reports]({{< relref path="project-page.md#reports-tab" lang="ja" >}}): メモ、Runs、グラフのスナップショット
* [Artifacts]({{< relref path="#artifacts-tab" lang="ja" >}}): すべての Runs と、その Run に関連づけられた Artifacts

## Overview タブ

* **Project name**: Project の名前。W&B は、project フィールドに指定した名前で Run を初期化すると Project を作成します。右上の **Edit** ボタンからいつでも名前を変更できます。
* **Description**: Project の説明。
* **Project visibility**: Project の公開範囲。誰がアクセスできるかを決める設定です。詳しくは [Project visibility]({{< relref path="/guides/hosting/iam/access-management/restricted-projects.md" lang="ja" >}}) を参照してください。
* **Last active**: この Project に最後にデータがログされた時刻のタイムスタンプ
* **Owner**: この Project を所有する Entity
* **Contributors**: この Project に貢献している Users の数
* **Total runs**: この Project に含まれる Runs の総数
* **Total compute**: Project 内の Run の実行時間を合計した値
* **Undelete runs**: ドロップダウン メニューから "Undelete all runs" を選ぶと、Project で削除された Runs を復元できます。
* **Delete project**: 右上のドットメニューから Project を削除します

[ライブ例を見る](https://app.wandb.ai/example-team/sweep-demo/overview)

{{< img src="/images/track/overview_tab_image.png" alt="Project の Overview タブ" >}}

Overview タブから Project の公開範囲を変更するには:

{{% readfile file="/content/en/_includes/project-visibility-settings.md" %}}

## Workspace タブ

Project の Workspace は、experiments を比較するための個人用サンドボックスです。異なるアーキテクチャー、ハイパーパラメーター、Datasets、前処理などで同じ課題に取り組む Models を整理するのに Project を使いましょう。

**Runs Sidebar**: Project 内のすべての Runs のリスト。

* **Dot menu**: サイドバーの行にカーソルを合わせると、左側にメニューが表示されます。このメニューで Run の名前変更、Run の削除、実行中の Run の停止ができます。
* **Visibility icon**: 目のアイコンをクリックして、グラフ上の Runs の表示・非表示を切り替えます
* **Color**: Run の色をプリセットまたはカスタム色に変更します
* **Search**: 名前で Runs を検索します。これはグラフに表示される Runs のフィルタにもなります。
* **Filter**: サイドバーのフィルタで、可視化対象の Runs を絞り込みます
* **Group**: 設定列を選んで Runs を動的にグループ化します（例: アーキテクチャー別）。グループ化すると、グラフは平均値の線と分散の領域を表示します。
* **Sort**: 値で Runs をソートします（例: 最小の loss や最大の accuracy）。ソートはグラフに表示される Runs にも影響します。
* **Expand button**: サイドバーをフルテーブルに展開します
* **Run count**: 先頭のかっこ内の数は Project の総 Runs 数です。"(N visualized)" は、目のアイコンがオンで各プロットで可視化可能な Runs の数です。下の例では、183 件中最初の 10 件だけが表示されています。グラフを編集して表示可能な Runs の最大数を増やせます。

[Runs タブ](#runs-tab) で列の固定、非表示、順序変更を行うと、そのカスタマイズは Runs サイドバーにも反映されます。

**Panels layout**: この作業スペースを使って結果を探索し、チャートを追加・削除し、異なるメトリクスに基づいて Models のバージョンを比較します

[ライブ例を見る](https://app.wandb.ai/example-team/sweep-demo)

{{< img src="/images/app_ui/workspace_tab_example.png" alt="Project Workspace" >}}


### パネルのセクションを追加

セクションのドロップダウン メニューから "Add section" をクリックして、パネル用の新しいセクションを作成します。セクション名の変更、ドラッグによる並べ替え、展開と折りたたみが可能です。

各セクションの右上には次のオプションがあります:

* **Switch to custom layout**: カスタム レイアウトでは、パネルを個別にリサイズできます。
* **Switch to standard layout**: 標準レイアウトでは、セクション内のすべてのパネルをまとめてリサイズでき、ページネーションが付きます。
* **Add section**: ドロップダウン メニューから上下いずれかにセクションを追加するか、ページ下部のボタンで新しいセクションを追加します。
* **Rename section**: セクションのタイトルを変更します。
* **Export section to report**: このパネル セクションを新しい Report に保存します。
* **Delete section**: セクション全体とすべてのチャートを削除します。Workspace バーの下部にある元に戻すボタンで取り消せます。
* **Add panel**: プラスボタンをクリックして、このセクションにパネルを追加します。

{{< img src="/images/app_ui/add-section.gif" alt="Workspace セクションの追加" >}}

### セクション間でパネルを移動

ドラッグ＆ドロップでパネルを並べ替え、セクションに整理します。パネル右上の "Move" ボタンをクリックして、移動先のセクションを選ぶこともできます。

{{< img src="/images/app_ui/move-panel.gif" alt="セクション間のパネル移動" >}}

### パネルのサイズ変更

* **Standard layout**: すべてのパネルが同じサイズで、ページに分かれて表示されます。パネルは右下の角をドラッグしてリサイズできます。セクションはセクション右下の角をドラッグしてリサイズします。
* **Custom layout**: すべてのパネルを個別にサイズ調整でき、ページ分割はありません。

{{< img src="/images/app_ui/resize-panel.gif" alt="パネルのサイズ変更" >}}

### メトリクスを検索

Workspace の検索ボックスでパネルを絞り込みます。この検索はパネルのタイトル（既定では可視化しているメトリクス名）にマッチします。

{{< img src="/images/app_ui/search_in_the_workspace.png" alt="Workspace 検索" >}}

## Runs タブ

Runs タブで Runs をフィルタ、グループ化、ソートできます。

{{< img src="/images/runs/run-table-example.png" alt="Runs テーブル" >}}

以下のタブで、Runs タブでよく行う操作を紹介します。

{{< tabpane text=true >}}
   {{% tab header="列をカスタマイズ" %}}
Runs タブは Project 内の Run の詳細を表示します。既定で多くの列が表示されます。

{{% alert %}}
Runs タブをカスタマイズすると、その内容は [Workspace タブ]({{< relref path="#workspace-tab" lang="ja" >}}) の **Runs** セレクタにも反映されます。
{{% /alert %}}

- 表示中の列をすべて見るには、ページを水平方向にスクロールします。
- 列の順序を変更するには、列を左右にドラッグします。
- 列を固定するには、列名にカーソルを合わせ、表示されるアクションメニュー `...` をクリックし、**Pin column** を選びます。固定した列は **Name** 列の直後、ページ左側付近に表示されます。固定を解除するには **Unpin column** を選びます。
- 列を非表示にするには、列名にカーソルを合わせ、表示されるアクションメニュー `...` をクリックし、**Hide column** を選びます。現在非表示の列を一覧表示するには **Columns** をクリックします。
- 複数の列をまとめて表示・非表示・固定・固定解除するには **Columns** をクリックします。
  - 非表示の列名をクリックすると表示に戻ります。
  - 表示中の列名をクリックすると非表示になります。
  - 表示中の列のピンアイコンをクリックすると固定されます。

   {{% /tab %}}

   {{% tab header="ソート" %}}
Table の任意の列の値で全行をソートします。

1. 列タイトルにマウスオーバーします。縦三点のケバブメニューが表示されます。
2. ケバブメニュー（縦三点）をクリックします。
3. **Sort Asc** または **Sort Desc** を選び、それぞれ昇順・降順に並べ替えます。

{{< img src="/images/data_vis/data_vis_sort_kebob.png" alt="ソートオプション" >}}

上の画像は、`val_acc` という Table 列のソートオプションの表示方法を示しています。   
   {{% /tab %}}
   {{% tab header="フィルタ" %}}
ダッシュボード左上の **Filter** ボタンで、式に基づいて全行をフィルタします。

{{< img src="/images/data_vis/filter.png" alt="フィルタの例" >}}

**Add filter** を選んで 1 つ以上のフィルタを追加します。3 つのドロップダウンが表示され、左から順に「列名」「演算子」「値」を選びます。

|                   | 列名 | 二項関係    | 値       |
| -----------       | ----------- | ----------- | ----------- |
| 受け付ける値   | String       |  &equals;, &ne;, &le;, &ge;, IN, NOT IN,  | Integer, float, string, timestamp, null |

式エディタは、列名のオートコンプリートと論理述語の構造に基づき、各項目の選択肢を一覧表示します。"and" や "or"（場合によってはかっこ）を使って、複数の論理述語を 1 つの式に結合できます。

{{< img src="/images/data_vis/filter_example.png" alt="検証損失で Runs をフィルタ" >}}
上の画像は、`val_loss` 列に基づくフィルタの例です。検証損失が 1 以下の Runs を表示しています。   
   {{% /tab %}}
   {{% tab header="グループ化" %}}
列ヘッダーの **Group by** ボタンで、特定の列の値ごとに全行をグループ化します。

{{< img src="/images/data_vis/group.png" alt="誤差分布の分析" >}}

既定では、他の数値列はヒストグラムに変換され、その列の値の分布がグループごとに表示されます。グループ化は、データの高次なパターンを理解するのに役立ちます。   
   {{% /tab %}}
{{< /tabpane >}}



## Automations タブ
Artifacts のバージョン管理に伴う下流アクションを自動化します。Automation を作成するには、トリガーイベントとそれに続くアクションを定義します。アクションには、Webhook の実行や W&B のジョブの起動などが含まれます。詳しくは [Automations]({{< relref path="/guides/core/automations/" lang="ja" >}}) を参照してください。

{{< img src="/images/app_ui/automations_tab.png" alt="Automations タブ" >}}

## Reports タブ

結果のスナップショットを 1 か所で確認し、学びをチームと共有します。

{{< img src="/images/app_ui/reports-tab.png" alt="Reports タブ" >}}

## Sweeps タブ

Project から新しい [sweep]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) を開始します。

{{< img src="/images/app_ui/sweeps-tab.png" alt="Sweeps タブ" >}}

## Artifacts タブ

Project に関連する [artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) をすべて表示します。トレーニング Datasets や [fine-tuned models]({{< relref path="/guides/core/registry/" lang="ja" >}})、[メトリクスとメディアの tables]({{< relref path="/guides/models/tables/tables-walkthrough.md" lang="ja" >}}) まで。

### Overview パネル

{{< img src="/images/app_ui/overview_panel.png" alt="Artifact の Overview パネル" >}}

Overview パネルには、artifact 名とバージョン、変更検知と重複防止に使うハッシュダイジェスト、作成日、エイリアスなど、artifact に関するさまざまな高レベル情報が表示されます。ここでエイリアスの追加・削除を行い、バージョンや artifact 全体にノートを残せます。

### Metadata パネル

{{< img src="/images/app_ui/metadata_panel.png" alt="Artifact の Metadata パネル" >}}

Metadata パネルでは、artifact 構築時に与えられたメタデータにアクセスできます。これには、artifact の再構築に必要な設定引数、詳細情報の URL、artifact をログした Run で生成されたメトリクスなどが含まれる場合があります。加えて、artifact を生成した Run の設定や、artifact をログした時点の履歴メトリクスも確認できます。

### Usage パネル

{{< img src="/images/app_ui/usage_panel.png" alt="Artifact の Usage パネル" >}}

Usage パネルには、Web アプリ外（例えばローカルマシン）で artifact をダウンロードするためのコードスニペットが表示されます。このセクションには、artifact を出力した Run と、その artifact を入力として使用する任意の Run へのリンクも表示されます。

### Files パネル

{{< img src="/images/app_ui/files_panel.png" alt="Artifact の Files パネル" >}}

Files パネルには、artifact に関連するファイルやフォルダが一覧表示されます。W&B は Run に対して一部のファイルを自動でアップロードします。例えば、`requirements.txt` には Run が使用した各ライブラリのバージョンが記録され、`wandb-metadata.json` や `wandb-summary.json` には Run に関する情報が含まれます。Run の設定によっては、Artifacts やメディアなど他のファイルがアップロードされることもあります。このファイル ツリーを辿って、W&B の Web アプリ内で内容を直接表示できます。

Artifacts に関連する [Tables]({{< relref path="/guides/models/tables//tables-walkthrough.md" lang="ja" >}}) は、このコンテキストで特にリッチかつインタラクティブです。Artifacts と Tables の使い方は [こちら]({{< relref path="/guides/models/tables//visualize-tables.md" lang="ja" >}}) を参照してください。

{{< img src="/images/app_ui/files_panel_table.png" alt="Artifact の Table ビュー" >}}

### Lineage パネル

{{< img src="/images/app_ui/lineage_panel.png" alt="Artifact の Lineage" >}}

Lineage パネルでは、Project に関連するすべての artifacts と、それらを相互に結びつける Runs を俯瞰できます。Run の種類はブロック、Artifact は円で表示され、特定の種類の Run が特定の種類の artifact を消費または生成する関係を矢印で示します。左側の列で選択した特定の artifact の種類がハイライトされます。

"Explode" トグルをクリックすると、個々の artifact バージョンと、それらを結びつける具体的な Runs を展開表示できます。

### Action History Audit タブ

{{< img src="/images/app_ui/action_history_audit_tab_1.png" alt="アクション履歴の監査" >}}

{{< img src="/images/app_ui/action_history_audit_tab_2.png" alt="アクション履歴" >}}

Action History Audit タブには、Collection のエイリアス操作とメンバーシップ変更がすべて表示され、リソースの進化の全履歴を監査できます。

### Versions タブ

{{< img src="/images/app_ui/versions_tab.png" alt="Artifact の Versions タブ" >}}

Versions タブには、artifact のすべてのバージョンと、そのバージョンをログした時点の Run History の数値列が表示されます。これにより、パフォーマンスを比較し、関心のあるバージョンをすばやく見つけられます。

## Project を作成する
W&B App から作成する方法と、`wandb.init()` 呼び出しで project を指定してプログラムから作成する方法があります。

{{< tabpane text=true >}}
   {{% tab header="W&B App" %}}
W&B App では、**Projects** ページまたは team のランディングページから Project を作成できます。

**Projects** ページから:
1. 左上のグローバル ナビゲーション アイコンをクリックします。ナビゲーション サイドバーが開きます。
1. ナビゲーションの **Projects** セクションで **View all** をクリックして Project の概要ページを開きます。
1. **Create new project** をクリックします。
1. **Team** に、この Project を所有する team 名を設定します。
1. **Name** フィールドで Project 名を指定します。 
1. **Project visibility** を設定します（既定は **Team**）。
1. 任意で **Description** を入力します。
1. **Create project** をクリックします。

team のランディングページから:
1. 左上のグローバル ナビゲーション アイコンをクリックします。ナビゲーション サイドバーが開きます。
1. ナビゲーションの **Teams** セクションで team 名をクリックし、ランディングページを開きます。
1. ランディングページで **Create new project** をクリックします。
1. **Team** は、表示中のランディングページを所有する team に自動設定されます。必要に応じて変更します。
1. **Name** フィールドで Project 名を指定します。 
1. **Project visibility** を設定します（既定は **Team**）。
1. 任意で **Description** を入力します。
1. **Create project** をクリックします。

   {{% /tab %}}
   {{% tab header="Python SDK" %}}
プログラムから Project を作成するには、`wandb.init()` 呼び出し時に `project` を指定します。Project が存在しない場合は自動的に作成され、指定した entity が所有者になります。例:

```python
import wandb with wandb.init(entity="<entity>", project="<project_name>") as run: run.log({"accuracy": .95})
```

[`wandb.init()` の API リファレンス]({{< relref path="/ref/python/sdk/functions/init/#examples" lang="ja" >}}) を参照してください。
   {{% /tab %}}  
{{< /tabpane >}}

## Project にスターを付ける

重要な Project にスターを付けてマークできます。あなたや team がスターで重要とマークした Projects は、組織のホームページの上部に表示されます。

例えば、次の画像では `zoo_experiment` と `registry_demo` の 2 つの Project にスターが付いており、どちらも組織のホームページ上部の **Starred projects** セクションに表示されています。
{{< img src="/images/track/star-projects.png" alt="Starred projects セクション" >}}

Project を重要としてマークする方法は 2 つあります。Project の Overview タブ内、または team のプロフィールページ内です。

{{< tabpane text=true >}}
    {{% tab header="Project overview" %}}
1. W&B App で `https://wandb.ai/<team>/<project-name>` の W&B Project に移動します。
2. Project サイドバーから **Overview** タブを選びます。
3. 右上の **Edit** ボタンの横にある星アイコンをクリックします。

{{< img src="/images/track/star-project-overview-tab.png" alt="Overview から Project にスターを付ける" >}}    
    {{% /tab %}}
    {{% tab header="Team profile" %}}
1. `https://wandb.ai/<team>/projects` の team プロフィールページに移動します。
2. **Projects** タブを選びます。
3. スターを付けたい Project の横にマウスオーバーし、表示される星アイコンをクリックします。

例えば、次の画像では "Compare_Zoo_Models" Project の横に星アイコンが表示されています。
{{< img src="/images/track/star-project-team-profile-page.png" alt="team ページから Project にスターを付ける" >}}    
    {{% /tab %}}
{{< /tabpane >}}

組織のランディングページに Project が表示されていることを確認するには、アプリ左上の組織名をクリックします。

## Project を削除する

Overview タブ右側の三点メニューから Project を削除できます。

{{< img src="/images/app_ui/howto_delete_project.gif" alt="Project 削除ワークフロー" >}}

Project が空の場合は、右上のドロップダウン メニューから **Delete project** を選択して削除できます。

{{< img src="/images/app_ui/howto_delete_project_2.png" alt="空の Project を削除" >}}

## Project にノートを追加する

Project には、説明（Overview）として、または Workspace 内の Markdown パネルとしてノートを追加できます。

### Project に説明（Overview）を追加

ページに追加した説明は、プロフィールの **Overview** タブに表示されます。

1. W&B の Project に移動します
2. Project サイドバーから **Overview** タブを選びます
3. 右上の Edit を選択します
4. **Description** フィールドにノートを追加します
5. **Save** ボタンを選択します

{{% alert title="Runs を比較する説明用のノートには Reports を使いましょう" %}}
W&B Report を作成して、プロットと Markdown を横に並べて追加することもできます。異なるセクションで別々の Runs を見せ、取り組んだ内容をストーリーとして伝えましょう。
{{% /alert %}}

### Run の Workspace にノートを追加

1. W&B の Project に移動します
2. Project サイドバーから **Workspace** タブを選びます
3. 右上の **Add panels** ボタンを選びます
4. 表示されるモーダルで **TEXT AND CODE** ドロップダウンを開きます
5. **Markdown** を選びます
6. Workspace に表示される Markdown パネルにノートを追加します