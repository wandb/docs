---
title: プロジェクト
description: モデルのバージョンを比較し、結果をスクラッチワークスペースで探索し、学びや可視化をレポートにエクスポートしてメモを保存しましょう
menu:
  default:
    identifier: project-page
    parent: experiments
weight: 3
---

*プロジェクト* は、結果の可視化や実験の比較、Artifacts の閲覧・ダウンロード、オートメーションの作成などを行う中心的な場所です。

{{% alert %}}
各プロジェクトには公開範囲の設定があり、誰がアクセスできるかを決めることができます。プロジェクトへのアクセス可否について詳しくは、[Project visibility]({{< relref "/guides/hosting/iam/access-management/restricted-projects.md" >}})をご覧ください。
{{% /alert %}}

各プロジェクトには以下のタブがあります：

* [Overview]({{< relref "project-page.md#overview-tab" >}})：プロジェクトのスナップショット
* [Workspace]({{< relref "project-page.md#workspace-tab" >}})： 個人用の可視化サンドボックス
* [Runs]({{< relref "#runs-tab" >}})：プロジェクト内の全ての Run を一覧表示するテーブル
* [Automations]({{< relref "#automations-tab">}})：プロジェクト内で設定したオートメーション
* [Sweeps]({{< relref "project-page.md#sweeps-tab" >}})：自動化された探索と最適化
* [Reports]({{< relref "project-page.md#reports-tab" >}})：ノートや Run、グラフのスナップショット
* [Artifacts]({{< relref "#artifacts-tab" >}})：すべての Run と、その Run に紐付く Artifacts

## Overview タブ

* **Project name**: プロジェクトの名前。W&B では、`wandb.init()`の際に指定した名前でプロジェクトが作成されます。右上の **Edit** ボタンからいつでも名前を変更できます。
* **Description**: プロジェクトの説明。
* **Project visibility**: プロジェクトの公開範囲。誰が閲覧できるかの設定。[Project visibility]({{< relref "/guides/hosting/iam/access-management/restricted-projects.md" >}}) もご参照ください。
* **Last active**: データが最後にこのプロジェクトに記録された日時
* **Owner**: このプロジェクトの所有 Entity
* **Contributors**: 本プロジェクトに貢献した User の数
* **Total runs**: プロジェクト内の Run の総数
* **Total compute**: プロジェクト内全ての Run 時間を合算した合計
* **Undelete runs**: ドロップダウンメニューをクリックし「Undelete all runs」を選ぶことで、削除された Run を復元できます
* **Delete project**: 右上のメニューからプロジェクトを削除できます

[ライブ例を見る](https://app.wandb.ai/example-team/sweep-demo/overview)

{{< img src="/images/track/overview_tab_image.png" alt="Project overview tab" >}}


## Workspace タブ

プロジェクトの *workspace* は、実験の比較を行うための個人サンドボックスです。プロジェクトごとにモデルを整理し、同じ課題に対して異なるアーキテクチャーやハイパーパラメータ、データセット、前処理などを比較できます。

**Runs サイドバー**：プロジェクト内の全 Run の一覧。

* **ドットメニュー**: 行にマウスオーバーすると左側にメニューが表示されます。ここから Run の名前変更、削除、停止などが行えます。
* **公開範囲アイコン**: 目のアイコンをクリックするとグラフ上の Run の表示/非表示を切り替えられます
* **色**: Run に割り当てる色をプリセットやカスタムカラーから変更可能
* **検索**: 名前で Run を検索し、プロット上で表示対象を絞ることができます
* **フィルタ**: サイドバーのフィルターで、可視化する Run を絞り込みます
* **グループ**: 設定カラムを選択すると、例えばアーキテクチャーごとに動的に Run をグループ化できます。グループ化すると、プロットには平均値の線や分散範囲が表示されます。
* **ソート**: 指定した値（例: 最小 loss や最大 accuracy）で Run を並び替え。グラフに表示される Run が変わります。
* **拡張ボタン**: サイドバーを全画面のテーブルに拡大
* **Run 数**: カッコ内の数字はプロジェクト中の全 Run 数です。N visualized の数字は、目のアイコンがオンで各プロットで可視化できる Run の数です。下記例では10/183 の Run のみグラフ表示。最大表示数はグラフで編集可能です。

[Runs タブ](#runs-tab)で列のピン留め、非表示、順序変更を行うと、Runs サイドバーもそのカスタマイズが反映されます。

**パネルレイアウト**：ここはアイディア出しスペースとして活用できます。結果の深掘り、新規チャートの追加・削除、異なるメトリクスでモデルのバージョンの比較が可能です。

[ライブ例を見る](https://app.wandb.ai/example-team/sweep-demo)

{{< img src="/images/app_ui/workspace_tab_example.png" alt="Project workspace" >}}

### パネル追加セクションを作る

セクションのドロップダウンメニューで「Add section」を選択して、新しいパネル用セクションを作成できます。セクションの名前変更、ドラッグによる並び替え、折りたたみも可能です。

各セクションの右上には以下のオプションがあります：

* **Switch to custom layout**: 各パネルを個別にサイズ変更できるカスタムレイアウト
* **Switch to standard layout**: 全てまとめてサイズ変更、ページ分割付きの標準レイアウト
* **Add section**: ドロップダウンまたはページ下部ボタンからセクションを追加
* **Rename section**: セクションタイトルの変更
* **Export section to report**: 指定のセクションを新しい Report に保存
* **Delete section**: セクションごと削除。ワークスペースバー下部の「元に戻す」ボタンで復元可能
* **Add panel**: プラスボタンでパネル追加

{{< img src="/images/app_ui/add-section.gif" alt="Adding workspace section" >}}

### セクション間でパネルを移動

パネルをドラッグ & ドロップで、セクション間の並び替えや整理ができます。パネル右上の「Move」ボタンからも移動先セクションの選択が可能です。

{{< img src="/images/app_ui/move-panel.gif" alt="Moving panels between sections" >}}

### パネルサイズ変更

* **標準レイアウト**: 全てのパネルが同一サイズで、ページ分割も可能。右下をドラッグして個々に、またはセクション全体をドラッグしてサイズ変更
* **カスタムレイアウト**: 各パネルを個別にサイズ変更、ページ分割なし

{{< img src="/images/app_ui/resize-panel.gif" alt="Resizing panels" >}}

### メトリクス検索

ワークスペースにある検索ボックスを使い、表示パネルを絞り込めます。検索対象はパネルタイトルです（デフォルトでは可視化しているメトリクス名になります）。

{{< img src="/images/app_ui/search_in_the_workspace.png" alt="Workspace search" >}}

## Runs タブ

Runs タブでは Run のフィルタ、グループ、小並び替えが行えます。

{{< img src="/images/runs/run-table-example.png" alt="Runs table" >}}

以下のタブで、Runs タブでよく利用される操作例を紹介します。

{{< tabpane text=true >}}
   {{% tab header="カラムのカスタマイズ" %}}
Runs タブは、プロジェクト内の Run の情報を多数のカラムで表示します。

{{% alert %}}
Runs タブのカスタマイズ内容は、[Workspace タブ]({{< relref "#workspace-tab" >}})の **Runs** セレクタにも反映されます。
{{% /alert %}}

- 水平方向にページをスクロールすると、全てのカラムを表示できます。
- 並び順を変更したい場合は、カラムを左右にドラッグしてください。
- カラムをピン留めするには、カラム名にマウスオーバーしてアクションメニュー `...` をクリックし、**Pin column** を選択します。ピン留めされたカラムはページ左側、**Name** カラムの後ろに表示されます。ピン留め解除するには **Unpin column** を選択。
- カラムを非表示にする場合は、カラム名にマウスオーバーしてアクションメニュー `...` から **Hide column** を選びます。現在非表示のカラム一覧は **Columns** で確認可能です。
- 複数カラムの一括表示/非表示/ピン留め/ピン解除には **Columns** を使います。
  - 非表示カラム名をクリックすると表示されます。
  - 表示中カラム名をクリックすると非表示になります。
  - 表示中カラム横のピンアイコンでピン留めできます。

   {{% /tab %}}

   {{% tab header="ソート" %}}
指定したカラムの値で Table の全行を並び替えられます。

1. カラムタイトルにマウスを合わせます。ケバブメニュー（三点リーダー）が表示されます。
2. ケバブメニューをクリック。
3. **Sort Asc** や **Sort Desc** を選択して、昇順/降順に並び替えできます。

{{< img src="/images/data_vis/data_vis_sort_kebob.png" alt="Confident predictions" >}}

前の画像は、`val_acc` という Table カラムでソートする方法を示しています。
   {{% /tab %}}
   {{% tab header="フィルター" %}}
ダッシュボード左上の **Filter** ボタンで全行を式でフィルターできます。

{{< img src="/images/data_vis/filter.png" alt="Incorrect predictions filter" >}}

**Add filter** を選択し、複数のフィルター条件を追加できます。三つのドロップダウンが表示され、左から順に：カラム名、演算子、値の設定です。

|                   | カラム名      | 条件                | 値              |
| -----------       | ----------- | ----------- | ----------- |
| 有効値           | String       |  =, ≠, ≤, ≥, IN, NOT IN | Integer, float, string, timestamp, null |

式エディタはカラム名補完や論理述語構造の補完をサポートします。複数の論理述語を "and" や "or"（括弧も利用可）で組み合わせられます。

{{< img src="/images/data_vis/filter_example.png" alt="Filtering runs by validation loss" >}}
上の画像は `val_loss` カラムを基準にフィルターし、バリデーションロスが1以下の Run だけを表示しています。   
   {{% /tab %}}
   {{% tab header="グループ化" %}}
**Group by** ボタンから、特定カラムの値で全行をグループ化できます。

{{< img src="/images/data_vis/group.png" alt="Error distribution analysis" >}}

デフォルトで数値カラムはヒストグラムになり、各グループごとの値分布が可視化できます。グループ化はデータの高次的なパターンの理解に役立ちます。   
   {{% /tab %}}
{{< /tabpane >}}


## Automations タブ
Artifacts のバージョン管理を自動化できます。オートメーションはトリガーイベントと実行アクションを定義して作成します。アクション例として Webhook の実行や W&B ジョブの起動があります。詳細は [Automations]({{< relref "/guides/core/automations/" >}}) をご覧ください。

{{< img src="/images/app_ui/automations_tab.png" alt="Automation tab" >}}

## Reports タブ

全ての結果スナップショットを一箇所で確認でき、学びをチームと共有できます。

{{< img src="/images/app_ui/reports-tab.png" alt="Reports tab" >}}

## Sweeps タブ

プロジェクトから新しい [sweep]({{< relref "/guides/models/sweeps/" >}}) を開始できます。

{{< img src="/images/app_ui/sweeps-tab.png" alt="Sweeps tab" >}}

## Artifacts タブ

プロジェクトに紐付くすべての [artifacts]({{< relref "/guides/core/artifacts/" >}}) を確認できます。トレーニング用 Datasets や [fine-tuned models]({{< relref "/guides/core/registry/" >}})、[メトリクス／メディアテーブル]({{< relref "/guides/models/tables/tables-walkthrough.md" >}}) もここから利用できます。

### Overview パネル

{{< img src="/images/app_ui/overview_panel.png" alt="Artifact overview panel" >}}

Overview パネルでは、Artifacts の名前やバージョン、重複防止用のハッシュ、作成日時、エイリアスなど上位情報を確認できます。ここでエイリアスの追加・削除、バージョンごとのノート追加もできます。

### Metadata パネル

{{< img src="/images/app_ui/metadata_panel.png" alt="Artifact metadata panel" >}}

Metadata パネルでは Artifacts のメタデータが確認可能です。これは作成時に記録された設定値や再構築時に必要な引数、追加情報の URL や Run 中に生成されたメトリクスなどを含みます。また、Artifacts を作成した Run の設定や、その時点での履歴メトリクスも参照できます。

### Usage パネル

{{< img src="/images/app_ui/usage_panel.png" alt="Artifact usage panel" >}}

Usage パネルでは Artifacts をウェブアプリ外で利用する際のコードスニペット（例：ローカルマシンでのダウンロード）などが確認できます。また、この Artifacts を出力した Run、入力として利用した Run へのリンクも表示されます。

### Files パネル

{{< img src="/images/app_ui/files_panel.png" alt="Artifact files panel" >}}

Files パネルには Artifacts に紐付くファイルやフォルダが一覧されます。W&B は Run に必要なファイル（例: `requirements.txt` でライブラリバージョン, `wandb-metadata.json`, `wandb-summary.json` で実行情報）を自動アップロードします。その他、Artifacts やメディアは Run 設定次第で追加されます。このファイルツリーはサイト上で直接閲覧可能です。

Artifacts と関連付けた [Tables]({{< relref "/guides/models/tables//tables-walkthrough.md" >}}) は特にリッチ&インタラクティブです。Tables の活用方法は[こちら]({{< relref "/guides/models/tables//visualize-tables.md" >}})もご参照ください。

{{< img src="/images/app_ui/files_panel_table.png" alt="Artifact table view" >}}

### Lineage パネル

{{< img src="/images/app_ui/lineage_panel.png" alt="Artifact lineage" >}}

Lineage パネルは、プロジェクト内の全 Artifacts と、それらを結びつける Run の関係を可視化します。Run の種類はブロック、Artifacts は円で表示され、どの Run がどの Artifacts を入出力しているか矢印で示されます。左側のカラムで選択した Artifacts の種類が強調表示されます。

「Explode」トグルで、全ての個別バージョンや、それらを繋ぐ特定の Run も閲覧可能です。

### Action History Audit タブ

{{< img src="/images/app_ui/action_history_audit_tab_1.png" alt="Action history audit" >}}

{{< img src="/images/app_ui/action_history_audit_tab_2.png" alt="Action history" >}}

Action History Audit タブでは Collection に対する全てのエイリアス操作やメンバー変更を表示します。リソースの進化過程を監査できます。

### Versions タブ

{{< img src="/images/app_ui/versions_tab.png" alt="Artifact versions tab" >}}

Versions タブでは Artifacts の全バージョンと、それぞれの Run 履歴の数値カラムを一覧できます。これによりパフォーマンス比較や、興味深いバージョンの特定が容易です。

## プロジェクトを作成する

W&B App またはプログラムから `wandb.init()` でプロジェクト名を指定することでプロジェクトを作成できます。

{{< tabpane text=true >}}
   {{% tab header="W&B App" %}}
W&B App では **Projects** ページ、またはチームのランディングページからプロジェクトを作成できます。

**Projects** ページから:
1. 左上のグローバルナビゲーションアイコンをクリックしてサイドバーを開きます。
1. ナビゲーションの **Projects** セクションで **View all** をクリックし、プロジェクト概要ページを開きます。
1. **Create new project** をクリック。
1. **Team** でプロジェクトの所有チームを選択。
1. **Name** フィールドにプロジェクト名を入力。
1. **Project visibility** を設定。デフォルトは **Team** です。
1. 必要に応じて **Description** を入力。
1. **Create project** をクリック。

チームのランディングページから:
1. 左上のグローバルナビゲーションをクリックしサイドバーを開く。
1. **Teams** セクションでチーム名をクリックし、ランディングページを開きます。
1. 「Create new project」をクリック。
1. **Team** は見ていたページのチームで自動的にセットされています。必要なら変更してください。
1. **Name** フィールドにプロジェクト名を指定してください。
1. **Project visibility** を選択（デフォルトは **Team**）。
1. 必要に応じて **Description** を入力。
1. **Create project** をクリック。

   {{% /tab %}}
   {{% tab header="Python SDK" %}}
プログラムからプロジェクトを作成するには、`wandb.init()` の際に `project` を指定します。存在しない場合は自動的に作成され、指定した Entity の所有となります。例：

```python
import wandb with wandb.init(entity="<entity>", project="<project_name>") as run: run.log({"accuracy": .95})
```

[`wandb.init()` APIリファレンス]({{< relref "/ref/python/sdk/functions/init/#examples" >}}) もご覧ください。
   {{% /tab %}}  
{{< /tabpane >}}

## プロジェクトにスターを付ける

プロジェクトにスターを付けると重要なプロジェクトとしてマークできます。スターを付けたプロジェクトは、組織のホームページの上部に表示されます。

以下の画像例では、`zoo_experiment` と `registry_demo` の2つのプロジェクトにスターが付いており、**Starred projects** セクションの一番上に表示されています。
{{< img src="/images/track/star-projects.png" alt="Starred projects section" >}}


プロジェクトにスターを付ける方法は2通りあります。プロジェクトの Overview タブ、またはチームのプロフィールページから行えます。

{{< tabpane text=true >}}
    {{% tab header="Project overview" %}}
1. `https://wandb.ai/<team>/<project-name>` にアクセスし、自分の W&B プロジェクトを開きます。
2. サイドバーから **Overview** タブを選択。
3. 右上の **Edit** ボタンの横にある星アイコンをクリック。

{{< img src="/images/track/star-project-overview-tab.png" alt="Star project from overview" >}}    
    {{% /tab %}}
    {{% tab header="Team profile" %}}
1. `https://wandb.ai/<team>/projects` で自分のチームプロフィールページへ。
2. **Projects** タブを選択。
3. スターを付けたいプロジェクトの横にマウスを合わせ、表示される星アイコンをクリック。

下記の例画像では "Compare_Zoo_Models" プロジェクトの横にスターアイコンが表示されています。
{{< img src="/images/track/star-project-team-profile-page.png" alt="Star project from team page" >}}    
    {{% /tab %}}
{{< /tabpane >}}

プロジェクトにスターを付けたことを、アプリ左上の組織名をクリックしてランディングページで確認できます。

## プロジェクトを削除する

Overview タブ右側にある三点ドットのメニューからプロジェクトを削除できます。

{{< img src="/images/app_ui/howto_delete_project.gif" alt="Delete project workflow" >}}

プロジェクトが空の場合は、右上ドロップダウンメニューから **Delete project** 選択で削除も可能です。

{{< img src="/images/app_ui/howto_delete_project_2.png" alt="Delete empty project" >}}

## プロジェクトにノートを追加する

ノートは、プロジェクトの説明（Overview）としても Workspace の Markdown パネルとしても追加できます。

### プロジェクトに説明を追加する

追加した説明はプロフィールの **Overview** タブに表示されます。

1. W&B プロジェクトにアクセス
2. サイドバーから **Overview** タブを選択
3. 右上の Edit を選択
4. **Description** フィールドにノートを記入
5. **Save** ボタンを押して保存

{{% alert title="Run 比較メモは Reports で！" %}}
W&B Reportでグラフや Markdown を組み合わせて Run の比較説明を記録できます。異なる Run ごとにセクションを作り、作業経過や結果をストーリーでまとめましょう。
{{% /alert %}}

### Run Workspace にノートを追加

1. W&B プロジェクトにアクセス
2. サイドバーから **Workspace** タブを選択
3. 右上の **Add panels** ボタンをクリック
4. 表示されるモーダルで **TEXT AND CODE** ドロップダウンを選択
5. **Markdown** を選択
6. Workspace に表示された Markdown パネルにノートを書き込み