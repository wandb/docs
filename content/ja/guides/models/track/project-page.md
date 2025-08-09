---
title: プロジェクト
description: モデルのバージョンを比較し、スクラッチワークスペースで結果を探索し、学びや可視化をレポートにエクスポートしてメモを保存しましょう
menu:
  default:
    identifier: ja-guides-models-track-project-page
    parent: experiments
weight: 3
---

*プロジェクト*は、結果の可視化、実験比較、Artifacts の閲覧・ダウンロード、オートメーションの作成などを一箇所で行える中心的なスペースです。

{{% alert %}}
各プロジェクトには公開範囲の設定があります。これにより誰がそのプロジェクトにアクセスできるかが決まります。プロジェクトへのアクセス権限についての詳細は、[プロジェクトの公開範囲]({{< relref path="/guides/hosting/iam/access-management/restricted-projects.md" lang="ja" >}})をご覧ください。
{{% /alert %}}

プロジェクトには以下のタブが含まれています：

* [Overview]({{< relref path="project-page.md#overview-tab" lang="ja" >}})：プロジェクトの概要スナップショット
* [Workspace]({{< relref path="project-page.md#workspace-tab" lang="ja" >}})：個人の可視化サンドボックス
* [Runs]({{< relref path="#runs-tab" lang="ja" >}})：プロジェクト内の全ての run を一覧表示するテーブル
* [Automations]({{< relref path="#automations-tab" lang="ja" >}})：プロジェクトで設定されているオートメーション
* [Sweeps]({{< relref path="project-page.md#sweeps-tab" lang="ja" >}})：自動探索・最適化
* [Reports]({{< relref path="project-page.md#reports-tab" lang="ja" >}})：ノート、run、グラフ等のスナップショット保存
* [Artifacts]({{< relref path="#artifacts-tab" lang="ja" >}})：すべての run と、それに紐づく Artifacts

## Overviewタブ

* **Project name**: プロジェクトの名前です。W&B では、run の初期化時に指定したプロジェクト名で新規プロジェクトが作成されます。右上の**Edit**ボタンから、いつでもプロジェクト名を変更できます。
* **Description**: プロジェクトの説明文。
* **Project visibility**: プロジェクトの公開範囲。誰がアクセス可能かを指定します。詳細は[プロジェクトの公開範囲]({{< relref path="/guides/hosting/iam/access-management/restricted-projects.md" lang="ja" >}})を参照ください。
* **Last active**: このプロジェクトに最後にデータがログされた日時
* **Owner**: プロジェクトの所有 Entity
* **Contributors**: プロジェクトに貢献しているユーザー数
* **Total runs**: プロジェクト内の run の総数
* **Total compute**: プロジェクト内すべての run の合計実行時間
* **Undelete runs**: ドロップダウンメニューから「Undelete all runs」をクリックすると、削除済みの run を復元できます。
* **Delete project**: 右上のメニュードットからプロジェクトを削除できます。

[ライブ例を見る](https://app.wandb.ai/example-team/sweep-demo/overview)

{{< img src="/images/track/overview_tab_image.png" alt="Project overview tab" >}}


## Workspaceタブ

プロジェクトの*workspace*は、実験を比較するための個人用サンドボックスです。異なるモデル、アーキテクチャ、ハイパーパラメーター、データセット、前処理などで同じ課題に取り組む際の整理に役立ちます。

**Runs サイドバー**: プロジェクト内のすべての run を一覧表示します。

* **ドットメニュー**: サイドバー内の行にカーソルを合わせると左側にメニューが表示されます。このメニューで run の名前変更、削除、またはアクティブな run の停止ができます。
* **表示アイコン(目のアイコン)**: グラフ上で run の表示/非表示を切り替え
* **カラー**: run の色をプリセットやカスタムカラーで変更
* **検索**: 名前で run を検索。プロットに表示される run も絞り込みます。
* **フィルター**: サイドバーのフィルターで可視化される run を絞り込み
* **グループ(集約)**: 設定カラムを選択して run を動的にグループ化。例えばアーキテクチャごとにグループ化可能。グループ化すると、グラフに平均値の線と分散範囲(シェーディング)が表示されます。
* **並び替え**: 最小ロス、最大精度など基準値で run をソート。並び替えた順でグラフ表示にも反映されます。
* **展開ボタン**: サイドバーをフルテーブルへ展開
* **run 数**: 上部のカッコ内の数字はプロジェクト内全 run 数。「(N visualized)」はグラフに表示可能な run 数です。以下の例では183 run のうち最初の10件のみがグラフ表示中。編集で最大可視 run 数を増やせます。

[Runs タブ](#runs-tab) で列のピン止め・非表示・順序変更を行うと、Runs サイドバーにもそれが反映されます。

**パネルレイアウト**: このスペースで結果探索やチャートの追加/削除、異なるメトリクスでモデル・バージョンを比較できます。

[ライブ例を見る](https://app.wandb.ai/example-team/sweep-demo)

{{< img src="/images/app_ui/workspace_tab_example.png" alt="Project workspace" >}}


### パネルセクションの追加

セクションドロップダウンメニューで「Add section」をクリックすると新しいパネル用セクションが作成できます。セクション名の変更、ドラッグによる移動、展開・折りたたみが可能です。

各セクションの右上メニューで以下が行えます：

* **Switch to custom layout**: 個別にパネルサイズの調整ができるカスタムレイアウトへ
* **Switch to standard layout**: セクション内全パネルを一括リサイズし、ページ切替も可能な標準レイアウトへ
* **Add section**: ドロップダウンから上または下に新セクション追加。ページ下部のボタンからも追加可能。
* **Rename section**: セクションのタイトル変更
* **Export section to report**: セクション内パネルを新規レポートとして保存
* **Delete section**: セクションごと削除（ページ下部のundoボタンで元に戻せます）
* **Add panel**: プラスを押してセクションにパネル追加

{{< img src="/images/app_ui/add-section.gif" alt="Adding workspace section" >}}

### セクション間でのパネル移動

パネルをドラッグ＆ドロップで再配置し各セクションに整理できます。パネル右上の「Move」ボタンからも、移動先セクションを選択して移動できます。

{{< img src="/images/app_ui/move-panel.gif" alt="Moving panels between sections" >}}

### パネルのリサイズ

* **Standard layout**: すべてのパネルが同一サイズを保ち、ページ切替で表示。パネル右下をドラッグでリサイズ、セクション自体も同様にリサイズできます。
* **Custom layout**: パネルごとに自由なサイズ指定可能。ページ切替はありません。

{{< img src="/images/app_ui/resize-panel.gif" alt="Resizing panels" >}}

### メトリクスの検索

ワークスペース内の検索ボックスでパネルをフィルタリングできます。検索はパネルタイトル（＝デフォルトでメトリクス名）に一致します。

{{< img src="/images/app_ui/search_in_the_workspace.png" alt="Workspace search" >}}

## Runsタブ

Runs タブでは、run のフィルター・グループ化・並び替えが可能です。

{{< img src="/images/runs/run-table-example.png" alt="Runs table" >}}

下記のタブで、Runs タブ内でよく使う操作例を紹介します。

{{< tabpane text=true >}}
   {{% tab header="Customize columns" %}}
Runs タブでは、プロジェクト内 run の詳細情報が各種カラムに表示され、多くのカラムがデフォルトで見られます。

{{% alert %}}
Runsタブのカスタマイズ内容は、[Workspaceタブ]({{< relref path="#workspace-tab" lang="ja" >}}) の **Runs** セレクターにも反映されます。
{{% /alert %}}

- 表示中の全カラムを見るにはページを横スクロールしてください。
- カラムの並び順を変更するには、カラムを左右にドラッグします。
- カラムをピン止めするには、カラム名にカーソルを合わせ、アクションメニュー `...` をクリックし、**Pin column** を選択。ピン止めしたカラムは**Name**列の右に固定表示されます。解除は **Unpin column** を選択。
- カラムを非表示にする場合もカラム名の `...` メニューから **Hide column** を選択。**Columns** をクリックで現在非表示のカラム一覧が見られます。
- 複数カラムの一括表示・非表示・ピン止め切替も **Columns** から操作可能。
  - 非表示カラム名をクリックで再表示
  - 表示中カラム名をクリックで非表示
  - ピンアイコンをクリックでピン止め可能

   {{% /tab %}}

   {{% tab header="Sort" %}}
Table の任意カラムを基準にすべての行をソートできます。

1. カラムタイトルにカーソルを合わせると「ケバブメニュー」（縦の点3つ）が表示されます。
2. ケバブメニューをクリック。
3. **Sort Asc** または **Sort Desc** を選択し、昇順・降順で並び替えてください。

{{< img src="/images/data_vis/data_vis_sort_kebob.png" alt="Confident predictions" >}}

上記画像は、`val_acc` カラムで並び替えを行う方法を示しています。   
   {{% /tab %}}
   {{% tab header="Filter" %}}
すべての行を **Filter** ボタンから式で絞り込みできます。ダッシュボード左上にあります。

{{< img src="/images/data_vis/filter.png" alt="Incorrect predictions filter" >}}

**Add filter** を選択すると1つ以上のフィルター条件を追加できます。左から順に：カラム名・比較演算子・値 の3つのドロップダウンが出ます。

|                   | カラム名        | 比較演算子          | 値                    |
| -----------       | -----------    | -----------         | -----------           |
| 入力例            | String         |  =, ≠, ≤, ≥, IN, NOT IN,  | Integer, float, string, timestamp, null |


式エディタはカラム名やロジカル述語構造を自動補完で候補表示します。and/or（およびカッコ）で複数述語の組み合わせも可能です。

{{< img src="/images/data_vis/filter_example.png" alt="Filtering runs by validation loss" >}}
この画像では `val_loss` カラムを使って「バリデーションロスが1以下」の run を表示しています。   
   {{% /tab %}}
   {{% tab header="Group" %}}
**Group by** ボタンで、指定カラムの値ごとにすべての行をグループ化できます。

{{< img src="/images/data_vis/group.png" alt="Error distribution analysis" >}}

グループ化すると他の数値カラムが自動でヒストグラム表示に切り替わり、そのカラムの値の分布がグループ単位で可視化されます。データ全体のパターン把握に役立ちます。   
   {{% /tab %}}
{{< /tabpane >}}



## Automationsタブ

Artifacts のバージョン管理用途や、下流アクションの自動化ができます。オートメーション作成時はトリガーとなるイベントとアクションを設定し、Webhook 呼び出しや W&B Job の起動が可能です。詳細は[Automations]({{< relref path="/guides/core/automations/" lang="ja" >}})をご覧ください。

{{< img src="/images/app_ui/automations_tab.png" alt="Automation tab" >}}

## Reportsタブ

結果のスナップショットを一元管理し、チームメンバーと学びを共有できます。

{{< img src="/images/app_ui/reports-tab.png" alt="Reports tab" >}}

## Sweepsタブ

プロジェクトから新しい [sweep]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) を開始できます。

{{< img src="/images/app_ui/sweeps-tab.png" alt="Sweeps tab" >}}

## Artifactsタブ

プロジェクトに紐づくすべての [artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) を閲覧できます。トレーニングデータセットや [ファインチューン済みモデル]({{< relref path="/guides/core/registry/" lang="ja" >}})、[メトリクスやメディアのテーブル]({{< relref path="/guides/models/tables/tables-walkthrough.md" lang="ja" >}}) などが含まれます。

### Overviewパネル

{{< img src="/images/app_ui/overview_panel.png" alt="Artifact overview panel" >}}

Overviewパネルでは、artifact の名前とバージョン、重複防止用ハッシュ、作成日時、エイリアスなど、基本的な情報が一覧できます。ここでエイリアスの追加/削除や、バージョン単位・artifact単位のノート追加も可能です。

### Metadataパネル

{{< img src="/images/app_ui/metadata_panel.png" alt="Artifact metadata panel" >}}

Metadataパネルではartifactを作成した時に付与したメタデータ情報が閲覧できます。artifact の再現に必要な設定引数や、詳細情報URL、run内で生成されたメトリクスなども含まれます。また、そのartifactを生んだ run の設定と、その時点の履歴メトリクスも確認できます。

### Usageパネル

{{< img src="/images/app_ui/usage_panel.png" alt="Artifact usage panel" >}}

Usageパネルでは、webアプリ外（例：ローカルマシン）でartifactをダウンロードし利用するためのコードスニペットを確認できます。また、そのartifactを出力した run や、入力として利用した run もインジケータやリンク付きで示されます。

### Filesパネル

{{< img src="/images/app_ui/files_panel.png" alt="Artifact files panel" >}}

Filesパネルにはartifactに紐づくファイルやフォルダが一覧されます。W&Bはrunごとに `requirements.txt`（ライブラリのバージョン記録）、`wandb-metadata.json`、`wandb-summary.json`（run 情報）など自動でアップロードします。他にもrunの設定に応じてartifactデータやメディアなどがアップロードされます。ファイルツリーをたどり、webアプリ内で内容を直接閲覧できます。

[テーブル]({{< relref path="/guides/models/tables//tables-walkthrough.md" lang="ja" >}})はArtifactと組み合わせることで特にリッチかつインタラクティブに利用できます。Artifacts と Tables の詳細は[こちら]({{< relref path="/guides/models/tables//visualize-tables.md" lang="ja" >}})から。

{{< img src="/images/app_ui/files_panel_table.png" alt="Artifact table view" >}}

### Lineageパネル

{{< img src="/images/app_ui/lineage_panel.png" alt="Artifact lineage" >}}

Lineageパネルでは、プロジェクト内すべてのArtifactと、それらを結びつける run の関係性を視覚的に表示します。runタイプはブロック、Artifactは円で表され、runがどのtypeのArtifactを入出力するかは矢印で示されます。左カラムで選択中のArtifact typeがハイライトされます。

「Explode」トグルをクリックすると、個別Artifactバージョンやそれらを結ぶrunもすべて個別表示されます。

### アクション履歴・監査タブ

{{< img src="/images/app_ui/action_history_audit_tab_1.png" alt="Action history audit" >}}

{{< img src="/images/app_ui/action_history_audit_tab_2.png" alt="Action history" >}}

アクション履歴監査タブでは、コレクション内のエイリアス追加・削除やメンバーシップ変更など、リソース進化の全履歴を確認できます。

### Versionsタブ

{{< img src="/images/app_ui/versions_tab.png" alt="Artifact versions tab" >}}

Versionsタブでは、そのartifactの全バージョンと、各バージョンが記録された際のRun Historyから数値系カラムを一覧表示します。パフォーマンス比較や注目バージョンの特定に役立ちます。

## プロジェクト作成

W&B App 内またはプログラムから `wandb.init()` の引数としてプロジェクト指定することで作成できます。

{{< tabpane text=true >}}
   {{% tab header="W&B App" %}}
W&B Appでは、**Projects** ページやチームページからプロジェクトを作成できます。

**Projects** ページからの作成手順:
1. 左上のグローバルナビゲーションアイコンをクリックし、サイドバーを開く
1. ナビゲーション内の **Projects** セクションで **View all** をクリックし、プロジェクト一覧ページを開く
1. **Create new project** をクリック
1. **Team** でプロジェクト所有チーム名を指定
1. **Name** フィールドにプロジェクト名を指定
1. **Project visibility** を設定（デフォルトは **Team**）
1. **Description**（任意）を記入
1. **Create project** をクリック

チームページからの作成手順:
1. 左上のグローバルナビゲーションアイコンをクリックし、サイドバーを開く
1. ナビゲーション内の **Teams** セクションで該当チーム名をクリックし、そのチームページへ
1. ページ内で **Create new project** をクリック
1. **Team** はそのページのチーム名が自動セット。必要に応じて変更
1. **Name** フィールドでプロジェクト名を指定
1. **Project visibility** を設定（デフォルトは **Team**）
1. **Description**（任意）を記入
1. **Create project** をクリック

   {{% /tab %}}
   {{% tab header="Python SDK" %}}
プログラムから作成するには、`wandb.init()` 実行時に `project` 引数を指定してください。プロジェクトがまだなければ自動作成 & 指定 entity の所有プロジェクトとなります。例：

```python
import wandb
with wandb.init(entity="<entity>", project="<project_name>") as run:
    run.log({"accuracy": .95})
```

[`wandb.init()` APIリファレンス]({{< relref path="/ref/python/sdk/functions/init/#examples" lang="ja" >}})もご参照ください。
   {{% /tab %}}  
{{< /tabpane >}}

## プロジェクトのお気に入り登録

プロジェクトにスターを付けることで、そのプロジェクトを重要なものとしてマークできます。あなたやチームがスターを付けたプロジェクトは、組織トップページの上部 **Starred projects** セクションに表示されます。

例えば以下の図は、`zoo_experiment` と `registry_demo` という 2つのプロジェクトが重要扱いで上位表示されている例です。
{{< img src="/images/track/star-projects.png" alt="Starred projects section" >}}


スター付与の方法は、プロジェクトの overview タブまたはチームのプロフィールページの2通りがあります。

{{< tabpane text=true >}}
    {{% tab header="Project overview" %}}
1. W&B App の `https://wandb.ai/<team>/<project-name>` で対象プロジェクトを開く
2. サイドバーから **Overview** タブを開く
3. 右上 **Edit** ボタン横のスターアイコンを選択

{{< img src="/images/track/star-project-overview-tab.png" alt="Star project from overview" >}}    
    {{% /tab %}}
    {{% tab header="Team profile" %}}
1. チームプロフィールページ（`https://wandb.ai/<team>/projects`）を開く
2. **Projects** タブを選択
3. スター付与したいプロジェクト名の横にマウスを乗せ、表示されるスターアイコンをクリック

例えば下図は「Compare_Zoo_Models」プロジェクト横に現れるスターアイコン例です。
{{< img src="/images/track/star-project-team-profile-page.png" alt="Star project from team page" >}}    
    {{% /tab %}}
{{< /tabpane >}}

プロジェクトが組織トップページで上部に表示されていることは、アプリ左上で組織名をクリックして確認できます。

## プロジェクト削除

Overviewタブ右側の三点メニュー（3点リーダ）からプロジェクトを削除できます。

{{< img src="/images/app_ui/howto_delete_project.gif" alt="Delete project workflow" >}}

プロジェクトが空の場合は、右上のドロップダウンメニューで **Delete project** を選択すれば削除可能です。

{{< img src="/images/app_ui/howto_delete_project_2.png" alt="Delete empty project" >}}


## プロジェクトにノートを追加

プロジェクトへのノートは、説明欄（Overview）またはワークスペース内のマークダウンパネルとして追加できます。

### プロジェクトに記述概要を追加

追加した説明はプロフィールの **Overview** タブに表示されます。

1. W&B プロジェクトページへアクセス
2. サイドバーから **Overview** タブを選択
3. 右上 **Edit** を選ぶ
4. **Description** フィールドにノートを追加
5. **Save** ボタンをクリック

{{% alert title="run 比較など記述ノート作成には Reports 利用がおすすめ" %}}
W&B の Report を使えば、チャートとマークダウンノートを並べて表示できます。セクション分けし、複数の run を比較しながら結果をストーリーとしてまとめることができます。
{{% /alert %}}


### ワークスペースにノートを追加

1. W&B プロジェクトページへアクセス
2. サイドバーから **Workspace** タブを選択
3. 右上 **Add panels** ボタンをクリック
4. 表示されるモーダルで **TEXT AND CODE** を選択
5. **Markdown** を選択
6. ワークスペースに出現したマークダウンパネルにノートを記入