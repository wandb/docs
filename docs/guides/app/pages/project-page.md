---
description: 'モデルのバージョンを比較し、スクラッチワークスペースで結果を調査し、学びをレポートにエクスポートしてメモや可視化を保存しましょう

  '
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Project Page

**Workspace** は、実験を比較するための個人的なサンドボックスを提供します。Projects を使用して比較可能なモデルを整理し、異なるアーキテクチャー、ハイパーパラメーター、データセット、前処理などで同じ問題に取り組むことができます。

Project page のタブ:

1. [**Overview**](project-page.md#overview-tab): プロジェクトのスナップショット
2. [**Workspace**](project-page.md#workspace-tab): 個人用の可視化サンドボックス
3. [**Table**](project-page.md#table-tab): 全ての runs を一望
4. [**Reports**](project-page.md#reports-tab): メモ、runs、グラフの保存されたスナップショット
5. [**Sweeps**](project-page.md#sweeps-tab): 自動探索と最適化

## Overview Tab

* **Project name**: プロジェクト名をクリックして編集
* **Project description**: プロジェクトの説明をクリックして編集し、メモを追加
* **Delete project**: 右上のドットメニューをクリックしてプロジェクトを削除
* **Project privacy**: runs と Reports を誰が閲覧できるか編集—ロックアイコンをクリック
* **Last active**: 最も最近データがログされた時刻を表示
* **Total compute**: プロジェクト内の全ての run 時間を合計
* **Undelete runs**: ドロップダウンメニューをクリックして "Undelete all runs" を選択し、削除された runs を復元

[View a live example →](https://app.wandb.ai/example-team/sweep-demo/overview)

![](/images/app_ui/overview_tab_image.png)

![](/images/app_ui/undelete.png)

## Workspace Tab

**Runs Sidebar**: プロジェクト内の全ての runs のリスト

* **Dot menu**: サイドバーの行にホバーするとメニューが左側に表示されます。このメニューを使用して、run の名前を変更、削除、またはアクティブな run を停止できます。
* **Visibility icon**: 目のアイコンをクリックしてグラフ上の runs の表示を切り替え
* **Color**: プリセットまたはカスタムカラーに run 色を変更
* **Search**: 名前で runs を検索。これはプロット内の表示 runs もフィルタリング
* **Filter**: サイドバーフィルターを使用して表示される runs を絞り込み
* **Group**: 設定カラムを選択して動的に runs をグループ化。例えばアーキテクチャーごとにグループ化。グループ化により、グラフ上で平均値の線と、変動量の影付き領域が表示されます。
* **Sort**: 特定の値で runs をソート。例えば、最も低い損失または最高の精度の run。ソートはグラフに表示される runs に影響
* **Expand button**: サイドバーを完全なテーブルに展開
* **Run count**: 上部のカッコ内の数値は、プロジェクト内の全ての runs の総数です。`N visualized`とは、目のアイコンがオンになっているグラフで可視化されている runs の数を示します。以下の例では、183個の runs のうち最初の10個のみがグラフに表示されています。グラフを編集して表示可能な runs の最大数を増やせます。

**Panels layout**: このスクラッチスペースを使用して結果を探り、チャートを追加および削除し、異なるメトリクスに基づいてモデルのバージョンを比較

[View a live example →](https://app.wandb.ai/example-team/sweep-demo)

![](/images/app_ui/workspace_tab_example.png)

### Search for runs

サイドバーで名前で run を検索。正規表現を使用して表示された runs をフィルタリングできます。検索ボックスはグラフに表示された runs に影響します。以下は例です:

![](/images/app_ui/project_page_search_for_runs.gif)

### Add a section of panels

セクションのドロップダウンメニューをクリックして「セクションを追加」を選択し、新しいパネルセクションを作成。セクションの名前を変更、ドラッグして再配置、展開・縮小可能。

各セクションには右上にオプションがあります:

* **Switch to custom layout**: カスタムレイアウトでは、各パネルを個別にサイズ変更可能です。
* **Switch to standard layout**: 標準レイアウトでは、セクション内の全てのパネルのサイズを一度に変更し、ページネーションを提供します。
* **Add section**: ドロップダウンメニューを使用して上または下にセクションを追加、またはページ下部のボタンをクリックして新しいセクションを追加。
* **Rename section**: セクションのタイトルを変更。
* **Export section to report**: このパネルセクションを新しいレポートに保存。
* **Delete section**: セクション全体と全てのチャートを削除。これはワークスペースバーのページ下部にある元に戻すボタンで元に戻せます。
* **Add panel**: プラスボタンをクリックしてセクションにパネルを追加。

![](@site/static/images/app_ui/add-section.gif)

### Move panels between sections

パネルをドラッグアンドドロップして再配置し、セクションに整理。パネルの右上にある「移動」ボタンをクリックして、移動先のセクションを選択することもできます。

![](@site/static/images/app_ui/move-panel.gif)

### Resize panels

* **Standard layout**: 全てのパネルが同じサイズを維持し、ページ単位で表示されます。パネルのサイズを変更するには、右下隅をクリックしてドラッグ。セクションのサイズを変更するには、セクションの右下隅をクリックしてドラッグ。
* **Custom layout**: 全てのパネルが個別にサイズ変更され、ページはありません。

![](@site/static/images/app_ui/resize-panel.gif)

### Search for metrics

ワークスペース内の検索ボックスを使用してパネルをフィルタリング。この検索はパネルタイトルに一致し、デフォルトでは可視化されたメトリクスの名前を示します。

![](/images/app_ui/search_in_the_workspace.png)

## Table Tab

テーブルを使用して結果をフィルタ、グループ、ソート。

[View a live example →](https://app.wandb.ai/example-team/sweep-demo/table?workspace=user-carey)

![](/images/app_ui/table_tab.png)

### Table operations

W&B App を使用して W&B Tables をソート、フィルタ、グループ。

<Tabs
  defaultValue="sort"
  values={[
    {label: 'Sort', value: 'sort'},
    {label: 'Filter', value: 'filter'},
    {label: 'Group', value: 'group'},
  ]}>
  <TabItem value="sort">

与えられたカラムの値で Table の全行をソート。

1. カラムタイトルにマウスをホバー。ケボブメニュー（二重縦線）が表示されます。
2. ケボブメニュー（三重縦線）を選択。
3. **Sort Asc** または **Sort Desc** を選択して、行を昇順または降順にソート。

![See the digits for which the model most confidently guessed "0".](/images/data_vis/data_vis_sort_kebob.png)

上記の画像は、`val_acc` というカラムのソートオプションを表示する方法を示しています。

</TabItem>
  <TabItem value="filter">

**Filter** ボタンを使って、フィルタ条件を指定し、表全体の行を絞り込み。

![See only examples which the model gets wrong.](/images/data_vis/filter.png)

**Add filter** を選択して、行に1つ以上のフィルターを追加。3つのドロップダウンメニューが表示されます。左から順に、カラム名、オペレーター、値 に基づいてフィルタ条件を設定。

|                   | カラム名 | 比較演算子    | 値       |
| -----------       | ----------- | ----------- | ----------- |
| 受け入れ可能な値   | 文字列       | =, ≠, ≤, ≥, IN, NOT IN,  | 整数、浮動小数点、文字列、タイムスタンプ、null |


式エディターは、カラム名と論理述語構造に基づいたオプションのリストを autocomplete で表示。複数の論理述語を "and" または "or"（および時にはカッコ）で結合して１つの式に。

![](/images/data_vis/filter_example.png)
上記の画像は、`val_loss` カラムに基づいたフィルターを示しています。このフィルターは、検証損失が1以下の runs を表示します。

</TabItem>
  <TabItem value="group">

特定のカラムの値で全行をグループ化するには、カラムヘッダーの **Group by** ボタンを使用。

![The truth distribution shows small errors: 8s and 2s are confused for 7s and 9s for 2s.](/images/data_vis/group.png)

デフォルトでは、数値カラムが各グループの値分布を示すヒストグラムに変わります。グループ化はデータの上位パターンを理解するのに役立ちます。

  </TabItem>
</Tabs>

## Reports Tab

成果のスナップショットを一箇所に集め、チームと共有。

![](@site/static/images/app_ui/reports-tab.png)

## Sweeps Tab

プロジェクトから新しい [sweep](../../sweeps/intro.md) を開始。

![](@site/static/images/app_ui/sweeps-tab.png)

## Artifacts Tab

プロジェクトに関連付けられた全ての [artifacts](../../artifacts/intro.md) を表示。トレーニングデータセットや[ファインチューンされたモデル](../../model_registry/intro.md)から[メトリクスやメディアのテーブル](../../tables/tables-walkthrough.md)まで。

### Overview panel

![](/images/app_ui/overview_panel.png)

概要パネルでは、artifact の名前とバージョン、重複を防ぐためのハッシュダイジェスト、作成日、エイリアスなど、artifact に関するさまざまな高レベルの情報を確認できます。ここでエイリアスを追加・削除したり、バージョンやアーティファクト全体に対してメモを取ったりできます。

### Metadata panel

![](/images/app_ui/metadata_panel.png)

メタデータパネルは、artifact の作成時に提供されるメタデータへのアクセスを提供します。このメタデータには、artifact を再構築するために必要な設定引数、詳細情報を見つけるためのURL、artifact をログする run 中に生成されたメトリクスが含まれます。また、artifact を生成した run の設定や、artifact をログした時点の履歴メトリクスも確認できます。

### Usage panel

![](/images/app_ui/usage_panel.png)

使用パネルでは、artifact を web アプリの外部（例：ローカルマシン）で使用するためのコードスニペットが提供されます。このセクションには、artifact を出力した run と入力として使用する runs へのリンクも表示されます。

### Files panel

![](/images/app_ui/files_panel.png)

ファイルパネルには、artifact に関連付けられたファイルとフォルダがリスト表示されます。このファイルツリーをナビゲートし、W&B web アプリで内容を直接確認できます。

artifact に関連付けられた [Tables](../../tables/tables-walkthrough.md) は特にリッチでインタラクティブです。Artifacts と Tables の使用方法についての詳細は [こちら](../../tables/visualize-tables.md) をご覧ください。

![](/images/app_ui/files_panel_table.png)

### Lineage panel

![](/images/app_ui/lineage_panel.png)

リネージ パネルは、プロジェクトに関連する全ての artifacts とそれらを相互に接続する runs を表示。run タイプはブロック、artifact は円で表示され、特定の run タイプが特定の artifact タイプを消費または生成する場合に矢印が表示されます。左のカラムで選択した特定の artifact タイプがハイライト表示されます。

個々の artifact バージョンとそれらを接続する特定の runs を表示するには、"Explode" トグルをクリック。

### Action History Audit tab

![](/images/app_ui/action_history_audit_tab_1.png)

![](/images/app_ui/action_history_audit_tab_2.png)

アクション履歴監査タブは、コレクションのエイリアスアクションやメンバーシップ変更を全て表示して、リソースの進化過程を監査できます。

### Versions tab

![](/images/app_ui/versions_tab.png)

バージョンタブには、artifact の全てのバージョンと、バージョンをログした時点の Run History の各数値が表示されます。これにより、パフォーマンスを比較し、注目すべきバージョンを迅速に特定できます。

## Project Defaults

User Settings の `/settings` でプロジェクトのデフォルト設定を _手動で_ 変更できます。

* **新しいプロジェクトを作成するデフォルトロケーション**: デフォルトで自身の個人エンティティに設定されています。ドロップダウンをクリックして、個人エンティティと所属するチームを切り替え可能。
* **個人アカウントのデフォルトのプロジェクトプライバシー**: デフォルトで 'Private' に設定されています。つまり、プロジェクトはプライベートであり、あなたしかアクセスできません。
* **個人アカウントでのコード保存の有効化**: デフォルトで無効になっています。有効にすると、主要なスクリプトやノートブックを W&B に保存できます。

:::note
これらの設定は引数を渡して指定することも可能です
[`wandb.init`](../../../ref/python/init.md).
:::

![](/images/app_ui/project_defaults.png)

## Frequently Asked Questions

### プロジェクトを削除するには？

Overviewタブの右側にある3つの点をクリックしてプロジェクトを削除できます。

![](/images/app_ui/howto_delete_project.gif)

プロジェクトが空である（つまり、run がない）場合は、右上のドロップダウンメニューをクリックして「プロジェクトを削除」を選択します。

![](/images/app_ui/howto_delete_project_2.png)

### プロジェクトのプライバシー設定はどこにありますか？ プロジェクトを公開または非公開にするには？

ページ上部のナビゲーションバーにあるロックアイコンをクリックしてプロジェクトのプライバシー設定を変更します。run または Reports を視聴または送信できるユーザーを編集できます。これらの設定はプロジェクト内の全ての run とレポートに適用されます。少人数と結果を共有したい場合は、[プライベートチーム](../features/teams.md)を作成できます。

![](/images/app_ui/privacy_settings.png)

### ワークスペースをリセットするには？

プロジェクトページに以下のようなエラーが表示された場合、ワークスペースをリセットする方法はこちらです。 `"objconv: "100000000000" overflows the maximum values of a signed 64 bits integer"`

URLの末尾に `?workspace=clear` を追加してEnterキーを押します。これでプロジェクトページのワークスペースがクリアされます。