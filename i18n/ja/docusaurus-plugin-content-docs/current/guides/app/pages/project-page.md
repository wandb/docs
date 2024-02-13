---
description: >-
  Compare versions of your model, explore results in a scratch workspace, and
  export findings to a report to save notes and visualizations
displayed_sidebar: ja
---

# プロジェクトページ

プロジェクトの**ワークスペース**は、実験を比較するための個人用のサンドボックスです。異なるアーキテクチャー、ハイパーパラメーター、データセット、前処理などで同じ問題に取り組むモデルを比較できるように、プロジェクトを使って整理します。

プロジェクトページのタブ：

1. [**概要**](project-page.md#overview-tab): プロジェクトのスナップショット
2. [**ワークスペース**](project-page.md#workspace-tab): 個人用の可視化サンドボックス
3. [**テーブル**](project-page.md#table-tab): すべてのrunsの俯瞰図
4. [**レポート**](project-page.md#reports-tab): ノート、runs、グラフの保存されたスナップショット
5. [**スイープ**](project-page.md#sweeps-tab): 自動化された探索と最適化

## 概要タブ

* **プロジェクト名**: クリックしてプロジェクト名を編集します
* **プロジェクトの説明**: クリックしてプロジェクトの説明を編集し、メモを追加します
* **プロジェクトの削除**: 右上のドットメニューをクリックしてプロジェクトを削除します
* **プロジェクトのプライバシー**: runsやレポートが誰に見られるかを編集 – 鍵アイコンをクリック
* **最終アクティブ**: このプロジェクトに最も最近データがログされた日時
* **合計コンピュート**: プロジェクト内のすべての実行時間を合計してこの数値が得られます
* **削除済みrunsの復元**: ドロップダウンメニューをクリックして「すべての削除済みrunsを復元」をクリックし、プロジェクト内の削除済みrunsを回復します。

[ライブ例を見る →](https://app.wandb.ai/example-team/sweep-demo/overview)

![](/images/app_ui/overview_tab_image.png)
![](/images/app_ui/undelete.png)

## ワークスペースタブ

**Runsサイドバー**: プロジェクト内のすべてのrunsのリスト

* **ドットメニュー**: サイドバーの行にカーソルを合わせると左側にメニューが現れます。このメニューでrunの名前を変更したり、runを削除したり、アクティブなrunを停止したりできます。
* **表示アイコン**: 目のアイコンをクリックして、グラフ上の runs の表示をオン / オフにします
* **色**: runの色を、プリセットの別の色やカスタムカラーに変更します
* **検索**: runsを名前で検索します。これにより、プロット内の表示されるrunsも絞り込まれます。
* **フィルター**: サイドバーのフィルター機能を利用して、表示されるrunsを絞り込むことができます
* **グループ**: 設定列を選択して、アーキテクチャーなどの項目でrunsを動的にグループ化してください。グルーピングすると、プロットに平均値に沿った線と、グラフ上の点の分散範囲が表示されます。
* **並び替え**: 例えば最も損失が少ないrunsや正確性が高いrunsなど、runsを並び替える基準を選択します。並び替えは、グラフに表示されるrunsに影響します。
* **拡張ボタン**: サイドバーを完全なテーブルに拡張します
* **Runの数**: 上部のかっこ内の数字は、プロジェクト内のrunsの合計数です。数値(Nが可視化されています)は、目のアイコンがオンになっており、各プロットで可視化できるrunsの数です。以下の例では、グラフは183のrunsのうち最初の10個だけを表示しています。表示される上限runsの数を増やすには、グラフを編集してください。

**パネルレイアウト**: このスクラッチスペースを使って結果を探索し、チャートを追加・削除したり、モデルの異なるバージョンをメトリクスに基づいて比較してください

[ライブ例を見る →](https://app.wandb.ai/example-team/sweep-demo)

![](/images/app_ui/workspace_tab_example.png)

### runs を検索する

サイドバーでrunの名前を検索します。正規表現を使って表示するrunsを絞り込むことができます。検索ボックスは、グラフに表示されるrunsに影響します。以下に例を示します:

![](/images/app_ui/project_page_search_for_runs.gif)

### パネルのセクションを追加する
セクションのドロップダウンメニューをクリックし、「セクションを追加」をクリックして、パネル用の新しいセクションを作成します。セクションの名前を変更したり、ドラッグして並べ替えたり、展開したり折りたたんだりできます。

各セクションには、右上隅にオプションがあります：

* **カスタムレイアウトに切り替え**: カスタムレイアウトでは、パネルを個別にサイズ変更できます。
* **標準レイアウトに切り替え**: 標準レイアウトでは、一度にセクション内のすべてのパネルのサイズを変更でき、ページ送りが可能です。
* **セクションの追加**: ドロップダウンメニューから上または下にセクションを追加するか、ページの下部にあるボタンをクリックして新しいセクションを追加します。
* **セクションの名前の変更**: セクションのタイトルを変更します。
* **レポートにセクションをエクスポート**: パネルのこのセクションを新しいレポートとして保存します。
* **セクションの削除**: セクション全体とすべてのグラフを削除します。これは、ワークスペースバーのページ下部にある元に戻すボタンで元に戻すことができます。
* **パネルの追加**: プラスボタンをクリックして、セクションにパネルを追加します。

![](@site/static/images/app_ui/add-section.gif)

### セクション間でパネルを移動する

パネルをドラッグアンドドロップして、セクションに並べ替えたり整理したりできます。また、パネルの右上隅にある「移動」ボタンをクリックして、パネルを移動するセクションを選択することもできます。

![](@site/static/images/app_ui/move-panel.gif)

### パネルのサイズ変更

* **標準レイアウト**: すべてのパネルは同じサイズを維持し、パネルのページがあります。右下隅をクリックしてドラッグすることで、パネルのサイズを変更できます。セクションの右下隅をクリックしてドラッグして、セクションのサイズを変更します。
* **カスタムレイアウト**: すべてのパネルは個別にサイズが設定され、ページはありません。

![](@site/static/images/app_ui/resize-panel.gif)

### メトリクスの検索

ワークスペース内の検索ボックスを使用して、パネルを絞り込みます。この検索は、パネルのタイトルに一致し、デフォルトでは表示されているメトリクスの名前になります。
![](/images/app_ui/search_in_the_workspace.png)

## テーブルタブ

テーブルを使用して、結果をフィルター、グループ化、並び替えします。

[ライブ例を見る →](https://app.wandb.ai/example-team/sweep-demo/table?workspace=user-carey)

![](/images/app_ui/table_tab.png)

## レポートタブ

結果のスナップショットを一か所で表示し、チームと調査結果を共有します。

![](@site/static/images/app_ui/reports-tab.png)

## スイープタブ

プロジェクトから新しい[スイープ](../../sweeps/intro.md)を開始します。

![](@site/static/images/app_ui/sweeps-tab.png)

## アーティファクトタブ

プロジェクトに関連するすべての[アーティファクト](../../artifacts/intro.md)を表示し、トレーニングデータセットや[微調整されたモデル](../../model_registry/intro.md)、[メトリクスとメディアのテーブル](../../tables/tables-walkthrough.md)を確認します。

### 概要パネル

![](/images/app_ui/overview_panel.png)
概要パネルでは、アーティファクトの名前やバージョン、変更を検出し重複を防ぐためのハッシュダイジェスト、作成日、エイリアスなど、アーティファクトに関する様々な高レベルの情報が表示されます。ここでエイリアスの追加や削除ができ、バージョンやアーティファクト全体についてメモを取ることができます。

### メタデータパネル

![](/images/app_ui/metadata_panel.png)

メタデータパネルは、アーティファクトのメタデータにアクセスする機能を提供します。このメタデータは、アーティファクトが構築されたときに提供されます。このメタデータには、アーティファクトを再構築するために必要な設定引数、詳細情報が入手できるURL、または、アーティファクトをログに記録したランで生成されたメトリクスが含まれることがあります。また、アーティファクトを生成したランの設定や、アーティファクトがログに記録された時点での履歴メトリクスを確認することができます。

### 使用法パネル

![](/images/app_ui/usage_panel.png)

Usageパネルでは、Webアプリケーション以外でアーティファクトを使用するためのコードをダウンロードするためのスニペットが提供されます。例えば、ローカルマシンなどです。このセクションでは、アーティファクトを出力したランと、アーティファクトを入力として使用するランについても表示・リンクがされています。

### ファイルパネル

![](/images/app_ui/files_panel.png)

ファイルパネルには、アーティファクトに関連するファイルやフォルダが一覧表示されます。このファイルツリーをナビゲートして、W&B Webアプリ内で直接コンテンツを表示することができます。

アーティファクトに関連するTablesは、このコンテキストで特に豊かでインタラクティブです。アーティファクトとTablesの使用方法についてはこちらで詳しく解説しています。

![](/images/app_ui/files_panel_table.png)

### 履歴パネル

![](/images/app_ui/lineage_panel.png)

履歴パネルでは、プロジェクトに関連するすべてのアーティファクトとそれらを互いに接続するランを表示します。ランのタイプはブロックで表示され、アーティファクトは円で表示されます。矢印は、特定のタイプのランが特定のタイプのアーティファクトを消費または生成することを示しています。左側の列で選択された特定のアーティファクトのタイプが強調表示されます。
Explodeトグルをクリックすると、個々のアーティファクトのバージョンとそれらを接続する特定のrunが表示されます。

### アクション履歴監査タブ

![](/images/app_ui/action_history_audit_tab_1.png)

![](/images/app_ui/action_history_audit_tab_2.png)

アクション履歴監査タブには、Collectionのエイリアスアクションとメンバーシップの変更がすべて表示されるため、リソースの全体的な進化を監査できます。

### バージョンタブ

![](/images/app_ui/versions_tab.png)

バージョンタブには、アーティファクトのすべてのバージョンと、バージョンのログ時点のRun履歴の数値値ごとの各列が表示されます。これにより、パフォーマンスを比較し、関心のあるバージョンをすばやく特定できます。

## プロジェクトのデフォルト設定

ユーザー設定の `/settings` でプロジェクトのデフォルト設定を_手動_で変更することができます。

* **新しいプロジェクトを作成するデフォルトの場所**：デフォルトでは、あなた自身の個人エンティティに設定されています。ドロップダウンメニューをクリックすることで、個人エンティティと参加しているチーム間で切り替えることができます。
* **個人アカウントでのデフォルトプロジェクトプライバシー**：デフォルトでは、「プライベート」に設定されています。つまり、プロジェクトはプライベートであり、あなただけがアクセスできます。
* **個人アカウントでのコード保存を有効にする**：デフォルトではオフになっています。これをオンにすると、メインスクリプトやノートブックをW&Bに保存できます。

:::note
これらの設定は、
[`wandb.init`](../../../ref/python/init.md)
に引数を渡すことで指定することもできます。
:::

![](/images/app_ui/project_defaults.png)
## よくある質問



### プロジェクトを削除する方法は？



プロジェクトを削除するには、概要タブの右側にある3つの点をクリックします。



![](/images/app_ui/howto_delete_project.gif)



プロジェクトが空（つまり、runがない）場合は、右上のドロップダウンメニューをクリックして「プロジェクトを削除」を選択することで削除できます。



![](/images/app_ui/howto_delete_project_2.png)



### プロジェクトのプライバシー設定はどこにありますか？プロジェクトを公開または非公開にする方法は？



ページ上部のナビゲーションバーにあるロックをクリックして、プロジェクトのプライバシー設定を変更できます。プロジェクトに対して誰がrunを閲覧または送信できるかを編集することができます。この設定は、プロジェクト内のすべてのrunやレポートに適用されます。結果を限られた数の人と共有したい場合は、[プライベートチーム](../features/teams.md)を作成できます。



![](/images/app_ui/privacy_settings.png)



### ワークスペースをリセットする方法は？



プロジェクトページで以下のようなエラーが表示された場合、ワークスペースをリセットする方法は次のとおりです。`"objconv: "100000000000" overflows the maximum values of a signed 64 bits integer"`



URLの末尾に`?workspace=clear`を追加してエンターを押します。これで、ワークスペースがクリアされたバージョンのプロジェクトページに移動するはずです。