---
title: 実験結果を表示
description: インタラクティブな可視化で run データを探索できるプレイグラウンド
menu:
  default:
    identifier: ja-guides-models-track-workspaces
    parent: experiments
weight: 4
---

W&B Workspace は、チャートのカスタマイズやモデル結果の探索を行うための個人用サンドボックスです。W&B Workspace は *Tables* と *Panel セクション* で構成されています。

* **Tables**: プロジェクトにログされたすべての Run がプロジェクトのテーブルに一覧表示されます。Run のオン/オフ切り替え、配色変更、テーブルの拡大などを行い、各 Run のノート、config、サマリーメトリクスを確認できます。
* **Panel セクション**: 1つ以上の [パネル]({{< relref path="/guides/models/app/features/panels/" lang="ja" >}}) を含むセクションです。新しいパネルの作成や整理、Workspace のスナップショットを保存するためにレポートへのエクスポートができます。

{{< img src="/images/app_ui/workspace_table_and_panels.png" alt="Workspace table and panels" >}}

## Workspace の種類
Workspace には大きく分けて **Personal workspaces** と **Saved views** の 2 種類があります。

* **Personal workspaces:** モデルやデータ可視化の詳細な分析に柔軟に使えるカスタマイズ可能な Workspace です。Workspace の所有者だけが編集・保存でき、チームメイトは表示のみ可能。他の人の Personal Workspace を編集することはできません。
* **Saved views:** Workspace のコラボレーティブなスナップショットです。チームメンバーなら誰でも参照・編集・保存が可能。Saved views は、Experiments や Runs などのレビュー・ディスカッションに使えます。

次の画像は Cécile-parker のチームメイトによって作成された複数の Personal Workspace を示しています。このプロジェクトには Saved views がありません:
{{< img src="/images/app_ui/Menu_No_views.jpg" alt="No saved views" >}}

## Saved workspace views
チームコラボレーションを強化するために、ニーズに合わせた Workspace Views を作成しましょう。Saved Views を作成して、好みのチャートやデータのセットアップを整理できます。

### 新しい saved workspace view の作成

1. Personal Workspace または Saved view へ移動します。
2. Workspace を編集します。
3. Workspace 右上の三点リーダー（縦三点）メニューをクリックし、**Save as a new view** を選択。

新しく作成された Saved view は、Workspace のナビゲーションメニューに表示されます。

{{< img src="/images/app_ui/Menu_Views.jpg" alt="Saved views menu" >}}

### Saved workspace view の更新
保存された変更は、その Saved view の以前の状態を上書きします。未保存の変更は保持されません。W&B で Saved workspace view を更新するには:

1. Saved view に移動します。
2. Workspace 内でチャートやデータを必要な形に編集します。
3. **Save** ボタンをクリックして変更を確定します。

{{% alert %}}
Workspace view の更新内容を保存するとき、確認ダイアログが表示されます。今後このダイアログを表示させたくない場合は、**Do not show this modal next time** を選択してから保存してください。
{{% /alert %}}

### Saved workspace view の削除
不要な Saved view を削除します。

1. 削除したい Saved view に移動します。
2. 画面右上の三点マーク（**...**）をクリックします。
3. **Delete view** を選択。
4. 削除を確定すると、Workspace メニューから削除されます。

### Workspace view の共有
Workspace の URL を直接共有することで、カスタマイズした Workspace をチームで共有できます。Workspace プロジェクトに access があるユーザーは、全員その Workspace の Saved Views を閲覧できます。

## Workspace テンプレート
{{% alert %}}この機能は [Enterprise](https://wandb.ai/site/pricing/) ライセンスが必要です。{{% /alert %}}

_ワークスペーステンプレート_ を使うと、[新しい Workspace のデフォルト設定]({{< relref path="#default-workspace-settings" lang="ja" >}}) ではなく、既存 Workspace と同じ設定で素早く Workspace を作成できます。現在、Workspace テンプレートではカスタムの [折れ線グラフの設定]({{< relref path="/guides/models/app/features/panels/line-plot/#all-line-plots-in-a-workspace" lang="ja" >}}) を定義できます。

### デフォルトの Workspace 設定
新しい Workspace は、折れ線グラフに関して以下のデフォルト設定が適用されます:

| Setting | Default |
|--------|---------
| X axis             | Step |
| Smoothing type     | Time weight EMA |
| Smoothing weight   | 0 |
| Max runs           | 10 |
| Grouping in charts | on |
| Group aggregation  | Mean |

### Workspace テンプレートの設定方法
1. 任意の Workspace を開くか、新規作成します。
1. Workspace の [折れ線グラフ設定]({{< relref path="/guides/models/app/features/panels/line-plot/#all-line-plots-in-a-workspace" lang="ja" >}}) を好みに合わせて設定します。
1. 設定を Workspace テンプレートとして保存します:
    1. Workspace 上部、**Undo** と **Redo** 矢印アイコン付近のアクションメニュー `...` をクリック。
    1. **Save personal workspace template** をクリック。
    1. テンプレートの折れ線グラフ設定を確認し、**Save** をクリック。

次回以降、新しい Workspace ではデフォルト設定の代わりにこれらの設定が使用されます。

### Workspace テンプレートの確認
Workspace テンプレートの現在の設定内容を確認するには:
1. どのページからでも、右上のユーザーアイコンをクリックし、ドロップダウンから **Settings** を選びます。
1. **Personal workspace template** セクションへ移動します。Workspace テンプレートを利用している場合、その設定内容が表示されます。未設定なら何も表示されません。

### Workspace テンプレートの更新
Workspace テンプレートを変更するには:

1. 任意の Workspace を開きます。
1. Workspace の設定を編集します（例: 表示する Run 数を `11` に変更）。
1. テンプレート保存のために、**Undo** と **Redo** アイコン付近のアクションメニュー `...` をクリックし、**Update personal workspace template** を選択します。
1. 設定を確認し **Update** をクリックすると、テンプレートが更新され、そのテンプレートを利用するすべての Workspace に再適用されます。

### Workspace テンプレートの削除
Workspace テンプレートを削除して元のデフォルト設定に戻すには:

1. どのページからでも、右上のユーザーアイコンをクリックし、ドロップダウンから **Settings** を選びます。
1. **Personal workspace template** セクションへ移動します。テンプレート内容が表示されます。
1. **Settings** 右側のゴミ箱アイコンをクリック。

{{% alert %}}
Dedicated Cloud および Self-Managed の場合、v0.70 以上では Workspace テンプレートの削除がサポートされています。より古い Server バージョンでは、[デフォルト設定]({{< relref path="#default-workspace-settings" lang="ja" >}}) にテンプレートを更新してください。
{{% /alert %}}

## プログラムから Workspace を作成する

[`wandb-workspaces`](https://github.com/wandb/wandb-workspaces/tree/main) は、[W&B](https://wandb.ai/) の Workspace や Reports をプログラムから操作するための Python ライブラリです。

[`wandb-workspaces`](https://github.com/wandb/wandb-workspaces/tree/main) を使って Workspace をプログラムから定義できます。[`wandb-workspaces`](https://github.com/wandb/wandb-workspaces/tree/main) は、W&B の Workspace や Reports をコードで操作するための Python ライブラリです。

Workspace の設定例は次の通りです:

* パネルレイアウト、色、セクション順の設定
* デフォルトの X 軸、セクション順、折りたたみ状態など Workspace 設定のカスタマイズ
* セクション内へのパネル追加・編集による Workspace View の構成
* 既存 Workspace の URL からのロードや変更
* 既存 Workspace への変更保存、または別名での新規保存
* シンプルな式を使ったプログラムによる Run のフィルタ・グループ化・ソート
* Run の色や表示状態など Appearance のカスタマイズ
* 別の Workspace への View のコピーや再利用によるインテグレーション

### Workspace API のインストール

`wandb` に加えて、`wandb-workspaces` のインストールも必要です:

```bash
pip install wandb wandb-workspaces
```

### Workspace view のコード定義と保存


```python
import wandb_workspaces.reports.v2 as wr

workspace = ws.Workspace(entity="your-entity", project="your-project", views=[...])
workspace.save()
```

### 既存の view の編集
```python
existing_workspace = ws.Workspace.from_url("workspace-url")
existing_workspace.views[0] = ws.View(name="my-new-view", sections=[...])
existing_workspace.save()
```

### Workspace の `saved view` を別 Workspace へコピー

```python
old_workspace = ws.Workspace.from_url("old-workspace-url")
old_workspace_view = old_workspace.views[0]
new_workspace = ws.Workspace(entity="new-entity", project="new-project", views=[old_workspace_view])

new_workspace.save()
```

より詳しい Workspace API の使用例は [`wandb-workspace examples`](https://github.com/wandb/wandb-workspaces/tree/main/examples/workspaces) をご覧ください。エンドツーエンドのチュートリアルについては [Programmatic Workspaces]({{< relref path="/tutorials/workspaces.md" lang="ja" >}}) を参照してください。