---
title: 実験の結果を閲覧する
description: 対話型の可視化で run データを自由に探索できるプレイグラウンド
menu:
  default:
    identifier: workspaces
    parent: experiments
weight: 4
---

W&B workspace は、チャートをカスタマイズしたり、モデルの結果を探索したりできるあなた専用のサンドボックスです。W&B workspace は *Tables* と *Panel sections* から構成されています。

* **Tables**: プロジェクトにログされたすべての Run がプロジェクトのテーブルに一覧表示されます。Run のオン・オフ切り替え、色の変更、ノート・config・summary metrics の展開表示も可能です。
* **Panel sections**: 1つ以上の [パネル]({{< relref "/guides/models/app/features/panels/" >}}) を含むセクションです。新しいパネルの作成や整理、さらにレポートへのエクスポートで Workspace のスナップショット保存も行えます。

{{< img src="/images/app_ui/workspace_table_and_panels.png" alt="Workspace table and panels" >}}

## Workspace の種類
Workspace には主に **Personal workspaces** と **Saved views** の2つのカテゴリがあります。

* **Personal workspaces:** モデルやデータ可視化を深く分析できる、カスタマイズ可能な workspace です。オーナーのみ編集や保存が可能です。他のチームメンバーは閲覧のみ可能ですが、他人の personal workspace を編集することはできません。
* **Saved views:** Workspace のスナップショットを共同で利用できる保存ビューです。チームの誰でも閲覧・編集・保存が可能です。Saved workspace views を使って、Experiments や Runs をレビューしたり、議論したりするのに便利です。

下記画像は Cécile-parker のチームメイトが作成した複数の personal workspace の例です。このプロジェクトには Saved view がありません。
{{< img src="/images/app_ui/Menu_No_views.jpg" alt="No saved views" >}}

## Saved workspace views
チームコラボレーション強化のために、最適化した workspace ビューを保存しましょう。Saved Views を作成することで、好みのチャートやデータのセットアップを整理できます。

### 新しい Saved workspace view の作成

1. Personal workspace または既存の Saved view に移動します。
2. Workspace を編集します。
3. Workspace 右上の三点リーダーメニュー（三本線）をクリックし、**Save as a new view** を選択します。

新しい Saved view は workspace のナビゲーションメニューに表示されます。

{{< img src="/images/app_ui/Menu_Views.jpg" alt="Saved views menu" >}}

### Saved workspace view の更新 
保存時に、前回の状態が上書きされます。未保存の変更は保持されません。更新手順は以下の通りです。

1. 編集したい Saved view に移動します。
2. Workspace 内でチャートやデータを変更します。
3. **Save** ボタンをクリックして変更を確定します。

{{% alert %}}
Workspace view を保存する際に確認ダイアログが表示されます。今後このダイアログを表示しない場合は、**Do not show this modal next time** にチェックを入れてから保存してください。
{{% /alert %}}

### Saved workspace view の削除
不要な Saved view を削除するには：

1. 削除対象の Saved view に移動します。
2. 右上の三点リーダー（**...**）をクリックします。
3. **Delete view** を選択します。
4. 削除を確認すると、そのビューが workspace メニューから削除されます。

### Workspace view の共有
Workspace の URL をチームに直接共有することで、カスタマイズした Workspace を共有できます。Workspace project へのアクセス権を持つすべてのユーザーが、その Workspace の Saved Views を閲覧可能です。

## Workspace テンプレート
{{% alert %}}この機能は [Enterprise](https://wandb.ai/site/pricing/) ライセンスが必要です。{{% /alert %}}

_ワークスペーステンプレート_ を使うと、既存の workspace の設定を使って素早く workspace を新規作成できます（[新規 workspace のデフォルト設定]({{< relref "#default-workspace-settings" >}}) を使わずに済みます）。現時点では、テンプレートで [line plot 設定]({{< relref "/guides/models/app/features/panels/line-plot/#all-line-plots-in-a-workspace" >}}) をカスタマイズできます。

### デフォルト workspace 設定
新しい workspace は、次の line plot のデフォルト設定が適用されます。

| 設定 | デフォルト値 |
|-------|----------
| X 軸               | Step |
| Smoothing type     | Time weight EMA |
| Smoothing weight   | 0 |
| Max runs           | 10 |
| Grouping in charts | on |
| Group aggregation  | Mean |

### Workspace テンプレートの設定
1. 既存の workspace を開くか、新規作成してください。
1. Workspace の [line plot 設定]({{< relref "/guides/models/app/features/panels/line-plot/#all-line-plots-in-a-workspace" >}}) を好みに合わせて設定します。
1. 設定を workspace テンプレートとして保存します:
    1. Workspace 上部、**Undo**・**Redo** アイコン近くにある `...` アクションメニューをクリックします。
    1. **Save personal workspace template** をクリックします。
    1. テンプレート用の line plot 設定を確認してから **Save** をクリックします。

以降、新規 Workspace にはこれらの設定がデフォルトとして適用されます。

### Workspace テンプレートの確認
Workspace テンプレートの現在の設定内容を確認するには：

1. 画面右上のユーザーアイコンから **Settings** を選択します。
1. **Personal workspace template** セクションに進みます。Templates 適用中なら内容が表示されます。テンプレート未使用の場合は情報がありません。

### Workspace テンプレートの更新
Workspace テンプレートを更新する手順：

1. Workspace を開く
1. 設定を変更（例: run 数の上限を `11` に設定）
1. 変更を保存するには、**Undo**・**Redo** アイコン近くの `...` アクションメニューをクリックし、**Update personal workspace template** をクリックします。
1. 設定を確認して **Update** をクリックします。テンプレートが更新され、これを利用する全 Workspace に再適用されます。

### Workspace テンプレートの削除
Workspace テンプレートを削除しデフォルト設定に戻したい場合：

1. 画面右上のユーザーアイコンから **Settings** を選択します。
1. **Personal workspace template** セクションに移動してください。テンプレート内容が表示されます。
1. **Settings** の隣にあるゴミ箱アイコンをクリックします。

{{% alert %}}
専用クラウドおよびセルフマネージド環境では v0.70 以上でテンプレート削除がサポートされています。古い Server バージョンの場合は、テンプレートを [デフォルト設定]({{< relref "#default-workspace-settings" >}}) に更新してください。
{{% /alert %}}

## ワークスペースのプログラムによる作成

[`wandb-workspaces`](https://github.com/wandb/wandb-workspaces/tree/main) は、[W&B](https://wandb.ai/) Workspace や Reports をプログラムで操作するための Python ライブラリです。

[`wandb-workspaces`](https://github.com/wandb/wandb-workspaces/tree/main) を使って、Workspace をプログラムで定義できます。このライブラリを活用すると、以下のような操作が可能です。

* パネルのレイアウト・色・セクション順序の設定
* Workspace 設定（デフォルト X 軸やセクション順、折りたたみ状態など）の指定
* セクションごとにパネルを追加・カスタマイズしてビューの構成
* ワークスペースの URL から既存 Workspace を取得・編集
* 既存 Workspace の変更保存、または新規ビューとして保存
* シンプルな式で Run をフィルタ・グループ化・ソート
* 色や可視性など Run の外観設定
* ビューを他の Workspace にコピーして統合や再利用

### Workspace API のインストール

`wandb` に加えて `wandb-workspaces` もインストールしてください:

```bash
pip install wandb wandb-workspaces
```

### プログラムで Workspace ビューを定義・保存する

```python
import wandb_workspaces.reports.v2 as wr

workspace = ws.Workspace(entity="your-entity", project="your-project", views=[...])
workspace.save()
```

### 既存ビューの編集
```python
existing_workspace = ws.Workspace.from_url("workspace-url")
existing_workspace.views[0] = ws.View(name="my-new-view", sections=[...])
existing_workspace.save()
```

### Workspace の `saved view` を他の Workspace にコピーする

```python
old_workspace = ws.Workspace.from_url("old-workspace-url")
old_workspace_view = old_workspace.views[0]
new_workspace = ws.Workspace(entity="new-entity", project="new-project", views=[old_workspace_view])

new_workspace.save()
```

ワークスペース API の詳細な例は [`wandb-workspace examples`](https://github.com/wandb/wandb-workspaces/tree/main/examples/workspaces) をご覧ください。より詳しいチュートリアルは [Programmatic Workspaces]({{< relref "/tutorials/workspaces.md" >}}) も参照してください。