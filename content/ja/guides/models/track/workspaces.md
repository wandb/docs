---
title: 実験結果を表示する
description: インタラクティブな可視化で run データを探索するためのプレイグラウンド
menu:
  default:
    identifier: ja-guides-models-track-workspaces
    parent: experiments
weight: 4
---

W&B workspace は、チャートをカスタマイズして モデルの結果 を探索するための個人用サンドボックスです。W&B workspace は、*テーブル* と *パネル セクション* で構成されます:

* **Tables**: あなたの Project にログされたすべての Runs は、その Project のテーブルに一覧表示されます。Runs の表示／非表示や色の変更ができ、テーブルを展開すると各 run のノート、config、サマリー メトリクスを確認できます。
* **Panel sections**: 1 つ以上の [パネル]({{< relref path="/guides/models/app/features/panels/" lang="ja" >}}) を含むセクションです。新しいパネルを作成して整理し、Workspace のスナップショットを保存するために Reports へエクスポートできます。

{{< img src="/images/app_ui/workspace_table_and_panels.png" alt="Workspace のテーブルとパネル" >}}

## Workspace の種類
Workspace には 2 つの主要なカテゴリがあります: **Personal workspaces** と **Saved views**。

* **Personal workspaces:** モデルやデータ可視化をじっくり 分析 するためにカスタマイズ可能な Workspace です。Workspace の所有者だけが編集や保存ができます。チームメンバーは他人の Personal workspace を閲覧できますが、編集はできません。
* **Saved views:** Workspace の共同編集可能なスナップショットです。チームの誰でも閲覧・編集・保存ができます。Saved workspace views は Experiments、Runs などのレビューやディスカッションに活用できます。

次の画像は、Cécile-parker のチームメンバーが作成した複数の Personal workspaces を示しています。この Project には Saved views がありません:
{{< img src="/images/app_ui/Menu_No_views.jpg" alt="Saved views がありません" >}}

## Saved workspace views
チームに合わせた Workspace ビューでコラボレーションを強化しましょう。好みのチャートやデータのセットアップを整理するために Saved Views を作成します。

### 新しい Saved workspace view を作成する

1. Personal workspace または saved view に移動します。
2. Workspace を編集します。
3. Workspace 右上のメニュー（3 つの横ドット、いわゆる meatball メニュー）をクリックし、**Save as a new view** をクリックします。

新しい Saved views は Workspace のナビゲーション メニューに表示されます。

{{< img src="/images/app_ui/Menu_Views.jpg" alt="Saved views メニュー" >}}

### Saved workspace view を更新する
保存すると、前の状態を上書きします。未保存の変更は保持されません。W&B で Saved workspace view を更新するには:

1. Saved view に移動します。
2. Workspace 内のチャートやデータを希望どおりに変更します。
3. **Save** ボタンをクリックして変更を確定します。

{{% alert %}}
Workspace view の更新を保存すると、確認ダイアログが表示されます。今後この確認を表示したくない場合は、保存を確定する前に **Do not show this modal next time** を選択してください。
{{% /alert %}}

### Saved workspace view を削除する
不要になった Saved views を削除します。

1. 削除したい saved view に移動します。
2. 右上の (**...**) を選択します。
3. **Delete view** を選びます。
4. 確認して Workspace メニューからそのビューを削除します。

### Workspace view を共有する
Workspace の URL を直接共有して、カスタマイズした Workspace をチームと共有できます。その Workspace の Project へ アクセス できるすべての Users は、その Workspace の Saved Views を閲覧できます。

## Workspace templates
{{% alert %}}この機能には [Enterprise](https://wandb.ai/site/pricing/) ライセンスが必要です。{{% /alert %}}

_Workspace templates_ を使うと、[新しい Workspace のデフォルト設定]({{< relref path="#default-workspace-settings" lang="ja" >}}) ではなく、既存の Workspace と同じ設定で Workspace をすばやく作成できます。現在、Workspace template ではカスタムの [line plot 設定]({{< relref path="/guides/models/app/features/panels/line-plot/#all-line-plots-in-a-workspace" lang="ja" >}}) を定義できます。

### デフォルトの Workspace 設定
新しい Workspace は、既定で次の line plot 設定を使用します:

| 設定 | デフォルト |
|-------|----------
| X 軸               | Step |
| スムージング タイプ | Time weight EMA |
| スムージング 重み   | 0 |
| 最大 Runs          | 10 |
| チャートでのグルーピング | on |
| グループ集約        | Mean |

### Workspace template を設定する
1. どれかの Workspace を開くか、新しく作成します。
1. 好みに合わせて Workspace の [line plot 設定]({{< relref path="/guides/models/app/features/panels/line-plot/#all-line-plots-in-a-workspace" lang="ja" >}}) を設定します。
1. 設定を Workspace template に保存します:
    1. Workspace 上部、**Undo** と **Redo** の矢印アイコンの近くにあるアクション メニュー `...` をクリックします。
    1. **Save personal workspace template** をクリックします。
    1. Template の line plot 設定を確認し、**Save** をクリックします。

新しい Workspace では、デフォルトではなくこの設定が使われます。

### Workspace template を表示する
Workspace template の現在の設定を確認するには:
1. 任意のページで、右上のユーザーアイコンを選択し、ドロップダウンから **Settings** を選びます。
1. **Personal workspace template** セクションに移動します。Workspace template を使用している場合は、その設定が表示されます。使用していない場合、このセクションに詳細は表示されません。

### Workspace template を更新する
Workspace template を更新するには:

1. どれかの Workspace を開きます。
1. Workspace の設定を変更します。例えば、含める Runs の数を `11` に設定します。
1. Template への変更を保存するには、**Undo** と **Redo** の矢印アイコン近くのアクション メニュー `...` をクリックし、**Update personal workspace template** をクリックします。
1. 設定を確認して **Update** をクリックします。Template が更新され、それを使うすべての Workspace に再適用されます。

### Workspace template を削除する
Workspace template を削除してデフォルト設定に戻すには:

1. 任意のページで、右上のユーザーアイコンを選択し、ドロップダウンから **Settings** を選びます。
1. **Personal workspace template** セクションに移動します。Workspace template の設定が表示されます。
1. **Settings** の横にあるゴミ箱アイコンをクリックします。

{{% alert %}}
専用クラウド と Self-Managed 環境では、Workspace template の削除は v0.70 以降でサポートされています。古い Server バージョンでは、Workspace template を更新して [デフォルト設定]({{< relref path="#default-workspace-settings" lang="ja" >}}) を使用してください。
{{% /alert %}}

## プログラムで Workspace を作成する

[`wandb-workspaces`](https://github.com/wandb/wandb-workspaces/tree/main) は、[W&B](https://wandb.ai/) の workspaces と Reports をプログラムで扱うための Python ライブラリです。

[`wandb-workspaces`](https://github.com/wandb/wandb-workspaces/tree/main) を使ってプログラムで Workspace を定義します。[`wandb-workspaces`](https://github.com/wandb/wandb-workspaces/tree/main) は、[W&B](https://wandb.ai/) の workspaces と Reports をプログラムで扱うための Python ライブラリです。

Workspace のプロパティとして、次のようなことを定義できます:

* パネルのレイアウト、色、セクション順序を設定。
* 既定の X 軸、セクション順序、折りたたみ状態などの Workspace 設定を構成。
* セクション内にパネルを追加・カスタマイズして Workspace view を整理。
* URL を使って既存の Workspace を読み込み・変更。
* 既存の Workspace への変更を保存、または新しい view として保存。
* シンプルな式で Runs をフィルタ、グループ化、ソート。
* 色や可視性などの設定で run の見た目をカスタマイズ。
* ある Workspace から別の Workspace へ view をコピーして、統合や再利用に活用。

### Workspace API をインストールする

`wandb` に加えて、`wandb-workspaces` をインストールしてください:

```bash
pip install wandb wandb-workspaces
```

### プログラムで Workspace view を定義して保存する

```python
import wandb_workspaces.reports.v2 as ws

workspace = ws.Workspace(entity="your-entity", project="your-project", views=[...])
workspace.save()
```

### 既存の view を編集する
```python
existing_workspace = ws.Workspace.from_url("workspace-url")
existing_workspace.views[0] = ws.View(name="my-new-view", sections=[...])
existing_workspace.save()
```

### Workspace の `saved view` を別の Workspace にコピーする

```python
old_workspace = ws.Workspace.from_url("old-workspace-url")
old_workspace_view = old_workspace.views[0]
new_workspace = ws.Workspace(entity="new-entity", project="new-project", views=[old_workspace_view])

new_workspace.save()
```

包括的な Workspace API の例は、[`wandb-workspace examples`](https://github.com/wandb/wandb-workspaces/tree/main/examples/workspaces) を参照してください。エンドツーエンドのチュートリアルは、[Programmatic Workspaces]({{< relref path="/tutorials/workspaces.md" lang="ja" >}}) チュートリアルをご覧ください。