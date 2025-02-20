---
title: View experiments results
description: インタラクティブな可視化を使って run データを探索するためのプレイグラウンド
menu:
  default:
    identifier: ja-guides-models-track-workspaces
    parent: experiments
weight: 4
---

W&B workspace は、チャートをカスタマイズし、モデル結果を探索するための個人的なサンドボックスです。W&B workspace は、*Tables* と *Panel sections* から構成されています：

* **Tables**: プロジェクトにログされているすべての runs は、プロジェクトのテーブルに一覧表示されます。runs のオン/オフを切り替え、色を変更し、拡大してメモ、config、各 run の summary metrics を確認できます。
* **Panel sections**: 一つ以上の[パネル]({{< relref path="/guides/models/app/features/panels/" lang="ja" >}})を含むセクションです。新しいパネルを作成し、それらを整理してレポートにエクスポートし、workspace のスナップショットを保存します。

{{< img src="/images/app_ui/workspace_table_and_panels.png" alt="" >}}

## Workspace の種類
主に2つの workspace カテゴリーがあります: **Personal workspaces** と **Saved views**。

* **Personal workspaces:** モデルとデータ可視化の詳細な分析のためのカスタマイズ可能なワークスペースです。workspace の所有者だけが編集と変更の保存ができます。チームメイトは個人の workspace を表示することはできますが、他の人の個人の workspace に変更を加えることはできません。
* **Saved views:** Saved views は workspace の共同スナップショットです。チームの誰もが閲覧、編集、変更の保存ができます。Saved Views を使用して、実験、runs、その他のレビューと議論を行います。

以下の画像は、Cécile-parker のチームメイトによって作成された複数の個人 workspaces を示しています。このプロジェクトには、保存されたビューがありません：
{{< img src="/images/app_ui/Menu_No_views.jpg" alt="" >}}

## Saved workspace views
カスタマイズされた workspace views でチームのコラボレーションを向上させます。Saved Views を作成して、チャートやデータのお好みの設定を整理します。

### 新しい保存済み workspace view を作成する

1. 個人 workspace または saved view に移動します。
2. workspace に変更を加えます。
3. workspace の右上隅のミートボールメニュー（三重線）をクリックします。**Save as a new view** をクリックします。

新しい保存済みビューは、workspace ナビゲーションメニューに表示されます。

{{< img src="/images/app_ui/Menu_Views.jpg" alt="" >}}

### 保存済み workspace view の更新
保存された変更は、保存されたビューの前の状態を上書きします。保存されてない変更は保持されません。W&B で保存済み workspace view を更新するには：

1. 保存済みビューに移動します。
2. workspace 内のチャートやデータに必要な変更を行います。
3. 変更を確定するために **Save** ボタンをクリックします。

{{% alert %}}
workspace view に更新を保存する際、確認ダイアログが表示されます。今後この確認プロンプトを表示しない場合は、確認する前に **Do not show this modal next time** オプションを選択してください。
{{% /alert %}}

### 保存済み workspace view の削除
不要になった保存済みビューを削除します。

1. 削除したい保存済みビューに移動します。
2. ビューの右上の三重線 (**...**) を選択します。
3. **Delete view** を選択します。
4. 削除を確認して、workspace メニューからビューを削除します。

### Workspace view を共有する
カスタマイズされた workspace を、workspace URL をチームと直接共有して、共有します。workspace プロジェクトにアクセスできるすべてのユーザーが、その workspace の保存済み Views を閲覧できます。

## プログラムによる workspace の作成

[`wandb-workspaces`](https://github.com/wandb/wandb-workspaces/tree/main) は、[W&B](https://wandb.ai/) workspaces と reports をプログラムで操作するための Python ライブラリです。

[`wandb-workspaces`](https://github.com/wandb/wandb-workspaces/tree/main) を使用してプログラムで workspace を定義します。 [`wandb-workspaces`](https://github.com/wandb/wandb-workspaces/tree/main) は、[W&B](https://wandb.ai/) workspaces と reports をプログラムで操作するための Python ライブラリです。

workspace のプロパティを定義するには、以下のようなことができます：

* パネルのレイアウト、色、およびセクションの順序を設定します。
* デフォルトの x 軸、セクション順序、折りたたみ状態などの workspace 設定を構成します。
* セクション内でパネルを追加およびカスタマイズして、workspace views を整理します。
* URL を使用して既存の workspace をロードおよび変更します。
* 既存の workspace に変更を保存するか、新しいビューとして保存します。
* 簡単な式を使用してプログラムで runs をフィルタリング、グループ化、および並べ替えます。
* 色や可視性などの設定で runs の外観をカスタマイズします。
* インテグレーションと再利用のために、一つの workspace から別の workspace へビューをコピーします。

### Workspace API のインストール

`wandb` に加えて、`wandb-workspaces` もインストールしてください：

```bash
pip install wandb wandb-workspaces
```

### プログラムで workspace view を定義して保存する

```python
import wandb_workspaces.reports.v2 as wr

workspace = ws.Workspace(entity="your-entity", project="your-project", views=[...])
workspace.save()
```

### 既存のビューを編集する
```python
existing_workspace = ws.Workspace.from_url("workspace-url")
existing_workspace.views[0] = ws.View(name="my-new-view", sections=[...])
existing_workspace.save()
```

### Workspace `saved view` を別の workspace にコピーする

```python
old_workspace = ws.Workspace.from_url("old-workspace-url")
old_workspace_view = old_workspace.views[0]
new_workspace = ws.Workspace(entity="new-entity", project="new-project", views=[old_workspace_view])

new_workspace.save()
```

包括的な workspace API の例については、[`wandb-workspace examples`](https://github.com/wandb/wandb-workspaces/tree/main/examples/workspaces) を参照してください。エンドツーエンドのチュートリアルの場合は、[Programmatic Workspaces]({{< relref path="/tutorials/workspaces.md" lang="ja" >}}) チュートリアルを参照してください。