---
title: 実験管理の結果を見る
description: ランデータを対話的な可視化で探求するためのプレイグラウンド
menu:
  default:
    identifier: ja-guides-models-track-workspaces
    parent: experiments
weight: 4
---

W&B ワークスペースは、チャートをカスタマイズしモデル結果を探索するための個人のサンドボックスです。W&B ワークスペースは *テーブル* と *パネルセクション* で構成されています:

* **Tables**: プロジェクトに記録されたすべてのRunがプロジェクトのテーブルに一覧表示されます。Runをオンオフしたり、色を変更したり、テーブルを拡張して各Runのノート、設定、およびサマリーメトリクスを表示することができます。
* **Panel sections**: 1つ以上の [パネル]({{< relref path="/guides/models/app/features/panels/" lang="ja" >}}) を含むセクションです。新しいパネルを作成し、整理し、ワークスペースのスナップショットを保存するためにReportsにエクスポートすることができます。

{{< img src="/images/app_ui/workspace_table_and_panels.png" alt="" >}}

## Workspaceの種類
主に2つのWorkspaceカテゴリがあります: **個人用ワークスペース** と **保存されたビュー** です。

* **個人用ワークスペース:** モデルとデータの可視化の詳細な分析のためのカスタマイズ可能なワークスペースです。ワークスペースの所有者のみが編集し、変更を保存できます。チームメイトは個人用ワークスペースを閲覧できますが、他の人の個人用ワークスペースには変更を加えることはできません。
* **保存されたビュー:** 保存されたビューは、ワークスペースの協力的なスナップショットです。チームの誰もが保存されたワークスペースビューを閲覧、編集、保存することができます。保存されたワークスペースビューを使用して、実験、Runなどをレビューおよび議論します。

次の画像は、Cécile-parkerのチームメイトによって作成された複数の個人用ワークスペースを示しています。このプロジェクトには保存されたビューがありません:
{{< img src="/images/app_ui/Menu_No_views.jpg" alt="" >}}

## 保存されたワークスペースビュー
チームのコラボレーションを向上させるために、カスタマイズされたワークスペースビューを作成します。保存されたビューを作成して、チャートとデータの好みのセットアップを整理します。

### 新しい保存されたワークスペースビューを作成する

1. 個人用ワークスペースまたは保存されたビューに移動します。
2. ワークスペースを編集します。
3. ワークスペースの右上隅にある三つポチメニュー（三本の横線）をクリックし、**新しいビューとして保存** をクリックします。

新しい保存されたビューはワークスペースナビゲーションメニューに表示されます。

{{< img src="/images/app_ui/Menu_Views.jpg" alt="" >}}

### 保存されたワークスペースビューを更新する
保存された変更は、保存されたビューの以前の状態を上書きします。保存されていない変更は保持されません。W&Bで保存されたワークスペースビューを更新するには:

1. 保存されたビューに移動します。
2. ワークスペース内のチャートとデータに必要な変更を加えます。
3. **保存** ボタンをクリックして変更を確認します。

{{% alert %}}
ワークスペースビューの更新を保存すると、確認ダイアログが表示されます。次回このプロンプトを表示しないようにするには、保存を確認する前に **今後このモーダルを表示しない** オプションを選択します。
{{% /alert %}}

### 保存されたワークスペースビューを削除する
不要になった保存されたビューを削除します。

1. 削除したい保存済みビューに移動します。
2. ビューの右上隅にある三本の横線（**...**）を選択します。
3. **ビューを削除** を選択します。
4. 削除を確認してワークスペースメニューからビューを削除します。

### ワークスペースビューを共有する
ワークスペースのURLを直接共有することで、カスタマイズしたワークスペースをチームと共有します。ワークスペースプロジェクトにアクセスできるすべてのユーザーが、そのワークスペースの保存されたビューを見ることができます。

## ワークスペースをプログラムで作成する

[`wandb-workspaces`](https://github.com/wandb/wandb-workspaces/tree/main) は、[W&B](https://wandb.ai/) のワークスペースとレポートをプログラムで操作するためのPythonライブラリです。

[`wandb-workspaces`](https://github.com/wandb/wandb-workspaces/tree/main) を使用してワークスペースをプログラムで定義します。[`wandb-workspaces`](https://github.com/wandb/wandb-workspaces/tree/main) は、[W&B](https://wandb.ai/) のワークスペースとレポートをプログラムで操作するためのPythonライブラリです。

ワークスペースのプロパティを定義できます、例:

* パネルのレイアウト、色、およびセクションの順序を設定します。
* ワークスペース設定としてデフォルトのX軸、セクションの順序、および折りたたみ状態を設定します。
* セクション内にパネルを追加してカスタマイズし、ワークスペースビューを整理します。
* URLを使用して既存のワークスペースを読み込み、変更します。
* 既存のワークスペースに変更を保存するか、新しいビューとして保存します。
* シンプルな式を使用してRunをプログラムでフィルタリング、グループ化、ソートします。
* 色や表示可否などの設定でRunの外観をカスタマイズします。
* ワークスペース間でビューをコピーして、インテグレーションと再利用を行います。

### Workspace API をインストール

`wandb`に加えて、`wandb-workspaces`をインストールすることを確認してください:

```bash
pip install wandb wandb-workspaces
```

### プログラムでワークスペースビューを定義し保存する

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

### ワークスペース `保存されたビュー` を別のワークスペースにコピーする

```python
old_workspace = ws.Workspace.from_url("old-workspace-url")
old_workspace_view = old_workspace.views[0]
new_workspace = ws.Workspace(entity="new-entity", project="new-project", views=[old_workspace_view])

new_workspace.save()
```

包括的なワークスペースAPIの例として、[`wandb-workspace examples`](https://github.com/wandb/wandb-workspaces/tree/main/examples/workspaces) を参照してください。エンドツーエンドのチュートリアルについては、[Programmatic Workspaces]({{< relref path="/tutorials/workspaces.md" lang="ja" >}}) チュートリアルを参照してください。