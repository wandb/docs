---
title: View experiments results
description: インタラクティブな 可視化 で run のデータを探索するためのプレイグラウンド
menu:
  default:
    identifier: ja-guides-models-track-workspaces
    parent: experiments
weight: 4
---

W&B workspace は、チャートをカスタマイズし、モデルの 結果 を 探索するための個人的なサンドボックスです。W&B workspace は、 *Tables* と *Panel sections* で構成されています。

* **Tables**: プロジェクト に 記録されたすべての run は、プロジェクト の テーブル にリストされます。run のオン/オフ、色の変更、テーブル の展開を行い、各 run のノート、config、およびサマリー メトリクス を確認します。
* **Panel sections**: 1つまたは複数の [パネル]({{< relref path="/guides/models/app/features/panels/" lang="ja" >}}) を含むセクション。新しいパネルを作成し、それらを整理して、レポート に エクスポート し、workspace の スナップショット を保存します。

{{< img src="/images/app_ui/workspace_table_and_panels.png" alt="" >}}

## Workspace の種類
主に、**Personal workspaces** と **Saved views** の2つの workspace カテゴリがあります。

* **Personal workspaces:** モデル と data visualization の詳細な 分析 のためのカスタマイズ可能な workspace。workspace の所有者のみが変更を編集および保存できます。チームメイト は personal workspace を表示できますが、他の ユーザー の personal workspace に変更を加えることはできません。
* **Saved views:** Saved views は、workspace の共同 スナップショット です。チームの メンバー は誰でも、保存された workspace ビューを表示、編集、および変更を保存できます。実験 、run などをレビューおよび議論するには、保存された workspace ビューを使用します。

次の図は、Cécile-parker のチームメイト によって作成された複数の personal workspaces を示しています。この プロジェクト には、保存された ビュー はありません。
{{< img src="/images/app_ui/Menu_No_views.jpg" alt="" >}}

## 保存された workspace ビュー
調整された workspace ビュー で チーム の コラボレーション を改善します。保存されたビューを作成して、チャート と データの好みの設定を整理します。

### 新しい保存された workspace ビュー を作成する

1. personal workspace または保存されたビュー に移動します。
2. workspace を編集します。
3. workspace の右上隅にあるミートボール メニュー（3つの水平ドット）をクリックします。[**Save as a new view**] をクリックします。

新しい保存されたビュー が workspace ナビゲーション メニュー に表示されます。

{{< img src="/images/app_ui/Menu_Views.jpg" alt="" >}}

### 保存された workspace ビュー を更新する
保存された変更は、保存されたビュー の以前の状態を上書きします。保存されていない変更は保持されません。W&B で保存された workspace ビュー を更新するには:

1. 保存されたビュー に移動します。
2. workspace 内でチャート と データ に必要な変更を加えます。
3. [**Save**] ボタンをクリックして、変更を確認します。

{{% alert %}}
workspace ビュー への更新を保存すると、確認ダイアログが表示されます。今後このプロンプトを表示しない場合は、保存を確認する前に [**Do not show this modal next time**] オプションを選択してください。
{{% /alert %}}

### 保存された workspace ビュー を削除する
不要になった保存されたビュー を削除します。

1. 削除する保存されたビュー に移動します。
2. ビュー の右上にある3つの水平線（**...**）を選択します。
3. [**Delete view**] を選択します。
4. 削除を確認して、workspace メニュー からビュー を削除します。

### Workspace ビュー を共有する
workspace の URL を直接共有して、カスタマイズした workspace を チーム と共有します。workspace プロジェクト への アクセス 権を持つすべての ユーザー が、その workspace の保存された ビュー を表示できます。

## プログラム で workspace を作成する

[`wandb-workspaces`](https://github.com/wandb/wandb-workspaces/tree/main) は、プログラム で [W&B](https://wandb.ai/) workspaces および レポート を操作するための Python ライブラリです。

[`wandb-workspaces`](https://github.com/wandb/wandb-workspaces/tree/main) を使用して、プログラム で workspace を定義します。[`wandb-workspaces`](https://github.com/wandb/wandb-workspaces/tree/main) は、プログラム で [W&B](https://wandb.ai/) workspaces および レポート を操作するための Python ライブラリです。

workspace のプロパティ は、次のように定義できます。

* パネル のレイアウト、色、およびセクション の順序を設定します。
* デフォルト の x軸、セクション の順序、および折りたたみ状態など、workspace の 設定 を構成します。
* セクション 内のパネル を追加およびカスタマイズして、workspace ビュー を整理します。
* URL を使用して、既存の workspaces をロードおよび変更します。
* 既存の workspaces への変更を保存するか、新しいビュー として保存します。
* 簡単な式を使用して、プログラム で run をフィルタリング、グループ化、およびソートします。
* 色や表示/非表示などの 設定 で run の外観をカスタマイズします。
* 統合 と再利用のために、ある workspace から別の workspace にビュー をコピーします。

### Workspace API のインストール

`wandb` に加えて、`wandb-workspaces` がインストールされていることを確認してください。

```bash
pip install wandb wandb-workspaces
```

### プログラム で workspace ビュー を定義して保存する

```python
import wandb_workspaces.reports.v2 as wr

workspace = ws.Workspace(entity="your-entity", project="your-project", views=[...])
workspace.save()
```

### 既存のビュー を編集する
```python
existing_workspace = ws.Workspace.from_url("workspace-url")
existing_workspace.views[0] = ws.View(name="my-new-view", sections=[...])
existing_workspace.save()
```

### workspace の `saved view` を別の workspace にコピーする

```python
old_workspace = ws.Workspace.from_url("old-workspace-url")
old_workspace_view = old_workspace.views[0]
new_workspace = ws.Workspace(entity="new-entity", project="new-project", views=[old_workspace_view])

new_workspace.save()
```

包括的な workspace API の例については、[`wandb-workspace examples`](https://github.com/wandb/wandb-workspaces/tree/main/examples/workspaces) を参照してください。エンドツーエンド のチュートリアル については、[Programmatic Workspaces]({{< relref path="/tutorials/workspaces.md" lang="ja" >}}) チュートリアル を参照してください。
