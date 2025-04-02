---
title: View experiments results
description: インタラクティブな 可視化 で run のデータを探索できるプレイグラウンド
menu:
  default:
    identifier: ja-guides-models-track-workspaces
    parent: experiments
weight: 4
---

W&B workspace は、チャートをカスタマイズし、モデルの 結果 を探索するための個人的なサンドボックスです。W&B workspace は、 *Tables* と *Panel sections* で構成されています。

* **Tables**: あなたの project に ログ されたすべての run は、その project のテーブルにリストされます。run のオン/オフ、色の変更、テーブルの 展開 などを行い、各 run のノート、config、サマリー metrics を確認できます。
* **Panel sections**: 1つまたは複数の [panels]({{< relref path="/guides/models/app/features/panels/" lang="ja" >}}) を含むセクション。新しい パネル を作成し、それらを整理し、workspace の スナップショット を保存するために Reports にエクスポートします。

{{< img src="/images/app_ui/workspace_table_and_panels.png" alt="" >}}

## Workspace の種類
workspace には、主に **Personal workspaces** と **Saved views** の2つのカテゴリがあります。

* **Personal workspaces:** model と data visualization を詳細に 分析 するための、カスタマイズ可能な workspace です。workspace のオーナーのみが 編集 して変更を保存できます。チームメイトは personal workspace を表示できますが、他の ユーザー の personal workspace を変更することはできません。
* **Saved views:** Saved views は、workspace のコラボレーティブな スナップショット です。あなたの team の誰でも、保存された workspace view を表示、 編集 、および変更を保存できます。実験管理 、runs などをレビューおよび議論するために、保存された workspace view を使用します。

次の図は、Cécile-parker のチームメイトによって作成された複数の personal workspace を示しています。この project には、保存された view はありません。
{{< img src="/images/app_ui/Menu_No_views.jpg" alt="" >}}

## 保存された workspace view
カスタマイズされた workspace view で team のコラボレーションを改善します。保存された View を作成して、チャートと data の好みの 設定 を整理します。

### 新しい保存済み workspace view を作成する

1. personal workspace または保存された view に移動します。
2. workspace を 編集 します。
3. workspace の右上隅にあるミートボールメニュー（3つの水平ドット）をクリックします。**Save as a new view** をクリックします。

新しい保存された view は、workspace ナビゲーションメニューに表示されます。

{{< img src="/images/app_ui/Menu_Views.jpg" alt="" >}}

### 保存された workspace view を 更新 する
保存された変更は、保存された view の以前の状態を 上書き します。保存されていない変更は保持されません。W&B で保存された workspace view を 更新 するには：

1. 保存された view に移動します。
2. workspace 内で、チャートと data に必要な変更を加えます。
3. **Save** ボタンをクリックして、変更を確定します。

{{% alert %}}
workspace view への 更新 を保存すると、確認ダイアログが表示されます。今後このプロンプトを表示しない場合は、保存を確定する前に **Do not show this modal next time** オプションを選択してください。
{{% /alert %}}

### 保存された workspace view を 削除 する
不要になった保存された view を削除します。

1. 削除する保存された view に移動します。
2. view の右上にある3つの水平線（**...**）を選択します。
3. **Delete view** を選択します。
4. 削除を確定して、workspace メニューから view を削除します。

### workspace view を共有する
workspace の URL を直接共有して、カスタマイズされた workspace を team と共有します。workspace project への アクセス 権を持つすべての ユーザー は、その workspace の保存された View を見ることができます。

## プログラムで workspace を作成する

[`wandb-workspaces`](https://github.com/wandb/wandb-workspaces/tree/main) は、[W&B](https://wandb.ai/) workspace と Reports をプログラムで 操作 するための Python library です。

[`wandb-workspaces`](https://github.com/wandb/wandb-workspaces/tree/main) を使用して、プログラムで workspace を定義します。[`wandb-workspaces`](https://github.com/wandb/wandb-workspaces/tree/main) は、[W&B](https://wandb.ai/) workspace と Reports をプログラムで 操作 するための Python library です。

workspace の プロパティ は、次のように定義できます。

* panel の レイアウト、色、およびセクションの 順序 を 設定 します。
* デフォルトのx軸、セクションの 順序 、および コラプス 状態など、workspace の 設定 を構成します。
* セクション内に パネル を追加およびカスタマイズして、workspace view を整理します。
* URL を使用して、既存の workspace を ロード および 変更 します。
* 既存の workspace への変更を保存するか、新しい view として保存します。
* 簡単な 式 を使用して、runs をプログラムで フィルタリング、グループ化、およびソートします。
* 色や 可視性 などの 設定 で、run の外観をカスタマイズします。
* 統合 と再利用のために、ある workspace から別の workspace に view をコピーします。

### Workspace API をインストールする

`wandb` に加えて、`wandb-workspaces` をインストールしてください。

```bash
pip install wandb wandb-workspaces
```

### プログラムで workspace view を定義して保存する

```python
import wandb_workspaces.reports.v2 as wr

workspace = ws.Workspace(entity="your-entity", project="your-project", views=[...])
workspace.save()
```

### 既存の view を 編集 する
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

workspace API の包括的な例については、[`wandb-workspace examples`](https://github.com/wandb/wandb-workspaces/tree/main/examples/workspaces) を参照してください。エンドツーエンドのチュートリアルについては、[Programmatic Workspaces]({{< relref path="/tutorials/workspaces.md" lang="ja" >}}) チュートリアルを参照してください。
