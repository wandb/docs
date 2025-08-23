---
title: ワークスペース、セクション、およびパネルの設定を管理する
menu:
  default:
    identifier: ja-guides-models-app-features-cascade-settings
    parent: w-b-app-ui-reference
url: guides/app/features/cascade-settings
---

ワークスペースページ内には、ワークスペース、セクション、パネルの3つの異なる設定レベルがあります。[Workspace 設定]({{< relref path="#workspace-settings" lang="ja" >}}) はワークスペース全体に適用されます。[Section 設定]({{< relref path="#section-settings" lang="ja" >}}) はセクション内のすべてのパネルに適用されます。[Panel 設定]({{< relref path="#panel-settings" lang="ja" >}}) は個々のパネルに適用されます。

## Workspace 設定

Workspace 設定は、すべてのセクションおよびそこに含まれるすべてのパネルに適用されます。編集できる Workspace 設定は [Workspace レイアウト]({{< relref path="#workspace-layout-options" lang="ja" >}}) と [Line plots]({{< relref path="#line-plots-options" lang="ja" >}}) の2種類です。**Workspace レイアウト**はワークスペースの構造を決定し、**Line plots** 設定はワークスペース内でのラインプロットのデフォルト設定を制御します。

ワークスペースの全体構成に影響する設定を編集するには：

1. 利用中のプロジェクト ワークスペースに移動します。
2. **New report** ボタンの隣にあるギアアイコンをクリックして Workspace 設定を開きます。
3. **Workspace layout** でワークスペースのレイアウトを変更、または **Line plots** で Line plot のデフォルト設定を編集します。
{{< img src="/images/app_ui/workspace_settings.png" alt="Workspace 設定のギアアイコン" >}}

{{% alert %}}
ワークスペースをカスタマイズした後、_workspace templates_ を使えば同じ設定で新しいワークスペースを素早く作成できます。詳しくは [Workspace templates]({{< relref path="/guides/models/track/workspaces.md#workspace-templates" lang="ja" >}}) をご覧ください。
{{% /alert %}}

### Workspace レイアウト オプション

ワークスペースのレイアウトを設定することで、ワークスペース全体の構造を定義できます。ここにはセクションの分割方法やパネルの並べ方が含まれます。

{{< img src="/images/app_ui/workspace_layout_settings.png" alt="Workspace レイアウトオプション" >}}

Workspace レイアウトオプションのページでは、パネルが自動生成か手動生成かが表示されます。パネル生成モードを調整したい場合は [Panels]({{< relref path="panels/" lang="ja" >}}) をご参照ください。

各 Workspace レイアウトオプションの説明は以下の通りです。

| Workspace 設定 | 説明 |
| ----- | ----- |
| **検索時に空のセクションを非表示** | パネルが含まれていないセクションをパネル検索時に非表示にします。|
| **パネルをアルファベット順にソート** | ワークスペース内のパネルをアルファベット順に並べます。|
| **セクションの整理** | 既存のすべてのセクションとパネルを削除し、新しいセクション名で再構成します。新しいセクションは先頭または末尾のプレフィックスでグループ化されます。|

{{% alert %}}
W&B では、セクションのグループ化は末尾のプレフィックスよりも先頭のプレフィックスで行うことを推奨しています。先頭プレフィックスでのグループ化はより少ないセクション数となり、パフォーマンスの向上につながります。
{{% /alert %}}

### Line plots オプション

**Line plots** の workspace 設定を変更することで、そのワークスペース内のラインプロットのグローバルデフォルトやカスタムルールを設定できます。

{{< img src="/images/app_ui/workspace_settings_line_plots.png" alt="Line plot 設定" >}}

**Line plots** 設定内で主に編集できるのは **Data** と **Display preferences** の2つです。**Data** タブには次のような設定があります：

| Line plot 設定 | 説明 |
| ----- | ----- |
| **X 軸** | ラインプロットの x 軸のスケール。デフォルトは **Step** になっています。x 軸オプション一覧は次の表を参照してください。|
| **範囲** | x軸に表示する最小値・最大値の設定。|
| **スムージング** | ラインプロットのスムージングの調整。[Smooth line plots]({{< relref path="/guides/models/app/features/panels/line-plot/smoothing.md" lang="ja" >}}) を参照してください。|
| **外れ値** | デフォルトのプロットの最小・最大スケールから外れ値を除外します。|
| **ポイント集約メソッド** | データ可視化の精度とパフォーマンスを向上させます。[Point aggregation]({{< relref path="/guides/models/app/features/panels/line-plot/sampling.md" lang="ja" >}}) をご参照ください。|
| **最大表示 run またはグループ数** | ラインプロット上に表示する run やグループの最大数を制限します。|

**Step** 以外にも、x 軸には以下のオプションがあります：

| X 軸オプション | 説明 |
| ------------- | ----------- |
| **Relative Time (Wall)**| プロセス開始からの経過時刻（タイムスタンプ）。例：run を開始し、翌日再開して何かをログすると24時間と記録されます。|
| **Relative Time (Process)** | 実行中プロセス内の経過時刻。例：run を始めて10秒間続ける、翌日再開しても記録は10秒のままです。|
| **Wall Time** | グラフ上で最初の run 開始からの経過分数。|
| **Step** | `wandb.Run.log()` を呼び出すたびにインクリメントされます。|



{{% alert %}}
個別のラインプロットの編集方法については、Line plots ガイドの [ラインパネル設定の編集]({{< relref path="/guides/models/app/features/panels/line-plot/#edit-line-panel-settings" lang="ja" >}}) をご参照ください。
{{% /alert %}}


**Display preferences** タブ内では、下記の設定を切り替えることができます：

| 表示設定 | 説明 |
| ----- | ----- |
| **全パネルの凡例を削除** | パネルの凡例を非表示にします |
| **ツールチップに run 名をカラー表示** | ツールチップ内で run 名を色付きテキストで表示します |
| **強調表示した run のみツールチップに表示** | チャートのツールチップにはハイライトした run のみ表示 |
| **ツールチップに表示する run 数** | ツールチップ内で表示する run の数を設定 |
| **メインチャートのツールチップで run 名をフル表示**| チャートツールチップに run のフルネームを表示 |




## Section 設定

Section 設定は、そのセクション内のすべてのパネルに適用されます。ワークスペースのセクション内では、パネルの並べ替えや入れ替え、セクション名の変更が可能です。

Section 設定は、セクション右上の3つのドット（**...**）を選択して編集します。

{{< img src="/images/app_ui/section_settings.png" alt="Section 設定メニュー" >}}

ドロップダウンメニューからセクション全体に適用される以下の設定が編集できます：

| Section 設定 | 説明 |
| ----- | ----- |
| **セクション名の変更** | セクションの名前を変更します |
| **パネルをA～Z順でソート** | セクション内のパネルをアルファベット順に並べ替え |
| **パネルの入れ替え** | セクション内のパネルをドラッグ＆ドロップで順序変更 |

下記のアニメーションは、セクション内のパネル入れ替えの例です：

{{< img src="/images/app_ui/rearrange_panels.gif" alt="パネルの入れ替え" >}}

{{% alert %}}
上記設定のほかにも、**Add section below**、**Add section above**、**Delete section**、**Add section to report** など、ワークスペースでのセクション表示のカスタマイズも可能です。
{{% /alert %}}

## Panel 設定

個々のパネルごとに、複数のラインの比較・カスタム軸の計算・ラベル名変更など様々な設定が行えます。パネルの設定を編集するには：

1. 編集したいパネルにマウスカーソルを合わせます。
2. 表示される鉛筆アイコンをクリックします。
{{< img src="/images/app_ui/panel_settings.png" alt="パネルの編集アイコン" >}}
3. 表示されたモーダル内で、パネルのデータや表示設定などを編集できます。
{{< img src="/images/app_ui/panel_settings_modal.png" alt="パネル設定モーダル" >}}

パネルで利用可能な設定の詳細は [ラインパネル設定の編集]({{< relref path="/guides/models/app/features/panels/line-plot/#edit-line-panel-settings" lang="ja" >}}) をご覧ください。