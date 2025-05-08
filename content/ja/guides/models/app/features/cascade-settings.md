---
title: ワークスペース、セクション、パネル設定を管理する
menu:
  default:
    identifier: ja-guides-models-app-features-cascade-settings
    parent: w-b-app-ui-reference
url: /ja/guides/app/features/cascade-settings
---

Within a given workspace page there are three different setting levels: workspaces, sections, and panels. [ワークスペース設定]({{< relref path="#workspace-settings" lang="ja" >}}) は、ワークスペース全体に適用されます。[セクション設定]({{< relref path="#section-settings" lang="ja" >}}) は、セクション内のすべてのパネルに適用されます。[パネル設定]({{< relref path="#panel-settings" lang="ja" >}}) は、個々のパネルに適用されます。

## ワークスペース設定

ワークスペース設定は、すべてのセクションとそれらのセクション内のすべてのパネルに適用されます。編集できるワークスペース設定は次の2種類です: [**Workspace layout**]({{< relref path="#workspace-layout-options" lang="ja" >}}) と [**Line plots**]({{< relref path="#line-plots-options" lang="ja" >}})。**Workspace layouts** はワークスペースの構造を決定し、**Line plots** 設定はワークスペース内のラインプロットのデフォルト設定を制御します。

このワークスペースの全体的な構造に適用される設定を編集するには:

1. プロジェクトワークスペースに移動します。
2. **New report** ボタンの横にある歯車のアイコンをクリックして、ワークスペース設定を表示します。
3. ワークスペースのレイアウトを変更するには **Workspace layout** を選択するか、ワークスペース内のラインプロットのデフォルト設定を設定するには **Line plots** を選択します。
{{< img src="/images/app_ui/workspace_settings.png" alt="" >}}

### ワークスペースレイアウトオプション

ワークスペースのレイアウトを設定して、ワークスペースの全体的な構造を定義します。これには、セクションのロジックとパネルの配置が含まれます。

{{< img src="/images/app_ui/workspace_layout_settings.png" alt="" >}}

ワークスペースレイアウトオプションページでは、ワークスペースがパネルを自動か手動で生成するかが表示されます。ワークスペースのパネル生成モードを調整するには、[Panels]({{< relref path="panels/" lang="ja" >}}) を参照してください。

この表は、各ワークスペースのレイアウトオプションについて説明しています。

| ワークスペース設定 | 説明 |
| ----- | ----- |
| **検索中に空のセクションを非表示** | パネルを検索するときにパネルを含まないセクションを非表示にします。 |
| **パネルをアルファベット順に並べ替え** | ワークスペース内のパネルをアルファベット順に並べ替えます。 |
| **セクションの組織化** | 既存のすべてのセクションとパネルを削除し、新しいセクション名で再配置します。また、新しく配置されたセクションを最初または最後のプレフィックスでグループ化します。 |

{{% alert %}}
W&B は、最後のプレフィックスでグループ化するのではなく、最初のプレフィックスでセクションをグループ化することをお勧めします。最初のプレフィックスでグループ化することで、セクション数が少なくなり、パフォーマンスが向上します。
{{% /alert %}}

### ラインプロットオプション

ワークスペースの**Line plots**設定を変更して、ラインプロットのグローバルデフォルトとカスタムルールを設定します。

{{< img src="/images/app_ui/workspace_settings_line_plots.png" alt="" >}}

**Line plots** 設定内で編集できる主要な設定は2つあります: **Data** と **Display preferences**。**Data** タブには次の設定が含まれています:

| ラインプロット設定 | 説明 |
| ----- | ----- |
| **X軸** | ラインプロットのx軸のスケール。x軸はデフォルトで **Step** に設定されています。x軸オプションのリストは次の表を参照してください。 |
| **範囲** | x軸に表示する最小値と最大値の設定。 |
| **平滑化** | ラインプロットの平滑化を変更します。平滑化の詳細については、[Smooth line plots]({{< relref path="/guides/models/app/features/panels/line-plot/smoothing.md" lang="ja" >}}) を参照してください。 |
| **異常値** | 異常値を除外するためにプロットの最小スケールと最大スケールを再設定します。 |
| **ポイント集計方法** | Data Visualization の精度とパフォーマンスを向上させます。詳細については、[Point aggregation]({{< relref path="/guides/models/app/features/panels/line-plot/sampling.md" lang="ja" >}}) を参照してください。 |
| **最大の runs またはグループの数** | ラインプロットに表示する最大の runs またはグループ数を制限します。 |

**Step** 以外にも、x軸には他のオプションがあります:

| X軸オプション | 説明 |
| ------------- | ----------- |
| **相対時間 (Wall)** | プロセスが開始してからのタイムスタンプ。例えば、run を開始して次の日にその run を再開したとします。その場合、記録されたポイントは24時間後です。|
| **相対時間 (Process)** | 実行中のプロセス内のタイムスタンプ。例えば、run を開始して10秒間続け、その後次の日に再開したとします。その場合、記録されたポイントは10秒です。|
| **ウォールタイム** | グラフで最初の run が開始してから経過した時間（分）。|
| **Step** | `wandb.log()` を呼び出すたびに増加します。|

{{% alert %}}
個別のラインプロットを編集する方法については、ラインプロット内の[Edit line panel settings]({{< relref path="/guides/models/app/features/panels/line-plot/#edit-line-panel-settings" lang="ja" >}})を参照してください。
{{% /alert %}}

**Display preferences** タブ内で、以下の設定を切り替えることができます:

| ディスプレイ設定 | 説明 |
| ----- | ----- |
| **すべてのパネルから凡例を削除** | パネルの凡例を削除します |
| **ツールチップ内でカラード run 名を表示** | ツールチップ内で run をカラードテキストとして表示します |
| **コンパニオンチャートツールチップで、ハイライトされた run のみを表示** | チャートツールチップ内でハイライトされた run のみを表示します |
| **ツールチップ内に表示される run の数** | ツールチップ内で表示される run の数を表示します |
| **プライマリチャートのツールチップにフル run 名を表示**| チャートツールチップで run のフルネームを表示します |

## セクション設定

セクション設定は、そのセクション内のすべてのパネルに適用されます。ワークスペースセクション内では、パネルをソートしたり、並べ替えたり、セクション名を変更したりできます。

セクション設定を変更するには、セクションの右上隅にある3つの水平ドット (**...**) を選択します。

{{< img src="/images/app_ui/section_settings.png" alt="" >}}

ドロップダウンから、セクション全体に適用される次の設定を編集できます:

| セクション設定 | 説明 |
| ----- | ----- |
| **セクションの名前を変更** | セクションの名前を変更します |
| **パネルを A-Z に並べ替え** | セクション内のパネルをアルファベット順に並べ替えます |
| **パネルを並べ替え** | セクション内でパネルを手動で並べ替えるために、パネルを選択してドラッグします |

以下のアニメーションは、セクション内でパネルを並べ替える方法を示しています:

{{< img src="/images/app_ui/rearrange_panels.gif" alt="" >}}

{{% alert %}}
この表で説明されている設定に加えて、ワークスペースでのセクションの表示方法も編集できます。たとえば、**Add section below**、**Add section above**、**Delete section**、**Add section to report** などです。
{{% /alert %}}

## パネル設定

個々のパネルの設定をカスタマイズして、同じプロットで複数のラインを比較したり、カスタム軸を計算したり、ラベルを変更したりすることができます。パネルの設定を編集するには:

1. 編集したいパネルにマウスを乗せます。
2. 現れる鉛筆アイコンを選択します。
{{< img src="/images/app_ui/panel_settings.png" alt="" >}}
3. 表示されたモーダル内で、パネルのデータ、ディスプレイの設定などに関連する設定を編集できます。
{{< img src="/images/app_ui/panel_settings_modal.png" alt="" >}}

パネルに適用できる設定の完全なリストについては、[Edit line panel settings]({{< relref path="/guides/models/app/features/panels/line-plot/#edit-line-panel-settings" lang="ja" >}}) を参照してください。