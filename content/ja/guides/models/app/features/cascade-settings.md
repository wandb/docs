---
title: Manage workspace, section, and panel settings
menu:
  default:
    identifier: ja-guides-models-app-features-cascade-settings
    parent: w-b-app-ui-reference
url: guides/app/features/cascade-settings
---

特定の ワークスペース ページ内には、ワークスペース、セクション、パネルという3つの異なる設定レベルがあります。[ワークスペース の 設定]({{< relref path="#workspace-settings" lang="ja" >}}) は、ワークスペース 全体に適用されます。[セクション の 設定]({{< relref path="#section-settings" lang="ja" >}}) は、セクション内のすべてのパネルに適用されます。[パネル の 設定]({{< relref path="#panel-settings" lang="ja" >}}) は、個々のパネルに適用されます。

## ワークスペース の 設定

ワークスペース の 設定は、すべてのセクションと、それらのセクション内のすべてのパネルに適用されます。編集できるワークスペース の 設定には、[**ワークスペース の レイアウト**]({{< relref path="#workspace-layout-options" lang="ja" >}}) と [**折れ線グラフ**]({{< relref path="#line-plots-options" lang="ja" >}}) の2種類があります。**ワークスペース の レイアウト** は ワークスペース の構造を決定し、**折れ線グラフ** の 設定は ワークスペース 内の折れ線グラフのデフォルト設定を制御します。

この ワークスペース の全体的な構造に適用される設定を編集するには:

1. プロジェクト の ワークスペース に移動します。
2. **New report** ボタンの横にある歯車アイコンをクリックして、ワークスペース の 設定を表示します。
3. ワークスペース のレイアウトを変更するには **Workspace layout** を選択し、ワークスペース 内の折れ線グラフのデフォルト設定を構成するには **Line plots** を選択します。
{{< img src="/images/app_ui/workspace_settings.png" alt="" >}}

### ワークスペース の レイアウト オプション

ワークスペース のレイアウトを構成して、ワークスペース の全体的な構造を定義します。これには、セクション分割ロジックとパネルの構成が含まれます。

{{< img src="/images/app_ui/workspace_layout_settings.png" alt="" >}}

ワークスペース のレイアウト オプション ページには、ワークスペース がパネルを自動的に生成するか、手動で生成するかが表示されます。ワークスペース のパネル生成モードを調整するには、[Panels]({{< relref path="panels/" lang="ja" >}}) を参照してください。

次の表は、各ワークスペース のレイアウト オプションについて説明したものです。

| ワークスペース の 設定 | 説明 |
| ----- | ----- |
| **検索時に空のセクションを非表示にする** | パネルを検索するときに、パネルを含まないセクションを非表示にします。|
| **パネルをアルファベット順に並べ替える** | ワークスペース 内のパネルをアルファベット順に並べ替えます。 |
| **セクション の 構成** | 既存のすべてのセクションとパネルを削除し、新しいセクション名で再作成します。新しく作成されたセクションを、最初のプレフィックスまたは最後のプレフィックスでグループ化します。 |

{{% alert %}}
W&B は、最後のプレフィックスでグループ化するのではなく、最初のプレフィックスでセクションを構成することを推奨します。最初のプレフィックスでグループ化すると、セクションの数が減り、パフォーマンスが向上する可能性があります。
{{% /alert %}}

### 折れ線グラフ の オプション
**Line plots** ワークスペース の 設定を変更して、ワークスペース 内の折れ線グラフのグローバルなデフォルトとカスタムルールを設定します。

{{< img src="/images/app_ui/workspace_settings_line_plots.png" alt="" >}}

**Line plots** の 設定では、**Data** と **Display preferences** という2つのメイン設定を編集できます。**Data** タブには、次の設定が含まれています。

| 折れ線グラフ の 設定 | 説明 |
| ----- | ----- |
| **X軸** | 折れ線グラフのX軸のスケール。X軸はデフォルトで **Step** に設定されています。X軸のオプションのリストについては、次の表を参照してください。 |
| **範囲** | X軸に表示する最小値と最大値の設定。 |
| **Smoothing** | 折れ線グラフの Smoothing を変更します。Smoothing の詳細については、[Smooth line plots]({{< relref path="/guides/models/app/features/panels/line-plot/smoothing.md" lang="ja" >}}) を参照してください。 |
| **Outliers** | デフォルトのプロットの最小スケールと最大スケールから外れ値を排除するためにリスケールします。 |
| **Point aggregation method** | Data Visualization の精度とパフォーマンスを向上させます。詳細については、[Point aggregation]({{< relref path="/guides/models/app/features/panels/line-plot/sampling.md" lang="ja" >}}) を参照してください。 |
| **Max number of runs or groups** | 折れ線グラフに表示される Runs またはグループの数を制限します。 |

**Step** に加えて、X軸には次のオプションがあります。

| X軸 オプション | 説明 |
| ------------- | ----------- |
| **相対時間 (Wall)**| プロセス の開始からのタイムスタンプ。たとえば、run を開始し、翌日にその run を再開するとします。その後、何かを log に記録すると、記録されたポイントは24時間になります。|
| **相対時間 (Process)** | 実行中のプロセス内のタイムスタンプ。たとえば、run を開始して10秒間続行するとします。翌日にその run を再開します。ポイントは10秒として記録されます。 |
| **Wall Time** | グラフ上の最初の run の開始からの経過時間 (分)。 |
| **Step** | `wandb.log()` を呼び出すたびに増分します。|

{{% alert %}}
個々の折れ線グラフの編集方法については、折れ線グラフの [Edit line panel settings]({{< relref path="/guides/models/app/features/panels/line-plot/#edit-line-panel-settings" lang="ja" >}}) を参照してください。
{{% /alert %}}

**Display preferences** タブでは、次の設定を切り替えることができます。

| 表示設定 | 説明 |
| ----- | ----- |
| **Remove legends from all panels** | パネルの凡例を削除します |
| **Display colored run names in tooltips** | ツールチップ内に Runs を色付きのテキストとして表示します |
| **Only show highlighted run in companion chart tooltip** | チャートのツールチップに強調表示された Runs のみを表示します |
| **Number of runs shown in tooltips** | ツールチップに Runs の数を表示します |
| **Display full run names on the primary chart tooltip**| チャートのツールチップに run のフルネームを表示します |

## セクション の 設定

セクション の 設定は、そのセクション内のすべてのパネルに適用されます。ワークスペース の セクション 内では、パネルの並べ替え、パネルの再配置、セクション名の変更を行うことができます。

セクション の 設定を変更するには、セクション の右上隅にある3つの水平ドット (**...**) を選択します。

{{< img src="/images/app_ui/section_settings.png" alt="" >}}

ドロップダウンから、セクション全体に適用される次の設定を編集できます。

| セクション の 設定 | 説明 |
| ----- | ----- |
| **Rename a section** | セクションの名前を変更します |
| **Sort panels A-Z** | セクション内のパネルをアルファベット順に並べ替えます |
| **Rearrange panels** | セクション内のパネルを選択してドラッグし、パネルを手動で並べ替えます |

次のアニメーションは、セクション内のパネルを再配置する方法を示しています。

{{< img src="/images/app_ui/rearrange_panels.gif" alt="" >}}

{{% alert %}}
上記の表で説明されている設定に加えて、**Add section below**、**Add section above**、**Delete section**、**Add section to report** など、ワークスペース でのセクションの表示方法を編集することもできます。
{{% /alert %}}

## パネル の 設定

個々のパネルの 設定をカスタマイズして、同じプロット上に複数の線を比較したり、カスタム軸を計算したり、ラベルの名前を変更したりできます。パネルの 設定を編集するには:

1. 編集するパネルにマウスを合わせます。
2. 表示される鉛筆アイコンを選択します。
{{< img src="/images/app_ui/panel_settings.png" alt="" >}}
3. 表示されるモーダル内で、パネルの データ、表示設定などに関連する設定を編集できます。
{{< img src="/images/app_ui/panel_settings_modal.png" alt="" >}}

パネルに適用できる設定の完全なリストについては、[Edit line panel settings]({{< relref path="/guides/models/app/features/panels/line-plot/#edit-line-panel-settings" lang="ja" >}}) を参照してください。
