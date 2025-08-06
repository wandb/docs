---
title: ワークスペース、セクション、パネルの設定を管理する
menu:
  default:
    identifier: cascade-settings
    parent: w-b-app-ui-reference
url: guides/app/features/cascade-settings
---

ある Workspace ページ内では、3 つの異なる設定レベルがあります：Workspace、Section、Panel です。[Workspace 設定]({{< relref "#workspace-settings" >}})は Workspace 全体に適用されます。[Section 設定]({{< relref "#section-settings" >}})は、そのセクション内のすべての Panel に適用されます。[Panel 設定]({{< relref "#panel-settings" >}})は、個々の Panel に適用されます。

## Workspace 設定

Workspace 設定は、その Workspace 内のすべての Section とすべての Panel に適用されます。2 種類の Workspace 設定を編集できます：[Workspace レイアウト]({{< relref "#workspace-layout-options" >}}) と [Line plots]({{< relref "#line-plots-options" >}}) です。**Workspace レイアウト** は Workspace の構造を決め、**Line plots** の設定は Workspace 内の Line plot のデフォルト設定を管理します。

Workspace の全体的な構造に関する設定を編集するには：

1. プロジェクトの Workspace に移動します。
2. **New report** ボタンの隣にあるギアアイコンをクリックして Workspace 設定を開きます。
3. **Workspace layout** を選択して Workspace のレイアウトを変更するか、**Line plots** を選択して Line plot のデフォルト設定を Workspace 内で設定します。
{{< img src="/images/app_ui/workspace_settings.png" alt="Workspace 設定 ギア アイコン" >}}

{{% alert %}}
Workspace をカスタマイズした後は、_Workspace テンプレート_ を使うことで同じ設定で新しい Workspace を素早く作成できます。詳しくは [Workspace テンプレート]({{< relref "/guides/models/track/workspaces.md#workspace-templates" >}}) をご参照ください。
{{% /alert %}}

### Workspace レイアウトオプション

Workspace レイアウトを設定することで、その Workspace の全体的な構造を定義します。これには Section の分割方法や Panel の配置などが含まれます。

{{< img src="/images/app_ui/workspace_layout_settings.png" alt="Workspace レイアウトオプション" >}}

Workspace レイアウトオプションのページでは、Workspace が Panel を自動で生成するか手動で生成するかが表示されます。Panel の生成方法を調整する場合は、[Panels]({{< relref "panels/" >}}) をご参照ください。

次の表は、各 Workspace レイアウトオプションについて説明しています。

| Workspace 設定 | 説明 |
| ----- | ----- |
| **検索時に空のセクションを非表示** | Panel が存在しない Section を、Panel 検索時に非表示にします。|
| **Panel をアルファベット順に並べ替え** | Workspace で Panel をアルファベット順に並べ替えます。|
| **Section の構成** | すべての既存の Section および Panel を削除し、新しい Section 名で再配置します。新しく配置した Section をプレフィックスの先頭または末尾でグループ化します。|

{{% alert %}}
W&B では、Section のグループ化はプレフィックスの末尾よりも先頭でまとめることをおすすめしています。先頭でグループ化することで Section 数が少なくなり、パフォーマンスも向上します。
{{% /alert %}}

### Line plots オプション

Workspace 設定の **Line plots** から、Line plot のグローバルデフォルトおよびカスタムルールを設定できます。

{{< img src="/images/app_ui/workspace_settings_line_plots.png" alt="Line plot 設定" >}}

**Line plots** 設定の中で編集できる主な設定は **Data** と **Display preferences** の 2 種類です。**Data** タブには次の項目があります：

| Line plot 設定 | 説明 |
| ----- | ----- |
| **X 軸** |  Line plot の x 軸のスケール。デフォルトは **Step** です。x 軸オプションの詳細は次の表を参照してください。|
| **範囲** |  x 軸に表示される最小・最大範囲の設定です。|
| **平滑化** | Line plot の平滑化を変更します。平滑化の詳細は [Smooth line plots]({{< relref "/guides/models/app/features/panels/line-plot/smoothing.md" >}}) をご覧ください。|
| **外れ値** | 外れ値を除外してデフォルトの plot 最小・最大スケールをリスケーリングします。|
| **ポイント集約方法** | データ可視化の精度やパフォーマンスを向上させます。詳細は [Point aggregation]({{< relref "/guides/models/app/features/panels/line-plot/sampling.md" >}}) をご覧ください。|
| **表示する run またはグループの最大数** | Line plot に表示される run またはグループの最大数を制限します。|

**Step** 以外にも、x 軸には他のオプションが用意されています：

| X 軸オプション | 説明 |
| ------------- | ----------- |
| **Relative Time (Wall)**| プロセスが開始してからのタイムスタンプ。たとえば、Run を開始して翌日に再開し、その時点で何かを記録すると、記録されるタイムスタンプは 24 時間になります。|
| **Relative Time (Process)** | 実行中プロセス内のタイムスタンプ。たとえば、Run を開始後 10 秒間走らせ、翌日 Run を再開したとき、記録されるポイントは 10 秒となります。|
| **Wall Time** | グラフ上で最初の Run の開始から経過した分数です。|
| **Step** | `wandb.Run.log()` を呼び出すたびに増加します。|


{{% alert %}}
個別の Line plot の編集方法については、[Line panel 設定を編集]({{< relref "/guides/models/app/features/panels/line-plot/#edit-line-panel-settings" >}}) をご覧ください。
{{% /alert %}}


**Display preferences** タブでは、次の設定のオン・オフを切り替えられます。

| 表示設定 | 説明 |
| ----- | ----- |
| **すべての Panel から凡例を削除** | Panel の凡例（ラベル）を削除します |
| **ツールチップに色付き run 名を表示** | ツールチップ内の run 名を色付きで表示します |
| **強調表示された run のみコンパニオンチャートツールチップに表示** | チャートのツールチップ内には強調表示された run のみ表示されます |
| **ツールチップに表示される run の数** | ツールチップで表示する run の数を指定します |
| **メインチャートのツールチップに run 名をすべて表示**| メインチャートのツールチップに run のフルネームを表示します |


## Section 設定

Section 設定は、その Section 内のすべての Panel に適用されます。Workspace の Section 内では Panel の並べ替え、再配置、Section 名の変更ができます。

Section 設定を変更するには、Section 右上の 3 つ並んだ点（**...**）を選択します。

{{< img src="/images/app_ui/section_settings.png" alt="Section 設定メニュー" >}}

ドロップダウンから、Section 全体に適用される次の設定が編集できます：

| Section 設定 | 説明 |
| ----- | ----- |
| **Section 名を変更** | Section 名の編集 |
| **Panel をアルファベット順で並べ替え** | Section 内の Panel をアルファベット順に並べ替え |
| **Panel を再配置** | Section 内の Panel を選択し、ドラッグして手動並び替え |

次のアニメーションは Section 内で Panel を再配置する方法を示しています：

{{< img src="/images/app_ui/rearrange_panels.gif" alt="Panel の再配置" >}}

{{% alert %}}
前述した設定に加えて、**下に Section を追加**、**上に Section を追加**、**Section の削除**、**Section を Report に追加** もワークスペースで編集できます。
{{% /alert %}}

## Panel 設定

個々の Panel 設定をカスタマイズして、同じ Plot 上で複数のラインを比較したり、カスタム軸の計算、ラベル名変更などができます。Panel の設定を編集するには：

1. 編集したい Panel の上にカーソルを乗せます。
2. 表示される鉛筆アイコンをクリックします。
{{< img src="/images/app_ui/panel_settings.png" alt="Panel 編集アイコン" >}}
3. 表示されるモーダル内で、その Panel のデータや表示設定などを編集できます。
{{< img src="/images/app_ui/panel_settings_modal.png" alt="Panel 設定モーダル" >}}

Panel に適用できる設定の一覧については、[Line panel 設定を編集]({{< relref "/guides/models/app/features/panels/line-plot/#edit-line-panel-settings" >}}) をご覧ください。