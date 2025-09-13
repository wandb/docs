---
title: Workspace、セクション、パネルの設定を管理する
menu:
  default:
    identifier: ja-guides-models-app-features-cascade-settings
    parent: w-b-app-ui-reference
url: guides/app/features/cascade-settings
---

特定の Workspace ページには、Workspace、セクション、パネルという 3 つの設定レベルがあります。[Workspace settings]({{< relref path="#workspace-settings" lang="ja" >}}) は Workspace 全体に適用されます。[Section settings]({{< relref path="#section-settings" lang="ja" >}}) はそのセクション内のすべてのパネルに適用されます。[Panel settings]({{< relref path="#panel-settings" lang="ja" >}}) は個々のパネルに適用されます。 

## Workspace の 設定

Workspace の 設定は、すべてのセクションと、それらのセクション内のすべてのパネルに適用されます。編集できる Workspace 設定は 2 種類あります: [Workspace layout]({{< relref path="#workspace-layout-options" lang="ja" >}}) と [Line plots]({{< relref path="#line-plots-options" lang="ja" >}})。**Workspace layouts** は Workspace の構造を決め、**Line plots** は Workspace 内の line plot の既定設定を制御します。

この Workspace の全体構造に関わる設定を編集するには:

1. 自分の Project の Workspace に移動します。
2. **New report** ボタンの横にある歯車アイコンをクリックして Workspace 設定を表示します。
3. Workspace のレイアウトを変更するには **Workspace layout** を、Workspace 内の line plot の既定設定を設定するには **Line plots** を選択します。
{{< img src="/images/app_ui/workspace_settings.png" alt="Workspace 設定の歯車アイコン" >}}

{{% alert %}}
Workspace をカスタマイズしたら、 _Workspace テンプレート_ を使用して、同じ設定で新しい Workspace をすばやく作成できます。詳しくは [Workspace templates]({{< relref path="/guides/models/track/workspaces.md#workspace-templates" lang="ja" >}}) を参照してください。
{{% /alert %}}

### Workspace layout の オプション

Workspace の レイアウトを 設定して、Workspace の全体構造を定義します。これには、セクション分割のロジックやパネルの配置が含まれます。 

{{< img src="/images/app_ui/workspace_layout_settings.png" alt="Workspace layout の オプション" >}}

Workspace layout オプションのページでは、Workspace がパネルを自動生成するか手動で生成するかを確認できます。Workspace のパネル生成モードを調整するには、[パネル]({{< relref path="panels/" lang="ja" >}}) を参照してください。

次の表は、それぞれの Workspace layout オプションを説明します。

| Workspace 設定 | 説明 |
| ----- | ----- |
| **検索中に空のセクションを非表示** | パネルを検索するとき、パネルを含まないセクションを非表示にします。|
| **パネルをアルファベット順に並べ替える** | Workspace 内のパネルをアルファベット順に並べ替えます。 |
| **セクションの整理** | 既存のセクションとパネルをすべて削除し、新しいセクション名で再構成します。新しく作成されたセクションは、先頭または末尾のプレフィックスでグループ化します。 |

{{% alert %}}
W&B は、末尾のプレフィックスでグループ化するよりも、先頭のプレフィックスでセクションをグループ化することを推奨します。先頭のプレフィックスでグループ化すると、セクション数が少なくなり、パフォーマンスが向上する場合があります。
{{% /alert %}}

### Line plots の オプション
**Line plots** の Workspace 設定を変更して、Workspace 内の line plot のグローバル既定値やカスタムルールを設定します。

{{< img src="/images/app_ui/workspace_settings_line_plots.png" alt="Line plot の 設定" >}}

**Line plots** 内では主に **Data** と **Display preferences** の 2 つを設定できます。**Data** タブには次の設定があります:


| Line plot の 設定 | 説明 |
| ----- | ----- |
| **X axis** | Line plot の X 軸のスケール。既定は **Step**。X 軸のオプションは次の表を参照してください。 |
| **Range** | X 軸に表示する最小値と最大値。 |
| **Smoothing** | Line plot のスムージングを変更します。スムージングの詳細は [Line plot の スムージング]({{< relref path="/guides/models/app/features/panels/line-plot/smoothing.md" lang="ja" >}}) を参照してください。 |
| **Outliers** | 既定の最小/最大スケールから外れ値を除外するように再スケールします。 |
| **Point aggregation method** | データ可視化 の 精度とパフォーマンスを改善します。詳しくは [Point aggregation]({{< relref path="/guides/models/app/features/panels/line-plot/sampling.md" lang="ja" >}}) を参照してください。 |
| **Max number of runs or groups** | Line plot に表示する runs またはグループの最大数を制限します。 |

**Step** に加えて、X 軸には次のオプションがあります:

| X 軸オプション | 説明 |
| ------------- | ----------- |
| **Relative Time (Wall)**| プロセスの開始からのタイムスタンプ。例えば、run を開始して翌日にその run を再開し、そのときに何かを ログ すると、記録されるポイントは 24 時間になります。|
| **Relative Time (Process)** | 実行中のプロセス内のタイムスタンプ。例えば、run を開始して 10 秒間実行し、翌日にその run を再開した場合、そのポイントは 10 秒として記録されます。 |
| **Wall Time** | グラフ上の最初の run の開始から経過した分数。 |
| **Step** | `wandb.Run.log()` を呼ぶたびに増加します。|



{{% alert %}}
個々の line plot を編集する方法は、Line plots の [Edit line panel settings]({{< relref path="/guides/models/app/features/panels/line-plot/#edit-line-panel-settings" lang="ja" >}}) を参照してください。 
{{% /alert %}}


**Display preferences** タブでは、次の設定を切り替えられます:

| 表示設定 | 説明 |
| ----- | ----- |
| **すべてのパネルから凡例を削除** | パネルの凡例を削除します。 |
| **ツールチップに色付きの run 名を表示** | ツールチップ内で run を色付きテキストで表示します。 |
| **関連チャートのツールチップにハイライトされた run のみ表示** | チャートのツールチップにハイライトされた run のみを表示します。 |
| **ツールチップに表示する run の数** | ツールチップに表示する run の数を指定します。 |
| **メインチャートのツールチップで run 名をフル表示**| チャートのツールチップで run のフルネームを表示します。 |




## セクションの 設定

セクションの設定は、そのセクション内のすべてのパネルに適用されます。Workspace のセクション内では、パネルの並べ替え、パネルの再配置、セクション名の変更ができます。

セクションの右上にある三点リーダー（**...**）を選択して、セクションの設定を変更します。

{{< img src="/images/app_ui/section_settings.png" alt="セクション設定メニュー" >}}

ドロップダウンから、セクション全体に適用される次の設定を編集できます:

| セクションの 設定 | 説明 |
| ----- | ----- |
| **セクション名の変更** | セクションの名前を変更します。 |
| **パネルを A-Z で並べ替え** | セクション内のパネルをアルファベット順に並べ替えます。 |
| **パネルを並べ替える** | セクション内のパネルを選択してドラッグし、手動で順序を変更します。 |

次のアニメーションは、セクション内でパネルを並べ替える方法を示します:

{{< img src="/images/app_ui/rearrange_panels.gif" alt="パネルの並べ替え" >}}

{{% alert %}}
上の表で説明した設定に加えて、**Add section below**、**Add section above**、**Delete section**、**Add section to report** のように、Workspace 内でのセクションの見え方も編集できます。 
{{% /alert %}}

## パネルの 設定

同じプロット上で複数の線を比較したり、カスタム軸を計算したり、ラベル名を変更したりするなど、個々のパネルの設定をカスタマイズできます。パネルの設定を編集するには:

1. 編集したいパネルにマウスカーソルを重ねます。 
2. 表示される鉛筆アイコンを選択します。
{{< img src="/images/app_ui/panel_settings.png" alt="パネル編集アイコン" >}}
3. 表示されるモーダルで、パネルの データ、表示設定 などに関する設定を編集できます。
{{< img src="/images/app_ui/panel_settings_modal.png" alt="パネル設定モーダル" >}}

パネルに適用できる設定の一覧は、[Edit line panel settings]({{< relref path="/guides/models/app/features/panels/line-plot/#edit-line-panel-settings" lang="ja" >}}) を参照してください。