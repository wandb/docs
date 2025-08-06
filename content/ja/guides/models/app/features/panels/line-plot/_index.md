---
title: 折れ線グラフ
description: メトリクスを可視化し、軸をカスタマイズして、プロット上で複数の線を比較できます
url: guides/app/features/panels/line-plot
menu:
  default:
    identifier: intro_line_plot
    parent: panels
cascade:
- url: guides/app/features/panels/line-plot/:filename
weight: 10
---

折れ線グラフは、`wandb.Run.log()` でメトリクスを時間経過で記録する際にデフォルトで表示されます。チャート設定をカスタマイズして、同じプロット上で複数のラインを比較したり、カスタム軸を計算したり、ラベル名を変更することができます。

{{< img src="/images/app_ui/line_plot_example.png" alt="折れ線グラフの例" >}}

## 折れ線グラフの設定を編集する

このセクションでは、個別の折れ線グラフパネル、セクション内のすべての折れ線グラフパネル、またはワークスペース全体の全折れ線グラフパネルに対して設定を編集する方法を説明します。

{{% alert %}}
カスタムの x 軸を使用したい場合は、y 軸を記録するのと同じ `wandb.Run.log()` の呼び出しで x 軸も記録してください。
{{% /alert %}} 

### 個別の折れ線グラフ
個々の折れ線グラフの設定は、セクションやワークスペースで設定した折れ線グラフの設定を上書きします。折れ線グラフをカスタマイズするには：

1. パネルにマウスカーソルを重ね、歯車アイコンをクリックします。
1. 表示されるドロワー内で、編集したいタブを選択し、[設定]({{< relref "#line-plot-settings" >}}) を開きます。
1. **適用** をクリックします。

#### 折れ線グラフの設定
折れ線グラフに対して以下の設定が可能です：

**データ**: グラフのデータ表示の詳細を設定します。
* **X軸**: X軸に使う値を選択します（デフォルトは **Step**）。x軸は **相対時間** に変更できたり、W&Bで記録した値をもとにカスタム軸を選択できます。さらにX軸のスケールや範囲も指定可能です。
  * **Relative Time (Wall)** はプロセス開始からの時計時間です。たとえば run を開始し、1日後に再開して何かを記録した場合、その点は24時間の位置にプロットされます。
  * **Relative Time (Process)** はプロセス内部での時間。run を10秒間実行し、1日後に再開した場合、そのポイントは10秒でプロットされます。
  * **Wall Time** はグラフ上の最初の run 開始から経過した分数です。
  * **Step** は `wandb.Run.log()` が呼ばれるごとにデフォルトでインクリメントされ、モデルから記録したトレーニングのステップ数を反映します。
* **Y軸**: 記録した値から1つ以上選択します。メトリクスや時間とともに変化するハイパーパラメーターなどが含まれます。Y軸のスケールや範囲も指定できます。
* **ポイント集約方法**: **ランダムサンプリング**（デフォルト）または **フルフィデリティ**。詳細は [Sampling]({{< relref "sampling.md" >}}) を参照してください。
* **スムージング**: ラインプロットのスムージングを変更します。デフォルトは **時系列重み付きEMA**。その他に **スムージングなし**、**移動平均**、**ガウス** などがあります。
* **外れ値**: 外れ値を除外し、表示範囲（最小・最大）を再スケールします。
* **最大runまたはグループ数**: この数値を増やすことで、同時により多くのラインを折れ線グラフに表示できます（デフォルトは10 run）。10個以上のrunが存在してもグラフで表示できる数が制限されている場合、画面上部に "Showing first 10 runs" が表示されます。
* **チャートタイプ**: 折れ線グラフ、面グラフ、パーセント面グラフを切り替えられます。

**グルーピング**: グラフ内で run をどのようにグループ化・集約するかを設定します。
* **グループ化**: 列を選ぶと、その列内で同じ値を持つ run がグループ化されます。
* **Agg**: グラフ上のラインを集約する方法。平均、中央値、最小値、最大値から選べます。

**チャート**: パネルやX/Y軸のタイトル、凡例の表示/非表示や位置の設定ができます。

**凡例**: パネルの凡例が有効な場合、その外観をカスタマイズできます。
* **凡例**: プロットの各ラインの凡例に表示されるフィールド名を設定します。
* **凡例テンプレート**: 上部やマウスオーバー時に表示する内容など、凡例用のテンプレートをフルカスタマイズ可能です。どのテキスト・変数を表示するかを定義します。

**式**: パネルにカスタム計算式を追加します。
* **Y軸式**: グラフに計算メトリクスを追加します。記録済みのメトリクスやハイパーパラメーターなどの設定値を用いてカスタムラインを計算可能です。
* **X軸式**: カスタム式で計算した値を使ってx軸を再スケールします。有用な変数にはデフォルトx軸の `*_step*` や、サマリー値参照の `${summary:value}` などがあります。

### セクション内の全折れ線グラフ

セクション内のすべての折れ線グラフに対するデフォルト設定をカスタマイズし、ワークスペースの設定より優先させるには：

1. セクションの歯車アイコンをクリックし、設定を開きます。
1. 表示されるドロワー内の **データ** または **表示設定** タブで、セクションのデフォルト設定を行います。各 **データ** 設定の詳細は前述の [個別の折れ線グラフ]({{< relref "#line-plot-settings" >}}) を、表示設定に関しては [セクションレイアウトの設定]({{< relref "../#configure-section-layout" >}}) をご参照ください。

### ワークスペース内の全折れ線グラフ
ワークスペース内すべての折れ線グラフに対するデフォルト設定をカスタマイズするには:

1. ワークスペースの設定（**Settings** とラベル表示され、歯車アイコン）をクリックします。
1. **Line plots** をクリックします。
1. 表示されるドロワー内で、**データ** または **表示設定** タブからワークスペースのデフォルト設定を行います。
    - 各 **データ** 設定の詳細は、前述の [個別の折れ線グラフ]({{< relref "#line-plot-settings" >}}) を参照してください。

    - 各 **表示設定** セクションについては [ワークスペースの表示設定]({{< relref "../#configure-workspace-layout" >}}) をご覧ください。ワークスペースレベルでは、折れ線グラフ同士でx軸キーが一致する場合の **ズーム操作の同期** デフォルト動作も構成可能です。この設定はデフォルトで無効です。



## グラフ上で平均値を可視化する

複数の異なる実験があり、それらの値の平均を可視化したい場合は、ランテーブルのグループ化機能を使えます。runテーブル上部の「Group」をクリックして「All」を選べば、グラフ内ですべての値の平均が表示されます。

平均化前のグラフは次のような外観です：

{{< img src="/images/app_ui/demo_precision_lines.png" alt="個別のprecisionライン" >}}

次の画像は、グループ化されたラインで run 全体の平均値を表現しています。

{{< img src="/images/app_ui/demo_average_precision_lines.png" alt="平均化されたprecisionライン" >}}

## NaN値をグラフで可視化する

`wandb.Run.log()` を使うことで、PyTorchのテンソルを含め `NaN` 値も折れ線グラフで可視化できます。例:

```python
with wandb.init() as run:
    # NaN値を記録
    run.log({"test": float("nan")})
```

{{< img src="/images/app_ui/visualize_nan.png" alt="NaN値の取扱い" >}}

## ひとつのチャートで2つのメトリクスを比較する

{{< img src="/images/app_ui/visualization_add.gif" alt="可視化パネルの追加" >}}

1. ページ右上の **Add panels** ボタンを選択。
2. 表示された左側のパネルで Evaluation ドロップダウンを展開。
3. **Run comparer** を選択。

## 折れ線グラフの色を変更する

デフォルトのrunの色では比較しづらい場合があります。そのため wandb では、色を手動で変更できる2つの方法を提供しています。

{{< tabpane text=true >}}
{{% tab header="ランテーブルから" value="run_table" %}}

  各runには初期化時にランダムな色が割り当てられます。

  {{< img src="/images/app_ui/line_plots_run_table_random_colors.png" alt="Runごとにランダムに色がつく" >}}

  どれかの色をクリックするとカラーパレットが表示され、手動で好きな色を選択できます。

  {{< img src="/images/app_ui/line_plots_run_table_color_palette.png" alt="カラーパレット" >}}

{{% /tab %}}

{{% tab header="チャートの凡例設定から" value="legend_settings" %}}

1. 編集したいパネルにマウスを重ねます。
2. 表示された鉛筆アイコンをクリックします。
3. **Legend** タブを選びます。

{{< img src="/images/app_ui/plot_style_line_plot_legend.png" alt="折れ線グラフ凡例の設定" >}}

{{% /tab %}}
{{< /tabpane >}}

## 異なるx軸で可視化する

実験の実行にかかった絶対時間や、どの日に実行したかを見たい場合は、x軸の切り替えが可能です。例えば、StepからRelative Time、Wall Timeへの切り替え例はこちらです。

{{< img src="/images/app_ui/howto_use_relative_time_or_wall_time.gif" alt="x軸の時間オプション" >}}

## 面グラフ

折れ線グラフの設定の「高度な」タブで、グラフスタイルを切り替えることで面グラフやパーセント面グラフを得られます。

{{< img src="/images/app_ui/line_plots_area_plots.gif" alt="面グラフのスタイル" >}}

## ズーム

縦横同時に矩形範囲をドラッグしてズームできます。これによりx軸・y軸共に拡大縮小が行えます。

{{< img src="/images/app_ui/line_plots_zoom.gif" alt="プロットのズーム機能" >}}

## チャートの凡例を非表示にする

以下の簡単な切り替え操作で折れ線グラフの凡例をオフにできます：

{{< img src="/images/app_ui/demo_hide_legend.gif" alt="凡例の非表示トグル" >}}

## runメトリクスの通知を作成する
[Automations]({{< relref "/guides/core/automations" >}}) を利用して、特定条件でrunメトリクスに基づいてチームへ通知を送れます。Slackのチャンネルへ投稿したり、ウェブフックを実行することも可能です。

折れ線グラフから、そのメトリックの [runメトリクス通知]({{< relref "/guides/core/automations/automation-events.md#run-events" >}}) を素早く作成できます：

1. パネルにマウスを重ねてベルアイコンをクリックします。
1. 基本または高度な設定からオートメーションを構成します（例：runフィルターで対象範囲を限定したり、絶対値のしきい値を設定したりできます）。

より詳細は [Automations]({{< relref "/guides/core/automations" >}}) をご覧ください。

## CoreWeave のインフラストラクチャーアラートを可視化する

W&Bに記録した機械学習実験中に発生したGPU障害や熱違反など、インフラストラクチャーアラートを監視することができます。[W&B run]({{< relref "/guides/models/track/runs/_index" >}})実行中は、[CoreWeave Mission Control](https://www.coreweave.com/mission-control)が計算インフラストラクチャーを監視します。

{{< alert>}}
この機能はプレビュー版であり、CoreWeave クラスター上でのトレーニング時のみ利用可能です。ご利用にはW&B担当者までお問い合わせください。
{{< /alert >}}

エラーが発生した場合、その情報はCoreWeaveからW&Bへ送信されます。W&Bはrunのチャートにインフラストラクチャー情報を表示します。一部の問題はCoreWeaveが自動的に解決を試み、その情報もrunページに表示されます。

### run 内のインフラストラクチャ問題を探す

W&Bは SLURM ジョブ問題とクラスター ノード問題の両方を表示します。 run 内のインフラエラーを見るには：

1. W&B Appでプロジェクトを開きます。
2. **Workspace** タブを選択してそのプロジェクトのワークスペースを表示します。
3. インフラストラクチャーに問題のある run 名を検索して選択します。CoreWeave が問題を検知していれば、そのrunのチャート上にビックリマーク付き赤い縦線が1本以上重なって表示されます。
4. プロット上の問題を選ぶか、ページ右上の **Issues** ボタンを選択します。CoreWeaveが報告した各問題のリストが表示されたドロワーが現れます。

{{< alert title="ヒント" >}}
インフラストラクチャーの問題があるrunを一覧で確認したい場合、**Issues** 列をW&Bワークスペースにピン固定しましょう。ピン方法については [runの表示カスタマイズ]({{< relref "/guides/models/track/runs/#customize-how-runs-are-displayed" >}}) を参照ください。
{{< /alert >}}

**Overall Grafana view**(ドロワー上部)は、そのSLURMジョブのGrafanaダッシュボードへリダイレクトし、runに関するシステムレベルの詳細情報を確認できます。**Issues summary** には CoreWeave Mission Control に SLURM ジョブが報告した根本的なエラーが記載されています。このサマリー部分にはCoreWeaveによる自動解決の試みについても記載があります。

{{< img src="/images/app_ui/cw_wb_observability.png" >}}

**All Issues** にはrun中に発生した全ての問題が新しい順で一覧化されます。リストにはジョブ問題とノード問題が含まれます。各アラート内には問題名、発生時刻、その問題に関するGrafanaダッシュボードへのリンク、問題内容を簡単に説明するサマリーがあります。

下表は各インフラストラクチャー問題カテゴリーごとのアラート例です：

| カテゴリー | アラート例 |
| -------- | ------------- |
| ノードの可用性・準備状況 | `KubeNodeNotReadyHGX`, `NodeExtendedDownTime` |
| GPU/アクセラレータエラー | `GPUFallenOffBusHGX`, `GPUFaultHGX`, `NodeTooFewGPUs` |
| ハードウェアエラー | `HardwareErrorFatal`, `NodeRAIDMemberDegraded` |
| ネットワーキング & DNS | `NodeDNSFailureHGX`, `NodeEthFlappingLegacyNonGPU` |
| 電源・冷却・管理 | `NodeCPUHZThrottle`, `RedfishDown` |
| DPU & NVSwitch | `DPUNcoreVersionBelowDesired`, `NVSwitchFaultHGX` |
| その他 | `NodePCISpeedRootGBT`, `NodePCIWidthRootSMC` |

エラー種別の詳細については [CoreWeave DocsのSLURMジョブメトリクス](https://docs.coreweave.com/docs/observability/managed-grafana/sunk/slurm-job-metrics#job-info-alerts#job-info-alerts) をご参照ください。

### インフラストラクチャ問題をデバッグする

W&Bで作成される各runは、CoreWeave上の1つのSLURMジョブに対応しています。失敗したジョブの [Grafana](https://grafana.com/) ダッシュボードや個々のノードの詳細を確認できます。**Issues**ドロワーの **Overview** セクションにあるリンクからSLURMジョブのGrafanaダッシュボードへ移動可能です。**All Issues** ドロップダウンを展開するとジョブ/ノードの問題とそれぞれのGrafanaダッシュボードへのリンクを確認できます。

{{< alert title="注意" >}}
GrafanaダッシュボードはCoreWeaveアカウントを持つW&Bユーザーのみ利用可能です。利用にはW&Bまでご連絡の上、お使いのW&B組織でGrafanaを設定してください。
{{< /alert >}}

問題によっては、SLURMジョブの設定を調整したり、ノードの状態を調査したり、ジョブを再起動したり、必要に応じて他の対応をとる必要があります。

CoreWeaveにおけるGrafanaのSLURMジョブについては [CoreWeave Docs](https://docs.coreweave.com/docs/observability/managed-grafana/sunk/slurm-job-metrics#job-info-alerts) を、ジョブアラートの詳細は [Job info: alerts](https://docs.coreweave.com/docs/observability/managed-grafana/sunk/slurm-job-metrics#job-info-alerts#job-info-alerts) をご参照ください。