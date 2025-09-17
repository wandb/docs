---
title: 折れ線グラフ
description: メトリクスの可視化、軸のカスタマイズ、プロット上での複数の線の比較
cascade:
- url: guides/app/features/panels/line-plot/:filename
menu:
  default:
    identifier: ja-guides-models-app-features-panels-line-plot-_index
    parent: panels
url: guides/app/features/panels/line-plot
weight: 10
---

メトリクスを `wandb.Run.log()` で時間に対してプロットすると、既定でラインプロットが表示されます。チャートの設定でカスタマイズして、同じプロット上で複数の線を比較したり、カスタム軸を計算したり、ラベル名を変更したりできます。

{{< img src="/images/app_ui/line_plot_example.png" alt="Line plot example" >}}

## ラインプロットの設定を編集

このセクションでは、個々のラインプロット パネル、セクション内のすべてのラインプロット パネル、または Workspace 内のすべてのラインプロット パネルの設定を編集する方法を説明します。

{{% alert %}}
カスタムの x 軸を使いたい場合は、y 軸をログするのと同じ `wandb.Run.log()` 呼び出しでログしていることを確認してください。
{{% /alert %}} 

### 個別のラインプロット
個別の設定は、セクションや Workspace に対するラインプロットの設定よりも優先されます。ラインプロットをカスタマイズするには:

1. パネルにマウスオーバーして、歯車アイコンをクリックします。
1. 表示されるドロワーで、タブを選んでその[設定]({{< relref path="#line-plot-settings" lang="ja" >}})を編集します。
1. **Apply** をクリックします。

#### Line plot settings
ラインプロットでは次の設定を行えます。

**Data**: プロットのデータ表示の詳細を設定します。
* **X axis**: X 軸に使う値を選択します（既定は **Step**）。x 軸を **Relative Time** に変更したり、W&B でログした値に基づくカスタム軸を選択できます。X 軸のスケールや範囲も設定できます。
  * **Relative Time (Wall)** はプロセス開始からの時計時間です。たとえば run を開始して 1 日後に再開して何かをログした場合、その点は 24hrs の位置にプロットされます。
  * **Relative Time (Process)** は実行プロセス内の経過時間です。たとえば run を開始して 10 秒実行し、その 1 日後に再開した場合、その点は 10s にプロットされます。
  * **Wall Time** は、そのグラフ上の最初の run 開始からの経過分です。
  * **Step** は既定では `wandb.Run.log()` が呼ばれるたびに増分され、モデルからログしたトレーニングステップ数を表します。
* **Y axis**: ログされた値（時間とともに変化するメトリクスやハイパーパラメーターを含む）から 1 つ以上の y 軸を選択します。X 軸のスケールや範囲も設定できます。
* **Point aggregation method**: **Random sampling**（既定）または **Full fidelity**。詳しくは [Sampling]({{< relref path="sampling.md" lang="ja" >}}) を参照してください。
* **Smoothing**: ラインプロットのスムージングを変更します。既定は **Time weighted EMA**。そのほか **No smoothing**、**Running average**、**Gaussian** があります。
* **Outliers**: 外れ値を除外するようにリスケールし、既定の最小・最大スケールから外します。
* **Max number of runs or groups**: 一度に表示する線の数を増やします（既定は 10 run）。利用可能な run が 10 を超えてチャートが表示数を制限している場合、チャート上部に "Showing first 10 runs" と表示されます。
* **Chart type**: ラインプロット、面グラフ、割合の面グラフを切り替えます.

**Grouping**: プロット内で run をグループ化・集約する方法を設定します。
* **Group by**: 列を 1 つ選ぶと、その列で同じ値を持つ run が同じグループにまとめられます。
* **Agg**: 集約。グラフ上の線の値です。オプションはグループの mean、median、min、max です。

**Chart**: パネル、X 軸、Y 軸のタイトルを指定し、凡例の表示・非表示や位置を設定します。

**Legend**: 有効化されている場合、パネルの凡例の見た目をカスタマイズします。
* **Legend**: 各線の凡例に使用するフィールド。
* **Legend template**: 凡例用の完全カスタマイズ可能なテンプレートを定義します。ラインプロット上部のテンプレートや、プロットにマウスオーバーしたときに表示される凡例に、どのテキストや変数を表示するかを正確に指定できます。

**Expressions**: パネルにカスタム計算式を追加します。
* **Y Axis Expressions**: 計算済みメトリクスをグラフに追加します。ログ済みの任意のメトリクスに加え、ハイパーパラメーターのような設定値を使ってカスタムの線を計算できます。
* **X Axis Expressions**: カスタム式を使って計算した値を x 軸に再スケールします。有用な変数としては既定の x 軸を表す \*\*_step\*\* などがあり、サマリー値を参照する構文は `${summary:value}` です。

### セクション内のすべてのラインプロット

セクション内のすべてのラインプロットの既定設定をカスタマイズして、Workspace のラインプロット設定を上書きするには:
1. そのセクションの歯車アイコンをクリックして設定を開きます。
1. 表示されるドロワーで **Data** または **Display preferences** タブを選んで、セクションの既定設定を構成します。各 **Data** 設定の詳細は前のセクション [Individual line plot]({{< relref path="#line-plot-settings" lang="ja" >}}) を参照してください。表示設定の詳細は [セクションレイアウトを設定]({{< relref path="../#configure-section-layout" lang="ja" >}}) を参照してください。

### Workspace 内のすべてのラインプロット 
Workspace 内のすべてのラインプロットの既定設定をカスタマイズするには:
1. ギアと **Settings** ラベルが付いた Workspace の設定をクリックします。
1. **Line plots** をクリックします。
1. 表示されるドロワーで **Data** または **Display preferences** タブを選び、Workspace の既定設定を構成します。
    - 各 **Data** 設定の詳細は、前のセクション [Individual line plot]({{< relref path="#line-plot-settings" lang="ja" >}}) を参照してください。

    - 各 **Display preferences** の詳細は [Workspace display preferences]({{< relref path="../#configure-workspace-layout" lang="ja" >}}) を参照してください。Workspace レベルでは、ラインプロットの既定の **Zooming** の振る舞いを設定できます。この設定は、x 軸キーが一致するラインプロット間でズームを同期するかどうかを制御します。既定では無効です。



## 平均値をプロットで可視化する

複数の異なる実験があり、それらの平均値をプロットで見たい場合は、テーブルの Grouping 機能を使えます。run テーブル上部の "Group" をクリックし、"All" を選ぶと、グラフに平均値が表示されます。

平均化前のグラフは次のとおりです:

{{< img src="/images/app_ui/demo_precision_lines.png" alt="Individual precision lines" >}}

次の画像は、run をグループ化した線を使って run 全体の平均値を表したグラフです。

{{< img src="/images/app_ui/demo_average_precision_lines.png" alt="Averaged precision lines" >}}

## NaN 値をプロットで可視化

`wandb.Run.log()` を使えば、PyTorch のテンソルを含む `NaN` 値もラインプロットに描画できます。例:

```python
with wandb.init() as run:
    # NaN 値をログする
    run.log({"test": float("nan")})
```

{{< img src="/images/app_ui/visualize_nan.png" alt="NaN 値の取り扱い" >}}

## 1 つのチャートで 2 つのメトリクスを比較

{{< img src="/images/app_ui/visualization_add.gif" alt="可視化パネルの追加" >}}

1. 右上の **Add panels** ボタンを選択します。
2. 表示される左側のパネルで、Evaluation ドロップダウンを展開します。
3. **Run comparer** を選択します。


## ラインプロットの色を変更

既定の run の色が比較に役立たない場合があります。そのために、手動で色を変更できる 2 つの方法を用意しています。

{{< tabpane text=true >}}
{{% tab header="run テーブルから" value="run_table" %}}

  各 run は初期化時に既定でランダムな色が割り当てられます。

  {{< img src="/images/app_ui/line_plots_run_table_random_colors.png" alt="run に割り当てられるランダムな色" >}}

  色をクリックするとカラーパレットが表示され、任意の色を手動で選択できます。

  {{< img src="/images/app_ui/line_plots_run_table_color_palette.png" alt="カラーパレット" >}}

{{% /tab %}}

{{% tab header="チャートの凡例設定から" value="legend_settings" %}}

1. 設定を編集したいパネルにマウスオーバーします。
2. 表示される鉛筆アイコンを選択します。
3. **Legend** タブを選びます。

{{< img src="/images/app_ui/plot_style_line_plot_legend.png" alt="ラインプロットの凡例設定" >}}

{{% /tab %}}
{{< /tabpane >}}

## 異なる x 軸で可視化

実験にかかった絶対時間や、実験を実行した日付を見たい場合は、x 軸を切り替えられます。以下は、Step から Relative Time、さらに Wall Time に切り替える例です。

{{< img src="/images/app_ui/howto_use_relative_time_or_wall_time.gif" alt="X-axis time options" >}}

## 面グラフ

ラインプロットの設定の Advanced タブで、プロットスタイルを切り替えると、面グラフや割合の面グラフになります。

{{< img src="/images/app_ui/line_plots_area_plots.gif" alt="Area plot styles" >}}

## ズーム

四角形をドラッグして、縦横同時にズームします。x 軸と y 軸のズームが変更されます。

{{< img src="/images/app_ui/line_plots_zoom.gif" alt="Plot zoom functionality" >}}

## チャートの凡例を非表示にする

トグル 1 つで、ラインプロットの凡例をオフにできます。

{{< img src="/images/app_ui/demo_hide_legend.gif" alt="Hide legend toggle" >}}

## run メトリクス通知を作成
[Automations]({{< relref path="/guides/core/automations" lang="ja" >}}) を使って、指定した条件に run のメトリクスが合致したときにチームへ通知を送れます。Automation は Slack のチャンネルへ投稿したり、webhook を実行できます。

ラインプロットから、表示中のメトリクスに対する[run メトリクス通知]({{< relref path="/guides/core/automations/automation-events.md#run-events" lang="ja" >}})をすばやく作成できます:

1. パネルにマウスオーバーして、ベルアイコンをクリックします。
1. 基本または高度な設定コントロールを使って Automation を設定します。たとえば、run フィルターを適用して適用範囲を絞ったり、絶対しきい値を設定したりできます。

[Automations]({{< relref path="/guides/core/automations" lang="ja" >}}) の詳細をご覧ください。

## CoreWeave のインフラストラクチャーアラートを可視化

機械学習の実験を W&B にログしている間、GPU の障害、熱違反などのインフラストラクチャーアラートを観測できます。[W&B run]({{< relref path="/guides/models/track/runs/_index" lang="ja" >}}) の実行中、[CoreWeave Mission Control](https://www.coreweave.com/mission-control) が計算インフラストラクチャーを監視します。

{{< alert>}}
この機能はプレビュー段階で、CoreWeave クラスター上でトレーニングしている場合にのみ利用できます。利用には W&B の担当者にお問い合わせください。
{{< /alert >}}

エラーが発生すると、CoreWeave はその情報を W&B に送信します。W&B はインフラ情報を Project の Workspace 内の run のプロットに表示します。CoreWeave は一部の問題を自動的に解決しようとし、W&B はその情報を run のページに表示します。

### run のインフラストラクチャー問題を見つける

W&B は SLURM ジョブの問題とクラスターのノード問題の両方を表示します。run 内のインフラエラーを見るには:

1. W&B App で自分の Project に移動します。 
2. **Workspace** タブを選択して Project の Workspace を表示します。
3. インフラストラクチャーの問題を含む run 名を検索して選択します。CoreWeave がインフラ問題を検出した場合、run のプロットに感嘆符付きの赤い縦線が 1 本以上重ねて表示されます。 
4. プロット上の問題を選択するか、ページ右上の **Issues** ボタンを選択します。CoreWeave が報告した各問題の一覧がドロワーに表示されます。 

{{< alert title="Tip" >}}
インフラ問題のある run をひと目で確認するには、W&B の Workspace に **Issues** 列をピン留めします。列のピン留め方法の詳細は、[run の表示方法をカスタマイズ]({{< relref path="/guides/models/track/runs/#customize-how-runs-are-displayed" lang="ja" >}}) を参照してください。
{{< /alert >}}

ドロワー上部の **Overall Grafana view** は、run のシステムレベルの詳細を含む SLURM ジョブの Grafana ダッシュボードへ遷移します。**Issues summary** には、SLURM ジョブが CoreWeave Mission Control に報告した根本的なエラーが記載されています。サマリーには、CoreWeave による自動復旧の試行内容も記載されます。

{{< img src="/images/app_ui/cw_wb_observability.png" >}}

**All Issues** には、実行中に発生したすべての問題が新しい順で一覧表示されます。この一覧にはジョブの問題とノードの問題のアラートが含まれます。各アラートには、問題名、発生時刻、その問題の Grafana ダッシュボードへのリンク、問題の概要が含まれます。

次の表は、インフラストラクチャー問題の各カテゴリにおけるアラート例です。

| カテゴリ | アラート例 |
| -------- | ------------- |
| Node Availability & Readiness | `KubeNodeNotReadyHGX`, `NodeExtendedDownTime` |
| GPU/Accelerator Errors | `GPUFallenOffBusHGX`, `GPUFaultHGX`, `NodeTooFewGPUs` |
| Hardware Errors | `HardwareErrorFatal`, `NodeRAIDMemberDegraded` |
| Networking & DNS | `NodeDNSFailureHGX`, `NodeEthFlappingLegacyNonGPU` |
| Power, Cooling, and Management | `NodeCPUHZThrottle`, `RedfishDown` |
| DPU & NVSwitch | `DPUNcoreVersionBelowDesired`, `NVSwitchFaultHGX` |
| Miscellaneous | `NodePCISpeedRootGBT`, `NodePCIWidthRootSMC` |

エラータイプの詳細は、[CoreWeave Docs の SLURM Job Metrics](https://docs.coreweave.com/docs/observability/managed-grafana/sunk/slurm-job-metrics#job-info-alerts#job-info-alerts) を参照してください。

### インフラストラクチャー問題のデバッグ

W&B で作成する各 run は、CoreWeave の単一の SLURM ジョブに対応します。失敗したジョブの [Grafana](https://grafana.com/) ダッシュボードを開いたり、特定のノードに関する詳細を確認できます。**Issues** ドロワーの **Overview** セクション内のリンクから、SLURM ジョブの Grafana ダッシュボードに移動できます。**All Issues** のドロップダウンを展開すると、ジョブとノードの両方の問題と、それぞれの Grafana ダッシュボードを確認できます。 

{{< alert title="Note" >}}
Grafana ダッシュボードは、CoreWeave アカウントを持つ W&B ユーザーのみが利用できます。W&B に連絡して、自組織の W&B と Grafana の連携を設定してください。
{{< /alert >}}

問題の種類に応じて、SLURM ジョブの設定を調整したり、ノードの状態を調査したり、ジョブを再起動したり、その他必要な対応を行ってください。

Grafana における CoreWeave の SLURM ジョブの詳細は、[CoreWeave Docs の Slurm/Job Metrics](https://docs.coreweave.com/docs/observability/managed-grafana/sunk/slurm-job-metrics#job-info-alerts) を参照してください。ジョブアラートの詳細は [Job info: alerts](https://docs.coreweave.com/docs/observability/managed-grafana/sunk/slurm-job-metrics#job-info-alerts#job-info-alerts) を参照してください。