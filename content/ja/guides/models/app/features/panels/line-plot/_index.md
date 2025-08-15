---
title: 折れ線グラフ
description: メトリクスを可視化し、軸をカスタマイズし、複数のラインをプロット上で比較できます
cascade:
- url: guides/app/features/panels/line-plot/:filename
menu:
  default:
    identifier: ja-guides-models-app-features-panels-line-plot-_index
    parent: panels
url: guides/app/features/panels/line-plot
weight: 10
---

Line プロットは、`wandb.Run.log()` でメトリクスを時間とともに記録するとデフォルトで表示されます。チャート設定をカスタマイズすることで、同じプロット上で複数ラインを比較したり、カスタム軸を計算したり、ラベル名を変更したりできます。

{{< img src="/images/app_ui/line_plot_example.png" alt="Line plot example" >}}

## Line プロットの設定を編集する

このセクションでは、個別の Line プロットパネル、セクション内のすべての Line プロットパネル、またはワークスペース全体の Line プロットパネルの設定を編集する方法を紹介します。

{{% alert %}}
カスタム X 軸を使いたい場合は、Y 軸と同じ `wandb.Run.log()` の呼び出しでログに記録されていることを確認してください。
{{% /alert %}} 

### 個別の Line プロット
個別の Line プロット設定は、セクションやワークスペースの設定よりも優先されます。Line プロットをカスタマイズするには:

1. パネルにマウスオーバーし、ギア アイコンをクリックします。
1. 表示されるドロワー内で、タブを選択し[設定]({{< relref path="#line-plot-settings" lang="ja" >}})を編集します。
1. **適用**をクリックします。

#### Line プロットの設定
Line プロットでは、次の設定を行うことができます。

**データ**: プロットで表示するデータの詳細を設定します。
* **X 軸**: X 軸として使う値を選択します（デフォルトは **Step**）。X軸を **相対時間** に変更したり、W&B で記録した値をもとにカスタム軸を選択することもできます。また、X軸のスケールや範囲も設定できます。
  * **Relative Time (Wall)** はプロセス開始からの実時間で、run を開始して一日後に再開した場合、その点は24時間後にプロットされます。
  * **Relative Time (Process)** は実行プロセス内の時間で、run を開始して10秒間実行し、翌日再開しても、その点は10秒時点にプロットされます。
  * **Wall Time** はグラフ上の最初の run 開始から経過した分数です。
  * **Step** はデフォルトで `wandb.Run.log()` が呼ばれるたびにインクリメントされ、モデルで記録したトレーニングステップ数を表します。
* **Y 軸**: ログに記録された値の中から、1つまたは複数のY軸を選択します。これには、時間とともに変化するメトリクスやハイパーパラメーターも含みます。X軸のスケールや範囲もここで設定できます。
* **ポイント集約方法**: **ランダムサンプリング**（デフォルト）または **Full fidelity** を選べます。[サンプリング]({{< relref path="sampling.md" lang="ja" >}})を参照してください。
* **スムージング**: 線グラフのスムーズ化レベルを変更します。デフォルトは **Time weighted EMA**。他には **スムージングなし**、**移動平均**、**ガウシアン**があります。
* **外れ値**: 外れ値を除外してプロットのmin/maxスケールを再調整します。
* **最大 run またはグループ数**: この数値を増やすと同時により多くのラインを表示できます。デフォルトは10 run です。run が10個以上ある場合、「最初の10 run を表示中」とメッセージが出ます。
* **チャートタイプ**: Line プロット、エリア（面）プロット、パーセンテージエリアプロットを切り替えられます。

**グルーピング**: プロット内で run をどのようにグループ化・集約するかを設定します。
* **Group by**: 列を選択すると、同じ値を持つ run がグループ化されます。
* **Agg**: グラフのライン値の集計方法。平均、中央値、最小、最大などから選択します。

**チャート**: パネルや X軸・Y軸のタイトル、凡例の表示/非表示と位置などを指定できます。

**凡例（レジェンド）**: パネルの凡例が有効な場合、その表示をカスタマイズできます。
* **Legend**: プロット内の各ラインごとに凡例に表示するフィールド。
* **Legend template**: 完全にカスタマイズ可能なテンプレートを定義できます。ラインプロット上部やマウスオーバー時の凡例で表示したいテキストや変数を指定できます。

**Expressions**: パネルにカスタム計算式を追加できます。
* **Y Axis Expressions**: 計算したメトリクスをグラフに追加します。ログに記録されたメトリクスや設定値（ハイパーパラメーターなど）を使ってカスタムラインを描くことができます。
* **X Axis Expressions**: カスタム式で計算した値でX軸を再スケーリングできます。よく使う変数として **_step_**（デフォルトのX軸）、サマリー値への参照は `${summary:value}` の形式です。

### セクション内のすべての Line プロット

セクション内のすべての Line プロットに対してデフォルト設定をカスタマイズしたい場合（ワークスペースの設定を上書き）:
1. セクションのギア アイコンをクリックして設定を開きます。
1. ドロワー内で **データ** または **表示設定** タブを選び、セクションのデフォルト設定を変更します。各 **データ** 設定の詳細は前述の[個別の Line プロット]({{< relref path="#line-plot-settings" lang="ja" >}})を参照してください。表示設定の詳細は[セクションレイアウトの設定]({{< relref path="../#configure-section-layout" lang="ja" >}})を参照してください。

### ワークスペース内のすべての Line プロット
ワークスペース内のすべての Line プロットに対するデフォルト設定をカスタマイズする手順:
1. ラベル **Settings** の付いたワークスペースの設定（ギアアイコン）をクリックします。
1. **Line plots** をクリックします。
1. ドロワー内で **データ** または **表示設定** タブを選び、ワークスペースのデフォルト設定を構成します。
    - 各 **データ** 設定の詳細は前述の[個別の Line プロット]({{< relref path="#line-plot-settings" lang="ja" >}})を参照してください。

    - **表示設定** セクションの詳細は[ワークスペース表示設定]({{< relref path="../#configure-workspace-layout" lang="ja" >}})を参照してください。ワークスペースレベルでは、Line プロットのデフォルト **ズーム** 振る舞いも設定可能です。この設定は、同じX軸キーを持つLineプロット間でズームを同期させるかどうかを制御します（デフォルトは無効）。



## プロットで平均値を可視化する

複数の異なる experiment を比較し、それらの平均値をプロットに表示したい場合は、テーブルのグルーピング機能が使えます。run テーブルの上にある「グループ」をクリックし、「All」を選択すると、グラフで平均値が表示されます。

平均前のグラフは次のとおりです:

{{< img src="/images/app_ui/demo_precision_lines.png" alt="Individual precision lines" >}}

以下の画像は、グループ化したラインを使って runs 間の平均値を示すグラフ例です。

{{< img src="/images/app_ui/demo_average_precision_lines.png" alt="Averaged precision lines" >}}

## NaN 値をプロットで可視化する

`wandb.Run.log()` を使って、PyTorch テンソルなども含む `NaN` 値のプロットが可能です。例:

```python
with wandb.init() as run:
    # NaN値をログに記録する
    run.log({"test": float("nan")})
```

{{< img src="/images/app_ui/visualize_nan.png" alt="NaN value handling" >}}

## 1つのチャート上で2つのメトリクスを比較する

{{< img src="/images/app_ui/visualization_add.gif" alt="Adding visualization panels" >}}

1. ページ右上の **Add panels** ボタンを選択します。
2. 左側に表示されるパネルの Evaluation ドロップダウンを展開します。
3. **Run comparer** を選択します。


## Line プロットの色を変更する

デフォルトの run 色が比較に適さない場合があります。そのため wandb では、手動で色を変更できる2つの手段を提供しています。

{{< tabpane text=true >}}
{{% tab header="run テーブルから" value="run_table" %}}

  各 run には初期化時にデフォルトでランダムな色が割り当てられます。

  {{< img src="/images/app_ui/line_plots_run_table_random_colors.png" alt="Random colors given to runs" >}}

  色をクリックするとカラーパレットが表示され、手動で好きな色を選択することができます。

  {{< img src="/images/app_ui/line_plots_run_table_color_palette.png" alt="The color palette" >}}

{{% /tab %}}

{{% tab header="チャートの凡例設定から" value="legend_settings" %}}

1. 設定を編集したいパネルにマウスを乗せます。
2. 表示される鉛筆アイコンを選択します。
3. **Legend** タブを選びます。

{{< img src="/images/app_ui/plot_style_line_plot_legend.png" alt="Line plot legend settings" >}}

{{% /tab %}}
{{< /tabpane >}}

## 異なるX軸で可視化する

experiment のかかった絶対時間や、どの日に実行されたかを見たい場合は、X軸を切り替えることができます。例えば、ステップから相対時間、そして壁時間への切り替え例です。

{{< img src="/images/app_ui/howto_use_relative_time_or_wall_time.gif" alt="X-axis time options" >}}

## エリアプロット

Line プロット設定の「詳細」タブで異なるプロットスタイルをクリックすると、エリアプロットやパーセンテージエリアプロットが選べます。

{{< img src="/images/app_ui/line_plots_area_plots.gif" alt="Area plot styles" >}}

## ズーム

長方形をドラッグすると、縦横同時にズーム可能です。これでX軸・Y軸ともにズーム範囲が変更されます。

{{< img src="/images/app_ui/line_plots_zoom.gif" alt="Plot zoom functionality" >}}

## チャート凡例の非表示

簡単なトグルで、Line プロットの凡例をオフにできます:

{{< img src="/images/app_ui/demo_hide_legend.gif" alt="Hide legend toggle" >}}

## run メトリクス通知を作成する
[Automations]({{< relref path="/guides/core/automations" lang="ja" >}}) を利用すれば、指定した条件に run メトリクスが達したときにチームへ通知できます。自動化処理で Slack チャンネルへ投稿したり、Webhook を起動可能です。

Line プロットから、可視化しているメトリクスに対してすぐに [run メトリクス通知]({{< relref path="/guides/core/automations/automation-events.md#run-events" lang="ja" >}}) を作成できます:

1. パネルにマウスを乗せ、ベルアイコンをクリックします。
1. 基本または詳細設定でオートメーションを設定します。たとえば、run フィルターで適用範囲を絞ったり、絶対値のしきい値を設定したりできます。

[Automations]({{< relref path="/guides/core/automations" lang="ja" >}}) の詳細はこちら。

## CoreWeave インフラストラクチャ警告の可視化

W&B へログをとった機械学習 experiments 中に、GPU 障害やサーマル違反などのインフラストラクチャ警告を観測できます。[W&B run]({{< relref path="/guides/models/track/runs/_index" lang="ja" >}}) の間、[CoreWeave Mission Control](https://www.coreweave.com/mission-control) があなたの計算インフラをモニタリングします。

{{< alert>}}
この機能はプレビュー版で、CoreWeave クラスター上でのトレーニング時のみ利用可能です。W&B 担当者までご連絡ください。
{{< /alert >}}

エラーが発生すると、CoreWeave がその情報を W&B へ送ります。W&B はインフラの情報を run のプロットやプロジェクトのワークスペースに反映します。CoreWeave が自動で修正を試みる場合もあり、その情報も run ページで可視化されます。

### run でインフラ障害を見つける

W&B は SLURM ジョブの問題やクラスター ノードの問題両方を表示します。run でインフラエラーを見るには：

1. W&B App でプロジェクトにアクセスします。
2. **Workspace** タブを選択してプロジェクトのワークスペースを表示します。
3. インフラ障害を含む run の名前を検索・選択します。CoreWeave が障害を検出した場合、run のプロット上に赤い縦線と感嘆符がオーバーレイ表示されます。
4. プロット上の障害を選択するか、ページ右上の **Issues** ボタンを選ぶと、CoreWeave が報告した各障害のリストが表示されるドロワーが現れます。

{{< alert title="ヒント" >}}
インフラ障害のある run を一目で確認したい場合は、**Issues** カラムをワークスペースにピン留めしましょう。ピン留め方法の詳細は[run の表示カスタマイズ]({{< relref path="/guides/models/track/runs/#customize-how-runs-are-displayed" lang="ja" >}})を参照してください。
{{< /alert >}}

ドロワーの最上部にある **Overall Grafana view** をクリックすると、その SLURM ジョブの Grafana ダッシュボード（システムレベルの詳細情報）に遷移します。**Issues summary** には、SLURM ジョブが CoreWeave Mission Control に報告した根本エラーの概要や、CoreWeave による自動修正の試み内容が記載されます。

{{< img src="/images/app_ui/cw_wb_observability.png" >}}

**All Issues** には、run 実行中に発生したすべての障害が新しい順に一覧表示されます。ジョブ障害やノード障害の警告も含まれます。各警告には、障害名・発生時刻・Grafana ダッシュボードへのリンク・障害の簡易説明が含まれます。

下記表はインフラ障害カテゴリごとの警告例です:

| カテゴリ | 警告の例 |
| -------- | ------------- |
| ノードの可用性・準備性 | `KubeNodeNotReadyHGX`, `NodeExtendedDownTime` |
| GPU/アクセラレータ エラー | `GPUFallenOffBusHGX`, `GPUFaultHGX`, `NodeTooFewGPUs` |
| ハードウェア エラー | `HardwareErrorFatal`, `NodeRAIDMemberDegraded` |
| ネットワーク・DNS | `NodeDNSFailureHGX`, `NodeEthFlappingLegacyNonGPU` |
| 電源・冷却・管理 | `NodeCPUHZThrottle`, `RedfishDown` |
| DPU・NVSwitch | `DPUNcoreVersionBelowDesired`, `NVSwitchFaultHGX` |
| その他 | `NodePCISpeedRootGBT`, `NodePCIWidthRootSMC` |

エラータイプの詳細は [CoreWeave Docs の SLURM Job Metrics](https://docs.coreweave.com/docs/observability/managed-grafana/sunk/slurm-job-metrics#job-info-alerts#job-info-alerts) を参照してください。

### インフラ障害のデバッグ

W&B で作成する各 run は、CoreWeave の一つの SLURM ジョブに対応します。失敗したジョブの [Grafana](https://grafana.com/) ダッシュボードや、特定ノードの追加情報も確認可能です。 **Issues** ドロワー内「Overview」セクションのリンクが SLURM ジョブ Grafana ダッシュボードに繋がります。 **All Issues** ドロップダウンを展開すると、ジョブ・ノード両方の問題とそのダッシュボードが確認できます。

{{< alert title="備考" >}}
Grafana ダッシュボードは CoreWeave アカウントを持つ W&B ユーザーのみ利用可能です。ご利用には W&B までご連絡のうえ、お使いの組織で Grafana を設定してください。
{{< /alert >}}

障害に応じて、SLURM ジョブの設定を見直したり、ノードの状態を調査したり、ジョブを再起動したり、その他必要な対応を行ってください。

CoreWeave の Grafana 上の SLURM ジョブ詳細については、[CoreWeave Docs: Slurm/Job Metrics](https://docs.coreweave.com/docs/observability/managed-grafana/sunk/slurm-job-metrics#job-info-alerts)を参照してください。[Job info: alerts](https://docs.coreweave.com/docs/observability/managed-grafana/sunk/slurm-job-metrics#job-info-alerts#job-info-alerts) でジョブの警告詳細が確認できます。