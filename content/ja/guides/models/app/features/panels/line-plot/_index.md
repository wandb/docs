---
title: 折れ線グラフ
description: メトリクスを可視化し、軸をカスタマイズし、プロット上で複数のラインを比較します
cascade:
- url: /ja/guides/app/features/panels/line-plot/:filename
menu:
  default:
    identifier: ja-guides-models-app-features-panels-line-plot-_index
    parent: panels
url: /ja/guides/app/features/panels/line-plot
weight: 10
---

ラインプロットは、**wandb.log()** でメトリクスを時間経過とともにプロットするとデフォルトで表示されます。複数のラインを同じプロットで比較したり、カスタム軸を計算したり、ラベルをリネームしたりするために、チャート設定をカスタマイズできます。

{{< img src="/images/app_ui/line_plot_example.png" alt="" >}}

## ラインプロット設定を編集する

このセクションでは、個々のラインプロットパネル、セクション内のすべてのラインプロットパネル、またはワークスペース内のすべてのラインプロットパネルの設定を編集する方法を紹介します。

{{% alert %}}
カスタムの x 軸を使用したい場合は、同じ `wandb.log()` の呼び出しで y 軸と一緒にログを取るようにしてください。
{{% /alert %}}

### 個別のラインプロット

個々のラインプロットの設定は、セクションまたはワークスペースのラインプロット設定を上書きします。ラインプロットをカスタマイズするには:

1. パネルの上にマウスをホバーさせ、ギアアイコンをクリックします。
2. 表示されるモーダル内で、[設定]({{< relref path="#line-plot-settings" lang="ja" >}}) を編集するタブを選択します。
3. **適用** をクリックします。

#### ラインプロット設定

次の設定をラインプロットに設定できます:

**データ**: プロットのデータ表示の詳細を設定します。
* **X**: X 軸に使用する値を選択します (デフォルトは **Step** です)。X 軸を **相対時間** に変更したり、W&B でログを取った値に基づいてカスタム軸を選択したりできます。
  * **相対時間 (Wall)** はプロセス開始以降の時計時間で、もし 1 日後に run を再開して何かをログした場合、それは 24 時間にプロットされます。
  * **相対時間 (プロセス)** は実行中のプロセス内の時間で、もし 10 秒間 run を実行し、1 日後に再開した場合、そのポイントは 10 秒にプロットされます。
  * **ウォール時間** はグラフ上の最初の run 開始からの経過時間を示します。
  * **Step** はデフォルトで `wandb.log()` が呼び出されるたびに増加し、モデルからログされたトレーニングステップの数を反映することになっています。
* **Y**: メトリクスや時間経過とともに変化するハイパーパラメーターなど、ログに取られた値から1つ以上の y 軸を選択します。
* **X軸** および **Y軸** の最小値と最大値 (オプション)。
* **ポイント集計メソッド**. **ランダムサンプリング** (デフォルト) または **フルフェデリティ**。詳細は [サンプリング]({{< relref path="sampling.md" lang="ja" >}}) を参照。
* **スムージング**: ラインプロットのスムージングを変更します。デフォルトは **時間加重EMA** です。その他の値には **スムージングなし**, **ランニング平均**, および **ガウシアン** があります。
* **外れ値**: 外れ値を除外して、デフォルトのプロット最小値および最大値スケールを再設定します。
* **最大 run またはグループ数**: この数値を増やすことで、ラインプロットに一度により多くのラインを表示します。デフォルトは 10 run です。チャートの一番上に "最初の 10 run を表示中" というメッセージが表示され、利用可能な run が 10 個を超える場合、チャートが表示できる数を制約していることが分かります。
* **チャートタイプ**: ラインプロット、エリアプロット、および パーセンテージエリアプロットの中で切り替えます。

**グルーピング**: プロット内で run をどのようにグループ化し集計するかを設定します。
* **グループ化基準**: 列を選択し、その列に同じ値を持つすべての run がグループ化されます。
* **Agg**: 集計— グラフ上のラインの値です。オプションはグループの平均、中央値、最小、最大です。

**チャート**: パネル、X軸、Y軸のタイトルを指定し、凡例を表示または非表示に設定し、その位置を設定します。

**凡例**: パネルの凡例の外観をカスタマイズします、もし有効化されている場合。
* **凡例**: プロットの各ラインに対する凡例のフィールド、それぞれのラインのプロット内の凡例。
* **凡例テンプレート**: テンプレートの上部に表示される凡例およびマウスオーバー時にプロットに表示される伝説で, 表示したい具体的なテキストおよび変数を定義します。

**式**: パネルにカスタム計算式を追加します。
* **Y軸式**: グラフに計算されたメトリクスを追加。ログされたメトリクスやハイパーパラメーターのような設定値を使用してカスタムラインを計算することができます。
* **X軸式**: 計算された値を使用して x 軸を再スケーリングします。有用な変数には、デフォルトの x 軸の **_step_** などが含まれ、サマリー値を参照するための構文は `${summary:value}` です。

### セクション内のすべてのラインプロット

ワークスペースの設定を上書きして、セクション内のすべてのラインプロットのデフォルトの設定をカスタマイズするには:
1. セクションのギアアイコンをクリックして設定を開きます。
2. 表示されるモーダル内で、**データ** または **表示設定** タブを選択して、セクションのデフォルトの設定を構成します。各 **データ** 設定の詳細については、前述のセクション [個別のラインプロット]({{< relref path="#line-plot-settings" lang="ja" >}}) を参照してください。各表示設定の詳細については、[セクションレイアウトの構成]({{< relref path="../#configure-section-layout" lang="ja" >}}) を参照してください。

### ワークスペース内のすべてのラインプロット
ワークスペース内のすべてのラインプロットのデフォルト設定をカスタマイズするには:
1. **設定** というラベルの付いたギアがあるワークスペースの設定をクリックします。
2. **ラインプロット** をクリックします。
3. 表示されるモーダル内で、**データ** または **表示設定** タブを選択して、ワークスペースのデフォルト設定を構成します。
    - 各 **データ** 設定の詳細については、前述のセクション [個別のラインプロット]({{< relref path="#line-plot-settings" lang="ja" >}}) を参照してください。

    - 各 **表示設定** セクションの詳細については、[ワークスペース表示設定]({{< relref path="../#configure-workspace-layout" lang="ja" >}}) を参照してください。ワークスペースレベルで、ラインプロットのデフォルト **ズーム** 振る舞いを構成できます。この設定は、x 軸キーが一致するラインプロット間でズームの同期を制御します。デフォルトでは無効です。

## プロット上で平均値を可視化する

複数の異なる実験があり、その値の平均をプロットで見たい場合は、テーブルのグルーピング機能を使用できます。runテーブルの上部で "グループ化" をクリックし、"すべて" を選択してグラフに平均値を表示します。

以下は平均化する前のグラフの例です:

{{< img src="/images/app_ui/demo_precision_lines.png" alt="" >}}

次の画像は、グループ化されたラインを使用して run における平均値を示すグラフです。

{{< img src="/images/app_ui/demo_average_precision_lines.png" alt="" >}}

## プロット上で NaN 値を可視化する

`wandb.log` を使用して、PyTorch テンソルを含む `NaN` 値をラインプロットでプロットすることもできます。例えば:

```python
wandb.log({"test": [..., float("nan"), ...]})
```

{{< img src="/images/app_ui/visualize_nan.png" alt="" >}}

## 2 つのメトリクスを 1 つのチャートで比較する

{{< img src="/images/app_ui/visualization_add.gif" alt="" >}}

1. ページの右上隅にある **パネルを追加** ボタンを選択します。
2. 表示される左側のパネルで評価のドロップダウンを展開します。
3. **Run comparer** を選択します。

## ラインプロットの色を変更する

時々、run のデフォルトの色が比較には適していないことがあります。この問題を解決するために、wandb は手動で色を変更できる2つの方法を提供しています。

{{< tabpane text=true >}}
{{% tab header="runテーブルから" value="run_table" %}}

  各 run は初期化時にデフォルトでランダムな色が割り当てられます。

  {{< img src="/images/app_ui/line_plots_run_table_random_colors.png" alt="Random colors given to runs" >}}

  どの色をクリックすると、手動で選択できるカラーパレットが表示されます。

  {{< img src="/images/app_ui/line_plots_run_table_color_palette.png" alt="The color palette" >}}

{{% /tab %}}

{{% tab header="チャート凡例設定から" value="legend_settings" %}}

1. 設定を編集したいパネルにマウスをホバーさせます。
2. 表示される鉛筆アイコンを選択します。
3. **凡例** タブを選択します。

{{< img src="/images/app_ui/plot_style_line_plot_legend.png" alt="" >}}

{{% /tab %}}
{{< /tabpane >}}

## 異なる x 軸で可視化する

実験がかかった絶対時間を見たい場合や、実験が実行された日を見たい場合は、x 軸を切り替えることができます。ここでは、ステップから相対時間、そして壁時間に切り替える例を示します。

{{< img src="/images/app_ui/howto_use_relative_time_or_wall_time.gif" alt="" >}}

## エリアプロット

詳細設定タブでさまざまなプロットスタイルをクリックすると、エリアプロットまたはパーセンテージエリアプロットを取得できます。

{{< img src="/images/app_ui/line_plots_area_plots.gif" alt="" >}}

## ズーム

直線をクリックしてドラッグすると垂直および水平方向に同時にズームします。これにより、x軸とy軸のズームが変更されます。

{{< img src="/images/app_ui/line_plots_zoom.gif" alt="" >}}

## チャートの凡例の非表示

この簡単なトグルでラインプロットの凡例をオフにできます:

{{< img src="/images/app_ui/demo_hide_legend.gif" alt="" >}}