---
title: 実験からプロットを作成し、トラッキングする
description: 機械学習実験からプロットを作成して追跡します。
menu:
  default:
    identifier: ja-guides-models-track-log-plots
    parent: log-objects-and-media
---

`wandb.plot` のメソッドを使うことで、`wandb.Run.log()` でグラフを記録できます。これにはトレーニング中に時間とともに変化するグラフも含まれます。独自のチャート作成フレームワークについては、[カスタムチャートのチュートリアル]({{< relref path="/guides/models/app/features/custom-charts/walkthrough.md" lang="ja" >}}) をご覧ください。

### 基本チャート

これらのシンプルなチャートは、メトリクスや結果の基本的な可視化を簡単に作成できます。

{{< tabpane text=true >}}
    {{% tab header="Line" %}}

カスタム折れ線グラフ（任意の軸に沿った連結・順序付きの点のリスト）を記録します。

```python
import wandb

with wandb.init() as run:
    data = [[x, y] for (x, y) in zip(x_values, y_values)]
    table = wandb.Table(data=data, columns=["x", "y"])
    run.log(
        {
            "my_custom_plot_id": wandb.plot.line(
                table, "x", "y", title="Custom Y vs X Line Plot"
            )
        }
    )
```

2つの次元で曲線を記録することができます。もし2つの値のリスト同士をプロットする場合は、リストの値の数が完全に一致する必要があります。例えば、各点は x, y の両方が必要です。

{{< img src="/images/track/line_plot.png" alt="Custom line plot" >}}

[アプリで見る](https://wandb.ai/wandb/plots/reports/Custom-Line-Plots--VmlldzoyNjk5NTA)

[コードを実行](https://tiny.cc/custom-charts)   
    {{% /tab %}}
    {{% tab header="Scatter" %}}

カスタム散布図（任意の x, y 軸上に点 (x, y) のリスト）を記録します。

```python
import wandb

with wandb.init() as run:
    data = [[x, y] for (x, y) in zip(class_x_scores, class_y_scores)]
    table = wandb.Table(data=data, columns=["class_x", "class_y"])
    run.log({"my_custom_id": wandb.plot.scatter(table, "class_x", "class_y")})
```

2つの次元上に散布点を記録できます。2つの値リスト同士をプロットする際は、リストの長さが完全に一致する必要があります。例えば、各点はxとyの両方が必要です。

{{< img src="/images/track/demo_scatter_plot.png" alt="Custom scatter plot" >}}

[アプリで見る](https://wandb.ai/wandb/plots/reports/Custom-Scatter-Plots--VmlldzoyNjk5NDQ)

[コードを実行](https://tiny.cc/custom-charts)    
    {{% /tab %}}
    {{% tab header="Bar" %}}

カスタム棒グラフ（ラベル付き値のリストを棒グラフで）を、数行のコードで簡単に記録できます。

```python
import wandb

with wandb.init() as run:
    data = [[label, val] for (label, val) in zip(labels, values)]
    table = wandb.Table(data=data, columns=["label", "value"])
    run.log(
        {
        "my_bar_chart_id": wandb.plot.bar(
            table, "label", "value", title="Custom Bar Chart"
        )
    }
)
```

任意の棒グラフを記録できます。ラベルと値のリストの数は必ず一致させてください。各データ点には両方が必要です。

{{< img src="/images/track/basic_charts_bar.png" alt="Custom bar chart" >}}

[アプリで見る](https://wandb.ai/wandb/plots/reports/Custom-Bar-Charts--VmlldzoyNzExNzk)

[コードを実行](https://tiny.cc/custom-charts)    
    {{% /tab %}}
    {{% tab header="Histogram" %}}

カスタムヒストグラム（値のリストをビン分けして出現数/頻度で可視化）も数行のコードで記録できます。例えば、予測の信頼度スコア（`scores`）の分布を可視化したい場合:

```python
import wandb

with wandb.init() as run:
    data = [[s] for s in scores]
    table = wandb.Table(data=data, columns=["scores"])
    run.log({"my_histogram": wandb.plot.histogram(table, "scores", title="Histogram")})
```

任意のヒストグラムを記録できます。`data` は行列（2次元配列）に対応したリストのリストです。

{{< img src="/images/track/demo_custom_chart_histogram.png" alt="Custom histogram" >}}

[アプリで見る](https://wandb.ai/wandb/plots/reports/Custom-Histograms--VmlldzoyNzE0NzM)

[コードを実行](https://tiny.cc/custom-charts)    
    {{% /tab %}}
    {{% tab header="Multi-line" %}}

複数の折れ線や異なる x-y 座標のリストを1つの x-y 軸上にプロットします。

```python
import wandb
with wandb.init() as run:
    run.log(
        {
            "my_custom_id": wandb.plot.line_series(
                xs=[0, 1, 2, 3, 4],
                ys=[[10, 20, 30, 40, 50], [0.5, 11, 72, 3, 41]],
            keys=["metric Y", "metric Z"],
            title="Two Random Metrics",
            xname="x units",
        )
    }
)
```

xとyの点の数は完全に一致させてください。複数の y 値リストに対して1つの x 値リスト、または各 y 値リストごとに別々の x リストを渡すことが可能です。

{{< img src="/images/track/basic_charts_histogram.png" alt="Multi-line plot" >}}

[アプリで見る](https://wandb.ai/wandb/plots/reports/Custom-Multi-Line-Plots--VmlldzozOTMwMjU)    
    {{% /tab %}}
{{< /tabpane >}}



### モデル評価チャート

これらのプリセットチャートは `wandb.plot()` メソッドで簡単にスクリプトから直接記録でき、UI 上で必要な情報をすぐに確認できます。

{{< tabpane text=true >}}
    {{% tab header="Precision-recall curves" %}}

[PR曲線](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve) を1行で作成：

```python
import wandb
with wandb.init() as run:
    # ground_truth は正解ラベルリスト、predictions は予測スコアのリストです
    # 例: ground_truth = [0, 1, 1, 0], predictions = [0.1, 0.4, 0.35, 0.8]
    ground_truth = [0, 1, 1, 0]
    predictions = [0.1, 0.4, 0.35, 0.8]
    run.log({"pr": wandb.plot.pr_curve(ground_truth, predictions)})
```

以下の情報がコードで取得できればいつでもこのチャートが記録できます：

* モデルの予測スコア（`predictions`）のリスト
* 対応する正解ラベル（`ground_truth`）
* （任意）ラベル・クラス名のリスト（例えば `labels=["cat", "dog", "bird"...]` で0=cat, 1=dog, 2=birdなど）
* （任意）プロットで可視化したいラベルのサブセット

{{< img src="/images/track/model_eval_charts_precision_recall.png" alt="Precision-recall curve" >}}

[アプリで見る](https://wandb.ai/wandb/plots/reports/Plot-Precision-Recall-Curves--VmlldzoyNjk1ODY)

[コードを実行](https://colab.research.google.com/drive/1mS8ogA3LcZWOXchfJoMrboW3opY1A8BY?usp=sharing)    
    {{% /tab %}}
    {{% tab header="ROC curves" %}}

[ROC曲線](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve) を1行で作成：

```python
import wandb

with wandb.init() as run:
    # ground_truth は正解ラベルリスト、predictions は予測スコアのリストです
    # 例: ground_truth = [0, 1, 1, 0], predictions = [0.1, 0.4, 0.35, 0.8]
    ground_truth = [0, 1, 1, 0]
    predictions = [0.1, 0.4, 0.35, 0.8]
    run.log({"roc": wandb.plot.roc_curve(ground_truth, predictions)})
```

以下の情報がコードで取得できればいつでもこのチャートが記録できます：

* モデルの予測スコア（`predictions`）のリスト
* 対応する正解ラベル（`ground_truth`）
* （任意）ラベル・クラス名のリスト（例えば `labels=["cat", "dog", "bird"...]` で0=cat, 1=dog, 2=birdなど）
* （任意）プロットで可視化したいラベルのサブセット

{{< img src="/images/track/demo_custom_chart_roc_curve.png" alt="ROC curve" >}}

[アプリで見る](https://wandb.ai/wandb/plots/reports/Plot-ROC-Curves--VmlldzoyNjk3MDE)

[コードを実行](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Plot_ROC_Curves_with_W%26B.ipynb)    
    {{% /tab %}}
    {{% tab header="Confusion matrix" %}}

マルチクラスの[混同行列](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html)を1行で作成：

```python
import wandb

cm = wandb.plot.confusion_matrix(
    y_true=ground_truth, preds=predictions, class_names=class_names
)

with wandb.init() as run:
    run.log({"conf_mat": cm})
```

以下の情報が取得できれば記録できます：

* モデルが予測したラベル（`preds`）または正規化済み確率スコア（`probs`）。確率スコアは（サンプル数, クラス数）の形状である必要があります。「確率」か「予測ラベル」のどちらか片方のみ渡してください。
* 対応する正解ラベル（`y_true`）
* クラス名の全リスト（例：`class_names=["cat", "dog", "bird"]` で 0=cat, 1=dog, 2=bird）

{{< img src="/images/experiments/confusion_matrix.png" alt="Confusion matrix" >}}

​[アプリで見る](https://wandb.ai/wandb/plots/reports/Confusion-Matrix--VmlldzozMDg1NTM)​

​[コードを実行](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Log_a_Confusion_Matrix_with_W%26B.ipynb)    
    {{% /tab %}}
{{< /tabpane >}}


### インタラクティブなカスタムチャート

完全なカスタマイズが必要な場合は、[カスタムチャートプリセット]({{< relref path="/guides/models/app/features/custom-charts/walkthrough.md" lang="ja" >}}) を編集または新規作成し、チャートを保存してください。チャートIDを使えば、そのカスタムプリセットに直接スクリプトからデータを記録できます。

```python
import wandb
# プロットしたい列でTableを作成
table = wandb.Table(data=data, columns=["step", "height"])

# Tableの列とチャートのフィールドをマッピング
fields = {"x": "step", "value": "height"}

# このTableで新しいカスタムチャートプリセットにデータを入れる
# 保存済みプリセットを使うには vega_spec_name を変更
# タイトルを編集したい場合は string_fields を変える
my_custom_chart = wandb.plot_table(
    vega_spec_name="carey/new_chart",
    data_table=table,
    fields=fields,
    string_fields={"title": "Height Histogram"},
)

with wandb.init() as run:
    # カスタムチャートを記録
    run.log({"my_custom_chart": my_custom_chart})
```

[コードを実行](https://tiny.cc/custom-charts)

### Matplotlib および Plotly プロット

`wandb.plot()` を使った W&B [Custom Charts]({{< relref path="/guides/models/app/features/custom-charts/walkthrough.md" lang="ja" >}}) の代わりに、[matplotlib](https://matplotlib.org/) や [Plotly](https://plotly.com/) で作成したグラフも記録できます。

```python
import wandb
import matplotlib.pyplot as plt

with wandb.init() as run:
    # シンプルなmatplotlibプロットを作成
    plt.figure()
    plt.plot([1, 2, 3, 4])
    plt.ylabel("some interesting numbers")
    
    # プロットをW&Bに記録
    run.log({"chart": plt})
```

`wandb.Run.log()` には `matplotlib` のプロット（または Figure）オブジェクトをそのまま渡すだけでOKです。デフォルトでは、このプロットが [Plotly](https://plot.ly/) グラフに変換されます。画像として記録したい場合は、`wandb.Image` に渡してください。Plotlyで作ったグラフも直接記録できます。

{{% alert %}}
「You attempted to log an empty plot」というエラーが出る場合は、`fig = plt.figure()` のようにプロットとは別に Figure を作成し、`wandb.Run.log()` で `fig` を渡してください。
{{% /alert %}}

### W&B Tables にカスタムHTMLを記録

W&B では、Plotly や Bokeh のインタラクティブグラフを HTML 形式で出力し、Tables に追加できます。

#### PlotlyグラフをHTMLとしてTablesに記録

インタラクティブなPlotlyグラフを wandb Tables にHTMLとして記録できます。

```python
import wandb
import plotly.express as px

# 新しいrunを初期化
with wandb.init(project="log-plotly-fig-tables", name="plotly_html") as run:

    # Tableを作成
    table = wandb.Table(columns=["plotly_figure"])

    # Plotly用HTMLファイルパス
    path_to_plotly_html = "./plotly_figure.html"

    # サンプルのPlotlyグラフ
    fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])

    # PlotlyグラフをHTMLとして保存
    # auto_play=False でアニメーションPlotlyグラフが
    # 自動再生されるのを防ぎます
    fig.write_html(path_to_plotly_html, auto_play=False)

    # PlotlyのHTMLファイルをTableに追加
    table.add_data(wandb.Html(path_to_plotly_html))

    # Tableを記録
    run.log({"test_table": table})
```

#### BokehグラフをHTMLとしてTablesに記録

インタラクティブなBokehグラフを wandb Tables にHTMLとして記録できます。

```python
from scipy.signal import spectrogram
import holoviews as hv
import panel as pn
from scipy.io import wavfile
import numpy as np
from bokeh.resources import INLINE

hv.extension("bokeh", logo=False)
import wandb


def save_audio_with_bokeh_plot_to_html(audio_path, html_file_name):
    sr, wav_data = wavfile.read(audio_path)
    duration = len(wav_data) / sr
    f, t, sxx = spectrogram(wav_data, sr)
    spec_gram = hv.Image((t, f, np.log10(sxx)), ["Time (s)", "Frequency (hz)"]).opts(
        width=500, height=150, labelled=[]
    )
    audio = pn.pane.Audio(wav_data, sample_rate=sr, name="Audio", throttle=500)
    slider = pn.widgets.FloatSlider(end=duration, visible=False)
    line = hv.VLine(0).opts(color="white")
    slider.jslink(audio, value="time", bidirectional=True)
    slider.jslink(line, value="glyph.location")
    combined = pn.Row(audio, spec_gram * line, slider).save(html_file_name)


html_file_name = "audio_with_plot.html"
audio_path = "hello.wav"
save_audio_with_bokeh_plot_to_html(audio_path, html_file_name)

wandb_html = wandb.Html(html_file_name)

with wandb.init(project="audio_test") as run:
    my_table = wandb.Table(columns=["audio_with_plot"], data=[[wandb_html]])
    run.log({"audio_table": my_table})
```