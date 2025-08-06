---
title: 実験からプロットを作成してトラッキングする
description: 機械学習実験からプロットを作成し、追跡しましょう。
menu:
  default:
    identifier: plots
    parent: log-objects-and-media
---

`wandb.plot` のメソッドを使って、`wandb.Run.log()` でチャートをトラッキングできます。これは、トレーニング中に変化するグラフも記録できます。カスタムチャートフレームワークについてさらに知りたい場合は、[カスタムチャートのウォークスルー]({{< relref "/guides/models/app/features/custom-charts/walkthrough.md" >}}) をご覧ください。

### 基本チャート

これらのシンプルなチャートを使うことで、メトリクスや結果の基本的な可視化が簡単に行えます。

{{< tabpane text=true >}}
    {{% tab header="Line" %}}

カスタム折れ線グラフ（任意の軸上で繋がれた順序付きデータ点のリスト）を記録します。

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

任意の2次元で曲線を記録する際に利用できます。2つのリストを使ってグラフ化する場合、リスト内の値の数は必ず一致する必要があります。たとえば、各ポイントは x と y を持たなくてはいけません。

{{< img src="/images/track/line_plot.png" alt="Custom line plot" >}}

[アプリで確認](https://wandb.ai/wandb/plots/reports/Custom-Line-Plots--VmlldzoyNjk5NTA)

[コードを実行](https://tiny.cc/custom-charts)   
    {{% /tab %}}
    {{% tab header="Scatter" %}}

カスタム散布図（任意の x, y 軸上の点のリスト）を記録します。

```python
import wandb

with wandb.init() as run:
    data = [[x, y] for (x, y) in zip(class_x_scores, class_y_scores)]
    table = wandb.Table(data=data, columns=["class_x", "class_y"])
    run.log({"my_custom_id": wandb.plot.scatter(table, "class_x", "class_y")})
```

任意の2次元で散布点を記録する際に利用できます。2つの値リストを使ってグラフ化する場合、リスト内の値の数は必ず一致する必要があります。各ポイントは x と y を持つ必要があります。

{{< img src="/images/track/demo_scatter_plot.png" alt="Custom scatter plot" >}}

[アプリで確認](https://wandb.ai/wandb/plots/reports/Custom-Scatter-Plots--VmlldzoyNjk5NDQ)

[コードを実行](https://tiny.cc/custom-charts)    
    {{% /tab %}}
    {{% tab header="Bar" %}}

カスタム棒グラフ（ラベルと値のリストをバーで表示）を数行でネイティブに記録します。

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

任意の棒グラフを記録するために利用できます。ラベルと値のリストの長さは必ず一致する必要があります。それぞれのデータポイントにラベルと値が必要です。

{{< img src="/images/track/basic_charts_bar.png" alt="Custom bar chart" >}}

[アプリで確認](https://wandb.ai/wandb/plots/reports/Custom-Bar-Charts--VmlldzoyNzExNzk)

[コードを実行](https://tiny.cc/custom-charts)    
    {{% /tab %}}
    {{% tab header="Histogram" %}}

カスタムヒストグラム（値のリストをビンごとに数/頻度で集計）を数行でネイティブに記録できます。例えば、予測の信頼スコア (`scores`) のリストをヒストグラム化して分布を可視化できます。

```python
import wandb

with wandb.init() as run:
    data = [[s] for s in scores]
    table = wandb.Table(data=data, columns=["scores"])
    run.log({"my_histogram": wandb.plot.histogram(table, "scores", title="Histogram")})
```

任意のヒストグラムを記録できます。`data` は2次元配列（行と列）をサポートするためのリストのリストです。

{{< img src="/images/track/demo_custom_chart_histogram.png" alt="Custom histogram" >}}

[アプリで確認](https://wandb.ai/wandb/plots/reports/Custom-Histograms--VmlldzoyNzE0NzM)

[コードを実行](https://tiny.cc/custom-charts)    
    {{% /tab %}}
    {{% tab header="Multi-line" %}}

複数本の折れ線や、異なるx-y座標ペアのリストを、1つの共有x-y軸にまとめて描画します。

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

x/yのデータ点の数は必ず一致する必要があります。1つのx値リストで複数のy値リストに対応させたり、それぞれのy値リストに個別のx値リストを割り当てることも可能です。

{{< img src="/images/track/basic_charts_histogram.png" alt="Multi-line plot" >}}

[アプリで確認](https://wandb.ai/wandb/plots/reports/Custom-Multi-Line-Plots--VmlldzozOTMwMjU)    
    {{% /tab %}}
{{< /tabpane >}}



### モデル評価チャート

これらのプリセットチャートは `wandb.plot()` メソッドが組み込まれており、スクリプトからダイレクトにチャートを素早く記録でき、UIで必要な情報をすぐに確認できます。

{{< tabpane text=true >}}
    {{% tab header="Precision-recall curves" %}}

[Precision-Recall曲線](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve) を1行で作成できます。

```python
import wandb
with wandb.init() as run:
    # ground_truth は正解ラベルのリスト、predictions は予測スコアのリスト
    # 例: ground_truth = [0, 1, 1, 0], predictions = [0.1, 0.4, 0.35, 0.8]
    ground_truth = [0, 1, 1, 0]
    predictions = [0.1, 0.4, 0.35, 0.8]
    run.log({"pr": wandb.plot.pr_curve(ground_truth, predictions)})
```

以下のデータがあれば、いつでも記録できます。

* モデルの予測スコア（`predictions`）、評価データ上での結果
* 対応する正解ラベル（`ground_truth`）
* （オプション）ラベルやクラス名（例：`labels=["cat", "dog", "bird"...]` で 0=cat, 1=dog, 2=bird などを指定可能）
* （オプション）プロットで可視化したいラベルのサブセット（リスト形式で）

{{< img src="/images/track/model_eval_charts_precision_recall.png" alt="Precision-recall curve" >}}

[アプリで確認](https://wandb.ai/wandb/plots/reports/Plot-Precision-Recall-Curves--VmlldzoyNjk1ODY)

[コードを実行](https://colab.research.google.com/drive/1mS8ogA3LcZWOXchfJoMrboW3opY1A8BY?usp=sharing)    
    {{% /tab %}}
    {{% tab header="ROC curves" %}}

[ROC曲線](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve) も1行で作成できます。

```python
import wandb

with wandb.init() as run:
    # ground_truth は正解ラベルのリスト、predictions は予測スコアのリスト
    # 例: ground_truth = [0, 1, 1, 0], predictions = [0.1, 0.4, 0.35, 0.8]
    ground_truth = [0, 1, 1, 0]
    predictions = [0.1, 0.4, 0.35, 0.8]
    run.log({"roc": wandb.plot.roc_curve(ground_truth, predictions)})
```

以下のデータがあれば、いつでも記録できます。

* モデルの予測スコア（`predictions`）、評価データ上での結果
* 対応する正解ラベル（`ground_truth`）
* （オプション）ラベルやクラス名（例：`labels=["cat", "dog", "bird"...]` で 0=cat, 1=dog, 2=bird などを指定可能）
* （オプション）プロットで可視化したいラベルのサブセット（リスト形式で）

{{< img src="/images/track/demo_custom_chart_roc_curve.png" alt="ROC curve" >}}

[アプリで確認](https://wandb.ai/wandb/plots/reports/Plot-ROC-Curves--VmlldzoyNjk3MDE)

[コードを実行](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Plot_ROC_Curves_with_W%26B.ipynb)    
    {{% /tab %}}
    {{% tab header="Confusion matrix" %}}

マルチクラス [混同行列](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html) も1行で作成できます。

```python
import wandb

cm = wandb.plot.confusion_matrix(
    y_true=ground_truth, preds=predictions, class_names=class_names
)

with wandb.init() as run:
    run.log({"conf_mat": cm})
```

以下のどちらかのデータがあれば記録できます。

* モデルによるサンプルごとの予測ラベル（`preds`）または正規化済み確率スコア（`probs`）。probsは（サンプル数, クラス数）の形で、probabilities か predictions のどちらか一方を指定します。
* 対応する正解ラベル（`y_true`）
* すべてのラベル/クラス名を文字列リストで（例：`class_names=["cat", "dog", "bird"]` 0=cat, 1=dog, 2=bird ）

{{< img src="/images/experiments/confusion_matrix.png" alt="Confusion matrix" >}}

​[アプリで確認](https://wandb.ai/wandb/plots/reports/Confusion-Matrix--VmlldzozMDg1NTM)​

​[コードを実行](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Log_a_Confusion_Matrix_with_W%26B.ipynb)    
    {{% /tab %}}
{{< /tabpane >}}


### インタラクティブなカスタムチャート

フルカスタマイズしたい場合は、組み込みの [Custom Chart プリセット]({{< relref "/guides/models/app/features/custom-charts/walkthrough.md" >}}) を調整したり新たなプリセットを作って保存しましょう。カスタムプリセットチャートのIDを使えば、スクリプトから直接データを記録できます。

```python
import wandb
# プロット用のカラムを持つテーブルを作成
table = wandb.Table(data=data, columns=["step", "height"])

# テーブルのカラムからチャートのフィールドへのマッピング
fields = {"x": "step", "value": "height"}

# テーブルを使って新しいカスタムチャートプリセットを生成
# 独自の保存済みチャートプリセットを使いたい場合は vega_spec_name を変更
# タイトルを編集したい場合は string_fields を変更
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

### Matplotlib・Plotly のプロット

W&B の [Custom Charts]({{< relref "/guides/models/app/features/custom-charts/walkthrough.md" >}}) や `wandb.plot()` を使う代わりに、[matplotlib](https://matplotlib.org/) や [Plotly](https://plotly.com/) で生成したチャートも記録できます。

```python
import wandb
import matplotlib.pyplot as plt

with wandb.init() as run:
    # シンプルな matplotlib プロットを作成
    plt.figure()
    plt.plot([1, 2, 3, 4])
    plt.ylabel("some interesting numbers")
    
    # プロットを W&B に記録
    run.log({"chart": plt})
```

`matplotlib` のグラフや Figure オブジェクトをそのまま `wandb.Run.log()` に渡せます。デフォルトでは [Plotly](https://plot.ly/) プロットに変換されます。図を画像として記録したい場合は `wandb.Image` に渡してください。Plotlyのチャートも直接記録可能です。

{{% alert %}}
「You attempted to log an empty plot」というエラーが出る場合は、`fig = plt.figure()` で Figure を明示的に作成し、その fig を `wandb.Run.log()` で渡してください。
{{% /alert %}}

### 独自HTMLをW&B Tablesに記録

W&Bは Plotly や Bokeh のインタラクティブなグラフを HTML で記録し、Tables に追加することができます。

#### Plotly 図を Tables にHTMLとして記録

Plotly のインタラクティブチャートは、HTML に変換して wandb Tables に追加できます。

```python
import wandb
import plotly.express as px

# 新しい run を初期化
with wandb.init(project="log-plotly-fig-tables", name="plotly_html") as run:

    # テーブルを作成
    table = wandb.Table(columns=["plotly_figure"])

    # Plotly 図の HTML ファイルパス
    path_to_plotly_html = "./plotly_figure.html"

    # Plotlyの例
    fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])

    # Plotly図を HTML で保存
    # auto_play=False ならテーブル内のアニメーション付きPlotlyチャートが自動再生されない
    fig.write_html(path_to_plotly_html, auto_play=False)

    # Plotly図をHTMLファイルとしてTableに追加
    table.add_data(wandb.Html(path_to_plotly_html))

    # テーブルを記録
    run.log({"test_table": table})
```

#### Bokeh 図を Tables に HTML として記録

Bokeh のインタラクティブチャートは、HTML に変換して wandb Tables に追加できます。

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