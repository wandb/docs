---
title: Create and track plots from experiments
description: 機械学習 の 実験 からプロットを作成し、追跡します。
menu:
  default:
    identifier: ja-guides-models-track-log-plots
    parent: log-objects-and-media
---

`wandb.plot` のメソッドを使用すると、トレーニング中に時間とともに変化するグラフを含め、`wandb.log` でグラフを追跡できます。カスタムグラフのフレームワークの詳細については、[こちらのガイド]({{< relref path="/guides/models/app/features/custom-charts/walkthrough.md" lang="ja" >}})をご覧ください。

### 基本的なグラフ

これらのシンプルなグラフを使用すると、メトリクスと結果の基本的な可視化を簡単に構築できます。

{{< tabpane text=true >}}
    {{% tab header="折れ線グラフ" %}}
`wandb.plot.line()`

カスタムの折れ線グラフ（任意の軸上の接続され順序付けられた点のリスト）をログに記録します。

```python
data = [[x, y] for (x, y) in zip(x_values, y_values)]
table = wandb.Table(data=data, columns=["x", "y"])
wandb.log(
    {
        "my_custom_plot_id": wandb.plot.line(
            table, "x", "y", title="Custom Y vs X Line Plot"
        )
    }
)
```

これを使用して、任意の2つのディメンションの曲線をログに記録できます。2つの 値 のリストを互いにプロットする場合、リスト内の 値 の数は完全に一致する必要があります。たとえば、各ポイントにはxとyが必要です。

{{< img src="/images/track/line_plot.png" alt="" >}}

[アプリで表示](https://wandb.ai/wandb/plots/reports/Custom-Line-Plots--VmlldzoyNjk5NTA)

[コードを実行](https://tiny.cc/custom-charts)   
    {{% /tab %}}
    {{% tab header="散布図" %}}
`wandb.plot.scatter()`

カスタムの散布図（任意の軸xとyのペア上の点（x、y）のリスト）をログに記録します。

```python
data = [[x, y] for (x, y) in zip(class_x_scores, class_y_scores)]
table = wandb.Table(data=data, columns=["class_x", "class_y"])
wandb.log({"my_custom_id": wandb.plot.scatter(table, "class_x", "class_y")})
```

これを使用して、任意の2つのディメンションの散布図ポイントをログに記録できます。2つの 値 のリストを互いにプロットする場合、リスト内の 値 の数は完全に一致する必要があります。たとえば、各ポイントにはxとyが必要です。

{{< img src="/images/track/demo_scatter_plot.png" alt="" >}}

[アプリで表示](https://wandb.ai/wandb/plots/reports/Custom-Scatter-Plots--VmlldzoyNjk5NDQ)

[コードを実行](https://tiny.cc/custom-charts)    
    {{% /tab %}}
    {{% tab header="棒グラフ" %}}
`wandb.plot.bar()`

カスタムの棒グラフ（ラベル付きの値のリストを棒として表示）を、数行でネイティブにログに記録します。

```python
data = [[label, val] for (label, val) in zip(labels, values)]
table = wandb.Table(data=data, columns=["label", "value"])
wandb.log(
    {
        "my_bar_chart_id": wandb.plot.bar(
            table, "label", "value", title="Custom Bar Chart"
        )
    }
)
```

これを使用して、任意の棒グラフをログに記録できます。リスト内のラベルと 値 の数は完全に一致する必要があります。各データポイントには、ラベルと 値 の両方が必要です。

{{< img src="/images/track/basic_charts_bar.png" alt="" >}}

[アプリで表示](https://wandb.ai/wandb/plots/reports/Custom-Bar-Charts--VmlldzoyNzExNzk)

[コードを実行](https://tiny.cc/custom-charts)    
    {{% /tab %}}
    {{% tab header="ヒストグラム" %}}
`wandb.plot.histogram()`

カスタムのヒストグラム（値のリストを、出現回数/頻度でビンにソート）を、数行でネイティブにログに記録します。たとえば、予測信頼度スコア（`scores`）のリストがあり、その分布を可視化するとします。

```python
data = [[s] for s in scores]
table = wandb.Table(data=data, columns=["scores"])
wandb.log({"my_histogram": wandb.plot.histogram(table, "scores", title="Histogram")})
```

これを使用して、任意のヒストグラムをログに記録できます。`data` はリストのリストであり、行と列の2D配列をサポートすることを目的としています。

{{< img src="/images/track/demo_custom_chart_histogram.png" alt="" >}}

[アプリで表示](https://wandb.ai/wandb/plots/reports/Custom-Histograms--VmlldzoyNzE0NzM)

[コードを実行](https://tiny.cc/custom-charts)    
    {{% /tab %}}
    {{% tab header="複数折れ線グラフ" %}}
`wandb.plot.line_series()`

複数の線、または複数の異なるx-y座標ペアのリストを、1つの共有x-y軸セット上にプロットします。

```python
wandb.log(
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

xとyの点の数は完全に一致する必要があることに注意してください。1つのx 値 のリストを提供して複数のy 値 のリストに一致させるか、y 値 の各リストに個別のx 値 のリストを提供できます。

{{< img src="/images/track/basic_charts_histogram.png" alt="" >}}

[アプリで表示](https://wandb.ai/wandb/plots/reports/Custom-Multi-Line-Plots--VmlldzozOTMwMjU)    
    {{% /tab %}}
{{< /tabpane >}}



### モデル評価チャート

これらのプリセットされたグラフには、`wandb.plot` メソッドが組み込まれており、スクリプトから直接グラフを簡単にログに記録し、UIで探している正確な情報を確認できます。

{{< tabpane text=true >}}
    {{% tab header="PR曲線" %}}
`wandb.plot.pr_curve()`

1行で[PR曲線](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve)を作成します。

```python
wandb.log({"pr": wandb.plot.pr_curve(ground_truth, predictions)})
```

コードが以下にアクセスできる場合は、いつでもこれをログに記録できます。

* 例のセットに対するモデルの予測スコア（`predictions`）
* それらの例に対応する正解ラベル（`ground_truth`）
* （オプション）ラベル/クラス名のリスト（`labels=["cat", "dog", "bird"...]`、ラベルインデックス0がcat、1 = dog、2 = birdなどを意味する場合）
* （オプション）プロットで可視化するラベルの サブセット （リスト形式のまま）

{{< img src="/images/track/model_eval_charts_precision_recall.png" alt="" >}}

[アプリで表示](https://wandb.ai/wandb/plots/reports/Plot-Precision-Recall-Curves--VmlldzoyNjk1ODY)

[コードを実行](https://colab.research.google.com/drive/1mS8ogA3LcZWOXchfJoMrboW3opY1A8BY?usp=sharing)    
    {{% /tab %}}
    {{% tab header="ROC曲線" %}}

`wandb.plot.roc_curve()`

1行で[ROC曲線](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve)を作成します。

```python
wandb.log({"roc": wandb.plot.roc_curve(ground_truth, predictions)})
```

コードが以下にアクセスできる場合は、いつでもこれをログに記録できます。

* 例のセットに対するモデルの予測スコア（`predictions`）
* それらの例に対応する正解ラベル（`ground_truth`）
* （オプション）ラベル/クラス名のリスト（`labels=["cat", "dog", "bird"...]`、ラベルインデックス0がcat、1 = dog、2 = birdなどを意味する場合）
* （オプション）プロットで可視化するこれらのラベルの サブセット （リスト形式のまま）

{{< img src="/images/track/demo_custom_chart_roc_curve.png" alt="" >}}

[アプリで表示](https://wandb.ai/wandb/plots/reports/Plot-ROC-Curves--VmlldzoyNjk3MDE)

[コードを実行](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Plot_ROC_Curves_with_W%26B.ipynb)    
    {{% /tab %}}
    {{% tab header="混同行列" %}}
`wandb.plot.confusion_matrix()`

1行でマルチクラス[混同行列](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html)を作成します。

```python
cm = wandb.plot.confusion_matrix(
    y_true=ground_truth, preds=predictions, class_names=class_names
)

wandb.log({"conf_mat": cm})
```

コードが以下にアクセスできる場合は、いつでもこれをログに記録できます。

* 例のセットに対するモデルの予測ラベル（`preds`）または正規化された確率スコア（`probs`）。確率は、（例の数、クラスの数）の形状である必要があります。確率または予測のいずれかを指定できますが、両方は指定できません。
* それらの例に対応する 正解 ラベル（`y_true`）
* `class_names` の文字列としてのラベル/クラス名の完全なリスト。例：インデックス0が `cat`、1が `dog`、2が `bird` の場合、`class_names=["cat", "dog", "bird"]`

{{< img src="/images/experiments/confusion_matrix.png" alt="" >}}

​[アプリで表示](https://wandb.ai/wandb/plots/reports/Confusion-Matrix--VmlldzozMDg1NTM)​

​[コードを実行](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Log_a_Confusion_Matrix_with_W%26B.ipynb)    
    {{% /tab %}}
{{< /tabpane >}}


### インタラクティブなカスタムグラフ

完全にカスタマイズするには、組み込みの[カスタムグラフプリセット]({{< relref path="/guides/models/app/features/custom-charts/walkthrough.md" lang="ja" >}})を調整するか、新しいプリセットを作成して、グラフを保存します。グラフIDを使用して、スクリプトから直接そのカスタムプリセットにデータをログに記録します。

```python
# プロットする列を含むテーブルを作成します
table = wandb.Table(data=data, columns=["step", "height"])

# テーブルの列からグラフのフィールドへのマッピング
fields = {"x": "step", "value": "height"}

# テーブルを使用して新しいカスタムチャートプリセットに入力します
# 独自の保存済みチャートプリセットを使用するには、vega_spec_nameを変更します
# タイトルを編集するには、string_fieldsを変更します
my_custom_chart = wandb.plot_table(
    vega_spec_name="carey/new_chart",
    data_table=table,
    fields=fields,
    string_fields={"title": "Height Histogram"},
)
```

[コードを実行](https://tiny.cc/custom-charts)

### Matplotlib と Plotly プロット

`wandb.plot` で W&B [カスタムグラフ]({{< relref path="/guides/models/app/features/custom-charts/walkthrough.md" lang="ja" >}})を使用する代わりに、[matplotlib](https://matplotlib.org/) と [Plotly](https://plotly.com/) で生成されたグラフをログに記録できます。

```python
import matplotlib.pyplot as plt

plt.plot([1, 2, 3, 4])
plt.ylabel("some interesting numbers")
wandb.log({"chart": plt})
```

`matplotlib` プロットまたは図オブジェクトを `wandb.log()` に渡すだけです。デフォルトでは、プロットを [Plotly](https://plot.ly/) プロットに変換します。プロットを画像としてログに記録する場合は、プロットを `wandb.Image` に渡すことができます。Plotlyグラフも直接受け入れます。

{{% alert %}}
「空のプロットをログに記録しようとしました」というエラーが表示される場合は、`fig = plt.figure()` を使用してプロットとは別に図を保存し、`wandb.log` の呼び出しで `fig` をログに記録できます。
{{% /alert %}}

### カスタムHTMLをW&B Tablesに記録する

W&B は、Plotly および Bokeh からのインタラクティブなグラフをHTMLとして記録し、Tablesに追加することをサポートしています。

#### Plotlyの図をHTMLとしてTablesに記録する

インタラクティブなPlotlyグラフをHTMLに変換することで、wandb Tablesに記録できます。

```python
import wandb
import plotly.express as px

# 新しいrunを初期化します
run = wandb.init(project="log-plotly-fig-tables", name="plotly_html")

# テーブルを作成します
table = wandb.Table(columns=["plotly_figure"])

# Plotlyの図のパスを作成します
path_to_plotly_html = "./plotly_figure.html"

# Plotlyの図の例
fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])

# Plotlyの図をHTMLに書き込みます
# auto_playをFalseに設定すると、アニメーション化されたPlotlyグラフが
# テーブル内で自動的に再生されるのを防ぎます
fig.write_html(path_to_plotly_html, auto_play=False)

# Plotlyの図をHTMLファイルとしてテーブルに追加します
table.add_data(wandb.Html(path_to_plotly_html))

# テーブルをログに記録します
run.log({"test_table": table})
wandb.finish()
```

#### Bokehの図をHTMLとしてTablesに記録する

インタラクティブなBokehグラフをHTMLに変換することで、wandb Tablesに記録できます。

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
run = wandb.init(project="audio_test")
my_table = wandb.Table(columns=["audio_with_plot"], data=[[wandb_html], [wandb_html]])
run.log({"audio_table": my_table})
run.finish()
```
