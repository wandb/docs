---
description: 機械学習 実験 からプロットを作成して追跡します。
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Log Plots

<head>
  <title>Log and Track Plots from W&B Experiments.</title>
</head>

`wandb.plot` メソッドを使って、`wandb.log` でグラフを追跡できます。トレーニング中に時間とともに変化するグラフも含まれます。カスタムグラフフレームワークについて詳しく知りたい方は、[このガイド](../../app/features/custom-charts/walkthrough.md)をチェックしてください。

### Basic Charts

これらのシンプルなチャートは、メトリクスと結果の基本的な可視化を簡単に構築することができます。

<Tabs
  defaultValue="line"
  values={[
    {label: 'Line', value: 'line'},
    {label: 'Scatter', value: 'scatter'},
    {label: 'Bar', value: 'bar'},
    {label: 'Histogram', value: 'histogram'},
    {label: 'Multi-line', value: 'multiline'},
  ]}>
  <TabItem value="line">

`wandb.plot.line()`

任意の軸上にリスト化されたカスタム折れ線グラフをログに記録します。

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

これは任意の2次元でカーブをログに記録するのに使用できます。値のリスト同士をプロットする場合、それらのリストの値の数は正確に一致している必要があります（例えば、各ポイントは x と y を持つ必要があります）。

![](/images/track/line_plot.png)

[See in the app →](https://wandb.ai/wandb/plots/reports/Custom-Line-Plots--VmlldzoyNjk5NTA)

[Run the code →](https://tiny.cc/custom-charts)
  </TabItem>
  <TabItem value="scatter">

`wandb.plot.scatter()`

任意の軸上にリスト化されたカスタム散布図をログに記録します。

```python
data = [[x, y] for (x, y) in zip(class_x_scores, class_y_scores)]
table = wandb.Table(data=data, columns=["class_x", "class_y"])
wandb.log({"my_custom_id": wandb.plot.scatter(table, "class_x", "class_y")})
```

これは任意の2次元で散布点をログに記録するのに使用できます。値のリスト同士をプロットする場合、それらのリストの値の数は正確に一致している必要があります（例えば、各ポイントは x と y を持つ必要があります）。

![](/images/track/demo_scatter_plot.png)

[See in the app →](https://wandb.ai/wandb/plots/reports/Custom-Scatter-Plots--VmlldzoyNjk5NDQ)

[Run the code →](https://tiny.cc/custom-charts)
  </TabItem>
  <TabItem value="bar">

`wandb.plot.bar()`

任意のラベル付き値のリストをバーとしてログに記録します。

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

任意のバーグラフをログに記録するためにこれを使用できます。リスト内のラベルと値の数が正確に一致している必要があります（例えば、各データポイントは両方を持つ必要があります）。

![](/images/track/basic_charts_bar.png)

[See in the app →](https://wandb.ai/wandb/plots/reports/Custom-Bar-Charts--VmlldzoyNzExNzk)

[Run the code →](https://tiny.cc/custom-charts)
  </TabItem>
  <TabItem value="histogram">

`wandb.plot.histogram()`

カスタムヒストグラムをログに記録します—発生頻度によって値のリストをビンごとにソートします。例えば、予測の信頼度スコア (`scores`) のリストがあり、その分布を可視化したい場合：

```python
data = [[s] for s in scores]
table = wandb.Table(data=data, columns=["scores"])
wandb.log({"my_histogram": wandb.plot.histogram(table, "scores", title="Histogram")})
```

これは任意のヒストグラムをログに記録するために使用できます。`data` はリストのリストであり、行と列の2次元配列をサポートすることを意図しています。

![](/images/track/demo_custom_chart_histogram.png)

[See in the app →](https://wandb.ai/wandb/plots/reports/Custom-Histograms--VmlldzoyNzE0NzM)

[Run the code →](https://tiny.cc/custom-charts)
  </TabItem>
  <TabItem value="multiline">

`wandb.plot.line_series()`

複数の線、または複数の異なる x-y 座標ペアのリストを 1 つの共通の x-y 軸上にプロットします：

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

x と y のポイントの数が正確に一致している必要があります。1つの x 値のリストを複数の y 値のリストに対応させるか、各 y 値リストに対応する別々の x 値リストを提供できます。

![](/images/track/basic_charts_histogram.png)

[See in the app →](https://wandb.ai/wandb/plots/reports/Custom-Multi-Line-Plots--VmlldzozOTMwMjU)
  </TabItem>
</Tabs>

### Model Evaluation Charts

これらのプリセットチャートには、スクリプトから直接チャートをログに記録し、UIで必要な情報を素早く簡単に確認できる組み込みの `wandb.plot` メソッドがあります。

<Tabs
  defaultValue="precision_recall"
  values={[
    {label: 'Precision-Recall Curves', value: 'precision_recall'},
    {label: 'ROC Curves', value: 'roc'},
    {label: 'Confusion Matrix', value: 'confusion_matrix'},
  ]}>
  <TabItem value="precision_recall">

`wandb.plot.pr_curve()`

1行で [Precision-Recall curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision\_recall\_curve.html#sklearn.metrics.precision\_recall\_curve) を作成します：

```python
wandb.log({"pr": wandb.plot.pr_curve(ground_truth, predictions)})
```

次の条件が満たされる際にこれをログに記録できます：

* モデルの予測スコア (`predictions`) が一連の例に対して取得できる
* その例に対応する正解ラベル (`ground_truth`)
* （オプション）ラベル/クラス名のリスト（例えば、ラベルのインデックス 0 は猫、1 は犬、2 は鳥などの場合、`labels=["cat", "dog", "bird"...]`）
* （オプション）プロットで視覚化するためのラベルのサブセット（リスト形式）

![](/images/track/model_eval_charts_precision_recall.png)

[See in the app →](https://wandb.ai/wandb/plots/reports/Plot-Precision-Recall-Curves--VmlldzoyNjk1ODY)

[Run the code →](https://colab.research.google.com/drive/1mS8ogA3LcZWOXchfJoMrboW3opY1A8BY?usp=sharing)
  </TabItem>
  <TabItem value="roc">

`wandb.plot.roc_curve()`

1行で [ROC curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc\_curve.html#sklearn.metrics.roc\_curve) を作成します：

```python
wandb.log({"roc": wandb.plot.roc_curve(ground_truth, predictions)})
```

次の条件が満たされる際にこれをログに記録できます：

* モデルの予測スコア (`predictions`) が一連の例に対して取得できる
* その例に対応する正解ラベル (`ground_truth`)
* （オプション）ラベル/クラス名のリスト（例えば、ラベルのインデックス 0 は猫、1 は犬、2 は鳥などの場合、`labels=["cat", "dog", "bird"...]`）
* （オプション）プロットで視覚化するためのこれらのラベルのサブセット（リスト形式）

![](/images/track/demo_custom_chart_roc_curve.png)

[See in the app →](https://wandb.ai/wandb/plots/reports/Plot-ROC-Curves--VmlldzoyNjk3MDE)

[Run the code →](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Plot\_ROC\_Curves\_with\_W%26B.ipynb)
  </TabItem>
  <TabItem value="confusion_matrix">

`wandb.plot.confusion_matrix()`

1行でマルチクラスの [confusion matrix](https://scikit-learn.org/stable/auto\_examples/model\_selection/plot\_confusion\_matrix.html) を作成します：

```python
cm = wandb.plot.confusion_matrix(
    y_true=ground_truth, preds=predictions, class_names=class_names
)

wandb.log({"conf_mat": cm})
```

次の条件が満たされる際にこれをログに記録できます：

* モデルの予測ラベルが一連の例に対して取得できる (`preds`)、または正規化された確率スコア (`probs`)。確率は (例の数、クラス数) の形状を持つ必要があります。確率または予測のどちらか一方を提供できますが、両方は提供できません。
* その例に対応する正解ラベル (`y_true`)
* 文字列としてのラベル/クラス名の完全なリスト（例えば、インデックス 0 が猫、1 が犬、2 が鳥などの場合、`class_names=["cat", "dog", "bird"]`）

![](/images/experiments/confusion_matrix.png)

​[See in the app →](https://wandb.ai/wandb/plots/reports/Confusion-Matrix--VmlldzozMDg1NTM)​

​[Run the code →](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Log\_a\_Confusion\_Matrix\_with\_W%26B.ipynb)
  </TabItem>
</Tabs>

### Interactive Custom Charts

完全にカスタマイズするために、組み込みの [Custom Chart プリセット](../../app/features/custom-charts/walkthrough.md) を調整するか、または新しいプリセットを作成し、チャートを保存します。スクリプトから直接そのカスタムプリセットにデータをログに記録するためにチャート ID を使用します。

```python
# プロットする列を持つテーブルを作成
table = wandb.Table(data=data, columns=["step", "height"])

# テーブルの列をチャートのフィールドにマッピング
fields = {"x": "step", "value": "height"}

# 新しいカスタムチャートプリセットのデータを使用
# 自分の保存したチャートプリセットを使うには、vega_spec_name を変更
# タイトルを編集するには、string_fields を変更
my_custom_chart = wandb.plot_table(
    vega_spec_name="carey/new_chart",
    data_table=table,
    fields=fields,
    string_fields={"title": "Height Histogram"},
)
```

[Run the code →](https://tiny.cc/custom-charts)

### Matplotlib and Plotly Plots

W&B の [Custom Charts](../../app/features/custom-charts/walkthrough.md) を `wandb.plot` で使用する代わりに、[matplotlib](https://matplotlib.org/) と [Plotly](https://plotly.com/) で生成したグラフをログに記録できます。

```python
import matplotlib.pyplot as plt

plt.plot([1, 2, 3, 4])
plt.ylabel("some interesting numbers")
wandb.log({"chart": plt})
```

単に `matplotlib` のプロットやフィギュアオブジェクトを `wandb.log()` に渡します。デフォルトでは、プロットを [Plotly](https://plot.ly/) プロットに変換します。プロットを画像としてログに記録したい場合は、プロットを `wandb.Image` に渡すこともできます。また、Plotly のチャートも直接受け付けます。

:::info
「空のプロットをログに記録しようとしました」というエラーが表示される場合は、`fig = plt.figure()` でプロットからフィギュアを別々に保存し、その後 `wandb.log` の呼び出しで `fig` をログに記録してください。
:::

### Log Custom HTML to W&B Tables

W&B では、Plotly や Bokeh からのインタラクティブなグラフを HTML としてログに記録し、Tables に追加することができます。

#### Log Plotly figures to Tables as HTML

Plotly のインタラクティブなグラフを HTML に変換して wandb Tables にログに記録できます。

```python
import wandb
import plotly.express as px

# 新しい run を初期化
run = wandb.init(project="log-plotly-fig-tables", name="plotly_html")

# テーブルを作成
table = wandb.Table(columns=["plotly_figure"])

# Plotly フィギュアのパスを作成
path_to_plotly_html = "./plotly_figure.html"

# Plotly フィギュアの例
fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])

# Plotly フィギュアを HTML に書き込み
# auto_play を False に設定すると、テーブル内のアニメーション Plotly グラフの自動再生を防止
fig.write_html(path_to_plotly_html, auto_play=False)

# Plotly フィギュアを HTML ファイルとしてテーブルに追加
table.add_data(wandb.Html(path_to_plotly_html))

# テーブルをログに記録
run.log({"test_table": table})
wandb.finish()
```

#### Log Bokeh figures to Tables as HTML

Bokeh のインタラクティブなグラフを HTML に変換して wandb Tables にログに記録できます。

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

