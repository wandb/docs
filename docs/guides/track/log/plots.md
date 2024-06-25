---
description: 機械学習実験からプロットを作成し、トラッキングします。
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Log Plots

<head>
  <title>W&B Experimentsからプロットをログおよびトラックする</title>
</head>

`wandb.plot`のメソッドを使用すると、`wandb.log`と共にチャートをトラックできます。トレーニング中に時間とともに変化するチャートも含まれます。カスタムチャートフレームワークについて詳しくは[このガイド](../../app/features/custom-charts/walkthrough.md)を参照してください。

### 基本的なチャート

これらのシンプルなチャートは、メトリクスや結果の基本的な可視化を簡単に構築できます。

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

カスタム折れ線グラフをログします。任意の軸上に接続された順序付きポイントのリストです。

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

これを使用して任意の2次元上に曲線をログできます。2つのリストの値を対でプロットする場合、リスト内の値の数が正確に一致する必要があります（つまり、各ポイントにxとyの両方が必要です）。

![](/images/track/line_plot.png)

[アプリで見る →](https://wandb.ai/wandb/plots/reports/Custom-Line-Plots--VmlldzoyNjk5NTA)

[コードを実行 →](https://tiny.cc/custom-charts)
  </TabItem>
  <TabItem value="scatter">

`wandb.plot.scatter()`

カスタム散布図をログします。任意の軸xとy上のポイント(x, y)のリストです。

```python
data = [[x, y] for (x, y) in zip(class_x_scores, class_y_scores)]
table = wandb.Table(data=data, columns=["class_x", "class_y"])
wandb.log({"my_custom_id": wandb.plot.scatter(table, "class_x", "class_y")})
```

任意の2次元上に散布ポイントをログするために使用できます。2つのリストの値を対でプロットする場合、リスト内の値の数が正確に一致する必要があります（つまり、各ポイントにxとyの両方が必要です）。

![](/images/track/demo_scatter_plot.png)

[アプリで見る →](https://wandb.ai/wandb/plots/reports/Custom-Scatter-Plots--VmlldzoyNjk5NDQ)

[コードを実行 →](https://tiny.cc/custom-charts)
  </TabItem>
  <TabItem value="bar">

`wandb.plot.bar()`

カスタムバーグラフをログします。ラベル付きの値のリストをバーとして表示します。

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

任意のバーグラフをログするために使用できます。ラベルと値のリストの数が正確に一致する必要があります（つまり、各データポイントに両方が必要です）。

![](/images/track/basic_charts_bar.png)

[アプリで見る →](https://wandb.ai/wandb/plots/reports/Custom-Bar-Charts--VmlldzoyNzExNzk)

[コードを実行 →](https://tiny.cc/custom-charts)
  </TabItem>
  <TabItem value="histogram">

`wandb.plot.histogram()`

カスタムヒストグラムをログします。値のリストをカウントや出現頻度に基づいてビンに分けます。予測の信頼度スコアのリスト(`scores`)があり、その分布を可視化したいとしましょう。

```python
data = [[s] for s in scores]
table = wandb.Table(data=data, columns=["scores"])
wandb.log({"my_histogram": wandb.plot.histogram(table, "scores", title="Histogram")})
```

任意のヒストグラムをログするために使用できます。`data`はリストのリストで、行と列の2次元配列をサポートすることを意図しています。

![](/images/track/demo_custom_chart_histogram.png)

[アプリで見る →](https://wandb.ai/wandb/plots/reports/Custom-Histograms--VmlldzoyNzE0NzM)

[コードを実行 →](https://tiny.cc/custom-charts)
  </TabItem>
  <TabItem value="multiline">

`wandb.plot.line_series()`

複数の線、または複数の異なるx-y座標ペアのリストを1つの共有x-y軸にプロットします。

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

xとyのポイントの数が正確に一致する必要があることに注意してください。複数のy値リストに一致する1つのx値リスト、または各y値リストに対応する個別のx値リストを提供できます。

![](/images/track/basic_charts_histogram.png)

[アプリで見る →](https://wandb.ai/wandb/plots/reports/Custom-Multi-Line-Plots--VmlldzozOTMwMjU)
  </TabItem>
</Tabs>

### モデル評価チャート

これらのプリセットチャートには、`wandb.plot`のメソッドが組み込まれており、スクリプトから直接チャートを素早く簡単にログして、UIで必要な情報を正確に確認できます。

<Tabs
  defaultValue="precision_recall"
  values={[
    {label: 'Precision-Recall Curves', value: 'precision_recall'},
    {label: 'ROC Curves', value: 'roc'},
    {label: 'Confusion Matrix', value: 'confusion_matrix'},
  ]}>
  <TabItem value="precision_recall">

`wandb.plot.pr_curve()`

[PR曲線](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision\_recall\_curve.html#sklearn.metrics.precision\_recall\_curve)を1行で作成します。

```python
wandb.log({"pr": wandb.plot.pr_curve(ground_truth, predictions)})
```

コードが以下にアクセスできるときにこれをログできます：

* モデルが予測したスコア（examplesのセットに対する`predictions`）
* それらのexamplesに対する対応する正解ラベル（`ground_truth`）
* （オプション）ラベル／クラス名のリスト（ラベルインデックスが0 = cat、1 = dog、2 = birdなどの場合、`labels=["cat", "dog", "bird"...]`）
* （オプション）プロットに可視化するラベルのサブセット（リスト形式のまま）

![](/images/track/model_eval_charts_precision_recall.png)

[アプリで見る →](https://wandb.ai/wandb/plots/reports/Plot-Precision-Recall-Curves--VmlldzoyNjk1ODY)

[コードを実行 →](https://colab.research.google.com/drive/1mS8ogA3LcZWOXchfJoMrboW3opY1A8BY?usp=sharing)
  </TabItem>
  <TabItem value="roc">

`wandb.plot.roc_curve()`

[ROC曲線](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc\_curve.html#sklearn.metrics.roc\_curve)を1行で作成します。

```python
wandb.log({"roc": wandb.plot.roc_curve(ground_truth, predictions)})
```

コードが以下にアクセスできるときにこれをログできます：

* モデルが予測したスコア（examplesのセットに対する`predictions`）
* それらのexamplesに対する対応する正解ラベル（`ground_truth`）
* （オプション）ラベル／クラス名のリスト（ラベルインデックスが0 = cat、1 = dog、2 = birdなどの場合、`labels=["cat", "dog", "bird"...]`）
* （オプション）プロットに可視化するこれらのラベルのサブセット（リスト形式のまま）

![](/images/track/demo_custom_chart_roc_curve.png)

[アプリで見る →](https://wandb.ai/wandb/plots/reports/Plot-ROC-Curves--VmlldzoyNjk3MDE)

[コードを実行 →](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Plot\_ROC\_Curves\_with\_W%26B.ipynb)
  </TabItem>
  <TabItem value="confusion_matrix">

`wandb.plot.confusion_matrix()`

マルチクラスの[混同行列](https://scikit-learn.org/stable/auto\_examples/model\_selection/plot\_confusion\_matrix.html)を1行で作成します。

```python
cm = wandb.plot.confusion_matrix(
    y_true=ground_truth, preds=predictions, class_names=class_names
)

wandb.log({"conf_mat": cm})
```

コードが以下にアクセスできるときにこれをログできます：

* モデルがexamplesのセットに対して予測したラベル（`preds`）または正規化された確率スコア（`probs`）。確率は（例の数、クラスの数）の形状を持つ必要があります。確率または予測のいずれか一方のみを供給できます。
* それらのexamplesに対する対応する正解ラベル（`y_true`）
* 文字列としてのラベル／クラス名のリスト全体（例：`class_names=["cat", "dog", "bird"]`、インデックス0はcat、1=dog、2=birdなど）

![](/images/experiments/confusion_matrix.png)

[アプリで見る →](https://wandb.ai/wandb/plots/reports/Confusion-Matrix--VmlldzozMDg1NTM)​

[コードを実行 →](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Log\_a\_Confusion\_Matrix\_with\_W%26B.ipynb)
  </TabItem>
</Tabs>

### インタラクティブなカスタムチャート

完全なカスタマイズのために、組み込みの[カスタムチャートプリセット](../../app/features/custom-charts/walkthrough.md)を調整するか、新しいプリセットを作成してからチャートを保存します。チャートIDを使用して、スクリプトから直接そのカスタムプリセットにデータをログします。

```python
# プロットする列を持つテーブルを作成
table = wandb.Table(data=data, columns=["step", "height"])

# テーブルの列とチャートのフィールドをマップ
fields = {"x": "step", "value": "height"}

# 新しいカスタムチャートプリセットをテーブルで埋める
# 自分の保存したチャートプリセットを使用するには、vega_spec_nameを変更
# タイトルを編集するには、string_fieldsを変更
my_custom_chart = wandb.plot_table(
    vega_spec_name="carey/new_chart",
    data_table=table,
    fields=fields,
    string_fields={"title": "Height Histogram"},
)
```

[コードを実行 →](https://tiny.cc/custom-charts)

### MatplotlibおよびPlotlyプロット

`wandb.plot`を使用したW&B[カスタムチャート](../../app/features/custom-charts/walkthrough.md)の代わりに、[matplotlib](https://matplotlib.org/)および[Plotly](https://plotly.com/)で生成されたチャートをログできます。

```python
import matplotlib.pyplot as plt

plt.plot([1, 2, 3, 4])
plt.ylabel("some interesting numbers")
wandb.log({"chart": plt})
```

`wandb.log()`に`matplotlib`プロットまたは図オブジェクトを渡すだけです。デフォルトでは、プロットを[Plotly](https://plot.ly/)プロットに変換します。プロットを画像としてログする場合は、`wandb.Image`にプロットを渡すことができます。Plotlyチャートも直接受け付けます。

:::info
「空のプロットをログしようとしました」というエラーが発生した場合、`fig = plt.figure()`を使用してプロットとは別に図を保存し、`wandb.log`の呼び出しで`fig`をログします。
:::

### カスタムHTMLをW&B Tablesにログする

W&Bは、PlotlyやBokehからのインタラクティブチャートをHTMLとしてログし、Tablesに追加することをサポートしています。

#### Plotlyの図をHTMLとしてTablesにログする

Plotlyのインタラクティブチャートをwandb Tablesにログするためには、HTMLに変換します。

```python
import wandb
import plotly.express as px

# 新しいrunを初期化
run = wandb.init(project="log-plotly-fig-tables", name="plotly_html")

# テーブルを作成
table = wandb.Table(columns=["plotly_figure"])

# Plotly図のパスを作成
path_to_plotly_html = "./plotly_figure.html"

# Plotlyの例の図
fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])

# Plotly図をHTMLに書き出し
# auto_playをFalseに設定すると、アニメーション付きのPlotlyチャートがテーブルで自動再生されなくなります
fig.write_html(path_to_plotly_html, auto_play=False)

# Plotly図をテーブルにHTMLファイルとして追加
table.add_data(wandb.Html(path_to_plotly_html))

# テーブルをログ
run.log({"test_table": table})
wandb.finish()
```

#### Bokehの図をHTMLとしてTablesにログする

Bokehのインタラクティブチャートをwandb Tablesにログするためには、HTMLに変換します。

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
    spec_gram = hv.Image((t, f, np.log10(sxx)), ["時間 (秒)", "周波数 (Hz)"]).opts(
        width=500, height=150, labelled=[]
    )
    audio = pn.pane.Audio(wav_data, sample_rate=sr, name="音声", throttle=500)
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