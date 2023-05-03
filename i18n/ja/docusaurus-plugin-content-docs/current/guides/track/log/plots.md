---
description: 機械学習実験から精度図を作成してトラッキングします。
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# ログ図

<head>
  <title>W&B実験からログ図とトラッキング図を作成する。</title>
</head>

`wandb.plot`のメソッドを使って、`wandb.log`でチャートをトラッキングできます。トレーニング中に時間経過で変化するチャートも含まれます。カスタムチャートフレームワークについて詳しく知りたい場合は、[こちらのガイド](../../app/features/custom-charts/walkthrough.md)を参照してください。

### 基本チャート

これらのシンプルなチャートは、メトリクスや結果の基本的なデータ可視化を簡単に構築するのに便利です。

<Tabs
  defaultValue="line"
  values={[
    {label: '折れ線', value: 'line'},
    {label: '散布図', value: 'scatter'},
    {label: '棒グラフ', value: 'bar'},
    {label: 'ヒストグラム', value: 'histogram'},
    {label: 'マルチライン', value: 'multiline'},
  ]}>
  <TabItem value="line">

`wandb.plot.line()`
カスタム線プロット（任意の軸で接続され順序付けされた点のリスト）をログに記録します。

```python
data = [[x, y] for (x, y) in zip(x_values, y_values)]
table = wandb.Table(data=data, columns=["x", "y"])
wandb.log(
    {"my_custom_plot_id": wandb.plot.line(table, "x", "y",
           title="カスタムY対X線プロット")})
```

これを使用して、任意の2つの次元で曲線をログに記録できます。2つの値のリストを互いにプロットする場合、リスト内の値の数は正確に一致する必要があります（つまり、各点にxとyが必要です）。

![](/images/track/line_plot.png)

[アプリで見る →](https://wandb.ai/wandb/plots/reports/Custom-Line-Plots--VmlldzoyNjk5NTA)

[コードを実行する →](https://tiny.cc/custom-charts)
  </TabItem>
  <TabItem value="scatter">

`wandb.plot.scatter()`

カスタム散布図（任意の軸xとyで点（x、y）のリスト）をログに記録します。

```python
data = [[x, y] for (x, y) in zip(class_x_scores, class_y_scores)]
table = wandb.Table(data=data, columns=["class_x", "class_y"])
wandb.log({"my_custom_id": wandb.plot.scatter(table,
                            "class_x", "class_y")})
```
この機能を使って、任意の2つの次元に散布図のポイントをログすることができます。2つの値のリストを互いにプロットする場合は、リスト内の値の数が完全に一致している必要があります（つまり、各点にはxとyの両方が必要です）。

![](/images/track/demo_scatter_plot.png)

[アプリで見る →](https://wandb.ai/wandb/plots/reports/Custom-Scatter-Plots--VmlldzoyNjk5NDQ)

[コードを実行する →](https://tiny.cc/custom-charts)
  </TabItem>
  <TabItem value="bar">

`wandb.plot.bar()`

独自の棒グラフを作成してログすることができます。これは、数行でラベル付きの値を棒としてリスト化したものです。

```python
data = [[label, val] for (label, val) in zip(labels, values)]
table = wandb.Table(data=data, columns = ["label", "value"])
wandb.log({"my_bar_chart_id" : wandb.plot.bar(table, "label",
                               "value", title="Custom Bar Chart")
```

この機能を使って、任意の棒グラフをログすることができます。リスト内のラベルと値の数が完全に一致していることに注意してください（つまり、各データポイントには両方が必要です）。

![](/images/track/basic_charts_bar.png)

[アプリで見る →](https://wandb.ai/wandb/plots/reports/Custom-Bar-Charts--VmlldzoyNzExNzk)

[コードを実行する →](https://tiny.cc/custom-charts)
  </TabItem>
  <TabItem value="histogram">
`wandb.plot.histogram()`

カスタムヒストグラムを記録するために、リスト内の値をカウント/出現頻度に基づいてビンに分類します。これは、予測信頼スコアのリスト（`scores`）がある場合に、その分布を視覚化したい場合に使用できます。

```python
data = [[s] for s in scores]
table = wandb.Table(data=data, columns=["scores"])
wandb.log({'my_histogram': wandb.plot.histogram(table, "scores",
                           title="Histogram")})
```

これを使って任意のヒストグラムを記録できます。`data`はリストのリストで、行と列の2D配列をサポートすることを意図しています。

![](/images/track/demo_custom_chart_histogram.png)

[アプリで見る →](https://wandb.ai/wandb/plots/reports/Custom-Histograms--VmlldzoyNzE0NzM)

[コードを実行 →](https://tiny.cc/custom-charts)
  </TabItem>
  <TabItem value="multiline">

`wandb.plot.line_series()`

複数の線または複数の異なるx-y座標ペアのリストを、共有されたx-y座標のセットにプロットします:

```python
wandb.log({"my_custom_id" : wandb.plot.line_series(
          xs=[0, 1, 2, 3, 4],
          ys=[[10, 20, 30, 40, 50], [0.5, 11, 72, 3, 41]],
          keys=["metric Y", "metric Z"],
          title="Two Random Metrics",
          xname="x units")})
```
xとyのポイントの数は正確に一致している必要があります。複数のy値のリストに一致するx値のリストを1つ提供するか、y値のリストごとに別のx値のリストを提供できます。

![](/images/track/basic_charts_histogram.png)

[アプリで見る →](https://wandb.ai/wandb/plots/reports/Custom-Multi-Line-Plots--VmlldzozOTMwMjU)
  </TabItem>
</Tabs>

### モデル評価チャート

これらのプリセットチャートには組み込まれた`wandb.plot`メソッドがあり、スクリプトから直接チャートをログに記録し、UIで正確な情報を簡単に表示できます。

<Tabs
  defaultValue="precision_recall"
  values={[
    {label: 'PR曲線', value: 'precision_recall'},
    {label: 'ROC曲線', value: 'roc'},
    {label: '混同行列', value: 'confusion_matrix'},
  ]}>
  <TabItem value="precision_recall">

`wandb.plot.pr_curve()`

1行で[PR曲線](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision\_recall\_curve.html#sklearn.metrics.precision\_recall\_curve)を作成します。

```python
wandb.log({"pr": wandb.plot.pr_curve(ground_truth, predictions)})
```

このログは、コードが次のものにアクセスできるときに記録できます。
* モデルの予測スコア（`predictions`）を一連の例に適用
* それらの例に対応する正解ラベル（`ground_truth`）
* （オプション）ラベル/クラス名のリスト（`labels=["cat", "dog", "bird"...]` index 0がcat、1= dog、2= birdなどを意味します）
* （オプション）プロットで表示するラベルのサブセット（リスト形式のまま）

![](/images/track/model_eval_charts_precision_recall.png)

[アプリで見る →](https://wandb.ai/wandb/plots/reports/Plot-Precision-Recall-Curves--VmlldzoyNjk1ODY)

[コードを実行 →](https://colab.research.google.com/drive/1mS8ogA3LcZWOXchfJoMrboW3opY1A8BY?usp=sharing)
  </TabItem>
  <TabItem value="roc">

`wandb.plot.roc_curve()`

1行で[ROCカーブ](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc\_curve.html#sklearn.metrics.roc\_curve)を生成：

```python
wandb.log({"roc": wandb.plot.roc_curve(ground_truth, predictions)})
```

コードが以下にアクセスできる場合は、これをログに記録できます。

* 一連の例に対するモデルの予測スコア（`predictions`）
* それらの例に対応する正解ラベル（`ground_truth`）
* （オプション）ラベル/クラス名のリスト（`labels=["cat", "dog", "bird"...]` index 0がcat、1= dog、2= birdなどを意味します）
* （オプション）プロットで表示するこれらのラベルのサブセット（リスト形式のまま）

![](/images/track/demo_custom_chart_roc_curve.png)
[アプリで見る →](https://wandb.ai/wandb/plots/reports/Plot-ROC-Curves--VmlldzoyNjk3MDE)

[コードを実行 →](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Plot\_ROC\_Curves\_with\_W%26B.ipynb)
  </TabItem>
  <TabItem value="confusion_matrix">

`wandb.plot.confusion_matrix()`

1行で多クラス[混同行列](https://scikit-learn.org/stable/auto\_examples/model\_selection/plot\_confusion\_matrix.html)を作成します。

```python
cm = wandb.plot.confusion_matrix(
    y_true=ground_truth,
    preds=predictions,
    class_names=class_names)
    
wandb.log({"conf_mat": cm})
```

以下にアクセス可能な場所でこれをログに記録できます。

* 一連の例でモデルが予測したラベル（`preds`）または正規化された確率スコア（`probs`）。確率の形状は（例の数、クラスの数）でなければなりません。確率と予測のどちらか一方を指定できますが、両方は指定できません。
* それらの例に対応する正解ラベル（`y_true`）
* ラベル・クラス名の文字列のフルリスト（`class_names`、例えば`class_names=["cat", "dog", "bird"]`でインデックス0が猫、1=犬、2=鳥、等の場合）

![](/images/experiments/confusion_matrix.png)

[アプリで見る →](https://wandb.ai/wandb/plots/reports/Confusion-Matrix--VmlldzozMDg1NTM)​

[コードを実行 →](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Log\_a\_Confusion\_Matrix\_with\_W%26B.ipynb)
  </TabItem>
</Tabs>
### インタラクティブなカスタムチャート

完全なカスタマイズについては、[カスタムチャートプリセット](../../app/features/custom-charts/walkthrough.md)を微調整するか新しいプリセットを作成し、チャートを保存してください。チャートIDを使用して、スクリプトから直接カスタムプリセットにデータをログに記録します。

```python
# プロットする列を持つ表を作成する
table = wandb.Table(data=data,
                    columns=["step", "height"])

# テーブルの列からチャートのフィールドへのマップ
fields = {"x": "step",
          "value": "height"}

# テーブルを使用して、新しいカスタムチャートプリセットを設定する
# 既存のチャートプリセットを使用する場合は、vega_spec_nameを変更してください
my_custom_chart = wandb.plot_table(
    vega_spec_name="carey/new_chart",
    data_table=table,
    fields=fields)
```

[コードを実行 →](https://tiny.cc/custom-charts)

### Matplotlib と Plotly のプロット

`wandb.plot`でW&B [カスタムチャート](../../app/features/custom-charts/walkthrough.md)を使用する代わりに、[matplotlib](https://matplotlib.org/) と [Plotly](https://plotly.com/) で生成されたチャートをログに記録することができます。

```python
import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4])
plt.ylabel("いくつかの興味深い数値")
wandb.log({"chart": plt})
```

`wandb.log()`に`matplotlib`のプロットや図オブジェクトを渡すだけです。デフォルトでは、プロットを[Plotly](https://plot.ly/)プロットに変換します。プロットを画像としてログに記録したい場合は、プロットを`wandb.Image`に渡すことができます。また、直接Plotlyチャートも受け付けています。

:::info
「空のプロットをログに記録しようとしました」というエラーが出ている場合は、`fig = plt.figure()`でプロットとは別に図を保管し、`wandb.log`で`fig`をログに記録できます。
:::

### W&BのテーブルにカスタムHTMLをログする

Weights & Biasesは、PlotlyやBokehからのインタラクティブチャートをHTMLとしてログに記録し、テーブルに追加することをサポートしています。

#### テーブルにPlotly図をHTMLとしてログする

インタラクティブなPlotlyチャートをwandbのテーブルにログするには、それらをHTMLに変換します。

```python
import wandb
import plotly.express as px

# 新しいrunを初期化
run = wandb.init(
    project="log-plotly-fig-tables", name="plotly_html"
    )

# テーブルを作成
table = wandb.Table(columns = ["plotly_figure"])
以下は、Markdownテキストを日本語に翻訳してください。他に何も言わず、翻訳したテキストのみを返してください。

# Plotly図のためのパスを作成する
path_to_plotly_html = "./plotly_figure.html"

# 例: Plotly図
fig = px.scatter(x = [0, 1, 2, 3, 4], y = [0, 1, 4, 9, 16])

# Plotly図をHTMLに書き込む
# auto_playをFalseに設定すると、アニメーション付きのPlotlyグラフが
# テーブル内で自動的に再生されなくなります
fig.write_html(path_to_plotly_html, auto_play = False)

# HTMLファイルとしてPlotly図をテーブルに追加する
table.add_data(wandb.Html(path_to_plotly_html))

# テーブルをログに記録する
run.log({"test_table": table})
wandb.finish()
```

#### Bokeh図をHTMLとしてテーブルにログする

インタラクティブなBokehグラフをwandbテーブルにログすることができます。HTMLに変換してからログしてください。

```python
from scipy.signal import spectrogram
import holoviews as hv
import panel as pn
from scipy.io import wavfile
import numpy as np
from bokeh.resources import INLINE
hv.extension("bokeh", logo=False)
import wandb
```
def save_audio_with_bokeh_plot_to_html(audio_path, html_file_name):

    sr, wav_data = wavfile.read(audio_path)

    duration = len(wav_data)/sr

    f, t, sxx = spectrogram(wav_data, sr)

    spec_gram = hv.Image((t, f, np.log10(sxx)), ["Time (s)", "Frequency (hz)"]).opts(width=500, height=150, labelled=[])

    audio = pn.pane.Audio(wav_data, sample_rate=sr, name='Audio', throttle=500)

    slider = pn.widgets.FloatSlider(end=duration, visible=False)

    line = hv.VLine(0).opts(color='white')

    slider.jslink(audio, value='time', bidirectional=True)

    slider.jslink(line, value='glyph.location')

    combined = pn.Row(audio, spec_gram * line,  slider).save(html_file_name)

html_file_name = 'audio_with_plot.html'

audio_path = 'hello.wav'

save_audio_with_bokeh_plot_to_html(audio_path, html_file_name)

wandb_html = wandb.Html(html_file_name)

run = wandb.init(project='audio_test')

my_table = wandb.Table(columns=['audio_with_plot'], data=[[wandb_html], [wandb_html]])

run.log({"audio_table": my_table})

run.finish()

```