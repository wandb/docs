---
description: >-
  Answers to frequently asked questions about tracking data from machine
  learning experiments with W&B Experiments.
displayed_sidebar: ja
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# ロギングのFAQ

<head>
  <title>実験からのデータロギングに関するよくある質問</title>
</head>

### W&B UIでログしたチャートやメディアをどのように整理できますか？

W&B UIでは、`/`を区切りとしてログしたパネルを整理します。デフォルトでは、ログされたアイテムの名前の`/`の前の部分が"パネルセクション"と呼ばれるパネルのグループを定義するために使用されます。

```python
wandb.log({'val/loss': 1.1, 'val/acc': 0.3})  
wandb.log({'train/loss': 0.1, 'train/acc': 0.94})  
```

[ワークスペース](../../app/pages/workspaces.md)の設定で、パネルが`/`で区切られた最初のコンポーネントだけでなく、すべてのコンポーネントでグループ化されるかどうかを変更できます。

### エポックやステップ間で画像やメディアを比較する方法は？

ステップから画像をログするたびに、UIに表示するためにそれらを保存します。画像パネルを展開し、ステップスライダーを使用して異なるステップの画像を表示します。これにより、モデルの出力がトレーニング中にどのように変化するかを簡単に比較できます。

### バッチごとに一部のメトリクスをログし、エポックごとに一部のメトリクスだけをログしたい場合はどうすればいいですか？

特定のメトリクスをすべてのバッチでログし、プロットを標準化したい場合、プロットしたいx軸の値とともにメトリクスをログできます。次に、カスタムプロットで編集をクリックし、カスタムx軸を選択します。
```python
wandb.log({'batch': batch_idx, 'loss': 0.3})
wandb.log({'epoch': epoch, 'val_acc': 0.94})
```

### 値のリストをログに記録する方法は？


<Tabs
  defaultValue="dictionary"
  values={[
    {label: '辞書を使う', value: 'dictionary'},
    {label: 'ヒストグラムとして', value: 'histogram'},
  ]}>
  <TabItem value="dictionary">

```python
wandb.log({f"losses/loss-{ii}": loss for ii, loss in enumerate(losses)})
```
  </TabItem>
  <TabItem value="histogram">

```python
wandb.log({"losses": wandb.Histogram(losses)})  # 損失をヒストグラムに変換
```
  </TabItem>
</Tabs>
### 複数の線をプロットに表示し、凡例を追加する方法は？

複数の線が表示されるカスタムチャートは、`wandb.plot.line_series()`を使用して作成できます。表示される折れ線グラフを確認するには、[プロジェクトページ](https://docs.wandb.ai/ref/app/pages/project-page)にアクセスする必要があります。プロットに凡例を追加するには、`wandb.plot.line_series()`内でキー引数を渡します。例はこちらです：

```python
wandb.log({"my_plot" : wandb.plot.line_series(
                         xs = x_data, 
                         ys = y_data, 
                         keys = ["metric_A", "metric_B"])}] 
```

Multi-lineプロットに関する詳細情報は、[こちら](https://docs.wandb.ai/guides/track/log/plots#basic-charts)のMulti-lineタブ内にあります。

### Plotly/Bokehチャートをテーブルに追加する方法は？

Plotly/Bokehフィギュアを直接テーブルに追加することはまだサポートされていません。代わりに、フィギュアをHTMLに書き込んでから、そのHTMLをテーブルに追加してください。以下に、インタラクティブなPlotlyとBokehチャートを使用した例を示します。

<Tabs
  defaultValue="plotly"
  values={[
    {label: 'Using Plotly', value: 'plotly'},
    {label: 'Using Bokeh', value: 'bokeh'},
  ]}>
  <TabItem value="plotly">

```python
import wandb
import plotly.express as px

# 新たなrunを初期化
run = wandb.init(
    project="log-plotly-fig-tables", 
    name="plotly_html"
    )
# テーブルを作成
table = wandb.Table(columns = ["plotly_figure"])

# Plotly図のパスを作成
path_to_plotly_html = "./plotly_figure.html"

# 例としてのPlotly図
fig = px.scatter(x = [0, 1, 2, 3, 4], y = [0, 1, 4, 9, 16])

# Plotly図をHTMLに書き込み
# auto_playをFalseに設定することで、アニメーション付きのPlotly
# グラフが自動で再生されるのを防ぐ
fig.write_html(path_to_plotly_html, auto_play = False) 

# テーブルにPlotly図をHTMLファイルとして追加
table.add_data(wandb.Html(path_to_plotly_html))

# テーブルをログ
run.log({"test_table": table})
wandb.finish()カスタムのx軸をどのように使用しますか？
```
  </TabItem>
  <TabItem value="bokeh">

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
  </TabItem>
</Tabs>

### なぜグラフに何も表示されないのですか？

「まだ可視化データが記録されていません」と表示されている場合、これはスクリプトからの最初の `wandb.log` コールがまだ取得されていないことを意味します。これは、実行がステップの完了に時間がかかるためです。エポックの終わりにログを記録している場合は、エポックごとに複数回ログを記録して、データがより迅速にストリーム化されるようにすることができます。
### 同じ指標が複数回表示されるのはなぜですか？

同じキーの下で異なるタイプのデータをログに記録している場合、データベース内でそれらを分割する必要があります。これにより、UIのドロップダウンに同じ指標名の複数のエントリが表示されます。グループ化するタイプは、`number`、`string`、`bool`、`other`（主に配列）、および任意の`wandb`データタイプ（`Histogram`、`Image`など）です。この振る舞いを防ぐために、各キーに1つのタイプのみを送信してください。

指標は大文字と小文字を区別しない形式で格納されるため、`"My-Metric"`と`"my-metric"`のような同じ名前の2つの指標がないことを確認してください。

### ログに記録されたデータに直接、プログラム的にアクセスする方法は？

`wandb.log`でログに記録されたメトリクスをトラッキングするために、historyオブジェクトが使用されます。[API](../public-api-guide)を使用すると、`run.history()`を介してhistoryオブジェクトにアクセスできます。

```python
api = wandb.Api()
run = api.run("username/project/run_id")
print(run.history())
```

### W&Bに何百万件ものステップをログに記録するとどうなりますか？ブラウザでの表示はどうなりますか？

より多くのポイントを送信すると、UIのグラフの読み込みに時間がかかります。線に1000ポイント以上ある場合、バックエンドで1000ポイントにダウンサンプリングしてから、ブラウザにデータを送信します。このサンプリングは非決定的であるため、ページを更新するとサンプリングされたポイントの別のセットが表示されます。

オリジナルのすべてのデータが必要な場合は、サンプリングされていないデータをダウンロードするために[データAPI](https://docs.wandb.com/library/api)を使用できます。

**ガイドライン**

メトリックごとに10,000ポイント未満のログを記録するようにすることをお勧めします。1つの線に100万ポイント以上のログを記録すると、ページの読み込みに時間がかかります。ログの足跡を減らす方法については、[こちらのColab](http://wandb.me/log-hf-colab)をチェックしてください。configやsummary metricsの列が500以上ある場合、テーブルには500のみ表示されます。

### W&Bをプロジェクトに統合したいが、画像やメディアをアップロードしたくない場合はどうすればいいですか？

W&Bは、スカラーのみをログに記録するプロジェクトにも使用できます。アップロードするファイルやデータを明示的に指定します。以下は、画像をログに記録しない[PyTorchの簡単な例](http://wandb.me/pytorch-colab)です。
### wandb.log()にクラス属性を渡すとどうなりますか？

一般的に、`wandb.log()`にクラス属性を渡すことはお勧めできません。なぜなら、ネットワーク呼び出しが行われる前に属性が変更される可能性があるからです。メトリクスをクラスの属性として保存している場合は、`wandb.log()`が呼ばれた時点での属性の値とログされたメトリクスが一致するように、属性をディープコピーすることをお勧めします。

### ログしたデータポイントよりも少ない数を見ているのはなぜですか？

X軸に`Step`以外のものを対してメトリクスを可視化している場合、期待したよりもデータポイントが少なくなることがあります。これは、互いにプロットされるメトリクスが同じ`Step`でログされることが必須であるためです。これにより、メトリクスが同期されるようになります。つまり、同じ`Step`でログされたメトリクスのみをサンプリングし、サンプル間を補間します。\
\
**ガイドライン**\
****\
****メトリクスを同じ`log()`呼び出しにまとめることをお勧めします。コードが以下のようになっている場合：

```python
wandb.log({"Precision" : precision})
...
wandb.log({"Recall" : recall})
```

以下のようにログする方が良いでしょう:

```python
wandb.log({
    "Precision" : precision,
    "Recall" : recall
})
```

別の方法として、手動でstepパラメータを制御し、自分のコード内でメトリクスを同期させることができます：

```python
wandb.log({"Precision" : precision}, step = step)
...
wandb.log({"Recall" : recall}, step = step)
```
`step` の値が `log()` の呼び出しで同じ場合、メトリクスは同じステップでログに記録され、一緒にサンプルされます。ただし、各呼び出しで step の値が単調に増加している必要があることに注意してください。そうでない場合、`log()` の呼び出し時に `step` の値は無視されます。