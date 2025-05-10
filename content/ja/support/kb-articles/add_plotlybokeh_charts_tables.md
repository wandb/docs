---
title: Tables に Plotly または Bokeh チャートを追加するにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-add_plotlybokeh_charts_tables
support:
  - experiments
  - tables
  - charts
toc_hide: true
type: docs
url: /ja/support/:filename
translationKey: add_plotlybokeh_charts_tables
---
Plotly または Bokeh の図をテーブルに直接統合することはサポートされていません。代わりに、図を HTML にエクスポートし、HTML をテーブルに含めてください。以下に、対話型の Plotly と Bokeh グラフを使用した例を示します。

{{< tabpane text=true >}}
{{% tab "Using Plotly" %}}
```python
import wandb
import plotly.express as px

# 新しい run の初期化
run = wandb.init(project="log-plotly-fig-tables", name="plotly_html")

# テーブルの作成
table = wandb.Table(columns=["plotly_figure"])

# Plotly 図のパスを定義
path_to_plotly_html = "./plotly_figure.html"

# Plotly 図の作成
fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])

# Plotly 図を HTML にエクスポート
# auto_play を False に設定すると、アニメーションされた Plotly グラフの自動再生が防止されます
fig.write_html(path_to_plotly_html, auto_play=False)

# Plotly 図を HTML ファイルとしてテーブルに追加
table.add_data(wandb.Html(path_to_plotly_html))

# テーブルのログ
run.log({"test_table": table})
wandb.finish()
```
{{% /tab %}}
{{% tab "Using Bokeh" %}}
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
    spec_gram = hv.Image((t, f, np.log10(sxx)), ["Time (s)", "Frequency (Hz)"]).opts(
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
{{% /tab %}}
{{% /tabpane %}}