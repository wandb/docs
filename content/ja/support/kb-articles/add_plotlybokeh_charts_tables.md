---
title: Plotly や Bokeh のチャートを Tables に追加するにはどうすればいいですか？
menu:
  support:
    identifier: ja-support-kb-articles-add_plotlybokeh_charts_tables
support:
- 実験
- テーブル
- チャート
toc_hide: true
type: docs
url: /support/:filename
---

Plotly や Bokeh の図を直接 Tables に統合することはサポートされていません。その代わり、図を HTML へエクスポートし、その HTML を テーブル に含めてください。以下は、インタラクティブな Plotly および Bokeh チャートでこれを実現する例です。

{{< tabpane text=true >}}
{{% tab "Plotly を使う" %}}
```python
import wandb
import plotly.express as px

# 新しい run を初期化
with wandb.init(project="log-plotly-fig-tables", name="plotly_html") as run:

    # テーブルを作成
    table = wandb.Table(columns=["plotly_figure"])

    # Plotly 図のパスを定義
    path_to_plotly_html = "./plotly_figure.html"

    # Plotly 図を作成
    fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])

    # Plotly 図を HTML にエクスポート
    # auto_play を False に設定すると、アニメーション付きの Plotly チャートが自動再生されません
    fig.write_html(path_to_plotly_html, auto_play=False)

    # Plotly 図を HTML ファイルとしてテーブルに追加
    table.add_data(wandb.Html(path_to_plotly_html))

    # テーブルをログ
    run.log({"test_table": table})

```
{{% /tab %}}
{{% tab "Bokeh を使う" %}}
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
with wandb.init(project="audio_test") as run:
    my_table = wandb.Table(columns=["audio_with_plot"], data=[[wandb_html], [wandb_html]])
    run.log({"audio_table": my_table})
```
{{% /tab %}}
{{% /tabpane %}}