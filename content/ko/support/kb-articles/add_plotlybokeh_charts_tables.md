---
title: 테이블에 Plotly 또는 Bokeh 차트를 어떻게 추가할 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-add_plotlybokeh_charts_tables
support:
- Experiments
- 테이블
- 차트
toc_hide: true
type: docs
url: /support/:filename
---

Plotly 또는 Bokeh figure를 테이블에 직접적으로 인테그레이트하는 것은 지원되지 않습니다. 대신, figure를 HTML로 내보내고 해당 HTML을 테이블에 포함할 수 있습니다. 아래는 Plotly와 Bokeh의 인터랙티브 차트를 사용한 예시입니다.

{{< tabpane text=true >}}
{{% tab "Plotly 사용하기" %}}
```python
import wandb
import plotly.express as px

# 새로운 run을 초기화합니다
with wandb.init(project="log-plotly-fig-tables", name="plotly_html") as run:

    # 테이블을 만듭니다
    table = wandb.Table(columns=["plotly_figure"])

    # Plotly figure의 경로를 지정합니다
    path_to_plotly_html = "./plotly_figure.html"

    # Plotly figure를 생성합니다
    fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])

    # Plotly figure를 HTML로 내보냅니다
    # auto_play를 False로 하면 애니메이션 플롯이 자동 재생되지 않습니다
    fig.write_html(path_to_plotly_html, auto_play=False)

    # Plotly figure의 HTML 파일을 테이블에 추가합니다
    table.add_data(wandb.Html(path_to_plotly_html))

    # 테이블을 로그합니다
    run.log({"test_table": table})

```
{{% /tab %}}
{{% tab "Bokeh 사용하기" %}}
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