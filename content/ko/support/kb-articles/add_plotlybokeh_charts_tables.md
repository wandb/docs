---
title: How do I add Plotly or Bokeh Charts into Tables?
menu:
  support:
    identifier: ko-support-kb-articles-add_plotlybokeh_charts_tables
support:
- experiments
- tables
- charts
toc_hide: true
type: docs
url: /ko/support/:filename
---

Plotly 또는 Bokeh figure를 테이블에 직접 통합하는 것은 지원되지 않습니다. 대신, figure를 HTML로 내보내고 해당 HTML을 테이블에 포함하세요. 아래는 인터랙티브 Plotly 및 Bokeh 차트를 사용하여 이를 보여주는 예제입니다.

{{< tabpane text=true >}}
{{% tab "Using Plotly" %}}
```python
import wandb
import plotly.express as px

# 새로운 run 초기화
run = wandb.init(project="log-plotly-fig-tables", name="plotly_html")

# 테이블 생성
table = wandb.Table(columns=["plotly_figure"])

# Plotly figure의 경로 정의
path_to_plotly_html = "./plotly_figure.html"

# Plotly figure 생성
fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])

# Plotly figure를 HTML로 내보내기
# auto_play를 False로 설정하면 애니메이션 Plotly 차트가 자동으로 재생되는 것을 방지합니다.
fig.write_html(path_to_plotly_html, auto_play=False)

# Plotly figure를 HTML 파일로 테이블에 추가
table.add_data(wandb.Html(path_to_plotly_html))

# 테이블 로그
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
