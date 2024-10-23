---
title: "How do I add Plotly or Bokeh Charts into Tables?"
displayed_sidebar: support
tags:
   - experiments
   - tables
   - charts
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

Direct integration of Plotly or Bokeh figures into tables is not supported. Instead, export the figures to HTML and include the HTML in the table. Below are examples demonstrating this with interactive Plotly and Bokeh charts.

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

# Initialize a new run
run = wandb.init(project="log-plotly-fig-tables", name="plotly_html")

# Create a table
table = wandb.Table(columns=["plotly_figure"])

# Define path for Plotly figure
path_to_plotly_html = "./plotly_figure.html"

# Create a Plotly figure
fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])

# Export Plotly figure to HTML
# Setting auto_play to False prevents animated Plotly charts from playing automatically
fig.write_html(path_to_plotly_html, auto_play=False)

# Add Plotly figure as HTML file to the table
table.add_data(wandb.Html(path_to_plotly_html))

# Log Table
run.log({"test_table": table})
wandb.finish()
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
  </TabItem>
</Tabs>