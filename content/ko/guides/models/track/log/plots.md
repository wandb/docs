---
title: 실험에서 플롯을 생성하고 추적하기
description: 기계학습 실험에서 플롯을 생성하고 추적하세요.
menu:
  default:
    identifier: ko-guides-models-track-log-plots
    parent: log-objects-and-media
---

`wandb.plot`의 메소드를 사용하면 트레이닝 중에 변화하는 차트를 포함해, 다양한 차트를 `wandb.Run.log()`로 기록할 수 있습니다. W&B의 커스텀 차트 프레임워크에 대해 더 자세히 알고 싶다면 [커스텀 차트 가이드]({{< relref path="/guides/models/app/features/custom-charts/walkthrough.md" lang="ko" >}})를 참고하세요.

### 기본 차트

이 차트들은 메트릭과 결과를 간단하게 시각화할 수 있도록 도와줍니다.

{{< tabpane text=true >}}
    {{% tab header="Line" %}}

커스텀 라인 플롯을 기록하세요. 任의의 축에서 연결된 점들의 리스트를 시각화합니다.

```python
import wandb

with wandb.init() as run:
    data = [[x, y] for (x, y) in zip(x_values, y_values)]
    table = wandb.Table(data=data, columns=["x", "y"])
    run.log(
        {
            "my_custom_plot_id": wandb.plot.line(
                table, "x", "y", title="Custom Y vs X Line Plot"
            )
        }
    )
```

이 방법으로 임의의 두 차원에 곡선을 기록할 수 있습니다. 두 리스트의 값 개수는 반드시 같아야 하며, 각 포인트마다 x, y가 있어야 합니다.

{{< img src="/images/track/line_plot.png" alt="Custom line plot" >}}

[앱에서 보기](https://wandb.ai/wandb/plots/reports/Custom-Line-Plots--VmlldzoyNjk5NTA)

[코드 실행하기](https://tiny.cc/custom-charts)   
    {{% /tab %}}
    {{% tab header="Scatter" %}}

커스텀 산점도를 기록하세요. 任의의 x, y 좌표 (x, y)의 리스트를 시각화합니다.

```python
import wandb

with wandb.init() as run:
    data = [[x, y] for (x, y) in zip(class_x_scores, class_y_scores)]
    table = wandb.Table(data=data, columns=["class_x", "class_y"])
    run.log({"my_custom_id": wandb.plot.scatter(table, "class_x", "class_y")})
```

이 방법으로 임의의 두 차원에 산점도를 기록할 수 있습니다. 두 리스트의 값 개수는 반드시 같아야 하며, 각 포인트마다 x, y가 있어야 합니다.

{{< img src="/images/track/demo_scatter_plot.png" alt="Custom scatter plot" >}}

[앱에서 보기](https://wandb.ai/wandb/plots/reports/Custom-Scatter-Plots--VmlldzoyNjk5NDQ)

[코드 실행하기](https://tiny.cc/custom-charts)    
    {{% /tab %}}
    {{% tab header="Bar" %}}

커스텀 바 차트를 간단한 코드로 기록할 수 있습니다. 몇 줄만으로 라벨이 있는 값들의 리스트를 바 차트로 표현합니다.

```python
import wandb

with wandb.init() as run:
    data = [[label, val] for (label, val) in zip(labels, values)]
    table = wandb.Table(data=data, columns=["label", "value"])
    run.log(
        {
        "my_bar_chart_id": wandb.plot.bar(
            table, "label", "value", title="Custom Bar Chart"
        )
    }
)
```

임의의 바 차트를 기록할 때 사용할 수 있습니다. 리스트의 라벨과 값의 개수는 반드시 같아야 하며, 각 데이터 포인트에는 둘 다 있어야 합니다.

{{< img src="/images/track/basic_charts_bar.png" alt="Custom bar chart" >}}

[앱에서 보기](https://wandb.ai/wandb/plots/reports/Custom-Bar-Charts--VmlldzoyNzExNzk)

[코드 실행하기](https://tiny.cc/custom-charts)    
    {{% /tab %}}
    {{% tab header="Histogram" %}}

커스텀 히스토그램을 기록하세요. 값들의 리스트를 빈(bins)별로 개수/빈도를 기준으로 나누어 시각화합니다. 예를 들어, 예측 신뢰도 점수(`scores`)의 분포를 그릴 수 있습니다.

```python
import wandb

with wandb.init() as run:
    data = [[s] for s in scores]
    table = wandb.Table(data=data, columns=["scores"])
    run.log({"my_histogram": wandb.plot.histogram(table, "scores", title="Histogram")})
```

임의의 값에 대한 히스토그램을 기록할 수 있습니다. `data`는 행과 열로 구성된 2D 배열, 즉 리스트의 리스트 형태여야 합니다.

{{< img src="/images/track/demo_custom_chart_histogram.png" alt="Custom histogram" >}}

[앱에서 보기](https://wandb.ai/wandb/plots/reports/Custom-Histograms--VmlldzoyNzE0NzM)

[코드 실행하기](https://tiny.cc/custom-charts)    
    {{% /tab %}}
    {{% tab header="Multi-line" %}}

여러 라인 또는 여러 리스트의 x-y 좌표 쌍을 같은 축 위에 그릴 수 있습니다.

```python
import wandb
with wandb.init() as run:
    run.log(
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

x와 y 포인트의 개수가 반드시 일치해야 합니다. 여러 y값 리스트에 대해 하나의 x 리스트를 제공하거나, 각각 y 리스트 별로 x 리스트를 따로 줄 수도 있습니다.

{{< img src="/images/track/basic_charts_histogram.png" alt="Multi-line plot" >}}

[앱에서 보기](https://wandb.ai/wandb/plots/reports/Custom-Multi-Line-Plots--VmlldzozOTMwMjU)    
    {{% /tab %}}
{{< /tabpane >}}



### 모델 평가 차트

이 사전설정된 차트들은 내장 `wandb.plot()` 메소드로 간단하게 스크립트에서 바로 차트를 기록하고 UI에서 원하는 정보를 빠르게 확인할 수 있도록 도와줍니다.

{{< tabpane text=true >}}
    {{% tab header="Precision-recall curves" %}}

한 줄의 코드로 [PR 곡선](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve)을 그릴 수 있습니다:

```python
import wandb
with wandb.init() as run:
    # ground_truth 는 정답 라벨 리스트, predictions 는 예측 점수 리스트입니다
    # 예시: ground_truth = [0, 1, 1, 0], predictions = [0.1, 0.4, 0.35, 0.8]
    ground_truth = [0, 1, 1, 0]
    predictions = [0.1, 0.4, 0.35, 0.8]
    run.log({"pr": wandb.plot.pr_curve(ground_truth, predictions)})
```

다음 정보를 코드에서 확인할 수 있을 때 기록 가능합니다:

* 데이터셋 내 예시에 대한 모델의 예측 점수(`predictions`)
* 해당 예시들의 실제 정답 라벨(`ground_truth`)
* (옵션) 라벨/클래스명 리스트 (`labels=["cat", "dog", "bird"…]` 등, 0=고양이, 1=개, 2=새 등)
* (옵션) 차트에 시각화하고 싶은 라벨 서브셋 (리스트 형태 유지)

{{< img src="/images/track/model_eval_charts_precision_recall.png" alt="Precision-recall curve" >}}

[앱에서 보기](https://wandb.ai/wandb/plots/reports/Plot-Precision-Recall-Curves--VmlldzoyNjk1ODY)

[코드 실행하기](https://colab.research.google.com/drive/1mS8ogA3LcZWOXchfJoMrboW3opY1A8BY?usp=sharing)    
    {{% /tab %}}
    {{% tab header="ROC curves" %}}

한 줄로 [ROC 곡선](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve)을 그릴 수 있습니다:

```python
import wandb

with wandb.init() as run:
    # ground_truth 는 정답 라벨 리스트, predictions 는 예측 점수 리스트입니다
    # 예시: ground_truth = [0, 1, 1, 0], predictions = [0.1, 0.4, 0.35, 0.8]
    ground_truth = [0, 1, 1, 0]
    predictions = [0.1, 0.4, 0.35, 0.8]
    run.log({"roc": wandb.plot.roc_curve(ground_truth, predictions)})
```

다음 정보를 코드에서 확인할 수 있을 때 기록 가능합니다:

* 데이터셋 내 예시에 대한 모델의 예측 점수(`predictions`)
* 해당 예시의 그라운드 트루스 라벨(`ground_truth`)
* (옵션) 라벨/클래스명 리스트(`labels=["cat", "dog", "bird"…]` 등, 0=고양이, 1=개, 2=새 등)
* (옵션) 이 라벨들의 서브셋을 시각화(리스트 형태 유지)

{{< img src="/images/track/demo_custom_chart_roc_curve.png" alt="ROC curve" >}}

[앱에서 보기](https://wandb.ai/wandb/plots/reports/Plot-ROC-Curves--VmlldzoyNjk3MDE)

[코드 실행하기](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Plot_ROC_Curves_with_W%26B.ipynb)    
    {{% /tab %}}
    {{% tab header="Confusion matrix" %}}

한 줄로 다중 클래스 [confusion matrix](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html)를 만들 수 있습니다:

```python
import wandb

cm = wandb.plot.confusion_matrix(
    y_true=ground_truth, preds=predictions, class_names=class_names
)

with wandb.init() as run:
    run.log({"conf_mat": cm})
```

다음 정보를 코드에서 확인할 수 있을 때 기록 가능합니다:

* 데이터셋 내 예시에 대한 모델의 예측 라벨(`preds`) 혹은 정규화된 확률(`probs`). 확률은 (샘플 개수, 클래스 개수) 형태여야 하며, 확률 또는 예측 중 하나만 입력해야 합니다.
* 해당 예시들의 그라운드 트루스 라벨(`y_true`)
* 클래스명 전체 리스트 (`class_names`), 예: `class_names=["cat", "dog", "bird"]` (0=고양이, 1=개, 2=새)

{{< img src="/images/experiments/confusion_matrix.png" alt="Confusion matrix" >}}

​[앱에서 보기](https://wandb.ai/wandb/plots/reports/Confusion-Matrix--VmlldzozMDg1NTM)​

​[코드 실행하기](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Log_a_Confusion_Matrix_with_W%26B.ipynb)    
    {{% /tab %}}
{{< /tabpane >}}


### 인터랙티브 커스텀 차트

완전한 커스터마이징을 원한다면, [Custom Chart 프리셋]({{< relref path="/guides/models/app/features/custom-charts/walkthrough.md" lang="ko" >}})을 수정하거나 새 프리셋을 만든 후 저장하세요. 차트 ID를 사용하여 스크립트에서 해당 커스텀 프리셋에 직접 데이터를 기록할 수 있습니다.

```python
import wandb
# 플롯할 컬럼이 있는 테이블 생성
table = wandb.Table(data=data, columns=["step", "height"])

# 테이블의 컬럼을 차트의 필드와 매핑
fields = {"x": "step", "value": "height"}

# 테이블을 사용하여 새 커스텀 차트 프리셋을 채움
# 직접 저장한 프리셋을 사용하려면 vega_spec_name을 변경
# 타이틀을 바꾸려면 string_fields를 수정
my_custom_chart = wandb.plot_table(
    vega_spec_name="carey/new_chart",
    data_table=table,
    fields=fields,
    string_fields={"title": "Height Histogram"},
)

with wandb.init() as run:
    # 커스텀 차트 기록
    run.log({"my_custom_chart": my_custom_chart})
```

[코드 실행하기](https://tiny.cc/custom-charts)

### Matplotlib 및 Plotly 플롯 기록

W&B [Custom Chart]({{< relref path="/guides/models/app/features/custom-charts/walkthrough.md" lang="ko" >}}) 대신 `wandb.plot()`을 쓰지 않고, [matplotlib](https://matplotlib.org/)과 [Plotly](https://plotly.com/)로 만든 차트도 기록할 수 있습니다.

```python
import wandb
import matplotlib.pyplot as plt

with wandb.init() as run:
    # 간단한 matplotlib 플롯 생성
    plt.figure()
    plt.plot([1, 2, 3, 4])
    plt.ylabel("some interesting numbers")
    
    # 플롯을 W&B에 기록
    run.log({"chart": plt})
```

`matplotlib` 플롯이나 figure 오브젝트를 `wandb.Run.log()`에 넘기면 됩니다. 기본적으로 플롯은 [Plotly](https://plot.ly/) 플롯으로 변환됩니다. 이미지로 기록하고 싶다면 플롯을 `wandb.Image`로 넘기면 됩니다. Plotly 차트도 바로 지원합니다.

{{% alert %}}
"빈 플롯을 기록하려 했습니다" 오류가 발생한다면, `fig = plt.figure()`로 figure를 따로 만들고 `wandb.Run.log()`에서 `fig`를 기록해 주세요.
{{% /alert %}}

### W&B Table에 커스텀 HTML 기록하기

W&B는 Plotly, Bokeh 등을 이용한 인터랙티브 차트를 HTML로 Table에 추가하는 것도 지원합니다.

#### Plotly 차트를 Table에 HTML로 기록하기

인터랙티브 Plotly 차트를 wandb Table에 HTML로 변환해 기록할 수 있습니다.

```python
import wandb
import plotly.express as px

# 새로운 run 시작
with wandb.init(project="log-plotly-fig-tables", name="plotly_html") as run:

    # 테이블 생성
    table = wandb.Table(columns=["plotly_figure"])

    # Plotly figure 저장 경로 생성
    path_to_plotly_html = "./plotly_figure.html"

    # 예시 Plotly figure
    fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])

    # Plotly figure를 HTML로 저장
    # auto_play 를 False로 하면, 애니메이션 Plotly 차트가 Table에서 자동 재생되지 않습니다
    fig.write_html(path_to_plotly_html, auto_play=False)

    # Plotly figure를 HTML 파일로 Table에 추가
    table.add_data(wandb.Html(path_to_plotly_html))

    # Table 기록
    run.log({"test_table": table})
```

#### Bokeh 차트를 Table에 HTML로 기록하기

인터랙티브 Bokeh 차트를 wandb Table에 HTML로 변환해 기록할 수 있습니다.

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

with wandb.init(project="audio_test") as run:
    my_table = wandb.Table(columns=["audio_with_plot"], data=[[wandb_html]])
    run.log({"audio_table": my_table})
```