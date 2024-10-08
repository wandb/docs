---
title: Create and track plots from experiments
description: 기계학습 실험에서 플롯을 생성하고 추적하세요.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

`wandb.plot` 메소드를 사용하여 `wandb.log`로 차트를 기록할 수 있습니다. 트레이닝 중 시간이 지나면서 변하는 차트를 포함할 수 있습니다. 사용자 지정 차트 프레임워크에 대한 자세한 내용을 보려면 [이 가이드](../../app/features/custom-charts/walkthrough.md)를 참조하세요.

### 기본 차트

이러한 간단한 차트를 사용하면 메트릭과 결과의 기본 시각화를 쉽게 구성할 수 있습니다.

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

임의의 축에 연결되고 정렬된 포인트의 리스트인 사용자 지정 선형 차트를 기록합니다.

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

이 메소드를 사용하여 모든 두 차원의 곡선을 기록할 수 있습니다. 서로 다른 두 리스트를 비교하여 플로팅할 때, 리스트의 값 개수가 정확히 일치해야 합니다 (즉, 각 포인트는 x값과 y값을 가져야 합니다).

![](/images/track/line_plot.png)

[앱에서 보기](https://wandb.ai/wandb/plots/reports/Custom-Line-Plots--VmlldzoyNjk5NTA)

[코드 실행](https://tiny.cc/custom-charts)
  </TabItem>

  <TabItem value="scatter">

`wandb.plot.scatter()`

임의의 쌍의 축 x와 y에 대한 포인트 (x, y)의 리스트로 사용자 지정 산점도를 기록합니다.

```python
data = [[x, y] for (x, y) in zip(class_x_scores, class_y_scores)]
table = wandb.Table(data=data, columns=["class_x", "class_y"])
wandb.log({"my_custom_id": wandb.plot.scatter(table, "class_x", "class_y")})
```

이 메소드를 사용하여 모든 두 차원의 산포점을 기록할 수 있습니다. 서로 다른 두 리스트를 비교하여 플로팅할 때, 리스트의 값 개수가 정확히 일치해야 합니다 (즉, 각 포인트는 x값과 y값을 가져야 합니다).

![](/images/track/demo_scatter_plot.png)

[앱에서 보기](https://wandb.ai/wandb/plots/reports/Custom-Scatter-Plots--VmlldzoyNjk5NDQ)

[코드 실행](https://tiny.cc/custom-charts)
  </TabItem>

  <TabItem value="bar">

`wandb.plot.bar()`

몇 줄의 코드로 길쭉한 막대기로 표시되는 라벨과 값의 리스트를 사용자 지정 바 차트로 기록합니다.

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

이 메소드를 사용하여 임의의 바 차트를 기록할 수 있습니다. 리스트의 라벨과 값의 수가 정확히 일치해야 합니다 (즉, 각 데이터 포인트는 둘 다 가져야 합니다).

![](/images/track/basic_charts_bar.png)

[앱에서 보기](https://wandb.ai/wandb/plots/reports/Custom-Bar-Charts--VmlldzoyNzExNzk)

[코드 실행](https://tiny.cc/custom-charts)
  </TabItem>

  <TabItem value="histogram">

`wandb.plot.histogram()`

값의 리스트를 빈도 또는 발생 횟수에 따라 정렬하여 사용자 지정 히스토그램으로 기록합니다. 예측값 신뢰도 점수 리스트 (`scores`)가 있으며 그 분포를 시각화하고 싶다고 가정해 봅시다:

```python
data = [[s] for s in scores]
table = wandb.Table(data=data, columns=["scores"])
wandb.log({"my_histogram": wandb.plot.histogram(table, "scores", title="Histogram")})
```

이 메소드를 사용하여 임의의 히스토그램을 기록할 수 있습니다. `data`는 행과 열의 2D 배열을 지원하기 위해 리스트의 리스트입니다.

![](/images/track/demo_custom_chart_histogram.png)

[앱에서 보기](https://wandb.ai/wandb/plots/reports/Custom-Histograms--VmlldzoyNzE0NzM)

[코드 실행](https://tiny.cc/custom-charts)
  </TabItem>

  <TabItem value="multiline">

`wandb.plot.line_series()`

한 개의 x-y 축 집합에 여러 선 또는 여러 서로 다른 x-y 좌표 쌍의 리스트를 플로팅합니다:

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

x와 y 포인트의 수가 정확히 일치해야 한다는 점에 유의하세요. 여러 리스트의 y값에 대응하는 하나의 x값 리스트를 제공할 수도 있고, 각 y값 리스트에 대해 별개의 x값 리스트를 제공할 수도 있습니다.

![](/images/track/basic_charts_histogram.png)

[앱에서 보기](https://wandb.ai/wandb/plots/reports/Custom-Multi-Line-Plots--VmlldzozOTMwMjU)
  </TabItem>
</Tabs>

### 모델 평가 차트

이러한 프리셋 차트는 `wandb.plot` 메소드를 내장하여 스크립트에서 차트를 직접 기록하고 UI에서 원하는 정보를 정확히 확인할 수 있도록 합니다.

<Tabs
  defaultValue="precision_recall"
  values={[
    {label: 'Precision-Recall Curves', value: 'precision_recall'},
    {label: 'ROC Curves', value: 'roc'},
    {label: 'Confusion Matrix', value: 'confusion_matrix'},
  ]}>

  <TabItem value="precision_recall">

`wandb.plot.pr_curve()`

[Precision-Recall curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve)를 한 줄로 생성합니다:

```python
wandb.log({"pr": wandb.plot.pr_curve(ground_truth, predictions)})
```

코드가 엑세스할 수 있을 때마다 이를 기록할 수 있습니다:

* 예제 집합에 대한 모델의 예측 점수 (`predictions`)
* 해당 예제에 대한 그라운드 트루스 라벨 (`ground_truth`)
* (선택사항) 라벨/클래스 이름의 리스트 (`labels=["cat", "dog", "bird"...]`, 여기서 라벨 인덱스 0은 cat, 1=dog, 2=bird를 의미합니다.)
* (선택사항) 플롯에서 시각화할 라벨의 서브셋 (여전히 리스트 형식)

![](/images/track/model_eval_charts_precision_recall.png)

[앱에서 보기](https://wandb.ai/wandb/plots/reports/Plot-Precision-Recall-Curves--VmlldzoyNjk1ODY)

[코드 실행](https://colab.research.google.com/drive/1mS8ogA3LcZWOXchfJoMrboW3opY1A8BY?usp=sharing)
  </TabItem>

  <TabItem value="roc">

`wandb.plot.roc_curve()`

[ROC curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve)를 한 줄로 생성합니다:

```python
wandb.log({"roc": wandb.plot.roc_curve(ground_truth, predictions)})
```

코드가 엑세스할 수 있을 때마다 이를 기록할 수 있습니다:

* 예제 집합에 대한 모델의 예측 점수 (`predictions`)
* 해당 예제에 대한 그라운드 트루스 라벨 (`ground_truth`)
* (선택사항) 라벨/클래스 이름의 리스트 (`labels=["cat", "dog", "bird"...]`, 여기서 라벨 인덱스 0은 cat, 1=dog, 2=bird를 의미합니다.)
* (선택사항) 플롯에서 시각화할 라벨의 서브셋 (여전히 리스트 형식)

![](/images/track/demo_custom_chart_roc_curve.png)

[앱에서 보기](https://wandb.ai/wandb/plots/reports/Plot-ROC-Curves--VmlldzoyNjk3MDE)

[코드 실행](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Plot_ROC_Curves_with_W%26B.ipynb)
  </TabItem>

  <TabItem value="confusion_matrix">

`wandb.plot.confusion_matrix()`

몇 줄로 멀티 클래스 [혼동 행렬](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html)을 생성합니다:

```python
cm = wandb.plot.confusion_matrix(
    y_true=ground_truth, preds=predictions, class_names=class_names
)

wandb.log({"conf_mat": cm})
```

코드가 엑세스할 수 있을 때마다 이를 기록할 수 있습니다:

* 예제 집합에 대한 모델의 예측 라벨 (`preds`) 또는 정규화된 확률 점수 (`probs`). 확률은 (예제의 수, 클래스의 수)의 형태여야 합니다. 확률 또는 예측 중 하나만 제공할 수 있습니다.
* 해당 예제에 대한 그라운드 트루스 라벨 (`y_true`)
* 문자열로 된 라벨/클래스 이름의 전체 리스트 (`class_names`, 예: `class_names=["cat", "dog", "bird"]`에서 인덱스 0은 cat, 1=dog, 2=bird를 의미)

![](/images/experiments/confusion_matrix.png)

​[앱에서 보기](https://wandb.ai/wandb/plots/reports/Confusion-Matrix--VmlldzozMDg1NTM)​

​[코드 실행](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Log_a_Confusion_Matrix_with_W%26B.ipynb)
  </TabItem>
</Tabs>

### 인터랙티브 사용자 지정 차트

전체 사용자 지정을 위해, 내장된 [Custom Chart 프리셋](../../app/features/custom-charts/walkthrough.md)을 수정하거나 새로운 프리셋을 생성한 뒤 차트를 저장합니다. 차트 ID를 사용하여 해당 사용자 지정 프리셋으로 데이터 로그를 기록합니다.

```python
# 차트에 그릴 열을 가진 테이블 생성
table = wandb.Table(data=data, columns=["step", "height"])

# 테이블의 열을 차트의 필드에 매핑
fields = {"x": "step", "value": "height"}

# 테이블을 사용하여 새로운 사용자 지정 차트 프리셋을 채웁니다.
# 자신의 저장된 차트 프리셋을 사용하려면 vega_spec_name을 변경하세요
# 제목을 편집하려면 string_fields를 수정하세요
my_custom_chart = wandb.plot_table(
    vega_spec_name="carey/new_chart",
    data_table=table,
    fields=fields,
    string_fields={"title": "Height Histogram"},
)
```

[코드 실행](https://tiny.cc/custom-charts)

### Matplotlib 및 Plotly 차트

`wandb.plot`으로 W&B [Custom Charts](../../app/features/custom-charts/walkthrough.md)를 사용하는 대신, [matplotlib](https://matplotlib.org/) 및 [Plotly](https://plotly.com/)를 사용하여 생성된 차트를 로그 기록할 수 있습니다.

```python
import matplotlib.pyplot as plt

plt.plot([1, 2, 3, 4])
plt.ylabel("some interesting numbers")
wandb.log({"chart": plt})
```

`wandb.log()`에 `matplotlib` 플롯 또는 피규어 오브젝트를 전달하십시오. 기본적으로 플롯을 [Plotly](https://plot.ly/) 플롯으로 변환합니다. 플롯을 이미지로 로그 기록하려면 `wandb.Image`에 플롯을 전달할 수 있습니다. 또한 Plotly 차트도 직접 수용합니다.

:::info
"빈 플롯을 기록하려고 시도했습니다"라는 오류가 발생하면 `fig = plt.figure()`로 피규어를 플롯에서 별도로 저장한 후 `wandb.log` 호출에서 `fig`를 기록할 수 있습니다.
:::

### W&B Tables에 사용자지정 HTML 로그 기록

W&B는 Plotly 및 Bokeh에서 제공하는 인터랙티브 차트를 HTML로 로그 기록하여 Tables에 추가하는 것을 지원합니다.

#### Plotly 차트들을 Tables로 HTML로 로그 기록

Plotly 차트를 wandb Tables에 인터랙티브하게 기록하려면 HTML로 변환하세요.

```python
import wandb
import plotly.express as px

# 새 run을 초기화
run = wandb.init(project="log-plotly-fig-tables", name="plotly_html")

# 테이블 생성
table = wandb.Table(columns=["plotly_figure"])

# Plotly 피규어의 경로 만들기
path_to_plotly_html = "./plotly_figure.html"

# 예시 Plotly 피규어
fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])

# Plotly 피규어를 HTML로 작성
# auto_play를 False로 설정하면
# 테이블에서 애니메이션 Plotly 차트의 자동 재생을 방지합니다
fig.write_html(path_to_plotly_html, auto_play=False)

# Plotly 피규어를 HTML 파일로 테이블에 추가
table.add_data(wandb.Html(path_to_plotly_html))

# 테이블을 기록
run.log({"test_table": table})
wandb.finish()
```

#### Bokeh 차트들을 Tables로 HTML로 로그 기록

Bokeh 차트를 wandb Tables에 인터랙티브하게 기록하려면 HTML로 변환하세요.

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
run = wandb.init(project="audio_test")
my_table = wandb.Table(columns=["audio_with_plot"], data=[[wandb_html], [wandb_html]])
run.log({"audio_table": my_table})
run.finish()
```