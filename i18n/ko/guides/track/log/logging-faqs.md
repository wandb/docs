---
description: Answers to frequently asked questions about tracking data from machine
  learning experiments with W&B Experiments.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 로깅 FAQ

<head>
  <title>실험에서 데이터 로깅에 대한 자주 묻는 질문</title>
</head>

### W&B UI에서 로그된 차트와 미디어를 어떻게 구성할 수 있나요?

W&B UI에서 로그된 패널을 구성하기 위해 `/`를 구분자로 취급합니다. 기본적으로 로그된 항목의 이름에서 `/` 앞의 구성 요소를 사용하여 "패널 섹션"이라고 하는 패널 그룹을 정의합니다.

```python
wandb.log({"val/loss": 1.1, "val/acc": 0.3})
wandb.log({"train/loss": 0.1, "train/acc": 0.94})
```

[워크스페이스](../../app/pages/workspaces.md) 설정에서 패널이 첫 번째 구성 요소만으로 그룹화되거나 `/`로 구분된 모든 구성 요소로 그룹화되는지 여부를 변경할 수 있습니다.

### 에포크나 스텝을 거치며 이미지나 미디어를 어떻게 비교할 수 있나요?

스텝마다 이미지를 로그할 때마다 UI에서 보여주기 위해 이미지를 저장합니다. 이미지 패널을 확장하고 스텝 슬라이더를 사용하여 다른 스텝의 이미지를 확인합니다. 이를 통해 모델의 출력이 학습 중에 어떻게 변하는지 쉽게 비교할 수 있습니다.

### 일부 메트릭은 배치 단위로, 일부 메트릭은 에포크 단위로만 로그하고 싶다면?

메트릭을 모든 배치에서 로그하고 플롯을 표준화하려면 메트릭과 함께 플롯하려는 x축 값을 로그할 수 있습니다. 그런 다음 사용자 정의 플롯에서 편집을 클릭하고 사용자 정의 x축을 선택합니다.

```python
wandb.log({"batch": batch_idx, "loss": 0.3})
wandb.log({"epoch": epoch, "val_acc": 0.94})
```

### 값 목록을 어떻게 로그하나요?



<Tabs
  defaultValue="dictionary"
  values={[
    {label: '사전 사용', value: 'dictionary'},
    {label: '히스토그램으로', value: 'histogram'},
  ]}>
  <TabItem value="dictionary">

```python
wandb.log({f"losses/loss-{ii}": loss for ii, loss in enumerate(losses)})
```
  </TabItem>
  <TabItem value="histogram">

```python
wandb.log({"losses": wandb.Histogram(losses)})  # losses를 히스토그램으로 변환
```
  </TabItem>
</Tabs>

### 범례가 있는 플롯에 여러 라인을 어떻게 그릴 수 있나요?

`wandb.plot.line_series()`를 사용하여 멀티 라인 사용자 정의 차트를 생성할 수 있습니다. 라인 차트를 보려면 [프로젝트 페이지](../../app/pages/project-page.md)로 이동해야 합니다. 플롯에 범례를 추가하려면 `wandb.plot.line_series()` 내에서 keys 인수를 전달합니다. 예를 들어:

```python
wandb.log(
    {
        "my_plot": wandb.plot.line_series(
            xs=x_data, ys=y_data, keys=["metric_A", "metric_B"]
        )
    }
)
```

멀티 라인 플롯에 대한 자세한 정보는 [여기](../../track/log/plots.md#basic-charts) 멀티 라인 탭에서 확인할 수 있습니다.

### 테이블에 Plotly/Bokeh 차트를 어떻게 추가하나요?

Plotly/Bokeh 그림을 테이블에 직접 추가하는 것은 아직 지원되지 않습니다. 대신 그림을 HTML로 작성한 다음 HTML을 테이블에 추가합니다. 아래는 인터랙티브 Plotly 및 Bokeh 차트 예제입니다.

<Tabs
  defaultValue="plotly"
  values={[
    {label: 'Plotly 사용', value: 'plotly'},
    {label: 'Bokeh 사용', value: 'bokeh'},
  ]}>
  <TabItem value="plotly">

```python
import wandb
import plotly.express as px

# 새로운 실행 초기화
run = wandb.init(project="log-plotly-fig-tables", name="plotly_html")

# 테이블 생성
table = wandb.Table(columns=["plotly_figure"])

# Plotly 그림 경로 생성
path_to_plotly_html = "./plotly_figure.html"

# Plotly 그림 예시
fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])

# Plotly 그림을 HTML로 작성
# auto_play를 False로 설정하면 애니메이션 Plotly
# 차트가 테이블에서 자동으로 재생되지 않음
fig.write_html(path_to_plotly_html, auto_play=False)

# Plotly 그림을 HTML 파일로 테이블에 추가
table.add_data(wandb.Html(path_to_plotly_html))

# 테이블 로그
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
  </TabItem>
</Tabs>

### 내 그래프에 아무것도 나타나지 않는 이유는 무엇인가요?

"아직 로그된 시각화 데이터가 없습니다"라는 메시지가 보이면 스크립트에서 첫 번째 `wandb.log` 호출을 아직 받지 못했다는 의미입니다. 이는 실행이 한 스텝을 완료하는 데 시간이 오래 걸릴 수 있기 때문입니다. 에포크마다 로깅하는 경우, 에포크당 몇 번 로깅하여 데이터가 더 빠르게 스트리밍되는 것을 볼 수 있습니다.

### 동일한 메트릭이 여러 번 나타나는 이유는 무엇인가요?

동일한 키 아래에서 다른 유형의 데이터를 로깅하는 경우 데이터베이스에서 분리해야 합니다. 이는 UI의 드롭다운에서 동일한 메트릭 이름의 여러 항목을 보게 됨을 의미합니다. 우리가 그룹화하는 유형에는 `number`, `string`, `bool`, `other`(대부분 배열), 그리고 모든 `wandb` 데이터 유형(`Histogram`, `Image` 등)이 포함됩니다. 이러한 동작을 피하려면 각 키에 하나의 유형만 보내십시오.

대소문자를 구분하지 않고 메트릭을 저장하므로 `"My-Metric"`과 `"my-metric"`과 같은 이름의 두 메트릭이 없는지 확인하세요.

### 실행에 로그된 데이터에 직접 프로그래매틱하게 어떻게 엑세스할 수 있나요?

history 개체는 `wandb.log`에 의해 로그된 메트릭을 추적하는 데 사용됩니다. [우리의 API](../public-api-guide.md)를 사용하여 `run.history()`를 통해 history 개체에 엑세스할 수 있습니다.

```python
api = wandb.Api()
run = api.run("username/project/run_id")
print(run.history())
```

### W&B에 수백만 스텝을 로그하면 어떻게 되나요? 브라우저에서는 어떻게 렌더링되나요?

보내는 포인트가 더 많을수록 그래프가 UI에서 로드되는 데 더 오래 걸립니다. 라인에 1000개 이상의 포인트가 있는 경우 백엔드에서 데이터를 브라우저로 보내기 전에 1000개의 포인트로 샘플링합니다. 이 샘플링은 비결정적이므로 페이지를 새로고침하면 다른 샘플링된 포인트 세트를 볼 수 있습니다.

**가이드라인**

메트릭 당 10,000개 미만의 포인트를 로그하도록 권장합니다. 라인에 100만 개 이상의 포인트를 로그하는 경우 페이지를 로드하는 데 시간이 걸립니다. 정확성을 희생하지 않고 로깅 발자국을 줄이는 전략에 대한 자세한 내용은 [이 Colab](http://wandb.me/log-hf-colab)을 확인하세요. 구성 및 요약 메트릭의 500개 이상의 열이 있는 경우 테이블에 500개만 표시됩니다.

### 이미지나 미디어를 업로드하지 않고 W&B를 프로젝트에 통합하고 싶다면?

W&B는 스칼라만 로그하는 프로젝트에도 사용할 수 있으며, 업로드하려는 파일이나 데이터를 명시적으로 지정합니다. 이미지를 로그하지 않는 [PyTorch의 간단한 예제](http://wandb.me/pytorch-colab)가 여기 있습니다.

### wandb.log()에 클래스 속성을 전달하면 어떻게 되나요?

`wandb.log()`에 클래스 속성을 전달하는 것은 일반적으로 권장되지 않으며, 네트워크 호출이 이루어지기 전에 속성이 변경될 수 있습니다. 클래스의 속성으로 메트릭을 저장하는 경우 `wandb.log()`가 호출된 시점의 속성 값과 일치하도록 메트릭을 로그하기 전에 속성을 깊은 복사하는 것이 권장됩니다.

### 로그한 것보다 적은 데이터 포인트를 보는 이유는 무엇인가요?

X축에 `Step`이 아닌 다른 것을 대비하여 메트릭을 시각화하는 경우 기대했던 것보다 적은 데이터 포인트를 볼 수 있습니다. 이는 메트릭을 서로 대비하여 플롯하려면 동일한 `Step`에서 로그되어야 하기 때문입니다 - 이것이 우리가 메트릭을 동기화하는 방법입니다. 즉, 동일한 `Step`에 로그된 메트릭만 샘플링하면서 샘플 사이에서 보간합니다.\
\
**가이드라인**\
****\
****메트릭을 동일한 `log()` 호출에 번들로 묶는 것을 권장합니다. 코드가 이렇게 보인다면:

```python
wandb.log({"Precision": precision})
...
wandb.log({"Recall": recall})
```

이렇게 로그하는 것이 더 좋습니다:

```python
wandb.log({"Precision": precision, "Recall": recall})
```

또는, 수동으로 step 파라미터를 제어하고 코드에서 메트릭을 동기화할 수 있습니다:

```python
wandb.log({"Precision": precision}, step=step)
...
wandb.log({"Recall": recall}, step=step)
```

`log()` 호출 모두에서 `step`의 값이 동일하면 메트릭이 동일한 스텝 아래에서 로그되어 함께 샘플링됩니다. 각 호출에서 step이 단조롭게 증가하지 않으면 `log()` 호출 중에 `step` 값이 무시됩니다.