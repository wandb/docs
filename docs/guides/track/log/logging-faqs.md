---
title: Logging FAQ
description: 기계학습 Experiments 에서 W&B Experiments 로 데이터 추적에 대한 자주 묻는 질문에 대한 답변.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

### W&B UI에서 로그된 차트와 미디어를 어떻게 조직할 수 있나요?

우리는 `/`를 W&B UI에서 로그된 패널을 조직하기 위한 구분자로 사용합니다. 기본적으로, 로그된 항목 이름의 `/` 이전 부분은 "패널 섹션"이라고 불리는 패널 그룹을 정의하는 데 사용됩니다.

```python
wandb.log({"val/loss": 1.1, "val/acc": 0.3})
wandb.log({"train/loss": 0.1, "train/acc": 0.94})
```

[Workspace](../../app/pages/workspaces.md) 설정에서 패널이 첫 번째 구성 요소로만 그룹화될 것인지 또는 `/`로 분리된 모든 구성 요소로 그룹화될 것인지를 변경할 수 있습니다.

### 에포크 또는 스텝에 걸쳐 이미지를 비교하려면 어떻게 하나요?

스텝마다 이미지를 로그할 때마다 우리는 그것들을 UI에 표시하기 위해 저장합니다. 이미지 패널을 확장하고 스텝 슬라이더를 사용하여 다른 스텝의 이미지를 살펴볼 수 있습니다. 이를 통해 트레이닝 중에 모델의 출력이 어떻게 변화하는지 쉽게 비교할 수 있습니다.

### 배치에서 몇 가지 메트릭을 로그하고 에포크에서만 몇 가지 메트릭을 로그하려면 어떻게 하나요?

모든 배치에서 특정 메트릭을 로그하고 플롯을 표준화하려면, 메트릭과 함께 플롯하고 싶은 x축 값을 로그할 수 있습니다. 그런 다음 커스텀 플롯에서 편집을 선택하고 커스텀 x축을 선택하세요.

```python
wandb.log({"batch": batch_idx, "loss": 0.3})
wandb.log({"epoch": epoch, "val_acc": 0.94})
```

### 값 목록을 어떻게 로그하나요?

<Tabs
  defaultValue="dictionary"
  values={[
    {label: 'Using a dictionary', value: 'dictionary'},
    {label: 'As a histogram', value: 'histogram'},
  ]}>
  <TabItem value="dictionary">

```python
wandb.log({f"losses/loss-{ii}": loss for ii, loss in enumerate(losses)})
```
  </TabItem>
  <TabItem value="histogram">

```python
wandb.log({"losses": wandb.Histogram(losses)})  # 손실을 히스토그램으로 변환합니다
```
  </TabItem>
</Tabs>

### 범례가 있는 플롯에 여러 곡선을 어떻게 그리나요?

`wandb.plot.line_series()`를 사용하여 멀티라인 커스텀 차트를 생성할 수 있습니다. 라인 차트를 보기 위해 [프로젝트 페이지](../../app/pages/project-page.md)로 이동해야 합니다. 플롯에 범례를 추가하려면 `wandb.plot.line_series()`의 키 인수를 전달하세요. 예를 들어:

```python
wandb.log(
    {
        "my_plot": wandb.plot.line_series(
            xs=x_data, ys=y_data, keys=["metric_A", "metric_B"]
        )
    }
)
```

멀티라인 플롯에 대한 더 많은 정보는 [여기](../../track/log/plots.md#basic-charts) 멀티라인 탭 아래에서 찾을 수 있습니다.

### Plotly/Bokeh 차트를 테이블에 어떻게 추가하나요?

Plotly/Bokeh 피규어를 테이블에 직접 추가하는 것은 아직 지원되지 않습니다. 대신 피규어를 HTML로 작성하여 해당 HTML을 테이블에 추가하세요. 아래에 인터랙티브 Plotly 및 Bokeh 차트의 예시가 있습니다.

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

# 새로운 run을 초기화
run = wandb.init(project="log-plotly-fig-tables", name="plotly_html")

# 테이블 생성
table = wandb.Table(columns=["plotly_figure"])

# Plotly 피규어 경로 생성
path_to_plotly_html = "./plotly_figure.html"

# Plotly 예제 피규어
fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])

# Plotly 피규어를 HTML로 작성
# auto_play를 False로 설정하면 애니메이션 된 Plotly 차트가 테이블에서 자동으로 재생되지 않습니다
fig.write_html(path_to_plotly_html, auto_play=False)

# Plotly 피규어를 HTML 파일로 테이블에 추가
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

### 내 그래프에 아무것도 표시되지 않는 이유는 무엇인가요?

"아직 시각화 데이터가 로그되지 않았음"을 보고 있다면, 이는 스크립트에서 첫 번째 `wandb.log` 호출을 받지 않았음을 의미합니다. 이는 run이 스텝을 완료하는 데 시간이 오래 걸리기 때문일 수 있습니다. 에포크가 끝날 때마다 로그하려면 에포크마다 몇 번 로그하여 데이터 스트림을 더 빨리 볼 수 있도록 하세요.

### 동일한 메트릭이 여러 번 나타나는 이유는 무엇인가요?

동일한 키로 다른 유형의 데이터를 로그하면 데이터베이스에서 이를 분리해야 합니다. 이를 통해 UI의 드롭다운에서 동일한 메트릭 이름의 여러 항목을 볼 수 있습니다. 우리가 그룹화하는 유형은 `number`, `string`, `bool`, `other` (주로 배열) 및 `wandb` 데이터 유형 (`Histogram`, `Image`, 등)입니다. 이 행동을 피하기 위해 각 키에 한 유형만 보내세요.

우리는 메트릭을 대소문자 구분 없이 저장하므로, `"My-Metric"`과 `"my-metric"` 같은 이름의 메트릭 두 개를 갖고 있지 않도록 하세요.

### 실험의 로그된 데이터를 직접적으로 프로그래밍적으로 엑세스하는 방법은 무엇인가요?

history 오브젝트는 `wandb.log`에 의해 로그된 메트릭을 추적하는 데 사용됩니다. [우리의 API](../public-api-guide.md)를 사용하여 `run.history()`를 통해 history 오브젝트에 엑세스할 수 있습니다.

```python
api = wandb.Api()
run = api.run("username/project/run_id")
print(run.history())
```

### 수백만 개의 스텝을 W&B에 로그할 때 어떤 일이 발생하나요? 브라우저에서는 어떻게 렌더링되나요?

더 많은 포인트를 보내면 UI에서 그래프를 로드하는 데 더 오래 걸립니다. 1,000개 이상의 포인트가 라인에 있으면, 우리는 백엔드에서 1,000개로 샘플링하여 브라우저로 데이터를 전송합니다. 이 샘플링은 비결정적이므로 페이지를 새로고침할 때마다 다른 샘플링 포인트 세트를 보게 됩니다.

**가이드라인**

각 메트릭 당 10,000점 이하로 로그하는 것이 좋습니다. 한 라인에 100만 점 이상 로그하면 페이지를 로드하는 데 시간이 걸립니다. 로깅 정확도를 손상시키지 않고 로깅 크기를 줄이는 전략에 대한 더 많은 정보는 [이 Colab](http://wandb.me/log-hf-colab)을 참조하세요. 구성 및 요약 메트릭의 열이 500개 이상이면 테이블에서 500개만 표시됩니다.

### 프로젝트에 W&B를 통합하고 싶지만 이미지를 업로드하고 싶지 않을 때는 어떻게 하나요?

W&B는 스칼라만 로그하는 프로젝트에서도 사용할 수 있습니다 — 업로드하고 싶은 파일이나 데이터를 명시적으로 지정할 수 있습니다. 이미지를 로그하지 않는 [PyTorch 예제](http://wandb.me/pytorch-colab)를 참고하세요.

### wandb.log()에 클래스 속성을 전달하면 어떻게 되나요?

일반적으로 `wandb.log()`에 클래스 속성을 전달하는 것은 권장되지 않습니다. 속성이 네트워크 호출이 이루어지기 전에 변경될 수 있기 때문입니다. 클래스의 속성으로 메트릭을 저장하는 경우, `wandb.log()`가 호출될 당시의 속성 값이 메트릭으로 정확히 로그되도록 속성을 깊은 복사할 것을 권장합니다.

### 로그한 데이터 포인트가 예상보다 적은 이유는 무엇인가요?

X-Axis에서 `Step`이 아닌 다른 항목에 대해 메트릭을 시각화하면 예상보다 적은 데이터 포인트를 볼 수 있습니다. 이는 메트릭을 서로 플롯하기 위해 동일한 `Step`으로 로그되어야 하기 때문입니다. 즉, 우리는 동일한 `Step`에 로그된 메트릭만 샘플링하여 서로 중간 샘플을 인터폴레이션합니다.  
**가이드라인**  
우리는 메트릭을 동일한 `log()` 호출에 번들링할 것을 권장합니다.  
만약 코드가 다음과 같다면:

```python
wandb.log({"Precision": precision})
...
wandb.log({"Recall": recall})
```

다음과 같이 로그하는 것이 더 좋습니다:

```python
wandb.log({"Precision": precision, "Recall": recall})
```

또는 step 파라미터를 수동으로 제어하여 메트릭을 코드 내에서 동기화할 수 있습니다:

```python
wandb.log({"Precision": precision}, step=step)
...
wandb.log({"Recall": recall}, step=step)
```

`log()`에 대한 두 호출의 `step` 값이 동일하면 메트릭이 동일한 스텝에 로그되고 함께 샘플링됩니다. 각 호출에서 step이 단조롭게 증가해야 한다는 점을 유의하세요, 그렇지 않으면 `log()` 호출 중에 `step` 값이 무시됩니다.