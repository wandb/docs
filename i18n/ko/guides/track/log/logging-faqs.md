---
description: Answers to frequently asked questions about tracking data from machine
  learning experiments with W&B Experiments.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 로깅 FAQ

<head>
  <title>실험에서 데이터를 로깅하는 것에 관한 자주 묻는 질문</title>
</head>

### W&B UI에서 로그된 차트와 미디어를 어떻게 구성할 수 있나요?

W&B UI에서 로그된 패널들을 구성하기 위해 `/`를 구분자로 사용합니다. 기본적으로 `/` 이전의 로그된 항목 이름의 구성 요소를 "패널 섹션"이라고 하는 패널 그룹을 정의하는 데 사용됩니다.

```python
wandb.log({"val/loss": 1.1, "val/acc": 0.3})
wandb.log({"train/loss": 0.1, "train/acc": 0.94})
```

[워크스페이스](../../app/pages/workspaces.md) 설정에서 패널이 `/`로 분리된 첫 번째 구성 요소나 모든 구성 요소에 따라 그룹화되는지 여부를 변경할 수 있습니다.

### 에포크 또는 스텝별로 이미지나 미디어를 어떻게 비교할 수 있나요?

스텝에서 이미지를 로깅할 때마다 UI에서 보여줄 수 있도록 이미지를 저장합니다. 이미지 패널을 확장하고 스텝 슬라이더를 사용하여 다른 스텝의 이미지를 봅니다. 이를 통해 모델의 출력이 트레이닝 동안 어떻게 변하는지 쉽게 비교할 수 있습니다.

### 일부 메트릭은 배치별로, 일부 메트릭은 에포크별로만 로깅하고 싶다면 어떻게 해야 하나요?

모든 배치에서 특정 메트릭을 로깅하고 플롯을 표준화하려면, 메트릭과 함께 플롯하려는 x 축 값들을 로깅할 수 있습니다. 그런 다음 사용자 정의 플롯에서 편집을 클릭하고 사용자 지정 x 축을 선택하십시오.

```python
wandb.log({"batch": batch_idx, "loss": 0.3})
wandb.log({"epoch": epoch, "val_acc": 0.94})
```

### 값 목록을 어떻게 로깅하나요?

<Tabs
  defaultValue="dictionary"
  values={[
    {label: '사전을 사용하여', value: 'dictionary'},
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

### 범례가 있는 플롯에서 여러 줄을 어떻게 그릴 수 있나요?

`wandb.plot.line_series()`를 사용하여 다중 선 사용자 정의 차트를 생성할 수 있습니다. 선 차트를 보려면 [프로젝트 페이지](../../app/pages/project-page.md)로 이동해야 합니다. 범례를 플롯에 추가하려면, `wandb.plot.line_series()` 내에 keys 인수를 전달하십시오. 예를 들면:

```python
wandb.log(
    {
        "my_plot": wandb.plot.line_series(
            xs=x_data, ys=y_data, keys=["metric_A", "metric_B"]
        )
    }
)
```

다중 줄 플롯에 대한 자세한 정보는 [여기](../../track/log/plots.md#basic-charts)에서 다중 줄 탭 아래에서 찾을 수 있습니다.

### 테이블에 Plotly/Bokeh 차트를 어떻게 추가하나요?

테이블에 직접 Plotly/Bokeh 그림을 추가하는 것은 아직 지원되지 않습니다. 대신, 그림을 HTML로 작성한 다음 HTML을 테이블에 추가하십시오. 아래는 상호 작용하는 Plotly 및 Bokeh 차트의 예입니다.

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

# 새로운 run 초기화
run = wandb.init(project="log-plotly-fig-tables", name="plotly_html")

# 테이블 생성
table = wandb.Table(columns=["plotly_figure"])

# Plotly 그림 경로 생성
path_to_plotly_html = "./plotly_figure.html"

# Plotly 그림 예시
fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])

# Plotly 그림을 HTML로 작성
# auto_play를 False로 설정하면 Plotly
# 차트가 테이블에서 자동으로 재생되는 것을 방지합니다
fig.write_html(path_to_plotly_html, auto_play=False)

# 테이블에 HTML 파일로 Plotly 그림 추가
table.add_data(wandb.Html(path_to_plotly_html))

# 테이블 로깅
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

### 그래프에 아무것도 나타나지 않는 이유는 무엇인가요?

"아직 로깅된 시각화 데이터가 없습니다"라는 메시지가 표시되면 스크립트에서 첫 번째 `wandb.log` 호출을 아직 받지 못했다는 의미입니다. 이는 실행이 한 단계를 마치는 데 시간이 오래 걸릴 수 있기 때문입니다. 에포크마다 끝에 로깅하고 있다면 에포크마다 몇 번 로깅하여 데이터가 더 빨리 스트리밍되도록 할 수 있습니다.

### 같은 메트릭이 여러 번 나타나는 이유는 무엇인가요?

동일한 키에 대해 다양한 유형의 데이터를 로깅하는 경우 데이터베이스에서 분리해야 합니다. 이는 UI의 드롭다운에서 동일한 메트릭 이름의 여러 항목을 볼 수 있음을 의미합니다. 우리가 그룹화하는 유형은 `number`, `string`, `bool`, `other` (대부분 배열), 그리고 모든 `wandb` 데이터 유형(`Histogram`, `Image` 등)입니다. 이러한 행동을 피하려면 각 키에 하나의 유형만 보내십시오.

메트릭을 대소문자 구분 없이 저장하므로, `"My-Metric"`과 `"my-metric"`과 같은 이름의 두 메트릭이 없는지 확인하십시오.

### 실행에 로깅된 데이터에 직접적이고 프로그래매틱하게 어떻게 엑세스할 수 있나요?

history 오브젝트는 `wandb.log`에 의해 로깅된 메트릭을 추적하는 데 사용됩니다. [우리의 API](../public-api-guide.md)를 사용하면 `run.history()`를 통해 history 오브젝트에 엑세스할 수 있습니다.

```python
api = wandb.Api()
run = api.run("username/project/run_id")
print(run.history())
```

### W&B에 수백만 스텝을 로깅하면 어떻게 될까요? 브라우저에서는 어떻게 렌더링될까요?

저희에게 보내는 포인트가 많을수록 UI에서 그래프를 로드하는 데 더 오래 걸립니다. 선 위에 1000개 이상의 포인트가 있는 경우, 데이터를 브라우저로 보내기 전에 백엔드에서 1000개의 포인트로 샘플링합니다. 이 샘플링은 비결정적이므로 페이지를 새로고침하면 다른 샘플링된 포인트 세트를 볼 수 있습니다.

**가이드라인**

메트릭 당 10,000개 이하의 포인트를 로깅하려고 노력하는 것이 좋습니다. 선에 100만 개 이상의 포인트를 로깅하면 페이지를 로드하는 데 시간이 걸립니다. 로깅 발자국을 줄이면서 정확성을 희생하지 않는 전략에 대한 자세한 내용은 [이 Colab](http://wandb.me/log-hf-colab)을 확인하세요. 설정 및 요약 메트릭의 열이 500개 이상인 경우 테이블에 500개만 표시됩니다.

### 이미지나 미디어를 업로드하지 않고 프로젝트에 W&B를 통합하고 싶다면 어떻게 해야 하나요?

W&B는 스칼라만 로깅하는 프로젝트에도 사용할 수 있습니다 — 업로드하고자 하는 파일이나 데이터를 명시적으로 지정합니다. 여기 [PyTorch에서의 빠른 예](http://wandb.me/pytorch-colab)가 있으며, 이미지를 로깅하지 않습니다.

### wandb.log()에 클래스 속성을 전달하면 어떻게 되나요?

일반적으로 `wandb.log()`에 클래스 속성을 전달하는 것은 권장되지 않습니다. 네트워크 호출이 이루어지기 전에 속성이 변경될 수 있기 때문입니다. 클래스의 속성으로 메트릭을 저장하는 경우, `wandb.log()`가 호출되었을 때 속성의 값과 일치하는 메트릭이 로깅되도록 속성을 깊은 복사하는 것이 권장됩니다.

### 로깅한 데이터 포인트가 예상보다 적은 이유는 무엇인가요?

X 축에 `Step` 이외의 것을 사용하여 메트릭을 시각화하는 경우 예상보다 데이터 포인트가 적게 보일 수 있습니다. 이는 메트릭이 서로 대응하여 로깅되어야 하기 때문입니다. 즉, 메트릭을 동기화하는 방법은 동일한 `Step`에 로깅된 메트릭만 샘플링하고 샘플 사이에서 보간하는 것입니다.\
\
**가이드라인**\
****\
****메트릭을 동일한 `log()` 호출로 번들링하는 것이 좋습니다. 코드가 다음과 같다면:

```python
wandb.log({"Precision": precision})
...
wandb.log({"Recall": recall})
```

다음과 같이 로깅하는 것이 더 좋습니다:

```python
wandb.log({"Precision": precision, "Recall": recall})
```

또는, step 파라미터를 수동으로 제어하고 코드에서 메트릭을 동기화할 수 있습니다:

```python
wandb.log({"Precision": precision}, step=step)
...
wandb.log({"Recall": recall}, step=step)
```

`log()` 호출에서 `step` 값이 동일하면, 메트릭은 동일한 스텝 아래에서 로깅되어 함께 샘플링됩니다. 단, 각 호출에서 step은 단조롭게 증가해야 하며, 그렇지 않으면 `log()` 호출 시 `step` 값이 무시됩니다.