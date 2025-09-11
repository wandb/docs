---
title: log
menu:
  reference:
    identifier: ko-ref-python-log
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L1613-L1873 >}}

run 데이터를 업로드합니다.

```python
log(
    data: dict[str, Any],
    step: (int | None) = None,
    commit: (bool | None) = None,
    sync: (bool | None) = None
) -> None
```

`log` 를 사용하여 스칼라, 이미지, 비디오, 히스토그램, 플롯 및 테이블과 같은 run의 데이터를 기록합니다.

실시간 예제, 코드 조각, 모범 사례 등에 대한 [로깅 가이드](https://docs.wandb.ai/guides/track/log) 를 참조하세요.

가장 기본적인 사용법은 `run.log({"train-loss": 0.5, "accuracy": 0.9})` 입니다. 이렇게 하면 손실 및 정확도가 run의 기록에 저장되고 이러한 메트릭에 대한 요약 값이 업데이트됩니다.

[wandb.ai](https://wandb.ai) 의 워크스페이스, 또는 W&B 앱의 [자체 호스팅 인스턴스](https://docs.wandb.ai/guides/hosting) 에서 기록된 데이터를 시각화하거나, [API](https://docs.wandb.ai/guides/track/public-api-guide) 를 사용하여 데이터를 내보내 로컬에서 시각화하고 탐색합니다 (예: Jupyter 노트북).

기록된 값은 스칼라일 필요가 없습니다. 모든 wandb 오브젝트 로깅이 지원됩니다. 예를 들어 `run.log({"example": wandb.Image("myimage.jpg")})` 는 W&B UI에 멋지게 표시될 예제 이미지를 기록합니다. 지원되는 다양한 유형에 대한 [참조 문서](https://docs.wandb.ai/ref/python/sdk/data-types/) 를 참조하거나 3D 분자 구조 및 분할 마스크에서 PR 곡선 및 히스토그램에 이르기까지 예제에 대한 [로깅 가이드](https://docs.wandb.ai/guides/track/log) 를 확인하세요. `wandb.Table` 을 사용하여 구조화된 데이터를 기록할 수 있습니다. 자세한 내용은 [테이블 로깅 가이드](https://docs.wandb.ai/guides/models/tables/tables-walkthrough) 를 참조하세요.

W&B UI는 이름에 슬래시 (`/`) 가 있는 메트릭을 마지막 슬래시 앞의 텍스트를 사용하여 명명된 섹션으로 구성합니다. 예를 들어 다음은 "train" 및 "validate" 라는 두 개의 섹션을 생성합니다.

```
run.log(
    {
        "train/accuracy": 0.9,
        "train/loss": 30,
        "validate/accuracy": 0.8,
        "validate/loss": 20,
    }
)
```

한 레벨의 중첩만 지원됩니다. `run.log({"a/b/c": 1})` 는 "a/b" 라는 섹션을 생성합니다.

`run.log` 는 초당 몇 번 이상 호출하도록 설계되지 않았습니다. 최적의 성능을 위해 로깅을 N번 반복할 때마다 한 번으로 제한하거나 여러 반복에 걸쳐 데이터를 수집하고 단일 단계로 기록합니다.

### W&B 단계

기본적인 사용법으로, `log` 를 호출할 때마다 새로운 "단계" 가 생성됩니다. 단계는 항상 증가해야 하며 이전 단계에 기록하는 것은 불가능합니다.

차트에서 모든 메트릭을 X축으로 사용할 수 있습니다. 대부분의 경우 W&B 단계를 트레이닝 단계가 아닌 타임스탬프처럼 취급하는 것이 좋습니다.

```
# 예시: X축으로 사용하기 위해 "epoch" 메트릭을 기록합니다.
run.log({"epoch": 40, "train-loss": 0.5})
```

[define_metric](https://docs.wandb.ai/ref/python/sdk/experiments/run/#method-rundefine_metric) 도 참조하세요.

여러 `log` 호출을 사용하여 `step` 및 `commit` 파라미터로 동일한 단계에 기록할 수 있습니다. 다음은 모두 동일합니다.

```
# 일반적인 사용법:
run.log({"train-loss": 0.5, "accuracy": 0.8})
run.log({"train-loss": 0.4, "accuracy": 0.9})

# 자동 증가 없이 암시적 단계:
run.log({"train-loss": 0.5}, commit=False)
run.log({"accuracy": 0.8})
run.log({"train-loss": 0.4}, commit=False)
run.log({"accuracy": 0.9})

# 명시적 단계:
run.log({"train-loss": 0.5}, step=current_step)
run.log({"accuracy": 0.8}, step=current_step)
current_step += 1
run.log({"train-loss": 0.4}, step=current_step)
run.log({"accuracy": 0.9}, step=current_step)
```

| Args |  |
| :--- | :--- |
|  `data` |  `str` 키와 직렬화 가능한 Python 오브젝트 (예: `int`, `float` 및 `string`; `wandb.data_types`; 직렬화 가능한 Python 오브젝트의 목록, 튜플 및 NumPy 배열; 이 구조의 다른 `dict`) 가 있는 `dict` 입니다. |
|  `step` |  기록할 단계 번호입니다. `None` 인 경우 암시적 자동 증가 단계가 사용됩니다. 설명의 메모를 참조하세요. |
|  `commit` |  true이면 단계를 완료하고 업로드합니다. false이면 단계에 대한 데이터를 누적합니다. 설명의 메모를 참조하세요. `step` 이 `None` 이면 기본값은 `commit=True` 입니다. 그렇지 않으면 기본값은 `commit=False` 입니다. |
|  `sync` |  이 인수는 더 이상 사용되지 않으며 아무 작업도 수행하지 않습니다. |

#### 예시:

더 자세한 예는 [로깅 가이드](https://docs.wandb.ai/guides/track/log) 를 참조하세요.

### 기본 사용법

```python
import wandb

with wandb.init() as run:
    run.log({"accuracy": 0.9, "epoch": 5})
```

### 증분 로깅

```python
import wandb

with wandb.init() as run:
    run.log({"loss": 0.2}, commit=False)
    # 이 단계를 보고할 준비가 되면 다른 곳에서:
    run.log({"accuracy": 0.8})
```

### 히스토그램

```python
import numpy as np
import wandb

# 정규 분포에서 임의로 그레이디언트 샘플링
gradients = np.random.randn(100, 100)
with wandb.init() as run:
    run.log({"gradients": wandb.Histogram(gradients)})
```

### numpy에서 이미지

```python
import numpy as np
import wandb

with wandb.init() as run:
    examples = []
    for i in range(3):
        pixels = np.random.randint(low=0, high=256, size=(100, 100, 3))
        image = wandb.Image(pixels, caption=f"random field {i}")
        examples.append(image)
    run.log({"examples": examples})
```

### PIL에서 이미지

```python
import numpy as np
from PIL import Image as PILImage
import wandb

with wandb.init() as run:
    examples = []
    for i in range(3):
        pixels = np.random.randint(
            low=0,
            high=256,
            size=(100, 100, 3),
            dtype=np.uint8,
        )
        pil_image = PILImage.fromarray(pixels, mode="RGB")
        image = wandb.Image(pil_image, caption=f"random field {i}")
        examples.append(image)
    run.log({"examples": examples})
```

### numpy에서 비디오

```python
import numpy as np
import wandb

with wandb.init() as run:
    # 축은 (시간, 채널, 높이, 너비) 입니다.
    frames = np.random.randint(
        low=0,
        high=256,
        size=(10, 3, 100, 100),
        dtype=np.uint8,
    )
    run.log({"video": wandb.Video(frames, fps=4)})
```

### Matplotlib 플롯

```python
from matplotlib import pyplot as plt
import numpy as np
import wandb

with wandb.init() as run:
    fig, ax = plt.subplots()
    x = np.linspace(0, 10)
    y = x * x
    ax.plot(x, y)  # 플롯 y = x^2
    run.log({"chart": fig})
```

### PR 곡선

```python
import wandb

with wandb.init() as run:
    run.log({"pr": wandb.plot.pr_curve(y_test, y_probas, labels)})
```

### 3D 오브젝트

```python
import wandb

with wandb.init() as run:
    run.log(
        {
            "generated_samples": [
                wandb.Object3D(open("sample.obj")),
                wandb.Object3D(open("sample.gltf")),
                wandb.Object3D(open("sample.glb")),
            ]
        }
    )
```

| Raises |  |
| :--- | :--- |
|  `wandb.Error` |  `wandb.init` 전에 호출된 경우 |
|  `ValueError` |  잘못된 데이터가 전달된 경우 |
```