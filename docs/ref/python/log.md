# 로그

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L1678-L1933' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>

run 데이터를 업로드합니다.

```python
log(
    data: Dict[str, Any],
    step: Optional[int] = None,
    commit: Optional[bool] = None,
    sync: Optional[bool] = None
) -> None
```

`log`를 사용하여 run에서 스칼라, 이미지, 비디오, 히스토그램, 플롯, 테이블과 같은 데이터를 로그합니다.

라이브 예제, 코드조각, 모범 사례 등을 보려면 [로그 가이드](https://docs.wandb.ai/guides/track/log)를 참조하십시오.

가장 기본적인 사용법은 `run.log({"train-loss": 0.5, "accuracy": 0.9})`입니다. 이는 손실과 정확도를 run의 기록에 저장하고 이러한 메트릭의 요약 값을 업데이트합니다.

[wandb.ai](https://wandb.ai)의 워크스페이스에서 로그된 데이터를 시각화하거나 [자가 호스팅 인스턴스](https://docs.wandb.ai/guides/hosting)에서 로컬로 시각화하거나 탐색할 데이터를 내보낼 수 있습니다. 예를 들어 Jupyter 노트북에서 [우리의 API](https://docs.wandb.ai/guides/track/public-api-guide)를 사용하여 가능합니다.

로그된 값이 스칼라일 필요는 없습니다. 모든 wandb 오브젝트를 로깅할 수 있습니다. 예를 들어 `run.log({"example": wandb.Image("myimage.jpg")})`는 예제 이미지를 로그하여 W&B UI에서 멋지게 표시됩니다. 지원되는 다양한 유형에 대한 모든 정보를 확인하려면 [참조 문서](https://docs.wandb.com/ref/python/data-types)를 참조하거나 3D 분자 구조와 세분화 마스크에서 PR 커브 및 히스토그램에 이르는 예제를 확인하려면 [로그 가이드](https://docs.wandb.ai/guides/track/log)를 참조하십시오. 구조화된 데이터를 로그하려면 `wandb.Table`을 사용할 수 있습니다. 자세한 내용은 [테이블 로그 가이드](https://docs.wandb.ai/guides/data-vis/log-tables)를 참조하십시오.

W&B UI는 이름에 슬래시(`/`)가 포함된 메트릭을 메트릭 이름의 마지막 슬래시 전 텍스트를 사용하여 섹션으로 정리합니다. 예를 들어, 다음은 "train"과 "validate"라는 두 섹션을 생성합니다:

```
run.log({
    "train/accuracy": 0.9,
    "train/loss": 30,
    "validate/accuracy": 0.8,
    "validate/loss": 20,
})
```

중첩은 한 단계만 지원됩니다. `run.log({"a/b/c": 1})`는 "a/b"라는 섹션을 생성합니다.

`run.log`는 초당 몇 번 이상 호출하는 것을 의도하지 않았습니다. 최적의 성능을 위해 로깅을 N 회 반복마다 한 번으로 제한하거나 여러 반복에 걸쳐 데이터를 수집하고 한 번에 로그하십시오.

### W&B 스텝

기본 사용법에서는 `log` 호출마다 새로운 "스텝"이 생성됩니다. 스텝은 항상 증가해야 하고 이전 단계에 로그할 수 없습니다.

차트에서 X 축으로 사용할 수 있는 모든 메트릭을 사용할 수 있습니다. 많은 경우 W&B 스텝을 타임스탬프로 취급하는 것이 트레이닝 스텝으로 취급하는 것보다 낫습니다.

```
# 예시: X 축으로 사용할 "epoch" 메트릭을 로그합니다.
run.log({"epoch": 40, "train-loss": 0.5})
```

또한 [define_metric](https://docs.wandb.ai/ref/python/run#define_metric)도 참조하십시오.

`step` 및 `commit` 파라미터를 사용하여 같은 스텝에 로그하도록 여러 `log` 호출을 사용할 수 있습니다. 다음은 모두 동등합니다:

```
# 일반적인 사용법:
run.log({"train-loss": 0.5, "accuracy": 0.8})
run.log({"train-loss": 0.4, "accuracy": 0.9})

# 자동 증가가 없는 암묵적 스텝:
run.log({"train-loss": 0.5}, commit=False)
run.log({"accuracy": 0.8})
run.log({"train-loss": 0.4}, commit=False)
run.log({"accuracy": 0.9})

# 명시적 스텝:
run.log({"train-loss": 0.5}, step=current_step)
run.log({"accuracy": 0.8}, step=current_step)
current_step += 1
run.log({"train-loss": 0.4}, step=current_step)
run.log({"accuracy": 0.9}, step=current_step)
```

| 인수 |  |
| :--- | :--- |
|  `data` |  `str` 키와 정수, 부동소수점, 문자열 등 직렬화 가능한 Python 객체를 포함하는 값이 있는 `dict`. 또한 `wandb.data_types`의 모든 유형, 직렬화 가능한 Python 객체의 리스트, 튜플 및 NumPy 배열, 또는 이 구조의 다른 `dict`를 포함합니다. |
|  `step` |  로그할 스텝 번호입니다. `None`인 경우 암묵적으로 자동 증가하는 스텝을 사용합니다. 설명에 있는 노트를 참조하십시오. |
|  `commit` |  true인 경우 스텝을 완료하고 업로드합니다. false인 경우 스텝에 대한 데이터를 누적합니다. 설명에 있는 노트를 참조하십시오. `step`이 `None`인 경우 기본값은 `commit=True`이고, 그렇지 않으면 기본값은 `commit=False`입니다. |
|  `sync` |  이 인수는 더 이상 사용되지 않으며 아무 작업도 수행하지 않습니다. |

#### 예제:

더 많고 자세한 예제를 보려면 [로그 가이드](https://docs.wandb.com/guides/track/log)를 참조하십시오.

### 기본 사용법

```python
import wandb

run = wandb.init()
run.log({"accuracy": 0.9, "epoch": 5})
```

### 증분 로그

```python
import wandb

run = wandb.init()
run.log({"loss": 0.2}, commit=False)
# 이 스텝을 보고할 준비가 됐을 때 다른 곳에서:
run.log({"accuracy": 0.8})
```

### 히스토그램

```python
import numpy as np
import wandb

# 정규분포에서 무작위로 그레이디언트 샘플링
gradients = np.random.randn(100, 100)
run = wandb.init()
run.log({"gradients": wandb.Histogram(gradients)})
```

### numpy에서 이미지

```python
import numpy as np
import wandb

run = wandb.init()
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

run = wandb.init()
examples = []
for i in range(3):
    pixels = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
    pil_image = PILImage.fromarray(pixels, mode="RGB")
    image = wandb.Image(pil_image, caption=f"random field {i}")
    examples.append(image)
run.log({"examples": examples})
```

### numpy에서 비디오

```python
import numpy as np
import wandb

run = wandb.init()
# 축은 (time, channel, height, width)
frames = np.random.randint(low=0, high=256, size=(10, 3, 100, 100), dtype=np.uint8)
run.log({"video": wandb.Video(frames, fps=4)})
```

### Matplotlib 플롯

```python
from matplotlib import pyplot as plt
import numpy as np
import wandb

run = wandb.init()
fig, ax = plt.subplots()
x = np.linspace(0, 10)
y = x * x
ax.plot(x, y)  # y = x^2를 플롯
run.log({"chart": fig})
```

### PR 곡선

```python
import wandb

run = wandb.init()
run.log({"pr": wandb.plot.pr_curve(y_test, y_probas, labels)})
```

### 3D 오브젝트

```python
import wandb

run = wandb.init()
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

| 발생 |  |
| :--- | :--- |
|  `wandb.Error` |  `wandb.init`가 호출되기 전에 호출된 경우 |
|  `ValueError` |  잘못된 데이터가 전달된 경우 |