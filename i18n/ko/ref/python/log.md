
# 로그

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/wandb_run.py#L1620-L1828' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>

현재 실행의 기록에 데이터 사전을 로그합니다.

```python
log(
    data: Dict[str, Any],
    step: Optional[int] = None,
    commit: Optional[bool] = None,
    sync: Optional[bool] = None
) -> None
```

`wandb.log`를 사용하여 실행에서 데이터를 로그하세요. 예를 들어 스칼라, 이미지, 비디오,
히스토그램, 플롯, 테이블 같은 것들입니다.

실제 예시, 코드조각, 모범 사례 등을 보려면 [로그 가이드](https://docs.wandb.ai/guides/track/log)를 참조하세요.

가장 기본적인 사용법은 `wandb.log({"train-loss": 0.5, "accuracy": 0.9})`입니다.
이렇게 하면 손실과 정확도가 실행의 기록에 저장되고 이 메트릭의 요약 값이 업데이트됩니다.

[wandb.ai](https://wandb.ai)에서 로그된 데이터를 시각화하거나, W&B 앱의 [자체 호스팅 인스턴스](https://docs.wandb.ai/guides/hosting)에서 로컬로 시각화하거나, Jupyter 노트북에서 로컬로 시각화하고 탐색하기 위해 데이터를 내보낼 수 있습니다. [API](https://docs.wandb.ai/guides/track/public-api-guide)를 참조하세요.

UI에서는 실행 테이블에서 실행 간의 단일 값 비교를 위해 요약 값이 표시됩니다.
요약 값은 `wandb.run.summary["키"] = 값`으로 직접 설정할 수도 있습니다.

로그된 값은 스칼라일 필요가 없습니다. wandb 오브젝트를 로깅하는 것이 지원됩니다.
예를 들어, `wandb.log({"example": wandb.Image("myimage.jpg")})`은 예제 이미지를 로깅하며 W&B UI에서 예쁘게 표시됩니다.
지원되는 다양한 유형에 대한 자세한 내용은 [참조 문서](https://docs.wandb.com/ref/python/data-types)를 참조하거나 3D 분자 구조와 분할 마스크부터 PR 곡선 및 히스토그램에 이르는 예시를 보려면 [로그 가이드](https://docs.wandb.ai/guides/track/log)를 확인하세요.
`wandb.Table`은 구조화된 데이터를 로깅하는 데 사용할 수 있습니다. 자세한 내용은 [테이블 로깅 가이드](https://docs.wandb.ai/guides/data-vis/log-tables)를 참조하세요.

중첩된 메트릭을 로깅하는 것이 권장되며 W&B UI에서 지원됩니다.
`wandb.log({"train": {"acc": 0.9}, "val": {"acc": 0.8}})`와 같이 중첩 사전으로 로그하면 메트릭이 W&B UI의 `train` 및 `val` 섹션으로 구성됩니다.

wandb는 기본적으로 `wandb.log` 호출 시마다 증가하는 전역 스텝을 추적합니다. 따라서 관련 메트릭을 함께 로깅하는 것이 권장됩니다.
관련 메트릭을 함께 로깅하기 불편한 경우
`wandb.log({"train-loss": 0.5}, commit=False)`를 호출한 다음
`wandb.log({"accuracy": 0.9})`를 호출하는 것은
`wandb.log({"train-loss": 0.5, "accuracy": 0.9})`를 호출하는 것과 동일합니다.

`wandb.log`는 초당 몇 번 이상 호출할 의도로 설계되지 않았습니다.
그보다 더 자주 로그하려면 클라이언트 측에서 데이터를 집계하는 것이 좋으며, 그렇지 않으면 성능이 저하될 수 있습니다.

| 인수 |  |
| :--- | :--- |
|  `data` |  (dict, 선택적) 직렬화 가능한 파이썬 오브젝트의 사전, 예: `str`, `ints`, `floats`, `Tensors`, `dicts`, 또는 `wandb.data_types` 중 하나. |
|  `commit` |  (boolean, 선택적) 메트릭 사전을 wandb 서버에 저장하고 스텝을 증가시킵니다. false인 경우 `wandb.log`는 현재 메트릭 사전을 데이터 인수로만 업데이트하며, `commit=True`로 `wandb.log`가 호출될 때까지 메트릭이 저장되지 않습니다. |
|  `step` |  (integer, 선택적) 처리할 글로벌 스텝입니다. 이전에 커밋되지 않은 모든 스텝이 유지되지만 지정된 스텝을 커밋하지 않는 것이 기본값입니다. |
|  `sync` |  (boolean, True) 이 인수는 더 이상 사용되지 않으며 현재 `wandb.log`의 동작을 변경하지 않습니다. |

#### 예시:

더 자세한 예시는 [로그 가이드](https://docs.wandb.com/guides/track/log)를 참조하세요.

### 기본 사용법




```python
import wandb

run = wandb.init()
run.log({"accuracy": 0.9, "epoch": 5})
```

### 증분 로깅




```python
import wandb

run = wandb.init()
run.log({"loss": 0.2}, commit=False)
# 나중에 이 단계를 보고할 준비가 되었을 때 어딘가에서:
run.log({"accuracy": 0.8})
```

### 히스토그램




```python
import numpy as np
import wandb

# 정규 분포에서 무작위로 그레이디언트를 샘플링합니다
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
# 축은 (시간, 채널, 높이, 너비)입니다
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
ax.plot(x, y)  # y = x^2 플롯
run.log({"chart": fig})
```

### PR 곡선

```python
import wandb

run = wandb.init()
run.log({"pr": wandb.plots.precision_recall(y_test, y_probas, labels)})
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

| 예외 |  |
| :--- | :--- |
|  `wandb.Error` |  `wandb.init` 호출 전에 사용된 경우 |
|  `ValueError` |  유효하지 않은 데이터가 전달된 경우 |