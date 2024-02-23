
# 실행

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/wandb_run.py#L432-L4041' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>

wandb에 의해 기록된 계산의 단위입니다. 일반적으로 이는 ML 실험입니다.

```python
Run(
    settings: Settings,
    config: Optional[Dict[str, Any]] = None,
    sweep_config: Optional[Dict[str, Any]] = None,
    launch_config: Optional[Dict[str, Any]] = None
) -> None
```

`wandb.init()`으로 실행을 생성합니다:

```python
import wandb

run = wandb.init()
```

어떠한 프로세스에서도 활성화된 `wandb.Run`은 하나만 존재하며, `wandb.run`으로 접근 가능합니다:

```python
import wandb

assert wandb.run is None

wandb.init()

assert wandb.run is not None
```

`wandb.log`를 사용하여 로그에 기록하는 모든 것은 해당 실행으로 전송됩니다.

동일한 스크립트나 노트북에서 더 많은 실행을 시작하려면, 진행 중인 실행을 완료해야 합니다. 실행은 `wandb.finish`를 사용하거나 `with` 블록에서 사용하여 완료할 수 있습니다:

```python
import wandb

wandb.init()
wandb.finish()

assert wandb.run is None

with wandb.init() as run:
    pass  # 여기서 데이터를 로그에 기록

assert wandb.run is None
```

실행을 생성하는 방법에 대해 자세히 알아보려면 `wandb.init` 문서를 확인하거나 [우리의 `wandb.init` 가이드](https://docs.wandb.ai/guides/track/launch)를 확인하세요.

분산 학습에서는 랭크 0 프로세스에서 단일 실행을 생성한 다음 해당 프로세스에서만 정보를 로그하거나, 각 프로세스에서 실행을 생성하고 각각을 별도로 로깅한 후 `wandb.init`에 `group` 인수를 사용하여 결과를 그룹화할 수 있습니다. W&B를 사용한 분산 학습에 대한 자세한 내용은 [우리의 가이드](https://docs.wandb.ai/guides/track/log/distributed-training)를 확인하세요.

현재, `wandb.Api`에서 병렬 `Run` 객체가 있습니다. 결국 이 두 객체는 통합될 것입니다.

| 속성 |  |
| :--- | :--- |
|  `summary` |  각 `wandb.log()` 키에 대해 설정된 단일 값입니다. 기본적으로 summary는 마지막에 로그된 값으로 설정됩니다. 최고의 값, 예를 들어 최대 정확도와 같은 summary를 수동으로 설정할 수 있습니다. |
|  `config` |  이 실행과 관련된 설정 객체입니다. |
|  `dir` |  실행과 관련된 파일이 저장되는 디렉터리입니다. |
|  `entity` |  실행과 관련된 W&B 엔티티의 이름입니다. 엔티티는 사용자 이름이거나 팀 또는 조직의 이름일 수 있습니다. |
|  `group` |  실행과 관련된 그룹의 이름입니다. 그룹을 설정하면 W&B UI가 실행을 합리적인 방식으로 구성하는 데 도움이 됩니다. 분산 학습을 수행하는 경우 학습의 모든 실행에 동일한 그룹을 지정해야 합니다. 교차 검증을 수행하는 경우 모든 교차 검증 폴드에 동일한 그룹을 지정해야 합니다. |
|  `id` |  이 실행의 식별자입니다. |
|  `mode` |  `0.9.x` 및 이전 버전과의 호환성을 위해, 결국 폐기될 예정입니다. |
|  `name` |  실행의 표시 이름입니다. 표시 이름은 고유하지 않을 수 있으며 설명적일 수 있습니다. 기본적으로 무작위로 생성됩니다. |
|  `notes` |  실행과 관련된 메모가 있는 경우입니다. 메모는 여러 줄 문자열일 수 있으며 `$$` 내에서 마크다운과 라텍스 수식을 사용할 수 있습니다, 예를 들어 `$x + 3$`. |
|  `path` |  실행의 경로입니다. 실행 경로에는 엔티티, 프로젝트 및 실행 ID가 포함되며, 형식은 `entity/project/run_id`입니다. |
|  `project` |  실행과 관련된 W&B 프로젝트의 이름입니다. |
|  `resumed` |  실행이 재개된 경우 True, 그렇지 않으면 False입니다. |
|  `settings` |  실행의 설정 객체의 동결된 복사본입니다. |
|  `start_time` |  실행이 시작된 때의 유닉스 타임스탬프(초 단위)입니다. |
|  `starting_step` |  실행의 첫 번째 단계입니다. |
|  `step` |  단계의 현재 값입니다. 이 카운터는 `wandb.log`에 의해 증가됩니다. |
|  `sweep_id` |  실행과 관련된 스윕의 ID입니다(있는 경우). |
|  `tags` |  실행과 관련된 태그가 있는 경우입니다. |
|  `url` |  실행과 관련된 W&B URL입니다. |

## 메서드

### `alert`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/wandb_run.py#L3338-L3371)

```python
alert(
    title: str,
    text: str,
    level: Optional[Union[str, 'AlertLevel']] = None,
    wait_duration: Union[int, float, timedelta, None] = None
) -> None
```

주어진 제목과 텍스트로 알림을 발생시킵니다.

| 인수 |  |
| :--- | :--- |
|  `title` |  (str) 알림의 제목, 64자를 초과하지 않아야 합니다. |
|  `text` |  (str) 알림의 텍스트 본문입니다. |
|  `level` |  (str 또는 wandb.AlertLevel, 선택사항) 사용할 알림 레벨, `INFO`, `WARN`, 또는 `ERROR` 중 하나입니다. |
|  `wait_duration` |  (int, float, 또는 timedelta, 선택사항) 이 제목으로 다른 알림을 보내기 전에 기다리는 시간(초)입니다. |

### `define_metric`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/wandb_run.py#L2543-L2577)

```python
define_metric(
    name: str,
    step_metric: Union[str, wandb_metric.Metric, None] = None,
    step_sync: Optional[bool] = None,
    hidden: Optional[bool] = None,
    summary: Optional[str] = None,
    goal: Optional[str] = None,
    overwrite: Optional[bool] = None,
    **kwargs
) -> wandb_metric.Metric
```

`wandb.log()`로 나중에 로그될 메트릭 속성을 정의합니다.

| 인수 |  |
| :--- | :--- |
|  `name` |  메트릭의 이름입니다. |
|  `step_metric` |  메트릭과 관련된 독립 변수입니다. |
|  `step_sync` |  필요한 경우 자동으로 `step_metric`을 기록에 추가합니다. step_metric이 지정된 경우 기본적으로 True입니다. |
|  `hidden` |  이 메트릭을 자동 플롯에서 숨깁니다. |
|  `summary` |  요약에 추가할 집계 메트릭을 지정합니다. 지원되는 집계: "min,max,mean,best,last,none" 기본 집계는 `copy`입니다. 집계 `best`는 `goal`==`minimize`로 기본 설정됩니다. |
|  `goal` |  메트릭을 최적화하는 방향을 지정합니다. 지원되는 방향: "minimize,maximize" |

| 반환값 |  |
| :--- | :--- |
|  추가로 지정할 수 있는 메트릭 객체가 반환됩니다. |

### `detach`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/wandb_run.py#L2699-L2700)

```python
detach() -> None
```

### `display`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/wandb_run.py#L1310-L1318)

```python
display(
    height: int = 420,
    hidden: bool = (False)
) -> bool
```

이 실행을 주피터에서 표시합니다.

### `finish`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/wandb_run.py#L1947-L1961)

```python
finish(
    exit_code: Optional[int] = None,
    quiet: Optional[bool] = None
) -> None
```

실행을 완료로 표시하고 모든 데이터의 업로드를 완료합니다.

이 메서드는 스크립트가 종료될 때나 실행 컨텍스트 관리자를 사용할 때 자동으로 호출됩니다.

| 인수 |  |
| :--- | :--- |
|  `exit_code` |  0이 아닌 것으로 설정하여 실행을 실패로 표시합니다 |
|  `quiet` |  로그 출력을 최소화하려면 true로 설정하세요 |

### `finish_artifact`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/wandb_run.py#L2953-L3005)

```python
finish_artifact(
    artifact_or_path: Union[Artifact, str],
    name: Optional[str] = None,
    type: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    distributed_id: Optional[str] = None
) -> Artifact
```

미완성 아티팩트를 실행의 결과물로 완성합니다.

동일한 분산 ID로 "upserts"하는 후속 작업은 새 버전을 생성합니다.

| 인수 |  |
| :--- | :--- |
|  `artifact_or_path` |  (str 또는 Artifact) 이 아티팩트의 내용에 대한 경로, 다음 형식 중 하나일 수 있습니다: - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` 또는 `wandb.Artifact`를 호출하여 생성된 Artifact 객체도 전달할 수 있습니다. |
|  `name` |  (str, 선택사항) 아티팩트의 이름입니다. 엔티티/프로젝트로 접두사가 붙을 수 있습니다. 유효한 이름 형식은 다음과 같습니다: - name:version - name:alias - digest 지정하지 않으면 현재 실행 ID가 접두사로 붙은 경로의 기본 이름으로 기본 설정됩니다. |
|  `type` |  (str) 로그할 아티팩트의 유형입니다. 예를 들어 `dataset`, `model` |
|  `aliases` |  (list, 선택사항) 이 아티팩트에 적용할 별칭으로, 기본값은 `["latest"]`입니다. |
|  `distributed_id` |  (string, 선택사항) 모든 분산 작업이 공유하는 고유 문자열입니다. None인 경우 실행의 그룹 이름으로 기본 설정됩니다. |

| 반환값 |  |
| :--- | :--- |
|  `Artifact` 객체입니다. |

### `get_project_url`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/wandb_run.py#L1192-L1200)

```python
get_project_url() -> Optional[str]
```

실행과 관련된 W&B 프로젝트의 URL을 반환합니다(있는 경우).

오프라인 실행은 프로젝트 URL이 없습니다.

### `get_sweep_url`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/wandb_run.py#L1202-L1207)

```python
get_sweep_url() -> Optional[str]
```

실행과 관련된 스윕의 URL을 반환합니다(있는 경우).

### `get_url`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/wandb_run.py#L1182-L1190)

```python
get_url() -> Optional[str]
```

W&B 실행과 관련된 URL을 반환합니다(있는 경우).

오프라인 실행은 URL이 없습니다.

### `join`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/wandb_run.py#L1995-L2005)

```python
join(
    exit_code: Optional[int] = None
) -> None
```

`finish()`의 사용되지 않는 별칭입니다 - 대신 finish를 사용하세요.

### `link_artifact`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/wandb_run.py#L2702-L2751)

```python
link_artifact(
    artifact: Artifact,
    target_path: str,
    aliases: Optional[List[str]] = None
) -> None
```

주어진 아티팩트를 포트폴리오(승진된 아티팩트 컬렉션)에 연결합니다.

연결된 아티팩트는 지정된 포트폴리오의 UI에서 볼 수 있습니다.

| 인수 |  |
| :--- | :--- |
|  `artifact` |  연결될 (공개 또는 로컬) 아티팩트입니다 |
|  `target_path` |  `str` - 다음 형식을 취합니다: {portfolio}, {project}/{portfolio}, 또는 {entity}/{project}/{portfolio} |
|  `aliases` |  `List[str]` - 선택적으로 이 포트폴리오 내에서 연결된 아티팩트에만 적용될 별칭입니다. "latest" 별칭은 항

### `log`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/wandb_run.py#L1620-L1828)

```python
log(
    data: Dict[str, Any],
    step: Optional[int] = None,
    commit: Optional[bool] = None,
    sync: Optional[bool] = None
) -> None
```

현재 실행의 기록에 데이터 사전을 로그합니다.

`wandb.log`를 사용하여 실행에서 데이터를 로그하세요. 예를 들면, 스칼라, 이미지, 비디오,
히스토그램, 플롯, 테이블 등이 있습니다.

실시간 예제, 코드 조각, 모범 사계, 그리고 더 많은 정보를 보려면 [로그 가이드](https://docs.wandb.ai/guides/track/log)를 참고하세요.

가장 기본적인 사용법은 `wandb.log({"train-loss": 0.5, "accuracy": 0.9})`입니다.
이렇게 하면 손실과 정확도가 실행의 기록에 저장되고 이 메트릭의 요약 값이 업데이트됩니다.

로그된 데이터를 [wandb.ai](https://wandb.ai)에서 워크스페이스로 시각화하거나, [자체 호스팅 인스턴스](https://docs.wandb.ai/guides/hosting)에서 W&B 앱을 로컬로 시각화하거나, Jupyter 노트북에서 데이터를 시각화하고 탐색하기 위해 데이터를 내보낼 수 있습니다.

UI에서 요약 값은 실행 테이블에서 실행 간 단일 값 비교를 위해 표시됩니다.
요약 값은 또한 `wandb.run.summary["key"] = value`로 직접 설정할 수 있습니다.

로그된 값은 스칼라일 필요가 없습니다. 모든 wandb 객체를 로깅하는 것이 지원됩니다.
예를 들어 `wandb.log({"example": wandb.Image("myimage.jpg")})`는 예시 이미지를 로그하며, W&B UI에서 예쁘게 표시됩니다.
지원되는 다양한 유형에 대한 전체 정보는 [참조 문서](https://docs.wandb.com/ref/python/data-types)를 확인하거나 3D 분자 구조부터 분할 마스크, PR 곡선, 히스토그램까지의 예제를 보려면 [로그 가이드](https://docs.wandb.ai/guides/track/log)를 참조하세요.
`wandb.Table`은 구조화된 데이터를 로깅하는 데 사용할 수 있습니다. 자세한 내용은 [테이블 로깅 가이드](https://docs.wandb.ai/guides/data-vis/log-tables)를 참조하세요.

중첩 메트릭 로깅은 권장되며 W&B UI에서 지원됩니다.
`wandb.log({"train": {"acc": 0.9}, "val": {"acc": 0.8}})`와 같은 중첩 사전으로 로그하면 메트릭이 W&B UI에서 `train`과 `val` 섹션으로 구성됩니다.

wandb는 전역 단계를 추적하며 기본적으로 `wandb.log` 호출 시마다 증가하므로 관련 메트릭을 함께 로깅하는 것이 권장됩니다.
관련 메트릭을 함께 로깅하는 것이 불편한 경우
`wandb.log({"train-loss": 0.5}, commit=False)`를 호출한 다음
`wandb.log({"accuracy": 0.9})`를 호출하는 것은
`wandb.log({"train-loss": 0.5, "accuracy": 0.9})`를 호출하는 것과 동일합니다.

`wandb.log`는 초당 몇 번 이상 호출할 의도가 아닙니다.
그 이상 빈번하게 로깅하려면 클라이언트 측에서 데이터를 집계하는 것이 좋으며, 그렇지 않으면 성능이 저하될 수 있습니다.

| 인수 |  |
| :--- | :--- |
|  `data` |  (사전, 선택적) 직렬화 가능한 파이썬 객체의 사전 예: `str`, `int`, `float`, `Tensor`, `사전`, 또는 `wandb.data_types` 중 하나. |
|  `commit` |  (불리언, 선택적) 메트릭 사전을 wandb 서버에 저장하고 단계를 증가시킵니다. 거짓인 경우 `wandb.log`는 현재 메트릭 사전을 데이터 인수로 업데이트하기만 하며 `wandb.log`가 `commit=True`로 호출될 때까지 메트릭이 저장되지 않습니다. |
|  `step` |  (정수, 선택적) 처리의 전역 단계입니다. 이전 단계를 유지하지만 지정된 단계를 커밋하지 않는 것이 기본값입니다. |
|  `sync` |  (불리언, 참) 이 인수는 더 이상 사용되지 않으며 현재 `wandb.log`의 동작을 변경하지 않습니다. |

#### 예제:

더 많고 자세한 예제는 [로그 가이드](https://docs.wandb.com/guides/track/log)를 참조하세요.

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
# 준비가 되었을 때 이 단계를 보고합니다:
run.log({"accuracy": 0.8})
```

### 히스토그램

```python
import numpy as np
import wandb

# 정규 분포에서 무작위로 그레이디언트 샘플링
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
# 축은 (시간, 채널, 높이, 너비)
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
run.log({"pr": wandb.plots.precision_recall(y_test, y_probas, labels)})
```

### 3D 개체

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

### `log_artifact`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/wandb_run.py#L2862-L2897)

```python
log_artifact(
    artifact_or_path: Union[Artifact, StrPath],
    name: Optional[str] = None,
    type: Optional[str] = None,
    aliases: Optional[List[str]] = None
) -> Artifact
```

실행의 출력으로 아티팩트를 선언합니다.

| 인수 |  |
| :--- | :--- |
|  `artifact_or_path` |  (문자열 또는 Artifact) 이 아티팩트의 내용에 대한 경로, 다음 형식 중 하나일 수 있습니다: - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` 또는 `wandb.Artifact`를 호출하여 생성된 Artifact 개체를 전달할 수도 있습니다. |
|  `name` |  (문자열, 선택적) 아티팩트 이름. 엔티티/프로젝트로 접두사를 붙일 수 있습니다. 유효한 이름은 다음 형식 중 하나일 수 있습니다: - name:version - name:alias - digest 지정하지 않으면 현재 실행 ID가 접두사로 붙은 경로의 기본 이름이 기본값이 됩니다. |
|  `type` |  (문자열) 로그하는 아티팩트의 유형, 예를 들면 `데이터세트`, `모델` |
|  `aliases` |  (리스트, 선택적) 이 아티팩트에 적용할 별칭, 기본값은 `["latest"]`입니다. |

| 반환 |  |
| :--- | :--- |
|  `Artifact` 개체. |

### `log_code`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/wandb_run.py#L1097-L1180)

```python
log_code(
    root: Optional[str] = ".",
    name: Optional[str] = None,
    include_fn: Union[Callable[[str, str], bool], Callable[[str], bool]] = _is_py_or_dockerfile,
    exclude_fn: Union[Callable[[str, str], bool], Callable[[str], bool]] = filenames.exclude_wandb_fn
) -> Optional[Artifact]
```

현재 코드 상태를 W&B 아티팩트로 저장합니다.

기본적으로 현재 디렉터리를 탐색하고 `.py`로 끝나는 모든 파일을 로그합니다.

| 인수 |  |
| :--- | :--- |
|  `root` |  코드를 재귀적으로 찾을 상대(현재 작업 디렉터리에 대해) 또는 절대 경로입니다. |
|  `name` |  (문자열, 선택적) 코드 아티팩트의 이름입니다. 기본적으로, 아티팩트 이름은 `source-$PROJECT_ID-$ENTRYPOINT_RELPATH`으로 지정됩니다. 여러 실행이 동일한 아티팩트를 공유하기를 원하는 시나리오가 있을 수 있습니다. 이름을 지정하면 그것을 달성할 수 있습니다. |
|  `include_fn` |  파일 경로를 받아들이고 (선택적으로) 루트 경로를 받아들이며 포함해야 하는 경우 True를, 그렇지 않은 경우 False를 반환하는 콜러블입니다. 기본값은 다음과 같습니다: `lambda path, root: path.endswith(".py")` |
|  `exclude_fn` |  파일 경로를 받아들이고 (선택적으로) 루트 경로를 받아들이며 제외해야 하는 경우 True를, 그렇지 않은 경우 False를 반환하는 콜러블입니다. 기본값은 `&lt;root&gt;/.wandb/` 및 `&lt;root&gt;/wandb/` 디렉터리 내의 모든 파일을 제외하는 함수입니다. |

#### 예제:

기본 사용법

```python
run.log_code()
```

고급 사용법

```python
run.log_code(
    "../",
    include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"),
    exclude_fn=lambda path, root: os.path.relpath(path, root).startswith("cache/"),
)
```

| 반환 |  |
| :--- | :--- |
|  코드가 로그되었다면 `Artifact` 개체 |

### `log_model`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/wandb_run.py#L3139-L3188)

```python
log_model(
    path: StrPath,
    name: Optional[str] = None,
    aliases: Optional[List[str]] = None
) -> None
```

실행에 모델 아티팩트를 로그하고 이 실행의 출력으로 표시합니다.

| 인수 |  |
| :--- | :--- |
|  `path` |  (문자열) 이 모델의 내용에 대한 경로, 다음 형식 중 하나일 수 있습니다: - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` |
|  `name` |  (문자열, 선택적) 파일 내용이 추가될 모델 아티팩트에 할당할 이름입니다. 문자열에는 대시, 밑줄, 점만 포함된 영숫자 문자만 포함할 수 있습니다. 지정하지 않으면 현재 실행 ID가 접두사로 붙은 경로의 기본 이름이 기본값이 됩니다. |
|  `aliases` |  (리스트, 선택적) 생성된 모델 아티팩트에 적용할 별칭, 기본값은 `["latest"]`입니다. |

#### 예제:

```python
run.log_model(
    path="/local/directory",
    name="my_model_artifact",
    aliases=["production"],
)
```

유효하지 않은 사용법

```python
run.log_model(
    path="/local/directory",
    name="my_entity/my_project/my_model_artifact",
    aliases=["production"],
)
```

| 예외 |  |
| :--- | :--- |
|  `ValueError` |  이름에 유효하지 않은 특수 문자가 있는 경우 |

| 반환 |  |
| :--- | :--- |
|  None |

### `mark_preempting`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/wandb_run.py#L3389-L3397)

```python
mark_preempting() -> None
```

이 실행을 선점 중으로 표시합니다.

또한 내부 프로세스에게 이를 서버에 즉시 보고하도록 지시합니다.

### `plot_table`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/wandb_run.py#L2032-L2053)

```python
@staticmethod
plot_table(
    vega_spec_name: str,
    data_table: "wandb.Table",
    fields: Dict[str, Any],
    string_fields: Optional[Dict[str, Any]] = None,
    split_table: Optional[bool

### `to_html`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/wandb_run.py#L1320-L1329)

```python
to_html(
    height: int = 420,
    hidden: bool = (False)
) -> str
```

현재 실행을 보여주는 iframe을 포함한 HTML을 생성합니다.

### `unwatch`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/wandb_run.py#L2660-L2662)

```python
unwatch(
    models=None
) -> None
```

### `upsert_artifact`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/wandb_run.py#L2899-L2951)

```python
upsert_artifact(
    artifact_or_path: Union[Artifact, str],
    name: Optional[str] = None,
    type: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    distributed_id: Optional[str] = None
) -> Artifact
```

실행의 출력으로 비종결 아티팩트를 선언(또는 추가)합니다.

아티팩트를 종결하기 위해 run.finish_artifact()를 호출해야 함에 유의하세요.
이는 분산 작업이 모두 동일한 아티팩트에 기여할 필요가 있을 때 유용합니다.

| 인수 |  |
| :--- | :--- |
|  `artifact_or_path` |  (str 또는 Artifact) 이 아티팩트의 내용에 대한 경로로, 다음 형식 중 하나일 수 있습니다: - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` 또는 `wandb.Artifact`를 호출하여 생성된 Artifact 객체를 전달할 수도 있습니다. |
|  `name` |  (str, 선택 사항) 아티팩트 이름입니다. 엔티티/프로젝트로 접두사가 붙을 수 있습니다. 유효한 이름 형식은 다음과 같습니다: - name:version - name:alias - digest 지정하지 않은 경우 현재 실행 ID로 시작하는 경로의 기본 이름이 기본값으로 설정됩니다. |
|  `type` |  (str) 로그할 아티팩트의 유형으로, 예시에는 `dataset`, `model`이 있습니다. |
|  `aliases` |  (list, 선택 사항) 이 아티팩트에 적용할 별칭으로, 기본값은 `["latest"]`입니다. |
|  `distributed_id` |  (string, 선택 사항) 모든 분산 작업이 공유하는 고유 문자열입니다. None인 경우 실행의 그룹 이름이 기본값으로 설정됩니다. |

| 반환값 |  |
| :--- | :--- |
|  `Artifact` 객체입니다. |

### `use_artifact`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/wandb_run.py#L2753-L2860)

```python
use_artifact(
    artifact_or_name: Union[str, Artifact],
    type: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    use_as: Optional[str] = None
) -> Artifact
```

아티팩트를 실행의 입력으로 선언합니다.

반환된 객체에서 `download` 또는 `file`을 호출하여 내용을 로컬로 가져옵니다.

| 인수 |  |
| :--- | :--- |
|  `artifact_or_name` |  (str 또는 Artifact) 아티팩트 이름입니다. 엔티티/프로젝트/로 접두사가 붙을 수 있습니다. 유효한 이름 형식은 다음과 같습니다: - name:version - name:alias - digest 또는 `wandb.Artifact`를 호출하여 생성된 Artifact 객체를 전달할 수도 있습니다. |
|  `type` |  (str, 선택 사항) 사용할 아티팩트의 유형입니다. |
|  `aliases` |  (list, 선택 사항) 이 아티팩트에 적용할 별칭입니다. |
|  `use_as` |  (string, 선택 사항) 아티팩트가 사용된 목적을 나타내는 선택적 문자열입니다. UI에 표시됩니다. |

| 반환값 |  |
| :--- | :--- |
|  `Artifact` 객체입니다. |

### `use_model`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/wandb_run.py#L3190-L3242)

```python
use_model(
    name: str
) -> FilePathStr
```

모델 아티팩트 'name'에 로그된 파일을 다운로드합니다.

| 인수 |  |
| :--- | :--- |
|  `name` |  (str) 모델 아티팩트 이름입니다. 'name'은 기존에 로그된 모델 아티팩트의 이름과 일치해야 합니다. 엔티티/프로젝트/로 접두사가 붙을 수 있습니다. 유효한 이름 형식은 다음과 같습니다: - model_artifact_name:version - model_artifact_name:alias - model_artifact_name:digest. |

#### 예시:

```python
run.use_model(
    name="my_model_artifact:latest",
)

run.use_model(
    name="my_project/my_model_artifact:v0",
)

run.use_model(
    name="my_entity/my_project/my_model_artifact:<digest>",
)
```

잘못된 사용

```python
run.use_model(
    name="my_entity/my_project/my_model_artifact",
)
```

| 예외 발생 |  |
| :--- | :--- |
|  `AssertionError` |  모델 아티팩트 'name'의 유형이 'model'이라는 문자열을 포함하지 않는 경우 발생합니다. |

| 반환값 |  |
| :--- | :--- |
|  `path` |  (str) 다운로드된 모델 아티팩트 파일의 경로입니다. |

### `watch`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/wandb_run.py#L2647-L2657)

```python
watch(
    models, criterion=None, log="gradients", log_freq=100, idx=None,
    log_graph=(False)
) -> None
```

### `__enter__`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/wandb_run.py#L3373-L3374)

```python
__enter__() -> "Run"
```

### `__exit__`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/wandb_run.py#L3376-L3387)

```python
__exit__(
    exc_type: Type[BaseException],
    exc_val: BaseException,
    exc_tb: TracebackType
) -> bool
```