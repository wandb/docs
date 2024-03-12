
# Run

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/wandb_run.py#L432-L4026' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>


wandb에서 기록된 계산 단위입니다. 일반적으로 이는 ML 실험입니다.

```python
Run(
    settings: Settings,
    config: Optional[Dict[str, Any]] = None,
    sweep_config: Optional[Dict[str, Any]] = None,
    launch_config: Optional[Dict[str, Any]] = None
) -> None
```

`wandb.init()`으로 run을 생성합니다:

```python
import wandb

run = wandb.init()
```

어떤 프로세스에도 항상 최대 하나의 활성 `wandb.Run`이 있으며, `wandb.run`으로 접근할 수 있습니다:

```python
import wandb

assert wandb.run is None

wandb.init()

assert wandb.run is not None
```

`wandb.log`로 기록하는 모든 것은 해당 run으로 전송됩니다.

동일한 스크립트나 노트북에서 더 많은 run을 시작하려면, 실행 중인 run을 종료해야 합니다. Run은 `wandb.finish`를 사용하거나 `with` 블록에서 사용하여 종료할 수 있습니다:

```python
import wandb

wandb.init()
wandb.finish()

assert wandb.run is None

with wandb.init() as run:
    pass  # 여기에 데이터 기록

assert wandb.run is None
```

run을 생성하는 방법에 대한 자세한 내용은 `wandb.init` 문서를 참조하거나 [wandb.init에 대한 가이드](https://docs.wandb.ai/guides/track/launch)를 확인하세요.

분산 트레이닝에서는 rank 0 프로세스에서 단일 run을 생성한 다음 해당 프로세스에서만 정보를 기록하거나, 각 프로세스에서 run을 생성하여 각각 별도로 기록하고 `wandb.init`의 `group` 인수를 사용하여 결과를 그룹화할 수 있습니다. W&B와 함께 분산 트레이닝에 대한 자세한 내용은 [가이드](https://docs.wandb.ai/guides/track/log/distributed-training)를 확인하세요.

현재, `wandb.Api`에는 병렬 `Run` 오브젝트가 있습니다. 결국 이 두 오브젝트는 병합될 것입니다.

| 속성 |  |
| :--- | :--- |
|  `summary` |  각 `wandb.log()` 키에 대해 설정된 단일 값입니다. 기본적으로 summary는 기록된 마지막 값으로 설정됩니다. 최종 값 대신 최고 값, 예를 들어 최대 정확도와 같은 최고의 값으로 summary를 수동으로 설정할 수 있습니다. |
|  `config` |  이 run과 관련된 설정 오브젝트입니다. |
|  `dir` |  run과 관련된 파일이 저장되는 디렉토리입니다. |
|  `entity` |  run과 관련된 W&B 엔티티의 이름입니다. 엔티티는 사용자 이름이거나 팀 또는 조직의 이름일 수 있습니다. |
|  `group` |  run과 관련된 그룹의 이름입니다. 그룹을 설정하면 W&B UI가 run을 합리적인 방식으로 구성하는 데 도움이 됩니다. 분산 트레이닝을 수행하는 경우 트레이닝의 모든 run에 동일한 그룹을 지정해야 합니다. 교차 검증을 수행하는 경우 모든 교차 검증 폴드에 동일한 그룹을 지정해야 합니다. |
|  `id` |  이 run의 식별자입니다. |
|  `mode` |  `0.9.x` 및 이전 버전과의 호환성을 위해 최종적으로 폐지됩니다. |
|  `name` |  run의 표시 이름입니다. 표시 이름은 고유하지 않을 수 있으며 설명적일 수 있습니다. 기본적으로 무작위로 생성됩니다. |
|  `notes` |  run과 관련된 노트가 있는 경우입니다. 노트는 다줄 문자열일 수 있으며 `$$` 안에 마크다운 및 라텍스 방정식을 사용할 수도 있습니다, 예를 들어 `$x + 3$`. |
|  `path` |  run의 경로입니다. Run 경로에는 엔티티, 프로젝트 및 run ID가 포함되며, 형식은 `entity/project/run_id`입니다. |
|  `project` |  run과 관련된 W&B 프로젝트의 이름입니다. |
|  `resumed` |  run이 재개되었는지 여부입니다. |
|  `settings` |  run의 설정 오브젝트의 동결된 복사본입니다. |
|  `start_time` |  run이 시작된 시간의 유닉스 타임스탬프(초)입니다. |
|  `starting_step` |  run의 첫 번째 단계입니다. |
|  `step` |  현재 단계의 값입니다. 이 카운터는 `wandb.log`에 의해 증가됩니다. |
|  `sweep_id` |  run과 관련된 스윕의 ID입니다(있는 경우). |
|  `tags` |  run과 관련된 태그가 있는 경우입니다. |
|  `url` |  run과 관련된 W&B URL입니다. |

## 메소드

### `alert`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/wandb_run.py#L3323-L3356)

```python
alert(
    title: str,
    text: str,
    level: Optional[Union[str, 'AlertLevel']] = None,
    wait_duration: Union[int, float, timedelta, None] = None
) -> None
```

주어진 제목과 텍스트로 알림을 발송합니다.

| 인수 |  |
| :--- | :--- |
|  `title` |  (str) 알림의 제목이며, 64자를 초과할 수 없습니다. |
|  `text` |  (str) 알림의 본문 텍스트입니다. |
|  `level` |  (str 또는 wandb.AlertLevel, 선택 사항) 사용할 알림 수준이며, `INFO`, `WARN`, 또는 `ERROR` 중 하나입니다. |
|  `wait_duration` |  (int, float, 또는 timedelta, 선택 사항) 이 제목으로 다른 알림을 보내기 전에 기다릴 시간(초)입니다. |

### `define_metric`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/wandb_run.py#L2535-L2569)

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

나중에 `wandb.log()`로 기록될 메트릭 속성을 정의합니다.

| 인수 |  |
| :--- | :--- |
|  `name` |  메트릭의 이름입니다. |
|  `step_metric` |  메트릭과 관련된 독립 변수입니다. |
|  `step_sync` |  필요한 경우 자동으로 `step_metric`을 기록에 추가합니다. step_metric이 지정된 경우 기본값은 True입니다. |
|  `hidden` |  이 메트릭을 자동 플롯에서 숨깁니다. |
|  `summary` |  요약에 추가될 집계 메트릭을 지정합니다. 지원되는 집계: "min,max,mean,best,last,none" 기본 집계는 `copy`입니다. 집계 `best`는 기본적으로 `goal`이 `minimize`일 때 적용됩니다. |
|  `goal` |  메트릭을 최적화하는 방향을 지정합니다. 지원되는 방향: "minimize,maximize" |

| 반환 |  |
| :--- | :--- |
|  추가로 지정할 수 있는 메트릭 오브젝트가 반환됩니다. |

### `detach`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/wandb_run.py#L2691-L2692)

```python
detach() -> None
```

### `display`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/wandb_run.py#L1310-L1318)

```python
display(
    height: int = 420,
    hidden: bool = (False)
) -> bool
```

이 run을 주피터에서 표시합니다.

### `finish`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/wandb_run.py#L1947-L1961)

```python
finish(
    exit_code: Optional[int] = None,
    quiet: Optional[bool] = None
) -> None
```

run을 완료로 표시하고 모든 데이터의 업로드를 완료합니다.

이 메소드는 동일한 프로세스에서 여러 run을 생성할 때 사용됩니다. 스크립트가 종료될 때나 run 컨텍스트 관리자를 사용할 때 이 메소드를 자동으로 호출합니다.

| 인수 |  |
| :--- | :--- |
|  `exit_code` |  0이 아닌 것으로 설정하여 run을 실패로 표시합니다 |
|  `quiet` |  로그 출력을 최소화하려면 true로 설정하세요 |

### `finish_artifact`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/wandb_run.py#L2941-L2993)

```python
finish_artifact(
    artifact_or_path: Union[Artifact, str],
    name: Optional[str] = None,
    type: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    distributed_id: Optional[str] = None
) -> Artifact
```

run의 출력으로 최종 결정되지 않은 아티팩트를 완료합니다.

동일한 분산 ID로 "업서트"하는 후속 작업은 새 버전을 생성합니다.

| 인수 |  |
| :--- | :--- |
|  `artifact_or_path` |  (str 또는 Artifact) 이 아티팩트의 내용에 대한 경로로, 다음 형식 중 하나일 수 있습니다: - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` `wandb.Artifact`를 호출하여 생성된 Artifact 오브젝트도 전달할 수 있습니다. |
|  `name` |  (str, 선택 사항) 아티팩트의 이름입니다. 엔티티/프로젝트로 접두사를 붙일 수 있습니다. 유효한 이름은 다음 형식 중 하나일 수 있습니다: - name:version - name:alias - digest 지정하지 않으면 기본적으로 현재 run ID로 시작하는 경로의 기본 이름이 사용됩니다. |
|  `type` |  (str) 기록할 아티팩트의 유형입니다. 예를 들어 `dataset`, `model`과 같은 예가 있습니다. |
|  `aliases` |  (list, 선택 사항) 이 아티팩트에 적용할 별칭으로, 기본값은 `["latest"]`입니다. |
|  `distributed_id` |  (string, 선택 사항) 모든 분산 작업이 공유하는 고유 문자열입니다. None인 경우 기본값은 run의 그룹 이름입니다. |

| 반환 |  |
| :--- | :--- |
|  `Artifact` 오브젝트입니다. |

### `get_project_url`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/wandb_run.py#L1192-L1200)

```python
get_project_url() -> Optional[str]
```

run과 관련된 W&B 프로젝트의 URL을 반환합니다(있는 경우).

오프라인 run은 프로젝트 URL이 없습니다.

### `get_sweep_url`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/wandb_run.py#L1202-L1207)

```python
get_sweep_url() -> Optional[str]
```

run과 관련된 스윕의 URL을 반환합니다(있는 경우).

### `get_url`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/wandb_run.py#L1182-L1190)

```python
get_url() -> Optional[str]
```

run과 관련된 W&B URL을 반환합니다(있는 경우).

오프라인 run은 URL이 없습니다.

### `join`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/wandb_run.py#L1995-L2005)

```python
join(
    exit_code: Optional[int] = None
) -> None
```

`finish()`의 사용 중단된 별칭입니다 - 대신 finish를 사용하세요.

### `link_artifact`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/wandb_run.py#L2694-L2740)

```python
link_artifact(
    artifact: Artifact,
    target_path: str,
    aliases: Optional[List[str]] = None
) -> None
```

주어진 아티팩트를 포트폴리오(승진된 아티팩트 모음)에 연결합니다.

연결된 아티팩트는 지정된 포트폴리오의 UI에서 볼 수 있습니다.

| 인수 |  |
| :--- | :--- |
|  `artifact` |  연결될 (공개 또는 로컬) 아티팩트입니다. |
|  `target_path` |  `str` - 다음 형식을 취합니다: {portfolio}, {project}/{portfolio}, 또는 {entity}/{project}/{portfolio} |
|  `aliases` |  `List[str]` - 이 포트폴리오 내에서 연결된 아티팩트에만 적용될 선택적 별칭입니다. "latest" 별칭은 항상 연결된 아티팩트의 최신 버전에 적용됩니다. |

| 반환 |  |
| :--- | :--- |
|  None |

### `link_model`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/wandb_run.py#L3229-L3321)

```python
link_model(
    path: StrPath,
    registered_model_name: str,
    name: Optional[str] = None,
    aliases: Optional[List[str]]

### numpy로부터의 비디오




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
ax.plot(x, y)  # y = x^2 그래프 그리기
run.log({"chart": fig})
```

### PR 커브

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

| 발생할 수 있는 오류 |  |
| :--- | :--- |
|  `wandb.Error` |  `wandb.init` 호출 전에 사용된 경우 |
|  `ValueError` |  잘못된 데이터가 전달된 경우 |

### `log_artifact`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/wandb_run.py#L2850-L2885)

```python
log_artifact(
    artifact_or_path: Union[Artifact, StrPath],
    name: Optional[str] = None,
    type: Optional[str] = None,
    aliases: Optional[List[str]] = None
) -> Artifact
```

아티팩트를 run의 출력물로 선언합니다.

| 인수 |  |
| :--- | :--- |
|  `artifact_or_path` |  (str 또는 Artifact) 이 아티팩트의 내용에 대한 경로, 다음 형식 중 하나일 수 있습니다: - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` `wandb.Artifact`를 호출하여 생성된 Artifact 오브젝트도 전달할 수 있습니다. |
|  `name` |  (str, 선택사항) 아티팩트 이름. entity/project로 접두사가 붙을 수 있습니다. 유효한 이름 형식은 다음과 같습니다: - name:version - name:alias - digest 지정하지 않은 경우 경로의 기본 이름에 현재 run id를 접두사로 붙입니다. |
|  `type` |  (str) 로그하는 아티팩트의 유형, 예를 들어 `dataset`, `model` |
|  `aliases` |  (list, 선택사항) 이 아티팩트에 적용할 에일리어스, 기본값은 `["latest"]` |

| 반환값 |  |
| :--- | :--- |
|  `Artifact` 오브젝트. |

### `log_code`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/wandb_run.py#L1097-L1180)

```python
log_code(
    root: Optional[str] = ".",
    name: Optional[str] = None,
    include_fn: Union[Callable[[str, str], bool], Callable[[str], bool]] = _is_py_or_dockerfile,
    exclude_fn: Union[Callable[[str, str], bool], Callable[[str], bool]] = filenames.exclude_wandb_fn
) -> Optional[Artifact]
```

현재 코드의 상태를 W&B 아티팩트로 저장합니다.

기본적으로 현재 디렉토리를 순회하여 `.py`로 끝나는 모든 파일을 로그합니다.

| 인수 |  |
| :--- | :--- |
|  `root` |  코드를 재귀적으로 찾을 상대적(대 `os.getcwd()`) 또는 절대 경로. |
|  `name` |  (str, 선택사항) 코드 아티팩트의 이름. 기본적으로, 아티팩트 이름은 `source-$PROJECT_ID-$ENTRYPOINT_RELPATH`로 지정됩니다. 여러 run이 동일한 아티팩트를 공유하고자 하는 시나리오에서 이름을 지정하면 그것을 달성할 수 있습니다. |
|  `include_fn` |  파일 경로와 (선택적으로) 루트 경로를 받아서 포함해야 할 경우 True를, 그렇지 않을 경우 False를 반환하는 callable입니다. 기본값은: `lambda path, root: path.endswith(".py")` |
|  `exclude_fn` |  파일 경로와 (선택적으로) 루트 경로를 받아서 제외해야 할 경우 True를, 그렇지 않을 경우 False를 반환하는 callable입니다. 기본값은 `&lt;root&gt;/.wandb/` 및 `&lt;root&gt;/wandb/` 디렉토리 내의 모든 파일을 제외하는 함수입니다. |

#### 예시:

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

| 반환값 |  |
| :--- | :--- |
|  코드가 로그되었다면 `Artifact` 오브젝트 |

### `log_model`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/wandb_run.py#L3125-L3174)

```python
log_model(
    path: StrPath,
    name: Optional[str] = None,
    aliases: Optional[List[str]] = None
) -> None
```

'path' 내부의 내용을 포함하는 모델 아티팩트를 run에 로그하고 이 run의 출력으로 표시합니다.

| 인수 |  |
| :--- | :--- |
|  `path` |  (str) 이 모델의 내용에 대한 경로, 다음 형식 중 하나일 수 있습니다: - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` |
|  `name` |  (str, 선택사항) 파일 내용이 추가될 모델 아티팩트에 할당할 이름. 문자열에는 다음 알파뉴메릭 문자만 포함될 수 있습니다: 대시, 밑줄, 점. 지정하지 않은 경우 경로의 기본 이름에 현재 run id를 접두사로 붙입니다. |
|  `aliases` |  (list, 선택사항) 생성된 모델 아티팩트에 적용할 에일리어스, 기본값은 `["latest"]` |

#### 예시:

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

| 발생할 수 있는 오류 |  |
| :--- | :--- |
|  `ValueError` |  이름에 유효하지 않은 특수 문자가 있는 경우 |

| 반환값 |  |
| :--- | :--- |
|  None |

### `mark_preempting`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/wandb_run.py#L3374-L3382)

```python
mark_preempting() -> None
```

이 run을 선점 중으로 표시합니다.

또한 내부 프로세스에게 즉시 서버에 이를 보고하도록 지시합니다.

### `plot_table`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/wandb_run.py#L2032-L2053)

```python
@staticmethod
plot_table(
    vega_spec_name: str,
    data_table: "wandb.Table",
    fields: Dict[str, Any],
    string_fields: Optional[Dict[str, Any]] = None,
    split_table: Optional[bool] = (False)
) -> CustomChart
```

테이블에 대한 사용자 지정 플롯을 생성합니다.

| 인수 |  |
| :--- | :--- |
|  `vega_spec_name` |  플롯에 대한 스펙의 이름 |
|  `data_table` |  시각화에 사용될 데이터를 포함하는 wandb.Table 오브젝트 |
|  `fields` |  사용자 지정 시각화에 필요한 필드로부터 테이블 키를 매핑하는 딕셔너리 |
|  `string_fields` |  사용자 지정 시각화에 필요한 모든 문자열 상수에 대한 값을 제공하는 딕셔너리 |

### `project_name`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/wandb_run.py#L1043-L1044)

```python
project_name() -> str
```

### `restore`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/wandb_run.py#L1932-L1945)

```python
restore(
    name: str,
    run_path: Optional[str] = None,
    replace: bool = (False),
    root: Optional[str] = None
) -> Union[None, TextIO]
```

클라우드 저장소에서 지정된 파일을 다운로드합니다.

기본적으로 이미 로컬에 파일이 존재하지 않는 경우에만 파일을 다운로드합니다.

| 인수 |  |
| :--- | :--- |
|  `name` |  파일의 이름 |
|  `run_path` |  파일을 가져올 run의 선택적 경로, 즉 `username/project_name/run_id` `wandb.init`이 호출되지 않은 경우 이는 필수입니다. |
|  `replace` |  이미 로컬에 존재하는 파일이라도 다운로드할지 여부 |
|  `root` |  파일을 다운로드할 디렉토리. 기본값은 현재 디렉토리 또는 `wandb.init`이 호출된 경우 run 디렉토리입니다. |

| 반환값 |  |
| :--- | :--- |
|  파일을 찾을 수 없는 경우 None, 그렇지 않으면 읽기 위해 열린 파일 오브젝트 |

| 발생할 수 있는 오류 |  |
| :--- | :--- |
|  `wandb.CommError` |  wandb 백엔드에 연결할 수 없는 경우 |
|  `ValueError` |  파일을 찾을 수 없거나 run_path를 찾을 수 없는 경우 |

### `save`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/wandb_run.py#L1830-L1860)

```python
save(
    glob_str: Optional[str] = None,
    base_path: Optional[str] = None,
    policy: "PolicyName" = "live"
) -> Union[bool, List[str]]
```

지정된 `glob_str`과 일치하는 모든 파일이 지정된 정책에 따라 wandb에 동기화되도록 합니다.

| 인수 |  |
| :--- | :--- |
|  `glob_str` |  (문자열) 유닉스 글롭 또는 정규 경로에 대한 상대적 또는 절대 경로. 지정되지 않은 경우 이 메소드는 아무 작업도 수행하지 않습니다. |
|  `base_path` |  글롭을 실행할 기본 경로 |
|  `policy` |  (문자열) `live`, `now`, `end` 중 하나 - live: 파일이 변경될 때마다 이전 버전을 덮어쓰면서 업로드 - now: 지금 한 번 업로드 - end: run이 끝날 때만 파일 업로드 |

### `status`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/wandb_run.py#L2007-L2030)

```python
status() -> RunStatus
```

내부 백엔드에서 현재 run의 동기화 상태에 대한 동기화 정보를 가져옵니다.

### `to_html`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/wandb_run.py#L1320-L1329)

```python
to_html(
    height: int = 420,
    hidden: bool = (False)
) -> str
```

현재 run을 표시하는 iframe을 포함하는 HTML을 생성합니다.

### `unwatch`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/wandb_run.py#L2652-L2654)

```python
unwatch(
    models=None
) -> None
```

### `upsert_artifact`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/wandb_run.py#L2887-L2939)

```python
upsert_artifact(
    artifact_or_path: Union[Artifact, str],
    name: Optional[str] = None,
    type: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    distributed_id: Optional[str] = None
) -> Artifact
```

run의 출력으로 아티팩트를 선언하거나 (아직 완료되지 않은) 아티팩트에 추가합니다.

분산 작업이 동일한 아티팩트에 모두 기여해야 할 때 유용합니다.

| 인수 |  |
| :--- | :--- |
|  `artifact_or_path` |  (str 또는 Artifact) 이 아티팩트의 내용에 대한 경로, 다음 형식 중 하나일 수 있습니다: - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` `wandb.Artifact`를 호출하여 생성된 Artifact 오브젝트도 전달할 수 있습니다. |
|  `name` |  (str, 선택사항) 아티팩트 이름. entity/project로 접두사가 붙을 수 있습니다. 유효한 이름 형식은 다음과 같습니다: - name:version - name:alias - digest 지정하지 않은 경우 경로의 기본 이름에 현재 run id를 접두사로 붙입니다. |
|  `type` |  (str) 로그하는 아티팩트의 유형, 예를 들어 `dataset`, `model` |
|  `aliases` |  (list, 선택사항) 이 아티팩트에 적용할 에일리어스, 기본값은 `["latest"]` |
|  `distributed_id` |  (문자열, 선택사항) 모든 분산 작업이 공유하는 고유 문자열. None인 경우 run의 그룹 이름을 기본값으로 사용합니다. |

| 반환값 |  |
| :--- | :--- |
|  `Artifact` 오브젝트. |

### `use_artifact`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/wandb_run.py#L2742-L2848)

```python
use_artifact(
    artifact_or_name: Union[str, Artifact],
    type: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    use_as: Optional[str] = None
) -> Artifact
```

아티팩트를 run의 입력으로 선언합니다.

반환된 객체에서 `download` 또는 `file`을 호출하여 내용을 로컬에 가져옵니다.

| 인수 |  |
| :--- | :--- |
|  `artifact