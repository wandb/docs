# Run

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L465-L4299' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

wandb에 의해 기록된 계산 단위입니다. 일반적으로, 이는 ML 실험입니다.

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

하나의 프로세스에 있어서 최대 하나의 활성 `wandb.Run`만 존재하며,
이는 `wandb.run`을 통해 엑세스할 수 있습니다:

```python
import wandb

assert wandb.run is None

wandb.init()

assert wandb.run is not None
```

`wandb.log`로 기록하는 모든 것은 해당 run으로 전송됩니다.

동일한 스크립트 또는 노트북에서 run을 더 시작하고 싶다면, 진행 중인 run을 종료해야 합니다. run은 `wandb.finish` 또는 `with` 블록에서 사용하여 종료할 수 있습니다:

```python
import wandb

wandb.init()
wandb.finish()

assert wandb.run is None

with wandb.init() as run:
    pass  # 여기에서 데이터를 기록합니다

assert wandb.run is None
```

run을 생성하는 방법에 대한 자세한 내용은 `wandb.init` 문서를 참조하거나
[wandb.init에 대한 우리의 가이드](https://docs.wandb.ai/guides/track/launch)를 확인하세요.

분산 트레이닝에서는 랭크 0 프로세스에서 단일 run을 생성한 후 해당 프로세스에서만 정보를 기록하거나, 각 프로세스에서 run을 생성하고 각각 별도로 기록한 다음 `wandb.init`에 `group` 인수를 사용하여 결과를 함께 그룹화할 수 있습니다. W&B를 사용한 분산 트레이닝에 대한 자세한 내용은 [우리의 가이드](https://docs.wandb.ai/guides/track/log/distributed-training)를 확인하세요.

현재 `wandb.Api`에는 병렬 `Run` 오브젝트가 있습니다. 궁극적으로 이 두 오브젝트는 병합될 것입니다.

| 속성 |  |
| :--- | :--- |
|  `summary` |  (Summary) 각 `wandb.log()` 키에 대해 설정된 단일 값입니다. 기본적으로 summary는 마지막으로 기록된 값으로 설정됩니다. 최종 값 대신 최대 정확도와 같은 최상의 값으로 summary를 수동으로 설정할 수 있습니다. |
|  `config` |  이 run과 관련된 Config 오브젝트입니다. |
|  `dir` |  run과 관련된 파일이 저장된 디렉토리입니다. |
|  `entity` |  run과 관련된 W&B 엔티티의 이름입니다. 엔티티는 사용자 이름 또는 팀 또는 조직의 이름일 수 있습니다. |
|  `group` |  run과 관련된 그룹의 이름입니다. 그룹을 설정하면 W&B UI가 run을 합리적인 방식으로 구성하는 데 도움이 됩니다. 분산 트레이닝을 수행하는 경우, 트레이닝의 모든 run에 동일한 그룹을 부여해야 합니다. 교차 검증을 수행할 경우, 모든 교차 검증 폴드에 동일한 그룹을 부여해야 합니다. |
|  `id` |  이 run의 식별자입니다. |
|  `mode` |  `0.9.x` 및 이전 버전과의 호환성을 위해 유지되며, 궁극적으로 폐기될 예정입니다. |
|  `name` |  run의 표시 이름입니다. 표시 이름은 고유성이 보장되지 않으며 설명적일 수 있습니다. 기본적으로 무작위로 생성됩니다. |
|  `notes` |  run과 연관된 메모가 있는 경우, 이를 표시합니다. 메모는 여러 줄의 문자열일 수 있으며 메모 내부는 마크다운과 라텍스 방정식을 사용할 수 있습니다. |
|  `path` |  run의 경로입니다. run 경로에는 entity, project, run ID가 포함되어 있으며, 이 포맷으로 표현됩니다 `entity/project/run_id`. |
|  `project` |  run과 관련된 W&B 프로젝트의 이름입니다. |
|  `resumed` |  run이 재개되었으면 True, 그렇지 않으면 False입니다. |
|  `settings` |  run의 Settings 오브젝트의 동결된 복사본입니다. |
|  `start_time` |  run이 시작된 시점의 유닉스 타임스탬프 (초)입니다. |
|  `starting_step` |  run의 첫 번째 단계입니다. |
|  `step` |  현재 단계의 값입니다. 이 카운터는 `wandb.log`에 의해 증가됩니다. |
|  `sweep_id` |  run과 관련된 sweep의 ID입니다. |
|  `tags` |  run과 관련된 태그가 있는 경우, 이를 표시합니다. |
|  `url` |  run과 관련된 W&B의 url입니다. |

## Methods

### `alert`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L3594-L3627)

```python
alert(
    title: str,
    text: str,
    level: Optional[Union[str, 'AlertLevel']] = None,
    wait_duration: Union[int, float, timedelta, None] = None
) -> None
```

제목과 텍스트를 사용하여 알림을 시작합니다.

| 인수 |  |
| :--- | :--- |
|  `title` |  (str) 알림의 제목이며, 64자 미만이어야 합니다. |
|  `text` |  (str) 알림의 본문 텍스트입니다. |
|  `level` |  (str 또는 AlertLevel, optional) 사용할 경고 수준으로, `INFO`, `WARN`, 또는 `ERROR` 중 하나입니다. |
|  `wait_duration` |  (int, float, 또는 timedelta, optional) 동일한 제목으로 또 다른 알림을 보내기 전에 대기할 시간(초)입니다. |

### `define_metric`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L2746-L2807)

```python
define_metric(
    name: str,
    step_metric: Union[str, wandb_metric.Metric, None] = None,
    step_sync: Optional[bool] = None,
    hidden: Optional[bool] = None,
    summary: Optional[str] = None,
    goal: Optional[str] = None,
    overwrite: Optional[bool] = None
) -> wandb_metric.Metric
```

`wandb.log()`에 기록된 메트릭을 맞춤 설정합니다.

| 인수 |  |
| :--- | :--- |
|  `name` |  맞춤 설정할 메트릭의 이름입니다. |
|  `step_metric` |  이 메트릭을 위한 자동 생성 차트에서 X-축으로 사용할 다른 메트릭의 이름입니다. |
|  `step_sync` |  명시적으로 제공되지 않은 경우 마지막 step_metric 값을 `run.log()`에 자동으로 삽입합니다. step_metric이 지정된 경우 기본값은 True입니다. |
|  `hidden` |  자동 플롯에서 이 메트릭을 숨깁니다. |
|  `summary` |  summary에 추가된 누적 메트릭을 지정합니다. 지원되는 집계는 "min", "max", "mean", "last", "best", "copy" 및 "none"입니다. "best"는 goal 파라미터와 함께 사용됩니다. "none"은 summary 생성을 방지합니다. "copy"는 더 이상 사용되지 않으며 사용하지 않아야 합니다. |
|  `goal` |  "best" summary 유형을 해석하는 방법을 지정합니다. 지원되는 옵션은 "minimize"와 "maximize"입니다. |
|  `overwrite` |  false 인 경우, 이 호출은 동일한 메트릭에 대한 이전 `define_metric` 호출과 결합되고, 지정되지 않은 파라미터의 값이 사용됩니다. true 인 경우, 지정되지 않은 파라미터는 이전 호출에서 지정한 값을 덮어씁니다. |

| 반환값 |  |
| :--- | :--- |
|  이 호출을 나타내지만, 그 외에는 버려질 수도 있는 오브젝트입니다. |

### `detach`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L2940-L2941)

```python
detach() -> None
```

### `display`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L1360-L1368)

```python
display(
    height: int = 420,
    hidden: bool = (False)
) -> bool
```

이 run을 jupyter에서 표시합니다.

### `finish`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L2142-L2156)

```python
finish(
    exit_code: Optional[int] = None,
    quiet: Optional[bool] = None
) -> None
```

run을 종료하고 모든 데이터를 업로드 완료합니다.

하나의 프로세스에서 여러 run을 생성할 때 사용됩니다. 스크립트가 종료되거나 run 컨텍스트 관리자를 사용할 경우, 이 메소드를 자동으로 호출합니다.

| 인수 |  |
| :--- | :--- |
|  `exit_code` |  0이 아닌 다른 값을 설정하여 run을 실패로 표시합니다. |
|  `quiet` |  로그 출력을 최소화하도록 true로 설정합니다. |

### `finish_artifact`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L3203-L3255)

```python
finish_artifact(
    artifact_or_path: Union[Artifact, str],
    name: Optional[str] = None,
    type: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    distributed_id: Optional[str] = None
) -> Artifact
```

비완료된 아티팩트를 run의 출력으로 마무리합니다.

동일한 분산 ID로 이후의 "upserts"는 새로운 버전을 생성합니다.

| 인수 |  |
| :--- | :--- |
|  `artifact_or_path` |  (str 또는 Artifact) 이 아티팩트의 내용을 가리키는 경로로, 다음 형태로 제공될 수 있습니다: - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` 또한 `wandb.Artifact`를 호출하여 생성한 Artifact 오브젝트를 전달할 수 있습니다. |
|  `name` |  (str, optional) 아티팩트의 이름을 지정합니다. entity/project 접두사를 포함할 수 있습니다. 유효한 이름은 다음 형태로 제공될 수 있습니다: - name:version - name:alias - digest 지정하지 않은 경우 기본으로 현재 run ID가 붙은 경로의 basename으로 설정됩니다. |
|  `type` |  (str) 기록할 아티팩트의 유형입니다. 예시는 `dataset`, `model` 등입니다. |
|  `aliases` |  (list, optional) 이 아티팩트에 적용할 에일리어스입니다, 기본값은 `["latest"]`입니다. |
|  `distributed_id` |  (string, optional) 모든 분산 작업이 공유하는 고유 문자열입니다. None인 경우, run의 그룹 이름을 기본값으로 합니다. |

| 반환값 |  |
| :--- | :--- |
|  `Artifact` 오브젝트입니다. |

### `get_project_url`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L1242-L1250)

```python
get_project_url() -> Optional[str]
```

run과 관련된 W&B 프로젝트의 url을 반환합니다. 만약 존재한다면.

오프라인 run은 project url을 가지지 않습니다.

### `get_sweep_url`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L1252-L1257)

```python
get_sweep_url() -> Optional[str]
```

run과 관련된 sweep의 url을 반환합니다. 만약 존재한다면.

### `get_url`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L1232-L1240)

```python
get_url() -> Optional[str]
```

W&B run의 url을 반환합니다. 만약 존재한다면.

오프라인 run은 url을 가지지 않습니다.

### `join`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L2210-L2221)

```python
join(
    exit_code: Optional[int] = None
) -> None
```

`finish()`에 대한 더 이상 사용되지 않는 에일리어스 - 대신 finish를 사용하세요.

### `link_artifact`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L2943-L2996)

```python
link_artifact(
    artifact: Artifact,
    target_path: str,
    aliases: Optional[List[str]] = None
) -> None
```

지정된 아티팩트를 포트폴리오(프로모션된 아티팩트 모음)로 링크합니다.

링크된 아티팩트는 지정된 포트폴리오의 UI에서 보이게 됩니다.

| 인수 |  |
| :--- | :--- |
|  `artifact` |  링크될 (공개 또는 로컬) 아티팩트입니다. |
|  `target_path` |  `str` - 다음 형식을 취합니다: `{portfolio}`, `{project}/{portfolio}`, or `{entity}/{project}/{portfolio}` |
|  `aliases` |  `List[str]` - 이 포트폴리오 내에서 이 링크된 아티팩트에만 적용될 선택적 에일리어스입니다. "latest"라는 에일리어스는 항상 링크된 아티팩트의 최신 버전에 적용됩니다. |

| 반환값 |  |
| :--- | :--- |
|  None |

### `link_model`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L3500-L3592)

```python
link_model(
    path: StrPath,
    registered_model_name: str,
    name: Optional[str] = None,
    aliases: Optional[List[str]] = None
) -> None
```

모델 아티팩트 버전을 기록하고 모델 레지스트리에 있는 등록된 모델에 연결합니다.

연결된 모델 버전은 지정된 등록된 모델의 UI에서 볼 수 있습니다.

#### Steps:

- 'name' 모델 아티팩트가 기록되었는지 확인합니다. 그렇다면, 'path'에 위치한 파일과 일치하는 아티팩트 버전을 사용하거나 새로운 버전을 기록합니다. 그렇지 않으면 'path' 아래 파일을 'model' 유형의 새로운 모델 아티팩트로 로그합니다.
- 등록된 모델이 'model-registry' 프로젝트에 'registered_model_name'이라는 이름으로 존재하는지 확인합니다. 그렇지 않으면 'registered_model_name'이라는 이름의 새로운 등록 모델을 만듭니다.
- 모델 아티팩트 'name'의 버전을 등록 모델 'registered_model_name'에 연결합니다.
- 'aliases' 목록의 에일리어스를 새로 연결된 모델 아티팩트 버전에 연결합니다.

| 인수 |  |
| :--- | :--- |
|  `path` |  (str) 이 모델의 내용을 가리키는 경로는 다음 형태 중 하나일 수 있습니다: - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` |
|  `registered_model_name` |  (str) - 모델 레지스트리에 연결할 등록된 모델의 이름입니다. 등록된 모델은 일반적으로 팀의 특정 ML 작업을 나타내며, 모델 레지스트리에 연결된 모델 버전 모음입니다. 이 등록된 모델이 속한 엔티티는 run에서 파생됩니다. |
|  `name` |  (str, optional) - 'path'의 파일 내용을 기록할 모델 아티팩트의 이름입니다. 지정하지 않으면 기본적으로 현재 run ID가 붙은 'path'의 basename으로 설정됩니다. |
|  `aliases` |  (List[str], optional) - 등록된 모델 내에서 이 연결된 아티팩트에만 적용될 에일리어스입니다. "latest" 에일리어스는 항상 연결된 아티팩트의 최신 버전에 적용됩니다. |

#### 예시:

```python
run.link_model(
    path="/local/directory",
    registered_model_name="my_reg_model",
    name="my_model_artifact",
    aliases=["production"],
)
```

잘못된 사용

```python
run.link_model(
    path="/local/directory",
    registered_model_name="my_entity/my_project/my_reg_model",
    name="my_model_artifact",
    aliases=["production"],
)

```python
run.link_model(
    path="/local/directory",
    registered_model_name="my_reg_model",
    name="my_entity/my_project/my_model_artifact",
    aliases=["production"],
)
```

| Raises |  |
| :--- | :--- |
|  `AssertionError` |  registered_model_name이 경로이거나 모델 아티팩트 'name' 유형이 'model' 문자열을 포함하지 않는 경우 |
|  `ValueError` |  name에 잘못된 특수 문자가 포함된 경우 |

| 반환값 |  |
| :--- | :--- |
|  None |

### `log`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L1678-L1933)

```python
log(
    data: Dict[str, Any],
    step: Optional[int] = None,
    commit: Optional[bool] = None,
    sync: Optional[bool] = None
) -> None
```

run 데이터를 업로드합니다.

`log`를 사용하여 run에서 스칼라, 이미지, 비디오, 히스토그램, 플롯 및 테이블과 같은 데이터를 기록합니다.

[로깅 가이드](https://docs.wandb.ai/guides/track/log)를 참조하여 라이브 예제, 코드 조각, 모범 사례 및 기타 정보를 확인하세요.

가장 기본적인 사용법은 `run.log({"train-loss": 0.5, "accuracy": 0.9})`입니다.
이렇게 하면 손실과 정확성이 run의 기록에 저장되고 이러한 메트릭의 요약 값이 업데이트됩니다.

기록된 데이터를 [wandb.ai](https://wandb.ai)의 워크스페이스에서 시각화하거나 
W&B 앱의 [자체 호스팅 인스턴스](https://docs.wandb.ai/guides/hosting)에서 로컬로 시각화하고 탐색합니다. 
또는 데이터를 로컬에서 Jupyter 노트북과 같은 곳으로 내보내어 시각화하고 탐색하세요. 
[API 문서](https://docs.wandb.ai/guides/track/public-api-guide)를 확인하세요.

기록된 값은 스칼라일 필요 없습니다. 어떤 wandb 오브젝트라도 기록하는 것이 지원됩니다.
예를 들어 `run.log({"example": wandb.Image("myimage.jpg")})`는 
W&B UI에서 잘 표시될 이미지 예제를 기록합니다. 
지원되는 유형에 대한 모든 참조 문서를 보거나 
예제를 보고 [로깅 가이드](https://docs.wandb.ai/guides/track/log)를 확인하세요.
3D 분자 구조 및 분할 마스크에서 PR 곡선 및 히스토그램까지입니다. 
구조화된 데이터를 기록하려면 `wandb.Table`을 사용할 수 있습니다. 
로그 테이블에 대한 자세한 내용은 [로그 테이블 가이드](https://docs.wandb.ai/guides/data-vis/log-tables)를 참조하세요.

W&B UI는 이름에 슬래시(`/`)가 포함된 메트릭을 섹션으로 구성하며, 
최종 슬래시 이전의 텍스트로 이름을 지정합니다. 예를 들어,

```
run.log({
    "train/accuracy": 0.9,
    "train/loss": 30,
    "validate/accuracy": 0.8,
    "validate/loss": 20,
})
```

이렇게 하면 "train" 및 "validate"라는 두 개의 섹션이 생성됩니다.

중첩 수준은 하나만 지원됩니다. `run.log({"a/b/c": 1})`는
"a/b"라는 섹션을 생성합니다.

`run.log`는 초당 여러 번 호출될 의도로 만들어지지 않았습니다.
최적의 성능을 위해서는 N개의 반복마다 한 번씩만 로깅하거나, 
여러 반복을 통해 데이터를 수집하여 단일 단계에서 로깅하세요.

### W&B 스텝

기본 사용법에서는 `log`를 호출할 때마다 새로운 "단계"가 생성됩니다.
단계는 항상 증가해야 하며 이전 단계에 로깅하는 것은 불가능합니다.

차트의 X축에서는 어떤 메트릭도 사용할 수 있다는 점에 유의하세요.
많은 경우, W&B 단계를 타임스탬프처럼 취급하는 것이 
트레이닝 단계처럼 취급하는 것보다 낫습니다.

```
# 예제: X축으로 사용할 "epoch" 메트릭 기록
run.log({"epoch": 40, "train-loss": 0.5})
```

다중 `log` 호출을 사용하여 
`step` 및 `commit` 파라미터와 함께 같은 단계에 로그할 수 있습니다.
다음은 모두 동일합니다:

```
# 일반적인 사용법:
run.log({"train-loss": 0.5, "accuracy": 0.8})
run.log({"train-loss": 0.4, "accuracy": 0.9})

# 자동 증가 없는 암시적 단계:
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

| 인수 |  |
| :--- | :--- |
|  `data` |  값이 직렬화 가능한 Python 객체(`int`, `float`, `string`, wandb.data_types 등)를 포함하는 `str` 키를 가진 `딕셔너리`. |
|  `step` |  기록할 단계 번호입니다. `None`이면 암시적 자동 증가 단계가 사용됩니다. |
|  `commit` |  true라면 단계를 확정하고 업로드합니다. false라면 단계에 대한 데이터를 누적합니다. |
|  `sync` |  이 인수는 더 이상 사용되지 않으며 아무 작업도 수행하지 않습니다. |

#### 예제:

자세한 예제는 
[로깅 가이드](https://docs.wandb.com/guides/track/log)를 참조하세요.

### 기본 사용법

```python
import wandb

run = wandb.init()
run.log({"accuracy": 0.9, "epoch": 5})
```

### 점진적 로깅

```python
import wandb

run = wandb.init()
run.log({"loss": 0.2}, commit=False)
# 보고할 준비가 되었을 때:
run.log({"accuracy": 0.8})
```

### 히스토그램

```python
import numpy as np
import wandb

# 정규 분포에서 임의의 그레이디언트 샘플 추출
gradients = np.random.randn(100, 100)
run = wandb.init()
run.log({"gradients": wandb.Histogram(gradients)})
```

### numpy로부터 이미지

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

### PIL로부터 이미지

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

### numpy로부터 비디오

```python
import numpy as np
import wandb

run = wandb.init()
# 축은 (시간, 채널, 높이, 너비)입니다
frames = np.random.randint(low=0, high=256, size=(10, 3, 100, 100), dtype=np.uint8)
run.log({"video": wandb.Video(frames, fps=4)})
```

### Matplotlib로 그리기

```python
from matplotlib import pyplot as plt
import numpy as np
import wandb

run = wandb.init()
fig, ax = plt.subplots()
x = np.linspace(0, 10)
y = x * x
ax.plot(x, y)  # y = x^2 그리기
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

| Raises |  |
| :--- | :--- |
|  `wandb.Error` |  `wandb.init` 이전에 호출한 경우 |
|  `ValueError` |  잘못된 데이터가 전달된 경우 |

### `log_artifact`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L3107-L3147)

```python
log_artifact(
    artifact_or_path: Union[Artifact, StrPath],
    name: Optional[str] = None,
    type: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    tags: Optional[List[str]] = None
) -> Artifact
```

run의 출력으로 아티팩트를 선언합니다.

| 인수 |  |
| :--- | :--- |
|  `artifact_or_path` |  (str 또는 Artifact) 이 아티팩트의 내용을 가리키는 경로로, 다음 형태로 제공될 수 있습니다: - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` 또한 `wandb.Artifact`를 호출하여 생성한 Artifact 오브젝트를 전달할 수 있습니다. |
|  `name` |  (str, optional) 아티팩트의 이름을 지정합니다. 유효한 이름은 다음 형태로 제공될 수 있습니다: - name:version - name:alias - digest 지정하지 않은 경우 기본적으로 현재 run ID가 붙은 경로의 basename으로 설정됩니다. |
|  `type` |  (str) 기록할 아티팩트의 유형입니다. 예시는 `dataset`, `model` 등입니다. |
|  `aliases` |  (list, optional) 이 아티팩트에 적용할 에일리어스입니다, 기본값은 `["latest"]`입니다. |
|  `tags` |  (list, optional) 이 아티팩트에 적용할 태그가 있는 경우. |

| 반환값 |  |
| :--- | :--- |
|  `Artifact` 오브젝트입니다. |

### `log_code`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L1147-L1230)

```python
log_code(
    root: Optional[str] = ".",
    name: Optional[str] = None,
    include_fn: Union[Callable[[str, str], bool], Callable[[str], bool]] = _is_py_or_dockerfile,
    exclude_fn: Union[Callable[[str, str], bool], Callable[[str], bool]] = filenames.exclude_wandb_fn
) -> Optional[Artifact]
```

현재 코드 상태를 W&B Artifact에 저장합니다.

기본적으로, 현재 디렉토리를 도보하면서 `.py`로 끝나는 모든 파일을 기록합니다.

| 인수 |  |
| :--- | :--- |
|  `root` |  코드 검색을 재귀적으로 찾을 상대(path) 혹은 절대 경로입니다. |
|  `name` |  (str, optional) 코드 아티팩트의 이름입니다. 기본적으로, `source-$PROJECT_ID-$ENTRYPOINT_RELPATH`라는 이름으로 설정됩니다. |
|  `include_fn` |  파일 경로 및(선택적으로) 루트 경로를 받아 들여 기록해야 할 때 True를 반환하고 그렇지 않은 경우 False를 반환하는 함수입니다. 기본값은: `lambda path, root: path.endswith(".py")` |
|  `exclude_fn` |  파일 경로 및(선택적으로) 루트 경로를 받아 들여 제외해야 할 때 True를 반환하고 그렇지 않은 경우 False를 반환하는 함수입니다. 기본값은 `&lt;root&gt;/.wandb/` 및 `&lt;root&gt;/wandb/` 디렉토리 내의 모든 파일을 제외하는 함수입니다. |

#### Examples:

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
|  코드가 기록된 경우 Artifact 오브젝트를 반환 |

### `log_model`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L3396-L3445)

```python
log_model(
    path: StrPath,
    name: Optional[str] = None,
    aliases: Optional[List[str]] = None
) -> None
```

'path'에 있는 내용을 포함하는 모델 아티팩트를 기록하고 이를 run에 출력으로 표시합니다.

| 인수 |  |
| :--- | :--- |
|  `path` |  (str) 이 모델의 내용을 가리키는 경로입니다. 다음 형태일 수 있습니다: - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` |
|  `name` |  (str, optional) 파일 내용을 추가할 모델 아티팩트에 할당할 이름입니다. 문자열에는 알파벳, 숫자, 대시, 밑줄, 점만 포함해야 합니다. 지정하지 않은 경우 기본적으로 현재 run ID가 붙은 경로의 basename으로 설정됩니다. |
|  `aliases` |  (list, optional) 생성된 모델 아티팩트에 적용할 에일리어스입니다, 기본값은 `["latest"]`입니다. |

#### 예시:

```python
run.log_model(
    path="/local/directory",
    name="my_model_artifact",
    aliases=["production"],
)
```

잘못된 사용

```python
run.log_model(
    path="/local/directory",
    name="my_entity/my_project/my_model_artifact",
    aliases=["production"],
)
```

| Raises |  |
| :--- | :--- |
|  `ValueError` |  name에 잘못된 특수 문자가 포함된 경우 |

| 반환값 |  |
| :--- | :--- |
|  None |

### `mark_preempting`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L3645-L3653)

```python
mark_preempting() -> None
```

이 run을 preempting으로 표시합니다.

또한 내부 프로세스에 이를 서버에 즉시 보고하라고 지시합니다.

### `plot_table`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L2248-L2269)

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

테이블에 커스텀 플롯을 생성합니다.

| 인수 |  |
| :--- | :--- |
|  `vega_spec_name` |  플롯의 사양 이름입니다. |
|  `data_table` |  시각화에 사용될 데이터를 포함하는 wandb.Table 오브젝트 |
|  `fields` |  커스텀 시각화가 필요한 필드로 테이블 키를 매핑하는 딕셔너리 |
|  `string_fields` |  커스텀 시각화에 필요한 문자열 상수에 대한 값을 제공하는 딕셔너리 |

### `project_name`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L1092-L1093)

```python
project_name() -> str
```

### `restore`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L2127-L2140)

```python
restore(
    name: str,
    run_path: Optional[str] = None,
    replace: bool = (False),
    root: Optional[str] = None
) -> Union[None, TextIO]
```

클라우드 저장소에서 지정된 파일을 다운로드합니다.

파일은 현재 디렉토리 또는 실행 디렉토리에 배치됩니다.
기본적으로, 파일이 이미 존재하지 않는 경우에만 다운로드합니다.

| 인수 |  |
| :--- | :--- |
|  `name` |  파일의 이름입니다. |
|  `run_path` |  파일을 가져올 run의 경로이며, 필요에 따라 `username/project_name/run_id` 형식으로 지정합니다. 만약 wandb.init가 호출되지 않았다면, 이는 필수입니다. |
|  `replace` |  파일이 로컬에 이미 존재해도 다운로드할지 여부입니다. |
|  `root` |  파일을 다운로드 받을 디렉토리를 지정합니다. wandb.init가 호출된 경우 기본값은 현재 디렉토리 또는 run 디렉토리입니다. |

| 반환값 |  |
| :--- | :--- |
|  파일을 찾을 수 없는 경우 None, 그렇지 않으면 읽을 수 있는 파일 객체 |

| Raises |  |
| :--- | :--- |
|  `wandb.CommError` |  wandb 백엔드에 연결할 수 없는 경우 |
|  `ValueError` |  파일을 찾을 수 없거나 run_path를 찾을 수 없는 경우 |

### `save`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L1935-L2041)

```python
save(
    glob_str: Optional[Union[str, os.PathLike]] = None,
    base_path: Optional[Union[str, os.PathLike]] = None,
    policy: PolicyName = "live"
) -> Union[bool, List[str]]
```

하나 이상의 파일을 W&B에 동기화합니다.

상대 경로는 현재 작업 디렉토리를 기준으로 합니다.

"myfiles/*"와 같은 유닉스 글로브는 `save`가
호출될 때 확장됩니다. `policy`와는 무관하게 호출됩니다.
특히, 새 파일은 자동으로 선택되지 않습니다.

업로드된 파일의 디렉토리 구조를 제어하기 위해 `base_path`를 제공할 수 있습니다. 이는 `glob_str`의 접두사여야 하며, 하위 디렉토리 구조는 유지됩니다. 예제를 통해 쉽게 이해할 수 있습니다:

```
wandb.save("these/are/myfiles/*")
# => run의 "these/are/myfiles/" 폴더에 파일을 저장합니다.

wandb.save("these/are/myfiles/*", base_path="these")
# => run의 "are/myfiles/" 폴더에 파일을 저장합니다.

wandb.save("/User/username/Documents/run123/*.txt")
# => run의 "run123/" 폴더에 파일을 저장합니다. 아래 참고사항을 확인하세요.

wandb.save("/User/username/Documents/run123/*.txt", base_path="/User")
# => run의 "username/Documents/run123/" 폴더에 파일을 저장합니다.

wandb.save("files/*/saveme.txt")
# => 각 "saveme.txt" 파일을 "files/"의 적절한 하위 디렉토리에 저장합니다.
```

참고: 절대 경로나 글로브를 사용하고 `base_path`를 제공하지 않을 때는 예제와 같이 한 디렉토리 수준이 유지됩니다.

| 인수 |  |
| :--- | :--- |
|  `glob_str` |  상대 또는 절대 경로나 유닉스 글로브입니다. |
|  `base_path` |  디렉토리 구조를 추론하는 데 사용할 경로로, 예제를 참조하세요. |
|  `policy` |  "live", "now", 또는 "end" 중 하나입니다. * live: 파일이 변경될 때 삽입하고 이전 버전을 덮어씁니다 * now: 현재 단일 업로드 * end: run이 끝날 때 파일 업로드 |

| 반환값 |  |
| :--- | :--- |
|  일치하는 파일에 대해 생성된 심볼릭 링크의 경로입니다. 역사적인 이유로, 레거시 코드에서는 불리언을 반환할 수 있습니다. |

### `status`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L2223-L2246)

```python
status() -> RunStatus
```

현재 run의 동기화 상태에 대한 내부 백엔드로부터의 동기화 정보를 가져옵니다.

### `to_html`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L1370-L1379)

```python
to_html(
    height: int = 420,
    hidden: bool = (False)
) -> str
```

현재 run을 표시하는 iframe을 포함하는 HTML을 생성합니다.

### `unwatch`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L2901-L2903)

```python
unwatch(
    models=None
) -> None
```

### `upsert_artifact`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L3149-L3201)

```python
upsert_artifact(
    artifact_or_path: Union[Artifact, str],
    name: Optional[str] = None,
    type: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    distributed_id: Optional[str] = None
) -> Artifact
```

run의 출력으로 비완성된 아티팩트를 선언(또는 추가)합니다.

artiface를 최종 완료하려면 반드시 run.finish_artifact()를 호출해야 한다는 점에 유의하세요. 분산 작업이 동일한 artifact에 모두 기여해야 할 때 유용합니다.

| 인수 |  |
| :--- | :--- |
|  `artifact_or_path` |  (str 또는 Artifact) 이 아티팩트의 내용을 가리키는 경로로, 다음 형태로 제공될 수 있습니다: - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` 또한 `wandb.Artifact`를 호출하여 생성한 Artifact 오브젝트를 전달할 수 있습니다. |
|  `name` |  (str, optional) 아티팩트의 이름을 지정합니다. entity/project 접두사를 포함할 수 있습니다. 유효한 이름은 다음 형태로 제공될 수 있습니다: - name:version - name:alias - digest 지정하지 않은 경우 기본으로 현재 run ID가 붙은 경로의 basename으로 설정됩니다. |
|  `type` |  (str) 기록할 아티팩트의 유형입니다. 예시는 `dataset`, `model` 등입니다. |
|  `aliases` |  (list, optional) 이 아티팩트에 적용할 에일리어스입니다, 기본값은 `["latest"]`입니다. |
|  `distributed_id` |  (string, optional) 모든 분산 작업이 공유하는 고유 문자열입니다. None일 경우, run의 그룹 이름을 기본값으로 합니다. |

| 반환값 |  |
| :--- | :--- |
|  `Artifact` 오브젝트입니다. |

### `use_artifact`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L2998-L3105)

```python
use_artifact(
    artifact_or_name: Union[str, Artifact],
    type: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    use_as: Optional[str] = None
) -> Artifact
```

run의 입력으로 아티팩트를 선언합니다.

로컬에서 내용을 얻으려면 반환된 오브젝트에서 `download`나 `file`을 호출합니다.

| 인수 |  |
| :--- | :--- |
|  `artifact_or_name` |  (str 또는 Artifact) 아티팩트의 이름입니다. entity/project/ 접두사를 포함할 수 있습니다. 유효한 이름은 다음 형태로 제공될 수 있습니다: - name:version - name:alias 또한 `wandb.Artifact`를 통해 생성된 Artifact 오브젝트를 전달할 수 있습니다. |
|  `type` |  (str, optional) 사용할 아티팩트의 유형입니다. |
|  `aliases` |  (list, optional) 이 아티팩트에 적용할 에일리어스입니다. |
|  `use_as` |  (string, optional) 아티팩트를 사용한 목적을 나타내는 선택적 문자열입니다. UI에 표시됩니다. |

| 반환값 |  |
| :--- | :--- |
|  `Artifact` 오브젝트입니다. |

### `use_model`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L3447-L3498)

```python
use_model(
    name: str
) -> FilePathStr
```

모델 아티팩트 'name'에 기록된 파일을 다운로드합니다.

| 인수 |  |
| :--- | :--- |
|  `name` |  (str) 모델 아티팩트의 이름입니다. 'name'은 기존에 기록된 모델 아티팩트의 이름과 일치해야 합니다. entity/project/ 접두사를 포함할 수 있습니다. 유효한 이름은 다음 형태로 제공될 수 있습니다: - model_artifact_name:version - model_artifact_name:alias |

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

| Raises |  |
| :--- | :--- |
|  `AssertionError` |  모델 아티팩트 'name'의 유형에 'model'이라는 문자열이 포함되어 있지 않은 경우. |

| 반환값 |  |
| :--- | :--- |
|  `path` |  (str) 다운로드된 모델 아티팩트 파일의 경로. |

### `watch`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L2888-L2898)

```python
watch(
    models, criterion=None, log="gradients", log_freq=100, idx=None,
    log_graph=(False)
) -> None
```

### `__enter__`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L3629-L3630)

```python
__enter__() -> "Run"
```

### `__exit__`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L3632-L3643)

```python
__exit__(
    exc_type: Type[BaseException],
    exc_val: BaseException,
    exc_tb: TracebackType
) -> bool
```