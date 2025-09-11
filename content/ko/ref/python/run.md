---
title: Run
menu:
  reference:
    identifier: ko-ref-python-run
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L461-L4042 >}}

wandb에 의해 기록되는 계산 단위입니다. 일반적으로 이는 ML 실험입니다.

```python
Run(
    settings: Settings,
    config: (dict[str, Any] | None) = None,
    sweep_config: (dict[str, Any] | None) = None,
    launch_config: (dict[str, Any] | None) = None
) -> None
```

`wandb.init()`으로 run을 생성합니다:

```python
import wandb

run = wandb.init()
```

어떤 프로세스에서든 최대 하나의 활성 `wandb.Run`만 존재하며,
`wandb.run`으로 엑세스할 수 있습니다:

```python
import wandb

assert wandb.run is None

wandb.init()

assert wandb.run is not None
```

`wandb.log`로 기록하는 모든 내용은 해당 run으로 전송됩니다.

동일한 스크립트 또는 노트북에서 더 많은 run을 시작하려면 진행 중인 run을
완료해야 합니다. Run은 `wandb.finish`를 사용하거나
`with` 블록에서 사용하여 완료할 수 있습니다:

```python
import wandb

wandb.init()
wandb.finish()

assert wandb.run is None

with wandb.init() as run:
    pass  # 여기에 데이터 기록

assert wandb.run is None
```

run 생성에 대한 자세한 내용은 `wandb.init` 문서를 참조하거나
[`wandb.init` 가이드](https://docs.wandb.ai/guides/track/launch)를 확인하세요.

분산 트레이닝에서 순위 0 프로세스에서 단일 run을 생성한 다음 해당 프로세스에서만 정보를 기록하거나, 각 프로세스에서 run을 생성하고 각 프로세스에서 별도로 기록한 다음 `wandb.init`에 대한 `group` 인수로 결과를 함께 그룹화할 수 있습니다. W&B를 사용한 분산 트레이닝에 대한 자세한 내용은
[가이드](https://docs.wandb.ai/guides/track/log/distributed-training)를 확인하세요.

현재 `wandb.Api`에 병렬 `Run` 오브젝트가 있습니다. 결국 이 두 오브젝트는 병합될 것입니다.

| 속성 |  |
| :--- | :--- |
| `summary` | (`Summary`) 각 `wandb.log()` 키에 대해 설정된 단일 값입니다. 기본적으로 summary는 마지막으로 기록된 값으로 설정됩니다. 최대 정확도와 같이 최종 값 대신 수동으로 summary를 최적 값으로 설정할 수 있습니다. |
| `config` | 이 run과 관련된 Config 오브젝트입니다. |
| `dir` | run과 관련된 파일이 저장되는 디렉토리입니다. |
| `entity` | run과 관련된 W&B 엔티티의 이름입니다. 엔티티는 사용자 이름이거나 팀 또는 조직의 이름일 수 있습니다. |
| `group` | run과 관련된 그룹의 이름입니다. 그룹을 설정하면 W&B UI에서 run을 합리적인 방식으로 구성하는 데 도움이 됩니다. 분산 트레이닝을 수행하는 경우 트레이닝의 모든 run에 동일한 그룹을 지정해야 합니다. 교차 검증을 수행하는 경우 모든 교차 검증 폴드에 동일한 그룹을 지정해야 합니다. |
| `id` | 이 run의 식별자입니다. |
| `mode` | `0.9.x` 및 이전 버전과의 호환성을 위해 결국 더 이상 사용되지 않습니다. |
| `name` | run의 표시 이름입니다. 표시 이름은 고유성이 보장되지 않으며 설명적일 수 있습니다. 기본적으로 무작위로 생성됩니다. |
| `notes` | run과 관련된 메모(있는 경우)입니다. 메모는 여러 줄 문자열일 수 있으며 `$$` 안에 마크다운 및 라텍스 수식을 사용할 수도 있습니다(예: `$x + 3$`). |
| `path` | run의 경로입니다. Run 경로는 `entity/project/run_id` 형식으로 엔티티, 프로젝트 및 run ID를 포함합니다. |
| `project` | run과 관련된 W&B 프로젝트의 이름입니다. |
| `resumed` | run이 재개된 경우 True, 그렇지 않으면 False입니다. |
| `settings` | run의 Settings 오브젝트의 고정된 복사본입니다. |
| `start_time` | run이 시작된 시점의 유닉스 타임스탬프(초)입니다. |
| `starting_step` | run의 첫 번째 단계입니다. |
| `step` | 단계의 현재 값입니다. 이 카운터는 `wandb.log`에 의해 증가됩니다. |
| `sweep_id` | run과 관련된 스윕의 ID(있는 경우)입니다. |
| `tags` | run과 관련된 태그(있는 경우)입니다. |
| `url` | run과 관련된 W&B URL입니다. |

## 메소드

### `alert`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L3567-L3600)

```python
alert(
    title: str,
    text: str,
    level: (str | AlertLevel | None) = None,
    wait_duration: (int | float | timedelta | None) = None
) -> None
```

지정된 제목과 텍스트로 알림을 시작합니다.

| 인수 |  |
| :--- | :--- |
| `title` | (str) 알림 제목입니다. 64자 미만이어야 합니다. |
| `text` | (str) 알림의 텍스트 본문입니다. |
| `level` | (str 또는 AlertLevel, 선택 사항) 사용할 알림 수준입니다. `INFO`, `WARN` 또는 `ERROR` 중 하나입니다. |
| `wait_duration` | (int, float 또는 timedelta, 선택 사항) 이 제목으로 다른 알림을 보내기 전에 대기할 시간(초)입니다. |

### `define_metric`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L2660-L2721)

```python
define_metric(
    name: str,
    step_metric: (str | wandb_metric.Metric | None) = None,
    step_sync: (bool | None) = None,
    hidden: (bool | None) = None,
    summary: (str | None) = None,
    goal: (str | None) = None,
    overwrite: (bool | None) = None
) -> wandb_metric.Metric
```

`wandb.log()`로 기록된 메트릭을 사용자 정의합니다.

| 인수 |  |
| :--- | :--- |
| `name` | 사용자 정의할 메트릭의 이름입니다. |
| `step_metric` | 자동으로 생성된 차트에서 이 메트릭의 X축 역할을 할 다른 메트릭의 이름입니다. |
| `step_sync` | step_metric이 명시적으로 제공되지 않은 경우 마지막 값을 `run.log()`에 자동으로 삽입합니다. step_metric이 지정된 경우 기본값은 True입니다. |
| `hidden` | 이 메트릭을 자동 플롯에서 숨깁니다. |
| `summary` | summary에 추가된 집계 메트릭을 지정합니다. 지원되는 집계에는 "min", "max", "mean", "last", "best", "copy" 및 "none"이 있습니다. "best"는 goal 파라미터와 함께 사용됩니다. "none"은 summary가 생성되지 않도록 합니다. "copy"는 더 이상 사용되지 않으며 사용해서는 안 됩니다. |
| `goal` | "best" summary 유형을 해석하는 방법을 지정합니다. 지원되는 옵션은 "minimize" 및 "maximize"입니다. |
| `overwrite` | False인 경우 이 호출은 지정되지 않은 파라미터에 대한 값을 사용하여 동일한 메트릭에 대한 이전 `define_metric` 호출과 병합됩니다. True인 경우 지정되지 않은 파라미터는 이전 호출에서 지정된 값을 덮어씁니다. |

| 반환 |  |
| :--- | :--- |
| 이 호출을 나타내는 오브젝트이지만 그렇지 않으면 삭제될 수 있습니다. |

### `detach`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L2885-L2886)

```python
detach() -> None
```

### `display`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L1219-L1236)

```python
display(
    height: int = 420,
    hidden: bool = (False)
) -> bool
```

이 run을 jupyter에서 표시합니다.

### `finish`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L2075-L2106)

```python
finish(
    exit_code: (int | None) = None,
    quiet: (bool | None) = None
) -> None
```

run을 완료하고 남은 데이터를 업로드합니다.

W&B run의 완료를 표시하고 모든 데이터가 서버에 동기화되도록 합니다.
run의 최종 상태는 종료 조건 및 동기화 상태에 따라 결정됩니다.

#### Run 상태:

- Running: 데이터를 기록하거나 하트비트를 보내는 활성 run입니다.
- Crashed: 예기치 않게 하트비트 전송을 중단한 run입니다.
- Finished: 모든 데이터가 동기화된 상태로 성공적으로 완료된 run입니다(`exit_code=0`).
- Failed: 오류가 발생하여 완료된 run입니다(`exit_code!=0`).

| 인수 |  |
| :--- | :--- |
| `exit_code` | run의 종료 상태를 나타내는 정수입니다. 성공에는 0을 사용하고 다른 값은 run을 실패로 표시합니다. |
| `quiet` | 더 이상 사용되지 않습니다. `wandb.Settings(quiet=...)`를 사용하여 로깅 verbosity를 구성합니다. |

### `finish_artifact`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L3163-L3215)

```python
finish_artifact(
    artifact_or_path: (Artifact | str),
    name: (str | None) = None,
    type: (str | None) = None,
    aliases: (list[str] | None) = None,
    distributed_id: (str | None) = None
) -> Artifact
```

run의 출력으로 아직 완료되지 않은 아티팩트를 완료합니다.

동일한 분산 ID를 사용한 후속 "업서트"는 새 버전을 생성합니다.

| 인수 |  |
| :--- | :--- |
| `artifact_or_path` | (str 또는 Artifact) 이 아티팩트의 콘텐츠에 대한 경로입니다. 다음 형식이 될 수 있습니다. - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` `wandb.Artifact`를 호출하여 생성된 Artifact 오브젝트를 전달할 수도 있습니다. |
| `name` | (str, 선택 사항) 아티팩트 이름입니다. 엔티티/프로젝트로 시작할 수 있습니다. 유효한 이름은 다음 형식이 될 수 있습니다. - name:version - name:alias - digest 지정하지 않으면 기본적으로 현재 run ID가 접두사로 붙은 경로의 기본 이름으로 설정됩니다. |
| `type` | (str) 기록할 아티팩트의 유형입니다. 예로는 `dataset`, `model`이 있습니다. |
| `aliases` | (list, 선택 사항) 이 아티팩트에 적용할 에일리어스입니다. 기본값은 `["latest"]`입니다. |
| `distributed_id` | (string, 선택 사항) 모든 분산 작업이 공유하는 고유한 문자열입니다. None인 경우 run의 그룹 이름으로 기본 설정됩니다. |

| 반환 |  |
| :--- | :--- |
| `Artifact` 오브젝트입니다. |

### `get_project_url`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L1091-L1099)

```python
get_project_url() -> (str | None)
```

run과 관련된 W&B 프로젝트의 URL을 반환합니다(있는 경우).

오프라인 run에는 프로젝트 URL이 없습니다.

### `get_sweep_url`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L1101-L1106)

```python
get_sweep_url() -> (str | None)
```

run과 관련된 스윕의 URL을 반환합니다(있는 경우).

### `get_url`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L1108-L1116)

```python
get_url() -> (str | None)
```

W&B run의 URL을 반환합니다(있는 경우).

오프라인 run에는 URL이 없습니다.

### `join`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L2148-L2159)

```python
join(
    exit_code: (int | None) = None
) -> None
```

`finish()`의 더 이상 사용되지 않는 에일리어스입니다. 대신 finish를 사용하세요.

### `link_artifact`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L2888-L2951)

```python
link_artifact(
    artifact: Artifact,
    target_path: str,
    aliases: (list[str] | None) = None
) -> None
```

지정된 아티팩트를 포트폴리오(승격된 아티팩트 모음)에 연결합니다.

연결된 아티팩트는 지정된 포트폴리오의 UI에 표시됩니다.

| 인수 |  |
| :--- | :--- |
| `artifact` | 연결될 (공개 또는 로컬) 아티팩트입니다. |
| `target_path` | `str` - 다음 형식을 취합니다. `{portfolio}`, `{project}/{portfolio}` 또는 `{entity}/{project}/{portfolio}` |
| `aliases` | `List[str]` - 선택적 에일리어스로, 포트폴리오 내의 이 연결된 아티팩트에만 적용됩니다. "latest" 에일리어스는 항상 연결된 아티팩트의 최신 버전에 적용됩니다. |

| 반환 |  |
| :--- | :--- |
| None |

### `link_model`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L3466-L3565)

```python
link_model(
    path: StrPath,
    registered_model_name: str,
    name: (str | None) = None,
    aliases: (list[str] | None) = None
) -> None
```

모델 아티팩트 버전을 기록하고 모델 레지스트리의 등록된 모델에 연결합니다.

연결된 모델 버전은 지정된 등록된 모델의 UI에 표시됩니다.

#### 단계:

- 'name' 모델 아티팩트가 기록되었는지 확인합니다. 그렇다면 'path'에 있는 파일과 일치하는 아티팩트 버전을 사용하거나 새 버전을 기록합니다. 그렇지 않으면 'path' 아래의 파일을 'model' 유형의 새 모델 아티팩트 'name'으로 기록합니다.
- 이름이 'registered_model_name'인 등록된 모델이 'model-registry' 프로젝트에 있는지 확인합니다. 그렇지 않으면 이름이 'registered_model_name'인 새 등록된 모델을 만듭니다.
- 모델 아티팩트 'name' 버전을 등록된 모델 'registered_model_name'에 연결합니다.
- 'aliases' 목록에서 에일리어스를 새로 연결된 모델 아티팩트 버전에 연결합니다.

| 인수 |  |
| :--- | :--- |
| `path` | (str) 이 모델의 콘텐츠에 대한 경로입니다. 다음 형식이 될 수 있습니다. - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` |
| `registered_model_name` | (str) - 모델을 연결할 등록된 모델의 이름입니다. 등록된 모델은 모델 레지스트리에 연결된 모델 버전의 모음으로, 일반적으로 팀의 특정 ML 작업을 나타냅니다. 이 등록된 모델이 속한 엔티티는 run에서 파생됩니다. |
| `name` | (str, 선택 사항) - 'path'의 파일이 기록될 모델 아티팩트의 이름입니다. 지정하지 않으면 기본적으로 현재 run ID가 접두사로 붙은 경로의 기본 이름으로 설정됩니다. |
| `aliases` | (List[str], 선택 사항) - 등록된 모델 내의 이 연결된 아티팩트에만 적용될 에일리어스입니다. "latest" 에일리어스는 항상 연결된 아티팩트의 최신 버전에 적용됩니다. |

#### 예:

```python
run.link_model(
    path="/local/directory",
    registered_model_name="my_reg_model",
    name="my_model_artifact",
    aliases=["production"],
)
```

잘못된 사용법

```python
run.link_model(
    path="/local/directory",
    registered_model_name="my_entity/my_project/my_reg_model",
    name="my_model_artifact",
    aliases=["production"],
)

run.link_model(
    path="/local/directory",
    registered_model_name="my_reg_model",
    name="my_entity/my_project/my_model_artifact",
    aliases=["production"],
)
```

| 발생 |  |
| :--- | :--- |
| `AssertionError` | registered_model_name이 경로이거나 모델 아티팩트 'name'이 'model' 하위 문자열을 포함하지 않는 유형인 경우 |
| `ValueError` | 이름에 잘못된 특수 문자가 있는 경우 |

| 반환 |  |
| :--- | :--- |
| None |

### `log`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L1613-L1873)

```python
log(
    data: dict[str, Any],
    step: (int | None) = None,
    commit: (bool | None) = None,
    sync: (bool | None) = None
) -> None
```

run 데이터를 업로드합니다.

`log`를 사용하여 스칼라, 이미지, 비디오,
히스토그램, 플롯 및 테이블과 같은 run 데이터를 기록합니다.

라이브 예제, 코드 조각, 모범 사례 등에 대한
[로깅 가이드](https://docs.wandb.ai/guides/track/log)를 참조하세요.

가장 기본적인 사용법은 `run.log({"train-loss": 0.5, "accuracy": 0.9})`입니다.
이렇게 하면 손실 및 정확도가 run의 기록에 저장되고
이러한 메트릭에 대한 summary 값이 업데이트됩니다.

[wandb.ai](https://wandb.ai)의 워크스페이스 또는
W&B 앱의 [자체 호스팅 인스턴스](https://docs.wandb.ai/guides/hosting)에서 기록된 데이터를 시각화하거나,
[API](https://docs.wandb.ai/guides/track/public-api-guide)를 사용하여 로컬에서 시각화하고 탐색할 데이터를 내보냅니다(예: Jupyter 노트북).

기록된 값은 스칼라일 필요가 없습니다. 모든 wandb 오브젝트의 로깅이 지원됩니다.
예를 들어 `run.log({"example": wandb.Image("myimage.jpg")})`는 W&B UI에 멋지게 표시될 예제 이미지를 기록합니다.
지원되는 모든 유형에 대한
[참조 문서](https://docs.wandb.ai/ref/python/sdk/data-types/)를 참조하거나
3D 분자 구조 및 분할 마스크에서 PR 곡선 및 히스토그램에 이르기까지 예제에 대한
[로깅 가이드](https://docs.wandb.ai/guides/track/log)를 확인하세요.
`wandb.Table`을 사용하여 구조화된 데이터를 기록할 수 있습니다. 자세한 내용은
[테이블 로깅 가이드](https://docs.wandb.ai/guides/models/tables/tables-walkthrough)를 참조하세요.

W&B UI는 이름에 슬래시(`/`)가 있는 메트릭을
최종 슬래시 앞의 텍스트를 사용하여 명명된 섹션으로 구성합니다. 예를 들어,
다음은 "train" 및 "validate"라는 두 개의 섹션을 생성합니다.

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

한 수준의 중첩만 지원됩니다. `run.log({"a/b/c": 1})`는
"a/b"라는 이름의 섹션을 생성합니다.

`run.log`는 초당 몇 번 이상 호출하도록 설계되지 않았습니다.
최적의 성능을 위해 로깅을 N번 반복할 때마다 한 번으로 제한하거나
여러 반복에 걸쳐 데이터를 수집하고 단일 단계로 기록합니다.

### W&B 단계

기본 사용법에서 `log`를 호출할 때마다 새 "단계"가 생성됩니다.
단계는 항상 증가해야 하며
이전 단계에 기록하는 것은 불가능합니다.

차트에서 모든 메트릭을 X축으로 사용할 수 있습니다.
많은 경우 W&B 단계를 트레이닝 단계가 아닌
타임스탬프로 취급하는 것이 좋습니다.

```
# 예: X축으로 사용할 "에포크" 메트릭을 기록합니다.
run.log({"epoch": 40, "train-loss": 0.5})
```

[define_metric](https://docs.wandb.ai/ref/python/sdk/experiments/run/#method-rundefine_metric)도 참조하세요.

여러 `log` 호출을 사용하여
`step` 및 `commit` 파라미터로 동일한 단계에 기록할 수 있습니다.
다음은 모두 동일합니다.

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

| 인수 |  |
| :--- | :--- |
| `data` | `str` 키와 값이 직렬화 가능한 Python 오브젝트인 `dict`입니다(예: `int`, `float` 및 `string`). `wandb.data_types`, 직렬화 가능한 Python 오브젝트의 목록, 튜플 및 NumPy 배열, 이 구조의 다른 `dict`를 포함합니다. |
| `step` | 기록할 단계 번호입니다. `None`인 경우 암시적 자동 증가 단계가 사용됩니다. 설명에서 참고 사항을 참조하세요. |
| `commit` | True이면 단계를 완료하고 업로드합니다. False이면 단계에 대한 데이터를 축적합니다. 설명에서 참고 사항을 참조하세요. `step`이 `None`이면 기본값은 `commit=True`입니다. 그렇지 않으면 기본값은 `commit=False`입니다. |
| `sync` | 이 인수는 더 이상 사용되지 않으며 아무 작업도 수행하지 않습니다. |

#### 예:

자세한 내용은
[로깅 가이드](https://docs.wandb.com/guides/track/log)를 참조하세요.

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
# 이 단계를 보고할 준비가 되면 다른 위치에서:
run.log({"accuracy": 0.8})
```

### 히스토그램

```python
import numpy as np
import wandb

# 정규 분포에서 임의로 그레이디언트 샘플링
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

run = wandb.init()
# 축은 (시간, 채널, 높이, 너비)입니다.
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
| `wandb.Error` | `wandb.init` 전에 호출된 경우 |
| `ValueError` | 잘못된 데이터가 전달된 경우 |

### `log_artifact`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L3067-L3107)

```python
log_artifact(
    artifact_or_path: (Artifact | StrPath),
    name: (str | None) = None,
    type: (str | None) = None,
    aliases: (list[str] | None) = None,
    tags: (list[str] | None) = None
) -> Artifact
```

아티팩트를 run의 출력으로 선언합니다.

| 인수 |  |
| :--- | :--- |
| `artifact_or_path` | (str 또는 Artifact) 이 아티팩트의 콘텐츠에 대한 경로입니다. 다음 형식이 될 수 있습니다. - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` `wandb.Artifact`를 호출하여 생성된 Artifact 오브젝트를 전달할 수도 있습니다. |
| `name` | (str, 선택 사항) 아티팩트 이름입니다. 유효한 이름은 다음 형식이 될 수 있습니다. - name:version - name:alias - digest 지정하지 않으면 기본적으로 현재 run ID가 접두사로 붙은 경로의 기본 이름으로 설정됩니다. |
| `type` | (str) 기록할 아티팩트의 유형입니다. 예로는 `dataset`, `model`이 있습니다. |
| `aliases` | (list, 선택 사항) 이 아티팩트에 적용할 에일리어스입니다. 기본값은 `["latest"]`입니다. |
| `tags` | (list, 선택 사항) 이 아티팩트에 적용할 태그입니다(있는 경우). |

| 반환 |  |
| :--- | :--- |
| `Artifact` 오브젝트입니다. |

### `log_code`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L1004-L1089)

```python
log_code(
    root: (str | None) = ".",
    name: (str | None) = None,
    include_fn: (Callable[[str, str], bool] | Callable[[str], bool]) = _is_py_requirements_or_dockerfile,
    exclude_fn: (Callable[[str, str], bool] | Callable[[str], bool]) = filenames.exclude_wandb_fn
) -> (Artifact | None)
```

코드의 현재 상태를 W&B Artifact에 저장합니다.

기본적으로 현재 디렉토리를 탐색하고 `.py`로 끝나는 모든 파일을 기록합니다.

| 인수 |  |
| :--- | :--- |
| `root` | `os.getcwd()`에 대한 상대 경로 또는 코드를 재귀적으로 찾을 절대 경로입니다. |
| `name` | (str, 선택 사항) 코드 아티팩트의 이름입니다. 기본적으로 아티팩트 이름을 `source-$PROJECT_ID-$ENTRYPOINT_RELPATH`로 지정합니다. 여러 run이 동일한 아티팩트를 공유하도록 하려는 시나리오가 있을 수 있습니다. 이름을 지정하면 이를 달성할 수 있습니다. |
| `include_fn` | 파일 경로와 (선택적으로) 루트 경로를 허용하고 포함해야 하는 경우 True를 반환하고 그렇지 않으면 False를 반환하는 호출 가능 항목입니다. 기본값은 `lambda path, root: path.endswith(".py")`입니다. |
| `exclude_fn` | 파일 경로와 (선택적으로) 루트 경로를 허용하고 제외해야 하는 경우 `True`를 반환하고 그렇지 않으면 `False`를 반환하는 호출 가능 항목입니다. 기본값은 `<root>/.wandb/` 및 `<root>/wandb/` 디렉토리 내의 모든 파일을 제외하는 함수입니다. |

#### 예:

기본 사용법

```python
run.log_code()
```

고급 사용법

```python
run.log_code(
    "../",
    include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"),
    exclude_fn=lambda path, root: os.path.relpath(path, root).startswith(
        "cache/"
    ),
)
```

| 반환 |  |
| :--- | :--- |
| 코드가 기록된 경우 `Artifact` 오브젝트입니다. |

### `log_model`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L3358-L3407)

```python
log_model(
    path: StrPath,
    name: (str | None) = None,
    aliases: (list[str] | None) = None
) -> None
```

'path' 내부의 콘텐츠를 포함하는 모델 아티팩트를 run에 기록하고 이 run에 대한 출력으로 표시합니다.

| 인수 |  |
| :--- | :--- |
| `path` | (str) 이 모델의 콘텐츠에 대한 경로입니다. 다음 형식이 될 수 있습니다. - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` |
| `name` | (str, 선택 사항) 파일 콘텐츠가 추가될 모델 아티팩트에 할당할 이름입니다. 문자열에는 대시, 밑줄 및 점과 같은 영숫자 문자만 포함되어야 합니다. 지정하지 않으면 기본적으로 현재 run ID가 접두사로 붙은 경로의 기본 이름으로 설정됩니다. |
| `aliases` | (list, 선택 사항) 생성된 모델 아티팩트에 적용할 에일리어스입니다. 기본값은 `["latest"]`입니다. |

#### 예:

```python
run.log_model(
    path="/local/directory",
    name="my_model_artifact",
    aliases=["production"],
)
```

잘못된 사용법

```python
run.log_model(
    path="/local/directory",
    name="my_entity/my_project/my_model_artifact",
    aliases=["production"],
)
```

| 발생 |  |
| :--- | :--- |
| `ValueError` | 이름에 잘못된 특수 문자가 있는 경우 |

| 반환 |  |
| :--- | :--- |
| None |

### `mark_preempting`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L3618-L3626)

```python
mark_preempting() -> None
```

이 run을 선점하는 것으로 표시합니다.

또한 내부 프로세스에 이를 즉시 서버에 보고하도록 지시합니다.

### `project_name`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L994-L996)

```python
project_name() -> str
```

### `restore`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L2060-L2073)

```python
restore(
    name: str,
    run_path: (str | None) = None,
    replace: bool = (False),
    root: (str | None) = None
) -> (None | TextIO)
```

클라우드 스토리지에서 지정된 파일을 다운로드합니다.

파일은 현재 디렉토리 또는 run 디렉토리에 배치됩니다.
기본적으로 파일이 아직 존재하지 않는 경우에만 다운로드합니다.

| 인수 |  |
| :--- | :--- |
| `name` | 파일 이름입니다. |
| `run_path` | 파일을 가져올 run에 대한 선택적 경로입니다(예: `username/project_name/run_id`). wandb.init가 호출되지 않은 경우 필수입니다. |
| `replace` | 파일이 로컬에 이미 있더라도 파일을 다운로드할지 여부입니다. |
| `root` | 파일을 다운로드할 디렉토리입니다. 기본값은 현재 디렉토리 또는 wandb.init가 호출된 경우 run 디렉토리입니다. |

| 반환 |  |
| :--- | :--- |
| 파일을 찾을 수 없으면 None이고, 그렇지 않으면 읽기용으로 열린 파일 오브젝트입니다. |

| 발생 |  |
| :--- | :--- |
| `wandb.CommError` | wandb 백엔드에 연결할 수 없는 경우 |
| `ValueError` | 파일을 찾을 수 없거나 run_path를 찾을 수 없는 경우 |

### `save`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L1875-L1979)

```python
save(
    glob_str: (str | os.PathLike | None) = None,
    base_path: (str | os.PathLike | None) = None,
    policy: PolicyName = "live"
) -> (bool | list[str])
```

하나 이상의 파일을 W&B에 동기화합니다.

상대 경로는 현재 작업 디렉토리를 기준으로 합니다.

"myfiles/*"와 같은 Unix glob은 `policy`에 관계없이 `save`가
호출될 때 확장됩니다. 특히 새 파일은 자동으로 선택되지 않습니다.

`base_path`는 업로드된 파일의 디렉토리 구조를 제어하기 위해 제공될 수 있습니다.
`glob_str`의 접두사여야 하며
그 아래의 디렉토리 구조가 유지됩니다. 다음 예를 통해 가장 잘 이해할 수 있습니다.

```
wandb.save("these/are/myfiles/*")
# => run에서 "these/are/myfiles/" 폴더에 파일을 저장합니다.

wandb.save("these/are/myfiles/*", base_path="these")
# => run에서 "are/myfiles/" 폴더에 파일을 저장합니다.

wandb.save("/User/username/Documents/run123/*.txt")
# => run에서 "run123/" 폴더에 파일을 저장합니다. 아래 참고 사항을 참조하세요.

wandb.save("/User/username/Documents/run123/*.txt", base_path="/User")
# => run에서 "username/Documents/run123/" 폴더에 파일을 저장합니다.

wandb.save("files/*/saveme.txt")
# => "files/"의 적절한 하위 디렉토리에 각 "saveme.txt" 파일을 저장합니다.
```

참고: 절대 경로 또는 glob이 제공되고 `base_path`가 없는 경우 위 예제와 같이 한 디렉토리 수준이 유지됩니다.

| 인수 |  |
| :--- | :--- |
| `glob_str` | 상대 경로 또는 절대 경로 또는 Unix glob입니다. |
| `base_path` | 디렉토리 구조를 추론하는 데 사용할 경로입니다. 예를 참조하세요. |
| `policy` | `live`, `now` 또는 `end` 중 하나입니다. * live: 파일이 변경될 때 파일을 업로드하고 이전 버전을 덮어씁니다. * now: 파일을 지금 한 번 업로드합니다. * end: run이 종료될 때 파일을 업로드합니다. |

| 반환 |  |
| :--- | :--- |
| 일치하는 파일에 대해 생성된 심볼릭 링크의 경로입니다. 레거시 코드에서는 이전 이유로 인해 부울을 반환할 수 있습니다. |

### `status`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L2161-L2183)

```python
status() -> RunStatus
```

내부 백엔드에서 현재 run의 동기화 상태에 대한 동기화 정보를 가져옵니다.

### `to_html`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L1238-L1247)

```python
to_html(
    height: int = 420,
    hidden: bool = (False)
) -> str
```

현재 run을 표시하는 iframe을 포함하는 HTML을 생성합니다.

### `unwatch`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L2838-L2848)

```python
unwatch(
    models: (torch.nn.Module | Sequence[torch.nn.Module] | None) = None
) -> None
```

pytorch 모델 토폴로지, 그레이디언트 및 파라미터 훅을 제거합니다.

| 인수 |  |
| :--- | :--- |
| models (torch.nn.Module | Sequence[torch.nn.Module]): watch가 호출된 pytorch 모델의 선택적 목록 |

### `upsert_artifact`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L3109-L3161)

```python
upsert_artifact(
    artifact_or_path: (Artifact | str),
    name: (str | None) = None,
    type: (str | None) = None,
    aliases: (list[str] | None) = None,
    distributed_id: (str | None) = None
) -> Artifact
```

run의 출력으로 아직 완료되지 않은 아티팩트를 선언하거나 추가합니다.

아티팩트를 완료하려면 run.finish_artifact()를 호출해야 합니다.
이는 분산 작업이 모두 동일한 아티팩트에 기여해야 하는 경우에 유용합니다.

| 인수 |  |
| :--- | :--- |
| `artifact_or_path` | (str 또는 Artifact) 이 아티팩트의 콘텐츠에 대한 경로입니다. 다음 형식이 될 수 있습니다. - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` `wandb.Artifact`를 호출하여 생성된 Artifact 오브젝트를 전달할 수도 있습니다. |
| `name` | (str, 선택 사항) 아티팩트 이름입니다. 엔티티/프로젝트로 시작할 수 있습니다. 유효한 이름은 다음 형식이 될 수 있습니다. - name:version - name:alias - digest 지정하지 않으면 기본적으로 현재 run ID가 접두사로 붙은 경로의 기본 이름으로 설정됩니다. |
| `type` | (str) 기록할 아티팩트의 유형입니다. 예로는 `dataset`, `model`이 있습니다. |
| `aliases` | (list, 선택 사항) 이 아티팩트에 적용할 에일리어스입니다. 기본값은 `["latest"]`입니다. |
| `distributed_id` | (string, 선택 사항) 모든 분산 작업이 공유하는 고유한 문자열입니다. None인 경우 run의 그룹 이름으로 기본 설정됩니다. |

| 반환 |  |
| :--- | :--- |
| `Artifact` 오브젝트입니다. |

### `use_artifact`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L2953-L3065)

```python
use_artifact(
    artifact_or_name: (str | Artifact),
    type: (str | None) = None,
    aliases: (list[str] | None) = None,
    use_as: (str | None) = None
) -> Artifact
```

아티팩트를 run의 입력으로 선언합니다.

반환된 오브젝트에서 `download` 또는 `file`을 호출하여 콘텐츠를 로컬로 가져옵니다.

| 인수 |  |
| :--- | :--- |
| `artifact_or_name` | (str 또는 Artifact) 아티팩트 이름입니다. 프로젝트/ 또는 엔티티/프로젝트/로 시작할 수 있습니다. 이름에 엔티티가 지정되지 않은 경우 Run 또는 API 설정의 엔티티가 사용됩니다. 유효한 이름은 다음 형식이 될 수 있습니다. - name:version - name:alias `wandb.Artifact`를 호출하여 생성된 Artifact 오브젝트를 전달할 수도 있습니다. |
| `type` | (str, 선택 사항) 사용할 아티팩트의 유형입니다. |
| `aliases` | (list, 선택 사항) 이 아티팩트에 적용할 에일리어스입니다. |
| `use_as` | (string, 선택 사항) 아티팩트가 사용된 목적을 나타내는 선택적 문자열입니다. UI에 표시됩니다. |

| 반환 |  |
| :--- | :--- |
| `Artifact` 오브젝트입니다. |

### `use_model`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L3409-L3464)

```python
use_model(
    name: str
) -> FilePathStr
```

모델 아티팩트 'name'에 기록된 파일을 다운로드합니다.

| 인수 |  |
| :--- | :--- |
| `name` | (str) 모델 아티팩트 이름입니다. 'name'은 기존에 기록된 모델 아티팩트의 이름과 일치해야 합니다. 엔티티/프로젝트/로 시작할 수 있습니다. 유효한 이름은 다음 형식이 될 수 있습니다. - model_artifact_name:version - model_artifact_name:alias |

#### 예:

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

잘못된 사용법

```python
run.use_model(
    name="my_entity/my_project/my_model_artifact",
)
```

| 발생 |  |
| :--- | :--- |
| `AssertionError` | 모델 아티팩트 'name'이 'model' 하위 문자열을 포함하지 않는 유형인 경우 |

| 반환 |  |
| :--- | :--- |
| `path` | (str) 다운로드된 모델 아티팩트 파일의 경로입니다. |

### `watch`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L2801-L2836)

```python
watch(
    models: (torch.nn.Module | Sequence[torch.nn.Module]),
    criterion: (torch.F | None) = None,
    log: (Literal['gradients', 'parameters', 'all'] | None) = "gradients",
    log_freq: int = 1000,
    idx: (int | None) = None,
    log_graph: bool = (False)
) -> None
```

지정된 PyTorch 모델에 훅을 연결하여 그레이디언트 및 모델의 계산 그래프를 모니터링합니다.

이 함수는 트레이닝 중에 파라미터, 그레이디언트 또는 둘 다를 추적할 수 있습니다. 향후 임의의 기계 학습 모델을 지원하도록 확장해야 합니다.

| 인수 |  |
| :--- | :--- |
| models (Union[torch.nn.Module, Sequence[torch.nn.Module]]): 모니터링할 단일 모델 또는 모델 시퀀스입니다. criterion (Optional[torch.F]): 최적화할 손실 함수입니다(선택 사항). log (Optional[Literal["gradients", "parameters", "all"]]): "gradients", "parameters" 또는 "all"을 기록할지 여부를 지정합니다. 로깅을 비활성화하려면 None으로 설정합니다. (기본값="gradients") log_freq (int): 그레이디언트 및 파라미터를 기록할 빈도(배치 단위)입니다. (기본값=1000) idx (Optional[int]): `wandb.watch`로 여러 모델을 추적할 때 사용되는 인덱스입니다. (기본값=None) log_graph (bool): 모델의 계산 그래프를 기록할지 여부입니다. (기본값=False) |

| 발생 |  |
| :--- | :--- |
| `ValueError` | `wandb.init`가 호출되지 않았거나 모델이 `torch.nn.Module`의 인스턴스가 아닌 경우 |

### `__enter__`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L3602-L3603)

```python
__enter__() -> Run
```

### `__exit__`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L3605-L3616)

```python
__exit__(
    exc_type: type[BaseException],
    exc_val: BaseException,
    exc_tb: TracebackType
) -> bool
```