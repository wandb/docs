
# init

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/wandb_init.py#L940-L1215' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>

W&B에 추적 및 로그를 시작하는 새로운 실행을 시작합니다.

```python
init(
    job_type: Optional[str] = None,
    dir: Optional[StrPath] = None,
    config: Union[Dict, str, None] = None,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    reinit: Optional[bool] = None,
    tags: Optional[Sequence] = None,
    group: Optional[str] = None,
    name: Optional[str] = None,
    notes: Optional[str] = None,
    magic: Optional[Union[dict, str, bool]] = None,
    config_exclude_keys: Optional[List[str]] = None,
    config_include_keys: Optional[List[str]] = None,
    anonymous: Optional[str] = None,
    mode: Optional[str] = None,
    allow_val_change: Optional[bool] = None,
    resume: Optional[Union[bool, str]] = None,
    force: Optional[bool] = None,
    tensorboard: Optional[bool] = None,
    sync_tensorboard: Optional[bool] = None,
    monitor_gym: Optional[bool] = None,
    save_code: Optional[bool] = None,
    id: Optional[str] = None,
    settings: Union[Settings, Dict[str, Any], None] = None
) -> Union[Run, RunDisabled, None]
```

ML 학습 파이프라인에서 학습 스크립트와 평가 스크립트 시작 부분에 `wandb.init()`을 추가할 수 있으며, 각 부분이 W&B에서 실행으로 추적됩니다.

`wandb.init()`은 실행에 데이터를 로그하기 위한 새로운 백그라운드 프로세스를 생성하며, 기본적으로 wandb.ai에 데이터를 동기화하여 실시간 시각화를 볼 수 있습니다.

데이터를 로깅하기 전에 `wandb.init()`을 호출하세요:

```python
import wandb

wandb.init()
# ... 메트릭 계산, 미디어 생성
wandb.log({"accuracy": 0.9})
```

`wandb.init()`은 실행 객체를 반환하며, `wandb.run`을 통해서도 실행 객체에 액세스할 수 있습니다:

```python
import wandb

run = wandb.init()

assert run is wandb.run
```

스크립트의 끝에서 `wandb.finish`를 자동으로 호출하여 실행을 마무리하고 정리합니다. 그러나 자식 프로세스에서 `wandb.init`을 호출하는 경우 자식 프로세스의 끝에서 명시적으로 `wandb.finish`를 호출해야 합니다.

`wandb.init()` 사용에 대한 자세한 예제를 포함한 자세한 내용은 [가이드 및 FAQs](https://docs.wandb.ai/guides/track/launch)를 확인하세요.

| 인수 |  |
| :--- | :--- |
|  `project` |  (str, optional) 새 실행을 보내는 프로젝트의 이름입니다. 프로젝트가 지정되지 않은 경우 실행은 "Uncategorized" 프로젝트에 배치됩니다. |
|  `entity` |  (str, optional) 실행을 보내는 사용자 이름 또는 팀 이름입니다. 이 엔터티는 실행을 보내기 전에 존재해야 하므로, 실행 로깅을 시작하기 전에 UI에서 계정이나 팀을 만들어야 합니다. 엔터티를 지정하지 않으면 실행이 기본 엔터티(보통 사용자 이름)로 보내집니다. "새 프로젝트 생성 기본 위치" 아래 [설정](https://wandb.ai/settings)에서 기본 엔터티를 변경할 수 있습니다. |
|  `config` |  (dict, argparse, absl.flags, str, optional) 작업에 대한 입력, 예를 들어 모델의 하이퍼파라미터나 데이터 전처리 작업의 설정을 저장하는 `wandb.config`, 사전 유사 객체를 설정합니다. 설정은 UI에서 그룹, 필터 및 정렬할 수 있는 테이블에 표시됩니다. 키 이름에는 `.`이 포함되지 않아야 하며, 값은 10MB 이하여야 합니다. dict, argparse 또는 absl.flags인 경우: `wandb.config` 객체에 키 값 쌍을 로드합니다. str인 경우: 해당 이름의 yaml 파일을 찾아서 `wandb.config` 객체에 설정을 로드합니다. |
|  `save_code` |  (bool, optional) W&B에 메인 스크립트 또는 노트북을 저장하려면 이 옵션을 켭니다. 이는 실험의 재현성을 향상시키고 UI에서 실험 간 코드를 비교하는 데 유용합니다. 기본적으로 이 옵션은 꺼져 있지만, [설정 페이지](https://wandb.ai/settings)에서 기본 동작을 켜도록 설정할 수 있습니다. |
|  `group` |  (str, optional) 개별 실행을 더 큰 실험으로 구성하기 위해 그룹을 지정합니다. 예를 들어, 교차 검증을 수행하거나 다양한 테스트 세트에 대해 모델을 학습하고 평가하는 여러 작업이 있을 수 있습니다. 그룹을 사용하면 실행을 더 큰 전체로 함께 구성할 수 있으며, UI에서 이를 켜고 끌 수 있습니다. 자세한 내용은 [실행 그룹화 가이드](https://docs.wandb.com/guides/runs/grouping)를 참조하세요. |
|  `job_type` |  (str, optional) 실행 유형을 지정하는데, 그룹을 사용하여 더 큰 실험으로 실행을 함께 그룹화할 때 유용합니다. 예를 들어, train 및 eval과 같은 작업 유형을 가진 그룹의 여러 작업이 있을 수 있습니다. 이를 설정하면 UI에서 유사한 실행을 필터링하고 그룹화하여 비교하기 쉽게 만듭니다. |
|  `tags` |  (list, optional) 이 실행의 태그 목록을 채우는 문자열 목록입니다. 태그는 실행을 함께 구성하거나 "baseline" 또는 "production"과 같은 임시 레이블을 적용하는 데 유용합니다. UI에서 태그를 쉽게 추가하거나 제거하거나 특정 태그가 있는 실행만 필터링할 수 있습니다. 실행을 재개하는 경우 `wandb.init()`에 전달한 태그로 기존 태그가 덮어씌워집니다. 기존 태그를 덮어쓰지 않고 재개된 실행에 태그를 추가하려면 `wandb.init()` 후에 `run.tags += ["new_tag"]`를 사용하세요. |
|  `name` |  (str, optional) 이 실행에 대한 짧은 표시 이름으로, UI에서 이 실행을 식별하는 방법입니다. 기본적으로 차트와 테이블을 쉽게 상호 참조할 수 있도록 임의의 두 단어 이름을 생성합니다. 이러한 실행 이름을 짧게 유지하면 차트 범례와 테이블을 읽기 쉽게 만듭니다. 하이퍼파라미터를 저장할 곳을 찾고 있다면, 이를 설정에 저장하는 것이 좋습니다. |
|  `notes` |  (str, optional) 실행에 대한 더 긴 설명으로, git에서 `-m` 커밋 메시지와 비슷합니다. 이를 통해 이 실행을 수행했을 때 무엇을 하고 있었는지 기억하는 데 도움이 됩니다. |
|  `dir` |  (str 또는 pathlib.Path, optional) 메타데이터가 저장될 디렉터리의 절대 경로입니다. 아티팩트에서 `download()`를 호출할 때 다운로드된 파일이 저장될 디렉터리입니다. 기본적으로 이는 `./wandb` 디렉터리입니다. |
|  `resume` |  (bool, str, optional) 재개 동작을 설정합니다. 옵션: `"allow"`, `"must"`, `"never"`, `"auto"` 또는 `None`. 기본값은 `None`. 경우: - `None` (기본값): 새 실행이 이전 실행과 동일한 ID를 가진 경우, 이 실행은 해당 데이터를 덮어씁니다. - `"auto"` (또는 `True`): 이전 실행이 이 기기에서 충돌했다면, 자동으로 재개합니다. 그렇지 않으면 새 실행을 시작합니다. - `"allow"`: `init(id="UNIQUE_ID")` 또는 `WANDB_RUN_ID="UNIQUE_ID"`로 ID가 설정되고 이전 실행과 동일한 경우, wandb는 해당 ID의 실행을 자동으로 재개합니다. 그렇지 않으면 wandb는 새 실행을 시작합니다. - `"never"`: `init(id="UNIQUE_ID")` 또는 `WANDB_RUN_ID="UNIQUE_ID"`로 ID가 설정되고 이전 실행과 동일한 경우, wandb는 충돌합니다. - `"must"`: `init(id="UNIQUE_ID")` 또는 `WANDB_RUN_ID="UNIQUE_ID"`로 ID가 설정되고 이전 실행과 동일한 경우, wandb는 해당 ID의 실행을 자동으로 재개합니다. 그렇지 않으면 wandb는 충돌합니다. 자세한 내용은 [실행 재개 가이드](https://docs.wandb.com/guides/runs/resuming)를 참조하세요. |
|  `reinit` |  (bool, optional) 동일한 프로세스에서 여러 `wandb.init()` 호출을 허용합니다. (기본값: `False`) |
|  `magic` |  (bool, dict, 또는 str, optional) bool은 스크립트에 더 많은 wandb 코드를 추가하지 않고도 실행의 기본 세부 정보를 자동으로 캡처하는 자동 계측을 시도할지 여부를 제어합니다. (기본값: `False`) 또한 dict, json 문자열, 또는 yaml 파일 이름을 전달할 수 있습니다. |
|  `config_exclude_keys` |  (list, optional) `wandb.config`에서 제외할 문자열 키입니다. |
|  `config_include_keys` |  (list, optional) `wandb.config`에 포함할 문자열 키입니다. |
|  `anonymous` |  (str, optional) 익명 데이터 로깅을 제어합니다. 옵션: - `"never"` (기본값): 실행을 추적하기 전에 W&B 계정을 연결해야 하므로 실수로 익명 실행을 생성하지 않습니다. - `"allow"`: 로그인한 사용자가 계정으로 실행을 추적할 수 있지만, W&B 계정이 없는 사용자가 스크립트를 실행할 때 UI에서 차트를 볼 수 있습니다. - `"must"`: 실행을 가입된 사용자 계정 대신 익명 계정으로 보냅니다. |
|  `mode` |  (str, optional) `"online"`, `"offline"` 또는 `"disabled"`일 수 있습니다. 기본값은 온라인입니다. |
|  `allow_val_change` |  (bool, optional) 키를 한 번 설정한 후 설정 값이 변경되는 것을 허용할지 여부입니다. 기본적으로 설정 값을 덮어쓰면 예외를 던집니다. 학습 중에 여러 번 변하는 학습률과 같은 것을 추적하려면 `wandb.log()`를 사용하세요. (스크립트에서 기본값: `False`, Jupyter에서는 `True`) |
|  `force` |  (bool, optional) `True`인 경우, 사용자가 W&B에 로그인하지 않았다면 스크립트를 충돌시킵니다. `False`인 경우, 사용자가 W&B에 로그인하지 않았다면 스크립트를 오프라인 모드로 실행할 수 있습니다. (기본값: `False`) |
|  `sync_tensorboard` |  (bool, optional) tensorboard 또는 tensorboardX에서 wandb 로그를 동기화하고 관련 이벤트 파일을 저장합니다. (기본값: `False`) |
|  `monitor_gym` |  (bool, optional) OpenAI Gym을 사용할 때 환경의 비디오를 자동으로 로그합니다. (기본값: `False`) [이 통합 가이드](https://docs.wandb.com/guides/integrations/openai-gym)를 참조하세요. |
|  `id` |  (str, optional) 이 실행에 대한 고유 ID로, 재개에 사용됩니다. 프로젝트에서 고유해야 하며, 실행을 삭제한 경우 ID를 재사용할 수 없습니다. 짧은 설명 이름에는 `name` 필드를 사용하거나 실행 간 하이퍼파라미터를 비교하기 위해 저장하려는 경우 `config`를 사용하세요. ID에는 다음 특수 문자를 포함할 수 없습니다: `/\#?%:`. [실행 재개 가이드](https://docs.wandb.com/guides/runs/resuming)를 참조하세요. |

#### 예시:

### 실행이 로그되는 위치 설정

git에서 조직, 저장소 및 브랜치를 변경하는 것처럼 실행이 로그되는 위치를 변경할 수 있습니다:

```python
import wandb

user = "geoff"
project = "capsules"
display_name = "experiment-2021-10-31"

wandb.init(entity=user, project=project, name=display_name)
```

### 실행에 대한 메타데이터를 설정에 추가

`config` 키워드 인수로 딕셔너리 스타일 객체를 전달하여 실행에 메타데이터, 예를 들어 하이퍼파라미터를 추가하세요.

```python
import wandb

config = {"lr": 3e-4, "batch_size": 32}
config.update({"architecture": "resnet", "depth": 34})
wandb.init(config=config)
```

| 예외 |  |
| :--- | :--- |
|  `Error` |  실행 초기화 중에 알 수 없거나 내부적인 오류가 발생한 경우. |
|  `AuthenticationError` |  사용자가 유효한 자격 증명을 제공하지 못한 경우. |
|  `CommError` |  WandB 서버와의 통신에 문제가 있는 경우. |
|  `UsageError` |  사용자가 유효하지 않은 인수를 제공한 경우. |
|  `KeyboardInterrupt` |  사용자가 실행을 중단한 경우. |

| 반환 |  |
| :--- | :--- |
|  `Run`