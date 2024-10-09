# init

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_init.py#L979-L1249' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

새로운 run을 시작하여 W&B에 추적하고 기록합니다.

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
    fork_from: Optional[str] = None,
    resume_from: Optional[str] = None,
    settings: Union[Settings, Dict[str, Any], None] = None
) -> Run
```

ML 트레이닝 파이프라인에서, `wandb.init()`을 트레이닝 스크립트의 시작 부분과 평가 스크립트에 추가할 수 있으며, 각 부분은 W&B에서 run으로 추적됩니다.

`wandb.init()`은 데이터를 run에 로그하고, wandb.ai에 데이터를 기본적으로 동기화하는 새로운 백그라운드 프로세스를 시작하여 실시간 시각화를 볼 수 있습니다.

`wandb.log()`로 데이터를 기록하기 전에 `wandb.init()`을 호출하여 run을 시작하세요:

```python
import wandb

wandb.init()
# ... 메트릭 계산, 미디어 생성
wandb.log({"accuracy": 0.9})
```

`wandb.init()`은 run 오브젝트를 반환하며, `wandb.run`을 통해 run 오브젝트에 엑세스할 수도 있습니다:

```python
import wandb

run = wandb.init()

assert run is wandb.run
```

스크립트의 끝에서, 우리는 자동으로 `wandb.finish`를 호출하여 run을 완료하고 정리합니다. 그러나 자식 프로세스에서 `wandb.init`을 호출한 경우, 자식 프로세스의 끝에서 명시적으로 `wandb.finish`를 호출해야 합니다.

`wandb.init()` 사용에 대한 자세한 내용은 우리의 [가이드와 FAQ](/guides/track/launch)를 참고하세요.

| Arguments |  |
| :--- | :--- |
|  `project` |  (str, optional) 새로운 run을 보낼 프로젝트의 이름입니다. 프로젝트가 지정되지 않은 경우, 우리는 git 루트 또는 현재 프로그램 파일에서 프로젝트 이름을 추론하려고 시도할 것입니다. 프로젝트 이름을 추론할 수 없으면 `"uncategorized"`로 기본값이 설정됩니다. |
|  `entity` |  (str, optional) run을 보내는 사용자 이름이나 팀 이름입니다. 이 엔터티는 run을 보내기 전에 존재해야 하므로, 계정이나 팀을 UI에서 생성하세요. 엔터티를 지정하지 않으면 run은 기본 엔터티로 보내집니다. 새로운 프로젝트를 생성할 기본 위치는 [설정](https://wandb.ai/settings)에서 변경할 수 있습니다. |
|  `config` |  (dict, argparse, absl.flags, str, optional) `wandb.config`를 설정하는 사전과 유사한 오브젝트로, 모델의 하이퍼파라미터나 데이터 처리 job의 설정 같은 job에 대한 인수를 저장합니다. 이 config는 UI의 표에 표시되며, 그룹핑, 필터링 및 정렬에 사용할 수 있습니다. 키 이름에 `.`을 포함해서는 안되며, 값은 10MB 이하여야 합니다. dict, argparse 또는 absl.flags인 경우: 키-값 쌍이 `wandb.config` 오브젝트로 로드됩니다. str인 경우: 동일한 이름의 yaml 파일을 찾고, 해당 파일에서 config를 `wandb.config` 오브젝트로 로드합니다. |
|  `save_code` |  (bool, optional) 주 스크립트나 노트북을 W&B에 저장하려면 이 옵션을 켭니다. 이것은 실험 재현성을 향상시키고 UI에서 실험 간 코드 차이를 확인하는 데 유용합니다. 기본적으로 꺼져 있지만, [설정 페이지](https://wandb.ai/settings)에서 기본 동작을 켜기로 변경할 수 있습니다. |
|  `group` |  (str, optional) 개별 run을 더 큰 실험으로 조직하기 위해 그룹을 지정합니다. 예를 들어, 교차검증을 하고 있거나 서로 다른 테스트 세트에 대해 모델을 학습시키고 평가하는 여러 job이 있을 수 있습니다. 그룹 기능은 run을 함께 더 큰 전체로 조직하는 방법을 제공하며, UI에서 켜고 끌 수 있습니다. 자세한 내용은 [run 그룹화에 대한 가이드](/guides/runs/grouping)를 참조하세요. |
|  `job_type` |  (str, optional) 그룹을 사용하여 더 큰 실험으로 run을 함께 그룹화할 때 유용한 run의 유형을 지정합니다. 예를 들어, 그룹에 여러 job이 있는 경우, train 및 eval과 같은 job 유형이 있을 수 있습니다. 이를 설정하면 UI에서 유사한 run을 쉽게 필터링하고 그룹화하여 비교할 수 있습니다. |
|  `tags` |  (list, optional) 이 run에 대한 태그 목록을 UI의 태그 목록에 채웁니다. 태그는 run을 함께 조직하거나 "baseline" 또는 "production"과 같은 임시 레이블을 적용하는 데 유용합니다. UI에서 쉽게 태그를 추가하고 제거하거나 특정 태그를 가진 run으로만 필터링할 수 있습니다. run을 재개하는 경우, 해당 run의 태그는 `wandb.init()`에 전달된 태그로 덮어쓰여집니다. 기존 태그를 덮어쓰지 않고 재개된 run에 태그를 추가하려면 `wandb.init()` 후에 `run.tags += ["new_tag"]`를 사용하세요. |
|  `name` |  (str, optional) 이 run의 짧은 표시 이름으로, UI에서 이 run을 식별하는 방법입니다. 기본적으로, 우리는 표와 차트를 쉽게 참조할 수 있도록 무작위 두 단어 이름을 생성합니다. run 이름을 짧게 유지하면 차트 범례와 표를 읽기 쉽게 만듭니다. 하이퍼파라미터를 저장할 곳을 찾고 있다면 config에 저장하는 것을 권장합니다. |
|  `notes` |  (str, optional) run에 대한 더 긴 설명으로, git의 `-m` 커밋 메시지와 유사합니다. 이 run을 실행할 때 무엇을 하고 있었는지 기억하는 데 도움이 됩니다. |
|  `dir` |  (str or pathlib.Path, optional) 메타데이터가 저장될 디렉토리의 절대 경로입니다. 아티팩트를 `download()`할 때, 이 디렉토리가 다운로드된 파일이 저장될 위치입니다. 기본적으로, 이 디렉토리는 `./wandb` 입니다. |
|  `resume` |  (bool, str, optional) 재개 행동을 설정합니다. 옵션: `"allow"`, `"must"`, `"never"`, `"auto"` 또는 `None`. 기본값은 `None`입니다. 사례: - `None` (기본값): 새 run이 이전 run과 동일한 ID를 가지면, 이 run이 해당 데이터를 덮어씁니다. - `"auto"` (또는 `True`): 이 머신에서 이전 run이 충돌한 경우 자동으로 재개됩니다. 그렇지 않으면 새 run을 시작합니다. - `"allow"`: `init(id="UNIQUE_ID")` 또는 `WANDB_RUN_ID="UNIQUE_ID"`와 함께 id가 설정되고 이전 run과 동일한 경우, wandb는 해당 id로 run을 자동으로 재개합니다. 그렇지 않으면, wandb는 새 run을 시작합니다. - `"never"`: `init(id="UNIQUE_ID")` 또는 `WANDB_RUN_ID="UNIQUE_ID"`와 함께 id가 설정되고 이전 run과 동일한 경우, wandb는 오류를 발생시킵니다. - `"must"`: `init(id="UNIQUE_ID")` 또는 `WANDB_RUN_ID="UNIQUE_ID"`와 함께 id가 설정되고 이전 run과 동일한 경우, wandb는 해당 id로 run을 자동으로 재개합니다. 그렇지 않으면 wandb는 오류를 발생시킵니다. 자세한 내용은 [run 재개 가이드](/guides/runs/resuming)를 참조하세요. |
|  `reinit` |  (bool, optional) 동일한 프로세스에서 여러 `wandb.init()` 호출을 허용합니다. (기본값: `False`) |
|  `magic` |  (bool, dict, or str, optional) 스크립트를 자동으로 계측하여 추가 wandb 코드 없이 run의 기본 세부 정보를 캡처하려고 할지 여부를 제어합니다. (기본값: `False`) 사전, json 문자열 또는 yaml 파일 이름을 전달할 수도 있습니다. |
|  `config_exclude_keys` |  (list, optional) `wandb.config`에서 제외할 키 문자열입니다. |
|  `config_include_keys` |  (list, optional) `wandb.config`에서 포함할 키 문자열입니다. |
|  `anonymous` |  (str, optional) 익명 데이터 로깅을 제어합니다. 옵션: - `"never"` (기본값): run을 추적하기 전에 W&B 계정과 연결해야 합니다. 그러지 않으면 익명 run이 생성될 수 있습니다. - `"allow"`: 로그인 한 사용자가 자신의 계정으로 run을 추적할 수 있지만, W&B 계정 없이 스크립트를 실행하는 사람이 UI에서 차트를 볼 수 있습니다. - `"must"`: run을 로그인 한 사용자 계정 대신 익명 계정으로 보냅니다. |
|  `mode` |  (str, optional) `"online"`, `"offline"` 또는 `"disabled"`로 설정할 수 있습니다. 기본값은 online입니다. |
|  `allow_val_change` |  (bool, optional) 키를 한 번 설정한 후 config 값의 변경을 허용할지 여부입니다. 기본적으로 config 값이 덮어쓰이면 예외를 발생시킵니다. 트레이닝 중 여러 시간에 걸쳐 변화하는 학습률과 같은 것을 추적하려면 대신 `wandb.log()`를 사용하세요. (기본값: 스크립트에서는 `False`, Jupyter에서는 `True`) |
|  `force` |  (bool, optional) `True`인 경우, 사용자가 W&B에 로그인하지 않으면 스크립트가 충돌합니다. `False`인 경우, 사용자가 W&B에 로그인하지 않으면 스크립트를 오프라인 모드로 실행합니다. (기본값: `False`) |
|  `sync_tensorboard` |  (bool, optional) tensorboard 또는 tensorboardX에서 wandb 로그를 동기화하고 관련 이벤트 파일을 저장합니다. (기본값: `False`) |
|  `tensorboard` |  (bool, optional) `sync_tensorboard`의 에일리어스이며, 폐지되었습니다. |
|  `monitor_gym` |  (bool, optional) OpenAI Gym을 사용할 때 환경의 비디오를 자동으로 기록합니다. (기본값: `False`) 이 인테그레이션에 대한 [가이드](/guides/integrations/openai-gym)를 참조하세요. |
|  `id` |  (str, optional) 이 run의 고유한 ID로, 재개에 사용됩니다. 프로젝트에서 고유해야 하며, run을 삭제하면 ID를 재사용할 수 없습니다. `name` 필드를 사용하여 짧은 설명적 이름을 지정하거나, `config`를 사용하여 across run을 비교할 하이퍼파라미터를 저장하세요. ID는 다음 특수 문자를 포함할 수 없습니다: `/\#?%:`. [run 재개 가이드](/guides/runs/resuming)를 참조하세요. |
|  `fork_from` |  (str, optional) 이전 run의 특정 순간에서 새로운 run을 포크하려는 `{run_id}?_step={step}` 형식의 문자열입니다. 지정된 run에서 지정된 순간에 로깅 기록을 가져와 새로운 run을 생성합니다. 대상 run은 현재 프로젝트에 있어야 합니다. 예: `fork_from="my-run-id?_step=1234"`. |
|  `resume_from` |  (str, optional) 이전 run의 특정 순간에서 run을 재개하려는 `{run_id}?_step={step}` 형식의 문자열입니다. 이는 중간 단계에서 run에 기록된 기록을 잘라내고 해당 단계에서 로깅을 재개할 수 있게 합니다. 이는 내부적으로 run 포크를 사용합니다. 대상 run은 현재 프로젝트에 있어야 합니다. 예: `resume_from="my-run-id?_step=1234"`. |
|  `settings` | `(dict, wandb.Settings, optional)` 이 run에서 사용할 설정입니다. (기본값: None) |

#### 예시:

### run이 기록될 위치 설정

git의 조직, 저장소 및 분기를 변경하는 것처럼 run이 기록될 위치를 변경할 수 있습니다:

```python
import wandb

user = "geoff"
project = "capsules"
display_name = "experiment-2021-10-31"

wandb.init(entity=user, project=project, name=display_name)
```

### run에 대한 메타데이터를 config에 추가

하이퍼파라미터와 같은 메타데이터를 run에 추가하려면 `config` 키워드 인수로 사전 스타일 오브젝트를 전달합니다.

```python
import wandb

config = {"lr": 3e-4, "batch_size": 32}
config.update({"architecture": "resnet", "depth": 34})
wandb.init(config=config)
```

| Raises |  |
| :--- | :--- |
|  `Error` |  run 초기화 중 알 수 없는 또는 내부 오류가 발생한 경우. |
|  `AuthenticationError` |  사용자가 유효한 자격 증명을 제공하지 못한 경우. |
|  `CommError` |  WandB 서버와의 통신에 문제가 발생한 경우. |
|  `UsageError` |  사용자가 잘못된 인수를 제공한 경우. |
|  `KeyboardInterrupt` |  사용자가 run을 중단한 경우. |

| Returns |  |
| :--- | :--- |
|  `Run` 오브젝트를 반환합니다. |