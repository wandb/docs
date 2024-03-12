
# init

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/wandb_init.py#L940-L1215' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>


W&B에 추적 및 로그를 기록할 새로운 run을 시작합니다.

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

ML 트레이닝 파이프라인에서는 `wandb.init()`을 트레이닝 스크립트와 평가 스크립트의 시작 부분에 추가할 수 있으며, 각 부분이 W&B에서 run으로 추적됩니다.

`wandb.init()`은 run에 데이터를 로그하기 위해 새로운 백그라운드 프로세스를 생성하며, 기본적으로 wandb.ai에 데이터를 동기화하여 실시간 시각화를 볼 수 있습니다.

`wandb.log()`를 사용하여 데이터를 로깅하기 전에 `wandb.init()`를 호출하여 run을 시작하세요:




```python
import wandb

wandb.init()
# ... 메트릭 계산, 미디어 생성
wandb.log({"accuracy": 0.9})
```

`wandb.init()`은 run 오브젝트를 반환하며, `wandb.run`을 통해서도 run 오브젝트에 접근할 수 있습니다:




```python
import wandb

run = wandb.init()

assert run is wandb.run
```

스크립트의 끝에서는 자동으로 `wandb.finish`를 호출하여 run을 마무리하고 정리합니다. 그러나, 자식 프로세스에서 `wandb.init`를 호출하는 경우 자식 프로세스의 끝에서 명시적으로 `wandb.finish`를 호출해야 합니다.

`wandb.init()` 사용에 대한 자세한 예시는 [가이드와 FAQ](https://docs.wandb.ai/guides/track/launch)를 확인하세요.

| 인수 |  |
| :--- | :--- |
|  `project` |  (str, optional) 새로운 run을 보내는 프로젝트의 이름입니다. 프로젝트가 지정되지 않은 경우 run은 "Uncategorized" 프로젝트에 배치됩니다. |
|  `entity` |  (str, optional) run을 보내는 사용자 이름 또는 팀 이름입니다. 이 entity는 run을 보내기 전에 존재해야 하므로, 로그를 시작하기 전에 UI에서 계정이나 팀을 생성하세요. entity를 지정하지 않으면, 기본적으로 사용자 이름인 기본 entity에 run이 전송됩니다. "새 프로젝트 생성 기본 위치" 아래 [설정](https://wandb.ai/settings)에서 기본 entity를 변경할 수 있습니다. |
|  `config` |  (dict, argparse, absl.flags, str, optional) `wandb.config`를 설정합니다. 이는 모델의 하이퍼파라미터나 데이터 전처리 작업 설정과 같이 작업에 대한 입력을 저장하는 dictionary와 같은 오브젝트입니다. config는 UI에서 그룹화, 필터링, 정렬할 수 있는 테이블에 표시됩니다. 키 이름에는 `.`이 포함되지 않아야 하며, 값은 10MB 이하이어야 합니다. dict, argparse 또는 absl.flags의 경우: `wandb.config` 오브젝트에 키 값 쌍을 로드합니다. str의 경우: 해당 이름의 yaml 파일을 찾아 `wandb.config` 오브젝트에 파일에서 config를 로드합니다. |
|  `save_code` |  (bool, optional) W&B에 메인 스크립트나 노트북을 저장하려면 이를 활성화합니다. 이는 실험 재현성을 향상시키고 UI에서 실험간 코드를 비교하는 데 유용합니다. 기본적으로 이 기능은 꺼져 있지만, [설정 페이지](https://wandb.ai/settings)에서 기본 동작을 켜도록 설정할 수 있습니다. |
|  `group` |  (str, optional) 개별 run을 더 큰 실험으로 조직화하는 데 사용할 그룹을 지정합니다. 예를 들어, 교차검증을 수행하거나 다른 테스트 세트에 대해 모델을 트레이닝하고 평가하는 여러 작업이 있을 수 있습니다. Group은 run을 더 큰 전체로 함께 조직화할 수 있는 방법을 제공하며, UI에서 이를 켜고 끌 수 있습니다. 자세한 내용은 [run 그룹화 가이드](https://docs.wandb.com/guides/runs/grouping)를 확인하세요. |
|  `job_type` |  (str, optional) run의 유형을 지정합니다. 이는 group을 사용하여 더 큰 실험으로 여러 run을 그룹화할 때 유용합니다. 예를 들어, train 및 eval과 같은 작업 유형을 가진 그룹에서 여러 작업이 있을 수 있습니다. 이를 설정하면 UI에서 유사한 run을 쉽게 필터링하고 그룹화하여 비교할 수 있습니다. |
|  `tags` |  (list, optional) 이 run의 태그 목록에 채워질 문자열의 목록입니다. 태그는 run을 함께 조직화하거나 "베이스라인" 또는 "프로덕션"과 같은 임시 레이블을 적용하는 데 유용합니다. UI에서 태그를 쉽게 추가하거나 제거하거나 특정 태그가 있는 run만 필터링할 수 있습니다. run을 재개하는 경우 `wandb.init()`에 전달한 태그로 기존 태그가 덮어쓰여집니다. 기존 태그를 덮어쓰지 않고 재개된 run에 태그를 추가하려면, `wandb.init()` 후에 `run.tags += ["new_tag"]`를 사용하세요. |
|  `name` |  (str, optional) 이 run에 대한 짧은 표시 이름으로, UI에서 이 run을 식별하는 방법입니다. 기본적으로 차트와 테이블을 쉽게 참조할 수 있도록 무작위로 생성된 두 단어의 이름을 생성합니다. 이러한 run 이름을 짧게 유지하면 차트 범례와 테이블을 읽기 쉽습니다. 하이퍼파라미터를 저장할 위치를 찾고 있다면, config에 저장하는 것이 좋습니다. |
|  `notes` |  (str, optional) run에 대한 더 긴 설명으로, git에서 `-m` 커밋 메시지와 같습니다. 이를 통해 이 run을 실행했을 때 무엇을 하고 있었는지 기억하는 데 도움이 됩니다. |
|  `dir` |  (str 또는 pathlib.Path, optional) 메타데이터가 저장될 디렉토리의 절대 경로입니다. 아티팩트에서 `download()`를 호출할 때 다운로드된 파일이 저장될 디렉토리입니다. 기본적으로 이는 `./wandb` 디렉토리입니다. |
|  `resume` |  (bool, str, optional) 재개 동작을 설정합니다. 옵션: `"allow"`, `"must"`, `"never"`, `"auto"` 또는 `None`. 기본값은 `None`입니다. 경우: - `None` (기본값): 새 run이 이전 run과 동일한 ID를 가지고 있다면, 이 run은 해당 데이터를 덮어씁니다. - `"auto"` (또는 `True`): 이전 run이 이 컴퓨터에서 충돌했다면, 자동으로 재개합니다. 그렇지 않으면 새로운 run을 시작합니다. - `"allow"`: `init(id="UNIQUE_ID")` 또는 `WANDB_RUN_ID="UNIQUE_ID"`로 id가 설정되어 있고 이전 run과 동일하다면, wandb는 해당 id의 run을 자동으로 재개합니다. 그렇지 않으면 wandb는 새로운 run을 시작합니다. - `"never"`: `init(id="UNIQUE_ID")` 또는 `WANDB_RUN_ID="UNIQUE_ID"`로 id가 설정되어 있고 이전 run과 동일하다면, wandb는 충돌합니다. - `"must"`: `init(id="UNIQUE_ID")` 또는 `WANDB_RUN_ID="UNIQUE_ID"`로 id가 설정되어 있고 이전 run과 동일하다면, wandb는 id의 run을 자동으로 재개합니다. 그렇지 않으면 wandb는 충돌합니다. [run 재개 가이드](https://docs.wandb.com/guides/runs/resuming)를 참조하세요. |
|  `reinit` |  (bool, optional) 같은 프로세스에서 여러 `wandb.init()` 호출을 허용합니다. (기본값: `False`) |
|  `magic` |  (bool, dict, 또는 str, optional) bool은 스크립트를 자동으로 계측하여 추가적인 wandb 코드 없이도 실행의 기본 세부 사항을 캡처할지 여부를 제어합니다. (기본값: `False`) dict, json 문자열 또는 yaml 파일 이름을 전달할 수도 있습니다. |
|  `config_exclude_keys` |  (list, optional) `wandb.config`에서 제외할 문자열 키입니다. |
|  `config_include_keys` |  (list, optional) `wandb.config`에 포함할 문자열 키입니다. |
|  `anonymous` |  (str, optional) 익명 데이터 로깅을 제어합니다. 옵션: - `"never"` (기본값): W&B 계정을 연결하기 전에 run을 추적하도록 요구하여 실수로 익명 run을 생성하지 않게 합니다. - `"allow"`: 로그인한 사용자가 계정으로 run을 추적할 수 있게 하지만, W&B 계정이 없는 사용자가 스크립트를 실행할 때 UI에서 차트를 볼 수 있게 합니다. - `"must"`: run을 가입한 사용자 계정 대신 익명 계정으로 보냅니다. |
|  `mode` |  (str, optional) `"online"`, `"offline"` 또는 `"disabled"`일 수 있습니다. 기본값은 온라인입니다. |
|  `allow_val_change` |  (bool, optional) 키를 한 번 설정한 후 config 값이 변경되는 것을 허용할지 여부입니다. 기본적으로 config 값이 덮어쓰여지면 예외를 발생시킵니다. 트레이닝 중 여러 번에 걸쳐 변하는 학습률과 같은 것을 추적하려면 대신 `wandb.log()`를 사용하세요. (기본값: 스크립트에서는 `False`, Jupyter에서는 `True`) |
|  `force` |  (bool, optional) `True`인 경우, 사용자가 W&B에 로그인하지 않은 경우 스크립트를 충돌시킵니다. `False`인 경우, 사용자가 W&B에 로그인하지 않은 경우 스크립트를 오프라인 모드로 실행할 수 있습니다. (기본값: `False`) |
|  `sync_tensorboard` |  (bool, optional) tensorboard 또는 tensorboardX에서 wandb 로그를 동기화하고 관련 이벤트 파일을 저장합니다. (기본값: `False`) |
|  `monitor_gym` |  (bool, optional) OpenAI Gym을 사용할 때 환경의 비디오를 자동으로 로깅합니다. (기본값: `False`) [이 인테그레이션 가이드](https://docs.wandb.com/guides/integrations/openai-gym)를 참조하세요. |
|  `id` |  (str, optional) 이 run의 고유 ID로, 재개에 사용됩니다. 프로젝트에서 고유해야 하며, run을 삭제한 경우 ID를 재사용할 수 없습니다. 짧은 설명 이름을 위해 `name` 필드를 사용하거나, run 간에 비교할 하이퍼파라미터를 저장하기 위해 `config`를 사용하세요. ID에는 다음 특수 문자를 포함할 수 없습니다: `/\#?%:`. [run 재개 가이드](https://docs.wandb.com/guides/runs/resuming)를 참조하세요. |

#### 예시:

### run이 기록되는 위치 설정

git의 조직, 저장소, 브랜치를 변경하는 것처럼 run이 기록되는 위치를 변경할 수 있습니다:

```python
import wandb

user = "geoff"
project = "capsules"
display_name = "experiment-2021-10-31"

wandb.init(entity=user, project=project, name=display_name)
```

### run에 대한 메타데이터를 config에 추가

`config` 키워드 인수로 dictionary 스타일 오브젝트를 전달하여 run에 메타데이터, 예를 들어 하이퍼파라미터를 추가하세요.




```python
import wandb

config = {"lr": 3e-4, "batch_size": 32}
config.update({"architecture": "resnet", "depth": 34})
wandb.init(config=config)
```

| 발생 |  |
| :--- | :--- |
|  `Error` |  run 초기화 중에 알 수 없거나 내부적인 오류가 발생한 경우. |
|  `AuthenticationError` |  사용자가 유효한 자격증명을 제공하지 못한 경우. |
|  `CommError` |  WandB 서버와 통신하는 데 문제가 있는 경우. |
|  `UsageError` |  사용자가 잘못된 인수를 제공한 경우. |
|  `KeyboardInterrupt` |