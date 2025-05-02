---
title: init
menu:
  reference:
    identifier: ko-ref-python-init
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_init.py#L1131-L1483 >}}

새로운 run을 시작하여 W&B에 추적하고 기록합니다.

```python
init(
    entity: (str | None) = None,
    project: (str | None) = None,
    dir: (StrPath | None) = None,
    id: (str | None) = None,
    name: (str | None) = None,
    notes: (str | None) = None,
    tags: (Sequence[str] | None) = None,
    config: (dict[str, Any] | str | None) = None,
    config_exclude_keys: (list[str] | None) = None,
    config_include_keys: (list[str] | None) = None,
    allow_val_change: (bool | None) = None,
    group: (str | None) = None,
    job_type: (str | None) = None,
    mode: (Literal['online', 'offline', 'disabled'] | None) = None,
    force: (bool | None) = None,
    anonymous: (Literal['never', 'allow', 'must'] | None) = None,
    reinit: (bool | None) = None,
    resume: (bool | Literal['allow', 'never', 'must', 'auto'] | None) = None,
    resume_from: (str | None) = None,
    fork_from: (str | None) = None,
    save_code: (bool | None) = None,
    tensorboard: (bool | None) = None,
    sync_tensorboard: (bool | None) = None,
    monitor_gym: (bool | None) = None,
    settings: (Settings | dict[str, Any] | None) = None
) -> Run
```

ML 트레이닝 파이프라인에서 트레이닝 스크립트의 시작 부분과 평가 스크립트에 `wandb.init()`을 추가할 수 있으며, 각 부분은 W&B에서 run으로 추적됩니다.

`wandb.init()`은 run에 데이터를 기록하기 위해 새로운 백그라운드 프로세스를 생성하며 기본적으로 데이터를 https://wandb.ai 에 동기화하므로 결과를 실시간으로 확인할 수 있습니다.

`wandb.log()`로 데이터를 기록하기 전에 `wandb.init()`을 호출하여 run을 시작합니다. 데이터 로깅이 완료되면 `wandb.finish()`를 호출하여 run을 종료합니다. `wandb.finish()`를 호출하지 않으면 스크립트가 종료될 때 run이 종료됩니다.

자세한 예제를 포함하여 `wandb.init()` 사용에 대한 자세한 내용은 [가이드 및 FAQ](https://docs.wandb.ai/guides/track/launch)를 확인하십시오.

#### 예시:

### entity 및 project를 명시적으로 설정하고 run 이름을 선택합니다.

```python
import wandb

run = wandb.init(
    entity="geoff",
    project="capsules",
    name="experiment-2021-10-31",
)

# ... 여기에 트레이닝 코드 작성 ...

run.finish()
```

### `config` 인수를 사용하여 run에 대한 메타데이터를 추가합니다.

```python
import wandb

config = {"lr": 0.01, "batch_size": 32}
with wandb.init(config=config) as run:
    run.config.update({"architecture": "resnet", "depth": 34})

    # ... 여기에 트레이닝 코드 작성 ...
```

`wandb.init()`을 컨텍스트 관리자로 사용하여 블록 끝에서 자동으로 `wandb.finish()`를 호출할 수 있습니다.

| 인수 |  |
| :--- | :--- |
|  `entity` |  run이 기록될 사용자 이름 또는 팀 이름입니다. entity는 이미 존재해야 하므로 run 로깅을 시작하기 전에 UI에서 계정 또는 팀을 생성했는지 확인하십시오. 지정하지 않으면 run은 기본 entity로 기본 설정됩니다. 기본 entity를 변경하려면 [설정](https://wandb.ai/settings)으로 이동하여 "기본 팀"에서 "새 프로젝트를 만들 기본 위치"를 업데이트하십시오. |
|  `project` |  이 run이 기록될 project의 이름입니다. 지정하지 않으면 git 루트 또는 현재 프로그램 파일 확인과 같이 시스템을 기반으로 프로젝트 이름을 추론하기 위한 경험적 방법을 사용합니다. 프로젝트 이름을 추론할 수 없으면 프로젝트는 기본적으로 `"uncategorized"`로 설정됩니다. |
|  `dir` |  실험 로그 및 메타데이터 파일이 저장되는 디렉토리의 절대 경로입니다. 지정하지 않으면 기본적으로 `./wandb` 디렉토리가 됩니다. 이는 `download()`를 호출할 때 Artifacts가 저장되는 위치에 영향을 미치지 않습니다. |
|  `id` |  다시 시작하는 데 사용되는 이 run에 대한 고유 식별자입니다. project 내에서 고유해야 하며 run이 삭제되면 다시 사용할 수 없습니다. 식별자는 다음 특수 문자 `/ \ # ? % :`를 포함해서는 안 됩니다. 짧은 설명 이름은 `name` 필드를 사용하고, run 간에 비교할 하이퍼파라미터를 저장하려면 `config`를 사용합니다. |
|  `name` |  UI에서 식별하는 데 도움이 되는 이 run의 짧은 표시 이름입니다. 기본적으로 테이블에서 차트로의 쉬운 상호 참조 run을 허용하는 임의의 두 단어 이름을 생성합니다. 이러한 run 이름을 간결하게 유지하면 차트 범례 및 테이블에서 가독성이 향상됩니다. 하이퍼파라미터를 저장하려면 `config` 필드를 사용하는 것이 좋습니다. |
|  `notes` |  Git의 커밋 메시지와 유사한 run에 대한 자세한 설명입니다. 이 인수를 사용하여 나중에 이 run의 목적이나 설정을 기억하는 데 도움이 될 수 있는 컨텍스트 또는 세부 정보를 캡처하십시오. |
|  `tags` |  UI에서 이 run에 레이블을 지정하는 태그 목록입니다. 태그는 run을 구성하거나 "베이스라인" 또는 "프로덕션"과 같은 임시 식별자를 추가하는 데 유용합니다. UI에서 태그를 쉽게 추가, 제거 또는 필터링할 수 있습니다. run을 다시 시작하는 경우 여기에 제공된 태그는 기존 태그를 대체합니다. 현재 태그를 덮어쓰지 않고 다시 시작된 run에 태그를 추가하려면 `run = wandb.init()`을 호출한 후 `run.tags += ["new_tag"]`를 사용하십시오. |
|  `config` |  모델 하이퍼파라미터 또는 데이터 전처리 설정과 같이 run에 대한 입력 파라미터를 저장하기 위한 딕셔너리와 유사한 오브젝트인 `wandb.config`를 설정합니다. config는 UI의 개요 페이지에 나타나 이러한 파라미터를 기반으로 run을 그룹화, 필터링 및 정렬할 수 있습니다. 키는 마침표(`.`)를 포함해서는 안 되며 값은 10MB보다 작아야 합니다. 딕셔너리, `argparse.Namespace` 또는 `absl.flags.FLAGS`가 제공되면 키-값 쌍이 `wandb.config`에 직접 로드됩니다. 문자열이 제공되면 구성 값이 `wandb.config`에 로드될 YAML 파일의 경로로 해석됩니다. |
|  `config_exclude_keys` |  `wandb.config`에서 제외할 특정 키 목록입니다. |
|  `config_include_keys` |  `wandb.config`에 포함할 특정 키 목록입니다. |
|  `allow_val_change` |  config 값을 처음 설정한 후 수정할 수 있는지 여부를 제어합니다. 기본적으로 config 값이 덮어쓰여지면 예외가 발생합니다. 학습률과 같이 트레이닝 중에 변경되는 변수를 추적하려면 `wandb.log()`를 대신 사용하는 것이 좋습니다. 기본적으로 스크립트에서는 `False`이고 노트북 환경에서는 `True`입니다. |
|  `group` |  더 큰 실험의 일부로 개별 run을 구성하기 위한 그룹 이름을 지정합니다. 이는 교차 검증 또는 서로 다른 테스트 세트에서 모델을 트레이닝하고 평가하는 여러 job을 실행하는 경우에 유용합니다. 그룹화를 통해 UI에서 관련 run을 집합적으로 관리할 수 있으므로 통합된 실험으로 결과를 쉽게 전환하고 검토할 수 있습니다. 자세한 내용은 [run 그룹화 가이드](https://docs.wandb.com/guides/runs/grouping)를 참조하십시오. |
|  `job_type` |  더 큰 실험의 일부로 그룹 내에서 run을 구성할 때 특히 유용한 run 유형을 지정합니다. 예를 들어 그룹에서 "train" 및 "eval"과 같은 job 유형으로 run에 레이블을 지정할 수 있습니다. job 유형을 정의하면 UI에서 유사한 run을 쉽게 필터링하고 그룹화하여 직접 비교할 수 있습니다. |
|  `mode` |  run 데이터가 관리되는 방식을 지정합니다. 다음 옵션을 사용할 수 있습니다. - `"online"` (기본값): 네트워크 연결이 가능할 때 W&B와의 실시간 동기화를 활성화하여 시각화에 대한 실시간 업데이트를 제공합니다. - `"offline"`: 에어 갭 또는 오프라인 환경에 적합합니다. 데이터는 로컬에 저장되며 나중에 동기화할 수 있습니다. 나중에 동기화할 수 있도록 run 폴더가 보존되었는지 확인하십시오. - `"disabled"`: 모든 W&B 기능을 비활성화하여 run의 메소드를 작동하지 않도록 합니다. 일반적으로 W&B 작업을 우회하기 위해 테스트에 사용됩니다. |
|  `force` |  스크립트를 실행하는 데 W&B 로그인이 필요한지 여부를 결정합니다. `True`이면 사용자가 W&B에 로그인해야 합니다. 그렇지 않으면 스크립트가 진행되지 않습니다. `False`(기본값)이면 사용자가 로그인하지 않은 경우 오프라인 모드로 전환하여 스크립트를 로그인 없이 진행할 수 있습니다. |
|  `anonymous` |  익명 데이터 로깅에 대한 제어 수준을 지정합니다. 사용 가능한 옵션은 다음과 같습니다. - `"never"` (기본값): run을 추적하기 전에 W&B 계정을 연결해야 합니다. 이렇게 하면 각 run이 계정과 연결되도록 하여 의도치 않은 익명 run 생성을 방지할 수 있습니다. - `"allow"`: 로그인한 사용자가 자신의 계정으로 run을 추적할 수 있지만 W&B 계정 없이 스크립트를 실행하는 사용자가 UI에서 차트와 데이터를 볼 수 있도록 허용합니다. - `"must"`: 사용자가 로그인했더라도 run을 익명 계정에 강제로 기록합니다. |
|  `reinit` |  동일한 프로세스 내에서 여러 `wandb.init()` 호출이 새 run을 시작할 수 있는지 여부를 결정합니다. 기본적으로(`False`) 활성 run이 있는 경우 `wandb.init()`을 호출하면 새 run을 만드는 대신 기존 run이 반환됩니다. `reinit=True`이면 새 run이 초기화되기 전에 활성 run이 완료됩니다. 노트북 환경에서는 `reinit`이 명시적으로 `False`로 설정되지 않은 경우 기본적으로 run이 다시 초기화됩니다. |
|  `resume` |  지정된 `id`로 run을 다시 시작할 때의 동작을 제어합니다. 사용 가능한 옵션은 다음과 같습니다. - `"allow"`: 지정된 `id`를 가진 run이 있으면 마지막 단계부터 다시 시작됩니다. 그렇지 않으면 새 run이 생성됩니다. - `"never"`: 지정된 `id`를 가진 run이 있으면 오류가 발생합니다. 그러한 run이 없으면 새 run이 생성됩니다. - `"must"`: 지정된 `id`를 가진 run이 있으면 마지막 단계부터 다시 시작됩니다. run이 없으면 오류가 발생합니다. - `"auto"`: 이 머신에서 이전 run이 충돌한 경우 자동으로 다시 시작합니다. 그렇지 않으면 새 run을 시작합니다. - `True`: 더 이상 사용되지 않습니다. 대신 `"auto"`를 사용하십시오. - `False`: 더 이상 사용되지 않습니다. 항상 새 run을 시작하려면 기본 동작( `resume`를 설정하지 않은 상태로 둠)을 사용하십시오. 참고: `resume`이 설정된 경우 `fork_from` 및 `resume_from`을 사용할 수 없습니다. `resume`이 설정되지 않은 경우 시스템은 항상 새 run을 시작합니다. 자세한 내용은 [run 다시 시작 가이드](https://docs.wandb.com/guides/runs/resuming)를 참조하십시오. |
|  `resume_from` |  `{run_id}?_step={step}` 형식을 사용하여 이전 run에서 run을 다시 시작할 시점을 지정합니다. 이를 통해 사용자는 중간 단계에서 run에 기록된 기록을 자르고 해당 단계부터 로깅을 다시 시작할 수 있습니다. 대상 run은 동일한 project에 있어야 합니다. `id` 인수가 함께 제공되면 `resume_from` 인수가 우선합니다. `resume`, `resume_from` 및 `fork_from`은 함께 사용할 수 없으며, 셋 중 하나만 한 번에 사용할 수 있습니다. 참고: 이 기능은 베타 버전이며 향후 변경될 수 있습니다. |
|  `fork_from` |  `{id}?_step={step}` 형식을 사용하여 이전 run에서 새 run을 포크할 시점을 지정합니다. 이렇게 하면 대상 run 기록의 지정된 단계에서 로깅을 다시 시작하는 새 run이 생성됩니다. 대상 run은 현재 프로젝트의 일부여야 합니다. `id` 인수가 함께 제공되면 `fork_from` 인수와 달라야 합니다. 동일하면 오류가 발생합니다. `resume`, `resume_from` 및 `fork_from`은 함께 사용할 수 없으며, 셋 중 하나만 한 번에 사용할 수 있습니다. 참고: 이 기능은 베타 버전이며 향후 변경될 수 있습니다. |
|  `save_code` |  실험 재현성을 지원하고 UI에서 run 간에 코드 비교를 허용하여 기본 스크립트 또는 노트북을 W&B에 저장할 수 있습니다. 기본적으로 이 기능은 비활성화되어 있지만 [설정 페이지](https://wandb.ai/settings)에서 기본값을 변경하여 활성화할 수 있습니다. |
|  `tensorboard` |  더 이상 사용되지 않습니다. 대신 `sync_tensorboard`를 사용하십시오. |
|  `sync_tensorboard` |  TensorBoard 또는 TensorBoardX에서 W&B 로그의 자동 동기화를 활성화하여 W&B UI에서 볼 수 있도록 관련 이벤트 파일을 저장합니다. W&B UI에서 볼 수 있도록 관련 이벤트 파일을 저장합니다. (기본값: `False`) |
|  `monitor_gym` |  OpenAI Gym을 사용할 때 환경 비디오의 자동 로깅을 활성화합니다. 자세한 내용은 [gym 인테그레이션 가이드](https://docs.wandb.com/guides/integrations/openai-gym)를 참조하십시오. |
|  `settings` |  run에 대한 고급 설정이 포함된 딕셔너리 또는 `wandb.Settings` 오브젝트를 지정합니다. |

| 반환 |  |
| :--- | :--- |
|  현재 run에 대한 핸들인 `Run` 오브젝트입니다. 이 오브젝트를 사용하여 데이터 로깅, 파일 저장 및 run 완료와 같은 작업을 수행합니다. 자세한 내용은 [Run API](https://docs.wandb.ai/ref/python/run)를 참조하십시오. |

| 발생 |  |
| :--- | :--- |
|  `Error` |  run 초기화 중에 알려지지 않거나 내부 오류가 발생한 경우. |
|  `AuthenticationError` |  사용자가 유효한 자격 증명을 제공하지 못한 경우. |
|  `CommError` |  W&B 서버와 통신하는 데 문제가 있는 경우. |
|  `UsageError` |  사용자가 함수에 유효하지 않은 인수를 제공한 경우. |
|  `KeyboardInterrupt` |  사용자가 run 초기화 프로세스를 중단한 경우. 사용자가 run 초기화 프로세스를 중단한 경우. |
