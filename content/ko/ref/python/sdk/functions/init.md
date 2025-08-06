---
title: 'init()

  '
data_type_classification: function
menu:
  reference:
    identifier: ko-ref-python-sdk-functions-init
object_type: python_sdk_actions
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_init.py >}}




### <kbd>function</kbd> `init`

```python
init(
    entity: 'str | None' = None,
    project: 'str | None' = None,
    dir: 'StrPath | None' = None,
    id: 'str | None' = None,
    name: 'str | None' = None,
    notes: 'str | None' = None,
    tags: 'Sequence[str] | None' = None,
    config: 'dict[str, Any] | str | None' = None,
    config_exclude_keys: 'list[str] | None' = None,
    config_include_keys: 'list[str] | None' = None,
    allow_val_change: 'bool | None' = None,
    group: 'str | None' = None,
    job_type: 'str | None' = None,
    mode: "Literal['online', 'offline', 'disabled'] | None" = None,
    force: 'bool | None' = None,
    anonymous: "Literal['never', 'allow', 'must'] | None" = None,
    reinit: "bool | Literal[None, 'default', 'return_previous', 'finish_previous', 'create_new']" = None,
    resume: "bool | Literal['allow', 'never', 'must', 'auto'] | None" = None,
    resume_from: 'str | None' = None,
    fork_from: 'str | None' = None,
    save_code: 'bool | None' = None,
    tensorboard: 'bool | None' = None,
    sync_tensorboard: 'bool | None' = None,
    monitor_gym: 'bool | None' = None,
    settings: 'Settings | dict[str, Any] | None' = None
) → Run
```

새로운 run 을 시작하여 W&B 에 메트릭을 추적하고 로그를 남길 수 있습니다.

ML 트레이닝 파이프라인에서는 트레이닝 스크립트나 평가 스크립트의 시작 부분에 `wandb.init()` 을 추가하면 각각이 W&B 의 개별 run 으로 추적됩니다.

`wandb.init()` 은 데이터를 run 에 기록하는 새로운 백그라운드 프로세스를 생성하며, 기본적으로 https://wandb.ai 로 데이터를 동기화하므로 결과를 실시간으로 확인할 수 있습니다. 데이터 기록이 끝나면 `wandb.finish()` 를 호출하여 run 을 종료하세요. 만약 `run.finish()` 를 호출하지 않으면, 스크립트가 종료될 때 자동으로 run 이 종료됩니다.

run ID에는 `/ \ # ? % :` 와 같은 특수 문자를 사용할 수 없습니다.



**인수(Args):**

 - `entity`:  run 이 기록될 사용자명 또는 팀명을 지정합니다. 해당 entity 는 미리 존재해야 하므로, 처음 로그를 남기기 전에 UI 에서 계정 또는 팀을 생성하세요. 지정하지 않으면 기본 entity 가 사용됩니다. 기본 entity 를 변경하려면 설정에서 "Default location to create new projects" 항목(“Default team” 아래)을 업데이트하세요.
 - `project`:  run 이 저장될 프로젝트 이름입니다. 지정하지 않으면 시스템에서 git 루트나 현재 프로그램 파일 등을 확인해 프로젝트 이름을 추정합니다. 만약 추정할 수 없으면 `"uncategorized"` 로 지정됩니다.
 - `dir`:  실험 로그 및 메타데이터 파일이 저장될 디렉토리의 절대경로입니다. 지정하지 않으면 기본적으로 `./wandb` 디렉토리가 사용됩니다. 이 경로는 `download()` 를 호출해 artifact 를 저장하는 위치에는 영향을 주지 않습니다.
 - `id`:  run 을 식별하는 고유 ID로, 재시작(resume) 시 사용합니다. 프로젝트 내에서 고유해야 하며, 삭제된 run 에는 재사용할 수 없습니다. 간단한 설명용 이름은 `name` 필드를, run 별 하이퍼파라미터 비교를 위해서는 `config` 를 사용하세요.
 - `name`:  run 에 표시될 짧은 표시 이름입니다. UI 에서 run 을 쉽게 식별할 수 있도록 도와줍니다. 기본적으로는 무작위 두 단어로 생성되어 차트나 테이블에서 run 을 쉽게 교차 참조할 수 있습니다. 짧은 이름을 사용하면 시각화에서 가독성이 좋아집니다. 하이퍼파라미터 저장에는 `config` 필드를 활용하세요.
 - `notes`:  run 에 대한 자세한 설명을 남기는 필드로, Git의 커밋 메시지와 유사하게 어떤 목적으로 실행한 run인지, 세부사항 등을 기록할 수 있습니다.
 - `tags`:  run 에 태그를 추가해 UI 에서 분류하거나 "baseline", "production" 등 임시 식별자를 추가할 수 있습니다. 태그로 run 을 쉽게 필터하거나 추가/제거할 수 있습니다. 재시작(run resume)시에는 여기 입력한 태그가 기존 태그를 덮어씌웁니다. 기존 태그를 덮지 않고 추가하려면, `run = wandb.init()` 호출 후 `run.tags += ("new_tag",)` 처럼 사용하세요.
 - `config`:  `wandb.config` 에 저장될 입력 파라미터(예: 모델 하이퍼파라미터, 데이터 전처리 설정 등)를 저장합니다. UI 의 개요 페이지에서 이를 기반으로 run 을 그룹화, 필터, 정렬할 수 있습니다. 키에 점(`.`)은 사용할 수 없고, 값은 10MB 이하로 저장해야 합니다. dict, `argparse.Namespace`, `absl.flags.FLAGS` 등을 넘기면 key-value 형식이 `wandb.config` 에 바로 적용됩니다. 문자열을 넘기면 YAML 파일 경로로 해석되어 해당 파일의 설정값이 `wandb.config` 로 불러와집니다.
 - `config_exclude_keys`:  `wandb.config` 에서 특정 키를 제외할 목록입니다.
 - `config_include_keys`:  `wandb.config` 에서 특정 키만 포함할 목록입니다.
 - `allow_val_change`:  config 값이 최초 설정 후 변경 가능한지 여부를 제어합니다. 기본적으로 값이 덮어써지면 예외가 발생합니다. 트레이닝 도중 값이 변하는 변수(예: learning rate 등)는 `wandb.log()` 로 기록하는 것을 추천합니다. 기본값은 script에서는 `False`, Notebook 환경에서는 `True` 입니다.
 - `group`:  여러 run 을 하나의 큰 실험으로 묶고 싶을 때 그룹명을 지정합니다. 예를 들어 cross-validation, 여러 데이터셋에 모델을 트레이닝/평가하는 경우 등 그룹으로 묶으면 관련 run 을 한꺼번에 관리/시각화할 수 있습니다.
 - `job_type`:  run 의 타입을 지정하는 값으로, 그룹 내에서 여러 개의 서로 다른 임무를 가진 run 을 구분할 때 유용합니다. 예를 들어 "train", "eval"같은 job type 을 붙이면 UI 에서 필터링과 비교가 쉬워집니다.
 - `mode`:  run 데이터의 동작 방식을 지정합니다.
    - `"online"` (기본): 네트워크가 연결되어 있으면 W&B 에 실시간 동기화 및 실시간 시각화 업데이트를 지원합니다.
    - `"offline"`: 오프라인 환경(또는 air-gapped 환경)에서 사용하며, 데이터는 로컬에 저장되고 나중에 동기화할 수 있습니다. 추후 동기화를 위해 run 폴더를 보존해야 합니다.
    - `"disabled"`: 모든 W&B 기능을 비활성화합니다(run 메소드가 아무 동작도 하지 않음). 보통 테스트 환경에서 사용합니다.
 - `force`:  이 값이 `True`이면 W&B 로그인이 완료되어야 스크립트가 실행됩니다. `False`(기본)면 로그인하지 않아도 실행되며, 미로그인 상태일 경우 자동으로 offline 모드로 전환됩니다.
 - `anonymous`:  익명 데이터 로깅 허용 수준을 지정합니다.
    - `"never"` (기본): W&B 계정 연결이 필요합니다. 이 옵션은 실수로 익명 run 이 생성되지 않도록 방지합니다.
    - `"allow"`: 로그인한 사용자는 계정과 함께 run 을 기록할 수 있지만, 미로그인 사용자도 UI 에서 차트와 데이터를 볼 수 있습니다.
    - `"must"`: 로그인 상태여도 run 을 익명 계정으로 기록합니다.
 - `reinit`:  "reinit" 설정의 단축어입니다. 이미 run 이 실행 중인 상태에서 `wandb.init()` 의 동작 방식을 정합니다.
 - `resume`:  지정한 `id`로 run 을 재시작할 때 동작을 제어합니다.
    - `"allow"`: 지정한 `id`의 run 이 있으면 마지막 스텝부터 재시작, 없으면 새 run 생성.
    - `"never"`: 동일 id의 run 이 있으면 에러, 없으면 새 run 생성.
    - `"must"`: 동일 id의 run 이 있으면 마지막 스텝부터 재시작, 없으면 에러.
    - `"auto"`: 이 머신에서 crash된 이전 run 이 있으면 자동 재시작, 없으면 새로 시작.
    - `True`: 더 이상 사용되지 않음. `"auto"` 사용 권장.
    - `False`: 더 이상 사용되지 않음. `resume` 미지정(default)이 항상 새 run 을 시작합니다.  `resume` 사용 시 `fork_from`, `resume_from` 을 함께 쓸 수 없으며, 세 가지 중 하나만 사용해야 합니다. `resume`을 미설정하면 무조건 새 run 이 시작됩니다.
 - `resume_from`:  이전 run 의 지정된 시점에서 {run_id}?_step={step} 형태로 run 을 재시작할 때 사용합니다. run 의 히스토리 일부만 남기고 해당 시점부터 다시 기록하려는 경우에 사용합니다. 대상 run 은 같은 프로젝트 내에 있어야 합니다. 만약 `id`도 함께 지정하면 `resume_from`이 우선 적용됩니다. `resume`, `resume_from`, `fork_from`은 동시에 사용할 수 없으며, 오직 하나만 사용할 수 있습니다. 아직 베타 기능이므로 추후 변경될 수 있습니다.
 - `fork_from`:  이전 run 의 지정 시점에서 {id}?_step={step} 형태로 새로운 run 을 fork 하고, 해당 시점부터 기록을 재개합니다. 대상 run 은 같은 프로젝트 내에 속해야 합니다. 만약 `id`도 지정하면, 이 값이 `fork_from`과 다르지 않으면 에러가 발생합니다. `resume`, `resume_from`, `fork_from` 중 하나만 사용할 수 있습니다. 이 기능도 베타 버전이므로 추후 변경될 수 있습니다.
 - `save_code`:  메인 스크립트나 노트북을 W&B 에 저장해 실험 재현성을 높이고, run 간 코드 비교를 가능하게 합니다. 기본값은 비활성화이며, 설정 페이지에서 기본값을 변경할 수 있습니다.
 - `tensorboard`:  더 이상 사용되지 않습니다. 대신 `sync_tensorboard` 를 사용하세요.
 - `sync_tensorboard`:  TensorBoard 나 TensorBoardX 에서 W&B 로그를 자동 동기화하며, 관련 이벤트 파일을 W&B UI 에서 볼 수 있도록 저장합니다.
 - `monitor_gym`:  OpenAI Gym 환경 사용 시 환경 동작 비디오를 자동으로 기록합니다.
 - `settings`:  run 에 대한 고급 설정을 담은 사전(dict) 또는 `wandb.Settings` 오브젝트를 지정합니다.



**예외(Raises):**

 - `Error`:  run 초기화 중 알 수 없거나 내부 오류가 발생한 경우
 - `AuthenticationError`:  사용자가 유효한 인증 정보 제공에 실패한 경우
 - `CommError`:  WandB 서버와 통신에 문제가 있는 경우
 - `UsageError`:  유효하지 않은 인수가 제공된 경우
 - `KeyboardInterrupt`:  사용자가 run 을 중단한 경우



**반환(Returns):**
 `Run` 오브젝트를 반환합니다.





**예시(Examples):**
 `wandb.init()` 은 `Run` 오브젝트를 반환하며, 이 오브젝트를 이용해 데이터 로깅, artifact 저장, run 라이프사이클 관리를 할 수 있습니다.

```python
import wandb

config = {"lr": 0.01, "batch_size": 32}
with wandb.init(config=config) as run:
    # run 에 정확도와 손실 값을 기록합니다.
    acc = 0.95  # 예시 정확도
    loss = 0.05  # 예시 손실 값
    run.log({"accuracy": acc, "loss": loss})
```