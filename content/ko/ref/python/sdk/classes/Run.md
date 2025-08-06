---
title: 'Run

  '
data_type_classification: class
menu:
  reference:
    identifier: ko-ref-python-sdk-classes-Run
object_type: python_sdk_actions
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_run.py >}}




## <kbd>class</kbd> `Run`
W&B가 기록하는 하나의 연산 단위입니다. 일반적으로 ML experiment를 의미합니다.

새로운 run을 생성하려면 [`wandb.init()`](https://docs.wandb.ai/ref/python/init/)을 호출하세요. `wandb.init()`은 새로운 run을 시작하고 `wandb.Run` 객체를 반환합니다. 각 run에는 고유한 ID(run ID)가 부여됩니다. W&B에서는 context(`with`문)을 사용해 run을 자동으로 종료하는 것을 권장합니다.

분산 트레이닝 experiment에서는 각 프로세스를 개별 run으로 추적하거나, 모든 프로세스를 하나의 run으로 추적할 수 있습니다. 자세한 내용은 [Log distributed training experiments](https://docs.wandb.ai/guides/track/log/distributed-training)를 참고하세요.

`wandb.Run.log()`로 run에 데이터를 기록할 수 있습니다. `wandb.Run.log()`를 사용해 기록되는 모든 데이터는 해당 run에 업로드됩니다. 더 자세한 내용은 [Create an experiment](https://docs.wandb.ai/guides/track/launch), 또는 [`wandb.init`](https://docs.wandb.ai/ref/python/init/) API 레퍼런스를 참고하세요.

[`wandb.apis.public`](https://docs.wandb.ai/ref/python/public-api/api/) 네임스페이스에도 또 다른 `Run` 객체가 있습니다. 이 객체는 이미 생성된 run들과 상호작용할 때 사용합니다.



**속성:**
 
 - `summary`:  (Summary) run의 요약 정보로, 딕셔너리처럼 사용할 수 있습니다. 자세한 내용은
 - [Log summary metrics](https://docs.wandb.ai/guides/track/log/log-summary/)를 참고하세요.



**예시:**
`wandb.init()`로 run을 생성하기:

```python
import wandb

# 새로운 run을 시작하고 데이터를 로깅합니다.
# with문(context manager)을 사용하면 run이 자동으로 종료됩니다.
with wandb.init(entity="entity", project="project") as run:
    run.log({"accuracy": acc, "loss": loss})
``` 


### <kbd>property</kbd> Run.config

이 run과 연결된 config 객체입니다.

---

### <kbd>property</kbd> Run.config_static

이 run과 연결된 static config 객체입니다.

---

### <kbd>property</kbd> Run.dir

run과 연결된 파일이 저장되는 디렉토리입니다.

---

### <kbd>property</kbd> Run.disabled

run이 비활성화된 경우 True, 아니면 False입니다.

---

### <kbd>property</kbd> Run.entity

이 run과 연결된 W&B entity의 이름입니다.

엔터티는 사용자명, 팀, 또는 조직 이름이 될 수 있습니다.

---

### <kbd>property</kbd> Run.group

이 run이 속한 group의 이름을 반환합니다.

run들을 그룹화하면 관련된 experiment를 W&B UI에서 한 번에 정리/시각화할 수 있습니다. 분산 트레이닝이나 교차 검증 등, 여러 run을 하나의 experiment로 관리해야 하는 경우에 특히 유용합니다.

모든 프로세스가 동일한 run 객체를 공유하는 shared 모드에서는, group을 별도로 설정할 필요가 없습니다.

---

### <kbd>property</kbd> Run.id

이 run의 식별자(ID)입니다.

---

### <kbd>property</kbd> Run.job_type

이 run과 연결된 job 유형의 이름입니다.

run의 Overview 페이지(실행 화면)에서 job 유형을 확인할 수 있습니다.

run을 "training", "evaluation", "inference" 등 job 별로 분류하거나 필터링할 때 유용합니다. 관련 내용은 [Organize runs](https://docs.wandb.ai/guides/runs/#organize-runs) 참고.

---

### <kbd>property</kbd> Run.name

run의 표시 이름(display name)입니다.

표시 이름은 반드시 고유해야 하는 것은 아니며 주로 설명 용도로 사용됩니다. 디폴트로는 랜덤하게 생성됩니다.

---

### <kbd>property</kbd> Run.notes

run과 연결된 노트(있을 경우)입니다.

노트는 여러 줄의 문자열로 남길 수 있으며, 마크다운 또는 `$x + 3$`과 같이 latex 수식(`$$` 안에 작성)도 지원합니다.

---

### <kbd>property</kbd> Run.offline

run이 오프라인 모드일 경우 True, 아니면 False입니다.

---

### <kbd>property</kbd> Run.path

run의 경로입니다.

형식은 `entity/project/run_id`이며, entity, project, run ID를 포함합니다.

---

### <kbd>property</kbd> Run.project

이 run과 연결된 W&B project의 이름입니다.

---

### <kbd>property</kbd> Run.project_url

이 run과 연결된 W&B project의 URL(있을 때만)입니다.

오프라인 run은 project URL이 없습니다.

---

### <kbd>property</kbd> Run.resumed

run이 resume(재시작)됐으면 True, 아니면 False입니다.

---

### <kbd>property</kbd> Run.settings

run의 Settings 객체의 frozen(고정) 복사본입니다.

---

### <kbd>property</kbd> Run.start_time

run이 시작된 시점의 Unix timestamp(초 단위)입니다.

---



### <kbd>property</kbd> Run.sweep_id

이 run과 연결된 sweep의 식별자(있을 경우)입니다.

---

### <kbd>property</kbd> Run.sweep_url

이 run과 연결된 sweep의 URL(있을 경우)입니다.

오프라인 run은 sweep URL이 없습니다.

---

### <kbd>property</kbd> Run.tags

run에 태깅된 태그 목록(있을 경우)입니다.

---

### <kbd>property</kbd> Run.url

이 run의 W&B URL(있을 경우)입니다.

오프라인 run은 URL이 없습니다.



---

### <kbd>method</kbd> `Run.alert`

```python
alert(
    title: 'str',
    text: 'str',
    level: 'str | AlertLevel | None' = None,
    wait_duration: 'int | float | timedelta | None' = None
) → None
```

제공한 타이틀과 텍스트로 알림을 생성합니다.



**인수:**
 
 - `title`:  알림의 제목으로, 64자 이하이어야 합니다.
 - `text`:  알림의 본문 내용입니다.
 - `level`:  사용할 알림 레벨로, `INFO`, `WARN`, `ERROR` 중 하나입니다.
 - `wait_duration`:  동일한 제목의 알림을 보낼 때까지 기다릴 시간(초 단위)입니다.

---

### <kbd>method</kbd> `Run.define_metric`

```python
define_metric(
    name: 'str',
    step_metric: 'str | wandb_metric.Metric | None' = None,
    step_sync: 'bool | None' = None,
    hidden: 'bool | None' = None,
    summary: 'str | None' = None,
    goal: 'str | None' = None,
    overwrite: 'bool | None' = None
) → wandb_metric.Metric
```

`wandb.Run.log()`로 기록되는 메트릭을 커스터마이즈합니다.



**인수:**
 
 - `name`:  커스터마이즈할 메트릭 이름입니다.
 - `step_metric`:  이 메트릭의 X축이 될 다른 메트릭의 이름입니다(자동 차트를 위해).
 - `step_sync`:  `wandb.Run.log()` 사용 시 명시적으로 제공하지 않아도 step_metric의 마지막 값을 자동으로 입력합니다. step_metric이 있을 땐 기본 True입니다.
 - `hidden`:  자동 차트에서 이 메트릭을 숨깁니다.
 - `summary`:  summary에 추가할 집계 메트릭을 지정합니다. 지원되는 집계는 "min", "max", "mean", "last", "best", "copy", "none" 등입니다. "best"는 goal 파라미터와 함께 사용합니다. "none"은 summary 생성을 방지합니다. "copy"는 더 이상 사용하지 않습니다.
 - `goal`:  "best" summary 타입을 해석하는 방법을 지정합니다. "minimize", "maximize"가 지원됩니다.
 - `overwrite`:  False면, 이전에 지정한 값 중 지정되지 않은 파라미터는 이전 값을 사용해 병합합니다. True면, 이전의 값을 지정되지 않은 파라미터도 덮어씁니다.



**반환값:**
 이 호출을 나타내는 객체(필요 없으면 버려도 됩니다).

---

### <kbd>method</kbd> `Run.display`

```python
display(height: 'int' = 420, hidden: 'bool' = False) → bool
```

이 run을 Jupyter에서 표시합니다.

---

### <kbd>method</kbd> `Run.finish`

```python
finish(exit_code: 'int | None' = None, quiet: 'bool | None' = None) → None
```

run을 종료하고 남은 데이터를 업로드합니다.

W&B run의 완료 상태를 표시하며, 모든 데이터가 서버에 동기화되도록 보장합니다. run의 최종 상태는 종료 조건과 동기화 상태에 따라 결정됩니다.

Run 상태:
- Running: 데이터 기록 중이거나 heartbeat를 보내는 활성 run입니다.
- Crashed: heartbeat 전송이 갑자기 중단된 run입니다.
- Finished: 모든 데이터가 정상적으로 동기화되면서 성공적으로 완료된 run입니다(`exit_code=0`).
- Failed: 에러와 함께 종료된 run입니다(`exit_code!=0`).
- Killed: 강제 종료되어 끝나지 못한 run입니다.



**인수:**
 
 - `exit_code`:  run의 종료 상태를 나타내는 정수입니다. 성공은 0, 이외의 값은 실패로 간주됩니다.
 - `quiet`:  더 이상 사용되지 않습니다. 로그 상세 수준은 `wandb.Settings(quiet=...)`으로 조절하세요.

---

### <kbd>method</kbd> `Run.finish_artifact`

```python
finish_artifact(
    artifact_or_path: 'Artifact | str',
    name: 'str | None' = None,
    type: 'str | None' = None,
    aliases: 'list[str] | None' = None,
    distributed_id: 'str | None' = None
) → Artifact
```

비최종(임시) 아티팩트를 run의 출력물로 마무리(finalize)합니다.

동일한 distributed ID로 여러 번 "upsert"할 경우, 새 버전이 생성됩니다.



**인수:**
 
 - `artifact_or_path`:  이 아티팩트의 내용이 위치한 경로로, 다음 중 하나의 경우를 지원합니다:
            - `/local/directory`
            - `/local/directory/file.txt`
            - `s3://bucket/path`  또한, `wandb.Artifact`로 생성한 Artifact 객체도 바로 전달할 수 있습니다.
 - `name`:  아티팩트 이름. entity/project를 접두어로 붙일 수도 있습니다.
            - name:version
            - name:alias
            - digest  지정하지 않으면 디폴트로 경로의 basename에 현재 run id가 덧붙은 값이 사용됩니다.
 - `type`:  기록할 아티팩트 유형. 예: `dataset`, `model`
 - `aliases`:  이 아티팩트에 추가할 에일리어스. 기본값은 `["latest"]`
 - `distributed_id`:  모든 분산 job이 공유할 고유 문자열. None일 경우 run의 group 이름이 기본값입니다.



**반환값:**
 `Artifact` 객체.

---




### <kbd>method</kbd> `Run.link_artifact`

```python
link_artifact(
    artifact: 'Artifact',
    target_path: 'str',
    aliases: 'list[str] | None' = None
) → Artifact | None
```

지정한 artifact를 포트폴리오(artifacts 컬렉션)에 연결합니다.

연결된 artifacts는 지정한 포트폴리오의 UI에서 볼 수 있습니다.



**인수:**
 
 - `artifact`:  연결할 (public 또는 local) artifact
 - `target_path`:  `{portfolio}`, `{project}/{portfolio}`, 또는 `{entity}/{project}/{portfolio}` 형태의 string
 - `aliases`:  이 포트폴리오 내 해당 artifact에만 적용할 선택적 에일리어스(목록). "latest"는 항상 마지막 버전에 자동 적용됩니다.



**반환값:**
 연결 성공 시 해당 artifact, 실패 시 None.

---

### <kbd>method</kbd> `Run.link_model`

```python
link_model(
    path: 'StrPath',
    registered_model_name: 'str',
    name: 'str | None' = None,
    aliases: 'list[str] | None' = None
) → Artifact | None
```

모델 artifact 버전을 기록하고, model registry의 registered model에 연결합니다.

연결된 모델 버전은 지정한 registered model의 UI에서 볼 수 있습니다.

이 함수는 다음을 수행합니다:
- 지정한 'name'의 모델 artifact가 이미 로깅된 경우, 'path'에 있는 파일들과 일치하는 artifact 버전을 사용하거나 새 버전을 로깅합니다. 없으면 주어진 path 하위의 파일을 새로운 모델 artifact('name') 타입 'model'로 기록합니다.
- 'model-registry' 프로젝트에 'registered_model_name'이 존재하는지 확인하고, 없으면 새로 생성합니다.
- 모델 artifact 버전('name')을 registered model('registered_model_name')에 연결합니다.
- 'aliases' 목록의 에일리어스를 새롭게 연결된 모델 버전에 붙입니다.



**인수:**
 
 - `path`:  (str) 모델 내용의 경로로, 다음 중 하나:
    - `/local/directory`
    - `/local/directory/file.txt`
    - `s3://bucket/path`
 - `registered_model_name`:  이 모델을 연결할 registered model의 이름. registered model은 팀의 특정 ML Task를 나타내는 모델 버전들의 컬렉션입니다. 등록된 모델의 entity는 run 정보에서 가져옵니다.
 - `name`:  'path'의 파일들이 기록될 모델 artifact 이름. 지정하지 않으면 path의 basename에 run id가 덧붙여집니다.
 - `aliases`:  이 registered model 안에서만 적용될 에일리어스 목록. "latest"는 항상 마지막 버전에 적용됩니다.



**예외 발생:**
 
 - `AssertionError`:  registered_model_name에 경로가 들어갔거나, model artifact 'name'이 'model'이 포함되지 않는 타입이면 발생.
 - `ValueError`:  name에 허용되지 않은 특수문자가 들어간 경우.



**반환값:**
 성공하면 연결된 artifact, 아니면 `None`.

---

### <kbd>method</kbd> `Run.log`

```python
log(
    data: 'dict[str, Any]',
    step: 'int | None' = None,
    commit: 'bool | None' = None
) → None
```

run 데이터를 업로드합니다.

run에서 생성된 데이터(스칼라, 이미지, 동영상, 히스토그램, 차트, 표 등)를 기록할 때 log를 사용하세요. 코드 예시, 모범 사례 등은 [Log objects and media](https://docs.wandb.ai/guides/track/log)에서 확인하세요.

기본 사용법:

```python
import wandb

with wandb.init() as run:
     run.log({"train-loss": 0.5, "accuracy": 0.9})
```

위 코드조각은 loss와 accuracy를 run의 history에 저장하고, summary 값도 업데이트합니다.

기록된 데이터는 [wandb.ai](https://wandb.ai) 워크스페이스, 또는 [자가 호스팅 인스턴스](https://docs.wandb.ai/guides/hosting)에서 로컬로 시각화할 수 있습니다. [Public API](https://docs.wandb.ai/guides/track/public-api-guide)를 이용하면 데이터를 내보내 Jupyter 노트북 등에서 직접 탐색할 수도 있습니다.

기록 값은 스칼라일 필요가 없습니다. 이미지, 오디오, 동영상 등 다양한 [W&B 지원 데이터 타입](https://docs.wandb.ai/ref/python/data-types/)을 모두 기록할 수 있습니다. 예를 들어, `wandb.Table`을 활용해 구조화된 데이터를 로깅할 수 있습니다. 자세한 내용 및 튜토리얼은 [Log tables, visualize and query data](https://docs.wandb.ai/guides/models/tables/tables-walkthrough)를 참고하세요.

W&B는 이름에 슬래시(`/`)가 포함된 metric을 마지막 슬래시 앞의 텍스트별 섹션으로 구분해서 나눕니다. 예:

```python
with wandb.init() as run:
     # "train", "validate"라는 두 개의 섹션에 각각 기록됩니다.
     run.log(
         {
             "train/accuracy": 0.9,
             "train/loss": 30,
             "validate/accuracy": 0.8,
             "validate/loss": 20,
         }
     )
```

중첩은 한 단계까지만 지원합니다. `run.log({"a/b/c": 1})`이면 "a/b"라는 섹션이 만들어집니다.

`run.log()`는 초당 여러 번 호출하는 용도가 아닙니다. 성능을 최적화하려면 N회마다 한 번씩 로그하거나, 여러 번 동안 데이터를 모아서 한 번에 기록하는 것이 좋습니다.

기본적으로 log를 호출할 때마다 새로운 "step"이 생성됩니다. step 값은 항상 증가해야 하며, 이전 step으로 기록할 수는 없습니다. 어떤 메트릭도 차트의 X 축으로 쓸 수 있습니다. 자세한 내용은 [Custom log axes](https://docs.wandb.ai/guides/track/log/customize-logging-axes/) 참고.

많은 경우 W&B의 step을 트레이닝 스텝이 아니라 timestamp처럼 다루는 것이 더 좋을 수 있습니다.

```python
with wandb.init() as run:
     # 예시: "epoch" 메트릭을 X축 용도로 기록합니다.
     run.log({"epoch": 40, "train-loss": 0.5})
```

`step`과 `commit` 파라미터로 여러 번의 `wandb.Run.log()` 호출을 동일한 step에 기록할 수 있습니다. 예:

```python
with wandb.init() as run:
     # 일반적인 사용:
     run.log({"train-loss": 0.5, "accuracy": 0.8})
     run.log({"train-loss": 0.4, "accuracy": 0.9})

     # step 자동 증가 없이 사용:
     run.log({"train-loss": 0.5}, commit=False)
     run.log({"accuracy": 0.8})
     run.log({"train-loss": 0.4}, commit=False)
     run.log({"accuracy": 0.9})

     # step을 명시적으로 설정:
     run.log({"train-loss": 0.5}, step=current_step)
     run.log({"accuracy": 0.8}, step=current_step)
     current_step += 1
     run.log({"train-loss": 0.4}, step=current_step)
     run.log({"accuracy": 0.9}, step=current_step)
```



**인수:**
 
 - `data`:  `str` 키와 직렬화 가능한 값이 들어간 `dict`
 - `Python objects including`:  `int`, `float`, `string` 외에, `wandb.data_types`, 직렬화 가능한 목록/튜플/NumPy 배열 및 구조체(dict).
 - `step`:  기록할 step 번호. None이면 자동으로 step이 증가합니다. 설명 참고.
 - `commit`:  True면 step을 고정하고 업로드합니다. False면 해당 step에 데이터를 누적합니다. 설명 참고. step이 None이면 기본값은 commit=True, step 값이 지정되면 기본값은 commit=False입니다.



**예시:**
 더 다양한 예시는 [로깅 가이드](https://docs.wandb.com/guides/track/log)를 참고하세요.

기본 사용 예시

```python
import wandb

with wandb.init() as run:
    run.log({"train-loss": 0.5, "accuracy": 0.9
``` 

증분 로깅 예시

```python
import wandb

with wandb.init() as run:
    run.log({"loss": 0.2}, commit=False)
    # 준비가 되면 해당 step 데이터를 기록
    run.log({"accuracy": 0.8})
``` 

히스토그램 로깅

```python
import numpy as np
import wandb

# 정규분포로 랜덤하게 샘플링된 그레이디언트
gradients = np.random.randn(100, 100)
with wandb.init() as run:
    run.log({"gradients": wandb.Histogram(gradients)})
``` 

NumPy 배열로 이미지 기록

```python
import numpy as np
import wandb

with wandb.init() as run:
    examples = []
    for i in range(3):
         pixels = np.random.randint(low=0, high=256, size=(100, 100, 3))
         image = wandb.Image(pixels, caption=f"random field {i}")
         examples.append(image)
    run.log({"examples": examples})
``` 

PIL로 이미지 기록

```python
import numpy as np
from PIL import Image as PILImage
import wandb

with wandb.init() as run:
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

NumPy로 비디오 기록

```python
import numpy as np
import wandb

with wandb.init() as run:
    # 차원 순서: (time, channel, height, width)
    frames = np.random.randint(
         low=0,
         high=256,
         size=(10, 3, 100, 100),
         dtype=np.uint8,
    )
    run.log({"video": wandb.Video(frames, fps=4)})
``` 

Matplotlib 차트 기록

```python
from matplotlib import pyplot as plt
import numpy as np
import wandb

with wandb.init() as run:
    fig, ax = plt.subplots()
    x = np.linspace(0, 10)
    y = x * x
    ax.plot(x, y)  # y = x^2 그래프
    run.log({"chart": fig})
``` 

PR 커브 기록

```python
import wandb

with wandb.init() as run:
    run.log({"pr": wandb.plot.pr_curve(y_test, y_probas, labels)})
``` 

3D 객체 기록

```python
import wandb

with wandb.init() as run:
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



**예외 발생:**
 
 - `wandb.Error`:  `wandb.init()`이 호출되기 전에 사용하면 발생.
 - `ValueError`:  잘못된 데이터를 전달하면 발생.

---

### <kbd>method</kbd> `Run.log_artifact`

```python
log_artifact(
    artifact_or_path: 'Artifact | StrPath',
    name: 'str | None' = None,
    type: 'str | None' = None,
    aliases: 'list[str] | None' = None,
    tags: 'list[str] | None' = None
) → Artifact
```

run의 출력 아티팩트로 artifact를 선언합니다.



**인수:**
 
 - `artifact_or_path`:  (str 또는 Artifact) 이 artifact의 내용 경로 또는 Artifact 객체로,
            - `/local/directory`
            - `/local/directory/file.txt`
            - `s3://bucket/path`  등 다양한 경로를 지원하며, wandb.Artifact로 생성한 Artifact 객체도 전달 가능합니다.
 - `name`:  (str, 선택) 아티팩트 이름.
            - name:version
            - name:alias
            - digest  지정하지 않으면 현재 run id가 추가된 path의 basename이 기본값입니다.
 - `type`:  (str) 기록할 artifact의 타입. 예: `dataset`, `model`
 - `aliases`:  (list, 선택) 이 artifact를 참조할 때 사용할 에일리어스. 기본값은 ["latest"].
 - `tags`:  (list, 선택) 이 artifact에 붙일 태그, 있다면 사용.



**반환값:**
 `Artifact` 객체.

---

### <kbd>method</kbd> `Run.log_code`

```python
log_code(
    root: 'str | None' = '.',
    name: 'str | None' = None,
    include_fn: 'Callable[[str, str], bool] | Callable[[str], bool]' = <function _is_py_requirements_or_dockerfile at 0x102da5f30>,
    exclude_fn: 'Callable[[str, str], bool] | Callable[[str], bool]' = <function exclude_wandb_fn at 0x103b4c5e0>
) → Artifact | None
```

현재 코드 상태를 W&B Artifact로 저장합니다.

기본적으로 현재 디렉토리에서 `.py`로 끝나는 모든 파일을 기록합니다.



**인수:**
 
 - `root`:  코드 파일을 재귀적으로 탐색할 기준 경로(상대/절대 경로).
 - `name`:  (str, 선택) 코드 artifact의 이름. 기본 값은 `source-$PROJECT_ID-$ENTRYPOINT_RELPATH`. 여러 run에서 동일한 artifact를 공유하려면 name을 명시적으로 지정할 수 있습니다.
 - `include_fn`:  파일 경로(및 필요시 root)를 받아 True/False를 반환하는 함수. 예: path.endswith(".py"). 기본값: `.py` 파일만 포함.
 - `exclude_fn`:  파일 경로(및 root)를 받아 True일 때 제외합니다. 기본은 `<root>/.wandb/`, `<root>/wandb/` 디렉토리 하위 파일 제외.



**예시:**
기본 사용법

```python
import wandb

with wandb.init() as run:
    run.log_code()
``` 

고급 사용법

```python
import wandb

with wandb.init() as run:
    run.log_code(
         root="../",
         include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"),
         exclude_fn=lambda path, root: os.path.relpath(path, root).startswith(
             "cache/"
         ),
    )
``` 



**반환값:**
 코드가 기록됐다면 `Artifact` 객체 반환

---

### <kbd>method</kbd> `Run.log_model`

```python
log_model(
    path: 'StrPath',
    name: 'str | None' = None,
    aliases: 'list[str] | None' = None
) → None
```

'path' 하위의 콘텐츠를 포함하는 모델 artifact를 run에 기록하여, 이 run의 출력물임을 표시합니다.

모델 artifact의 이름에는 영문/숫자, 언더스코어, 하이픈만 사용 가능합니다.



**인수:**
 
 - `path`:  (str) 모델 파일이 위치한 경로. 예시:
            - `/local/directory`
            - `/local/directory/file.txt`
            - `s3://bucket/path`
 - `name`:  기록될 모델 artifact의 이름. 지정하지 않으면 path의 basename에 현재 run id가 붙습니다.
 - `aliases`:  생성된 모델 artifact에 적용할 에일리어스. 기본값은 ["latest"].



**예외 발생:**
 
 - `ValueError`:  name에 허용되지 않은 특수문자가 포함된 경우.



**반환값:**
 없음

---

### <kbd>method</kbd> `Run.mark_preempting`

```python
mark_preempting() → None
```

이 run을 preempting 상태로 표시합니다.

또한 내부 프로세스에서 서버에 즉시 이를 보고합니다.

---


### <kbd>method</kbd> `Run.restore`

```python
restore(
    name: 'str',
    run_path: 'str | None' = None,
    replace: 'bool' = False,
    root: 'str | None' = None
) → None | TextIO
```

클라우드 저장소에서 지정 파일을 다운로드합니다.

파일은 현재 디렉토리 또는 run 디렉토리에 저장됩니다. 기본적으로 파일이 이미 있으면 다운로드하지 않습니다.



**인수:**
 
 - `name`:  파일 이름입니다.
 - `run_path`:  파일을 가져올 run의 경로. e.g. `username/project_name/run_id`. wandb.init을 호출하지 않았다면 필수입니다.
 - `replace`:  로컬에 파일이 이미 있어도 다시 받을 것인지 여부.
 - `root`:  파일을 다운로드할 디렉토리. 기본적으로 현재 디렉토리 또는 wandb.init이 호출된 경우 run 디렉토리.



**반환값:**
 파일을 못 찾으면 None, 있으면 읽기용 file 객체.



**예외 발생:**
 
 - `CommError`:  W&B 백엔드에 연결할 수 없는 경우.
 - `ValueError`:  파일을 찾지 못했거나 run_path를 찾지 못한 경우.

---

### <kbd>method</kbd> `Run.save`

```python
save(
    glob_str: 'str | os.PathLike',
    base_path: 'str | os.PathLike | None' = None,
    policy: 'PolicyName' = 'live'
) → bool | list[str]
```

하나 이상의 파일을 W&B에 동기화합니다.

상대 경로는 현재 작업 디렉토리를 기준으로 합니다.

"myfiles/*"와 같은 Unix glob은 `save`가 호출된 시점에 확장되며, 그 뒤에 추가된 파일은 자동으로 업로드되지 않습니다.

업로드 파일의 디렉토리 구조를 제어하려면 base_path를 사용할 수 있습니다. base_path는 glob_str의 prefix이어야 하며, base_path 아래의 구조가 유지됩니다.

절대 경로나 glob, base_path가 없는 경우에는 한 단계의 디렉토리 정보만 보존합니다.



**인수:**
 
 - `glob_str`:  상대 또는 절대경로/Unix glob.
 - `base_path`:  업로드 파일 구조에 반영할 경로 prefix.
 - `policy`:  'live', 'now', 'end' 중 하나 선택.
    - live: 파일이 바뀔 때마다 업로드 (기존 버전 덮어씀)
    - now: 현재 시점 한 번 업로드
    - end: run이 종료될 때 업로드



**반환값:**
 매칭된 파일들의 심볼릭 링크 경로 리스트.

(과거 코드와 호환 때문에 bool을 반환할 수도 있음)

```python
import wandb

run = wandb.init()

run.save("these/are/myfiles/*")
# => run 내 "these/are/myfiles/" 폴더에 파일 저장

run.save("these/are/myfiles/*", base_path="these")
# => run 내 "are/myfiles/" 폴더에 파일 저장

run.save("/User/username/Documents/run123/*.txt")
# => run 내 "run123/" 폴더에 파일 저장(아래 노트 참고).

run.save("/User/username/Documents/run123/*.txt", base_path="/User")
# => run 내 "username/Documents/run123/" 폴더에 파일 저장

run.save("files/*/saveme.txt")
# => 각 "saveme.txt" 파일을 "files/" 아래 적절한 하위 폴더에 저장

# context manager를 사용하지 않은 경우 run을 명시적으로 종료하세요.
run.finish()
``` 

---

### <kbd>method</kbd> `Run.status`

```python
status() → RunStatus
```

현재 run의 동기화(sync) 상태에 대한 내부 백엔드 정보를 가져옵니다.

---


### <kbd>method</kbd> `Run.unwatch`

```python
unwatch(
    models: 'torch.nn.Module | Sequence[torch.nn.Module] | None' = None
) → None
```

PyTorch 모델의 topology, gradient, 파라미터 훅을 제거합니다.



**인수:**
 
 - `models`:  watch가 호출된 pytorch 모델의 리스트(선택).

---

### <kbd>method</kbd> `Run.upsert_artifact`

```python
upsert_artifact(
    artifact_or_path: 'Artifact | str',
    name: 'str | None' = None,
    type: 'str | None' = None,
    aliases: 'list[str] | None' = None,
    distributed_id: 'str | None' = None
) → Artifact
```

비최종(임시) artifact를 run의 출력으로 선언하거나 추가합니다.

완성하려면 반드시 run.finish_artifact()를 따로 호출해야 합니다. 여러 분산 job에서 하나의 artifact에 기여해야 할 때 유용합니다.



**인수:**
 
 - `artifact_or_path`:  artifact의 내용 경로:
    - `/local/directory`
    - `/local/directory/file.txt`
    - `s3://bucket/path`
 - `name`:  entity/project를 접두사로 붙일 수 있는 artifact 이름. 지정하지 않으면 path의 basename에 현재 run ID가 추가됩니다.
    - name:version
    - name:alias
    - digest
 - `type`:  기록할 artifact의 타입. 예시로 `dataset`, `model` 등.
 - `aliases`:  적용할 에일리어스. 기본값은 ["latest"].
 - `distributed_id`:  분산 job이 공유할 고유 문자열. None이면 run의 group 이름이 기본값.



**반환값:**
 `Artifact` 객체.

---

### <kbd>method</kbd> `Run.use_artifact`

```python
use_artifact(
    artifact_or_name: 'str | Artifact',
    type: 'str | None' = None,
    aliases: 'list[str] | None' = None,
    use_as: 'str | None' = None
) → Artifact
```

run의 입력 artifact로 선언합니다.

반환된 객체에 `download`나 `file`을 호출해 로컬에 가져올 수 있습니다.



**인수:**
 
 - `artifact_or_name`:  사용할 artifact의 이름 또는 객체입니다. artifact가 로깅된 project 이름(`<entity>`, `<entity>/<project>`)을 접두사로 붙일 수 있습니다. entity가 지정되지 않으면 Run 또는 API에서 지정한 entity 사용. 유효한 이름 형태:
    - name:version
    - name:alias
 - `type`:  사용할 artifact의 타입.
 - `aliases`:  artifact에 적용할 에일리어스
 - `use_as`:  (더 이상 사용되지 않음, 효과 없음)



**반환값:**
 `Artifact` 객체.



**예시:**
 ```python
import wandb

run = wandb.init(project="<example>")

# 이름과 alias로 artifact 사용
artifact_a = run.use_artifact(artifact_or_name="<name>:<alias>")

# 이름과 버전으로 artifact 사용
artifact_b = run.use_artifact(artifact_or_name="<name>:v<version>")

# entity/project/name:alias로 artifact 사용
artifact_c = run.use_artifact(
    artifact_or_name="<entity>/<project>/<name>:<alias>"
)

# entity/project/name:version으로 artifact 사용
artifact_d = run.use_artifact(
    artifact_or_name="<entity>/<project>/<name>:v<version>"
)

# context manager를 쓰지 않은 경우 run을 명시적으로 종료
run.finish()
``` 

---

### <kbd>method</kbd> `Run.use_model`

```python
use_model(name: 'str') → FilePathStr
```

모델 artifact 'name'에 기록된 파일을 다운로드합니다.



**인수:**
 
 - `name`:  모델 artifact 이름. 반드시 이미 기록된 모델 artifact의 이름과 일치해야 합니다. entity/project/를 접두사로 붙일 수 있습니다.
    - model_artifact_name:version
    - model_artifact_name:alias



**반환값:**
 
 - `path` (str):  다운로드된 모델 artifact 파일의 경로.



**예외 발생:**
 
 - `AssertionError`:  모델 artifact 'name'이 'model'이 포함되지 않은 타입일 때

---

### <kbd>method</kbd> `Run.watch`

```python
watch(
    models: 'torch.nn.Module | Sequence[torch.nn.Module]',
    criterion: 'torch.F | None' = None,
    log: "Literal['gradients', 'parameters', 'all'] | None" = 'gradients',
    log_freq: 'int' = 1000,
    idx: 'int | None' = None,
    log_graph: 'bool' = False
) → None
```

지정한 PyTorch 모델에 훅(hook)를 적용해, 그레이디언트와 모델의 계산 그래프를 모니터링합니다.

트레이닝 중 파라미터, 그레이디언트(또는 둘 다)를 추적할 수 있습니다.



**인수:**
 
 - `models`:  모니터링할(추적할) 모델 한 개 또는 모델들의 시퀀스
 - `criterion`:  최적화할 loss 함수(선택)
 - `log`:  "gradients", "parameters", "all" 중 로깅 옵션. None이면 로깅 비활성화 (기본값: "gradients")
 - `log_freq`:  그레이디언트와 파라미터를 로깅할 주기(배치 단위, 기본 1000)
 - `idx`:  여러 모델을 `wandb.watch`로 추적할 때 쓸 인덱스(기본 None)
 - `log_graph`:  모델의 computational graph를 로깅할지 여부(기본 False)



**예외 발생:**
 ValueError:  `wandb.init()`이 호출되지 않았거나 모델이 `torch.nn.Module` 인스턴스가 아닌 경우
