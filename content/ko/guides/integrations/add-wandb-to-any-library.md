---
title: 모든 라이브러리에 wandb 추가하기
menu:
  default:
    identifier: ko-guides-integrations-add-wandb-to-any-library
    parent: integrations
weight: 10
---

## wandb를 모든 라이브러리에 추가하기

이 가이드는 강력한 Experiment Tracking, GPU 및 시스템 모니터링, 모델 체크포인팅 등 다양한 기능을 여러분의 Python 라이브러리에서 활용하기 위한 wandb 연동 모범 사례를 안내합니다.

{{% alert %}}
wandb 사용법을 익히는 중이라면, 이 문서의 다른 W&B 가이드들(예: [Experiment Tracking]({{< relref path="/guides/models/track" lang="ko" >}}))을 먼저 살펴보시길 추천드립니다.
{{% /alert %}}

아래에서는 단일 Python 트레이닝 스크립트나 Jupyter 노트북보다 더 복잡한 코드베이스를 다룰 때 적용할 수 있는 주요 팁과 모범 사례들을 다룹니다. 다루는 주제는 다음과 같습니다:

* 설치 요구 사항
* 사용자 로그인
* wandb Run 시작하기
* Run Config 정의하기
* W&B로 로그 남기기
* 분산 트레이닝
* 모델 체크포인팅 및 기타
* 하이퍼파라미터 튜닝
* 고급 인테그레이션

### 설치 요구 사항

시작하기 전에, 라이브러리 의존성에 W&B 포함 여부를 결정해야 합니다:

#### 설치 시 W&B 필수로 지정하기

W&B Python 라이브러리(`wandb`)를 의존성 파일에 추가하세요. 예를 들어, `requirements.txt` 파일에 아래와 같이 입력합니다:

```python
torch==1.8.0 
...
wandb==0.13.*
```

#### 설치 시 W&B 선택적으로 지정하기

W&B SDK(`wandb`)를 선택적으로 포함시키는 방법은 두 가지가 있습니다:

A. 사용자가 직접 `wandb`를 설치하지 않은 상태에서 해당 기능을 사용하려고 하면 에러를 발생시키고, 적합한 안내 메시지를 출력하기:

```python
try: 
    import wandb 
except ImportError: 
    raise ImportError(
        "wandb가 현재 설치되어 있지 않습니다. "
        "pip install wandb 명령어로 설치해주세요."
    ) 
```

B. Python 패키지 빌드 시 `pyproject.toml` 파일에 `wandb`를 선택적 의존성으로 추가하기:

```toml
[project]
name = "my_awesome_lib"
version = "0.1.0"
dependencies = [
    "torch",
    "sklearn"
]

[project.optional-dependencies]
dev = [
    "wandb"
]
```

### 사용자 로그인

#### API 키 생성하기

API 키는 클라이언트 또는 머신이 W&B에 인증할 때 사용됩니다. 내 사용자 프로필에서 API 키를 생성할 수 있습니다.

{{% alert %}}
더 간편하게는 [W&B 인증 페이지](https://wandb.ai/authorize)에서 즉시 API 키를 생성할 수 있습니다. 표시된 API 키를 복사한 뒤, 비밀번호 관리자와 같이 안전한 위치에 보관하세요.
{{% /alert %}}

1. 오른쪽 상단의 사용자 프로필 아이콘을 클릭합니다.
1. **User Settings**에 들어가서 **API Keys** 섹션까지 스크롤합니다.
1. **Reveal**을 클릭해 API 키를 복사합니다. API 키를 숨기려면 페이지를 새로고침하세요.

#### `wandb` 라이브러리 설치 및 로그인

로컬에서 `wandb` 라이브러리를 설치하고 로그인하려면:

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [환경 변수]({{< relref path="/guides/models/track/environment-variables.md" lang="ko" >}})에 API 키를 등록하세요.

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb` 라이브러리를 설치하고 로그인합니다.

    ```shell
    pip install wandb

    wandb login
    ```

{{% /tab %}}

{{% tab header="Python" value="python" %}}

```bash
pip install wandb
```
```python
import wandb
wandb.login()
```

{{% /tab %}}

{{% tab header="Python notebook" value="python-notebook" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

위의 과정을 거치지 않고 처음 wandb를 사용할 경우, 스크립트에서 `wandb.init`을 호출하면 자동으로 로그인 안내가 표시됩니다.

### Run 시작하기

W&B Run은 W&B가 기록하는 연산 단위입니다. 일반적으로 하나의 트레이닝 experiment 당 하나의 Run을 연결하게 됩니다.

wandb를 초기화하고 Run을 시작하려면 코드에서 다음과 같이 작성합니다:

```python
run = wandb.init()
```

프로젝트 이름은 파라미터로 전달하거나, 사용자가 직접 선택할 수 있습니다. 예를 들어, 코드에 `wandb_project`와 사용자명 또는 팀명에 해당하는 `wandb_entity`도 함께 넘겨줄 수 있습니다:

```python
run = wandb.init(project=wandb_project, entity=wandb_entity)
```

Run을 종료하려면 반드시 `run.finish()`를 호출해야 합니다. 인테그레이션 구조가 이에 맞는 경우, context manager를 활용하면 다음과 같이 종료를 자동으로 처리할 수 있습니다:

```python
# 이 블록을 벗어날 때 run.finish()가 자동 호출됩니다.
# 예외로 빠져나갈 경우 run.finish(exit_code=1)로
# run을 실패로 마칩니다.
with wandb.init() as run:
    ...
```

#### 언제 `wandb.init`을 호출해야 할까요?

콘솔에 출력되는 모든 메시지(에러 메시지 포함)가 W&B Run에 함께 기록됩니다. 그러므로 Run은 가능한 한 빨리 생성하는 것이 디버깅에 도움이 됩니다.

#### `wandb`를 선택적 의존성으로 사용하기

사용자가 wandb 유무를 선택하도록 하려면, 아래와 같이 할 수 있습니다:

* 예를 들어 `wandb` 플래그를 추가하여 사용:

{{< tabpane text=true >}}

{{% tab header="Python" value="python" %}}

```python
trainer = my_trainer(..., use_wandb=True)
```
{{% /tab %}}

{{% tab header="Bash" value="bash" %}}

```bash
python train.py ... --use-wandb
```
{{% /tab %}}

{{< /tabpane >}}

* 또는, `wandb.init` 호출 시 `disabled` 모드로 비활성화할 수도 있습니다:

{{< tabpane text=true >}}

{{% tab header="Python" value="python" %}}

```python
wandb.init(mode="disabled")
```
{{% /tab %}}

{{% tab header="Bash" value="bash" %}}

```bash
export WANDB_MODE=disabled
```

또는

```bash
wandb disabled
```
{{% /tab %}}

{{< /tabpane >}}

* 또는, `wandb`를 오프라인 모드로 설정할 수도 있습니다. 이때도 wandb는 실행되지만 인터넷을 통한 서버와의 통신만 중단됩니다:

{{< tabpane text=true >}}

{{% tab header="Environment Variable" value="environment" %}}

```bash
export WANDB_MODE=offline
```

또는

```python
os.environ['WANDB_MODE'] = 'offline'
```
{{% /tab %}}

{{% tab header="Bash" value="bash" %}}

```bash
wandb offline
```
{{% /tab %}}

{{< /tabpane >}}

### Run config 정의하기
`wandb` run config를 사용하면 모델, 데이터셋 등에 대한 메타데이터를 W&B Run 생성 시 함께 기록할 수 있습니다. 이 정보는 여러 experiment을 비교할 때 주요 차이점을 빠르게 파악하는 데 유용합니다.

{{< img src="/images/integrations/integrations_add_any_lib_runs_page.png" alt="W&B Runs table" >}}

주로 기록할 수 있는 config 파라미터는 다음과 같습니다:

* 모델 이름, 버전, 아키텍처 파라미터 등
* 데이터셋 이름, 버전, 학습/검증 데이터 수 등
* 트레이닝 파라미터(learning rate, 배치 크기, 옵티마이저 등)

다음 코드조각은 config를 로그로 남기는 예시입니다:

```python
config = {"batch_size": 32, ...}
wandb.init(..., config=config)
```

#### Run config 업데이트하기
`wandb.Run.config.update`를 이용해 config를 업데이트할 수 있습니다. 이는 사전 정의 이후에 파라미터를 얻었을 때 편리합니다. 예를 들어, 모델 인스턴스 생성 후 파라미터 정보를 추가하고 싶을 때 유용합니다.

```python
run.config.update({"model_parameters": 3500})
```

config 파일 정의법에 관한 자세한 내용은 [실험 구성하기]({{< relref path="/guides/models/track/config" lang="ko" >}})를 참고하세요.

### W&B로 로그 남기기

#### 메트릭 기록하기

사전 형태로 메트릭의 이름을 key로 하여 객체를 생성합니다. 이 dictionary 오브젝트를 [`run.log`]({{< relref path="/guides/models/track/log" lang="ko" >}})에 전달하면 됩니다:

```python
for epoch in range(NUM_EPOCHS):
    for input, ground_truth in data: 
        prediction = model(input) 
        loss = loss_fn(prediction, ground_truth) 
        metrics = { "loss": loss } 
        run.log(metrics)
```

메트릭이 많을 경우, 메트릭 이름에 접두사를 붙여서 UI에서 자동으로 그룹화할 수 있습니다(예: `train/...`, `val/...`). 이렇게 하면 W&B Workspace 내에서 트레이닝/검증 메트릭 또는 원하는 분류별로 섹션을 만들 수 있습니다:

```python
metrics = {
    "train/loss": 0.4,
    "train/learning_rate": 0.4,
    "val/loss": 0.5, 
    "val/accuracy": 0.7
}
run.log(metrics)
```

{{< img src="/images/integrations/integrations_add_any_lib_log.png" alt="W&B Workspace" >}}

[`wandb.Run.log()` 레퍼런스]({{< relref path="/guides/models/track/log" lang="ko" >}})도 참고하세요.

#### x-축 불일치 방지하기

동일한 트레이닝 스텝에서 여러 번 `run.log`를 호출하면, wandb SDK가 내부적으로 스텝 카운터를 매번 증가시킵니다. 이 카운터가 실제 트레이닝 루프의 스텝과 맞지 않을 수 있습니다.

이런 상황을 방지하려면, `wandb.init` 호출 직후 한 번만 `run.define_metric`을 이용해 x-축 기준 스텝을 명시하세요:

```python
with wandb.init(...) as run:
    run.define_metric("*", step_metric="global_step")
```

여기서 `*` 패턴은 모든 메트릭에 대해 `global_step`을 x축으로 사용한다는 의미입니다. 특정 메트릭만 적용하고 싶다면 해당 이름만 지정하면 됩니다:

```python
run.define_metric("train/loss", step_metric="global_step")
```

이제, 각각의 `run.log` 호출 때마다 메트릭과 함께 스텝 정보를 함께 기록하세요:

```python
for step, (input, ground_truth) in enumerate(data):
    ...
    run.log({"global_step": step, "train/loss": 0.1})
    run.log({"global_step": step, "eval/loss": 0.2})
```

만약 validation 루프 등에서 "global_step"과 같은 변수에 엑세스할 수 없다면, wandb는 마지막으로 로그된 "global_step" 값을 자동으로 사용합니다. 이런 경우, 해당 metric이 필요할 때 미리 초기 값을 기록해두세요.

#### 이미지, 테이블, 오디오 등 기록하기

메트릭 외에도 플롯, 히스토그램, 테이블, 텍스트, 그리고 이미지, 비디오, 오디오, 3D 등 다양한 미디어를 기록할 수 있습니다.

데이터 로깅 시 고려할 점은:

* 얼마나 자주 메트릭을 기록할지? (선택적으로 할 것인지)
* 데이터 유형에 따라 어떤 시각화가 유용할지?
  * 예를 들어, 이미지는 예측 샘플, 세그멘테이션 마스크 등 다양한 변화를 확인할 수 있습니다.
  * 텍스트의 경우, 예측 결과 샘플을 테이블로 기록해 탐색할 수 있습니다.

미디어, 오브젝트, 플롯 등 다양한 로깅에 대해서는 [로깅 가이드]({{< relref path="/guides/models/track/log" lang="ko" >}})를 참고하세요.

### 분산 트레이닝

분산 환경을 지원하는 프레임워크라면, 다음 중 한 가지 워크플로우를 활용할 수 있습니다:

* "main" 프로세스를 감지해서 해당 프로세스에서만 `wandb`를 사용하고, 필요한 데이터는 다른 프로세스에서 메인으로 전달합니다. (추천 워크플로우)
* 모든 프로세스에서 개별적으로 wandb를 호출하고, 동일한 유니크 `group` 이름을 지정해서 자동 그룹화합니다.

자세한 내용은 [분산 트레이닝 실험 기록하기]({{< relref path="/guides/models/track/log/distributed-training.md" lang="ko" >}})를 참고하세요.

### 모델 체크포인트 및 기타 기록하기

프레임워크에서 모델 또는 데이터셋을 다룬다면, wandb를 이용해 전체 파이프라인까지 추적 가능한 Artifacts로 자동 모니터링할 수 있습니다.

{{< img src="/images/integrations/integrations_add_any_lib_dag.png" alt="W&B에 저장된 Datasets 및 Model Checkpoints" >}}

Artifacts 사용 시, 사용자에게 다음과 같은 항목을 선택적으로 정의하도록 할 수 있습니다:

* 모델 체크포인트 또는 데이터셋 로그 남기기 여부 (옵션으로 만들고 싶은 경우)
* 입력으로 사용된 artifact의 경로/레퍼런스 (예: `user/project/artifact`)
* Artifacts 기록 빈도

#### 모델 체크포인트 기록하기

모델 체크포인트를 W&B에 기록할 수 있습니다. Run 별로 유니크한 wandb Run ID를 이용해 체크포인트 파일명을 지정하고, 메타데이터도 추가할 수 있습니다. 또한, 아래처럼 alias도 지정할 수 있습니다:

```python
metadata = {"eval/accuracy": 0.8, "train/steps": 800} 

artifact = wandb.Artifact(
                name=f"model-{run.id}", 
                metadata=metadata, 
                type="model"
                ) 
artifact.add_dir("output_model") # 모델 가중치가 저장된 로컬 디렉토리

aliases = ["best", "epoch_10"] 
run.log_artifact(artifact, aliases=aliases)
```

커스텀 에일리어스 생성 방법은 [Create a Custom Alias]({{< relref path="/guides/core/artifacts/create-a-custom-alias/" lang="ko" >}})를 참고하세요.

출력 Artifacts는 원하는 모든 빈도(에포크마다, 500스텝마다 등)로 기록 가능하며, 자동으로 버전 관리됩니다.

#### 사전학습된 모델이나 데이터셋 로깅 및 추적

모델 또는 데이터셋 등 트레이닝에 입력으로 쓰이는 아티팩트도 로그로 기록할 수 있습니다. 아래는 Artifacts를 로그로 남기고, 현재 Run의 입력으로 추가하는 예시입니다.

```python
artifact_input_data = wandb.Artifact(name="flowers", type="dataset")
artifact_input_data.add_file("flowers.npy")
run.use_artifact(artifact_input_data)
```

#### 아티팩트 다운로드하기

Artifact(데이터셋, 모델 등)를 재사용할 경우, wandb가 로컬에 사본을 다운로드(캐시)합니다:

```python
artifact = run.use_artifact("user/project/artifact:latest")
local_path = artifact.download("./tmp")
```

Artifacts는 W&B의 Artifacts 섹션에서 찾을 수 있으며, 자동(`latest`, `v2`, `v3`) 또는 수동(`best_accuracy` 등) 에일리어스로 참조할 수 있습니다.

`wandb` run(`wandb.init`) 없이 Artifact만 다운로드(예: 분산 환경, 추론 등)하려면 [wandb API]({{< relref path="/ref/python/public-api/index.md" lang="ko" >}})로 직접 참조하면 됩니다:

```python
artifact = wandb.Api().artifact("user/project/artifact:latest")
local_path = artifact.download()
```

자세한 내용은 [아티팩트 다운받기 및 사용하기]({{< relref path="/guides/core/artifacts/download-and-use-an-artifact" lang="ko" >}})를 참고하세요.

### 하이퍼파라미터 튜닝하기

라이브러리에서 W&B의 하이퍼파라미터 튜닝을 활용하려면 [W&B Sweeps]({{< relref path="/guides/models/sweeps/" lang="ko" >}})를 추가할 수도 있습니다.

### 고급 인테그레이션

아래 예시에서 보다 고급 W&B 인테그레이션 구현 사례를 볼 수 있습니다. 대부분의 인테그레이션은 이 정도로 복잡하지 않습니다:

* [Hugging Face Transformers `WandbCallback`](https://github.com/huggingface/transformers/blob/49629e7ba8ef68476e08b671d6fc71288c2f16f1/src/transformers/integrations.py#L639)
* [PyTorch Lightning `WandbLogger`](https://github.com/Lightning-AI/lightning/blob/18f7f2d3958fb60fcb17b4cb69594530e83c217f/src/pytorch_lightning/loggers/wandb.py#L53)