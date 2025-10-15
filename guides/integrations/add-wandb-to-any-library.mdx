---
title: Add wandb to any library
menu:
  default:
    identifier: ko-guides-integrations-add-wandb-to-any-library
    parent: integrations
weight: 10
---

## 모든 라이브러리에 wandb 추가하기

이 가이드는 강력한 Experiment Tracking, GPU 및 시스템 모니터링, 모델 체크포인팅 등을 사용자 라이브러리에서 활용할 수 있도록 W&B를 Python 라이브러리에 통합하는 모범 사례를 제공합니다.

{{% alert %}}
W&B 사용법을 배우는 중이라면 계속 읽기 전에 [실험 추적]({{< relref path="/guides/models/track" lang="ko" >}})과 같은 이 문서의 다른 W&B 가이드를 살펴보는 것이 좋습니다.
{{% /alert %}}

다음은 작업 중인 코드베이스가 단일 Python 트레이닝 스크립트 또는 Jupyter 노트북보다 복잡할 때 유용한 팁과 모범 사례입니다. 다루는 주제는 다음과 같습니다.

* 설정 요구 사항
* 사용자 로그인
* wandb Run 시작하기
* Run 설정 정의하기
* W&B에 로깅하기
* 분산 트레이닝
* 모델 체크포인팅 등
* 하이퍼파라미터 튜닝
* 고급 인테그레이션

### 설정 요구 사항

시작하기 전에 라이브러리의 종속성에 W&B가 필요한지 여부를 결정하세요.

#### 설치 시 W&B 필요

W&B Python 라이브러리(`wandb`)를 종속성 파일에 추가합니다(예: `requirements.txt` 파일):

```python
torch==1.8.0
...
wandb==0.13.*
```

#### 설치 시 W&B를 선택 사항으로 만들기

W&B SDK(`wandb`)를 선택 사항으로 만드는 방법에는 두 가지가 있습니다.

A. 사용자가 수동으로 설치하지 않고 `wandb` 기능을 사용하려고 할 때 오류를 발생시키고 적절한 오류 메시지를 표시합니다.

```python
try:
    import wandb
except ImportError:
    raise ImportError(
        "You are trying to use wandb which is not currently installed."
        "Please install it using pip install wandb"
    )
```

B. Python 패키지를 빌드하는 경우 `pyproject.toml` 파일에 `wandb`를 선택적 종속성으로 추가합니다.

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

#### API 키 만들기

API 키는 클라이언트 또는 머신을 W&B에 인증합니다. 사용자 프로필에서 API 키를 생성할 수 있습니다.

{{% alert %}}
보다 간소화된 접근 방식을 위해 [https://wandb.ai/authorize](https://wandb.ai/authorize)로 직접 이동하여 API 키를 생성할 수 있습니다. 표시된 API 키를 복사하여 비밀번호 관리자와 같은 안전한 위치에 저장하세요.
{{% /alert %}}

1. 오른쪽 상단 모서리에 있는 사용자 프로필 아이콘을 클릭합니다.
2. **사용자 설정**을 선택한 다음 **API 키** 섹션으로 스크롤합니다.
3. **표시**를 클릭합니다. 표시된 API 키를 복사합니다. API 키를 숨기려면 페이지를 새로 고침합니다.

#### `wandb` 라이브러리 설치 및 로그인

로컬에서 `wandb` 라이브러리를 설치하고 로그인하려면:

{{< tabpane text=true >}}
{{% tab header="커맨드라인" value="cli" %}}

1. API 키로 `WANDB_API_KEY` [환경 변수]({{< relref path="/guides/models/track/environment-variables.md" lang="ko" >}})를 설정합니다.

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

2. `wandb` 라이브러리를 설치하고 로그인합니다.

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

{{% tab header="Python 노트북" value="python-notebook" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

사용자가 위에 언급된 단계를 따르지 않고 처음으로 wandb를 사용하는 경우 스크립트에서 `wandb.init`을 호출할 때 자동으로 로그인하라는 메시지가 표시됩니다.

### Run 시작하기

W&B Run은 W&B에서 로깅하는 계산 단위입니다. 일반적으로 트레이닝 실험당 단일 W&B Run을 연결합니다.

코드 내에서 W&B를 초기화하고 Run을 시작합니다.

```python
run = wandb.init()
```

선택적으로, 코드에서 `wandb_project`와 같은 파라미터와 함께 프로젝트 이름을 제공하거나 사용자 이름 또는 팀 이름(예: 엔터티 파라미터에 대한 `wandb_entity`)과 함께 프로젝트 이름을 제공하도록 할 수 있습니다.

```python
run = wandb.init(project=wandb_project, entity=wandb_entity)
```

Run을 완료하려면 `run.finish()`를 호출해야 합니다. 이것이 통합 디자인에 적합한 경우 Run을 컨텍스트 관리자로 사용합니다.

```python
# 이 블록이 종료되면 run.finish()를 자동으로 호출합니다.
# 예외로 인해 종료되면 run.finish(exit_code=1)을 사용하여 Run을 실패로 표시합니다.
with wandb.init() as run:
    ...
```

#### `wandb.init`을 언제 호출해야 할까요?

라이브러리는 가능한 한 빨리 W&B Run을 만들어야 합니다. 오류 메시지를 포함하여 콘솔의 모든 출력이 W&B Run의 일부로 로깅되기 때문입니다. 이렇게 하면 디버깅이 더 쉬워집니다.

#### `wandb`를 선택적 종속성으로 사용하기

사용자가 라이브러리를 사용할 때 `wandb`를 선택 사항으로 만들려면 다음 중 하나를 수행할 수 있습니다.

* 다음과 같은 `wandb` 플래그를 정의합니다.

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

* 또는 `wandb.init`에서 `wandb`를 `disabled`로 설정합니다.

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

* 또는 `wandb`를 오프라인으로 설정합니다. 이렇게 하면 여전히 `wandb`가 실행되지만 인터넷을 통해 W&B에 다시 통신하려고 시도하지 않습니다.

{{< tabpane text=true >}}

{{% tab header="환경 변수" value="environment" %}}

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

### Run 설정 정의하기
`wandb` Run 설정으로 W&B Run을 만들 때 모델, 데이터셋 등에 대한 메타데이터를 제공할 수 있습니다. 이 정보를 사용하여 다양한 Experiments를 비교하고 주요 차이점을 빠르게 이해할 수 있습니다.

{{< img src="/images/integrations/integrations_add_any_lib_runs_page.png" alt="W&B Runs table" >}}

로깅할 수 있는 일반적인 구성 파라미터는 다음과 같습니다.

* 모델 이름, 버전, 아키텍처 파라미터 등
* 데이터셋 이름, 버전, 트레이닝/검증 예제 수 등
* 학습률, 배치 크기, 옵티마이저 등과 같은 트레이닝 파라미터

다음 코드 조각은 구성을 로깅하는 방법을 보여줍니다.

```python
config = {"batch_size": 32, ...}
wandb.init(..., config=config)
```

#### Run 설정 업데이트하기
`run.config.update`를 사용하여 구성을 업데이트합니다. 구성 사전이 정의된 후에 파라미터를 가져올 때 구성 사전을 업데이트하는 것이 유용합니다. 예를 들어 모델이 인스턴스화된 후에 모델의 파라미터를 추가할 수 있습니다.

```python
run.config.update({"model_parameters": 3500})
```

구성 파일을 정의하는 방법에 대한 자세한 내용은 [실험 구성]({{< relref path="/guides/models/track/config" lang="ko" >}})을 참조하세요.

### W&B에 로깅하기

#### 메트릭 로깅

키 값이 메트릭 이름인 사전을 만듭니다. 이 사전 오브젝트를 [`run.log`]({{< relref path="/guides/models/track/log" lang="ko" >}})에 전달합니다.

```python
for epoch in range(NUM_EPOCHS):
    for input, ground_truth in data:
        prediction = model(input)
        loss = loss_fn(prediction, ground_truth)
        metrics = { "loss": loss }
        run.log(metrics)
```

메트릭이 많은 경우 메트릭 이름에 `train/...` 및 `val/...`와 같은 접두사를 사용하여 UI에서 자동으로 그룹화할 수 있습니다. 이렇게 하면 트레이닝 및 검증 메트릭 또는 분리하려는 다른 메트릭 유형에 대해 W&B Workspace에 별도의 섹션이 생성됩니다.

```python
metrics = {
    "train/loss": 0.4,
    "train/learning_rate": 0.4,
    "val/loss": 0.5,
    "val/accuracy": 0.7
}
run.log(metrics)
```

{{< img src="/images/integrations/integrations_add_any_lib_log.png" alt="A W&B Workspace with 2 separate sections" >}}

[`run.log`에 대해 자세히 알아보세요]({{< relref path="/guides/models/track/log" lang="ko" >}}).

#### x축 정렬 오류 방지

동일한 트레이닝 단계에 대해 `run.log`를 여러 번 호출하는 경우 wandb SDK는 `run.log`를 호출할 때마다 내부 단계 카운터를 증가시킵니다. 이 카운터는 트레이닝 루프의 트레이닝 단계와 정렬되지 않을 수 있습니다.

이러한 상황을 피하려면 `wandb.init`을 호출한 직후에 `run.define_metric`을 사용하여 x축 단계를 명시적으로 한 번 정의하세요.

```python
with wandb.init(...) as run:
    run.define_metric("*", step_metric="global_step")
```

글로브 패턴 `*`는 모든 메트릭이 차트에서 `global_step`을 x축으로 사용함을 의미합니다. 특정 메트릭만 `global_step`에 대해 로깅하려면 대신 지정할 수 있습니다.

```python
run.define_metric("train/loss", step_metric="global_step")
```

이제 `run.log`를 호출할 때마다 메트릭, `step` 메트릭 및 `global_step`을 로깅합니다.

```python
for step, (input, ground_truth) in enumerate(data):
    ...
    run.log({"global_step": step, "train/loss": 0.1})
    run.log({"global_step": step, "eval/loss": 0.2})
```

예를 들어 검증 루프 중에 "global_step"을 사용할 수 없는 경우 독립 단계 변수에 액세스할 수 없는 경우 "global_step"에 대해 이전에 로깅된 값이 wandb에서 자동으로 사용됩니다. 이 경우 메트릭에 필요한 경우 정의되도록 메트릭에 대한 초기 값을 로깅해야 합니다.

#### 이미지, 테이블, 오디오 등 로깅

메트릭 외에도 플롯, 히스토그램, 테이블, 텍스트, 이미지, 비디오, 오디오, 3D 등과 같은 미디어를 로깅할 수 있습니다.

데이터 로깅 시 고려해야 할 사항은 다음과 같습니다.

* 메트릭을 얼마나 자주 로깅해야 할까요? 선택 사항이어야 할까요?
* 시각화하는 데 어떤 유형의 데이터가 도움이 될 수 있을까요?
  * 이미지의 경우 시간 경과에 따른 진화를 확인하기 위해 샘플 예측, 분할 마스크 등을 로깅할 수 있습니다.
  * 텍스트의 경우 나중에 탐색할 수 있도록 샘플 예측 테이블을 로깅할 수 있습니다.

미디어, 오브젝트, 플롯 등을 [로깅하는 방법에 대해 자세히 알아보세요]({{< relref path="/guides/models/track/log" lang="ko" >}}).

### 분산 트레이닝

분산 환경을 지원하는 프레임워크의 경우 다음 워크플로우 중 하나를 적용할 수 있습니다.

* "메인" 프로세스를 감지하고 거기에서만 `wandb`를 사용합니다. 다른 프로세스에서 오는 필요한 데이터는 먼저 메인 프로세스로 라우팅되어야 합니다. (이 워크플로우가 권장됩니다).
* 모든 프로세스에서 `wandb`를 호출하고 모두 동일한 고유 `group` 이름을 지정하여 자동으로 그룹화합니다.

자세한 내용은 [분산 트레이닝 실험 로깅]({{< relref path="/guides/models/track/log/distributed-training.md" lang="ko" >}})을 참조하세요.

### 모델 체크포인트 등 로깅

프레임워크에서 모델 또는 데이터셋을 사용하거나 생성하는 경우 전체 추적성을 위해 로깅하고 wandb가 W&B Artifacts를 통해 전체 파이프라인을 자동으로 모니터링하도록 할 수 있습니다.

{{< img src="/images/integrations/integrations_add_any_lib_dag.png" alt="Stored Datasets and Model Checkpoints in W&B" >}}

Artifacts를 사용할 때 사용자가 다음을 정의하도록 하는 것이 유용하지만 필수는 아닙니다.

* 모델 체크포인트 또는 데이터셋을 로깅하는 기능(선택 사항으로 만들려는 경우).
* 입력으로 사용되는 아티팩트의 경로/참조(있는 경우). 예를 들어 `user/project/artifact`입니다.
* Artifacts 로깅 빈도.

#### 모델 체크포인트 로깅

모델 체크포인트를 W&B에 로깅할 수 있습니다. 고유한 `wandb` Run ID를 활용하여 출력 모델 체크포인트 이름을 지정하여 Runs 간에 차별화하는 것이 유용합니다. 유용한 메타데이터를 추가할 수도 있습니다. 또한 아래와 같이 각 모델에 에일리어스를 추가할 수도 있습니다.

```python
metadata = {"eval/accuracy": 0.8, "train/steps": 800}

artifact = wandb.Artifact(
                name=f"model-{run.id}",
                metadata=metadata,
                type="model"
                )
artifact.add_dir("output_model") # 모델 가중치가 저장되는 로컬 디렉토리

aliases = ["best", "epoch_10"]
run.log_artifact(artifact, aliases=aliases)
```

사용자 지정 에일리어스를 만드는 방법에 대한 자세한 내용은 [사용자 지정 에일리어스 만들기]({{< relref path="/guides/core/artifacts/create-a-custom-alias/" lang="ko" >}})를 참조하세요.

출력 Artifacts를 임의의 빈도(예: 모든 에포크, 500단계마다 등)로 로깅할 수 있으며 자동으로 버전이 지정됩니다.

#### 사전 학습된 모델 또는 데이터셋 로깅 및 추적

사전 학습된 모델 또는 데이터셋과 같이 트레이닝에 입력으로 사용되는 아티팩트를 로깅할 수 있습니다. 다음 코드 조각은 Artifact를 로깅하고 위의 그래프에 표시된 대로 진행 중인 Run에 입력으로 추가하는 방법을 보여줍니다.

```python
artifact_input_data = wandb.Artifact(name="flowers", type="dataset")
artifact_input_data.add_file("flowers.npy")
run.use_artifact(artifact_input_data)
```

#### 아티팩트 다운로드

Artifact(데이터셋, 모델 등)를 다시 사용하면 `wandb`가 로컬에 복사본을 다운로드합니다(그리고 캐시합니다).

```python
artifact = run.use_artifact("user/project/artifact:latest")
local_path = artifact.download("./tmp")
```

Artifacts는 W&B의 Artifacts 섹션에서 찾을 수 있으며 자동으로 생성된 에일리어스(`latest`, `v2`, `v3`) 또는 로깅할 때 수동으로 생성된 에일리어스(`best_accuracy` 등)로 참조할 수 있습니다.

분산 환경 또는 간단한 추론과 같이(`wandb.init`을 통해) `wandb` Run을 만들지 않고 Artifact를 다운로드하려면 대신 [wandb API]({{< relref path="/ref/python/public-api" lang="ko" >}})로 아티팩트를 참조할 수 있습니다.

```python
artifact = wandb.Api().artifact("user/project/artifact:latest")
local_path = artifact.download()
```

자세한 내용은 [Artifacts 다운로드 및 사용]({{< relref path="/guides/core/artifacts/download-and-use-an-artifact" lang="ko" >}})을 참조하세요.

### 하이퍼파라미터 튜닝

라이브러리에서 W&B 하이퍼파라미터 튜닝을 활용하려면 [W&B Sweeps]({{< relref path="/guides/models/sweeps/" lang="ko" >}})를 라이브러리에 추가할 수도 있습니다.

### 고급 인테그레이션

다음 인테그레이션에서 고급 W&B 인테그레이션이 어떻게 보이는지 확인할 수도 있습니다. 대부분의 인테그레이션은 이만큼 복잡하지 않습니다.

* [Hugging Face Transformers `WandbCallback`](https://github.com/huggingface/transformers/blob/49629e7ba8ef68476e08b671d6fc71288c2f16f1/src/transformers/integrations.py#L639)
* [PyTorch Lightning `WandbLogger`](https://github.com/Lightning-AI/lightning/blob/18f7f2d3958fb60fcb17b4cb69594530e83c217f/src/pytorch_lightning/loggers/wandb.py#L53)
