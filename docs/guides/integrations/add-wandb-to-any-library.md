---
title: Add wandb to any library
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

이 가이드는 여러분의 파이썬 라이브러리에 W&B를 통합하여 강력한 실험 추적, GPU 및 시스템 모니터링, 모델 체크포인팅 등을 구현할 수 있는 모범 사례를 제공합니다.

:::note
W&B 사용법을 아직 배우고 있다면, 더 읽기 전에 이 문서의 다른 W&B 가이드, 예를 들어 [Experiment Tracking](/guides/track)을 탐색하는 것을 권장합니다.
:::

아래에서는 작업 중인 코드베이스가 단일 파이썬 트레이닝 스크립트나 주피터 노트북보다 복잡할 때 가장 좋은 팁과 모범 사례를 다룹니다. 다루는 주제는 다음과 같습니다:

* 요구 사항 설정
* 사용자 로그인
* wandb Run 시작하기
* Run 설정 정의
* W&B로 로깅
* 분산 트레이닝
* 모델 체크포인팅 및 기타
* 하이퍼-파라미터 튜닝
* 고급 인테그레이션

### 요구 사항 설정

시작하기 전에, W&B를 여러분의 라이브러리의 종속성으로 요구할지 여부를 결정해야 합니다:

#### 설치 시 W&B 필수

W&B 파이썬 라이브러리 (`wandb`)를 종속성 파일에 추가합니다. 예를 들어, `requirements.txt` 파일에 아래와 같이 추가할 수 있습니다:

```python
torch==1.8.0 
...
wandb==0.13.*
```

#### 설치 시 W&B 선택적

W&B SDK (`wandb`)를 선택적으로 만드는 두 가지 방법이 있습니다:

A. 사용자가 수동으로 설치하지 않고 `wandb` 기능을 사용하려고 할 때 오류를 발생시키고 적절한 오류 메시지를 표시합니다:

```python
try: 
    import wandb 
except ImportError: 
    raise ImportError(
        “You are trying to use wandb which is not currently installed”
        “Please install it using pip install wandb”
    ) 
```

B. 파이썬 패키지를 빌드하는 경우, 선택적 종속성으로 `wandb`를 `pyproject.toml` 파일에 추가하십시오.

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

사용자가 W&B에 로그인하는 몇 가지 방법이 있습니다:

<Tabs
  defaultValue="bash"
  values={[
    {label: 'Bash', value: 'bash'},
    {label: 'Notebook', value: 'notebook'},
    {label: 'Environment Variable', value: 'environment'},
  ]}>
  <TabItem value="bash">
터미널에서 bash 코맨드로 W&B에 로그인

```bash
wandb login $MY_WANDB_KEY
```
  </TabItem>
  <TabItem value="notebook">
주피터 또는 Colab 노트북에 있는 경우, 다음과 같이 W&B에 로그인

```python
import wandb
wandb.login
```
  </TabItem>
  <TabItem value="environment">

API 키에 대한 [W&B 환경 변수](../track/environment-variables.md) 설정

```bash
export WANDB_API_KEY=$YOUR_API_KEY
```

또는

```
os.environ['WANDB_API_KEY'] = "abc123..."
```
  </TabItem>
</Tabs>

사용자가 위에 언급된 단계를 따르지 않고 처음으로 wandb를 사용할 경우, 여러분의 스크립트가 `wandb.init`을 호출하면 자동으로 로그인하라는 메시지가 나타납니다.

### wandb Run 시작하기

W&B Run은 W&B에 의해 기록된 작업의 단위입니다. 일반적으로 트레이닝 실험당 단일 W&B Run을 연결합니다.

W&B를 초기화하고 `wandb.init()`을 사용하여 코드 내에서 Run을 시작하십시오:

```python
wandb.init()
```

옵션으로 프로젝트 이름을 제공하거나 코드 내에서 사용자 또는 팀 이름과 함께 `wandb_project`와 같은 파라미터를 사용하여 사용자가 직접 설정할 수 있습니다:

```python
wandb.init(project=wandb_project, entity=wandb_entity)
```

#### `wandb.init`을 어디에 배치해야 하는가?

여러분의 라이브러리는 가능한 한 빨리 W&B Run을 생성해야 합니다. 콘솔의 모든 출력, 오류 메시지를 포함하여 W&B Run의 일부로 기록되기 때문입니다. 이는 디버깅을 용이하게 합니다.

#### `wandb`를 선택 사항으로 설정하여 라이브러리 실행

사용자가 라이브러리를 사용할 때 `wandb`를 선택 사항으로 설정하려면, 다음 중 하나를 수행할 수 있습니다:

* `wandb` 플래그를 정의합니다, 예를 들어:

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python', value: 'python'},
    {label: 'Bash', value: 'bash'},
  ]}>
  <TabItem value="python">

```python
trainer = my_trainer(..., use_wandb=True)
```
  </TabItem>
  <TabItem value="bash">

```bash
python train.py ... --use-wandb
```
  </TabItem>
</Tabs>

* 또는, `wandb.init`에서 `wandb`를 비활성화로 설정합니다

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python', value: 'python'},
    {label: 'Bash', value: 'bash'},
  ]}>
  <TabItem value="python">

```python
wandb.init(mode=“disabled”)
```
  </TabItem>
  <TabItem value="bash">

```bash
export WANDB_MODE=disabled
```
또는

```bash
wandb disabled
```
  </TabItem>
</Tabs>

* 또는, `wandb`를 오프라인으로 설정합니다. 이는 여전히 `wandb`가 실행되지만 W&B와의 인터넷 통신을 시도하지 않음을 의미합니다.

<Tabs
  defaultValue="environment"
  values={[
    {label: 'Environment Variable', value: 'environment'},
    {label: 'Bash', value: 'bash'},
  ]}>
  <TabItem value="environment">

```bash
export WANDB_MODE=offline
```

또는

```python
os.environ['WANDB_MODE'] = 'offline'
```
  </TabItem>
  <TabItem value="bash">

```
wandb offline
```
  </TabItem>
</Tabs>

### wandb Run 설정 정의

`wandb` run 설정을 사용하여 W&B Run을 생성할 때 여러분의 모델, 데이터셋 등과 관련된 메타데이터를 제공할 수 있습니다. 이를 사용하여 다른 실험을 비교하고 주요 차이점을 빠르게 이해할 수 있습니다.

![W&B Runs 테이블](/images/integrations/integrations_add_any_lib_runs_page.png)

일반적인 설정 파라미터로 로그할 수 있는 항목은 다음과 같습니다:

* 모델 이름, 버전, 아키텍처 파라미터 등
* 데이터셋 이름, 버전, 학습/검증 예제 수 등
* 트레이닝 파라미터 예: 학습률, 배치 크기, 옵티마이저 등

다음 코드조각은 설정을 로그하는 방법을 보여줍니다:

```python
config = {“batch_size”:32, …}
wandb.init(…, config=config)
```

#### wandb 설정 업데이트하기

설정을 업데이트하려면 `wandb.config.update`를 사용하십시오. 설정 사전이 정의된 후 파라미터가 취득되는 경우, 예를 들어 모델이 인스턴스화된 후 모델의 파라미터를 추가하려는 경우, 유용하게 설정을 업데이트할 수 있습니다.

```python
wandb.config.update({“model_parameters” = 3500})
```

설정 파일 정의 방법에 대한 자세한 내용은 [Configure Experiments with wandb.config](/guides/track/config)를 참조하세요.

### W&B로 로깅

#### 메트릭 로그하기

메트릭의 이름을 키 값으로 하는 사전을 생성합니다. 이 사전 오브젝트를 [`wandb.log`](/guides/track/log)에 전달합니다:

```python
for epoch in range(NUM_EPOCHS):
    for input, ground_truth in data: 
        prediction = model(input) 
        loss = loss_fn(prediction, ground_truth) 
        metrics = { “loss”: loss } 
        wandb.log(metrics)
```

많은 메트릭이 있는 경우, 메트릭 이름에 접두사를 사용하여 UI에서 자동으로 그룹화할 수 있습니다. 예를 들어 `train/...` 및 `val/...` 등을 사용하여 W&B 워크스페이스에서 트레이닝 및 검증 메트릭 또는 다른 메트릭 유형을 별도의 섹션으로 분리할 수 있습니다.

```python
metrics = {
    “train/loss”: 0.4,
    “train/learning_rate”: 0.4,
    “val/loss”: 0.5, 
    “val/accuracy”: 0.7
}
wandb.log(metrics)
```

![2개의 별도 섹션이 있는 W&B 워크스페이스](/images/integrations/integrations_add_any_lib_log.png)

`wandb.log`에 대한 자세한 내용은 [Log Data with wandb.log](/guides/track/log)를 참조하세요.

#### x축 불일치 방지

때로는 동일한 트레이닝 단계에 대해 여러 번의 `wandb.log` 호출을 수행해야 할 수 있습니다. wandb SDK에는 `wandb.log` 호출이 이루어질 때마다 증가하는 자체 내부 단계 카운터가 있습니다. 이는 wandb 로그 카운터가 트레이닝 루프의 단계와 일치하지 않을 가능성이 있음을 의미합니다.

아래 예제의 첫 번째 패스에서 `train/loss`에 대한 내부 `wandb` 단계는 0이 되고, `eval/loss`에 대한 `wandb` 단계는 1이 됩니다. 다음 패스에서는 `train/loss`는 2가 되고, `eval/loss` wandb 단계는 3이 됩니다, 등등.

```python
for input, ground_truth in data:
    ...
    wandb.log(“train/loss”: 0.1)  
    wandb.log(“eval/loss”: 0.2)
```

이를 방지하기 위해서는 x축 단계를 명시적으로 정의하는 것을 권장합니다. `wandb.define_metric`으로 x축을 정의할 수 있으며, 이는 `wandb.init`이 호출된 후 한 번만 수행하면 됩니다:

```
wandb.init(...)
wandb.define_metric("*", step_metric="global_step")
```

glob 패턴 "\*"은 모든 메트릭이 차트에서 x축으로 "global_step"을 사용함을 의미합니다. 특정 메트릭만 "global_step"에 대해 기록하려면 대신 지정할 수 있습니다:

```
wandb.define_metric("train/loss", step_metric="global_step")
```

`wandb.define_metric`을 호출했으므로 `wandb.log`를 호출할 때마다 메트릭뿐만 아니라 `step_metric`, "global_step"도 로그해야 합니다:

```python
for step, (input, ground_truth) in enumerate(data):
    ...
    wandb.log({“global_step”: step, “train/loss”: 0.1})
    wandb.log({“global_step”: step, “eval/loss”: 0.2})
```

독립적인 단계 변수에 엑세스할 수 없는 경우, 예를 들어 검증 루프에서 "global_step"이 제공되지 않는 경우, wandb는 "global_step"에 대해 이전에 기록된 값을 자동으로 사용합니다. 이러한 경우 메트릭에 대한 초기 값을 기록하여 필요한 경우 정의되도록 하십시오.

#### 이미지, 테이블, 텍스트, 오디오 및 더 많은 미디어 로그

메트릭 외에도 플롯, 히스토그램, 테이블, 텍스트 및 이미지, 비디오, 오디오, 3D 등과 같은 미디어를 로그할 수 있습니다.

데이터 로깅 시 고려해야 할 몇 가지 사항은 다음과 같습니다:

* 메트릭은 얼마나 자주 로그되어야 하는가? 선택적이어야 하는가?
* 시각화에 도움이 될 수 있는 데이터 유형은 무엇인가?
  * 이미지의 경우, 시간 경과에 따른 진화를 보기 위해 샘플 예측값, 분할 마스크 등을 로그할 수 있습니다.
  * 텍스트의 경우, 나중에 탐색할 샘플 예측값의 테이블을 로그할 수 있습니다.

미디어, 오브젝트, 플롯 등을 로깅하는 전체 가이드는 [Log Data with wandb.log](/guides/track/log)를 참조하세요.

### 분산 트레이닝

분산 환경을 지원하는 프레임워크의 경우, 다음 워크플로우 중 하나를 적응할 수 있습니다:

* 어떤 것이 "주" 프로세스인지 감지하고 `wandb`는 거기서만 사용하십시오. 다른 프로세스에서 오는 필요한 데이터는 먼저 주 프로세스로 라우팅해야 합니다. (이 워크플로우가 권장됩니다).
* 모든 프로세스에서 `wandb`를 호출하고 동일한 고유 `group` 이름을 부여하여 자동으로 그룹화합니다.

자세한 내용은 [Log Distributed Training Experiments](../track/log/distributed-training.md)를 참조하세요.

### 모델 체크포인트 및 기타 로깅

프레임워크가 모델 또는 데이터셋을 사용하거나 생성하는 경우, W&B Artifacts를 통해 전체 파이프라인을 자동으로 모니터링하고 wandb로 모델과 데이터셋을 기록할 수 있습니다.

![W&B에 저장된 데이터셋 및 모델 체크포인트](/images/integrations/integrations_add_any_lib_dag.png)

Artifacts를 사용할 때, 사용자에게 다음을 정의하도록 허용하는 것이 유용할 수 있지만 필수는 아닙니다:

* 모델 체크포인트 또는 데이터셋 기록 기능(선택 사항으로 만들고자 하는 경우)
* 입력으로 사용되는 아티팩트의 경로/참조. 예를 들어 "user/project/artifact"
* Artifacts 기록 주기

#### 모델 체크포인트 로그

W&B에 모델 체크포인트를 로그할 수 있습니다. wandb의 고유 Run ID를 활용하여 Run 간에 구별할 수 있도록 출력 모델 체크포인트에 이름을 지정하는 것이 유용합니다. 또한 유용한 메타데이터를 추가할 수 있습니다. 아래에 표시된 것처럼 각 모델에 에일리어스를 추가할 수도 있습니다:

```python
metadata = {“eval/accuracy”: 0.8, “train/steps”: 800} 

artifact = wandb.Artifact(
                name=f”model-{wandb.run.id}”, 
                metadata=metadata, 
                type=”model”
                ) 
artifact.add_dir(“output_model”) # 로컬 디렉토리, 모델 가중치 저장 위치

aliases = [“best”, “epoch_10”] 
wandb.log_artifact(artifact, aliases=aliases)
```

커스텀 에일리어스 생성을 위한 정보는 [Create a Custom Alias](/guides/artifacts/create-a-custom-alias)를 참조하세요.

출력 Artifacts는 어떤 주기로도 (예: 매 에포크, 매 500단계 등) 로그할 수 있으며, 자동으로 버전 관리됩니다.

#### 사전 학습된 모델 또는 데이터셋 로그 및 추적

사전 학습된 모델 또는 데이터셋과 같은 트레이닝에 사용되는 아티팩트를 로그할 수 있습니다. 아래의 코드조각은 아티팩트를 로그하고 진행 중인 Run의 입력으로 추가하는 방법을 보여줍니다:

```python
artifact_input_data = wandb.Artifact(name=”flowers”, type=”dataset”)
artifact_input_data.add_file(“flowers.npy”)
wandb.use_artifact(artifact_input_data)
```

#### W&B 아티팩트 다운로드

아티팩트(데이터셋, 모델 등)를 재사용하고 `wandb`가 로컬로 복사본을 다운로드(및 캐시)하도록 할 수 있습니다:

```python
artifact = wandb.run.use_artifact(“user/project/artifact:latest”)
local_path = artifact.download(“./tmp”)
```

Artifacts는 W&B의 Artifacts 섹션에서 찾을 수 있으며, 자동으로 생성된 에일리어스("latest", "v2", "v3") 또는 로깅 시 수동으로("best_accuracy" 등) 참조할 수 있습니다.

`wandb` run을 생성하지 않고(`wandb.init`을 통해) 아티팩트를 다운로드하려면, 예를 들어 분산 환경에서의 간단한 추론의 경우, 대신 [wandb API](/ref/python/public-api)를 사용하여 아티팩트를 참조할 수 있습니다:

```python
artifact = wandb.Api().artifact(“user/project/artifact:latest”)
local_path = artifact.download()
```

자세한 정보는 [Download and Use Artifacts](/guides/artifacts/download-and-use-an-artifact)를 참조하세요.

### 하이퍼 파라미터 튜닝

W&B의 하이퍼 파라미터 튜닝을 사용하고자 한다면, [W&B Sweeps](/guides/sweeps)를 여러분의 라이브러리에 추가할 수 있습니다.

### 고급 인테그레이션

다음 인테그레이션에서 고급 W&B 인테그레이션이 어떻게 보이는지 확인할 수 있습니다. 대부분의 인테그레이션은 이들만큼 복잡하지 않을 것입니다:

* [Hugging Face Transformers `WandbCallback`](https://github.com/huggingface/transformers/blob/49629e7ba8ef68476e08b671d6fc71288c2f16f1/src/transformers/integrations.py#L639)
* [PyTorch Lightning `WandbLogger`](https://github.com/Lightning-AI/lightning/blob/18f7f2d3958fb60fcb17b4cb69594530e83c217f/src/pytorch_lightning/loggers/wandb.py#L53)