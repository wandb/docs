---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# wandb를 모든 라이브러리에 추가하기

이 가이드는 강력한 실험 추적, GPU 및 시스템 모니터링, 모델 체크포인팅 등을 위해 Python 라이브러리에 W&B를 통합하는 방법에 대한 모범 사례를 제공합니다.

:::note
W&B 사용 방법을 아직 배우고 있다면, 이 문서의 다른 W&B 가이드를 탐색하는 것이 좋습니다. 예를 들어, [실험 추적](https://docs.wandb.ai/guides/track)과 같은 가이드를 참고하세요.
:::

아래에서는 코드베이스가 단일 Python 트레이닝 스크립트나 Jupyter 노트북보다 복잡할 때 최고의 팁과 모범 사례를 다룹니다. 다루는 주제는 다음과 같습니다:

* 설치 요구사항
* 사용자 로그인
* wandb Run 시작하기
* Run 설정 정의하기
* W&B에 로깅하기
* 분산 트레이닝
* 모델 체크포인팅 등
* 하이퍼파라미터 튜닝
* 고급 통합

### 설치 요구사항

시작하기 전에 W&B를 라이브러리의 의존성으로 포함할지 여부를 결정하세요:

#### 설치 시 W&B 필수로 요구하기

W&B Python 라이브러리(`wandb`)를 의존성 파일에 추가하세요. 예를 들어, `requirements.txt` 파일에 다음과 같이 추가합니다.

```python
torch==1.8.0 
...
wandb==0.13.*
```

#### 설치 시 W&B를 선택사항으로 만들기

W&B SDK(`wandb`)를 선택사항으로 만드는 두 가지 방법이 있습니다:

A. 사용자가 수동으로 설치하지 않고 `wandb` 기능을 사용하려고 할 때 적절한 오류 메시지와 함께 오류를 발생시킵니다:

```python
try: 
    import wandb 
except ImportError: 
    raise ImportError(
        "wandb를 현재 설치하지 않았습니다"
        "pip install wandb를 사용하여 설치하세요"
    ) 
```

B. Python 패키지를 빌드하는 경우 `pyproject.toml` 파일에 `wandb`를 선택적 의존성으로 추가합니다.

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
    {label: '노트북', value: 'notebook'},
    {label: '환경 변수', value: 'environment'},
  ]}>
  <TabItem value="bash">
터미널에서 bash 명령어로 W&B에 로그인

```bash
wandb login $MY_WANDB_KEY
```
  </TabItem>
  <TabItem value="notebook">
Jupyter 또는 Colab 노트북에서 다음과 같이 W&B에 로그인

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

사용자가 위에서 언급한 단계를 따르지 않고 처음으로 wandb를 사용하는 경우, 스크립트가 `wandb.init`를 호출할 때 자동으로 로그인하라는 메시지가 표시됩니다.

### wandb Run 시작하기

W&B Run은 W&B에 의해 로깅된 계산의 단위입니다. 일반적으로 단일 W&B Run을 트레이닝 실험마다 연관시킵니다.

코드 내에서 W&B를 초기화하고 Run을 시작하려면:

```python
wandb.init()
```

선택적으로 프로젝트의 이름을 제공하거나 `wandb_project`와 같은 파라미터를 코드에 추가하여 사용자가 직접 설정하도록 할 수 있습니다. 또한 사용자 이름이나 팀 이름과 같은 엔터티 파라미터를 위해 `wandb_entity`와 같이 설정할 수 있습니다:

```python
wandb.init(project=wandb_project, entity=wandb_entity)
```

#### `wandb.init`을 어디에 배치해야 하나요?

라이브러리는 가능한 한 빨리 W&B Run을 생성해야 합니다. 왜냐하면 콘솔의 모든 출력, 오류 메시지를 포함하여 W&B Run의 일부로 로깅되기 때문입니다. 이를 통해 디버깅이 더 쉬워집니다.

#### `wandb`를 선택사항으로 라이브러리와 함께 실행하기

사용자가 라이브러리를 사용할 때 `wandb`를 선택사항으로 만들고 싶다면, 다음 중 하나를 수행할 수 있습니다:

* `wandb` 플래그를 정의합니다. 예를 들면:

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
wandb.init(mode="disabled")
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

* 또는, `wandb`를 오프라인으로 설정합니다 - 이는 여전히 `wandb`를 실행하지만 인터넷을 통해 W&B로 통신을 시도하지 않습니다

<Tabs
  defaultValue="environment"
  values={[
    {label: '환경 변수', value: 'environment'},
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

### wandb Run 설정 정의하기

`wandb` run 설정을 통해 모델, 데이터셋 등에 대한 메타데이터를 제공할 수 있습니다. 이 정보를 사용하여 다양한 실험을 비교하고 주요 차이점을 빠르게 이해할 수 있습니다.

![W&B Runs table](/images/integrations/integrations_add_any_lib_runs_page.png)

로그할 수 있는 전형적인 설정 파라미터에는 다음이 포함됩니다:

* 모델 이름, 버전, 아키텍처 파라미터 등
* 데이터셋 이름, 버전, 훈련/검증 예제 수 등
* 학습 파라미터 예: 학습률, 배치 크기, 옵티마이저 등

다음 코드 조각은 설정을 로깅하는 방법을 보여줍니다:

```python
config = {"batch_size":32, …}
wandb.init(…, config=config)
```

#### wandb 설정 업데이트하기

`wandb.config.update`를 사용하여 설정을 업데이트합니다. 설정 사전이 정의된 후에 파라미터를 얻을 때 설정 사전을 업데이트하는 것이 유용할 수 있습니다. 예를 들어, 모델이 인스턴스화된 후 모델의 파라미터를 추가하고 싶을 수 있습니다.

```python
wandb.config.update({"model_parameters" = 3500})
```

설정 파일을 정의하는 방법에 대한 자세한 내용은 [wandb.config로 실험을 구성하기](https://docs.wandb.ai/guides/track/config)를 참조하세요.

### W&B에 로깅하기

#### 메트릭 로깅

키 값이 메트릭의 이름인 사전을 생성합니다. 이 사전 객체를 [`wandb.log`](https://docs.wandb.ai/guides/track/log)에 전달합니다:

```python
for epoch in range(NUM_EPOCHS):
    for input, ground_truth in data: 
        prediction = model(input) 
        loss = loss_fn(prediction, ground_truth) 
        metrics = { "loss": loss } 
        wandb.log(metrics)
```

메트릭이 많은 경우, 메트릭 이름에 접두사를 사용하여 UI에서 자동으로 그룹화할 수 있습니다. 예를 들어 `train/...` 및 `val/...` 이렇게 하면 훈련 및 검증 메트릭 또는 분리하고 싶은 기타 메트릭 유형에 대해 W&B 워크스페이스에서 별도의 섹션이 생성됩니다.

```python
metrics = {
    "train/loss": 0.4,
    "train/learning_rate": 0.4,
    "val/loss": 0.5, 
    "val/accuracy": 0.7
}
wandb.log(metrics)
```

![A W&B Workspace with 2 separate sections](/images/integrations/integrations_add_any_lib_log.png)

`wandb.log`에 대한 자세한 내용은 [wandb.log로 데이터 로깅하기](https://docs.wandb.ai/guides/track/log)를 참조하세요.

#### x축 불일치 방지

동일한 훈련 단계에 대해 여러 번 `wandb.log`를 호출해야 할 수도 있습니다. wandb SDK에는 `wandb.log` 호출이 이루어질 때마다 증가하는 자체 내부 단계 카운터가 있습니다. 이는 wandb 로그 카운터가 훈련 루프의 훈련 단계와 일치하지 않을 가능성이 있음을 의미합니다.

아래 예제의 첫 번째 패스에서, `train/loss`의 내부 `wandb` 단계는 0일 것이고, `eval/loss`의 내부 `wandb` 단계는 1일 것입니다. 다음 패스에서, `train/loss`는 2가 될 것이고, `eval/loss`의 wandb 단계는 3이 될 것입니다 등등

```python
for input, ground_truth in data:
    ...
    wandb.log("train/loss": 0.1)  
    wandb.log("eval/loss": 0.2)
```

이를 방지하기 위해, x축 단계를 명시적으로 정의하는 것이 좋습니다. `wandb.define_metric`을 사용하여 x축을 정의할 수 있으며, `wandb.init`이 호출된 후 한 번만 이 작업을 수행하면 됩니다:

```
wandb.init(...)
wandb.define_metric("*", step_metric="global_step")
```

글로브 패턴 "\*"은 모든 메트릭이 차트에서 "global_step"을 x축으로 사용한다는 것을 의미합니다. "global_step"에 대해 로깅되기를 원하는 특정 메트릭만 지정하고 싶다면 대신 그것들을 명시할 수 있습니다:

```
wandb.define_metric("train/loss", step_metric="global_step")
```

`wandb.define_metric`를 호출한 이후, `wandb.log`를 호출할 때마다 메트릭뿐만 아니라 `step_metric`, "global_step"도 로깅해야 합니다:

```python
for step, (input, ground_truth) in enumerate(data):
    ...
    wandb.log({"global_step": step, "train/loss": 0.1})
    wandb.log({"global_step": step, "eval/loss": 0.2})
```

독립적인 단계 변수에 액세스할 수 없는 경우, 예를 들어 검증 루프 중에 "global_step"이 사용 가능하지 않다면, 이전에 로깅된 "global_step" 값이 wandb에 의해 자동으로 사용됩니다. 이 경우, 필요할 때 정의되었을 때 메트릭에 대한 초기 값이 로깅되도록 합니다.

#### 이미지, 테이블, 텍스트, 오디오 등 로깅

메트릭 외에도, 플롯, 히스토그램, 테이블, 텍스트 및 이미지, 비디오, 오디오, 3D 등과 같은 미디어를 로깅할 수 있습니다.

데이터 로깅 시 고려해야 할 몇 가지 사항은 다음과 같습니다:

* 메트릭은 얼마나 자주 로깅되어야 하나요? 선택사항이어야 하나요?
* 어떤 유형의 데이터가 시각화에 도움이 될 수 있나요?
  * 이미지의 경우, 예측 샘플, 세분화 마스크 등을 로깅하여 시간이 지남에 따라 진화를 볼 수 있습니다.
  * 텍스트의 경우, 나중에 탐색할 수 있는 샘플 예측의 테이블을 로깅할 수 있습니다.

미디어, 오브젝트, 플롯 등을 로깅하는 전체 가이드는 [wandb.log로 데이터 로깅하기](https://docs.wandb.ai/guides/track/log)를 참조하세요.

### 분산 트레이닝

분산 환경을 지원하는 프레임워크의 경우, 다음 워크플로 중 하나를 적용할 수 있습니다:

* 어느 프로세스가 "메인" 프로세스인지 감지하고 `wandb`를 거기서만 사용합니다. 다른 프로세스에서 오는 필요한 데이터는 먼저 메인 프로세스로 라우팅해야 합니다. (이 워크플로가 권장됩니다).
* 모든 프로세스에서 `wandb`를 호출하고 모두 동일한 고유한 `group` 이름을 제공하여 자동으로 그룹화합니다.

자세한 내용은 [분산 트레이닝 실험 로깅하기](../track/log/distributed-training.md)를 참조하세요.

### 모델 체크포인팅 등 로깅하기

프레임워크가 모델 또는 데이터셋을 사용하거나 생성하는 경우, W&B Artifacts를 통해 전체 추적성을 로깅하고 wandb가 전체 파이프라인을 자동으로 모니터링할 수 있습니다.

![Stored Datasets and Model Checkpoints in W&B](/images/integrations/integrations_add_any_lib_dag.png)

Artifacts를 사용할 때, 사용자가 모델 체크포인트나 데이터셋을 로깅할 수 있는 기능을 정의할 수 있지만 필수는 아닙니다.