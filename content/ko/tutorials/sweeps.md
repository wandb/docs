---
title: 스윕으로 하이퍼파라미터 튜닝하기
menu:
  tutorials:
    identifier: ko-tutorials-sweeps
    parent: null
weight: 3
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W&B.ipynb" >}}

원하는 메트릭(예: 모델 정확도)을 충족하는 기계학습 모델을 찾는 과정은 여러 번 반복해야 하는 지루한 작업일 때가 많습니다. 상황에 따라서는 트레이닝 run 에 사용할 하이퍼파라미터 조합이 무엇인지조차 불확실할 수 있습니다.

W&B Sweeps를 활용하면 학습률, 배치 크기, 히든 레이어 수, 옵티마이저 타입 등 다양한 하이퍼파라미터 값의 조합을 자동으로 탐색하면서, 원하는 메트릭 기준으로 모델이 잘 동작하는 값들을 쉽게 찾을 수 있습니다.

이 튜토리얼에서는 W&B PyTorch 인테그레이션과 함께 하이퍼파라미터 탐색을 만들어볼 것입니다. [영상 튜토리얼](https://wandb.me/sweeps-video)을 참고해 따라해보세요.

{{< img src="/images/tutorials/sweeps-1.png" alt="하이퍼파라미터 스윕 결과" >}}

## Sweeps: 개요

W&B로 하이퍼파라미터 스윕을 실행하는 과정은 매우 간단합니다. 아래 3단계만 따르세요:

1. **스윕 정의:** 사전(dictionary) 또는 [YAML 파일]({{< relref path="/guides/models/sweeps/define-sweep-configuration" lang="ko" >}})을 만들어 탐색할 파라미터, 탐색 방식, 최적화 메트릭 등을 지정합니다.

2. **스윕 초기화:** 코드 한 줄로 스윕을 초기화하고, 스윕 구성을 담은 사전을 전달합니다:
   `sweep_id = wandb.sweep(sweep_config)`

3. **스윕 에이전트 실행:** 역시 코드 한 줄로 `wandb.agent()`를 호출해 `sweep_id`와, 모델 아키텍처를 정의하고 학습시키는 함수를 전달해 실행합니다:
   `wandb.agent(sweep_id, function=train)`

## 시작 전 준비하기

W&B를 설치하고, Python SDK를 노트북에 임포트합니다:

1. `!pip install`로 설치:

```
!pip install wandb -Uq
```

2. W&B 임포트:

```
import wandb
```

3. W&B에 로그인하고 API 키를 입력:

```
wandb.login()
```

## Step 1️: 스윕 정의하기

W&B Sweep은 다양한 하이퍼파라미터 값을 탐색하는 전략과, 이를 평가하는 코드가 결합된 것입니다.
스윕을 시작하기 전, _스윕 구성(sweep configuration)_ 을 정의해야 합니다.

{{% alert %}}
Jupyter Notebook에서 스윕을 시작하려면, 반드시 중첩된 사전(nested dictionary) 형식으로 스윕 구성을 작성해야 합니다.

커맨드라인에서 스윕을 실행할 경우, [YAML 파일]({{< relref path="/guides/models/sweeps/define-sweep-configuration" lang="ko" >}})로 스윕 구성을 지정해야 합니다.
{{% /alert %}}

### 탐색 방식 선택

먼저, 스윕 구성 사전에 하이퍼파라미터 탐색 방식을 지정합니다. [그리드, 랜덤, 베이지안 탐색 세 가지 방법 중 선택 가능합니다]({{< relref path="/guides/models/sweeps/define-sweep-configuration/sweep-config-keys/#method" lang="ko" >}}).

이 튜토리얼에서는 랜덤 검색 방식을 사용합니다. 노트북에서 사전을 만들고, `method` 키에 `random`을 지정하세요.

```
sweep_config = {
    'method': 'random'
    }
```

최적화할 메트릭을 지정합니다. 랜덤 탐색 방식을 사용할 때는 메트릭과 목표(goal)를 꼭 지정할 필요는 없지만, 나중을 대비해 목표를 명시해두는 것이 좋습니다.

```
metric = {
    'name': 'loss',
    'goal': 'minimize'   
    }

sweep_config['metric'] = metric
```

### 탐색할 하이퍼파라미터 지정하기

이제 스윕 구성에 탐색 방식을 지정했으니, 탐색할 하이퍼파라미터들을 추가합니다.

`parameter` 키에 하이퍼파라미터 이름을, `value` 혹은 `values` 키에 탐색할 값들을 나열합니다.

어떤 하이퍼파라미터를 실험하는지에 따라 탐색할 값의 종류가 달라집니다.

예를 들어, 기계학습 옵티마이저를 탐색하려면, Adam 옵티마이저나 확률적 경사 하강법(sgd) 등 유한한 옵션 중에서 선택할 수 있습니다.

```
parameters_dict = {
    'optimizer': {
        'values': ['adam', 'sgd']
        },
    'fc_layer_size': {
        'values': [128, 256, 512]
        },
    'dropout': {
          'values': [0.3, 0.4, 0.5]
        },
    }

sweep_config['parameters'] = parameters_dict
```

가끔 하이퍼파라미터를 추적은 하고 싶지만 값을 바꾸고 싶지 않을 때가 있습니다. 이 경우, 스윕 구성에 해당 하이퍼파라미터를 추가하고 사용할 값을 그대로 명시하면 됩니다. 아래 코드는 `epochs`를 1로 고정합니다.

```
parameters_dict.update({
    'epochs': {
        'value': 1}
    })
```

`random` 탐색의 경우,
각 파라미터 `values`의 값들이 run 에서 똑같은 확률로 선택됩니다.

또는,
특정한 분포(distribution)와 파라미터(예: normal 분포의 평균 mu, 표준편차 sigma 등)를 지정할 수도 있습니다.

```
parameters_dict.update({
    'learning_rate': {
        # 0~0.1 사이에서 균일하게 값 샘플링
        'distribution': 'uniform',
        'min': 0,
        'max': 0.1
      },
    'batch_size': {
        # 32~256 사이의 정수, 로그 스케일로 균등 분포
        'distribution': 'q_log_uniform_values',
        'q': 8,
        'min': 32,
        'max': 256,
      }
    })
```

완성된 `sweep_config`는
우리가 시도해볼 `parameters`와 사용할 `method`가
중첩된 사전 형태로 정의되어 있습니다.

아래처럼 스윕 구성이 어떻게 생겼는지 확인할 수 있습니다:

```
import pprint
pprint.pprint(sweep_config)
```

자세한 구성 옵션은 [스윕 구성 옵션]({{< relref path="/guides/models/sweeps/define-sweep-configuration/sweep-config-keys/" lang="ko" >}})에서 확인하세요.

{{% alert %}}
무한에 가까운 옵션이 있는 하이퍼파라미터의 경우,
몇 가지 `values`만 선택하는 것이 보통 더 효율적입니다. 위 예시에서처럼 `layer_size`와 `dropout` 파라미터는 유한한 값 리스트를 지정했습니다.
{{% /alert %}}

## Step 2️: 스윕 초기화하기

탐색 전략을 정의했다면, 이제 이를 실제로 구현할 차례입니다.

W&B는 Sweep Controller를 사용해 클라우드 또는 로컬, 여러 머신에서 스윕을 관리합니다. 이 튜토리얼에서는 W&B에서 관리하는 sweep controller를 사용합니다.

Sweep Controller가 스윕을 관리하는 동안, 실제로 스윕을 실행하는 컴포넌트는 _스윕 에이전트_입니다.

{{% alert %}}
기본적으로, Sweep Controller 컴포넌트는 W&B 서버에서, 스윕 에이전트(실제 스윕 실행)는 사용자의 로컬 머신에서 시작됩니다.
{{% /alert %}}

노트북 내에서 `wandb.sweep` 메소드로 sweep controller를 활성화할 수 있습니다. 앞서 정의한 sweep 구성 사전을 `sweep_config` 파라미터에 전달해주세요:

```
sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")
```

`wandb.sweep` 함수는 추후 스윕 실행에 사용할 `sweep_id`를 반환합니다.

{{% alert %}}
커맨드라인에서는 다음 python 코드로 기능을 대체합니다.
```python
wandb sweep config.yaml
```
{{% /alert %}}

터미널에서 W&B Sweep을 생성하는 방법의 자세한 설명은 [W&B Sweep 워크스루]({{< relref path="/guides/models/sweeps/walkthrough" lang="ko" >}})를 참고하세요.

## Step 3: 기계학습 코드 작성

스윕을 실행하기 전에,
실험에 사용할 하이퍼파라미터 값을 받아서 트레이닝하는 학습 함수를 정의합니다. W&B Sweeps와 학습 코드를 연동하는 핵심은, 매 실험마다 스윕 구성에 명시된 하이퍼파라미터 값을 트레이닝 로직이 엑세스할 수 있게 하는 것입니다.

아래 예시에서는 보조 함수(`build_dataset`, `build_network`, `build_optimizer`, `train_epoch`)가 스윕 하이퍼파라미터 구성 사전을 사용합니다.

노트북에서 아래 기계학습 트레이닝 코드를 실행해보세요. 이 함수들은 PyTorch로 기본적인 완전연결 신경망을 정의합니다.

```python
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(config=None):
    # 새로운 wandb run을 초기화합니다
    with wandb.init(config=config) as run:
        # 아래와 같이 wandb.agent가 호출하면,
        # 이 config는 Sweep Controller에 의해 설정됩니다
        config = run.config

        loader = build_dataset(config.batch_size)
        network = build_network(config.fc_layer_size, config.dropout)
        optimizer = build_optimizer(network, config.optimizer, config.learning_rate)

        for epoch in range(config.epochs):
            avg_loss = train_epoch(network, loader, optimizer)
            run.log({"loss": avg_loss, "epoch": epoch})           
```

위 `train` 함수에는 다음과 같은 W&B Python SDK 메소드들이 등장합니다.
* [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init/" lang="ko" >}}): 새로운 W&B Run을 초기화합니다. 각 run은 학습 함수의 한 번 실행을 의미합니다.
* [`run.config`]({{< relref path="/guides/models/track/config" lang="ko" >}}): 실험할 하이퍼파라미터가 담긴 스윕 구성을 전달받습니다.
* [`run.log()`]({{< relref path="/ref/python/sdk/classes/run/#method-runlog" lang="ko" >}}): 각 에포크의 트레이닝 loss를 기록합니다.

아래 코드는 네 가지 함수를 정의합니다:
`build_dataset`, `build_network`, `build_optimizer`, `train_epoch`.
이 함수들은 표준적인 PyTorch 파이프라인에 속하는 것으로,
W&B와는 독립적으로 구현됩니다.

```python
def build_dataset(batch_size):
   
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
    # MNIST 트레이닝 데이터셋 다운로드
    dataset = datasets.MNIST(".", train=True, download=True,
                             transform=transform)
    sub_dataset = torch.utils.data.Subset(
        dataset, indices=range(0, len(dataset), 5))
    loader = torch.utils.data.DataLoader(sub_dataset, batch_size=batch_size)

    return loader


def build_network(fc_layer_size, dropout):
    network = nn.Sequential(  # 완전 연결, 단일 히든 레이어
        nn.Flatten(),
        nn.Linear(784, fc_layer_size), nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(fc_layer_size, 10),
        nn.LogSoftmax(dim=1))

    return network.to(device)
        

def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),
                               lr=learning_rate)
    return optimizer


def train_epoch(network, loader, optimizer):
    cumu_loss = 0

    with wandb.init() as run:
        for _, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # ➡ forward 패스
            loss = F.nll_loss(network(data), target)
            cumu_loss += loss.item()

            # ⬅ backward 패스 + 가중치 업데이트
            loss.backward()
            optimizer.step()

            run.log({"batch loss": loss.item()})

    return cumu_loss / len(loader)
```

PyTorch와 함께 W&B를 연동하는 자세한 방법은 [이 Colab 문서](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb)를 참고하세요.

## Step 4: 스윕 에이전트 실행

이제 스윕 구성도 정의했고, 하이퍼파라미터를 실험에 사용할 수 있는 트레이닝 스크립트도 작성했으니, 스윕 에이전트를 실행해봅시다. 스윕 에이전트는 스윕 구성에서 정의한 하이퍼파라미터 값 조합으로 실험을 반복 실행하는 역할을 합니다.

`wandb.agent` 메소드로 스윕 에이전트를 생성합니다. 다음의 인자가 필요합니다.
1. 에이전트가 소속된 sweep (`sweep_id`)
2. sweep에서 반복 실행할 함수. 여기서는 `train` 함수가 대상입니다.
3. (선택) sweep controller로부터 받을 config 개수(`count`)

{{% alert %}}
동일한 `sweep_id`로 여러 대의 컴퓨팅 자원에서 스윕 에이전트를 동시에 실행할 수 있습니다. Sweep Controller가 스윕 구성에 맞게 전체 작업을 조율합니다.
{{% /alert %}}

아래 코드는 트레이닝 함수(`train`)를 5회 실행하는 스윕 에이전트를 활성화합니다:

```python
wandb.agent(sweep_id, train, count=5)
```

{{% alert %}}
스윕 구성에서 `random` 탐색 방식을 지정했으므로, 스윕 컨트롤러가 임의로 생성한 하이퍼파라미터 값을 제공합니다.
{{% /alert %}}

터미널에서 W&B Sweep을 만드는 법은 [W&B Sweep 워크스루]({{< relref path="/guides/models/sweeps/walkthrough" lang="ko" >}})를 참고하세요.

## 스윕 결과 시각화

### Parallel Coordinates Plot
이 플롯은 하이퍼파라미터 값을 모델 메트릭에 매핑합니다. 최적의 모델 성능을 보인 하이퍼파라미터 조합을 찾는 데 유용합니다.

{{< img src="/images/tutorials/sweeps-2.png" alt="스윕 에이전트 실행 결과" >}}

### Hyperparameter Importance Plot
하이퍼파라미터 중요도 플롯은 어떤 하이퍼파라미터가 메트릭의 예측에 가장 큰 영향을 미쳤는지 시각적으로 보여줍니다.
(랜덤 포레스트 모델의 feature 중요도와 상관계수 - 선형관계 기준 - 를 함께 제공합니다.)

{{< img src="/images/tutorials/sweeps-3.png" alt="W&B 스윕 대시보드" >}}

이런 시각화는 가장 중요한 파라미터(및 값 범위)에 집중해 리소스를 절약하고, 의미 있는 추가 실험 설계에도 도움이 됩니다.

## W&B Sweeps 더 알아보기

간단한 트레이닝 스크립트와 [스윕 구성 샘플](https://github.com/wandb/examples/tree/master/examples/keras/keras-cnn-fashion)을 만들어두었으니 부담 없이 실험해보세요!

해당 저장소에는 [Bayesian Hyperband](https://app.wandb.ai/wandb/examples-keras-cnn-fashion/sweeps/us0ifmrf?workspace=user-lavanyashukla), [Hyperopt](https://app.wandb.ai/wandb/examples-keras-cnn-fashion/sweeps/xbs2wm5e?workspace=user-lavanyashukla) 등 더 수준 높은 스윕 예제도 포함되어 있습니다.