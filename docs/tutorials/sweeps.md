---
title: Tune hyperparameters with sweeps
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

<CTAButtons colabLink='https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W&B.ipynb'/>

원하는 메트릭(예: 모델 정확도)을 충족하는 기계학습 모델을 찾는 것은 일반적으로 여러 번 반복해야 하는 번거로운 작업입니다. 더 나쁜 것은, 주어진 트레이닝 run에 사용할 하이퍼파라미터 조합이 무엇인지 불분명할 수 있습니다.

W&B Sweeps를 사용하여 하이퍼파라미터 값의 조합을 자동으로 탐색하고 모델을 원하는 메트릭에 따라 최적화할 값을 찾기 위한 체계적이고 효율적인 방법을 만드세요. 예를 들어, 학습률, 배치 크기, 숨겨진 레이어의 수, 옵티마이저 유형 등을 조정할 수 있습니다.

이 튜토리얼에서는 W&B PyTorch 인테그레이션을 사용하여 하이퍼파라미터 탐색을 만듭니다. [비디오 튜토리얼](http://wandb.me/sweeps-video)을 따라오세요!

![](/images/tutorials/sweeps-1.png)

## Sweeps: 개요

Weights & Biases에서 하이퍼파라미터 스윕을 실행하는 것은 매우 쉽습니다. 다음 3단계만 따르면 됩니다:

1. **스윕 정의하기:** 검색해야 할 파라미터, 검색 전략, 최적화 메트릭 등을 지정하는 사전 또는 [YAML 파일](/guides/sweeps/define-sweep-configuration)을 생성하여 이를 수행합니다.

2. **스윕 초기화하기:** 한 줄의 코드로 스윕을 초기화하고 스윕 구성 사전을 전달합니다:
   `sweep_id = wandb.sweep(sweep_config)`

3. **스윕 에이전트 실행하기:** 한 줄의 코드로 수행되며, `wandb.agent()`를 호출하고, 모델 아키텍처를 정의하고 트레이닝할 함수를 함께 `sweep_id`에 전달합니다:
   `wandb.agent(sweep_id, function=train)`

## 시작하기 전에

W&B를 설치하고 W&B Python SDK를 노트북에 임포트합니다:

1. `!pip install`로 설치하세요:

```
!pip install wandb -Uq
```

2. W&B를 임포트합니다:

```
import wandb
```

3. W&B에 로그인하고 API 키를 제공하세요:

```
wandb.login()
```

## Step 1️: 스윕 정의하기

W&B Sweep은 수많은 하이퍼파라미터 값을 시도하는 전략을 그 값을 평가하는 코드와 결합합니다.
스윕을 시작하기 전에, _스윕 구성_을 통해 스윕 전략을 정의해야 합니다.

:::info
Jupyter 노트북에서 스윕을 시작하는 경우, 생성한 스윕 구성은 중첩된 사전이어야 합니다.

커맨드라인에서 스윕을 실행하는 경우, [YAML 파일](/guides/sweeps/define-sweep-configuration)을 사용하여 스윕 구성을 지정해야 합니다.
:::

### 검색 메소드 선택하기

먼저, 설정 사전 내에 하이퍼파라미터 검색 메소드를 지정합니다. [세 가지 하이퍼파라미터 검색 전략이 있으며, 그리드, 랜덤, 베이지안 탐색 중에서 선택할 수 있습니다](/guides/sweeps/sweep-config-keys#method).

이 튜토리얼에서는 랜덤 검색을 사용합니다. 노트북 내에서 사전을 만들고 `method` 키에 `random`을 지정하세요.

```
sweep_config = {
    'method': 'random'
    }
```

최적화하고 싶은 메트릭을 지정하세요. 랜덤 검색 방식의 스윕을 사용할 경우 메트릭과 목표를 지정할 필요가 없습니다. 그러나 나중에 스윕 목표를 참조할 수 있도록 스윕 목표를 기록해 두는 것이 좋은 습관입니다.

```
metric = {
    'name': 'loss',
    'goal': 'minimize'   
    }

sweep_config['metric'] = metric
```

### 탐색할 하이퍼파라미터 지정하기

스윕 구성에 검색 메소드를 지정했으므로, 이제 탐색할 하이퍼파라미터를 지정하세요.

이를 위해, `parameter` 키에 하나 이상의 하이퍼파라미터 이름을 지정하고, `value` 키에 하나 이상의 하이퍼파라미터 값을 지정하세요.

주어진 하이퍼파라미터에 대해 탐색하는 값은 조사 중인 하이퍼파라미터의 유형에 따라 다릅니다.

예를 들어 기계학습 옵티마이저를 선택한 경우 Adam 옵티마이저 및 확률적 그레이디언트 하강(Stochastic Gradient Descent)과 같은 하나 이상의 옵티마이저 이름을 지정해야 합니다.

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

하이퍼파라미터를 추적하고 싶지만 값을 변경하지 않으려는 경우, 스윕 구성에 하이퍼파라미터를 추가하고 사용하려는 정확한 값을 지정하세요. 예를 들어, 다음 코드 셀에서 `epochs`는 1로 설정되어 있습니다.

```
parameters_dict.update({
    'epochs': {
        'value': 1}
    })
```

`random` 검색의 경우, 주어진 run에서 파라미터의 모든 `값`이 동일한 확률로 선택됩니다.

대안으로, 이름이 지정된 `분포`와 함께 평균 `mu` 및 표준 편차 `sigma`와 같은 `normal` 분포의 파라미터를 지정할 수 있습니다.

```
parameters_dict.update({
    'learning_rate': {
        # 0과 0.1 사이의 평탄한 분포
        'distribution': 'uniform',
        'min': 0,
        'max': 0.1
      },
    'batch_size': {
        # 32부터 256까지의 정수
        # 균등 분포의 로그
        'distribution': 'q_log_uniform_values',
        'q': 8,
        'min': 32,
        'max': 256,
      }
    })
```

마침내, `sweep_config`는 우리가 시도해보고자 하는 정확한 `parameters`를 지정하고 사용할 `method`를 지정하는 중첩된 사전이 됩니다.

스윕 구성이 어떻게 되는지 봅시다:

```
import pprint
pprint.pprint(sweep_config)
```

구성 옵션의 전체 목록은 [Sweep 구성 옵션](/guides/sweeps/sweep-config-keys)을 참조하세요.

:::tip
잠재적으로 무한한 선택지를 가진 하이퍼파라미터의 경우, 몇 가지 선택된 `값`을 시도해보는 것이 보통 합리적입니다. 예를 들어, 이전 스윕 구성에는 `layer_size` 및 `dropout` 파라미터 키에 대해 지정된 유한한 값 목록이 있습니다.
:::

## Step 2️: 스윕 초기화

검색 전략을 정의한 후, 이를 구현하기 위한 설정을 해야 할 때입니다.

W&B는 클라우드 또는 로컬에서 하나 이상의 머신에 대해 스윕을 관리하기 위해 Sweep Controller를 사용합니다. 이 튜토리얼에서는 W&B가 관리하는 스윕 컨트롤러를 사용할 것입니다.

스윕 컨트롤러가 스윕을 관리하는 동안, 실제로 스윕을 실행하는 구성 요소는 _스윕 에이전트_라고 불립니다.

:::info
기본적으로, 스윕 컨트롤러 구성 요소는 W&B의 서버에서 시작되고, 스윕 에이전트는 사용자의 로컬 머신에서 활성화됩니다.
:::

노트북 내에서 `wandb.sweep` 메소드를 사용하여 스윕 컨트롤러를 활성화할 수 있습니다. 앞에서 정의한 스윕 구성 사전을 `sweep_config` 필드에 전달하세요:

```
sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")
```

`wandb.sweep` 함수는 나중 단계에서 스윕을 활성화하는 데 사용할 `sweep_id`를 반환합니다.

:::info
커맨드라인에서는 이 함수가 
```python
wandb sweep config.yaml
```
로 대체됩니다.
:::

터미널에서 W&B Sweeps를 생성하는 방법에 대한 자세한 내용은 [W&B Sweep walkthrough](/guides/sweeps/walkthrough)를 참조하세요.

## Step 3: 기계학습 코드 정의하기

스윕을 실행하기 전에,
시도해보고자 하는 하이퍼파라미터 값을 사용하는 트레이닝 절차를 정의하세요. W&B Sweeps를 트레이닝 코드에 통합하는 데 중요한 것은 각 트레이닝 실험에 대해 스윕 구성에 정의된 하이퍼파라미터 값에 엑세스할 수 있도록 하는 것입니다.

다음의 코드 예제에서, 도우미 함수 `build_dataset`, `build_network`, `build_optimizer` 및 `train_epoch`는 스윕 하이퍼파라미터 구성 사전에 엑세스합니다. 

기계학습 트레이닝 코드를 노트북에서 실행하세요. 함수들은 PyTorch에서 기본적인 완전 연결 신경망(fully-connected neural network)을 정의합니다.

```python
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(config=None):
    # 새로운 wandb run을 초기화합니다.
    with wandb.init(config=config):
        # wandb.agent에 의해 호출된 경우,
        # 이 설정은 Sweep Controller에 의해 설정됩니다.
        config = wandb.config

        loader = build_dataset(config.batch_size)
        network = build_network(config.fc_layer_size, config.dropout)
        optimizer = build_optimizer(network, config.optimizer, config.learning_rate)

        for epoch in range(config.epochs):
            avg_loss = train_epoch(network, loader, optimizer)
            wandb.log({"loss": avg_loss, "epoch": epoch})           
```

`train` 함수 내에서는 다음과 같은 W&B Python SDK 메소드를 볼 수 있습니다:
* [`wandb.init()`](/ref/python/init) – 새로운 W&B run을 초기화합니다. 각 run은 트레이닝 함수의 단일 실행입니다.
* [`wandb.config`](/guides/track/config) – 실험하려는 하이퍼파라미터와 함께 스윕 구성을 전달합니다.
* [`wandb.log()`](/ref/python/log) – 각 에포크에 대한 트레이닝 손실을 로그합니다.

다음 셀은 네 가지 함수를 정의합니다:
`build_dataset`, `build_network`, `build_optimizer` 및 `train_epoch`.
이 함수들은 기본적인 PyTorch 파이프라인의 표준적인 부분이며, W&B의 사용에 영향을 받지 않습니다.

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
    network = nn.Sequential(  # 완전 연결된, 단일 숨겨진 레이어
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
    for _, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # ➡ Forward pass
        loss = F.nll_loss(network(data), target)
        cumu_loss += loss.item()

        # ⬅ Backward pass + 가중치 업데이트
        loss.backward()
        optimizer.step()

        wandb.log({"batch loss": loss.item()})

    return cumu_loss / len(loader)
```

PyTorch와 W&B를 연계하는 것에 대한 자세한 내용은 이 [Colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb)을 참조하세요.

## Step 4: 스윕 에이전트 활성화
스윕 구성을 정의하고 해당 하이퍼파라미터를 대화형으로 활용할 수 있는 트레이닝 스크립트를 준비했으므로, 이제 스윕 에이전트를 활성화할 준비가 되었습니다. 스윕 에이전트는 스윕 구성에서 정의한 하이퍼파라미터 값 집합을 사용하여 실험을 수행하는 책임이 있습니다.

`wandb.agent` 메소드를 사용하여 스윕 에이전트를 만드세요. 다음을 제공하세요:
1. 에이전트가 속한 스윕 (`sweep_id`)
2. 스윕이 실행해야 하는 함수. 이 예제에서는 스윕이 `train` 함수를 사용할 것입니다.
3. (선택적으로) 스윕 컨트롤러에서 요청할 구성의 수 (`count`)

:::tip
같은 `sweep_id`를 가진 여러 스윕 에이전트를 서로 다른 컴퓨팅 자원에서 시작할 수 있습니다. 스윕 컨트롤러는 정의된 스윕 구성에 따라 이들이 협력하여 작동하도록 보장합니다.
:::

다음 셀은 트레이닝 함수(`train`)를 5번 실행하는 스윕 에이전트를 활성화합니다:

```python
wandb.agent(sweep_id, train, count=5)
```

:::info
스윕 구성에서 `random` 검색 방식이 지정되었기 때문에, 스윕 컨트롤러는 무작위로 생성된 하이퍼파라미터 값을 제공합니다.
:::

터미널에서 W&B Sweeps를 생성하는 방법에 대한 자세한 내용은 [W&B Sweep walkthrough](/guides/sweeps/walkthrough)를 참조하세요.

## 스윕 결과 시각화

### 병렬 좌표 플롯
이 플롯은 하이퍼파라미터 값을 모델 메트릭에 매핑합니다. 최고의 모델 성능을 달성한 하이퍼파라미터 조합에 초점을 맞추는 데 유용합니다.

![](/images/tutorials/sweeps-2.png)

### 하이퍼파라미터 중요도 플롯
하이퍼파라미터 중요도 플롯은 급격한 메트릭 예측에 가장 큰 영향을 미친 하이퍼파라미터를 표면화합니다.
우리는 랜덤 포레스트 모델의 피처 중요도와 상관관계를 보고합니다(암시적으로 선형 모델).

![](/images/tutorials/sweeps-3.png)

이러한 시각화는 중요한 파라미터(및 값 범위)에 집중하고, 추가 탐색 가치가 있는 파라미터를 식별함으로써 고비용의 하이퍼파라미터 최적화를 수행하는 데 시간과 자원을 절약할 수 있도록 돕습니다.


## W&B Sweeps에 대해 더 알아보기

여러분이 실험할 수 있는 간단한 트레이닝 스크립트와 [여러 스윕 구성]을 만들었습니다.(https://github.com/wandb/examples/tree/master/examples/keras/keras-cnn-fashion). 이를 시도해 보실 것을 강력히 권장합니다.

해당 저장소에는 [Bayesian Hyperband](https://app.wandb.ai/wandb/examples-keras-cnn-fashion/sweeps/us0ifmrf?workspace=user-lavanyashukla), 및 [Hyperopt](https://app.wandb.ai/wandb/examples-keras-cnn-fashion/sweeps/xbs2wm5e?workspace=user-lavanyashukla)와 같은 고급 스윕 기능을 시도하는 데 도움이 될 예제가 포함되어 있습니다.