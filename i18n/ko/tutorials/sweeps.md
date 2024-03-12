
# 하이퍼파라미터 튜닝하기

[**Colab 노트북에서 시도해보기 →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W&B.ipynb)

높은 차원의 하이퍼파라미터 공간을 탐색하여 가장 성능이 좋은 모델을 찾는 것은 매우 복잡해질 수 있습니다. 하이퍼파라미터 스윕은 모델들의 대결에서 가장 정확한 모델을 선택할 수 있는 조직적이고 효율적인 방법을 제공합니다. 이는 하이퍼파라미터 값(예: 학습률, 배치 크기, 은닉층의 수, 옵티마이저 유형)의 조합을 자동으로 탐색하여 가장 최적의 값을 찾음으로써 가능합니다.

이 튜토리얼에서는 Weights & Biases를 사용하여 3단계만으로 고급 하이퍼파라미터 스윕을 실행하는 방법을 살펴보겠습니다.

### [비디오 튜토리얼](http://wandb.me/sweeps-video)을 따라해 보세요!

![](https://i.imgur.com/WVKkMWw.png)

# 🚀 설정

실험 추적 라이브러리를 설치하고 무료 W&B 계정을 설정하여 시작하세요:

1. `!pip install`로 설치
2. Python에 라이브러리를 `import`
3. 프로젝트에 메트릭을 로그할 수 있도록 `.login()`

Weights & Biases를 처음 사용하는 경우,
`login` 호출은 계정에 가입할 수 있는 링크를 제공합니다.
W&B는 개인 및 학술 프로젝트에 무료로 사용할 수 있습니다!


```python
!pip install wandb -Uq
```


```python
import wandb

wandb.login()
```

# 1️⃣ 단계. 스윕 정의하기

기본적으로, 스윕은 많은 하이퍼파라미터 값들을 시도해보는 전략과 이를 평가하는 코드를 결합합니다.
[설정](https://docs.wandb.com/sweeps/configuration)의 형태로 _전략을 정의_하기만 하면 됩니다.

노트북에서 스윕을 설정할 때는
config 오브젝트가 중첩된 사전입니다.
커맨드라인을 통해 스윕을 실행할 때는
config 오브젝트가
[YAML 파일](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration)입니다.

함께 스윕 config를 정의해 봅시다.
천천히 진행해서 각 구성 요소를 설명할 기회를 갖겠습니다.
일반적인 스윕 파이프라인에서는
이 단계가 단일 할당에서 이루어집니다.

### 👈 `method` 선택하기

새로운 파라미터 값을 선택하는 `method`가 첫 번째로 정의해야 할 것입니다.

다음과 같은 검색 `methods`를 제공합니다:
*   **`grid` 검색** – 하이퍼파라미터 값의 모든 조합을 반복합니다.
매우 효과적이지만 계산 비용이 많이 들 수 있습니다.
*   **`random` 검색** – 제공된 `distribution`들에 따라 각 새로운 조합을 무작위로 선택합니다. 의외로 효과적입니다!
*   **`bayes`ian 검색** – 메트릭 점수를 하이퍼파라미터의 함수로 하는 확률 모델을 만들고, 메트릭을 개선할 가능성이 높은 파라미터를 선택합니다. 연속적인 파라미터의 소수에 대해 잘 작동하지만 규모가 커지면 잘 확장되지 않습니다.

우리는 `random`을 사용할 것입니다.


```python
sweep_config = {
    'method': 'random'
    }
```

`bayes`ian 스윕의 경우,
메트릭에 대해 약간의 정보를 더 알려주어야 합니다.
모델 출력에서 찾을 수 있도록 메트릭의 `name`을 알아야 하고
`goal`이 메트릭을 `minimize`하는 것인지
(예: 제곱 오차인 경우)
아니면 `maximize`하는 것인지
(예: 정확도인 경우) 알아야 합니다.


```python
metric = {
    'name': 'loss',
    'goal': 'minimize'   
    }

sweep_config['metric'] = metric
```

`bayes`ian 스윕을 실행하지 않는 경우에도,
나중에 마음이 바뀌었을 때를 대비해 `sweep_config`에 이를 포함하는 것이 나쁘지 않습니다.
또한 6개월 또는 6년 후에 당신이나 다른 사람이
당신의 스윕으로 돌아왔을 때 `val_G_batch`가 높아야 하는지 낮아야 하는지 모르는 경우와 같이,
재현성을 위한 좋은 관행입니다.

### 📃 하이퍼`parameters` 이름 지정하기

새로운 하이퍼파라미터 값의 새로운 `method`를 선택하면,
그 `parameters`가 무엇인지 정의해야 합니다.

대부분의 경우 이 단계는 간단합니다:
`parameter`에 이름을 지정하고
파라미터의 합법적인 `values` 목록을 지정하기만 하면 됩니다.

예를 들어, 네트워크의 `optimizer`를 선택할 때,
옵션이 유한한 수입니다.
여기서는 가장 인기 있는 두 가지 선택인 `adam`과 `sgd`를 사용합니다.
하이퍼파라미터가 무한한 옵션을 가질 수 있더라도,
여기서와 같이 은닉 `layer_size`와 `dropout`에 대해
몇 가지 선택된 `values`만 시도하는 것이 일반적으로 의미가 있습니다.


```python
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

이 스윕에서는 변화시키고 싶지 않지만,
여전히 `sweep_config`에서 설정하고 싶은 하이퍼파라미터가 종종 있습니다.

이 경우, `value`를 직접 설정합니다:


```python
parameters_dict.update({
    'epochs': {
        'value': 1}
    })
```

`grid` 검색의 경우, 이것이 전부입니다.

`random` 검색의 경우,
주어진 실행에서 파라미터의 모든 `values`가 선택될 확률이 동일합니다.

이것으로 충분하지 않다면,
대신 명명된 `distribution`과
평균 `mu`와 표준 편차 `sigma`와 같은 매개변수를 지정할 수 있습니다.

무작위 변수의 분포를 설정하는 방법에 대한 자세한 내용은 [여기](https://docs.wandb.com/sweeps/configuration#distributions)에서 확인할 수 있습니다.


```python
parameters_dict.update({
    'learning_rate': {
        # 0과 0.1 사이의 균일 분포
        'distribution': 'uniform',
        'min': 0,
        'max': 0.1
      },
    'batch_size': {
        # 32와 256 사이의 정수
        # 로그가 균등하게 분포됨
        'distribution': 'q_log_uniform_values',
        'q': 8,
        'min': 32,
        'max': 256,
      }
    })
```

완료되면, `sweep_config`은 정확히 어떤 `parameters`에 관심이 있는지
그리고 그것들을 시도하기 위해 어떤 `method`를 사용할 것인지를
명시하는 중첩된 사전입니다.


```python
import pprint

pprint.pprint(sweep_config)
```

하지만 이것이 모든 설정 옵션은 아닙니다!

예를 들어, [HyperBand](https://arxiv.org/pdf/1603.06560.pdf) 스케줄링 알고리즘을 사용하여 실행을 `early_terminate`하는 옵션도 제공합니다. 자세한 내용은 [여기](https://docs.wandb.com/sweeps/configuration#stopping-criteria)에서 확인하세요.

모든 구성 옵션 목록은 [여기](https://docs.wandb.com/library/sweeps/configuration)에서 찾을 수 있으며,
YAML 형식의 큰 예제 모음은 [여기](https://github.com/wandb/examples/tree/master/examples/keras/keras-cnn-fashion)에서 확인할 수 있습니다.

# 2️⃣ 단계. 스윕 초기화하기

검색 전략을 정의한 후에는 이를 구현할 무언가를 설정할 시간입니다.

우리의 스윕을 담당하는 시계작업장이자는 _스윕 컨트롤러_로 알려져 있습니다.
각 실행이 완료될 때마다, 실행할 새로운 실행을 설명하는 새로운 일련의 지침을 발행합니다.
이 지침은 실행을 실제로 수행하는 _에이전트_가 수집합니다.

일반적인 스윕에서는 컨트롤러가 _우리의_ 기계에 존재하고,
실행을 완료하는 에이전트가 _당신의_ 기계(들)에 존재합니다.
이 작업 분담은 에이전트를 실행하기 위해 더 많은 기계를 추가하기만 하면 스윕을 쉽게 확장할 수 있게 합니다!

아래의 다이어그램처럼.
<img src="https://i.imgur.com/zlbw3vQ.png" alt="sweeps-diagram" width="500"/>

적절한 `sweep_config`와 `프로젝트` 이름으로 `wandb.sweep`을 호출하여 스윕 컨트롤러를 작동시킬 수 있습니다.

이 함수는 나중에 이 컨트롤러에 에이전트를 할당하기 위해 사용할 `sweep_id`를 반환합니다.

> _사이드 노트_: 커맨드라인에서, 이 함수는 다음과 같이 대체됩니다.
```python
wandb sweep config.yaml
```
[커맨드라인에서 스윕 사용에 대해 더 알아보기 ➡](https://docs.wandb.ai/guides/sweeps/walkthrough)


```python
sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")
```

# 3️⃣ 단계. 스윕 에이전트 실행하기

### 💻 트레이닝 절차 정의하기

스윕을 실제로 실행하기 전에,
그 값을 사용하는 트레이닝 절차를 정의해야 합니다.

아래 함수에서는 PyTorch에서 간단한 완전 연결 신경망을 정의하고 다음 `wandb` 도구를 추가하여 모델 메트릭을 로그하고, 성능과 출력을 시각화하며, 실험을 추적합니다:
* [**`wandb.init()`**](https://docs.wandb.com/library/init) – 새로운 W&B 실행을 초기화합니다. 각 실행은 트레이닝 함수의 단일 실행입니다.
* [**`wandb.config`**](https://docs.wandb.com/library/config) – 모든 하이퍼파라미터를 구성 오브젝트에 저장하여 로그합니다. `wandb.config` 사용 방법에 대해 자세히 알아보려면 [여기](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Configs_in_W%26B.ipynb)를 참조하세요.
* [**`wandb.log()`**](https://docs.wandb.com/library/log) – W&B에 모델 동작을 로그합니다. 여기서는 성능만 로그합니다; `wandb.log`로 로그할 수 있는 다른 모든 리치 미디어는 [이 Colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Log_(Almost)_Anything_with_W%26B_Media.ipynb)에서 확인하세요.

PyTorch와 W&B를 함께 사용하는 더 자세한 내용은 [이 Colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb)을 참조하세요.


```python
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(config=None):
    # 새로운 wandb 실행을 초기화합니다.
    with wandb.init(config=config):
        # wandb.agent 아래에서 호출된 경우,
        # 이 config는 스윕 컨트롤러에 의해 설정됩니다.
        config = wandb.config

        loader = build_dataset(config.batch_size)
        network = build_network(config.fc_layer_size, config.dropout)
        optimizer = build_optimizer(network, config.optimizer, config.learning_rate)

        for epoch in range(config.epochs):
            avg_loss = train_epoch(network, loader, optimizer)
            wandb.log({"loss": avg_loss, "epoch": epoch})           
```

이 셀은 우리의 트레이닝 절차의 네 부분을 정의합니다:
`build_dataset`, `build_network`, `build_optimizer`, `train_epoch`.

이 모든 것은 기본 PyTorch 파이프라인의 표준 부분이며,
W&B 사용에 영향을 받지 않으므로
이에 대해 논평하지 않겠습니다.


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
    network = nn.Sequential(  # 완전 연결, 단일 은닉층
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
    cumu