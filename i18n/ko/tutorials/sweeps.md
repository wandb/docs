
# 하이퍼파라미터 조정

[**Colab 노트북에서 시도해보기 →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W&B.ipynb)

높은 차원의 하이퍼파라미터 공간을 검색하여 가장 성능이 좋은 모델을 찾는 것은 매우 번거로울 수 있습니다. 하이퍼파라미터 스윕은 모델들의 배틀 로얄을 조직적이고 효율적으로 수행하여 가장 정확한 모델을 선택할 수 있는 방법을 제공합니다. 이를 통해 하이퍼파라미터 값(예: 학습률, 배치 크기, 은닉층의 수, 옵티마이저 유형)의 조합을 자동으로 검색하여 가장 최적의 값을 찾을 수 있습니다.

이 튜토리얼에서는 Weights & Biases를 사용하여 3단계로 고급 하이퍼파라미터 스윕을 실행하는 방법을 살펴보겠습니다.

### [비디오 튜토리얼](http://wandb.me/sweeps-video) 따라하기!

![](https://i.imgur.com/WVKkMWw.png)

# 🚀 설정

실험 추적 라이브러리를 설치하고 무료 W&B 계정을 설정하여 시작하세요:

1. `!pip install`로 설치
2. 라이브러리를 Python에 `import`
3. 프로젝트에 메트릭을 로그할 수 있도록 `.login()`

Weights & Biases를 처음 사용한다면,
`login` 호출은 계정을 등록할 수 있는 링크를 제공할 것입니다.
W&B는 개인 및 학술 프로젝트에 무료로 사용할 수 있습니다!


```python
!pip install wandb -Uq
```


```python
import wandb

wandb.login()
```

# 1️⃣단계. 스윕 정의

기본적으로, 스윕은 많은 하이퍼파라미터 값을 시도하는 전략과 그것을 평가하는 코드를 결합합니다.
_전략을 정의_해야 합니다.
[구성](https://docs.wandb.com/sweeps/configuration)의 형태로.

노트북에서 스윕을 설정할 때,
해당 구성 객체는 중첩된 사전입니다.
커맨드 라인을 통해 스윕을 실행할 때,
구성 객체는
[YAML 파일](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration)입니다.

스윕 구성을 함께 정의하는 방법을 살펴보겠습니다.
천천히 진행하여 각 구성 요소를 설명하는 기회를 갖겠습니다.
일반적인 스윕 파이프라인에서,
이 단계는 단일 할당으로 수행됩니다.

### 👈 `method` 선택

새로운 파라미터 값 선택을 위한 `method`를 정의하는 것이 첫 번째 단계입니다.

다음과 같은 검색 `methods`를 제공합니다:
*   **`grid` 검색** - 하이퍼파라미터 값의 모든 조합을 반복합니다.
매우 효과적이지만, 계산 비용이 많이 들 수 있습니다.
*   **`random` 검색** - 제공된 `distribution`에 따라 각 새로운 조합을 무작위로 선택합니다. 놀랍게도 효과적입니다!
*   **`bayes`ian 검색** - 메트릭 점수를 하이퍼파라미터의 함수로 하는 확률적 모델을 생성하고, 메트릭을 개선할 가능성이 높은 파라미터를 선택합니다. 연속 파라미터의 작은 수에 대해서는 잘 작동하지만 규모가 커질수록 성능이 저하됩니다.

우리는 `random`을 사용할 것입니다.


```python
sweep_config = {
    'method': 'random'
    }
```

`bayes`ian 스윕의 경우,
메트릭에 대해 조금 더 알려줄 필요가 있습니다.
메트릭의 `name`을 알아야 하며,
메트릭을 `minimize`할 것인지
(예: 제곱 오류인 경우)
아니면 `maximize`할 것인지
(예: 정확도인 경우) 알아야 합니다.


```python
metric = {
    'name': 'loss',
    'goal': 'minimize'   
    }

sweep_config['metric'] = metric
```

`bayes`ian 스윕을 실행하지 않는 경우에도,
나중에 마음을 바꾸거나,
6개월이나 6년 후에 스윕으로 돌아와서
`val_G_batch`가 높거나 낮아야 하는지 모르는 경우를 대비하여
`sweep_config`에 이를 포함하는 것이 좋습니다.
또한, 재현성을 위한 좋은 실천 방법입니다.

### 📃 하이퍼`parameters` 이름 지정

하이퍼파라미터의 새로운 값을 시도할 `method`를 선택한 후에는
그 `parameters`가 무엇인지 정의해야 합니다.

대부분의 경우, 이 단계는 간단합니다:
`parameter`에 이름을 지정하고
파라미터의 합법적인 `values` 목록을 지정하기만 하면 됩니다.

예를 들어, 네트워크의 `optimizer`를 선택할 때,
옵션의 수는 제한되어 있습니다.
여기서는 가장 인기 있는 두 가지 옵션인 `adam`과 `sgd`를 사용합니다.
무한히 많은 옵션이 있는 하이퍼파라미터도,
여기서와 같이 은닉 `layer_size`와 `dropout`에 대해
몇 가지 선택된 `values`만 시도하는 것이 일반적입니다.


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

이 스윕에서 변화시키고 싶지 않은 하이퍼파라미터가 있지만,
여전히 `sweep_config`에 설정하고 싶은 경우가 종종 있습니다.

이 경우, `value`를 직접 설정하면 됩니다:


```python
parameters_dict.update({
    'epochs': {
        'value': 1}
    })
```

`grid` 검색의 경우, 이것이 필요한 전부입니다.

`random` 검색의 경우,
주어진 실행에서 파라미터의 모든 `values`가 선택될 확률은 동일합니다.

만약 이것이 충분하지 않다면,
대신 명명된 `distribution`과 그 매개변수, 예를 들어 `normal` 분포의 평균 `mu`
및 표준편차 `sigma`를 지정할 수 있습니다.

[여기](https://docs.wandb.com/sweeps/configuration#distributions)에서 무작위 변수의 분포를 설정하는 방법에 대해 자세히 알아보세요.


```python
parameters_dict.update({
    'learning_rate': {
        # 0과 0.1 사이의 평평한 분포
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

완료되면, `sweep_config`은 우리가 시도하고자 하는 `parameters`와
그것들을 시도할 `method`를 정확히 지정하는 중첩된 사전입니다.


```python
import pprint

pprint.pprint(sweep_config)
```

하지만 구성 옵션은 이것뿐만이 아닙니다!

예를 들어, [HyperBand](https://arxiv.org/pdf/1603.06560.pdf) 스케줄링 알고리즘을 사용하여 실행을 `early_terminate`할 수 있는 옵션도 제공합니다. [여기](https://docs.wandb.com/sweeps/configuration#stopping-criteria)에서 더 자세히 알아보세요.

모든 구성 옵션 목록을 [여기](https://docs.wandb.com/library/sweeps/configuration)에서 찾을 수 있고, YAML 형식의 큰 예제 모음을 [여기](https://github.com/wandb/examples/tree/master/examples/keras/keras-cnn-fashion)에서 찾을 수 있습니다.

# 2️⃣단계. 스윕 초기화

검색 전략을 정의한 후, 이를 구현할 무언가를 설정할 시간입니다.

우리 스윕의 시계 작업장은 _Sweep Controller_로 알려져 있습니다.
각 실행이 완료될 때마다, 실행할 새로운 실행 세트에 대한 지침을 발행합니다.
이러한 지침은 실행을 실제로 수행하는 _에이전트_에 의해 수집됩니다.

일반적인 스윕에서, 컨트롤러는 _우리_ 기계에 존재하는 반면,
실행을 완료하는 에이전트는 _당신_ 기계(들)에 존재합니다.
이 작업 분담은 에이전트를 실행하는 기계를 추가함으로써 스윕을 쉽게 확장할 수 있게 합니다!

적절한 `sweep_config` 및 `project` 이름으로 `wandb.sweep`을 호출하여 스윕 컨트롤러를 시작할 수 있습니다.

이 함수는 나중에 에이전트를 이 컨트롤러에 할당하는 데 사용할 `sweep_id`를 반환합니다.

> _부가 설명_: 커맨드 라인에서, 이 함수는 다음과 같이 대체됩니다.
```python
wandb sweep config.yaml
```
[커맨드 라인에서 스윕 사용에 대해 자세히 알아보기 ➡](https://docs.wandb.ai/guides/sweeps/walkthrough)


```python
sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")
```

# 3️⃣단계. 스윕 에이전트 실행

### 💻 학습 절차 정의

스윕을 실제로 실행하기 전에,
그 값들을 사용하는 학습 절차를 정의해야 합니다.

아래 함수에서, PyTorch에서 간단한 완전 연결 신경망을 정의하고, 모델 메트릭을 로그하고, 성능과 출력을 시각화하며, 실험을 추적하기 위해 다음과 같은 `wandb` 도구를 추가합니다:
* [**`wandb.init()`**](https://docs.wandb.com/library/init) – 새로운 W&B 실행을 초기화합니다. 각 실행은 학습 함수의 단일 실행입니다.
* [**`wandb.config`**](https://docs.wandb.com/library/config) – 모든 하이퍼파라미터를 구성 객체에 저장하여 로그합니다. `wandb.config` 사용 방법에 대해 자세히 알아보려면 [여기](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Configs_in_W%26B.ipynb)를 참조하세요.
* [**`wandb.log()`**](https://docs.wandb.com/library/log) – 모델 동작을 W&B에 로그합니다. 여기서는 성능만 로그하지만, `wandb.log`로 로그할 수 있는 다른 모든 리치 미디어는 [이 Colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Log_(Almost)_Anything_with_W%26B_Media.ipynb)에서 확인할 수 있습니다.

PyTorch와 W&B를 함께 사용하는 자세한 내용은 [이 Colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb)을 참조하세요.


```python
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(config=None):
    # 새로운 wandb 실행을 초기화
    with wandb.init(config=config):
        # wandb.agent에 의해 호출된 경우,
        # 이 구성은 스윕 컨트롤러에 의해 설정됩니다
        config = wandb.config

        loader = build_dataset(config.batch_size)
        network = build_network(config.fc_layer_size, config.dropout)
        optimizer = build_optimizer(network, config.optimizer, config.learning_rate)

        for epoch in range(config.epochs):
            avg_loss = train_epoch(network, loader, optimizer)
            wandb.log({"loss": avg_loss, "epoch": epoch})           
```

이 셀은 학습 절차의 네 부분을 정의합니다:
`build_dataset`, `build_network`, `build_optimizer`, `train_epoch`.

모두 기본 PyTorch 파이프라인의 표준 부분이며,
W&B 사용에 영향을 받지 않으므로 설명하지 않겠습니다.


```python
def build_dataset(batch_size):
   
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
    # MNIST 학습 데이터세트 다운로드
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
    cumu_loss = 0
    for _, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # ➡ 순방향 전달
        loss = F.nll_loss(network(data), target)
        cumu_loss += loss.item()

        # ⬅ 역방향 전달 + 가중치 업데이트
        loss.backward()
        optimizer.step()

        wandb.log({"batch loss": loss.item()})

    return cumu_loss / len(loader)
```

이제 스윕을 시작할 준비