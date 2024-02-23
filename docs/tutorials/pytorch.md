
# PyTorch
[Weights & Biases](https://wandb.com)를 사용하여 머신 러닝 실험 추적, 데이터세트 버전 관리 및 프로젝트 협업을 수행하세요.

[**Colab 노트북에서 시도해 보세요 →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb)

<div><img /></div>

<img src="https://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />

<div><img /></div>

## 이 노트북이 다루는 내용:

PyTorch 코드와 Weights & Biases를 통합하여 파이프라인에 실험 추적을 추가하는 방법을 보여줍니다.

## 결과적으로 나온 인터랙티브한 W&B 대시보드는 다음과 같습니다:
![](https://i.imgur.com/z8TK2Et.png)

## 의사코드에서 우리가 할 일은:
```python
# 라이브러리를 import합니다
import wandb

# 새 실험을 시작합니다
wandb.init(project="new-sota-model")

# config로 하이퍼파라미터의 사전을 캡처합니다
wandb.config = {"learning_rate": 0.001, "epochs": 100, "batch_size": 128}

# 모델과 데이터를 설정합니다
model, dataloader = get_model(), get_data()

# 선택사항: 그레이디언트를 추적합니다
wandb.watch(model)

for batch in dataloader:
  metrics = model.training_step()
  # 학습 루프 내에서 메트릭을 로그하여 모델 성능을 시각화합니다
  wandb.log(metrics)

# 선택사항: 마지막에 모델을 저장합니다
model.to_onnx()
wandb.save("model.onnx")
```

## [비디오 튜토리얼을 따라하세요](http://wandb.me/pytorch-video)!
**참고**: _Step_으로 시작하는 섹션들은 기존 파이프라인에 W&B를 통합하는 데 필요한 모든 것입니다. 나머지는 데이터를 로드하고 모델을 정의하는 것입니다.

# 🚀 설치, Import, 그리고 로그인


```python
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm.auto import tqdm

# 결정적인 동작을 보장합니다
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# 장치 구성
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# MNIST 미러 목록에서 느린 미러를 제거합니다
torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                      if not mirror.startswith("http://yann.lecun.com")]
```

### 0️⃣ 단계 0: W&B 설치

시작하려면 라이브러리를 받아야 합니다.
`wandb`는 `pip`을 사용하여 쉽게 설치할 수 있습니다.


```python
!pip install wandb onnx -Uq
```

### 1️⃣ 단계 1: W&B Import 및 로그인

우리의 웹 서비스에 데이터를 로그하기 위해서는,
로그인해야 합니다.

W&B를 처음 사용하는 경우,
등장하는 링크에서 무료 계정에 가입해야 합니다.


```
import wandb

wandb.login()
```

# 👩‍🔬 실험 및 파이프라인 정의

## 2️⃣ 단계 2: `wandb.init`으로 메타데이터 및 하이퍼파라미터 추적

프로그래밍적으로, 우리가 하는 첫 번째 일은 우리의 실험을 정의하는 것입니다:
하이퍼파라미터는 무엇인가요? 이 실행과 관련된 메타데이터는 무엇인가요?

이 정보를 `config` 사전에 저장하고 필요할 때마다 액세스하는 것이 일반적인 워크플로입니다.

이 예제에서는 몇 가지 하이퍼파라미터만 변경하고 나머지는 하드코딩합니다.
하지만 모델의 어떤 부분도 `config`의 일부가 될 수 있습니다!

우리는 또한 몇 가지 메타데이터를 포함합니다: 우리는 MNIST 데이터세트와 컨볼루션
아키텍처를 사용합니다. 나중에 같은 프로젝트에서 CIFAR에서 완전 연결 아키텍처로 작업한다면,
이것은 우리가 실행을 분리하는 데 도움이 될 것입니다.


```python
config = dict(
    epochs=5,
    classes=10,
    kernels=[16, 32],
    batch_size=128,
    learning_rate=0.005,
    dataset="MNIST",
    architecture="CNN")
```

이제 전체 파이프라인을 정의하겠습니다,
이는 모델 학습에 꽤 전형적입니다:

1. 우리는 먼저 모델을 `만들고`, 관련 데이터와 옵티마이저도 만들고,
2. 그에 따라 모델을 `학습`하고 마지막으로
3. 학습이 어떻게 진행되었는지 보기 위해 `테스트`합니다.

아래에서 이 함수들을 구현하겠습니다.


```python
def model_pipeline(hyperparameters):

    # wandb를 시작하라고 알립니다
    with wandb.init(project="pytorch-demo", config=hyperparameters):
      # 모든 HP를 wandb.config를 통해 액세스하므로 로깅이 실행과 일치합니다!
      config = wandb.config

      # 모델, 데이터 및 최적화 문제를 만듭니다
      model, train_loader, test_loader, criterion, optimizer = make(config)
      print(model)

      # 이를 사용하여 모델을 학습합니다
      train(model, train_loader, criterion, optimizer, config)

      # 최종 성능을 테스트합니다
      test(model, test_loader)

    return model
```

여기에서 표준 파이프라인과의 유일한 차이점은
모두 `wandb.init`의 컨텍스트 내에서 발생한다는 것입니다.
이 함수를 호출하면 코드와 서버 간의 통신 라인이 설정됩니다.

`config` 사전을 `wandb.init`에 전달하면
그 정보가 즉시 우리에게 로그되므로,
실험에 사용한 하이퍼파라미터 값이 항상 무엇인지 알 수 있습니다.

모델에서 선택하고 로그한 값이 항상 사용되는 값이 되도록 하기 위해,
`wandb.config` 사본을 사용하는 것이 좋습니다.
아래 `make`의 정의를 확인하면 몇 가지 예시를 볼 수 있습니다.

> *사이드 노트*: 우리는 코드가 별도의 프로세스에서 실행되도록 주의를 기울입니다,
그래서 우리 쪽에 문제가 있어도
(예: 거대한 해양 몬스터가 우리 데이터 센터를 공격)
당신의 코드가 충돌하지 않습니다.
문제가 해결되면 (예: 크라켄이 깊은 바다로 돌아감)
`wandb sync`로 데이터를 로그할 수 있습니다.


```python
def make(config):
    # 데이터를 만듭니다
    train, test = get_data(train=True), get_data(train=False)
    train_loader = make_loader(train, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)

    # 모델을 만듭니다
    model = ConvNet(config.kernels, config.classes).to(device)

    # 손실 및 옵티마이저를 만듭니다
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, test_loader, criterion, optimizer
```

# 📡 데이터 로딩 및 모델 정의

이제 데이터가 어떻게 로드되는지 및 모델이 어떤 모습인지를 명시해야 합니다.

이 부분은 매우 중요하지만,
`wandb` 없이도 똑같을 것이므로,
우리는 이에 대해 크게 언급하지 않겠습니다.


```python
def get_data(slice=5, train=True):
    전체 데이터세트 = torchvision.datasets.MNIST(root=".",
                                              train=train, 
                                              transform=transforms.ToTensor(),
                                              download=True)
    # [::slice]로 슬라이싱하는 것과 동등합니다
    서브 데이터세트 = torch.utils.data.Subset(
      전체 데이터세트, indices=range(0, len(전체 데이터세트), slice))
    
    return 서브 데이터세트


def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size, 
                                         shuffle=True,
                                         pin_memory=True, num_workers=2)
    return loader
```

모델을 정의하는 것은 보통 재미있는 부분입니다!

하지만 `wandb`와 함께라도 변하는 것은 없으므로,
우리는 표준 ConvNet 아키텍처를 고수할 것입니다.

이것을 수정하고 몇 가지 실험을 해보지 마십시오 --
모든 결과는 [wandb.ai](https://wandb.ai)에 로그될 것입니다!




```python
# 전통적이고 컨볼루션 신경망

class ConvNet(nn.Module):
    def __init__(self, kernels, classes=10):
        super(ConvNet, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, kernels[0], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, kernels[1], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * kernels[-1], classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
```

# 👟 학습 로직 정의

우리의 `model_pipeline`에서 계속해서 `학습`을 어떻게 하는지 명시할 차례입니다.

여기에는 두 가지 `wandb` 함수가 사용됩니다: `watch`와 `log`.

### 3️⃣ 단계 3. `wandb.watch`로 그레이디언트를 추적하고 `wandb.log`로 모든 것을 추적합니다

`wandb.watch`는 학습의 각 `log_freq` 단계마다 모델의 그레이디언트와 파라미터를 로그합니다.

학습을 시작하기 전에 호출하기만 하면 됩니다.

나머지 학습 코드는 동일하게 유지됩니다:
에포크와 배치를 반복하며,
순방향 및 역방향 패스를 실행하고
`옵티마이저`를 적용합니다.


```python
def train(model, loader, criterion, optimizer, config):
    # 모델이 무엇을 하는지 wandb에 알립니다: 그레이디언트, 가중치 등!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # 학습을 실행하고 wandb로 추적합니다
    total_batches = len(loader) * config.epochs
    example_ct = 0  # 본 예시의 수
    batch_ct = 0
    for epoch in tqdm(range(config.epochs)):
        for _, (images, labels) in enumerate(loader):

            loss = train_batch(images, labels, model, optimizer, criterion)
            example_ct +=  len(images)
            batch_ct += 1

            # 25번째 배치마다 메트릭을 보고합니다
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)


def train_batch(images, labels, model, optimizer, criterion):
    images, labels = images.to(device), labels.to(device)
    
    # 순방향 전달 ➡
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # 역방향 전달 ⬅
    optimizer.zero_grad()
    loss.backward()

    # 옵티마이저로 스텝을 진행합니다
    optimizer.step()

    return loss
```

로그 코드에서의 유일한 차이점은:
이전에는 터미널에 메트릭을 보고했을 수도 있지만,
이제는 동일한 정보를 `wandb.log`에 전달합니다.

`wandb.log`는 문자열을 키로 가지는 사전을 기대합니다.
이 문자열은 로그되는 객체를 식별하며, 이 객체들이 값으로 구성됩니다.
또한 선택적으로 학습의 어떤 `단계`에 있는지 로그할 수도 있습니다.

> *사이드 노트*: 저는 모델이 본 예시의 수를 사용하는 것을 선호합니다,
이는 배치 크기를 걸쳐 더 쉽게 비교할 수 있기 때문입니다,
하지만 원시 단계나 배치 수를 사용할 수도 있습니다. 더 긴 학습 실행의 경우, `에포크`별로 로그하는 것도 의미가 있을 수 있습니다.


```python
def train_log(loss, example_ct, epoch):
    # 마법이 일어나는 곳
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")
```

# 🧪 테스트 로직 정의

모델이 학습을 마친 후, 우리는 그것을 테스트하고 싶습니다:
생산에서 새로운 데이터에 대해 실행하거나,
일부 수작업으로 선별된 "어려운 예시"에 적용해 보세요.

#### 4️⃣ 선택적 단계 4: `wandb.save` 호출

이것은 또한 모델의 아키텍처와 최종 파라미터를 디스크에 저장하는 좋은 시간입니다.
최대 호환성을 위해, 우리는 모델을
[Open Neural Network Exchange (ONNX) 포맷](https://onnx.ai/)으로 `내보냅니다`.

해당 파일 이름을 `wandb.save`에 전달하면 모델 파라미터가
W&B 서버에 저장됩니다: 어떤 `.h5` 또는 `.pb`가 어떤 학습 실행과 대응하는지 더 이상 추적하지 않아도 됩니다!

모델을 저장, 버전 관리 및 배포하기 위한 `wandb`의 더 고급 기능에 대해서는,
[아