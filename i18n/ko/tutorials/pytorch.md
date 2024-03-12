
# PyTorch
[Weights & Biases](https://wandb.com)를 사용하여 기계학습 실험 추적, 데이터셋 버전 관리 및 프로젝트 협업을 수행하세요.

[**여기서 Colab 노트북으로 시도해보세요 →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb)

<div><img /></div>

<img src="https://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />

<div><img /></div>

## 이 노트북이 다루는 내용:

이 노트북에서는 PyTorch 코드와 Weights & Biases를 통합하여 파이프라인에 실험 추적을 추가하는 방법을 보여줍니다.

## 결과적으로 나타나는 대화형 W&B 대시보드는 다음과 같습니다:
![](https://i.imgur.com/z8TK2Et.png)

## 의사코드로, 우리가 할 일은:
```python
# 라이브러리를 불러옵니다
import wandb

# 새로운 실험을 시작합니다
wandb.init(project="new-sota-model")

# 설정으로 하이퍼파라미터 사전을 캡처합니다
wandb.config = {"learning_rate": 0.001, "epochs": 100, "batch_size": 128}

# 모델과 데이터를 설정합니다
model, dataloader = get_model(), get_data()

# 선택적: 그레이디언트를 추적합니다
wandb.watch(model)

for batch in dataloader:
  metrics = model.training_step()
  # 트레이닝 루프 내에서 메트릭을 로그하여 모델 성능을 시각화합니다
  wandb.log(metrics)

# 선택적: 마지막에 모델을 저장합니다
model.to_onnx()
wandb.save("model.onnx")
```

## [비디오 튜토리얼을 따라해 보세요](http://wandb.me/pytorch-video)!
**주의**: _Step_으로 시작하는 섹션들은 기존 파이프라인에 W&B를 통합하기 위해 필요한 전부입니다. 나머지는 데이터를 로드하고 모델을 정의하는 것입니다.

# 🚀 설치, 가져오기 및 로그인


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

# 장치 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# MNIST 미러 목록에서 느린 미러 제거
torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                      if not mirror.startswith("http://yann.lecun.com")]
```

### 0️⃣ 단계 0: W&B 설치

시작하려면 라이브러리를 가져와야 합니다.
`wandb`는 `pip`을 사용하여 쉽게 설치할 수 있습니다.


```python
!pip install wandb onnx -Uq
```

### 1️⃣ 단계 1: W&B 가져오기 및 로그인

웹 서비스에 데이터를 로그하기 위해서는 로그인해야 합니다.

W&B를 처음 사용하는 경우 나타나는 링크에서 무료 계정에 가입해야 합니다.


```
import wandb

wandb.login()
```

# 👩‍🔬 실험 및 파이프라인 정의

## 2️⃣ 단계 2: `wandb.init`으로 메타데이터와 하이퍼파라미터 추적

프로그래밍적으로, 우리가 하는 첫 번째 일은 실험을 정의하는 것입니다:
어떤 하이퍼파라미터가 있는가? 이 run과 관련된 메타데이터는 무엇인가?

이 정보를 `config` 사전(또는 유사 객체)에 저장하고 필요에 따라 액세스하는 것이 일반적인 워크플로우입니다.

이 예제에서는 몇 가지 하이퍼파라미터만 변동시키고 나머지는 하드 코딩합니다.
하지만 모델의 어떤 부분도 `config`의 일부가 될 수 있습니다!

또한 일부 메타데이터를 포함합니다: 우리는 MNIST 데이터셋과 컨볼루셔널
아키텍처를 사용합니다. 나중에, 예를 들어,
같은 프로젝트에서 CIFAR에서 완전 연결 아키텍처를 작업할 경우,
이를 통해 우리의 run을 구분하는 데 도움이 됩니다.


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

이제 전체 파이프라인을 정의합시다,
이는 모델 트레이닝에 대해 꽤 전형적입니다:

1. 먼저 모델, 관련 데이터 및 옵티마이저를 `만듭니다`, 그리고
2. 모델을 `트레이닝`합니다, 그리고 마지막으로
3. 트레이닝이 어떻게 진행되었는지 보기 위해 `테스트`합니다.

이 함수들을 아래에서 구현하겠습니다.


```python
def model_pipeline(hyperparameters):

    # wandb가 시작되도록 알립니다
    with wandb.init(project="pytorch-demo", config=hyperparameters):
      # wandb.config를 통해 모든 HP에 액세스하므로, 로깅이 실행과 일치합니다!
      config = wandb.config

      # 모델, 데이터 및 최적화 문제를 만듭니다
      model, train_loader, test_loader, criterion, optimizer = make(config)
      print(model)

      # 그것들을 사용하여 모델을 트레이닝합니다
      train(model, train_loader, criterion, optimizer, config)

      # 최종 성능을 테스트합니다
      test(model, test_loader)

    return model
```

여기서 표준 파이프라인과 다른 점은
모든 것이 `wandb.init`의 컨텍스트 내에서 발생한다는 것입니다.
이 함수를 호출하면 코드와 서버 간의 통신 라인이 설정됩니다.

`config` 사전을 `wandb.init`에 전달하면
그 정보가 즉시 우리에게 로그됩니다,
그래서 어떤 하이퍼파라미터 값을 실험에 설정했는지 항상 알 수 있습니다.

모델에서 선택하고 로그한 값이 항상 사용되는 값이 되도록 하기 위해
`wandb.config` 복사본을 사용하는 것이 좋습니다.
`make`의 정의를 확인하여 몇 가지 예를 보세요.

> *부가적인 주의*: 우리의 코드를 별도의 프로세스에서 실행하도록 주의를 기울입니다,
그래서 우리 쪽에서 발생하는 문제
(예: 거대한 바다 괴물이 우리 데이터 센터를 공격함)
가 당신의 코드를 충돌시키지 않습니다.
문제가 해결되면 (예: 크라켄이 깊은 곳으로 돌아감)
`wandb sync`로 데이터를 로그할 수 있습니다.


```python
def make(config):
    # 데이터를 만듭니다
    train, test = get_data(train=True), get_data(train=False)
    train_loader = make_loader(train, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)

    # 모델을 만듭니다
    model = ConvNet(config.kernels, config.classes).to(device)

    # 손실과 옵티마이저를 만듭니다
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, test_loader, criterion, optimizer
```

# 📡 데이터 로딩 및 모델 정의

이제 데이터가 어떻게 로드되는지, 모델이 어떻게 생겼는지를 명시해야 합니다.

이 부분은 매우 중요하지만,
`wandb` 없이도 동일하므로,
이 부분에 대해 많이 다루지 않겠습니다.


```python
def get_data(slice=5, train=True):
    full_dataset = torchvision.datasets.MNIST(root=".",
                                              train=train, 
                                              transform=transforms.ToTensor(),
                                              download=True)
    # [::slice]로 슬라이싱하는 것과 동일합니다
    sub_dataset = torch.utils.data.Subset(
      full_dataset, indices=range(0, len(full_dataset), slice))
    
    return sub_dataset


def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size, 
                                         shuffle=True,
                                         pin_memory=True, num_workers=2)
    return loader
```

모델을 정의하는 것은 보통 재미있는 부분입니다!

하지만 `wandb`에서는 아무것도 변하지 않으므로,
표준 ConvNet 아키텍처를 사용할 것입니다.

이를 바꾸어 실험해보는 것을 두려워하지 마세요 --
모든 결과는 [wandb.ai](https://wandb.ai)에서 로그됩니다!




```python
# 전통적이고 컨볼루셔널한 신경망

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

# 👟 트레이닝 로직 정의

`model_pipeline`에서 `train`을 지정할 차례입니다.

여기서 두 가지 `wandb` 함수가 사용됩니다: `watch`와 `log`.

### 3️⃣ 단계 3. `wandb.watch`로 그레이디언트를 추적하고 `wandb.log`로 모든 것을 추적

`wandb.watch`는 트레이닝의 `log_freq` 단계마다 모델의 그레이디언트와 파라미터를 로그합니다.

트레이닝을 시작하기 전에 호출하기만 하면 됩니다.

트레이닝 코드의 나머지 부분은 동일하게 유지됩니다:
에포크와 배치를 반복하며,
forward 및 backward 패스를 실행하고
`옵티마이저`를 적용합니다.


```python
def train(model, loader, criterion, optimizer, config):
    # 모델이 무엇을 하는지 wandb에 알리세요: 그레이디언트, 가중치 등!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # 트레이닝을 실행하고 wandb로 추적합니다
    total_batches = len(loader) * config.epochs
    example_ct = 0  # 본 예제의 수
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
    
    # Forward pass ➡
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # 옵티마이저로 스텝합니다
    optimizer.step()

    return loss
```

차이점은 로깅 코드에 있습니다:
이전에는 터미널에 메트릭을 보고했을 수 있지만,
이제는 동일한 정보를 `wandb.log`에 전달합니다.

`wandb.log`는 문자열을 키로 가지는 사전을 기대합니다.
이 문자열은 로그되는 객체를 식별하며, 값으로 구성됩니다.
또한 선택적으로 트레이닝의 `step`을 로그할 수도 있습니다.

> *부가적인 주의*: 배치 크기를 통해 비교가 쉬워지므로 모델이 본 예제 수를 사용하는 것을 선호합니다,
하지만 원시 단계나 배치 수를 사용할 수도 있습니다. 더 긴 트레이닝 실행의 경우, `epoch`별로 로그하는 것도 의미가 있을 수 있습니다.


```python
def train_log(loss, example_ct, epoch):
    # 마법이 일어나는 곳
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"{str(example_ct).zfill(5)} 예제 후 손실: {loss:.3f}")
```

# 🧪 테스트 로직 정의

모델이 트레이닝을 마치면 테스트하고 싶습니다:
프로덕션에서 새로운 데이터에 모델을 실행하거나,
어려운 "하드 예제"에 적용해보세요.

#### 4️⃣ 선택적 단계 4: `wandb.save` 호출

이는 모델의 아키텍처와 최종 파라미터를 디스크에 저장하기에 좋은 시기입니다.
최대 호환성을 위해 모델을
[Open Neural Network Exchange (ONNX) 형식](https://onnx.ai/)으로 `내보냅니다`.

해당 파일 이름을 `wandb.save`에 전달하면 모델 파라미터가 W&B 서버에 저장됩니다: 어떤 `.h5` 또는 `.pb`
가 어떤 트레이닝 실행과 일치하는지 더 이상 헷갈리지 않습니다!

모델을 저장, 버전 관리 및 배포하기 위한 더 고급 `wandb` 기능에 대해서는 [Artifacts 도구](https://www.wandb.com/artifacts)를 확인하세요.


```python
def test(model, test_loader):
    model.eval()

    # 몇몇 테스트 예제에서 모델을 실행합니다
    with torch.no_grad():
        correct, total = 0