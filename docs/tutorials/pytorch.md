---
title: PyTorch
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb"></CTAButtons>

[Weights & Biases](https://wandb.com)를 사용하여 기계학습 실험 추적, 데이터셋 버전 관리 및 프로젝트 협업을 수행하세요.

![](/images/tutorials/huggingface-why.png)

## 이 노트북에서 다루는 내용:

PyTorch 코드에 Weights & Biases를 통합하여 실험 추적을 파이프라인에 추가하는 방법을 보여드립니다.

## 결과로 나오는 인터랙티브 W&B 대시보드는 다음과 같습니다:

![](/images/tutorials/pytorch.png)

## 의사코드로, 우리가 할 일은:

```python
# 라이브러리 import
import wandb

# 새로운 실험 시작
wandb.init(project="new-sota-model")

# 하이퍼파라미터를 config로 저장
wandb.config = {"learning_rate": 0.001, "epochs": 100, "batch_size": 128}

# 모델과 데이터 설정
model, dataloader = get_model(), get_data()

# 선택: 그레이디언트 추적
wandb.watch(model)

for batch in dataloader:
  metrics = model.training_step()
  # 트레이닝 루프 내에서 메트릭을 기록하여 모델 성능 시각화
  wandb.log(metrics)

# 선택: 마지막에 모델 저장
model.to_onnx()
wandb.save("model.onnx")
```

## [비디오 튜토리얼](http://wandb.me/pytorch-video)과 함께 따라하세요!
**참고**: _Step_으로 시작하는 섹션은 기존 파이프라인에 W&B를 통합하기 위해 필요한 모든 것입니다. 나머지는 데이터 로드 및 모델 정의입니다.

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

# 결정적 행동 보장
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# 기기 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# MNIST 미러 리스트에서 느린 미러 제거
torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                      if not mirror.startswith("http://yann.lecun.com")]
```

### 0️⃣ Step 0: W&B 설치

시작하려면, 라이브러리를 가져와야 합니다.
`wandb`는 `pip`을 통해 쉽게 설치할 수 있습니다.

```python
!pip install wandb onnx -Uq
```

### 1️⃣ Step 1: W&B 가져오기 및 로그인

데이터를 웹 서비스에 로그하려면,
로그인이 필요합니다.

W&B를 처음 사용하는 경우, 화면에 나타나는 링크에서 무료 계정을 등록해야 합니다.

```
import wandb

wandb.login()
```

# 👩‍🔬 실험과 파이프라인 정의

## 2️⃣ Step 2: `wandb.init`으로 메타데이터와 하이퍼파라미터 추적

프로그래밍적 관점에서 보면, 우리가 먼저 하는 일은 실험을 정의하는 것입니다:
어떤 하이퍼파라미터를 사용할까요? 이 run과 관련된 메타데이터는 무엇일까요?

일반적인 워크플로우는 이러한 정보를 `config` 사전(또는 유사한 오브젝트)에 저장한 다음 필요할 때 액세스하는 것입니다.

이 예에서는 몇 개의 하이퍼파라미터만 다르게 하고 나머지는 하드코드했습니다.
그러나 모델의 어떤 부분이라도 `config`의 일부가 될 수 있습니다!

또한 몇 가지 메타데이터를 포함합니다: 우리는 MNIST 데이터셋과 컨볼루션 아키텍처를 사용하고 있습니다. 나중에 동일한 프로젝트에서 CIFAR와 같은 완전 연결 아키텍처로 작업할 경우, 이는 run을 분리하는 데 도움이 됩니다.

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

이제 전체 파이프라인을 정의해보겠습니다,
이는 모델 트레이닝에 대한 매우 일반적인 흐름입니다:

1. 먼저 관련 데이터 및 옵티마이저와 함께 모델을 `만드`, 그리고
2. 모델을 해당대로 `트레이닝`한 후,
3. 마지막으로 `테스트`하여 트레이닝이 어떻게 완료되었는지 확인합니다.

이 함수들은 아래에서 구현됩니다.

```python
def model_pipeline(hyperparameters):

    # wandb에게 시작하라고 알림
    with wandb.init(project="pytorch-demo", config=hyperparameters):
      # wandb.config를 통해 모든 HP에 액세스, 그래서 로그가 실행과 일치합니다!
      config = wandb.config

      # 모델, 데이터, 최적화 문제 설정
      model, train_loader, test_loader, criterion, optimizer = make(config)
      print(model)

      # 그리고 이를 사용하여 모델을 트레이닝
      train(model, train_loader, criterion, optimizer, config)

      # 그리고 최종 성능 테스트
      test(model, test_loader)

    return model
```

여기에서 표준 파이프라인과의 유일한 차이점은
모두 `wandb.init` 컨텍스트 내에서 발생한다는 것입니다.
이 함수를 호출하여
코드와 서버 간의 통신 라인을 설정합니다.

`config` 사전을 `wandb.init`에 전달함으로써
즉시 모든 정보를 우리에게 로그하며,
설정한 하이퍼파라미터 값을 항상 알 수 있습니다.

선택한 값이 항상 모델에서 사용되는지 확실히 하려면
당신의 오브젝트 사본인 `wandb.config`를 사용하는 것을 추천합니다.
아래의 `make` 정의를 참조하여 몇 가지 예를 확인하세요.

> *부가 참고*: 우리는 별도의 프로세스에서 코드를 실행하여,
우리 측에서 발생할 수 있는 문제로 인한
(예: 거대한 바다 괴물이 데이터 센터를 공격하는 경우)
코드 충돌을 방지합니다.
문제가 해결된 후
(예: 크라켄이 바다 깊숙이 돌아간 후)
`wandb sync`로 데이터를 로그할 수 있습니다.

```python
def make(config):
    # 데이터 생성
    train, test = get_data(train=True), get_data(train=False)
    train_loader = make_loader(train, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)

    # 모델 생성
    model = ConvNet(config.kernels, config.classes).to(device)

    # 손실 및 옵티마이저 생성
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, test_loader, criterion, optimizer
```

# 📡 데이터 로딩 및 모델 정의

이제 데이터가 어떻게 로드되는지와 모델이 어떻게 보이는지 지정해야 합니다.

이 부분은 매우 중요하지만,
`wandb`가 없을 때와 다르지 않으므로
이에 대해 깊이 다루지 않겠습니다.

```python
def get_data(slice=5, train=True):
    full_dataset = torchvision.datasets.MNIST(root=".",
                                              train=train, 
                                              transform=transforms.ToTensor(),
                                              download=True)
    # [::slice]로 슬라이싱 등가 
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

모델 정의는 보통 재미있는 부분입니다!

그러나 `wandb`와는 아무런 변화가 없으니,
표준 ConvNet 아키텍처로 진행해 보겠습니다.

이 부분에서 실험을 해서 결과를 확인하는 것을 두려워하지 마세요 --
모든 결과가 [wandb.ai](https://wandb.ai)에 기록될 것입니다!

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

# 👟 트레이닝 로직 정의

우리의 `model_pipeline`을 계속하여 `train`을 지정할 시간입니다.

여기서 두 가지 `wandb` 기능이 활용됩니다: `watch`와 `log`.

### 3️⃣ Step 3. `wandb.watch`와 `wandb.log`로 그레이디언트 및 모든 것 추적

`wandb.watch`는 트레이닝 중 매 `log_freq` 단계마다 모델의 그레이디언트와 파라미터를 기록합니다.

트레이닝을 시작하기 전에 호출만 하면 됩니다.

나머지 트레이닝 코드는 변하지 않습니다:
우리는 에포크와 배치를 반복하면서,
forward 및 backward 패스를 실행하여
우리의 `옵티마이저`를 적용합니다.

```python
def train(model, loader, criterion, optimizer, config):
    # 모델이 무엇을 하는지 지켜보라고 wandb에게 알림: 그레이디언트, 가중치, 등을!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # 트레이닝 실행 및 wandb로 추적
    total_batches = len(loader) * config.epochs
    example_ct = 0  # 본 예제 수
    batch_ct = 0
    for epoch in tqdm(range(config.epochs)):
        for _, (images, labels) in enumerate(loader):

            loss = train_batch(images, labels, model, optimizer, criterion)
            example_ct +=  len(images)
            batch_ct += 1

            # 매 25번째 배치 마다 메트릭 보고
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)


def train_batch(images, labels, model, optimizer, criterion):
    images, labels = images.to(device), labels.to(device)
    
    # Forward 패스 ➡
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # Backward 패스 ⬅
    optimizer.zero_grad()
    loss.backward()

    # 옵티마이저로 스텝
    optimizer.step()

    return loss
```

로그 코드에서의 유일한 차이점:
예전에는 터미널에 메트릭을 보고했을 수도 있지만,
이제는 동일한 정보를 `wandb.log`로 전달합니다.

`wandb.log`는 문자열을 키로 하는 사전을 예상합니다.
이 문자열들은 로그된 오브젝트를 식별하며, 값들을 이루고 있습니다.
또한 선택적으로 트레이닝 중인 `step`을 로그할 수 있습니다.

> *부가 참고*: 저는 모델이 본 예제의 수를 사용하는 것을 좋아합니다,
이는 배치 크기를 넘어서도 비교가 쉽기 때문입니다,
그러나 원시 단계나 배치 수를 사용할 수 있습니다. 더 긴 트레이닝 실행에서는 `epoch`에 따라 로그하는 것이 합리적일 수 있습니다.

```python
def train_log(loss, example_ct, epoch):
    # 마법이 일어나는 곳
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")
```

# 🧪 테스트 로직 정의

모델이 트레이닝을 완료한 후, 우리는 테스트하고자 합니다:
아마도 프로덕션에서의 새로운 데이터에 대한 실행,
또는 손으로 큐레이션한 "어려운 예제"에 적용하십시오.

#### 4️⃣ 선택 Step 4: `wandb.save` 호출

이 시점은 모델의 아키텍처와 최종 파라미터를 디스크에 저장하기에 좋은 시점입니다.
최대 호환성을 위해, [Open Neural Network eXchange (ONNX) 형식](https://onnx.ai/)으로 모델을 `내보내`겠습니다.

파일 이름을 `wandb.save`에 전달하면
모델 파라미터가 W&B의 서버에 저장되도록 보장합니다: 어떤 `.h5` 또는 `.pb`가 어느 트레이닝 run과 대응되는지를 잃지 마세요!

모델 저장, 버전 관리, 배포에 대한 W&B의 고급 기능은 [Artifacts 툴](https://www.wandb.com/artifacts)을 확인하세요.

```python
def test(model, test_loader):
    model.eval()

    # 일부 테스트 예제에서 모델 실행
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Accuracy of the model on the {total} " +
              f"test images: {correct / total:%}")
        
        wandb.log({"test_accuracy": correct / total})

    # 교환 가능한 ONNX 형식으로 모델 저장
    torch.onnx.export(model, images, "model.onnx")
    wandb.save("model.onnx")
```

# 🏃‍♀️ wandb.ai에서 실시간으로 메트릭을 보며 트레이닝 실행!

이제 전체 파이프라인을 정의하고
W&B 코드의 몇 줄을 추가했으니,
완벽히 추적된 실험을 실행할 준비가 되었습니다.

몇 가지 링크를 보고할 것입니다:
우리의 문서,
프로젝트 페이지는 프로젝트의 모든 run을 정리하고,
이 run의 결과가 저장될 Run 페이지.

Run 페이지로 이동하여 다음 탭을 확인하세요:

1. **Charts**, 여기서 모델 그레이디언트, 파라미터 값, 손실이 트레이닝 중에 로그됩니다
2. **System**, 디스크 IO 이용, CPU 및 GPU 메트릭(온도가 상승하는 것을 주목하세요 🔥), 등을 포함한 다양한 시스템 메트릭이 포함되어 있습니다
3. **Logs**, 트레이닝 중 표준 출력으로 푸시된 모든 항목의 사본이 있습니다
4. **Files**, 트레이닝이 완료되면, `model.onnx`를 클릭하여 [Netron 모델 뷰어](https://github.com/lutzroeder/netron)와 네트워크를 볼 수 있습니다.

run이 종료되면
(즉, `with wandb.init` 블록이 종료되면),
셀 출력에 결과 요약도 출력될 것입니다.

```
# 파이프라인으로 모델 구축, 훈련 및 분석
model = model_pipeline(config)
```

# 🧹 하이퍼파라미터를 Sweeps로 테스트

우리는 이 예제에서 하이퍼파라미터의 단일 세트만 보았습니다.
하지만 대부분의 ML 워크플로우의 중요한 부분은
다양한 하이퍼파라미터를 반복하는 것입니다.

Weights & Biases Sweeps을 사용하여 하이퍼파라미터 테스트를 자동화하고
가능한 모델 및 최적화 전략의 공간을 탐색할 수 있습니다.

## [PyTorch에서 W&B Sweeps를 사용한 하이퍼파라미터 최적화 확인하기](http://wandb.me/sweeps-colab)

Weights & Biases로 하이퍼파라미터 탐색을 실행하는 것은 매우 간단합니다. 다음은 3단계 간단합니다:

1. **탐색 정의:** 사전이나 검색 전략, 최적화 메트릭 등을 지정하는 [YAML 파일](/guides/sweeps/define-sweep-configuration)을 작성합니다.

2. **탐색 초기화:**
`sweep_id = wandb.sweep(sweep_config)`

3. **탐색 에이전트 실행:**
`wandb.agent(sweep_id, function=train)`

그리고 Voilà! 이것이 하이퍼파라미터 탐색을 실행하는 전부입니다!

![](/images/tutorials/pytorch-2.png)

# 🖼️ 예제 갤러리

W&B로 추적 및 시각화된 프로젝트의 예제를 [갤러리 →](https://app.wandb.ai/gallery)에서 확인하세요.

# 🤓 고급 설정
1. [환경 변수](/guides/hosting/env-vars): API 키를 환경 변수에 설정하여 관리형 클러스터에서 트레이닝을 실행할 수 있습니다.
2. [오프라인 모드](/guides/technical-faq/setup/#can-i-run-wandb-offline): `dryrun` 모드를 사용하여 오프라인으로 트레이닝하고 나중에 결과를 동기화하세요.
3. [온프레미스](/guides/hosting/hosting-options/self-managed): 프라이빗 클라우드 또는 자체 인프라의 공중 차단 서버에 W&B를 설치하십시오. 우리는 학계에서 엔터프라이즈 팀에 이르기까지 모든 사람을 위한 로컬 설치를 제공합니다.
4. [Sweeps](/guides/sweeps): 하이퍼파라미터 탐색을 빠르게 설정하기 위한 경량 툴을 사용하여 튜닝합니다.