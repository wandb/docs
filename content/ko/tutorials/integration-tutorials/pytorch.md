---
title: PyTorch
menu:
  tutorials:
    identifier: ko-tutorials-integration-tutorials-pytorch
    parent: integration-tutorials
weight: 1
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb" >}}

기계 학습 실험 추적, 데이터셋 버전 관리 및 프로젝트 협업을 위해 [Weights & Biases](https://www.wandb.ai)를 사용하세요.

{{< img src="/images/tutorials/huggingface-why.png" alt="" >}}

## 이 노트북에서 다루는 내용

Weights & Biases 와 PyTorch 코드를 통합하여 파이프라인에 실험 추적을 추가하는 방법을 보여줍니다.

{{< img src="/images/tutorials/pytorch.png" alt="" >}}

```python
# 라이브러리 임포트
import wandb

# 새로운 실험 시작
wandb.init(project="new-sota-model")

# config 로 하이퍼파라미터 사전을 캡처합니다
wandb.config = {"learning_rate": 0.001, "epochs": 100, "batch_size": 128}

# 모델 및 데이터 설정
model, dataloader = get_model(), get_data()

# 선택 사항: 그레이디언트 추적
wandb.watch(model)

for batch in dataloader:
  metrics = model.training_step()
  # 모델 성능을 시각화하기 위해 트레이닝 루프 내에서 메트릭을 기록합니다
  wandb.log(metrics)

# 선택 사항: 마지막에 모델 저장
model.to_onnx()
wandb.save("model.onnx")
```

[비디오 가이드](http://wandb.me/pytorch-video)를 따라해보세요.

**참고**: _Step_ 으로 시작하는 섹션은 기존 파이프라인에 W&B 를 통합하는 데 필요한 전부입니다. 나머지는 데이터를 로드하고 모델을 정의합니다.

## 설치, 임포트 및 로그인

```python
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm.auto import tqdm

# 결정론적 동작 보장
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

### 0단계: W&B 설치

시작하려면 라이브러리를 가져와야 합니다.
`wandb` 는 `pip` 를 사용하여 쉽게 설치할 수 있습니다.

```python
!pip install wandb onnx -Uq
```

### 1단계: W&B 임포트 및 로그인

웹 서비스에 데이터를 기록하려면
로그인해야 합니다.

W&B 를 처음 사용하는 경우
나타나는 링크에서 무료 계정에 가입해야 합니다.

```
import wandb

wandb.login()
```

## 실험 및 파이프라인 정의

### `wandb.init` 로 메타데이터 및 하이퍼파라미터 추적

프로그래밍 방식으로 가장 먼저 하는 일은 실험을 정의하는 것입니다.
하이퍼파라미터는 무엇입니까? 이 run 과 연결된 메타데이터는 무엇입니까?

이 정보를 `config` 사전에 저장하는 것은 매우 일반적인 워크플로우입니다
(또는 유사한 오브젝트)
필요에 따라 엑세스합니다.

이 예에서는 몇 가지 하이퍼파라미터만 변경하고
나머지는 수동으로 코딩합니다.
그러나 모델의 모든 부분이 `config` 의 일부가 될 수 있습니다.

또한 MNIST 데이터셋과 컨볼루션
아키텍처를 사용하고 있다는 메타데이터도 포함합니다. 나중에 동일한 프로젝트에서 CIFAR 에 대한 완전 연결 아키텍처를 사용하는 경우
run 을 분리하는 데 도움이 됩니다.

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

이제 전체 파이프라인을 정의해 보겠습니다.
이는 모델 트레이닝에 매우 일반적입니다.

1. 먼저 모델과 관련 데이터 및 옵티마이저를 `make` 한 다음
2. 모델을 적절하게 `train` 하고 마지막으로
3. `test` 하여 트레이닝이 어떻게 진행되었는지 확인합니다.

이러한 함수는 아래에서 구현합니다.

```python
def model_pipeline(hyperparameters):

    # wandb 에게 시작하라고 알립니다
    with wandb.init(project="pytorch-demo", config=hyperparameters):
      # wandb.config 를 통해 모든 HP 에 엑세스하므로 로깅이 실행과 일치합니다.
      config = wandb.config

      # 모델, 데이터 및 최적화 문제 만들기
      model, train_loader, test_loader, criterion, optimizer = make(config)
      print(model)

      # 모델을 트레이닝하는 데 사용합니다
      train(model, train_loader, criterion, optimizer, config)

      # 최종 성능 테스트
      test(model, test_loader)

    return model
```

표준 파이프라인과의 유일한 차이점은
모두 `wandb.init` 의 컨텍스트 내에서 발생한다는 것입니다.
이 함수를 호출하면 코드와 서버 간에 통신 라인이 설정됩니다.

`config` 사전을 `wandb.init` 에 전달하면
해당 정보가 즉시 기록되므로
실험에서 사용할 하이퍼파라미터 값을 항상 알 수 있습니다.

선택하고 기록한 값이 항상 모델에서 사용되는지 확인하려면
오브젝트의 `wandb.config` 사본을 사용하는 것이 좋습니다.
몇 가지 예제를 보려면 아래 `make` 의 정의를 확인하세요.

> *참고*: 코드를 별도의 프로세스에서 실행하도록 주의합니다.
따라서 당사 측의 문제
(예: 거대한 바다 괴물이 데이터 센터를 공격하는 경우)
코드가 충돌하지 않습니다.
크라켄이 심해로 돌아갈 때와 같이 문제가 해결되면
`wandb sync` 로 데이터를 기록할 수 있습니다.

```python
def make(config):
    # 데이터 만들기
    train, test = get_data(train=True), get_data(train=False)
    train_loader = make_loader(train, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)

    # 모델 만들기
    model = ConvNet(config.kernels, config.classes).to(device)

    # 손실 및 옵티마이저 만들기
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, test_loader, criterion, optimizer
```

### 데이터 로딩 및 모델 정의

이제 데이터를 로드하는 방법과 모델의 모양을 지정해야 합니다.

이 부분은 매우 중요하지만
`wandb` 가 없으면 달라지지 않으므로
자세히 설명하지 않겠습니다.

```python
def get_data(slice=5, train=True):
    full_dataset = torchvision.datasets.MNIST(root=".",
                                              train=train, 
                                              transform=transforms.ToTensor(),
                                              download=True)
    #  [::slice] 로 슬라이싱하는 것과 같습니다
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

모델을 정의하는 것은 일반적으로 재미있는 부분입니다.

하지만 `wandb` 에서는 아무것도 변경되지 않으므로
표준 ConvNet 아키텍처를 고수할 것입니다.

이것저것 만지작거리고 실험을 두려워하지 마세요.
모든 결과는 [wandb.ai](https://wandb.ai) 에 기록됩니다.

```python
# 기존 컨볼루션 신경망

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

### 트레이닝 로직 정의

`model_pipeline` 에서 계속 진행하여 `train` 방법을 지정할 시간입니다.

여기서 두 개의 `wandb` 함수가 사용됩니다. `watch` 와 `log` 입니다.

## `wandb.watch` 로 그레이디언트를 추적하고 `wandb.log` 로 다른 모든 것을 추적합니다

`wandb.watch` 는 트레이닝의 모든 `log_freq` 단계에서 모델의 그레이디언트와 파라미터를 기록합니다.

트레이닝을 시작하기 전에 호출하기만 하면 됩니다.

나머지 트레이닝 코드는 동일하게 유지됩니다.
에포크와 배치를 반복하고,
forward 및 backward 패스를 실행하고
`옵티마이저` 를 적용합니다.

```python
def train(model, loader, criterion, optimizer, config):
    # wandb 에게 모델이 무엇을 하는지 (그레이디언트, 가중치 등) 감시하라고 지시합니다.
    wandb.watch(model, criterion, log="all", log_freq=10)

    # wandb 로 트레이닝을 실행하고 추적합니다
    total_batches = len(loader) * config.epochs
    example_ct = 0  # 확인된 예제 수
    batch_ct = 0
    for epoch in tqdm(range(config.epochs)):
        for _, (images, labels) in enumerate(loader):

            loss = train_batch(images, labels, model, optimizer, criterion)
            example_ct +=  len(images)
            batch_ct += 1

            # 25번째마다 메트릭 보고
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

    # 옵티마이저로 단계 진행
    optimizer.step()

    return loss
```

유일한 차이점은 로깅 코드에 있습니다.
이전에는 터미널에 인쇄하여 메트릭을 보고했을 수 있지만,
이제 동일한 정보를 `wandb.log` 에 전달합니다.

`wandb.log` 는 문자열을 키로 사용하는 사전을 예상합니다.
이러한 문자열은 기록되는 오브젝트를 식별하며, 값을 구성합니다.
선택적으로 트레이닝의 `step` 을 기록할 수도 있습니다.

> *참고*: 모델이 확인한 예제 수를 사용하는 것을 좋아합니다.
배치 크기 간에 더 쉽게 비교할 수 있기 때문입니다.
하지만 원시 단계 또는 배치 수를 사용할 수 있습니다. 트레이닝 run 이 더 길어지면 `에포크` 로 기록하는 것이 좋습니다.

```python
def train_log(loss, example_ct, epoch):
    # 마법이 일어나는 곳
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")
```

### 테스팅 로직 정의

모델 트레이닝이 완료되면 테스트해야 합니다.
프로덕션의 최신 데이터에 대해 실행하거나
수동으로 큐레이팅된 예제에 적용합니다.

## (선택 사항) `wandb.save` 호출

모델 아키텍처를 저장하는 것도 좋은 시기입니다
디스크에 최종 파라미터를 저장합니다.
최대 호환성을 위해 모델을
[ONNX (Open Neural Network eXchange) 형식](https://onnx.ai/)으로 `내보냅니다`.

해당 파일 이름을 `wandb.save` 에 전달하면 모델 파라미터가
W&B 서버에 저장됩니다. 더 이상 어떤 `.h5` 또는 `.pb` 가
어떤 트레이닝 run 에 해당하는지 추적하지 않아도 됩니다.

모델 저장, 버전 관리 및 배포를 위한 더 고급 `wandb` 기능은
[Artifacts 툴](https://www.wandb.ai/artifacts)을 확인하세요.

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

### wandb.ai 에서 트레이닝을 실행하고 메트릭을 실시간으로 확인하세요

전체 파이프라인을 정의하고
몇 줄의 W&B 코드를 추가했으므로
완전히 추적된 실험을 실행할 준비가 되었습니다.

몇 가지 링크를 보고합니다.
당사의 설명서,
프로젝트의 모든 run 을 구성하는 프로젝트 페이지,
이 run 의 결과가 저장될 Run 페이지입니다.

Run 페이지로 이동하여 다음 탭을 확인하세요.

1. **차트**, 트레이닝 전반에 걸쳐 모델 그레이디언트, 파라미터 값 및 손실이 기록됩니다.
2. **시스템**, 디스크 I/O 활용률, CPU 및 GPU 메트릭 (온도가 급상승하는 것을 확인하세요) 등을 포함한 다양한 시스템 메트릭이 포함되어 있습니다.
3. **로그**, 트레이닝 중에 표준 출력으로 푸시된 모든 항목의 사본이 있습니다.
4. **파일**, 트레이닝이 완료되면 `model.onnx` 를 클릭하여 [Netron 모델 뷰어](https://github.com/lutzroeder/netron)로 네트워크를 볼 수 있습니다.

`with wandb.init` 블록이 종료될 때 run 이 완료되면
셀 출력에 결과 요약도 인쇄합니다.

```python
# 파이프라인으로 모델 빌드, 트레이닝 및 분석
model = model_pipeline(config)
```

### Sweeps 로 하이퍼파라미터 테스트

이 예에서는 하이퍼파라미터의 단일 세트만 살펴보았습니다.
그러나 대부분의 ML 워크플로우에서 중요한 부분은
여러 하이퍼파라미터를 반복하는 것입니다.

Weights & Biases Sweeps 를 사용하여 하이퍼파라미터 테스팅을 자동화하고 가능한 모델 및 최적화 전략 공간을 탐색할 수 있습니다.

## [W&B Sweeps 를 사용하여 PyTorch 에서 하이퍼파라미터 최적화 확인](http://wandb.me/sweeps-colab)

Weights & Biases 로 하이퍼파라미터 스윕을 실행하는 것은 매우 쉽습니다. 다음과 같은 3가지 간단한 단계가 있습니다.

1. **스윕 정의**: 검색할 파라미터, 검색 전략, 최적화 메트릭 등을 지정하는 사전 또는 [YAML 파일]({{< relref path="/guides/models/sweeps/define-sweep-configuration" lang="ko" >}})을 만들어 이 작업을 수행합니다.

2. **스윕 초기화**:
`sweep_id = wandb.sweep(sweep_config)`

3. **스윕 에이전트 실행**:
`wandb.agent(sweep_id, function=train)`

하이퍼파라미터 스윕을 실행하는 데 필요한 전부입니다.

{{< img src="/images/tutorials/pytorch-2.png" alt="" >}}

## 예제 갤러리

[갤러리 →](https://app.wandb.ai/gallery)에서 W&B 로 추적 및 시각화된 프로젝트의 예제를 확인하세요

## 고급 설정
1. [환경 변수]({{< relref path="/guides/hosting/env-vars/" lang="ko" >}}): 관리형 클러스터에서 트레이닝을 실행할 수 있도록 환경 변수에 API 키를 설정합니다.
2. [오프라인 모드]({{< relref path="/support/kb-articles/run_wandb_offline.md" lang="ko" >}}): `dryrun` 모드를 사용하여 오프라인으로 트레이닝하고 나중에 결과를 동기화합니다.
3. [온프레미스]({{< relref path="/guides/hosting/hosting-options/self-managed" lang="ko" >}}): 프라이빗 클라우드 또는 자체 인프라의 에어 갭 서버에 W&B 를 설치합니다. 학계에서 엔터프라이즈 팀에 이르기까지 모든 사람을 위한 로컬 설치가 있습니다.
4. [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ko" >}}): 튜닝을 위한 경량 툴을 사용하여 하이퍼파라미터 검색을 빠르게 설정합니다.
