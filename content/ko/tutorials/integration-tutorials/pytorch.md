---
title: PyTorch
menu:
  tutorials:
    identifier: ko-tutorials-integration-tutorials-pytorch
    parent: integration-tutorials
weight: 1
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb" >}}

[W&B](https://wandb.ai)를 사용하여 기계학습 실험 트래킹, 데이터셋 버전 관리, 그리고 프로젝트 협업을 진행할 수 있습니다.

{{< img src="/images/tutorials/huggingface-why.png" alt="W&B 사용의 이점" >}}

## 이 노트북에서 다루는 내용

이 노트북에서는 W&B를 PyTorch 코드에 연동하여 파이프라인에 실험 트래킹 기능을 추가하는 방법을 안내합니다.

{{< img src="/images/tutorials/pytorch.png" alt="PyTorch와 W&B 연동 다이어그램" >}}

```python
# 라이브러리 import
import wandb

# 새로운 experiment 시작
with wandb.init(project="new-sota-model") as run:
 
    # 하이퍼파라미터 사전을 config로 저장
    run.config = {"learning_rate": 0.001, "epochs": 100, "batch_size": 128}

    # 모델과 데이터 셋업
    model, dataloader = get_model(), get_data()

    # 선택사항: gradient 트래킹
    run.watch(model)

    for batch in dataloader:
    metrics = model.training_step()
    # 트레이닝 루프 안에서 메트릭을 로그하여 모델 성능을 시각화
    run.log(metrics)

    # 선택사항: 마지막에 모델 저장
    model.to_onnx()
    run.save("model.onnx")
```

[비디오 튜토리얼](https://wandb.me/pytorch-video)을 따라 진행해보세요.

**참고:** _Step_으로 시작하는 섹션만 따라하면 W&B를 기존 파이프라인에 쉽게 연동할 수 있습니다. 나머지 코드는 데이터 로드와 모델 정의에 관한 부분입니다.

## 설치, import, 그리고 로그인

```python
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm.auto import tqdm

# 재현성 보장을 위한 설정
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# 디바이스 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# MNIST 미러 중 느린 미러 제거
torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                      if not mirror.startswith("http://yann.lecun.com")]
```

### Step 0: W&B 설치

시작하려면 라이브러리를 설치해야 합니다.
`wandb`는 `pip`으로 간단하게 설치할 수 있습니다.

```python
!pip install wandb onnx -Uq
```

### Step 1: W&B import 및 로그인

데이터를 웹 서비스에 로그하려면,
로그인이 필요합니다.

W&B를 처음 사용한다면,
나타나는 링크에서 무료 계정을 만들어야 합니다.

```
import wandb

wandb.login()
```

## Experiment 및 파이프라인 정의

### `wandb.init`으로 메타데이터와 하이퍼파라미터 트래킹

프로그래밍적으로, 가장 먼저 할 일은 experiment를 정의하는 것입니다:
어떤 하이퍼파라미터를 사용할 것인지? 이 run에 어떤 메타데이터가 연결되는지 결정합니다.

보통 이 정보는 `config` 딕셔너리(혹은 비슷한 오브젝트)에 저장하고
필요할 때 참조합니다.

이 예시에서는 일부 하이퍼파라미터만 바꿀 수 있도록 하는 대신
나머지는 하드코딩했습니다.
하지만 모델의 어느 부분이든 `config`에 포함시킬 수 있습니다.

또한 몇 가지 메타데이터를 추가합니다: 여기서는 MNIST 데이터셋과 컨볼루션 아키텍처를 사용했습니다. 나중에 동일 프로젝트에서 예를 들어 CIFAR에 fully connected 아키텍처를 사용할 경우,
run을 분리하는 데 도움이 됩니다.

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

이제 전체 파이프라인을 정의해봅시다.
이는 모델 트레이닝에서 흔히 볼 수 있는 구조입니다:

1. 먼저 모델과 관련 데이터, 옵티마이저를 `make`로 준비하고
2. 모델을 트레이닝하고
3. 트레이닝 결과를 평가합니다.

아래에서 이 함수들을 구현해봅니다.

```python
def model_pipeline(hyperparameters):

    # wandb에게 시작하라고 알리기
    with wandb.init(project="pytorch-demo", config=hyperparameters) as run:
        # run.config를 통해 모든 하이퍼파라미터에 엑세스하여 실행과 로그를 일치시킵니다.
        config = run.config

        # 모델, 데이터, 최적화 문제 정의
        model, train_loader, test_loader, criterion, optimizer = make(config)
        print(model)

        # 이들로 모델 트레이닝
        train(model, train_loader, criterion, optimizer, config)

        # 마지막 성능 테스트
        test(model, test_loader)

    return model
```

여기서 기존 파이프라인과의 유일한 차이점은
모든 과정이 `wandb.init` 컨텍스트 내에서 발생한다는 점입니다.
이 함수 호출은 코드와 W&B 서버 사이에 통신채널을 생성합니다.

`config` 딕셔너리를 `wandb.init`에 전달하면
입력값이 즉시 로그되므로,
experiment에 사용한 하이퍼파라미터 값을 항상 알 수 있습니다.

모델에서 항상 실제 사용한 값과 로그된 값이 일치하도록 하려면
`run.config` 오브젝트를 사용하는 것이 좋습니다.
아래 `make` 함수 구현에서 예시를 확인하세요.

> *참고:* W&B는 코드를 별도 프로세스에서 실행하므로,
(예를 들어 데이터센터를 거대한 괴물이 공격해도)
코드가 영향을 받지 않습니다.
문제가 해결된 뒤(예: 괴물이 물러난 뒤)엔 `wandb sync`로 데이터를 다시 로그할 수 있습니다.

```python
def make(config):
    # 데이터 준비
    train, test = get_data(train=True), get_data(train=False)
    train_loader = make_loader(train, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)

    # 모델 생성
    model = ConvNet(config.kernels, config.classes).to(device)

    # 손실 함수와 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, test_loader, criterion, optimizer
```

### 데이터 로딩과 모델 정의

이제 데이터를 어떻게 로드할지, 모델 아키텍처가 어떤지 지정할 차례입니다.

이 부분도 매우 중요하지만
`wandb` 없이도 마찬가지로 설정하는 부분이니,
자세한 설명은 생략하겠습니다.

```python
def get_data(slice=5, train=True):
    full_dataset = torchvision.datasets.MNIST(root=".",
                                              train=train, 
                                              transform=transforms.ToTensor(),
                                              download=True)
    #  [::slice]로 슬라이싱한 것과 동일
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

모델을 정의하는 부분은 언제나 즐거운 부분이죠.

하지만 여기서도 `wandb` 관련 변경사항은 없으니,
일반적인 ConvNet 아키텍처를 사용하겠습니다.

마음껏 실험해보세요 -- 모든 결과가 [wandb.ai](https://wandb.ai)에 자동으로 기록됩니다.

```python
# 전통적인 컨볼루션 신경망

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

이제 `model_pipeline`에서 실제 트레이닝 과정을 정의할 차례입니다.

여기서 `wandb`의 두 기능인 `watch`와 `log`가 사용됩니다.

## `run.watch()`로 gradient, `run.log()`로 모든 것 추적

`run.watch`는 트레이닝 중 정해진 주기(`log_freq`)마다
모델의 gradient와 파라미터 값을 자동으로 추적합니다.

트레이닝 시작 전에 한 번 호출해주면 됩니다.

나머지 트레이닝 코드에는 큰 변화가 없습니다:
에포크와 배치를 반복하면서,
forward/backward 패스를 실행하고,
`optimizer`를 적용합니다.

```python
def train(model, loader, criterion, optimizer, config):
    # 모델이 무엇을 하는지(gradient, weight 등) wandb에게 추적하라고 지시합니다.
    run = wandb.init(project="pytorch-demo", config=config)
    run.watch(model, criterion, log="all", log_freq=10)

    # wandb로 트레이닝을 추적
    total_batches = len(loader) * config.epochs
    example_ct = 0  # 처리한 예시 개수
    batch_ct = 0
    for epoch in tqdm(range(config.epochs)):
        for _, (images, labels) in enumerate(loader):

            loss = train_batch(images, labels, model, optimizer, criterion)
            example_ct +=  len(images)
            batch_ct += 1

            # 25번째 배치마다 메트릭 보고
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)


def train_batch(images, labels, model, optimizer, criterion):
    images, labels = images.to(device), labels.to(device)
    
    # 순방향 패스 ➡
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # 역방향 패스 ⬅
    optimizer.zero_grad()
    loss.backward()

    # 옵티마이저로 한 스텝
    optimizer.step()

    return loss
```

여기서 바뀐 점은 로그 함수뿐입니다. 
기존에는 터미널로만 메트릭을 출력했다면,
이제는 동일한 정보를 `run.log()`로 전달합니다.

`run.log()`에는 문자열 키를 갖는 딕셔너리를 전달하면 됩니다.
이 키로 어떤 객체가 로그되는지 식별할 수 있고,
원한다면 트레이닝의 `step`도 함께 기록할 수 있습니다.

> *참고:* 저는 보통 모델이 본 전체 예시 개수로 step을 지정합니다.
배치 크기가 달라도 비교가 쉽기 때문입니다. 
길게 트레이닝할 때는 raw step이나 batch 카운트, 혹은 epoch에 맞춰 로그해도 좋습니다.

```python
def train_log(loss, example_ct, epoch):
    with wandb.init(project="pytorch-demo") as run:
        # 손실과 에포크 번호를 로그
        # 여기서 W&B로 메트릭을 기록합니다
        run.log({"epoch": epoch, "loss": loss}, step=example_ct)
        print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")
```

### 테스트 로직 정의

트레이닝이 끝난 후에는 모델을 평가해야 합니다.
프로덕션에서 새 데이터를 넣어보거나,
특별히 선정한 예시로 정확도를 측정할 수 있습니다.

## (선택) `run.save()` 사용하기

이 때 모델의 아키텍처와 최종 파라미터를 디스크에 저장해두면 좋습니다.
최대 호환성을 위해
[ONNX(Open Neural Network eXchange) 포맷](https://onnx.ai/)으로 모델을 내보냅니다.

이 파일명을 `run.save()`에 전달하면,
모델 파라미터가 W&B 서버에 안전하게 저장됩니다.
어떤 `.h5`나 `.pb` 파일이 어떤 training run 결과인지
잊어버릴 걱정이 없습니다.

모델을 저장, 버전 관리, 배포하는 W&B의 고급 기능은
[Artifacts 도구](https://www.wandb.com/artifacts)를 참고하세요.

```python
def test(model, test_loader):
    model.eval()

    with wandb.init(project="pytorch-demo") as run:
        # 테스트용 데이터로 모델 실행
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
            
            run.log({"test_accuracy": correct / total})

        # ONNX 포맷으로 모델 저장
        torch.onnx.export(model, images, "model.onnx")
        run.save("model.onnx")
```

### 트레이닝 실행 및 wandb.ai에서 메트릭 실시간 관찰

이제 전체 파이프라인을 정의했고,
W&B 코드를 몇 줄만 추가했으니,
완전히 트래킹되는 experiment를 실행할 준비가 되었습니다.

몇 가지 유용한 링크도 제공됩니다:
문서,
Project 페이지 (프로젝트 안의 모든 run 정리),
Run 페이지 (이 run의 결과 저장).

Run 페이지에서 이런 탭들을 확인할 수 있습니다:

1. **Charts**: 모델 gradient, 파라미터 값, 손실 등이 트레이닝 동안 실시간으로 기록됩니다
2. **System**: 디스크 I/O, CPU, GPU 등 다양한 시스템 메트릭 제공(온도 변화를 확인해보세요)
3. **Logs**: 트레이닝 중 표준 출력을 로그로 남깁니다
4. **Files**: 트레이닝이 끝나면, `model.onnx` 파일을 클릭하여 [Netron model viewer](https://github.com/lutzroeder/netron)로 네트워크 구조를 확인할 수 있습니다.

`with wandb.init` 블록이 끝나면 실행도 종료되고,
결과 요약도 셀 출력에서 확인할 수 있습니다.

```python
# 파이프라인으로 모델 빌드, 트레이닝, 분석 실행
model = model_pipeline(config)
```

### Sweeps로 하이퍼파라미터 실험

여기 예제에서는 하나의 하이퍼파라미터 조합만 살펴봤습니다.
그러나 대부분의 ML 워크플로우에서는
여러 하이퍼파라미터를 반복적으로 실험하는 것이 중요합니다.

W&B Sweeps를 사용하면 하이퍼파라미터 테스트를 자동화하고,
다양한 모델 및 최적화 전략 가능성을 쉽게 탐색할 수 있습니다.

[W&B Sweeps로 하이퍼파라미터 최적화하는 Colab 노트북](https://wandb.me/sweeps-colab)도 참고해보세요.

W&B를 이용한 하이퍼파라미터 sweep은 매우 간단하며, 단 3단계면 준비 완료입니다:

1. **스윕 정의:** 
   탐색할 파라미터, 검색 전략, 최적화 메트릭 등이 담긴 딕셔너리 또는 [YAML 파일]({{< relref path="/guides/models/sweeps/define-sweep-configuration" lang="ko" >}})을 생성합니다.

2. **스윕 초기화:**  
   `sweep_id = wandb.sweep(sweep_config)`

3. **스윕 에이전트 실행:**  
   `wandb.agent(sweep_id, function=train)`

이것만으로 하이퍼파라미터 스윕이 실행됩니다.

{{< img src="/images/tutorials/pytorch-2.png" alt="PyTorch 트레이닝 대시보드" >}}

## 예제 갤러리

W&B로 트래킹하고 시각화한 여러 프로젝트 예제는 [갤러리 →](https://app.wandb.ai/gallery)에서 볼 수 있습니다.

## 고급 설정
1. [환경 변수]({{< relref path="/guides/hosting/env-vars/" lang="ko" >}}): API 키를 환경 변수에 저장해 관리형 클러스터에서 트레이닝을 실행할 수 있습니다.
2. [오프라인 모드]({{< relref path="/support/kb-articles/run_wandb_offline.md" lang="ko" >}}): `dryrun` 모드로 오프라인에서 트레이닝 후 결과를 나중에 동기화할 수 있습니다.
3. [온프레미스]({{< relref path="/guides/hosting/hosting-options/self-managed" lang="ko" >}}): W&B를 프라이빗 클라우드 또는 자체 인프라 내 air-gapped 서버에 설치할 수 있습니다. 학계부터 엔터프라이즈 팀까지 모두를 위한 로컬 설치가 준비되어 있습니다.
4. [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ko" >}}): 하이퍼파라미터 탐색을 위한 경량화된 튜닝 툴을 빠르게 설정할 수 있습니다.