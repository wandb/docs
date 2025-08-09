---
title: 실험 추적하기
menu:
  tutorials:
    identifier: ko-tutorials-experiments
weight: 1
---

{{< cta-button 
    colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_&_Biases.ipynb" 
>}}

[W&B](https://wandb.ai/site)를 사용하여 기계학습 experiment 추적, 모델 체크포인팅, 팀과의 협업 등 다양한 작업을 진행해보세요.

이 노트북에서는 간단한 PyTorch 모델을 사용해 기계학습 experiment 를 생성하고 추적합니다. 노트북이 끝나면, 팀원들과 공유하고 맞춤 설정할 수 있는 대화형 Project 대시보드를 만들 수 있습니다. [예시 대시보드를 여기에서 확인하세요.](https://wandb.ai/wandb/wandb_example)

## 사전 준비

W&B Python SDK를 설치하고 로그인을 진행하세요:

```shell
!pip install wandb -qU
```

```python
# W&B 계정에 로그인합니다
import wandb
import random
import math

# wandb의 새로운 백엔드 임시 전환을 위해 wandb-core 사용
wandb.require("core")
```

```python
wandb.login()
```

## W&B로 기계학습 experiment 를 시뮬레이션하고 추적하기

기계학습 experiment 를 생성, 추적, 시각화하세요. 진행 방법은 다음과 같습니다:

1. [run]({{< relref path="/guides/models/track/runs/" lang="ko" >}}) 을 초기화하고 추적할 하이퍼파라미터를 전달합니다.
2. 트레이닝 루프 안에서, 정확도와 손실과 같은 메트릭을 로그합니다.

```python
import wandb
import random

project="basic-intro"
config = {
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10,
}

with wandb.init(project=project, config=config) as run:
  # 이 블록은 메트릭을 로그하는 트레이닝 루프를 시뮬레이션합니다
  epochs = 10
  offset = random.random() / 5
  for epoch in range(2, epochs):
      acc = 1 - 2 ** -epoch - random.random() / epoch - offset
      loss = 2 ** -epoch + random.random() / epoch + offset

      # 2️. 스크립트에서 W&B로 메트릭을 기록합니다
      run.log({"acc": acc, "loss": loss})    
```

W&B Project에서 기계학습 성능을 확인해보세요. 이전 셀에서 출력된 URL 링크를 복사해 브라우저에 붙여넣기 하면, 모델 성능을 그래프로 시각화한 대시보드를 확인할 수 있습니다.

아래 이미지는 대시보드의 예시 화면입니다:

{{< img src="/images/tutorials/experiments-1.png" alt="W&B experiment tracking dashboard" >}}

이제 W&B를 간단한 기계학습 트레이닝 루프에 통합하는 방법을 배웠으니, 기본적인 PyTorch 신경망으로 experiment 를 추적해보겠습니다. 아래 코드는 모델 체크포인트도 W&B로 업로드하여, 조직 내 다른 팀과도 쉽게 공유할 수 있습니다.

## PyTorch로 기계학습 experiment 추적하기

아래 코드 셀은 간단한 MNIST 분류기를 정의하고 트레이닝합니다. 트레이닝하는 동안 W&B가 출력한 URL을 클릭하여 실시간으로 결과가 W&B Project로 스트리밍되는 모습을 확인할 수 있습니다.

W&B run은 [메트릭]({{< relref path="/guides/models/track/runs/#workspace-tab" lang="ko" >}}),
시스템 정보,
[하이퍼파라미터]({{< relref path="/guides/models/track/runs/#overview-tab" lang="ko" >}}),
[터미널 출력]({{< relref path="/guides/models/track/runs/#logs-tab" lang="ko" >}}) 등을 자동으로 로그합니다.
또한, 모델 입력 및 출력이 담긴 [대화형 테이블]({{< relref path="/guides/models/tables/" lang="ko" >}})
도 확인할 수 있습니다.

### PyTorch Dataloader 세팅하기
아래 셀은 기계학습 모델을 트레이닝할 때 필요한 여러 함수들을 정의합니다. 이 함수들은 W&B에만 특화된 내용이 아니므로 여기서는 자세히 다루지 않습니다. [forward 및 backward 트레이닝 루프](https://pytorch.org/tutorials/beginner/nn_tutorial.html) 정의 방법, [PyTorch DataLoader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) 사용 방법, [`torch.nn.Sequential` 클래스](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html)를 통한 PyTorch 모델 정의법 등은 PyTorch 공식 문서를 참고하세요.

```python
import wandb
import torch, torchvision
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision.transforms as T

MNIST.mirrors = [
    mirror for mirror in MNIST.mirrors if "http://yann.lecun.com/" not in mirror
]

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_dataloader(is_train, batch_size, slice=5):
    "트레이닝 Dataloader 가져오기"
    full_dataset = MNIST(
        root=".", train=is_train, transform=T.ToTensor(), download=True
    )
    sub_dataset = torch.utils.data.Subset(
        full_dataset, indices=range(0, len(full_dataset), slice)
    )
    loader = torch.utils.data.DataLoader(
        dataset=sub_dataset,
        batch_size=batch_size,
        shuffle=True if is_train else False,
        pin_memory=True,
        num_workers=2,
    )
    return loader


def get_model(dropout):
    "간단한 모델"
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, 10),
    ).to(device)
    return model


def validate_model(model, valid_dl, loss_func, log_images=False, batch_idx=0):
    "검증 데이터셋에서 모델의 성능을 계산하고 wandb.Table을 로그합니다"
    model.eval()
    val_loss = 0.0
    with torch.inference_mode():
        correct = 0
        for i, (images, labels) in enumerate(valid_dl):
            images, labels = images.to(device), labels.to(device)

            # Forward 패스 ➡
            outputs = model(images)
            val_loss += loss_func(outputs, labels) * labels.size(0)

            # 정확도 계산 및 누적
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            # 대시보드에 한 배치 이미지를 로그 (항상 같은 batch_idx)
            if i == batch_idx and log_images:
                log_image_table(images, predicted, labels, outputs.softmax(dim=1))
    return val_loss / len(valid_dl.dataset), correct / len(valid_dl.dataset)
```

### 예측값과 실제 값을 비교하는 테이블 만들기

아래 함수는 W&B에만 특화된 것이므로 자세히 설명합니다.

셀에서는 `log_image_table`이라는 함수를 정의합니다. 이 함수는 W&B Table 오브젝트를 생성하며, 테이블 오브젝트를 사용해 이미지별로 모델이 예측한 값과 실제 라벨을 한눈에 보여줍니다.

좀 더 구체적으로, 각 행마다 모델에 입력된 이미지, 예측값, 실제값(라벨)이 담깁니다.

```python
def log_image_table(images, predicted, labels, probs):
    "wandb.Table에 (img, pred, target, score) 로그"
    # 이미지, 라벨, 예측값을 기록할 wandb Table 생성
    table = wandb.Table(
        columns=["image", "pred", "target"] + [f"score_{i}" for i in range(10)]
    )
    for img, pred, targ, prob in zip(
        images.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")
    ):
        table.add_data(wandb.Image(img[0].numpy() * 255), pred, targ, *prob.numpy())

    with wandb.init() as run:
        run.log({"predictions_table": table}, commit=False)
```

### 모델을 트레이닝하고 체크포인트 업로드하기

아래 코드는 모델을 트레이닝하며 체크포인트를 Project에 저장합니다. 저장된 체크포인트는 일반적으로 모델의 트레이닝 성능을 평가할 때 활용됩니다.

W&B를 이용하면 팀이나 조직 내 다른 멤버들과 쉽게 모델 및 모델 체크포인트도 공유할 수 있습니다. 팀 외부와 모델, 체크포인트를 공유하는 방법은 [W&B Registry]({{< relref path="/guides/core/registry/" lang="ko" >}}) 를 참고하세요.

```python
import wandb

config = {
    "epochs": 5,
    "batch_size": 128,
    "lr": 1e-3,
    "dropout": random.uniform(0.01, 0.80),
}

project = "pytorch-intro"

# wandb run을 초기화합니다
with wandb.init(project=project, config=config) as run:

    # config 복사(선택사항)
    config = run.config

    # 데이터 불러오기
    train_dl = get_dataloader(is_train=True, batch_size=config.batch_size)
    valid_dl = get_dataloader(is_train=False, batch_size=2 * config.batch_size)
    n_steps_per_epoch = math.ceil(len(train_dl.dataset) / config.batch_size)

    # 간단한 MLP 모델 정의
    model = get_model(config.dropout)

    # 손실 함수 및 옵티마이저 설정
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # 트레이닝 루프
    example_ct = 0
    step_ct = 0
    for epoch in range(config.epochs):
        model.train()
        for step, (images, labels) in enumerate(train_dl):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            train_loss = loss_func(outputs, labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            example_ct += len(images)
            metrics = {
                "train/train_loss": train_loss,
                "train/epoch": (step + 1 + (n_steps_per_epoch * epoch))
                / n_steps_per_epoch,
                "train/example_ct": example_ct,
            }

            if step + 1 < n_steps_per_epoch:
                # wandb로 트레인 메트릭 기록
                run.log(metrics)

            step_ct += 1

        val_loss, accuracy = validate_model(
            model, valid_dl, loss_func, log_images=(epoch == (config.epochs - 1))
        )

        # 트레인/검증 메트릭을 wandb에 로그
        val_metrics = {"val/val_loss": val_loss, "val/val_accuracy": accuracy}
        run.log({**metrics, **val_metrics})

        # 모델 체크포인트를 wandb에 저장
        torch.save(model, "my_model.pt")
        run.log_model(
            "./my_model.pt",
            "my_mnist_model",
            aliases=[f"epoch-{epoch+1}_dropout-{round(run.config.dropout, 4)}"],
        )

        print(
            f"Epoch: {epoch+1}, Train Loss: {train_loss:.3f}, Valid Loss: {val_loss:3f}, Accuracy: {accuracy:.2f}"
        )

    # 테스트 세트가 있다면 Summary 메트릭으로 이렇게 기록할 수 있습니다
    run.summary["test_accuracy"] = 0.8
```

이제 W&B로 첫 번째 모델을 트레이닝해보았습니다. 위의 링크 중 하나를 눌러 프로젝트의 메트릭을 확인하고, 저장된 모델 체크포인트가 W&B App UI의 Artifacts 탭에 잘 나타나는지 확인해보세요.

## (선택) W&B Alert 설정하기

코드에서 바로 Slack 또는 이메일로 알림을 받아보려면 [W&B Alerts]({{< relref path="/guides/models/track/runs/alert/" lang="ko" >}})를 활용하세요.

처음 알림을 보내기 위해서는 아래 두 단계를 따라주세요:

1) W&B [User Settings](https://wandb.ai/settings) 에서 Alerts를 켭니다.
2) 코드에 `run.alert()` 를 추가하세요. 예시는 다음과 같습니다:

```python
run.alert(title="Low accuracy", text=f"Accuracy is below the acceptable threshold")
```

아래 셀은 `run.alert()` 사용 예제를 보여줍니다.

```python
import wandb

# wandb run 시작
with wandb.init(project="pytorch-intro") as run:

    # 모델 트레이닝 루프 시뮬레이션
    acc_threshold = 0.3
    for training_step in range(1000):

        # 정확도를 임의로 발생
        accuracy = round(random.random() + random.random(), 3)
        print(f"Accuracy is: {accuracy}, {acc_threshold}")

        # wandb로 정확도 로그
        run.log({"Accuracy": accuracy})

        # 정확도가 임계값 이하면 W&B Alert 전송 후 run 종료
        if accuracy <= acc_threshold:
            # wandb Alert 전송
            run.alert(
                title="Low Accuracy",
                text=f"Accuracy {accuracy} at step {training_step} is below the acceptable threshold, {acc_threshold}",
            )
            print("Alert triggered")
            break
```

자세한 내용은 [W&B Alerts 개요]({{< relref path="/guides/models/track/runs/alert" lang="ko" >}})를 참고하세요.

## 다음 단계
다음 튜토리얼에서는 W&B Sweeps를 활용해 하이퍼파라미터 최적화(HPO)하는 방법을 배웁니다:
[PyTorch에서 하이퍼파라미터 스윕 사용하기](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb)
