---
title: Track experiments
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

<CTAButtons colabLink='https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_&_Biases.ipynb'/>

[W&B](https://wandb.ai/site?utm_source=intro_colab&utm_medium=code&utm_campaign=intro)를 사용하여 기계학습 실험 추적, 모델 체크포인트 생성, 팀과의 협업 등을 수행하세요. 전체 W&B 문서는 [여기](/)에서 확인할 수 있습니다.

이 노트북에서는 간단한 PyTorch 모델을 사용하여 기계학습 실험을 생성하고 추적합니다. 노트북이 끝날 때쯤에는 팀의 다른 멤버와 공유하고 커스터마이즈할 수 있는 대화형 프로젝트 대시보드가 준비됩니다. [예제 대시보드를 여기서 확인하세요](https://wandb.ai/wandb/wandb_example).

## 사전 준비

W&B Python SDK를 설치하고 로그인하세요:

```shell
!pip install wandb -qU
```

```python
# W&B 계정에 로그인하기
import wandb
import random
import math

# wandb의 새로운 백엔드를 위한 wandb-core 사용  
wandb.require("core")
```

```python
wandb.login()
```

## W&B로 기계학습 실험을 시뮬레이션하고 추적하기

기계학습 실험을 생성하고, 추적하고, 시각화합니다. 다음과 같은 단계로 수행합니다:

1. [W&B run](/guides/runs)을 초기화하고 추적하려는 하이퍼파라미터를 전달합니다.
2. 트레이닝 루프 내에서 정확도와 손실과 같은 메트릭을 로그합니다.

```
import random
import math

# 5개의 시뮬레이션된 실험 실행
total_runs = 5
for run in range(total_runs):
  # 1️. 이 스크립트를 추적하기 위해 새로운 run 시작하기
  wandb.init(
      # 이 run이 로그될 프로젝트 설정
      project="basic-intro",
      # run의 이름을 지정하기 (지정하지 않으면 임의로, 예: sunshine-lollypop-10 같이 이름이 부여됨)
      name=f"experiment_{run}",
      # 하이퍼파라미터와 run 메타데이터를 추적하기
      config={
      "learning_rate": 0.02,
      "architecture": "CNN",
      "dataset": "CIFAR-100",
      "epochs": 10,
      })

  # 이 간단한 블록은 트레이닝 루프를 시뮬레이트하여 메트릭을 로그합니다
  epochs = 10
  offset = random.random() / 5
  for epoch in range(2, epochs):
      acc = 1 - 2 ** -epoch - random.random() / epoch - offset
      loss = 2 ** -epoch + random.random() / epoch + offset

      # 2️. 스크립트에서 메트릭을 W&B에 로그하기
      wandb.log({"acc": acc, "loss": loss})

  # run을 종료로 표시합니다
  wandb.finish()
```

W&B 프로젝트에서 기계학습이 어떻게 수행되었는지를 확인하세요. 이전 셀에서 출력된 URL 링크를 복사하여 붙여넣으세요. URL은 그래프가 들어 있는 대시보드를 보여주는 W&B 프로젝트로 안내할 것입니다.

다음 이미지는 대시보드가 어떻게 보일 수 있는지를 보여줍니다:

![](/images/tutorials/experiments-1.png)

이제 W&B를 pseudo 기계학습 트레이닝 루프에 통합하는 방법을 알았으니, 기본 PyTorch 신경망을 사용하여 기계학습 실험을 추적해 보겠습니다. 다음 코드 또한 W&B에 모델 체크포인트를 업로드하여 조직 내 다른 팀과 공유할 수 있도록 합니다.

## PyTorch를 사용하여 기계학습 실험 추적하기

다음 코드 셀은 간단한 MNIST 분류기를 정의하고 트레이닝합니다. 트레이닝 동안, W&B가 URL들을 출력하는 것을 보게 될 것입니다. 프로젝트 페이지 링크를 클릭하여 결과를 실시간으로 W&B 프로젝트에 스트리밍하는 것을 보세요.

W&B run은 자동으로 [메트릭](/guides/app/pages/run-page/#workspace-tab),
[시스템 정보](/guides/app/pages/run-page/#system-tab),
[하이퍼파라미터](/guides/app/pages/run-page/#overview-tab),
[터미널 출력](/guides/app/pages/run-page/#logs-tab)을 로그하고
모델 입력 및 출력을 포함한 [대화형 테이블](/guides/tables)을 볼 수 있습니다.

### PyTorch Dataloader 설정하기
다음 셀은 우리의 기계학습 모델을 트레이닝하는 데 필요한 유용한 함수를 정의합니다. 함수 자체는 W&B에 독특한 것이 아니므로, 여기에서는 자세히 다루지 않습니다. [forward 및 backward 트레이닝 루프](https://pytorch.org/tutorials/beginner/nn_tutorial.html)를 정의하는 방법, 트레이닝에 데이터를 로드하기 위해 [PyTorch DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)를 사용하는 방법, 및 [`torch.nn.Sequential` 클래스](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html)를 사용하여 PyTorch 모델을 정의하는 방법에 대한 자세한 내용은 PyTorch 문서를 참조하세요.

```python
#@title
import torch, torchvision
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision.transforms as T

MNIST.mirrors = [mirror for mirror in MNIST.mirrors if "http://yann.lecun.com/" not in mirror]

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_dataloader(is_train, batch_size, slice=5):
    "트레이닝 dataloader 가져오기"
    full_dataset = MNIST(root=".", train=is_train, transform=T.ToTensor(), download=True)
    sub_dataset = torch.utils.data.Subset(full_dataset, indices=range(0, len(full_dataset), slice))
    loader = torch.utils.data.DataLoader(dataset=sub_dataset,
                                         batch_size=batch_size,
                                         shuffle=True if is_train else False,
                                         pin_memory=True, num_workers=2)
    return loader

def get_model(dropout):
    "간단한 모델"
    model = nn.Sequential(nn.Flatten(),
                         nn.Linear(28*28, 256),
                         nn.BatchNorm1d(256),
                         nn.ReLU(),
                         nn.Dropout(dropout),
                         nn.Linear(256,10)).to(device)
    return model

def validate_model(model, valid_dl, loss_func, log_images=False, batch_idx=0):
    "검증 데이터셋에 대한 모델 성능을 계산하고 wandb.Table을 로그합니다"
    model.eval()
    val_loss = 0.
    with torch.inference_mode():
        correct = 0
        for i, (images, labels) in enumerate(valid_dl):
            images, labels = images.to(device), labels.to(device)

            # Forward 패스 ➡
            outputs = model(images)
            val_loss += loss_func(outputs, labels)*labels.size(0)

            # 정확도 계산 및 누적
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            # 항상 동일한 batch_idx에서 대시보드에 이미지 한 배치를 로그합니다.
            if i==batch_idx and log_images:
                log_image_table(images, predicted, labels, outputs.softmax(dim=1))
    return val_loss / len(valid_dl.dataset), correct / len(valid_dl.dataset)
```

### 예측 값과 실제 값을 비교하는 테이블 생성하기

다음 셀은 W&B에 독특한 것이므로 자세히 살펴보겠습니다.

셀에서 `log_image_table`이라는 함수를 정의합니다. 기술적으로 선택 사항이지만, 이 함수는 W&B Table 오브젝트를 생성합니다. 우리는 테이블 오브젝트를 사용하여 각 이미지에 대해 모델이 예측한 것을 보여주는 테이블을 만듭니다.

더 자세히 말하면, 각 행은 모델에 입력된 이미지와 예측 값, 실제 값(레이블)이 포함됩니다.

```python
def log_image_table(images, predicted, labels, probs):
    "wandb.Table을 (img, pred, target, scores)로 로그합니다"
    # 이미지, 레이블 및 예측을 로그할 wandb 테이블 만들기
    table = wandb.Table(columns=["image", "pred", "target"]+[f"score_{i}" for i in range(10)])
    for img, pred, targ, prob in zip(images.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")):
        table.add_data(wandb.Image(img[0].numpy()*255), pred, targ, *prob.numpy())
    wandb.log({"predictions_table":table}, commit=False)
```

### 모델을 트레이닝하고 체크포인트 업로드하기

다음 코드는 모델을 트레이닝하고 모델 체크포인트를 프로젝트에 저장합니다. 모델 체크포인트는 트레이닝 동안 모델이 어떻게 수행되었는지를 평가하기 위해 일반적으로 사용하는 것처럼 사용하세요.

W&B는 또한 팀이나 조직의 다른 멤버들과 저장된 모델과 모델 체크포인트를 쉽게 공유할 수 있도록 합니다. 팀 외부의 멤버들과 모델 및 모델 체크포인트를 공유하는 방법에 대한 자세한 내용은 [W&B Registry](/guides/registry)를 참조하세요.

```python
# 서로 다른 dropout 비율을 시도하며 3개의 실험 실행
for _ in range(3):
    # wandb run 초기화하기
    wandb.init(
        project="pytorch-intro",
        config={
            "epochs": 5,
            "batch_size": 128,
            "lr": 1e-3,
            "dropout": random.uniform(0.01, 0.80),
            })

    # config를 복사합니다
    config = wandb.config

    # 데이터 가져오기
    train_dl = get_dataloader(is_train=True, batch_size=config.batch_size)
    valid_dl = get_dataloader(is_train=False, batch_size=2*config.batch_size)
    n_steps_per_epoch = math.ceil(len(train_dl.dataset) / config.batch_size)

    # 간단한 MLP 모델
    model = get_model(config.dropout)

    # 손실 및 옵티마이저 만들기
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # 트레이닝
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
            metrics = {"train/train_loss": train_loss,
                       "train/epoch": (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch,
                       "train/example_ct": example_ct}

            if step + 1 < n_steps_per_epoch:
                # wandb에 트레이닝 메트릭 로그하기
                wandb.log(metrics)

            step_ct += 1

        val_loss, accuracy = validate_model(model, valid_dl, loss_func, log_images=(epoch==(config.epochs-1)))

        # wandb에 트레이닝 및 검증 메트릭 로그하기
        val_metrics = {"val/val_loss": val_loss,
                       "val/val_accuracy": accuracy}
        wandb.log({**metrics, **val_metrics})

        # wandb에 모델 체크포인트 저장하기
        torch.save(model, "my_model.pt")
        wandb.log_model("./my_model.pt", "my_mnist_model", aliases=[f"epoch-{epoch+1}_dropout-{round(wandb.config.dropout, 4)}"])

        print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.3f}, Valid Loss: {val_loss:3f}, Accuracy: {accuracy:.2f}")

    # 테스트 세트가 있었다면, 이 방법으로 Summary 메트릭으로 로그할 수 있습니다
    wandb.summary['test_accuracy'] = 0.8

    # wandb run 닫기
    wandb.finish()
```

이제 W&B를 사용하여 첫 번째 모델을 트레이닝했습니다. 위의 링크 중 하나를 클릭하여 메트릭을 확인하고 W&B 앱 UI의 Artifacts 탭에서 저장된 모델 체크포인트를 확인하세요.

## (옵션) W&B 알림 설정하기

Python 코드에서 Slack이나 이메일로 알림을 보내는 [W&B Alerts](/guides/runs/alert/)를 생성하세요.

코드에서 Slack이나 이메일 알림을 처음 보내고 싶을 때 따라야 할 단계는 두 가지입니다:

1) W&B [사용자 설정](https://wandb.ai/settings)에서 알림을 켜세요
2) 코드에 `wandb.alert()`를 추가하세요. 예를 들어:

```python
wandb.alert(
    title="Low accuracy",
    text=f"Accuracy is below the acceptable threshold"
)
```

다음 셀은 `wandb.alert`를 사용하는 방법을 보여주는 간단 예제를 제공합니다.

```python
# wandb run 시작하기
wandb.init(project="pytorch-intro")

# 모델 트레이닝 루프 시뮬레이션
acc_threshold = 0.3
for training_step in range(1000):

    # 정확도를 위한 랜덤 숫자 생성
    accuracy = round(random.random() + random.random(), 3)
    print(f'Accuracy is: {accuracy}, {acc_threshold}')

    # wandb에 정확도 로그하기
    wandb.log({"Accuracy": accuracy})

    # 정확도가 임계값 이하인 경우, W&B Alert를 발송하고 run을 중지합니다
    if accuracy <= acc_threshold:
        # wandb Alert 보내기
        wandb.alert(
            title='Low Accuracy',
            text=f'Accuracy {accuracy} at step {training_step} is below the acceptable theshold, {acc_threshold}',
        )
        print('Alert triggered')
        break

# run을 종료로 표시합니다 (Jupyter 노트북에서 유용합니다)
wandb.finish()
```

[W&B Alerts에 대한 전체 문서](/guides/runs/alert)는 여기서 찾을 수 있습니다.

## 다음 단계
다음 튜토리얼에서는 W&B Sweeps를 사용한 하이퍼파라미터 최적화 방법을 배우게 됩니다:
[PyTorch를 사용한 하이퍼파라미터 스윕](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb)