---
title: 테이블로 예측값 시각화
menu:
  tutorials:
    identifier: ko-tutorials-tables
    parent: null
weight: 2
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W&B_Tables_Quickstart.ipynb" >}}

이 튜토리얼에서는 PyTorch와 MNIST 데이터를 사용하여 트레이닝 과정에서 모델의 예측값을 추적, 시각화, 비교하는 방법을 다룹니다.

여기서 배울 내용은 다음과 같습니다:
1. 모델 트레이닝 또는 평가 중에 `wandb.Table()`을 사용해 메트릭, 이미지, 텍스트 등을 로그
2. 이 테이블들을 보고, 정렬하고, 필터링하고, 그룹화하고, 조인하며, 인터랙티브하게 쿼리하고 탐색하는 방법
3. 모델 예측값이나 결과를 비교: 특정 이미지, 하이퍼파라미터/모델 버전, 또는 시간 단계별로 동적으로 분석

## 예시
### 특정 이미지에 대한 예측 점수 비교하기

[라이브 예시: 트레이닝 1 에포크 대 5 에포크 후 예측값 비교 →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#compare-predictions-after-1-vs-5-epochs)

{{< img src="/images/tutorials/tables-1.png" alt="Training epoch comparison" >}}

히스토그램은 두 모델의 클래스별 점수를 비교한 것입니다. 각 히스토그램의 위쪽 녹색 막대는 1 에포크만 트레이닝된 모델 "CNN-2, 1 epoch" (id 0), 아래쪽 보라색 막대는 5 에포크 트레이닝된 "CNN-2, 5 epochs" (id 1)를 의미합니다. 이미지는 두 모델의 예측이 다를 때만 필터링되어 보여집니다. 예를 들어, 첫 번째 행의 "4"의 경우, 1 에포크에서는 모든 숫자에 대해 점수가 높지만, 5 에포크 후에는 정답 레이블에 가장 높은 점수, 나머지에는 매우 낮은 점수를 보입니다.

### 시간에 따른 주요 에러 집중 보기
[라이브 예시 →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#top-errors-over-time)

테스트 전체 데이터셋에서 잘못된 예측만 (즉 "guess" != "truth" 조건) 필터링해서 볼 수 있습니다. 1 에포크 트레이닝 후 틀린 예측은 229개, 5 에포크 후에는 98개로 줄어듭니다.

{{< img src="/images/tutorials/tables-2.png" alt="Side-by-side epoch comparison" >}}

### 모델 성능 비교 및 패턴 찾기

[라이브 전체 예시 보기 →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#false-positives-grouped-by-guess)

정답인 행을 제외하고, 예측값(guess)별로 그룹화해 잘못 분류된 이미지 예시와 실제 라벨 분포를 두 모델을 나란히 비교해 볼 수 있습니다. 왼쪽은 레이어 크기와 learning rate를 2배로 키운 모델 변형이고, 오른쪽은 베이스라인입니다. 베이스라인 모델이 각 예측 클래스별로 약간 더 많은 에러를 내는 것을 볼 수 있습니다.

{{< img src="/images/tutorials/tables-3.png" alt="Error comparison" >}}

## 회원가입 또는 로그인

[회원가입 또는 로그인](https://wandb.ai/login) 하여 W&B에서 웹브라우저로 직접 본인의 실험을 보고, 인터랙티브하게 탐색하세요.

이 예시에서는 Google Colab을 사용하지만, 직접 환경에서 트레이닝 스크립트를 실행하고 W&B의 experiment tracking tool로 메트릭을 시각화할 수도 있습니다.


```python
!pip install wandb -qqq
```

본인 계정에 로그인을 진행하세요.


```python

import wandb
wandb.login()

WANDB_PROJECT = "mnist-viz"
```

## 0. 설정

필요한 패키지 설치, MNIST 데이터 다운로드, PyTorch를 이용한 트레인/테스트 데이터셋 만들기


```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T 
import torch.nn.functional as F


device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 트레인/테스트 데이터로더 만들기
def get_dataloader(is_train, batch_size, slice=5):
    "트레이닝 데이터로더를 가져옵니다"
    ds = torchvision.datasets.MNIST(root=".", train=is_train, transform=T.ToTensor(), download=True)
    loader = torch.utils.data.DataLoader(dataset=ds, 
                                         batch_size=batch_size, 
                                         shuffle=True if is_train else False, 
                                         pin_memory=True, num_workers=2)
    return loader
```

## 1. 모델 및 트레이닝 스케줄 정의

* 실행할 에포크 수를 지정합니다. 각 에포크에는 트레이닝 및 검증(테스트) 단계가 포함됩니다. 테스트마다 로그할 데이터양을 옵션으로 설정할 수 있습니다. 예시에서는 각 테스트 단계에서 시각화할 배치/이미지 개수를 적게 설정해 데모를 간소화했습니다.
* 간단한 컨볼루션 신경망을 정의합니다 ([pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial) 코드 참고).
* PyTorch로 트레인/테스트셋을 불러옵니다.



```python
# 실행할 에포크 수
# 각 에포크에 트레이닝/테스트 단계가 있으므로,
# 테스트 예측값 테이블이 그만큼 생성됨
EPOCHS = 1

# 각 테스트 스텝에서 로그할 테스트 배치 수
# (데모 간소화를 위해 낮게 설정)
NUM_BATCHES_TO_LOG = 10 #79

# 각 테스트 배치에서 로그할 이미지 수
# (데모 간소화를 위해 낮게 설정)
NUM_IMAGES_PER_BATCH = 32 #128

# 트레이닝 설정 및 하이퍼파라미터
NUM_CLASSES = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
L1_SIZE = 32
L2_SIZE = 64
# 이 값을 변경할 때는 인접 레이어의 shape도 맞춰줘야 함
CONV_KERNEL_SIZE = 5

# 두 레이어로 구성된 컨볼루션 신경망 정의
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, L1_SIZE, CONV_KERNEL_SIZE, stride=1, padding=2),
            nn.BatchNorm2d(L1_SIZE),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(L1_SIZE, L2_SIZE, CONV_KERNEL_SIZE, stride=1, padding=2),
            nn.BatchNorm2d(L2_SIZE),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*L2_SIZE, NUM_CLASSES)
        self.softmax = nn.Softmax(NUM_CLASSES)

    def forward(self, x):
        # 각 레이어의 shape를 보고 싶다면 주석 해제:
        #print("x: ", x.size())
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

train_loader = get_dataloader(is_train=True, batch_size=BATCH_SIZE)
test_loader = get_dataloader(is_train=False, batch_size=2*BATCH_SIZE)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

## 2. 트레이닝 실행 및 테스트 예측값 로그

각 에포크마다 트레이닝 단계와 테스트 단계를 진행합니다. 각 테스트 단계에서는 테스트 예측값을 저장할 `wandb.Table()`을 만듭니다. 이 테이블은 브라우저에서 다이내믹하게 쿼리, 비교, 시각화할 수 있습니다.


```python
# 테스트 이미지 배치의 예측값을 로그하는 편의 함수
def log_test_predictions(images, labels, outputs, predicted, test_table, log_counter):
  # 모든 클래스에 대한 confidence score 얻기
  scores = F.softmax(outputs.data, dim=1)
  log_scores = scores.cpu().numpy()
  log_images = images.cpu().numpy()
  log_labels = labels.cpu().numpy()
  log_preds = predicted.cpu().numpy()
  # 이미지 순서대로 id 부여
  _id = 0
  for i, l, p, s in zip(log_images, log_labels, log_preds, log_scores):
    # 데이터 테이블에 필수 정보 추가:
    # id, 이미지 픽셀, 모델 예측값, 실제 라벨, 각 클래스에 대한 score
    img_id = str(_id) + "_" + str(log_counter)
    test_table.add_data(img_id, wandb.Image(i), p, l, *s)
    _id += 1
    if _id == NUM_IMAGES_PER_BATCH:
      break

# W&B: 이 모델의 트레이닝을 추적할 새 run 초기화
with wandb.init(project="table-quickstart") as run:

    # W&B: config를 사용해 하이퍼파라미터 로그
    cfg = run.config
    cfg.update({"epochs" : EPOCHS, "batch_size": BATCH_SIZE, "lr" : LEARNING_RATE,
                "l1_size" : L1_SIZE, "l2_size": L2_SIZE,
                "conv_kernel" : CONV_KERNEL_SIZE,
                "img_count" : min(10000, NUM_IMAGES_PER_BATCH*NUM_BATCHES_TO_LOG)})

    # 모델, loss, 옵티마이저 정의
    model = ConvNet(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 모델 트레이닝
    total_step = len(train_loader)
    for epoch in range(EPOCHS):
        # 트레이닝 스텝
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            # forward 패스
            outputs = model(images)
            loss = criterion(outputs, labels)
            # backward 및 최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            # W&B: UI에서 실시간 시각화를 위해 loss를 트레이닝 스텝마다 로그
            run.log({"loss" : loss})
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, EPOCHS, i+1, total_step, loss.item()))
                

        # W&B: 각 테스트 단계마다 예측값을 저장할 테이블 생성
        columns=["id", "image", "guess", "truth"]
        for digit in range(10):
        columns.append("score_" + str(digit))
        test_table = wandb.Table(columns=columns)

        # 모델 테스트
        model.eval()
        log_counter = 0
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                if log_counter < NUM_BATCHES_TO_LOG:
                log_test_predictions(images, labels, outputs, predicted, test_table, log_counter)
                log_counter += 1
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            acc = 100 * correct / total
            # W&B: UI에서 시각화할 수 있도록 에포크별 정확도 로그
            run.log({"epoch" : epoch, "acc" : acc})
            print('Test Accuracy of the model on the 10000 test images: {} %'.format(acc))

        # W&B: wandb에 예측값 테이블 로그
        run.log({"test_predictions" : test_table})
```

## 다음은 무엇을 배울까요?
다음 튜토리얼에서는 [W&B Sweeps를 사용해 하이퍼파라미터를 최적화하는 방법]({{< relref path="sweeps.md" lang="ko" >}})을 배웁니다.