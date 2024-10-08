---
title: Visualize predictions with tables
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

<CTAButtons colabLink='https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W&B_Tables_Quickstart.ipynb'/>

이 문서는 PyTorch로 MNIST 데이터를 사용하여 트레이닝 과정에서 모델 예측값을 추적하고, 시각화하며 비교하는 방법을 설명합니다.

여러분은 다음을 배울 것입니다:
1. 모델 트레이닝 또는 평가 중에 `wandb.Table()`에 메트릭, 이미지, 텍스트 등을 로그하기
2. 이 테이블들을 보기, 정렬, 필터링, 그룹화, 조인, 인터랙티브 쿼리 및 탐색
3. 모델 예측값 또는 결과 비교: 특정 이미지, 하이퍼파라미터/모델 버전, 또는 시간 단계에 따라 동적으로

## Examples
### 특정 이미지에 대한 예측 점수 비교하기

[실시간 예제: 1 vs 5 에포크 트레이닝 후 예측 비교하기 →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#compare-predictions-after-1-vs-5-epochs)

![1 epoch vs 5 epochs of training](/images/tutorials/tables-1.png)

히스토그램은 두 모델 간의 클래스별 점수를 비교합니다. 각 히스토그램의 상단 녹색 막대는 "CNN-2, 1 epoch" (id 0) 모델을 나타내며, 이는 1 에포크 동안만 트레이닝되었습니다. 하단 보라색 막대는 "CNN-2, 5 epochs" (id 1) 모델을 나타내며, 5 에포크 동안 트레이닝되었습니다. 이미지는 모델들이 일치하지 않는 경우로 필터링되었습니다. 예를 들어, 첫 번째 행에서 "4"는 1 에포크 후 모든 가능한 숫자에 대해 높은 점수를 받았지만, 5 에포크 후에는 정확한 레이블에서 가장 높은 점수를 받고 나머지에서는 매우 낮은 점수를 받습니다.

### 시간이 지남에 따라 주요 오류에 집중하기
[실시간 예제 →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#top-errors-over-time)

전체 테스트 데이터에서 잘못된 예측을 확인합니다("guess" != "truth"인 행으로 필터링). 1 트레이닝 에포크 후에는 229개의 잘못된 예측이 있지만, 5 에포크 후에는 98개만 남습니다.

![side by side, 1 vs 5 epochs of training](/images/tutorials/tables-2.png)

### 모델 성능 비교 및 패턴 찾기

[실시간 예제에서 자세히 보기 →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#false-positives-grouped-by-guess)

정확한 답변은 필터링한 후, 추측에 따라 그룹화하여 잘못 분류된 이미지와 실제 레이블의 분포를 양옆으로 비교합니다. 레이어 크기와 학습률이 2배인 모델 변형은 좌측에, 베이스라인은 우측에 위치합니다. 베이스라인은 각 추측된 클래스에서 약간 더 많은 실수를 한다는 점을 주목하세요.

![grouped errors for baseline vs double variant](/images/tutorials/tables-3.png)

## 회원가입 또는 로그인

W&B에 [회원가입 또는 로그인](https://wandb.ai/login)하여 브라우저에서 직접 실험을 보고 상호작용할 수 있습니다.

이 예제에서는 편리한 호스팅 환경으로 Google Colab을 사용하고 있지만, 어디서든 자신의 트레이닝 스크립트를 실행하고 W&B의 실험 추적 툴로 메트릭을 시각화할 수 있습니다.

```python
!pip install wandb -qqq
```

계정에 로그

```python

import wandb
wandb.login()

WANDB_PROJECT = "mnist-viz"
```

## 0. 설정

필수 모듈 설치, MNIST 다운로드 및 PyTorch를 사용하여 트레인과 테스트 데이터셋을 생성합니다.


```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T 
import torch.nn.functional as F


device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 트레인 및 테스트 데이터로더 생성
def get_dataloader(is_train, batch_size, slice=5):
    "트레이닝 데이터로더 가져오기"
    ds = torchvision.datasets.MNIST(root=".", train=is_train, transform=T.ToTensor(), download=True)
    loader = torch.utils.data.DataLoader(dataset=ds, 
                                         batch_size=batch_size, 
                                         shuffle=True if is_train else False, 
                                         pin_memory=True, num_workers=2)
    return loader
```

## 1. 모델 및 트레이닝 스케줄 정의

* 각 에포크는 트레이닝 스텝과 검증(테스트) 스텝으로 구성되는데, 이때 실행할 에포크 수를 설정합니다. 선택적으로, 각 테스트 스텝당 로그할 데이터의 양을 설정합니다. 여기서는 데모를 단순화하기 위해 시각화할 배치 및 이미지 수를 낮게 설정했습니다.
* 간단한 합성곱 신경망 정의 ([pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial) 코드 참조)
* PyTorch를 사용하여 트레인 및 테스트셋 로드

```python
# 실행할 에포크 수
# 각 에포크는 트레이닝 스텝과 테스트 스텝을 포함하므로, 이는 로그할 테스트 예측 테이블 수를 설정합니다
EPOCHS = 1

# 각 테스트 스텝당 테스트 데이터에서 로그할 배치 수
# (데모 단순화를 위해 기본값을 낮게 설정)
NUM_BATCHES_TO_LOG = 10 #79

# 각 테스트 배치당 로그할 이미지 수
# (데모 단순화를 위해 기본값을 낮게 설정)
NUM_IMAGES_PER_BATCH = 32 #128

# 트레이닝 설정 및 하이퍼파라미터
NUM_CLASSES = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
L1_SIZE = 32
L2_SIZE = 64
# 변경 시에는 인접 레이어의 모양을 변경해야 할 수 있습니다
CONV_KERNEL_SIZE = 5

# 이중 레이어 합성곱 신경망 정의
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
        # 주어진 레이어의 모양을 보려면 주석 해제:
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

각 에포크마다 트레이닝 스텝과 테스트 스텝을 실행합니다. 각 테스트 스텝마다, 테스트 예측값을 저장할 wandb.Table()을 만듭니다. 이를 통해 브라우저에서 시각화하고, 동적으로 쿼리하고, 옆으로 비교할 수 있습니다.

```python
# ✨ W&B: 이 모델의 트레이닝을 추적하기 위해 새로운 run을 초기화합니다
wandb.init(project="table-quickstart")

# ✨ W&B: 설정을 사용하여 하이퍼파라미터를 로그합니다
cfg = wandb.config
cfg.update({"epochs" : EPOCHS, "batch_size": BATCH_SIZE, "lr" : LEARNING_RATE,
            "l1_size" : L1_SIZE, "l2_size": L2_SIZE,
            "conv_kernel" : CONV_KERNEL_SIZE,
            "img_count" : min(10000, NUM_IMAGES_PER_BATCH*NUM_BATCHES_TO_LOG)})

# 모델, 손실, 옵티마이저 정의
model = ConvNet(NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 테스트 이미지 배치에 대한 예측값을 로그하기 위한 편의 함수
def log_test_predictions(images, labels, outputs, predicted, test_table, log_counter):
  # 모든 클래스에 대한 신뢰 점수 획득
  scores = F.softmax(outputs.data, dim=1)
  log_scores = scores.cpu().numpy()
  log_images = images.cpu().numpy()
  log_labels = labels.cpu().numpy()
  log_preds = predicted.cpu().numpy()
  # 이미지 순서에 따라 id 추가
  _id = 0
  for i, l, p, s in zip(log_images, log_labels, log_preds, log_scores):
    # 데이터 테이블에 필요한 정보 추가:
    # id, 이미지 픽셀, 모델의 추측, 실제 레이블, 모든 클래스에 대한 점수
    img_id = str(_id) + "_" + str(log_counter)
    test_table.add_data(img_id, wandb.Image(i), p, l, *s)
    _id += 1
    if _id == NUM_IMAGES_PER_BATCH:
      break

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
  
        # ✨ W&B: 트레이닝 단계 동안 손실을 로그하여 UI에 실시간으로 시각화
        wandb.log({"loss" : loss})
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                .format(epoch+1, EPOCHS, i+1, total_step, loss.item()))
            

    # ✨ W&B: 각 테스트 단계에 대한 예측값을 저장하기 위한 테이블 생성
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
        # ✨ W&B: 트레이닝 에포크 간 정확도를 로그하여 UI에 시각화
        wandb.log({"epoch" : epoch, "acc" : acc})
        print('Test Accuracy of the model on the 10000 test images: {} %'.format(acc))

    # ✨ W&B: 예측 테이블을 wandb에 로그
    wandb.log({"test_predictions" : test_table})

# ✨ W&B: run을 완료로 표시 (여러 셀 노트북에서 유용)
wandb.finish()
```

## 다음 단계는?
다음 튜토리얼에서는 W&B Sweeps를 사용하여 하이퍼파라미터를 최적화하는 방법을 배웁니다:
## 👉 [하이퍼파라미터 최적화하기](sweeps)