
# 예측 시각화하기

[**Colab 노트북에서 시도해보기 →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W&B_Tables_Quickstart.ipynb)

이 문서는 PyTorch를 사용하여 MNIST 데이터에 대한 모델 예측을 추적, 시각화, 비교하는 방법을 다룹니다.

다음을 배울 수 있습니다:
1. 모델 학습이나 평가 중에 메트릭, 이미지, 텍스트 등을 `wandb.Table()`에 로그하기
2. 이러한 테이블을 보고, 정렬하고, 필터링하고, 그룹화하고, 조인하고, 대화형으로 쿼리하고, 탐색하기
3. 모델 예측이나 결과를 비교하기: 특정 이미지, 하이퍼파라미터/모델 버전 또는 시간 단계에 걸쳐 동적으로 비교하기

# 예시

## 특정 이미지에 대한 예측 점수 비교

[실제 예시: 1 에포크 대비 5 에포크 학습 후 예측 비교 →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#compare-predictions-after-1-vs-5-epochs)
<img src="https://i.imgur.com/NMme6Qj.png" alt="1 epoch vs 5 epochs of training"/>
히스토그램은 두 모델 간의 클래스별 점수를 비교합니다. 각 히스토그램에서 상단의 초록색 바는 모델 "CNN-2, 1 에포크"(id 0)를 나타내며, 1 에포크만 학습했습니다. 하단의 보라색 바는 모델 "CNN-2, 5 에포크"(id 1)를 나타내며, 5 에포크 동안 학습했습니다. 이미지는 모델이 동의하지 않는 경우에 필터링됩니다. 예를 들어, 첫 번째 행에서 "4"는 1 에포크 후 모든 가능한 숫자에 대해 높은 점수를 받지만, 5 에포크 후에는 올바른 레이블에 가장 높은 점수를 받고 나머지에는 매우 낮은 점수를 받습니다.

## 시간에 따른 주요 오류 집중
[실제 예시 →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#top-errors-over-time)

전체 테스트 데이터에서 잘못된 예측("추측" != "진실")을 보고 필터링합니다. 1번의 학습 에포크 후에는 229개의 잘못된 추측이 있지만, 5 에포크 후에는 98개 뿐입니다.
<img src="https://i.imgur.com/7g8nodn.png" alt="side by side, 1 vs 5 epochs of training"/>

## 모델 성능 비교 및 패턴 찾기

[실제 예시에서 전체 상세 보기 →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#false-positives-grouped-by-guess)

정답을 필터링한 후 추측별로 그룹화하여 두 모델 옆에 오분류된 이미지와 진짜 레이블의 기본 분포를 확인합니다. 모델 변형 중 레이어 크기와 학습률이 2배인 것이 왼쪽에 있고, 기준이 오른쪽에 있습니다. 기준은 각 추측된 클래스에 대해 약간 더 많은 실수를 합니다.
<img src="https://i.imgur.com/i5PP9AE.png" alt="grouped errors for baseline vs double variant"/>

# 가입하거나 로그인하기

[W&B에 가입하거나 로그인](https://wandb.ai/login)하여 브라우저에서 실험을 보고 상호 작용합니다.

이 예시에서는 편리한 호스팅 환경으로 Google Colab을 사용하고 있지만, 어디에서나 자신의 학습 스크립트를 실행하고 W&B의 실험 추적 도구로 메트릭을 시각화할 수 있습니다.


```python
!pip install wandb -qqq
```

계정으로 로그인하기


```python

import wandb
wandb.login()

WANDB_PROJECT = "mnist-viz"
```

# 0. 준비

의존성 설치, MNIST 다운로드, PyTorch를 사용하여 학습 및 테스트 데이터세트 생성하기


```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T 
import torch.nn.functional as F


device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 학습 및 테스트 데이터로더 생성
def get_dataloader(is_train, batch_size, slice=5):
    "학습 데이터로더 가져오기"
    ds = torchvision.datasets.MNIST(root=".", train=is_train, transform=T.ToTensor(), download=True)
    loader = torch.utils.data.DataLoader(dataset=ds, 
                                         batch_size=batch_size, 
                                         shuffle=True if is_train else False, 
                                         pin_memory=True, num_workers=2)
    return loader
```

# 1. 모델 및 학습 일정 정의하기

* 실행할 에포크 수 설정하기, 여기서 각 에포크는 학습 단계와 검증(테스트) 단계로 구성됩니다. 선택적으로 테스트 단계마다 로그할 데이터의 양을 구성할 수 있습니다. 여기서는 데모를 단순화하기 위해 배치 수와 배치당 이미지 수를 낮게 설정했습니다.
* 간단한 컨볼루션 신경망 정의하기 ([pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial) 코드를 따름).
* PyTorch를 사용하여 학습 및 테스트 세트 로드하기



```python
# 실행할 에포크 수
# 각 에포크는 학습 단계와 테스트 단계를 포함하므로, 이는 테스트 예측을 로그할 테이블의 수를 설정합니다
EPOCHS = 1

# 각 테스트 단계마다 로그할 테스트 데이터의 배치 수
# (데모를 단순화하기 위해 기본값이 낮게 설정됨)
NUM_BATCHES_TO_LOG = 10 #79

# 테스트 배치당 로그할 이미지 수
# (데모를 단순화하기 위해 기본값이 낮게 설정됨)
NUM_IMAGES_PER_BATCH = 32 #128

# 학습 구성 및 하이퍼파라미터
NUM_CLASSES = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
L1_SIZE = 32
L2_SIZE = 64
# 이를 변경하면 인접한 레이어의 모양을 변경해야 할 수도 있습니다
CONV_KERNEL_SIZE = 5

# 두 레이어 컨볼루션 신경망 정의
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
        # 주어진 레이어의 모양을 보려면 주석 해제하세요:
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

# 2. 학습 실행 및 테스트 예측 로그하기

매 에포크마다 학습 단계와 테스트 단계를 실행합니다. 각 테스트 단계마다, 테스트 예측을 저장할 wandb.Table()을 생성합니다. 이들은 브라우저에서 시각적으로, 동적으로 쿼리하고, 나란히 비교할 수 있습니다.


```python
# ✨ W&B: 이 모델 학습을 추적하기 위해 새로운 실행을 초기화합니다
wandb.init(project="table-quickstart")

# ✨ W&B: 하이퍼파라미터를 사용하여 로그하기
cfg = wandb.config
cfg.update({"epochs" : EPOCHS, "batch_size": BATCH_SIZE, "lr" : LEARNING_RATE,
            "l1_size" : L1_SIZE, "l2_size": L2_SIZE,
            "conv_kernel" : CONV_KERNEL_SIZE,
            "img_count" : min(10000, NUM_IMAGES_PER_BATCH*NUM_BATCHES_TO_LOG)})

# 모델, 손실, 옵티마이저 정의
model = ConvNet(NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 테스트 이미지 배치에 대한 예측을 로그하는 편리한 함수
def log_test_predictions(images, labels, outputs, predicted, test_table, log_counter):
  # 모든 클래스에 대한 신뢰도 점수 얻기
  scores = F.softmax(outputs.data, dim=1)
  log_scores = scores.cpu().numpy()
  log_images = images.cpu().numpy()
  log_labels = labels.cpu().numpy()
  log_preds = predicted.cpu().numpy()
  # 이미지 순서에 따라 id 추가
  _id = 0
  for i, l, p, s in zip(log_images, log_labels, log_preds, log_scores):
    # 데이터 테이블에 필요한 정보 추가:
    # id, 이미지 픽셀, 모델의 추측, 진짜 레이블, 모든 클래스에 대한 점수
    img_id = str(_id) + "_" + str(log_counter)
    test_table.add_data(img_id, wandb.Image(i), p, l, *s)
    _id += 1
    if _id == NUM_IMAGES_PER_BATCH:
      break

# 모델 학습
total_step = len(train_loader)
for epoch in range(EPOCHS):
    # 학습 단계
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        # 순방향 전달
        outputs = model(images)
        loss = criterion(outputs, labels)
        # 역방향 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
  
        # ✨ W&B: 학습 단계별로 손실 로그, UI에서 실시간으로 시각화
        wandb.log({"loss" : loss})
        if (i+1) % 100 == 0:
            print ('에포크 [{}/{}], 단계 [{}/{}], 손실: {:.4f}'
                .format(epoch+1, EPOCHS, i+1, total_step, loss.item()))
            

    # ✨ W&B: 각 테스트 단계에 대한 예측을 저장할 테이블 생성
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
        # ✨ W&B: 학습 에포크별로 정확도 로그, UI에서 시각화
        wandb.log({"epoch" : epoch, "acc" : acc})
        print('10000개의 테스트 이미지에 대한 모델의 정확도: {} %'.format(acc))

    # ✨ W&B: 예측 테이블을 wandb에 로그하기
    wandb.log({"test_predictions" : test_table})

# ✨ W&B: 실행을 완료로 표시(여러 셀 노트북에 유용)
wandb.finish()
```

# 다음 단계는?
다음 튜토리얼에서는 W&B 스윕을 사용하여 하이퍼파라미터를 최적화하는 방법을 배울 것입니다:

## 👉 [하이퍼파라미터 최적화하기](sweeps)