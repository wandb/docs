
# 예측 시각화하기

[**여기에서 Colab 노트북으로 시도해보세요 →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W&B_Tables_Quickstart.ipynb)

이 문서에서는 PyTorch를 사용하여 MNIST 데이터로 모델 트레이닝 중 예측값을 추적, 시각화 및 비교하는 방법을 다룹니다.

다음을 배울 수 있습니다:
1. 모델 트레이닝 또는 평가 중 `wandb.Table()`에 메트릭, 이미지, 텍스트 등을 로그하기
2. 이러한 테이블을 보고, 정렬하고, 필터링하고, 그룹화하고, 합치고, 대화식으로 쿼리하고 탐색하기
3. 특정 이미지, 하이퍼파라미터/모델 버전 또는 시간 단계에서 동적으로 모델 예측값 또는 결과 비교하기

# 예시

## 특정 이미지에 대한 예측 점수 비교하기

[실시간 예시: 트레이닝 1 에포크 대 5 에포크 후의 예측 비교 →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#compare-predictions-after-1-vs-5-epochs)
<img src="https://i.imgur.com/NMme6Qj.png" alt="1 에포크 대 5 에포크의 트레이닝"/>
히스토그램은 두 모델 간의 클래스별 점수를 비교합니다. 각 히스토그램에서 상단의 초록색 막대는 모델 "CNN-2, 1 에포크"(id 0)를 나타내며, 1 에포크만 트레이닝되었습니다. 하단의 보라색 막대는 모델 "CNN-2, 5 에포크"(id 1)를 나타내며, 5 에포크 동안 트레이닝되었습니다. 이미지는 모델이 동의하지 않는 경우에 필터링됩니다. 예를 들어, 첫 번째 행에서 "4"는 1 에포크 후 모든 가능한 숫자에 대해 높은 점수를 받지만, 5 에포크 후에는 올바른 라벨에 가장 높은 점수를 받고 나머지에는 매우 낮은 점수를 받습니다.

## 시간에 따른 주요 오류에 집중하기
[실시간 예시 →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#top-errors-over-time)

전체 테스트 데이터에서 잘못된 예측("추측" != "진실")을 보고, 행을 필터링합니다. 1 에포크 트레이닝 후에 229개의 잘못된 추측이 있지만, 5 에포크 후에는 오직 98개입니다.
<img src="https://i.imgur.com/7g8nodn.png" alt="side by side, 1 대 5 에포크의 트레이닝"/>

## 모델 성능 비교 및 패턴 찾기

[실시간 예시에서 전체 세부 정보 보기 →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#false-positives-grouped-by-guess)

정답을 필터링한 후, 추측별로 그룹화하여 두 모델 옆으로 오분류된 이미지 예시와 진짜 라벨의 기본 분포를 봅니다. 레이어 크기와 학습률이 2배인 모델 변형이 왼쪽에 있고, 베이스라인은 오른쪽에 있습니다. 베이스라인이 각 추측된 클래스에 대해 약간 더 많은 실수를 한다는 것을 알 수 있습니다.
<img src="https://i.imgur.com/i5PP9AE.png" alt="기본값 대비 두 배 변형에 대한 그룹화된 오류"/>

# 가입 또는 로그인

[W&B에 가입하거나 로그인](https://wandb.ai/login)하여 브라우저에서 실험을 보고 상호 작용하세요.

이 예시에서는 편리한 호스팅 환경으로 Google Colab를 사용하고 있지만, 어디서든 자신만의 트레이닝 스크립트를 실행하고 W&B의 실험 추적 툴로 메트릭을 시각화할 수 있습니다.


```python
!pip install wandb -qqq
```

계정에 로그인


```python

import wandb
wandb.login()

WANDB_PROJECT = "mnist-viz"
```

# 0. 준비

의존성을 설치하고, MNIST를 다운로드하고, PyTorch를 사용하여 트레인 및 테스트 데이터셋을 생성합니다.


```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T 
import torch.nn.functional as F


device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 트레인 및 테스트 데이터로더 생성
def get_dataloader(is_train, batch_size, slice=5):
    "트레이닝 데이터로더 얻기"
    ds = torchvision.datasets.MNIST(root=".", train=is_train, transform=T.ToTensor(), download=True)
    loader = torch.utils.data.DataLoader(dataset=ds, 
                                         batch_size=batch_size, 
                                         shuffle=True if is_train else False, 
                                         pin_memory=True, num_workers=2)
    return loader
```

# 1. 모델 및 트레이닝 일정 정의하기

* 실행할 에포크 수를 설정하고, 각 에포크는 트레이닝 단계와 검증(테스트) 단계로 구성됩니다. 선택적으로 테스트 단계 당 로그하는 데이터 양을 구성할 수 있습니다. 여기에서는 데모를 단순화하기 위해 배치 수와 배치 당 이미지 수가 낮게 설정되어 있습니다.
* 간단한 컨볼루션 신경망 정의하기([pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial) 코드를 따름).
* PyTorch를 사용하여 트레인 및 테스트 세트 로드하기



```python
# 실행할 에포크 수
# 각 에포크는 트레이닝 단계와 테스트 단계를 포함하므로, 이는
# 테스트 예측을 로그할 테이블 수를 설정합니다
EPOCHS = 1

# 각 테스트 단계에 대해 로그할 테스트 데이터의 배치 수
# (데모를 단순화하기 위해 기본값이 낮게 설정됨)
NUM_BATCHES_TO_LOG = 10 #79

# 테스트 배치 당 로그할 이미지 수
# (데모를 단순화하기 위해 기본값이 낮게 설정됨)
NUM_IMAGES_PER_BATCH = 32 #128

# 트레이닝 설정 및 하이퍼파라미터
NUM_CLASSES = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
L1_SIZE = 32
L2_SIZE = 64
# 이를 변경하면 인접한 레이어의 형태를 변경해야 할 수 있습니다
CONV_KERNEL_SIZE = 5

# 두 레이어 컨볼루션 신경망 정의하기
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
        # 주어진 레이어의 형태를 보고 싶다면 주석을 해제하세요:
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

# 2. 트레이닝 실행 및 테스트 예측 로그하기

매 에포크마다, 트레이닝 단계와 테스트 단계를 실행합니다. 각 테스트 단계에 대해, 테스트 예측을 저장할 wandb.Table()을 생성합니다. 이들은 브라우저에서 시각화되고, 동적으로 쿼리되며, 나란히 비교될 수 있습니다.


```python
# ✨ W&B: 이 모델의 트레이닝을 추적하기 위해 새로운 run 초기화하기
wandb.init(project="table-quickstart")

# ✨ W&B: config를 사용하여 하이퍼파라미터 로그하기
cfg = wandb.config
cfg.update({"epochs" : EPOCHS, "batch_size": BATCH_SIZE, "lr" : LEARNING_RATE,
            "l1_size" : L1_SIZE, "l2_size": L2_SIZE,
            "conv_kernel" : CONV_KERNEL_SIZE,
            "img_count" : min(10000, NUM_IMAGES_PER_BATCH*NUM_BATCHES_TO_LOG)})

# 모델, 손실, 옵티마이저 정의하기
model = ConvNet(NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 테스트 이미지 배치에 대한 예측을 로그하기 위한 편리한 함수
def log_test_predictions(images, labels, outputs, predicted, test_table, log_counter):
  # 모든 클래스에 대한 신뢰도 점수 얻기
  scores = F.softmax(outputs.data, dim=1)
  log_scores = scores.cpu().numpy()
  log_images = images.cpu().numpy()
  log_labels = labels.cpu().numpy()
  log_preds = predicted.cpu().numpy()
  # 이미지 순서에 따라 id 추가하기
  _id = 0
  for i, l, p, s in zip(log_images, log_labels, log_preds, log_scores):
    # 데이터 테이블에 필요한 정보 추가하기:
    # id, 이미지 픽셀, 모델의 추측, 진짜 라벨, 모든 클래스에 대한 점수
    img_id = str(_id) + "_" + str(log_counter)
    test_table.add_data(img_id, wandb.Image(i), p, l, *s)
    _id += 1
    if _id == NUM_IMAGES_PER_BATCH:
      break

# 모델 트레이닝하기
total_step = len(train_loader)
for epoch in range(EPOCHS):
    # 트레이닝 단계
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        # forward 패스
        outputs = model(images)
        loss = criterion(outputs, labels)
        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
  
        # ✨ W&B: 트레이닝 단계에서 손실을 로그하기, UI에서 실시간으로 시각화됨
        wandb.log({"loss" : loss})
        if (i+1) % 100 == 0:
            print ('에포크 [{}/{}], 스텝 [{}/{}], 손실: {:.4f}'
                .format(epoch+1, EPOCHS, i+1, total_step, loss.item()))
            

    # ✨ W&B: 각 테스트 단계에 대한 예측을 저장할 테이블 생성하기
    columns=["id", "image", "guess", "truth"]
    for digit in range(10):
      columns.append("score_" + str(digit))
    test_table = wandb.Table(columns=columns)

    # 모델 테스트하기
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
        # ✨ W&B: 트레이닝 에포크에 걸쳐 정확도를 로그하여 UI에서 시각화하기
        wandb.log({"epoch" : epoch, "acc" : acc})
        print('10000 테스트 이미지에 대한 모델의 테스트 정확도: {} %'.format(acc))

    # ✨ W&B: 예측 테이블을 wandb에 로그하기
    wandb.log({"test_predictions" : test_table})

# ✨ W&B: (다중 셀 노트북에 유용함) 실행을 완료로 표시하기
wandb.finish()
```

# 다음은 무엇인가요?
다음 튜토리얼에서는 W&B Sweeps를 사용하여 하이퍼파라미터를 최적화하는 방법을 배웁니다:

## 👉 [하이퍼파라미터 최적화하기](sweeps)