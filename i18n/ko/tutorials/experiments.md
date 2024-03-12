
# 실험 추적하기


[**Colab 노트북에서 시도해 보세요 →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_&_Biases.ipynb)

빠른 실험은 기계학습에 있어 근본적입니다. 이 튜토리얼에서는 W&B를 사용하여 실험을 추적하고 시각화하여 빠르게 반복하고 결과를 이해할 수 있습니다.

## 🤩 실험을 위한 공유 대시보드

단 몇 줄의 코드로,
당신은 풍부하고 인터랙티브하며 공유 가능한 대시보드를 얻을 수 있습니다 [여기에서 직접 확인하세요](https://wandb.ai/wandb/wandb_example).
![](https://i.imgur.com/Pell4Oo.png)

## 🔒 데이터 & 개인정보 보호

우리는 보안을 매우 심각하게 생각하며, 클라우드 호스팅 대시보드는 업계 표준 모범 사례를 사용하여 암호화합니다. 만약 여러분이 기업 클러스터를 떠날 수 없는 데이터셋을 다루고 있다면, 우리는 [온프레미스](https://docs.wandb.com/self-hosted) 설치를 제공합니다.

또한 모든 데이터를 쉽게 다운로드하고 다른 툴로 내보낼 수 있습니다 — 예를 들어, Jupyter 노트북에서의 사용자 지정 분석과 같은. 여기에서 [우리의 API에 대해 더 알아보세요](https://docs.wandb.com/library/api).

---

## 🪄 `wandb` 라이브러리 설치 및 로그인


라이브러리를 설치하고 무료 계정에 로그인하기로 시작합니다.




```python
!pip install wandb -qU
```


```python
# W&B 계정에 로그인
import wandb
wandb.login()
```

## 👟 실험 실행하기
1️⃣. **새 run을 시작**하고 추적할 하이퍼파라미터 전달하기

2️⃣. **트레이닝 또는 평가에서 메트릭 로그하기**

3️⃣. **대시보드에서 결과 시각화하기**


```python
import random

# 5개의 시뮬레이션된 실험 실행
total_runs = 5
for run in range(total_runs):
  # 🐝 1️⃣ 이 스크립트를 추적할 새 run 시작하기
  wandb.init(
      # 이 run이 기록될 프로젝트 설정
      project="basic-intro", 
      # run 이름을 전달합니다 (그렇지 않으면 무작위로 할당됩니다, 예: sunshine-lollypop-10)
      name=f"experiment_{run}", 
      # 하이퍼파라미터와 run 메타데이터 추적
      config={
      "learning_rate": 0.02,
      "architecture": "CNN",
      "dataset": "CIFAR-100",
      "epochs": 10,
      })
  
  # 이 간단한 블록은 메트릭 로깅을 시뮬레이션하는 트레이닝 루프를 시뮬레이션합니다
  epochs = 10
  offset = random.random() / 5
  for epoch in range(2, epochs):
      acc = 1 - 2 ** -epoch - random.random() / epoch - offset
      loss = 2 ** -epoch + random.random() / epoch + offset
      
      # 🐝 2️⃣ 스크립트에서 W&B로 메트릭 로그하기
      wandb.log({"acc": acc, "loss": loss})
      
  # run을 완료로 표시하기
  wandb.finish()
```

3️⃣ 이 코드를 실행하면 위의 👆 wandb 링크를 클릭하여 대화형 대시보드를 찾을 수 있습니다.

# 🔥 간단한 Pytorch 신경망

💪 이 모델을 실행하여 간단한 MNIST 분류기를 트레이닝하고, 프로젝트 페이지 링크를 클릭하여 W&B 프로젝트에 결과가 실시간으로 스트리밍되는 것을 확인하세요.


`wandb`에서 실행하는 모든 run은 자동으로 [메트릭](https://docs.wandb.ai/ref/app/pages/run-page#charts-tab),
[시스템 정보](https://docs.wandb.ai/ref/app/pages/run-page#system-tab),
[하이퍼파라미터](https://docs.wandb.ai/ref/app/pages/run-page#overview-tab),
[터미널 출력](https://docs.wandb.ai/ref/app/pages/run-page#logs-tab)을 로그하며,
모델 입력 및 출력과 함께 [인터랙티브 테이블](https://docs.wandb.ai/guides/tables)을 볼 수 있습니다.

## Dataloader 설정하기

이 예제를 실행하려면 PyTorch를 설치해야 합니다. Google Colab을 사용하는 경우 이미 사전 설치되어 있습니다. 


```python
!pip install torch torchvision
```


```python
import wandb
import math
import random
import torch, torchvision
import torch.nn as nn
import torchvision.transforms as T

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_dataloader(is_train, batch_size, slice=5):
    "트레이닝을 위한 dataloader 가져오기"
    full_dataset = torchvision.datasets.MNIST(root=".", train=is_train, transform=T.ToTensor(), download=True)
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
    "검증 데이터셋에서 모델의 성능을 계산하고 wandb.Table로 로그하기"
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

            # 대시보드에 한 배치의 이미지 로그하기, 항상 같은 batch_idx.
            if i==batch_idx && log_images:
                log_image_table(images, predicted, labels, outputs.softmax(dim=1))
    return val_loss / len(valid_dl.dataset), correct / len(valid_dl.dataset)

def log_image_table(images, predicted, labels, probs):
    "wandb.Table로 (img, pred, target, scores) 로그하기"
    # 🐝 이미지, 라벨, 예측을 로그하기 위한 wandb Table 생성하기
    table = wandb.Table(columns=["image", "pred", "target"]+[f"score_{i}" for i in range(10)])
    for img, pred, targ, prob in zip(images.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")):
        table.add_data(wandb.Image(img[0].numpy()*255), pred, targ, *prob.numpy())
    wandb.log({"predictions_table":table}, commit=False)
```

## 모델 트레이닝하기


```python
# 다양한 드롭아웃 비율을 시도하며 5개의 실험 실행
for _ in range(5):
    # 🐝 wandb run 초기화하기
    wandb.init(
        project="pytorch-intro",
        config={
            "epochs": 10,
            "batch_size": 128,
            "lr": 1e-3,
            "dropout": random.uniform(0.01, 0.80),
            })
    
    # 설정 복사하기
    config = wandb.config

    # 데이터 가져오기
    train_dl = get_dataloader(is_train=True, batch_size=config.batch_size)
    valid_dl = get_dataloader(is_train=False, batch_size=2*config.batch_size)
    n_steps_per_epoch = math.ceil(len(train_dl.dataset) / config.batch_size)
    
    # 간단한 MLP 모델
    model = get_model(config.dropout)

    # 손실과 옵티마이저 만들기
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
                # 🐝 트레이닝 메트릭을 wandb에 로그하기
                wandb.log(metrics)
                
            step_ct += 1

        val_loss, accuracy = validate_model(model, valid_dl, loss_func, log_images=(epoch==(config.epochs-1)))

        # 🐝 트레이닝 및 검증 메트릭을 wandb에 로그하기
        val_metrics = {"val/val_loss": val_loss, 
                       "val/val_accuracy": accuracy}
        wandb.log({**metrics, **val_metrics})
        
        print(f"Train Loss: {train_loss:.3f}, Valid Loss: {val_loss:3f}, Accuracy: {accuracy:.2f}")

    # 테스트 세트가 있었다면, 이렇게 요약 메트릭으로 로그할 수 있습니다.
    wandb.summary['test_accuracy'] = 0.8

    # 🐝 wandb run을 마무리하기
    wandb.finish()
```

이제 wandb를 사용하여 첫 번째 모델을 트레이닝했습니다! 👆 위의 wandb 링크를 클릭하여 메트릭을 확인하세요.

# 🔔 W&B 알림 시도하기

**[W&B 알림](https://docs.wandb.ai/guides/track/alert)** 은 Python 코드에서 트리거된 알림을 Slack이나 이메일로 보낼 수 있게 해줍니다. 코드에서 Slack이나 이메일 알림을 보내고 싶은 첫 번째 시도에 따라야 할 2단계가 있습니다:

1) W&B [사용자 설정](https://wandb.ai/settings)에서 알림을 켜기

2) 코드에 `wandb.alert()` 추가하기:

```python
wandb.alert(
    title="정확도 낮음", 
    text=f"정확도가 허용 가능한 임계값 아래입니다"
)
```

**[W&B 알림](https://docs.wandb.ai/guides/track/alert)** 에 대한 전체 문서를 찾을 수 있는 아래의 최소 예제를 확인하여 `wandb.alert` 사용 방법을 확인하세요.


```python
# wandb run 시작하기
wandb.init(project="pytorch-intro")

# 모델 트레이닝 루프 시뮬레이션
acc_threshold = 0.3
for training_step in range(1000):

    # 정확도를 위한 무작위 숫자 생성
    accuracy = round(random.random() + random.random(), 3)
    print(f'정확도는: {accuracy}, {acc_threshold}')
    
    # 🐝 wandb에 정확도 로그하기
    wandb.log({"Accuracy": accuracy})

    # 🔔 정확도가 임계값 이하인 경우, W&B 알림을 발생시키고 run을 멈추기
    if accuracy <= acc_threshold:
        # 🐝 wandb 알림 보내기
        wandb.alert(
            title='정확도 낮음',
            text=f'{training_step} 단계에서 정확도 {accuracy}가 허용 가능한 임계값, {acc_threshold} 아래입니다',
        )
        print('알림이 트리거되었습니다')
        break

# run이 완료되었다고 표시하기 (Jupyter 노트북에서 유용)
wandb.finish()
```

# 다음은 무엇인가요?
다음 튜토리얼에서는 W&B 테이블을 사용하여 모델 예측을 보고 분석하는 방법을 배웁니다:

## 👉 [모델 예측 보기 & 분석하기](tables)