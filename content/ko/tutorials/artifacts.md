---
title: 모델과 데이터셋 추적하기
menu:
  tutorials:
    identifier: ko-tutorials-artifacts
weight: 4
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Pipeline_Versioning_with_W&B_Artifacts.ipynb" >}}
이 노트북에서는 W&B Artifacts 를 사용해 ML 실험 파이프라인을 추적하는 방법을 보여드릴게요.

[비디오 튜토리얼](https://tiny.cc/wb-artifacts-video)도 함께 참고해주세요.

## 아티팩트(Artifacts)란?

아티팩트(artifact)는 그리스의 [앰포라](https://en.wikipedia.org/wiki/Amphora)처럼 어떤 프로세스의 결과물, 즉 생성된 오브젝트입니다.  
ML에서 가장 중요한 artifact 는 _데이터셋_ 과 _모델_ 입니다.

그리고, [코로나도의 십자가](https://indianajones.fandom.com/wiki/Cross_of_Coronado)처럼 이러한 중요한 artifact 들은 마치 박물관에 소장되어야 합니다.  
즉, 여러분과 팀, 전체 ML 커뮤니티가 이로부터 더 많은 것을 배울 수 있도록 잘 정리하고 관리해야 합니다.  
트레이닝 과정을 추적하지 않으면, 같은 실수를 반복하기 마련이니까요!

Artifiacts API를 활용하면, W&B의 `Run`에서 생성된 결과물을 `Artifact`로 기록할 수 있고, 다른 `Run`에서 입력값으로 해당 `Artifact`를 사용할 수도 있습니다.  
아래 다이어그램처럼, 트레이닝 run 이 데이터셋을 입력 받아 모델을 생성하는 형태입니다.
 
 {{< img src="/images/tutorials/artifacts-diagram.png" alt="Artifacts workflow diagram" >}}

하나의 run에서 나온 output을 다른 run에서 input으로 사용할 수 있기 때문에, `Artifact`와 `Run`은 함께 방향성 있는 그래프(양분 [DAG](https://en.wikipedia.org/wiki/Directed_acyclic_graph))를 형성하게 됩니다.  
노드는 각각 `Artifact`와 `Run`이고, 화살표로 어떤 `Run`이 어떤 `Artifact`를 생성/소비하는지 표현합니다.

## Artifacts로 모델과 데이터셋을 추적하기

### 설치 및 임포트

Artifacts 는 wandb Python 라이브러리의 `0.9.2` 버전부터 지원됩니다.

주요 ML Python 생태계와 같이, `pip`로 설치할 수 있습니다.

```python
# wandb 0.9.2+ 버전에서 동작합니다
!pip install wandb -qqq
!apt install tree
```

```python
import os
import wandb
```

### 데이터셋 기록(Log a Dataset)

이제 Artifacts 를 정의해봅시다.

아래 예제는 PyTorch의
["Basic MNIST Example"](https://github.com/pytorch/examples/tree/master/mnist/) 를 기반으로 합니다.  
물론 [TensorFlow](https://wandb.me/artifacts-colab)나 다른 프레임워크 또는 pure Python 으로도 적용 가능합니다.

데이터셋을 먼저 준비합니다:
- 파라미터(모델 파라미터) 선택에 사용하는 `train` 세트,
- 하이퍼파라미터를 위해 사용하는 `validation` 세트,
- 최종 모델 평가에 사용하는 `test` 세트

아래 첫 번째 셀에서 이 세 가지 데이터셋을 정의합니다.

```python
import random 

import torch
import torchvision
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm

# 일관된 결과를 위해 결정론적으로 만듦
torch.backends.cudnn.deterministic = True
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# 디바이스 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 데이터 파라미터
num_classes = 10
input_shape = (1, 28, 28)

# MNIST 미러 사이트 중 느린 곳 제외
torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                      if not mirror.startswith("http://yann.lecun.com")]

def load(train_size=50_000):
    """
    # 데이터 로드
    """

    # train/test 세트로 데이터 분할
    train = torchvision.datasets.MNIST("./", train=True, download=True)
    test = torchvision.datasets.MNIST("./", train=False, download=True)
    (x_train, y_train), (x_test, y_test) = (train.data, train.targets), (test.data, test.targets)

    # 하이퍼파라미터 튜닝을 위해 validation 세트 분리
    x_train, x_val = x_train[:train_size], x_train[train_size:]
    y_train, y_val = y_train[:train_size], y_train[train_size:]

    training_set = TensorDataset(x_train, y_train)
    validation_set = TensorDataset(x_val, y_val)
    test_set = TensorDataset(x_test, y_test)

    datasets = [training_set, validation_set, test_set]

    return datasets
```

이제 '데이터를 기록하는 코드'는 '데이터를 생성하는 코드'를 감싸는 형태가 됩니다.  
즉, 데이터를 `load`하는 코드는 따로 분리하고,  
이를 기록하는 `load_and_log` 함수는 그 바깥에서 데이터를 받아 기록합니다.

이 방식이 좋은 습관입니다.

이제 이 데이터셋들을 Artifacts 로 기록하려면,
1.  `wandb.init()`로 `Run`을 생성하고 (L4)
2.  데이터셋을 위한 `Artifact`를 만들고 (L10)
3.  연관된 파일을 저장하고 기록합니다 (L20, L23)

아래 예제 코드를 보고,  
더 자세한 설명은 다음 섹션들을 펼쳐보세요.

```python
def load_and_log():

    # Run 생성, 타입(job_type) 지정 및 프로젝트 설정
    with wandb.init(project="artifacts-example", job_type="load-data") as run:
        
        datasets = load()  # 데이터셋 로드
        names = ["training", "validation", "test"]

        # 🏺 Artifact 생성
        raw_data = wandb.Artifact(
            "mnist-raw", type="dataset",
            description="Raw MNIST dataset, split into train/val/test",
            metadata={"source": "torchvision.datasets.MNIST",
                      "sizes": [len(dataset) for dataset in datasets]})

        for name, data in zip(names, datasets):
            # 🐣 Artifact에 새 파일을 저장하고, 내용을 쓴다.
            with raw_data.new_file(name + ".pt", mode="wb") as file:
                x, y = data.tensors
                torch.save((x, y), file)

        # ✍️ Artifact를 W&B에 저장
        run.log_artifact(raw_data)

load_and_log()
```

#### `wandb.init()`

`Artifact`를 생성할 `Run`을 시작할 때는  
어떤 `project`에 소속될 건지 지정해야 합니다.

워크플로우에 따라  
프로젝트의 범위는 'car-that-drives-itself'처럼 크거나  
'iterative-architecture-experiment-117'처럼 작을 수 있습니다.

> **베스트 프랙티스**: 같은 `Artifact`를 공유하는 모든 `Run`은 한 프로젝트에 두세요. 관리가 쉬워집니다.  
> 하지만 `Artifact`는 프로젝트 간 이동도 가능하니 너무 걱정하지 마세요.

여러 종류의 작업을 구분해서 추적할 수 있도록,  
`Run`을 만들 때 `job_type`을 지정하는 것이 좋습니다.  
이렇게 하면 Artifacts 그래프가 군더더기 없이 깔끔해집니다.

> **베스트 프랙티스**: `job_type`은 파이프라인의 단계를 잘 설명하고, 한 단계만 나타내도록 작성하세요.  
> 여기서는 데이터를 `load`하는 것과 `preprocess`하는 것을 분리합니다.

#### `wandb.Artifact`

무언가를 `Artifact`로 기록하려면  
우선 `Artifact` 오브젝트를 생성해야 합니다.

모든 `Artifact`는 `name`을 가집니다 — 첫 번째 인수가 그 역할을 합니다.

> **베스트 프랙티스**: `name`은 직관적이고 타이핑하기 편하도록 하이픈(-)으로 구분하며, 코드 내 변수명과 대응되면 좋습니다.

그리고 `type`도 지정해야 합니다.  
`Run`의 `job_type`처럼,  
전체 실험의 `Run`과 `Artifact` 그래프를 구조화하는 데 쓰입니다.

> **베스트 프랙티스**: `type`은 간단하게!  
> `dataset`이나 `model`처럼 명확하게 작성하고,  
> `mnist-data-YYYYMMDD`처럼 너무 구체적이지 않아도 좋습니다.

`description`과 추가적인 `metadata`도 딕셔너리로 달 수 있습니다.  
여기서 `metadata`는 JSON으로 직렬화 가능한 값이면 됩니다.

> **베스트 프랙티스**: `metadata`에는 가능한 한 자세한 정보를 넣으세요.

#### `artifact.new_file` 및 `run.log_artifact()`

`Artifact` 오브젝트를 만들었으면, 여기에 파일을 추가해야 합니다.

정확히 말씀드리면, _파일들_ 입니다.  
`Artifact`는 디렉토리처럼 구조화되어 있고  
여러 파일과 서브디렉토리를 담을 수 있습니다.

> **베스트 프랙티스**: 가능하다면 `Artifact` 내용을 여러 파일로 나눠 관리하세요.  
> 확장성을 위해 권장합니다.

`new_file` 메소드는 파일 작성과 동시에  
Artifact 에 파일을 부착합니다.  
아래에 나오는 `add_file` 메소드는 두 단계를 나눠서 처리합니다.

모든 파일을 추가했다면, 마지막으로 [wandb.ai](https://wandb.ai) 에 `log_artifact` 합니다.

실행 결과를 보면,  
`Run` 페이지로 연결되는 URL이 나오고  
여기서 기록된 `Artifact`도 확인할 수 있습니다.

아래에서 Run 페이지의 다른 영역도 활용하는 예시를 소개합니다.

### 기록된 Dataset Artifact 활용하기

W&B의 `Artifact`는 박물관에 전시된 것과 달리  
_실제로 사용_하기 위해 만들어진 것입니다.  
저장만 하는 게 아니라 직접 활용할 수 있어요.

아래 셀에서는 파이프라인의 한 단계로  
raw 데이터셋을 입력받아 이를  
전처리(`preprocess`)해서 올바른 형태로 변경합니다.

마찬가지로, 핵심 구현 함수 `preprocess`와  
wandb 인터페이스를 위한 코드를 분리하는 패턴을 사용합니다.

```python
def preprocess(dataset, normalize=True, expand_dims=True):
    """
    ## 데이터 준비
    """
    x, y = dataset.tensors

    if normalize:
        # 이미지를 [0, 1] 범위로 스케일 조정
        x = x.type(torch.float32) / 255

    if expand_dims:
        # 이미지 모양을 (1, 28, 28)로 맞추기
        x = torch.unsqueeze(x, 1)
    
    return TensorDataset(x, y)
```

이번엔 이 `preprocess` 단계를 wandb.Artifact 로 기록하는 코드입니다.

아래 코드는  
기존 단계와 달리  
`Artifact`를 `use`해서 입력으로 쓰고,  
(`use`는 새 기능!)  
`log`하는 부분은 이전 단계와 같습니다.  
즉, `Artifact`는 `Run`의 입력도 될 수 있고 출력도 될 수 있습니다.

여기서는 새로운 `job_type`인 `preprocess-data`를 사용해  
이 단계가 이전과 다르다는 걸 분명히 합니다.

```python
def preprocess_and_log(steps):

    with wandb.init(project="artifacts-example", job_type="preprocess-data") as run:

        processed_data = wandb.Artifact(
            "mnist-preprocess", type="dataset",
            description="Preprocessed MNIST dataset",
            metadata=steps)
         
        # ✔️ 사용할 artifact를 명시적으로 선언
        raw_data_artifact = run.use_artifact('mnist-raw:latest')

        # 📥 필요시 artifact를 다운로드
        raw_dataset = raw_data_artifact.download()
        
        for split in ["training", "validation", "test"]:
            raw_split = read(raw_dataset, split)
            processed_dataset = preprocess(raw_split, **steps)

            with processed_data.new_file(split + ".pt", mode="wb") as file:
                x, y = processed_dataset.tensors
                torch.save((x, y), file)

        run.log_artifact(processed_data)


def read(data_dir, split):
    filename = split + ".pt"
    x, y = torch.load(os.path.join(data_dir, filename))

    return TensorDataset(x, y)
```

여기서 주목할 점:  
전처리 `steps`가  
`preprocessed_data`의 `metadata`로 함께 저장됩니다.

실험을 재현(reproducible)하게 만들고 싶다면,  
메타데이터를 최대한 꼼꼼히 기록하세요.

또한, 우리의 데이터셋이 꽤 "큰 artifact"임에도 불구하고,  
`download` 단계가 1초도 채 걸리지 않습니다.

아래 마크다운 셀을 펼쳐서 자세한 내용을 확인해보세요.

```python
steps = {"normalize": True,
         "expand_dims": True}

preprocess_and_log(steps)
```

#### `run.use_artifact()`

여기서는 간단하게 사용할 수 있습니다.  
사용자는 `Artifact`의 `name`, 그리고 약간의 부가 정보를 알면 됩니다.

바로 그 부가 정보가 바로 버전을 선택할 때 사용하는 `alias` 입니다.

기본적으로, 마지막으로 업로드된 버전은 `latest`로 태그됩니다.  
아니면 `v0`, `v1` 등으로 이전 버전을 선택할 수도 있고,  
`best`, `jit-script`와 같이 직접 에일리어스(alias)를 지정해줄 수도 있습니다.  
[Docker Hub](https://hub.docker.com/)의 태그와 마찬가지로,  
이름과 alias는 `:`로 구분합니다.  
즉, 우리가 원하는 `Artifact`는 `mnist-raw:latest`가 됩니다.

> **베스트 프랙티스**: 에일리어스는 짧고 기억하기 쉽도록 하세요.  
> `latest`나 `best`처럼 커스텀 alias로 특성을 바로 알 수 있게 하는 게 좋습니다.

#### `artifact.download`

`download` 호출이 걱정되실 수도 있겠죠.  
"파일을 한 번 더 다운로드하면 메모리 부담이 커지지 않을까?"

걱정하지 마세요! 다운로드 전에  
해당 버전이 이미 로컬에 있는지 먼저 체크합니다.  
이는 [토렌트](https://en.wikipedia.org/wiki/Torrent_file)나  
[git 버전 관리](https://blog.thoughtram.io/git/2014/11/18/the-anatomy-of-a-git-commit.html)에서 사용되는 해시 기술을 활용합니다.

Artifact가 생성되고 기록될 때마다  
작업 디렉토리에 `artifacts` 폴더가 생기고  
각 Artifact마다 하위 폴더가 채워집니다.  
아래 명령어로 구조를 살펴볼 수 있어요:

```python
!tree artifacts
```

#### Artifacts 페이지

이제 Artifact를 기록하고 사용했으니,  
Run 페이지의 Artifacts 탭을 확인해봅시다.

wandb 출력에서 Run 페이지 URL로 이동해  
왼쪽 사이드바에서 "Artifacts" 탭을 선택하세요  
(데이터베이스 아이콘, 즉 하키 퍽 3개가 쌓인 모양입니다).

**Input Artifacts** 테이블이나 **Output Artifacts** 테이블에서  
행을 클릭해보면  
(**Overview**, **Metadata**) 탭에서 Artifact에 기록된  
모든 정보를 확인할 수 있습니다.

특히 **Graph View**를 추천합니다.  
기본적으로  
Artifact의 `type`과
Run의 `job_type`이 노드가 되고,  
소비와 생성 관계가 화살표로 나타납니다.

### 모델 기록(Log a Model)

여기까지가 Artifact API의 기본 원리이지만,  
마지막 단계까지 파이프라인을 따라가면서  
어떻게 ML 워크플로우를 개선할 수 있는지 보여드릴게요.

아래 셀에서는 PyTorch로 심플한 ConvNet 구조의 DNN 모델을 만듭니다.

우선 모델을 초기화만 하고 학습은 하지 않겠습니다.  
이렇게 하면 트레이닝을 여러 번 반복해도 나머지는 동일하게 유지할 수 있습니다.

```python
from math import floor

import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, hidden_layer_sizes=[32, 64],
                  kernel_sizes=[3],
                  activation="ReLU",
                  pool_sizes=[2],
                  dropout=0.5,
                  num_classes=num_classes,
                  input_shape=input_shape):
      
        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential(
              nn.Conv2d(in_channels=input_shape[0], out_channels=hidden_layer_sizes[0], kernel_size=kernel_sizes[0]),
              getattr(nn, activation)(),
              nn.MaxPool2d(kernel_size=pool_sizes[0])
        )
        self.layer2 = nn.Sequential(
              nn.Conv2d(in_channels=hidden_layer_sizes[0], out_channels=hidden_layer_sizes[-1], kernel_size=kernel_sizes[-1]),
              getattr(nn, activation)(),
              nn.MaxPool2d(kernel_size=pool_sizes[-1])
        )
        self.layer3 = nn.Sequential(
              nn.Flatten(),
              nn.Dropout(dropout)
        )

        fc_input_dims = floor((input_shape[1] - kernel_sizes[0] + 1) / pool_sizes[0]) # layer 1 output size
        fc_input_dims = floor((fc_input_dims - kernel_sizes[-1] + 1) / pool_sizes[-1]) # layer 2 output size
        fc_input_dims = fc_input_dims*fc_input_dims*hidden_layer_sizes[-1] # layer 3 output size

        self.fc = nn.Linear(fc_input_dims, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc(x)
        return x
```

여기서는 W&B로 run을 추적합니다.  
그래서 모든 하이퍼파라미터를  
[`run.config`](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-config/Configs_in_W%26B.ipynb)  
오브젝트에 저장합니다.

이 config 오브젝트의 `dict` 버전은  
메타데이터로서 유용하게 쓰이니 꼭 포함하세요.

```python
def build_model_and_log(config):
    with wandb.init(project="artifacts-example", job_type="initialize", config=config) as run:
        config = run.config
        
        model = ConvNet(**config)

        model_artifact = wandb.Artifact(
            "convnet", type="model",
            description="Simple AlexNet style CNN",
            metadata=dict(config))

        torch.save(model.state_dict(), "initialized_model.pth")
        # ➕ Artifact에 파일을 추가하는 또다른 방법
        model_artifact.add_file("initialized_model.pth")

        run.save("initialized_model.pth")

        run.log_artifact(model_artifact)

model_config = {"hidden_layer_sizes": [32, 64],
                "kernel_sizes": [3],
                "activation": "ReLU",
                "pool_sizes": [2],
                "dropout": 0.5,
                "num_classes": 10}

build_model_and_log(model_config)
```

#### `artifact.add_file()`

데이터셋 기록 예제에서는  
`new_file`로 파일 작성과 동시에 Artifact 등록을 했지만,  
여기서는 파일을 먼저 저장(`torch.save`)하고  
그 다음 `add_file`로 Artifact에 추가하는 방식을 사용했습니다.

> **베스트 프랙티스**: 중복 저장을 막으려면 `new_file`을 권장합니다.

#### 기록된 모델 Artifact 사용하기

데이터셋처럼  
`use_artifact`를 통해  
`initialized_model`도 다른 Run에서 사용할 수 있습니다.

이번에는 모델을 `train`해보겠습니다.

더 자세한 정보는  
[PyTorch + W&B 연동 Colab](https://wandb.me/pytorch-colab)을 참고하세요.

```python
import wandb
import torch.nn.functional as F

def train(model, train_loader, valid_loader, config):
    optimizer = getattr(torch.optim, config.optimizer)(model.parameters())
    model.train()
    example_ct = 0
    for epoch in range(config.epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            example_ct += len(data)

            if batch_idx % config.batch_log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0%})]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    batch_idx / len(train_loader), loss.item()))
                
                train_log(loss, example_ct, epoch)

        # 매 epoch 마다 validation 세트로 모델 평가
        loss, accuracy = test(model, valid_loader)  
        test_log(loss, accuracy, example_ct, epoch)

    
def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum')  # 배치 손실 합
            pred = output.argmax(dim=1, keepdim=True)  # 예측 결과
            correct += pred.eq(target.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)
    
    return test_loss, accuracy


def train_log(loss, example_ct, epoch):
    loss = float(loss)

    # magic이 일어나는 곳
    with wandb.init(project="artifacts-example", job_type="train") as run:
        run.log({"epoch": epoch, "train/loss": loss}, step=example_ct)
        print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")
    

def test_log(loss, accuracy, example_ct, epoch):
    loss = float(loss)
    accuracy = float(accuracy)

    # magic이 일어나는 곳
    with wandb.init() as run:
        run.log({"epoch": epoch, "validation/loss": loss, "validation/accuracy": accuracy}, step=example_ct)
        print(f"Loss/accuracy after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}/{accuracy:.3f}")
```

이번엔 두 개의 `Artifact`를 생성하는 Run을 따로 사용합니다.

첫 번째 Run에서 모델 트레이닝을 끝내면,  
두 번째 Run에서는  
`trained-model` Artifact를 받아  
테스트 데이터셋에서 평가합니다.

또한, 네트워크가 가장 많이 헷갈린  
즉, `categorical_crossentropy`가 가장 컸던 32개의 예제를 뽑아봅니다.

이 방법은 데이터와 모델의 문제를  
진단하기에 매우 유용합니다.

```python
def evaluate(model, test_loader):
    """
    ## 훈련된 모델 평가
    """

    loss, accuracy = test(model, test_loader)
    highest_losses, hardest_examples, true_labels, predictions = get_hardest_k_examples(model, test_loader.dataset)

    return loss, accuracy, highest_losses, hardest_examples, true_labels, predictions

def get_hardest_k_examples(model, testing_set, k=32):
    model.eval()

    loader = DataLoader(testing_set, 1, shuffle=False)

    # 데이터셋의 각 항목에 대해 loss와 예측값 계산
    losses = None
    predictions = None
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            
            if losses is None:
                losses = loss.view((1, 1))
                predictions = pred
            else:
                losses = torch.cat((losses, loss.view((1, 1))), 0)
                predictions = torch.cat((predictions, pred), 0)

    argsort_loss = torch.argsort(losses, dim=0)

    highest_k_losses = losses[argsort_loss[-k:]]
    hardest_k_examples = testing_set[argsort_loss[-k:]][0]
    true_labels = testing_set[argsort_loss[-k:]][1]
    predicted_labels = predictions[argsort_loss[-k:]]

    return highest_k_losses, hardest_k_examples, true_labels, predicted_labels
```

여기서의 로깅 함수는 특별한 Artifact 기능을 더하지는 않습니다.  
단순히 `use`, `download`,  
그리고 `log`만 할 뿐입니다.

```python
from torch.utils.data import DataLoader

def train_and_log(config):

    with wandb.init(project="artifacts-example", job_type="train", config=config) as run:
        config = run.config

        data = run.use_artifact('mnist-preprocess:latest')
        data_dir = data.download()

        training_dataset =  read(data_dir, "training")
        validation_dataset = read(data_dir, "validation")

        train_loader = DataLoader(training_dataset, batch_size=config.batch_size)
        validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size)
        
        model_artifact = run.use_artifact("convnet:latest")
        model_dir = model_artifact.download()
        model_path = os.path.join(model_dir, "initialized_model.pth")
        model_config = model_artifact.metadata
        config.update(model_config)

        model = ConvNet(**model_config)
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
 
        train(model, train_loader, validation_loader, config)

        model_artifact = wandb.Artifact(
            "trained-model", type="model",
            description="Trained NN model",
            metadata=dict(model_config))

        torch.save(model.state_dict(), "trained_model.pth")
        model_artifact.add_file("trained_model.pth")
        run.save("trained_model.pth")

        run.log_artifact(model_artifact)

    return model

    
def evaluate_and_log(config=None):
    
    with wandb.init(project="artifacts-example", job_type="report", config=config) as run:
        data = run.use_artifact('mnist-preprocess:latest')
        data_dir = data.download()
        testing_set = read(data_dir, "test")

        test_loader = torch.utils.data.DataLoader(testing_set, batch_size=128, shuffle=False)

        model_artifact = run.use_artifact("trained-model:latest")
        model_dir = model_artifact.download()
        model_path = os.path.join(model_dir, "trained_model.pth")
        model_config = model_artifact.metadata

        model = ConvNet(**model_config)
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        loss, accuracy, highest_losses, hardest_examples, true_labels, preds = evaluate(model, test_loader)

        run.summary.update({"loss": loss, "accuracy": accuracy})

        run.log({"high-loss-examples":
            [wandb.Image(hard_example, caption=str(int(pred)) + "," +  str(int(label)))
             for hard_example, pred, label in zip(hardest_examples, preds, true_labels)]})
```

```python
train_config = {"batch_size": 128,
                "epochs": 5,
                "batch_log_interval": 25,
                "optimizer": "Adam"}

model = train_and_log(train_config)
evaluate_and_log()
```