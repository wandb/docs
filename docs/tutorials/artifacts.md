---
title: Track models and datasets
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

<CTAButtons colabLink='https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Pipeline_Versioning_with_W&B_Artifacts.ipynb'/>

이 노트북에서는 W&B Artifacts를 사용하여 ML 실험 파이프라인을 추적하는 방법을 보여 드리겠습니다.

### [동영상 튜토리얼](http://tiny.cc/wb-artifacts-video)을 함께 보세요!

### 🤔 Artifacts란 무엇이며 왜 중요한가요?

"Artifact"는 그리스의 [암포라 🏺](https://en.wikipedia.org/wiki/Amphora)와 같은, 과정의 결과물인 생산된 오브젝트입니다.
ML에서는 가장 중요한 artifacts가 _Datasets_와 _Models_입니다.

그리고 [코로나도 십자가](https://indianajones.fandom.com/wiki/Cross_of_Coronado)처럼, 이러한 중요한 artifacts는 박물관에 있어야 합니다.
즉, 이들은 카탈로그화되고 조직되어야 하며,
이를 통해 여러분과 팀, 그리고 대규모 ML 커뮤니티가 배우게 됩니다.
결국 트레이닝을 추적하지 않는 사람들은 그것을 반복할 운명에 처하게 됩니다.

우리의 Artifacts API를 사용하여, W&B `Run`의 출력으로 `Artifact`를 로그하거나, `Artifact`를 `Run`의 입력으로 사용할 수 있습니다. 아래 다이어그램에서, 트레이닝 run이 데이터셋을 입력으로 받아서 모델을 생성합니다.
 
 ![](/images/tutorials/artifacts-diagram.png)

하나의 run이 다른 run의 출력을 입력으로 사용할 수 있기 때문에, Artifacts와 Runs는 함께 `Artifact`와 `Run`에 대한 노드를 가지고, 그들이 소비하거나 생산하는 `Artifact`로 `Run`을 연결하는 화살표가 있는 이중 [DAG](https://en.wikipedia.org/wiki/Directed_acyclic_graph)를 형성합니다.

# 0️⃣ 설치 및 가져오기

Artifacts는 버전 `0.9.2`부터 포함된 Python 라이브러리의 일부입니다.

ML 파이썬 스택의 대부분의 부분과 마찬가지로, `pip`로 사용할 수 있습니다.

```python
# wandb 버전 0.9.2+와 호환 가능
!pip install wandb -qqq
!apt install tree
```

```python
import os
import wandb
```

# 1️⃣ Dataset 로그

먼저 몇 가지 Artifacts를 정의해 봅시다.

이 예제는 PyTorch ["기본 MNIST 예제"](https://github.com/pytorch/examples/tree/master/mnist/)를 기반으로 하지만, [TensorFlow](http://wandb.me/artifacts-colab) 또는 다른 프레임워크나 순수 파이썬에서 쉽게 수행될 수 있습니다.

우리는 `Dataset`s부터 시작합니다:
- 파라미터를 선택하기 위한 `train`ing set,
- 하이퍼파라미터를 선택하기 위한 `validation` set,
- 최종 모델을 평가하기 위한 `test`ing set

아래 첫 번째 셀은 이 세 가지 데이터를 정의합니다.

```python
import random 

import torch
import torchvision
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm

# 결정적 행동 보장
torch.backends.cudnn.deterministic = True
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# 장치 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 데이터 파라미터
num_classes = 10
input_shape = (1, 28, 28)

# MNIST 미러 목록에서 느린 미러 삭제
torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                      if not mirror.startswith("http://yann.lecun.com")]

def load(train_size=50_000):
    """
    # 데이터 로드
    """

    # 데이터, train 및 test 세트로 분할
    train = torchvision.datasets.MNIST("./", train=True, download=True)
    test = torchvision.datasets.MNIST("./", train=False, download=True)
    (x_train, y_train), (x_test, y_test) = (train.data, train.targets), (test.data, test.targets)

    # 하이퍼파라미터 튜닝을 위한 검증 세트를 분할
    x_train, x_val = x_train[:train_size], x_train[train_size:]
    y_train, y_val = y_train[:train_size], y_train[train_size:]

    training_set = TensorDataset(x_train, y_train)
    validation_set = TensorDataset(x_val, y_val)
    test_set = TensorDataset(x_test, y_test)

    datasets = [training_set, validation_set, test_set]

    return datasets
```

이는 이 예제에서 반복될 패턴을 설정합니다:
데이터를 Artifact로 로그하는 코드는 해당 데이터를 생성하기 위한 코드 주위에 래핑됩니다.
이 경우, 데이터 `load` 코드가 데이터를 `load_and_log`하는 코드와 분리되어 있습니다.

이는 좋은 실천입니다!

이 Datasets를 Artifacts로 로그하려면,
1. `wandb.init`으로 `Run`을 생성하고, (L4)
2. 데이터셋을 위한 `Artifact`를 생성하며 (L10),
3. 관련 `file`s를 저장하고 로그해야 합니다 (L20, L23).

아래 코드 셀의 예제를 확인하고
또한 이후의 섹션을 확장하여 자세한 내용을 보십시오.

```python
def load_and_log():

    # 🚀 run을 시작하고, 이를 라벨링하고 홈으로 호출할 수 있는 프로젝트를 설정합니다.
    with wandb.init(project="artifacts-example", job_type="load-data") as run:
        
        datasets = load()  # 데이터셋 로딩을 위한 독립된 코드
        names = ["training", "validation", "test"]

        # 🏺 우리의 Artifact 생성
        raw_data = wandb.Artifact(
            "mnist-raw", type="dataset",
            description="Raw MNIST dataset, split into train/val/test",
            metadata={"source": "torchvision.datasets.MNIST",
                      "sizes": [len(dataset) for dataset in datasets]})

        for name, data in zip(names, datasets):
            # 🐣 artifact의 새 파일 저장, 파일 내용을 기록합니다.
            with raw_data.new_file(name + ".pt", mode="wb") as file:
                x, y = data.tensors
                torch.save((x, y), file)

        # ✍️ 아티팩트를 W&B에 저장합니다.
        run.log_artifact(raw_data)

load_and_log()
```

### 🚀 `wandb.init`

`Artifact`를 생성할 `Run`을 만들 때,
해당 `project`에 속해 있다는 것을 표기해야 합니다.

작업 흐름에 따라,
프로젝트는 `car-that-drives-itself`처럼 큰 것일 수도 있고
`iterative-architecture-experiment-117`처럼 작은 것일 수도 있습니다.

> **👍 규칙**: `Artifact`를 공유하는 모든 `Run`을 단일 프로젝트 내에 두면 좋습니다. 이렇게 하면 간단하게 유지되지만 걱정 하지 마세요 -- `Artifact`는 프로젝트 간에 이동이 가능합니다!

수행할 수 있는 다양한 작업을 추적하기 위해,
`Runs`를 만들 때 `job_type`을 제공하는 것이 유용합니다.
이것은 Artifacts의 그래프를 깔끔하게 유지합니다.

> **👍 규칙**: `job_type`은 설명적이어야 하며 파이프라인의 단일 단계에 해당해야 합니다. 여기서는 `load` 데이터를 `preprocess` 데이터와 분리합니다.

### 🏺 `wandb.Artifact`

무언가를 `Artifact`로 로그하려면, 먼저 `Artifact` 오브젝트를 만들어야 합니다.

모든 `Artifact`에는 `name`이 있습니다. 첫 번째 인수가 그 것을 설정합니다.

> **👍 규칙**: `name`은 설명적이어야 하며 기억하기 쉽고 입력하기 쉬워야 합니다 -- 우리는 코드의 변수 이름과 일치하고 하이픈으로 구분된 이름을 사용하는 것을 선호합니다.

또한 `type`도 있습니다. `Run`의 `job_type`와 마찬가지로,
`Run`와 `Artifact`의 그래프를 위해 사용됩니다.

> **👍 규칙**: `type`은 `dataset`이나 `model`과 같은 간단한 것이어야 합니다, `mnist-data-YYYYMMDD` 같은 것이 아니라.

또한 `설명`과 사전으로 `metadata`를 첨부할 수 있습니다.
`metadata`는 JSON으로 직렬화 가능해야 합니다.

> **👍 규칙**: `metadata`는 가능한 한 설명적으로 작성해야 합니다.

### 🐣 `artifact.new_file` 및 ✍️ `run.log_artifact`

일단 `Artifact` 오브젝트를 만들었으면, 그것에 파일을 추가해야 합니다.

맞습니다: _파일_입니다. `Artifact`는 디렉토리처럼 구조화되어 있으며,
파일 및 하위 디렉토리를 포함하고 있습니다.

> **👍 규칙**: 만약 `Artifact`의 내용을 여러 파일로 분할할 수 있다면, 그렇게 하십시오. 이는 규모를 늘려야 할 때 도움이 됩니다!

`new_file` 메소드를 사용하여
파일을 동시에 쓰고 `Artifact`에 첨부합니다.
아래에서는
두 단계를 분리하는 `add_file` 메소드를 사용할 것입니다.

모든 파일을 추가한 후에는 [wandb.ai](https://wandb.ai)에 `log_artifact`해야 합니다.

출력에 몇 가지 URL이 나타나는 것을 확인할 수 있습니다.
`Run` 페이지의 URL도 포함됩니다.
여기에서 `Run`의 결과를 볼 수 있으며,
로그된 `Artifact`도 포함됩니다.

아래에서는 `Run` 페이지의 다른 구성 요소를 보다 잘 활용하는 몇 가지 예시를 볼 것입니다.

# 2️⃣ 로깅된 Dataset Artifact 사용

박물관에 있는 artifacts와는 달리, W&B의 `Artifact`는 저장만 되는 것이 아니라 _사용_하게 되어 있습니다.

어떻게 그렇게 가능한지 봅시다.

아래 셀은 원시 데이터셋을 사용하여 `정규화`되어 올바르게 형성된 `preprocess`된 데이터셋을 생성하는 파이프라인 단계를 정의합니다.

또한 우리는 `preprocess`의 코드 실체를 wandb와 인터페이스하는 코드와 분리하는 것을 주목하십시오.

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
        # 이미지의 형태가 (1, 28, 28)인지 확인
        x = torch.unsqueeze(x, 1)

    return TensorDataset(x, y)
```

이제 `wandb.Artifact` 로그와 함께 이 `preprocess` 단계를 도구화하는 코드입니다.

아래 예제는 `Artifact`를 `use`하는 새로운 것과 `로그`하는 이전 단계와 동일한 것을 포함합니다.
`Artifact`는 `Run`의 입력 및 출력 모두가 됩니다!

우리는 새로운 `job_type`인 `preprocess-data`를 사용하여, 이전 단계와는 다른 유형의 작업임을 명확히 합니다.

```python
def preprocess_and_log(steps):

    with wandb.init(project="artifacts-example", job_type="preprocess-data") as run:

        processed_data = wandb.Artifact(
            "mnist-preprocess", type="dataset",
            description="Preprocessed MNIST dataset",
            metadata=steps)
         
        # ✔️ 어떤 artifact를 사용할지 선언
        raw_data_artifact = run.use_artifact('mnist-raw:latest')

        # 📥 필요에 따라 artifact 다운로드
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

여기서 눈에 띄는 것 중 하나는 preprocessing의 `steps`가 `preprocessed_data`에 `metadata`로 저장된다는 것입니다.

실험을 재현 가능하게 하려면, 많은 메타데이터를 캡처하는 것이 좋은 아이디어입니다!

또한 데이터셋이 "`large artifact`"임에도 불구하고 `download` 단계는 1초보다 훨씬 짧은 시간 내에 완료됩니다.

아래 markdown 셀을 펼쳐서 자세한 내용을 확인하십시오.

```python
steps = {"normalize": True,
         "expand_dims": True}

preprocess_and_log(steps)
```

### ✔️ `run.use_artifact`

이 단계들은 더 간단합니다. 소비자는 `Artifact`의 `name`만 알면 됩니다. 플러스 조금 더.

이 "조금 더"는 `Artifact`의 특정 버전의 `에일리어스`입니다.

기본적으로 마지막으로 업로드된 버전은 `latest`로 태그됩니다.
그렇지 않으면 `v0`/`v1` 등으로 더 오래된 버전을 선택할 수 있으며,
또는 `best`나 `jit-script`와 같은 별명을 제공할 수 있습니다.
[Docker Hub](https://hub.docker.com/) 태그와 마찬가지로,
에일리어스는 `:`으로 이름과 분리되며,
필요한 `Artifact`는 `mnist-raw:latest`입니다.

> **👍 규칙**: 별칭은 간단하고 직관적으로 유지하십시오. 특정 속성을 만족하는 `Artifact`가 필요할 때는 `latest`나 `best`와 같은 사용자 정의 `alias`를 사용하십시오.

### 📥 `artifact.download`

이제 `download` 호출에 대해 걱정을 하실 수 있습니다.
다른 복사본을 다운로드한다면 메모리에 대한 부담이 두 배가 되지 않을까요?

걱정하지 마세요. 친구. 우리는 실제로 다운로드하기 전에,
해당 버전이 로컬에서 사용 가능한지를 확인합니다.
이는 [torrenting](https://en.wikipedia.org/wiki/Torrent_file)과 [`git`을 사용한 버전 관리](https://blog.thoughtram.io/git/2014/11/18/the-anatomy-of-a-git-commit.html)를 지원하는 동일한 기술을 사용합니다: 해싱.

`Artifact`가 생성되고 로그되면,
작업 디렉토리의 `artifacts`라는 폴더가 생성되며,
각 `Artifact`에 대한 서브 디렉토리로 채워지기 시작합니다.
`!tree artifacts`로 내용을 확인하세요:

```python
!tree artifacts
```

### 🌐 [wandb.ai](https://wandb.ai)의 Artifacts 페이지

이제 `Artifact`를 기록하고 사용했으니,
Run 페이지의 Artifacts 탭을 확인해 보십시오.

`wandb` 출력의 Run 페이지 URL로 이동하여
왼쪽 사이드바에서 "Artifacts" 탭을 선택합니다
(데이터베이스 아이콘, 즉 세 개의 하키 퍽이 쌓인 것처럼 보이는 아이콘).

"Input Artifacts" 테이블 또는 "Output Artifacts" 테이블에서 한 줄을 클릭한 다음,
탭("Overview", "Metadata")을 확인하여 로그된 `Artifact`에 대한 모든 내용을 확인하세요.

우리는 특히 "Graph View"를 좋아합니다.
기본적으로 이것은 `Artifact`의 `type`과 `Run`의 `job_type`을 두 가지 유형의 노드로 표시하며,
소비 및 생산을 나타내는 화살표를 제공합니다.

# 3️⃣ 모델 로그

`Artifact`의 API가 어떻게 동작하는지 보는 것은 충분하지만,
파이프라인 끝까지 예제를 따라가서, `Artifact`가 ML 워크플로에 어떻게 도움이 될 수 있는지 보도록 합시다.

여기 첫 번째 셀은 매우 간단한 ConvNet인 DNN `모델`을 PyTorch로 구축합니다.

우리는 `model`을 초기화하기만 하고, 트레이닝하지 않을 것입니다.
그렇게 하면 트레이닝을 계속 변경하지 않고 유지할 수 있습니다.

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

### ➕ `artifact.add_file`

`Artifact`에 추가하기 위한 파일을 생성하는 것은
데이터셋 로깅 예제와 달리
`새 파일`을 동시에 작성하고 `Artifact`에 추가하는 것과 다르게
여기서는 파일을 작성하는 한 단계
(여기서는 `torch.save`)에서 수행하고,
다른 단계로 그것을 `Artifact`에 `추가`합니다.

> **👍 규칙**: 중복을 방지하기 위해 가능할 때 `새 파일`을 사용하십시오.

# 4️⃣ 기록된 모델 Artifact 사용

데이터셋에서 `use_artifact`를 호출할 수 있는 것처럼,
다른 `Run`에서 `initialized_model`을 사용하여
해당 `Run`에서 사용될 수 있습니다.

이번에는 `모델`을 `train`할 것입니다.

자세한 내용은 PyTorch와 W&B를 [연결하는 Colab](http://wandb.me/pytorch-colab)를 확인해 보세요.

```python
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

        # 매 에포크마다 검증 세트에서 모델 평가
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
            test_loss += F.cross_entropy(output, target, reduction='sum')  # 배치 손실 합계
            pred = output.argmax(dim=1, keepdim=True)  # 최대 로그 확률의 인덱스 가져오기
            correct += pred.eq(target.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)
    
    return test_loss, accuracy


def train_log(loss, example_ct, epoch):
    loss = float(loss)

    # 마법이 일어나는 곳
    wandb.log({"epoch": epoch, "train/loss": loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")
    

def test_log(loss, accuracy, example_ct, epoch):
    loss = float(loss)
    accuracy = float(accuracy)

    # 마법이 일어나는 곳
    wandb.log({"epoch": epoch, "validation/loss": loss, "validation/accuracy": accuracy}, step=example_ct)
    print(f"Loss/accuracy after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}/{accuracy:.3f}")
```

이번에는 두 개의 별도 `Artifact`-생성 `Run`s를 실행할 것입니다.

첫 번째 `Run`이 `모델`을 `train`하기를 마치면,
`두 번째`는 `test_dataset`에서 성능을 `평가`하여
`trained-model` `Artifact`를 소비할 것입니다.

또한, 네트워크가 가장 헷갈리는 32개의 예제를 끌어낼 것입니다 --
`categorical_crossentropy`가 가장 높은 예제입니다.

이것은 데이터셋과 모델 문제를 진단하는 좋은 방법입니다!

```python
def evaluate(model, test_loader):
    """
    ## 학습된 모델을 평가합니다
    """

    loss, accuracy = test(model, test_loader)
    highest_losses, hardest_examples, true_labels, predictions = get_hardest_k_examples(model, test_loader.dataset)

    return loss, accuracy, highest_losses, hardest_examples, true_labels, predictions

def get_hardest_k_examples(model, testing_set, k=32):
    model.eval()

    loader = DataLoader(testing_set, 1, shuffle=False)

    # 데이터셋의 각 항목에 대한 손실 및 예측 가져오기
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

이 로깅 함수들은 새로운 `Artifact` 기능을 추가하지 않으므로,
코멘트하지 않겠습니다:
우리는 그냥 `Artifact`들을 `사용`하고, `다운로드`하고,
`로그`하고 있습니다.

```python
from torch.utils.data import DataLoader

def train_and_log(config):

    with wandb.init(project="artifacts-example", job_type="train", config=config) as run:
        config = wandb.config

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
        wandb.save("trained_model.pth")

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

        wandb.log({"high-loss-examples":
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

### 🔁 그래프 보기

`Artifact`의 `타입`을 변경한 것을 주목하세요:
이 `Run`들은 `dataset` 대신 `model`을 사용했습니다.
`dataset`을 생성하는 `Run`들은 `model`을 생성하는 `Run`들과는
Artifact 페이지의 그래프 보기에서 분리됩니다.

가서 확인해 보세요! 이전과 마찬가지로, `Run` 페이지로 이동하고,
왼쪽 사이드바에서 "Artifacts" 탭을 선택하고,
`Artifact`를 선택한 다음 "Graph View" 탭을 클릭하십시오.

### 💣 Exploded Graphs

"Explode"라는 버튼을 보셨을 겁니다. 그걸 클릭하지 마세요, 왜냐하면 그것은 W&B 본사의 저희 겸손한 저자의 책상 아래에 작은 폭탄을 터뜨릴 것이기 때문입니다!

농담입니다. 그것은 그래프를 훨씬 더 온화하게 "폭발"시킵니다:
`Artifact`와 `Run`은 단일 인스턴스 수준에서 분리됩니다,
`타입` 수준이 아니라:
노드는 `dataset` 및 `load-data`가 아니라 `dataset:mnist-raw:v1` 및 `load-data:sunny-smoke-1` 등입니다.

이는 여러분의 파이프라인에 대한 완전한 통찰력을 제공하며,
로그된 메트릭, 메타데이터 등을
여러분의 손끝에서 사용할 수 있습니다 --
여러분이 우리와 로그하는 것을 선택하는 것에 의해만 제한됩니다.