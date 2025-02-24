---
title: Track models and datasets
menu:
  tutorials:
    identifier: ko-tutorials-artifacts
weight: 4
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Pipeline_Versioning_with_W&B_Artifacts.ipynb" >}}
이 노트북에서는 W&B Artifacts를 사용하여 ML 실험 파이프라인을 추적하는 방법을 보여드리겠습니다.

[비디오 튜토리얼](http://tiny.cc/wb-artifacts-video)을 따라해 보세요.

## Artifacts 정보

Artifact는 그리스의 [암포라](https://en.wikipedia.org/wiki/Amphora)처럼,
프로세스의 결과물로 생산된 오브젝트입니다.
ML에서 가장 중요한 Artifact는 _데이터셋_ 과 _모델_ 입니다.

그리고 [코로나도 십자가](https://indianajones.fandom.com/wiki/Cross_of_Coronado)처럼, 이러한 중요한 Artifact는 박물관에 있어야 합니다.
즉, 카탈로그화되고 정리되어야 합니다.
그래야 여러분, 여러분의 팀, 그리고 더 나아가 ML 커뮤니티가 이를 통해 배울 수 있습니다.
결국 트레이닝을 추적하지 않는 사람들은 그것을 반복할 운명에 처하게 됩니다.

Artifacts API를 사용하여 W&B `Run`의 출력으로 `Artifact`를 기록하거나, 아래 다이어그램처럼 `Artifact`를 `Run`의 입력으로 사용할 수 있습니다.
여기서 트레이닝 run은 데이터셋을 입력으로 받아 모델을 생성합니다.

 {{< img src="/images/tutorials/artifacts-diagram.png" alt="" >}}

하나의 run이 다른 run의 출력을 입력으로 사용할 수 있기 때문에, `Artifact`와 `Run`은 함께 방향성 그래프(이분 [DAG](https://en.wikipedia.org/wiki/Directed_acyclic_graph))를 형성합니다.
이 그래프는 `Artifact`와 `Run`의 노드,
그리고 `Run`을 소비하거나 생성하는 `Artifact`에 연결하는 화살표로 구성됩니다.

## Artifacts를 사용하여 모델과 데이터셋 추적

### 설치 및 임포트

Artifacts는 `0.9.2` 버전부터 Python 라이브러리의 일부입니다.

ML Python 스택의 대부분과 마찬가지로 `pip`를 통해 사용할 수 있습니다.

```python
# Compatible with wandb version 0.9.2+
!pip install wandb -qqq
!apt install tree
```

```python
import os
import wandb
```

### 데이터셋 로깅

먼저 몇 가지 Artifact를 정의해 보겠습니다.

이 예제는 PyTorch의
["Basic MNIST Example"](https://github.com/pytorch/examples/tree/master/mnist/)를 기반으로 하지만,
[TensorFlow](http://wandb.me/artifacts-colab)나 다른 프레임워크,
또는 순수 Python에서도 쉽게 수행할 수 있습니다.

`Dataset`부터 시작하겠습니다.
- 파라미터를 선택하기 위한 `train`ing 세트,
- 하이퍼파라미터를 선택하기 위한 `validation` 세트,
- 최종 모델을 평가하기 위한 `test`ing 세트

아래의 첫 번째 셀은 이러한 세 가지 데이터셋을 정의합니다.

```python
import random 

import torch
import torchvision
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data parameters
num_classes = 10
input_shape = (1, 28, 28)

# drop slow mirror from list of MNIST mirrors
torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                      if not mirror.startswith("http://yann.lecun.com")]

def load(train_size=50_000):
    """
    # Load the data
    """

    # the data, split between train and test sets
    train = torchvision.datasets.MNIST("./", train=True, download=True)
    test = torchvision.datasets.MNIST("./", train=False, download=True)
    (x_train, y_train), (x_test, y_test) = (train.data, train.targets), (test.data, test.targets)

    # split off a validation set for hyperparameter tuning
    x_train, x_val = x_train[:train_size], x_train[train_size:]
    y_train, y_val = y_train[:train_size], y_train[train_size:]

    training_set = TensorDataset(x_train, y_train)
    validation_set = TensorDataset(x_val, y_val)
    test_set = TensorDataset(x_test, y_test)

    datasets = [training_set, validation_set, test_set]

    return datasets
```

여기서는 이 예제에서 반복되는 패턴을 설정합니다.
데이터를 Artifact로 로깅하는 코드는
해당 데이터를 생성하는 코드 주위에 래핑됩니다.
이 경우 데이터를 `load`하는 코드는
데이터를 `load_and_log`하는 코드와 분리됩니다.

이것은 좋은 습관입니다.

이러한 데이터셋을 Artifacts로 로깅하려면,
다음과 같이 해야 합니다.
1. `wandb.init`으로 `Run`을 생성하고 (L4)
2. 데이터셋에 대한 `Artifact`를 생성하고 (L10)
3. 관련 `file`을 저장하고 로깅합니다 (L20, L23).

아래 코드 셀의 예제를 확인하고
자세한 내용은 아래 섹션을 확장하십시오.

```python
def load_and_log():

    # 🚀 start a run, with a type to label it and a project it can call home
    with wandb.init(project="artifacts-example", job_type="load-data") as run:
        
        datasets = load()  # separate code for loading the datasets
        names = ["training", "validation", "test"]

        # 🏺 create our Artifact
        raw_data = wandb.Artifact(
            "mnist-raw", type="dataset",
            description="Raw MNIST dataset, split into train/val/test",
            metadata={"source": "torchvision.datasets.MNIST",
                      "sizes": [len(dataset) for dataset in datasets]})

        for name, data in zip(names, datasets):
            # 🐣 Store a new file in the artifact, and write something into its contents.
            with raw_data.new_file(name + ".pt", mode="wb") as file:
                x, y = data.tensors
                torch.save((x, y), file)

        # ✍️ Save the artifact to W&B.
        run.log_artifact(raw_data)

load_and_log()
```

#### `wandb.init`

`Artifact`를 생성할 `Run`을 만들 때,
어떤 `project`에 속하는지 명시해야 합니다.

워크플로우에 따라,
프로젝트는 `car-that-drives-itself`만큼 클 수도 있고
`iterative-architecture-experiment-117`만큼 작을 수도 있습니다.

> **경험 법칙**: 가능하다면 `Artifact`를 공유하는 모든 `Run`을
단일 프로젝트 내부에 보관하십시오. 이렇게 하면 간단하게 유지할 수 있습니다.
하지만 걱정하지 마십시오. `Artifact`는 프로젝트 간에 이동할 수 있습니다.

실행할 수 있는 모든 다양한 종류의 작업을 추적하는 데 도움이 되도록,
`Runs`를 만들 때 `job_type`을 제공하는 것이 유용합니다.
이렇게 하면 Artifacts 그래프가 깔끔하게 유지됩니다.

> **경험 법칙**: `job_type`은 설명적이어야 하며 파이프라인의 단일 단계에 해당해야 합니다. 여기서는 데이터를 `load`하는 것과 데이터를 `preprocess`하는 것을 분리합니다.

#### `wandb.Artifact`

무언가를 `Artifact`로 로깅하려면 먼저 `Artifact` 오브젝트를 만들어야 합니다.

모든 `Artifact`에는 `name`이 있습니다. 이것이 첫 번째 인수를 설정하는 것입니다.

> **경험 법칙**: `name`은 설명적이어야 하지만 기억하고 입력하기 쉬워야 합니다.
우리는 하이픈으로 구분되고 코드의 변수 이름에 해당하는 이름을 사용하는 것을 좋아합니다.

또한 `type`이 있습니다. `Run`의 `job_type`과 마찬가지로,
이는 `Run`과 `Artifact`의 그래프를 구성하는 데 사용됩니다.

> **경험 법칙**: `type`은 간단해야 합니다.
`mnist-data-YYYYMMDD`보다는 `dataset` 또는 `model`과 같아야 합니다.

`description`과 일부 `metadata`를 사전으로 첨부할 수도 있습니다.
`metadata`는 JSON으로 직렬화할 수 있어야 합니다.

> **경험 법칙**: `metadata`는 가능한 한 설명적이어야 합니다.

#### `artifact.new_file` 및 `run.log_artifact`

`Artifact` 오브젝트를 만들었으면 파일을 추가해야 합니다.

맞습니다. _파일_입니다.
`Artifact`는 디렉토리처럼 구성되며,
파일과 하위 디렉토리가 있습니다.

> **경험 법칙**: 가능하다면 `Artifact`의 내용을
여러 파일로 분할하십시오. 이렇게 하면 확장해야 할 때 도움이 됩니다.

`new_file` 메서드를 사용하여
파일을 작성하고 `Artifact`에 동시에 첨부합니다.
아래에서는 `add_file` 메서드를 사용합니다.
이 메서드는 두 단계를 분리합니다.

모든 파일을 추가했으면 [wandb.ai](https://wandb.ai)에 `log_artifact`해야 합니다.

출력에 몇 개의 URL이 나타나는 것을 알 수 있습니다.
그중 하나는 Run 페이지에 대한 것입니다.
거기에서 로깅된 `Artifact`를 포함하여 `Run`의 결과를 볼 수 있습니다.

아래에서는 Run 페이지의 다른 구성 요소를 더 잘 활용하는 몇 가지 예제를 살펴보겠습니다.

### 로깅된 데이터셋 Artifact 사용

박물관의 Artifact와 달리 W&B의 `Artifact`는
단순히 저장되는 것이 아니라 _사용_되도록 설계되었습니다.

그것이 어떻게 보이는지 살펴봅시다.

아래 셀은 원시 데이터셋을 입력으로 사용하여
`preprocess`된 데이터셋을 생성하는 파이프라인 단계를 정의합니다.
`normalize`되고 올바르게 모양이 지정되었습니다.

다시 말하지만 코드의 핵심인 `preprocess`를
`wandb`와 상호 작용하는 코드와 분리했습니다.

```python
def preprocess(dataset, normalize=True, expand_dims=True):
    """
    ## Prepare the data
    """
    x, y = dataset.tensors

    if normalize:
        # Scale images to the [0, 1] range
        x = x.type(torch.float32) / 255

    if expand_dims:
        # Make sure images have shape (1, 28, 28)
        x = torch.unsqueeze(x, 1)
    
    return TensorDataset(x, y)
```

이제 `wandb.Artifact` 로깅으로 이 `preprocess` 단계를 계측하는 코드입니다.

아래 예제에서는 `Artifact`를 `use`하고
(새로운 기능)
`log`합니다.
(마지막 단계와 동일)
`Artifact`는 `Run`의 입력이자 출력입니다.

새로운 `job_type`인 `preprocess-data`를 사용하여
이것이 이전 작업과는 다른 종류의 작업임을 명확히 합니다.

```python
def preprocess_and_log(steps):

    with wandb.init(project="artifacts-example", job_type="preprocess-data") as run:

        processed_data = wandb.Artifact(
            "mnist-preprocess", type="dataset",
            description="Preprocessed MNIST dataset",
            metadata=steps)
         
        # ✔️ declare which artifact we'll be using
        raw_data_artifact = run.use_artifact('mnist-raw:latest')

        # 📥 if need be, download the artifact
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

여기서 주목해야 할 한 가지는 전처리 `steps`가
`preprocessed_data`와 함께 `metadata`로 저장된다는 것입니다.

실험을 재현 가능하게 만들려면
많은 메타데이터를 캡처하는 것이 좋습니다.

또한 데이터셋이 "`large artifact`"임에도 불구하고
`download` 단계는 1초도 안 걸립니다.

자세한 내용은 아래 markdown 셀을 확장하십시오.

```python
steps = {"normalize": True,
         "expand_dims": True}

preprocess_and_log(steps)
```

#### `run.use_artifact`

이러한 단계는 더 간단합니다. 소비자는 `Artifact`의 `name`과 약간의 정보를 더 알아야 합니다.

그 "약간의 정보"는 원하는 특정 버전의 `Artifact`의 `alias`입니다.

기본적으로 마지막으로 업로드된 버전에는 `latest` 태그가 지정됩니다.
그렇지 않으면 `v0`/`v1` 등으로 이전 버전을 선택하거나,
`best` 또는 `jit-script`와 같은 사용자 지정 에일리어스를 제공할 수 있습니다.
[Docker Hub](https://hub.docker.com/) 태그와 마찬가지로
에일리어스는 `:`로 이름과 구분되므로,
원하는 `Artifact`는 `mnist-raw:latest`입니다.

> **경험 법칙**: 에일리어스를 짧고 간결하게 유지하십시오.
일부 속성을 충족하는 `Artifact`를 원할 때는
`latest` 또는 `best`와 같은 사용자 지정 `alias`를 사용하십시오.

#### `artifact.download`

이제 `download` 호출에 대해 걱정할 수 있습니다.
다른 복사본을 다운로드하면 메모리에 대한 부담이 두 배가 되지 않을까요?

걱정하지 마세요. 실제로 다운로드하기 전에
올바른 버전을 로컬에서 사용할 수 있는지 확인합니다.
여기에는 [토렌트](https://en.wikipedia.org/wiki/Torrent_file) 및 [`git`을 사용한 버전 관리](https://blog.thoughtram.io/git/2014/11/18/the-anatomy-of-a-git-commit.html)의 기본 기술인 해싱이 사용됩니다.

`Artifact`가 생성되고 로깅되면
작업 디렉토리의 `artifacts`라는 폴더가
각 `Artifact`에 대해 하나씩 하위 디렉토리로 채워지기 시작합니다.
`!tree artifacts`로 내용을 확인하십시오.

```python
!tree artifacts
```

#### Artifacts 페이지

이제 `Artifact`를 로깅하고 사용했으므로
Run 페이지에서 Artifacts 탭을 확인해 보겠습니다.

`wandb` 출력에서 Run 페이지 URL로 이동하고
왼쪽 사이드바에서 "Artifacts" 탭을 선택합니다.
(데이터베이스 아이콘이 있는 탭이며,
하키 퍽 세 개가 서로 위에 쌓여 있는 것처럼 보입니다.)

**Input Artifacts** 테이블
또는 **Output Artifacts** 테이블에서 행을 클릭하고
탭(**Overview**, **Metadata**)을 확인하여
`Artifact`에 대해 로깅된 모든 것을 확인합니다.

특히 **Graph View**가 마음에 듭니다.
기본적으로 `Artifact`의 `type`과
`Run`의 `job_type`을 두 가지 유형의 노드로 표시하고,
화살표를 사용하여 소비와 생산을 나타냅니다.

### 모델 로깅

`Artifact`용 API가 어떻게 작동하는지 확인하기에는 충분하지만,
이 예제를 파이프라인의 끝까지 따라가서
`Artifact`가 ML 워크플로우를 어떻게 개선할 수 있는지 살펴보겠습니다.

여기 첫 번째 셀은 PyTorch에서 DNN `model`을 빌드합니다.
매우 간단한 ConvNet입니다.

먼저 `model`을 초기화하기만 하고 트레이닝하지는 않겠습니다.
이렇게 하면 다른 모든 것을 일정하게 유지하면서 트레이닝을 반복할 수 있습니다.

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

여기서는 W&B를 사용하여 run을 추적하고 있으므로
[`wandb.config`](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-config/Configs_in_W%26B.ipynb)
오브젝트를 사용하여 모든 하이퍼파라미터를 저장합니다.

해당 `config` 오브젝트의 `dict`ionary 버전은 매우 유용한 `metadata`이므로 포함해야 합니다.

```python
def build_model_and_log(config):
    with wandb.init(project="artifacts-example", job_type="initialize", config=config) as run:
        config = wandb.config
        
        model = ConvNet(**config)

        model_artifact = wandb.Artifact(
            "convnet", type="model",
            description="Simple AlexNet style CNN",
            metadata=dict(config))

        torch.save(model.state_dict(), "initialized_model.pth")
        # ➕ another way to add a file to an Artifact
        model_artifact.add_file("initialized_model.pth")

        wandb.save("initialized_model.pth")

        run.log_artifact(model_artifact)

model_config = {"hidden_layer_sizes": [32, 64],
                "kernel_sizes": [3],
                "activation": "ReLU",
                "pool_sizes": [2],
                "dropout": 0.5,
                "num_classes": 10}

build_model_and_log(model_config)
```

#### `artifact.add_file`

데이터셋 로깅 예제에서처럼
`new_file`을 동시에 작성하고 `Artifact`에 추가하는 대신,
한 단계에서 파일을 작성할 수도 있습니다.
(여기서는 `torch.save`)
그런 다음 다른 단계에서 `Artifact`에 `add`합니다.

> **경험 법칙**: 중복을 방지하려면 가능한 경우 `new_file`을 사용하십시오.

#### 로깅된 모델 Artifact 사용

`dataset`에서 `use_artifact`를 호출할 수 있는 것처럼
`initialized_model`에서 호출하여
다른 `Run`에서 사용할 수 있습니다.

이번에는 `model`을 `train`해 보겠습니다.

자세한 내용은 다음 Colab을 참조하십시오.
[PyTorch로 W&B 계측](http://wandb.me/pytorch-colab).

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

        # evaluate the model on the validation set at each epoch
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
            test_loss += F.cross_entropy(output, target, reduction='sum')  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)
    
    return test_loss, accuracy


def train_log(loss, example_ct, epoch):
    loss = float(loss)

    # where the magic happens
    wandb.log({"epoch": epoch, "train/loss": loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")
    

def test_log(loss, accuracy, example_ct, epoch):
    loss = float(loss)
    accuracy = float(accuracy)

    # where the magic happens
    wandb.log({"epoch": epoch, "validation/loss": loss, "validation/accuracy": accuracy}, step=example_ct)
    print(f"Loss/accuracy after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}/{accuracy:.3f}")
```

이번에는 두 개의 개별 `Artifact` 생성 `Run`을 실행합니다.

첫 번째 `model` 트레이닝이 완료되면
`second`는 `trained-model` `Artifact`를 소비합니다.
`test_dataset`에서 성능을 `evaluate`합니다.

또한 네트워크가 가장 혼란스러워하는 32개의 예제를 추출합니다.
`categorical_crossentropy`가 가장 높은 예제입니다.

이는 데이터셋 및 모델의 문제를 진단하는 좋은 방법입니다.

```python
def evaluate(model, test_loader):
    """
    ## Evaluate the trained model
    """

    loss, accuracy = test(model, test_loader)
    highest_losses, hardest_examples, true_labels, predictions = get_hardest_k_examples(model, test_loader.dataset)

    return loss, accuracy, highest_losses, hardest_examples, true_labels, predictions

def get_hardest_k_examples(model, testing_set, k=32):
    model.eval()

    loader = DataLoader(testing_set, 1, shuffle=False)

    # get the losses and predictions for each item in the dataset
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

이러한 로깅 함수는 새로운 `Artifact` 기능을 추가하지 않으므로
설명하지 않겠습니다.
`Artifact`를 `use`하고, `download`하고, `log`하고 있습니다.

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