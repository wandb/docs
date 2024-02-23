
# 모델과 데이터세트 추적하기

[**여기서 Colab 노트북에서 시도해보세요 →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Pipeline_Versioning_with_W&B_Artifacts.ipynb)

이 노트북에서는 W&B Artifacts를 사용하여 ML 실험 파이프라인을 추적하는 방법을 보여드립니다.

### [비디오 튜토리얼](http://tiny.cc/wb-artifacts-video)을 따라 해보세요!

### 🤔 Artifacts란 무엇이며 왜 중요한가요?

"artifact"는 그리스 [앰포라 🏺](https://en.wikipedia.org/wiki/Amphora)처럼,
프로세스의 출력물인 생성된 개체입니다.
ML에서 가장 중요한 아티팩트는 _데이터세트_와 _모델_입니다.

그리고 [코로나도의 십자가](https://indianajones.fandom.com/wiki/Cross_of_Coronado)처럼, 이 중요한 아티팩트는 박물관에 속합니다!
즉, 그것들은 카탈로그되고 구성되어야 합니다.
그래서 당신, 당신의 팀, 그리고 전체 ML 커뮤니티가 그것들로부터 배울 수 있습니다.
결국, 학습을 추적하지 않는 이들은 그것을 반복하게 됩니다.

우리의 Artifacts API를 사용하면, W&B `Run`의 출력물로 `Artifact`을 로그하거나 `Run`에 `Artifact`을 입력으로 사용할 수 있습니다. 이 다이어그램에서는 학습 실행이 데이터세트를 입력으로 받아 모델을 생성합니다.
 
 ![](https://gblobscdn.gitbook.com/assets%2F-Lqya5RvLedGEWPhtkjU%2F-M94QAXA-oJmE6q07_iT%2F-M94QJCXLeePzH1p_fW1%2Fsimple%20artifact%20diagram%202.png?alt=media&token=94bc438a-bd3b-414d-a4e4-aa4f6f359f21)

한 실행이 다른 실행의 출력을 입력으로 사용할 수 있기 때문에, Artifacts와 Runs는 함께 방향성 그래프 -- 실제로는 이분(DAG) [DAG](https://en.wikipedia.org/wiki/Directed_acyclic_graph)! -- 를 형성합니다. 여기서 `Artifact`와 `Run`은 노드로,
`Run`과 그들이 소비하거나 생성하는 `Artifact`를 연결하는 화살표가 있습니다.

# 0️⃣ 설치 및 임포트

Artifacts는 우리 Python 라이브러리의 일부로, `0.9.2` 버전부터 시작됩니다.

ML Python 스택의 대부분과 마찬가지로, `pip`을 통해 사용할 수 있습니다.


```python
# wandb 버전 0.9.2+와 호환됩니다
!pip install wandb -qqq
!apt install tree
```


```python
import os
import wandb
```

# 1️⃣ 데이터세트 로그하기

먼저, 몇 가지 Artifacts를 정의해 보겠습니다.

이 예제는 PyTorch의
["기본 MNIST 예제"](https://github.com/pytorch/examples/tree/master/mnist/)를 기반으로 하지만, [TensorFlow](http://wandb.me/artifacts-colab)에서나 다른 프레임워크에서, 또는 순수 Python에서도 마찬가지로 수행될 수 있습니다.

우리는 `Dataset`으로 시작합니다:
- 파라미터를 선택하기 위한 `train` 학습 세트,
- 하이퍼파라미터를 선택하기 위한 `validation` 검증 세트,
- 최종 모델을 평가하기 위한 `test` 테스트 세트

아래의 첫 번째 셀은 이 세 데이터세트를 정의합니다.


```python
import random 

import torch
import torchvision
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm

# 결정적인 동작을 보장합니다
torch.backends.cudnn.deterministic = True
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# 디바이스 구성
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 데이터 파라미터
num_classes = 10
input_shape = (1, 28, 28)

# MNIST 미러 목록에서 느린 미러 제거
torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                      if not mirror.startswith("http://yann.lecun.com")]

def load(train_size=50_000):
    """
    # 데이터 로드하기
    """

    # 학습 및 테스트 세트로 분할된 데이터
    train = torchvision.datasets.MNIST("./", train=True, download=True)
    test = torchvision.datasets.MNIST("./", train=False, download=True)
    (x_train, y_train), (x_test, y_test) = (train.data, train.targets), (test.data, test.targets)

    # 하이퍼파라미터 튜닝을 위한 검증 세트 분리
    x_train, x_val = x_train[:train_size], x_train[train_size:]
    y_train, y_val = y_train[:train_size], y_train[train_size:]

    training_set = TensorDataset(x_train, y_train)
    validation_set = TensorDataset(x_val, y_val)
    test_set = TensorDataset(x_test, y_test)

    datasets = [training_set, validation_set, test_set]

    return datasets
```

이것은 우리가 이 예제에서 반복해서 보게 될 패턴을 설정합니다:
데이터를 생성하는 코드 주위에 데이터를 로그하는 코드를 래핑합니다.
이 경우, `load`하는 코드는
`load_and_log`하는 코드에서 분리되어 있습니다.

이것은 좋은 관행입니다!

이 데이터세트를 Artifacts로 로그하려면,
1. `wandb.init`으로 `Run`을 생성합니다. (L4)
2. 데이터세트에 대한 `Artifact`를 생성합니다. (L10)
3. 관련된 `file`들을 저장하고 로그합니다. (L20, L23).

아래 코드 셀의 예제를 확인한 다음, 자세한 내용은 이후 섹션을 확장하세요.


```python
def load_and_log():

    # 🚀 실행을 시작하고, 그것을 분류할 타입과 프로젝트를 지정합니다
    with wandb.init(project="artifacts-example", job_type="load-data") as run:
        
        datasets = load()  # 데이터세트를 로드하는 별도의 코드
        names = ["training", "validation", "test"]

        # 🏺 우리의 Artifact를 생성합니다
        raw_data = wandb.Artifact(
            "mnist-raw", type="dataset",
            description="Raw MNIST 데이터세트, train/val/test로 분할",
            metadata={"source": "torchvision.datasets.MNIST",
                      "sizes": [len(dataset) for dataset in datasets]})

        for name, data in zip(names, datasets):
            # 🐣 새 파일을 아티팩트에 저장하고, 그 내용에 무언가를 작성합니다.
            with raw_data.new_file(name + ".pt", mode="wb") as file:
                x, y = data.tensors
                torch.save((x, y), file)

        # ✍️ W&B에 아티팩트를 저장합니다.
        run.log_artifact(raw_data)

load_and_log()
```

### 🚀 `wandb.init`

`Artifact`를 생성할 `Run`을 만들 때, 그것이 속한 `project`를 명시해야 합니다.

귀하의 워크플로에 따라,
프로젝트는 `car-that-drives-itself`처럼 크거나 `iterative-architecture-experiment-117`처럼 작을 수 있습니다.

> **👍의 규칙**: 가능하다면, `Artifact`를 공유하는 모든 `Run`을 하나의 프로젝트 내에 유지하세요. 이것은 사물을 단순하게 유지합니다만, 걱정 마세요 -- `Artifact`는 프로젝트 간에 이동 가능합니다!

다양한 종류의 작업을 추적하기 위해 유용하게, `Run`을 만들 때 `job_type`을 제공하는 것이 좋습니다.
이것은 당신의 Artifacts 그래프를 깔끔하게 유지합니다.

> **👍의 규칙**: `job_type`은 설명적이어야 하며 파이프라인의 단일 단계에 해당해야 합니다. 여기서, 우리는 데이터 `load`와 데이터 `preprocess`를 분리합니다.

### 🏺 `wandb.Artifact`

무언가를 `Artifact`로 로그하려면, 먼저 `Artifact` 객체를 만들어야 합니다.

모든 `Artifact`에는 `name`이 있습니다 -- 첫 번째 인수가 설정하는 것입니다.

> **👍의 규칙**: `name`은 설명적이지만 기억하고 입력하기 쉬워야 합니다 --
코드에서 변수 이름에 해당하는 하이픈으로 구분된 이름을 사용하는 것을 좋아합니다.

또한 `type`이 있습니다. `Run`의 `job_type`처럼,
이것은 `Run`과 `Artifact`의 그래프를 구성하는 데 사용됩니다.

> **👍의 규칙**: `type`은 단순해야 합니다:
`dataset`이나 `model`처럼,
`mnist-data-YYYYMMDD`보다는 단순해야 합니다.

설명과 일부 `metadata`도 첨부할 수 있습니다. `metadata`는 사전입니다.
`metadata`는 JSON으로 직렬화될 수 있어야 합니다.

> **👍의 규칙**: `metadata`는 가능한 한 설명적이어야 합니다.

### 🐣 `artifact.new_file` 및 ✍️ `run.log_artifact`

`Artifact` 객체를 만든 후, 그것에 파일을 추가해야 합니다.

맞습니다: _복수의_ 파일들.
`Artifact`는 디렉터리처럼 구성되어 있으며,
파일과 하위 디렉터리가 있습니다.

> **👍의 규칙**: 가능할 때마다, `Artifact`의 내용을 여러 파일로 분할하세요. 확장할 시간이 올 때 도움이 됩니다!

`new_file` 메서드를 사용하여
파일을 동시에 작성하고 `Artifact`에 첨부합니다.
아래에서는 `add_file` 메서드를 사용할 것입니다.
이 두 단계를 분리합니다.

우리가 모든 파일을 추가한 후에는, [wandb.ai](https://wandb.ai)에 `log_artifact` 해야 합니다.

출력에서 몇 가지 URL이 나타났음을 알 수 있습니다.
그 중 하나는 Run 페이지 URL입니다.
그곳에서 `Run`의 결과를 볼 수 있습니다.
로그된 모든 `Artifact`를 포함합니다.

아래에서는 Run 페이지의 다른 구성 요소를 더 잘 활용하는 예제를 볼 것입니다.

# 2️⃣ 로그된 데이터세트 Artifact 사용하기

W&B의 `Artifact`는 박물관의 유물과 달리,
저장되기만 하는 것이 아니라 _사용_되도록 설계되었습니다.

어떻게 보이는지 살펴보겠습니다.

아래 셀은 원시 데이터세트를 입력으로 받아
`preprocess`된 데이터세트를 생성하는 파이프라인 단계를 정의합니다:
`normalize`되고 올바르게 형태가 지정됩니다.

다시 한번, `wandb`와 인터페이스하는 코드와 코드의 핵심인 `preprocess`를 분리했음을 알 수 있습니다.


```python
def preprocess(dataset, normalize=True, expand_dims=True):
    """
    ## 데이터 준비하기
    """
    x, y = dataset.tensors

    if normalize:
        # 이미지를 [0, 1] 범위로 스케일링합니다
        x = x.type(torch.float32) / 255

    if expand_dims:
        # 이미지가 (1, 28, 28) 형태를 갖도록 합니다
        x = torch.unsqueeze(x, 1)
    
    return TensorDataset(x, y)
```

이제 `wandb.Artifact` 로깅으로 이 `preprocess` 단계를 계측하는 코드입니다.

아래 예제는 새로운 `Artifact`을 `use`하는 것과,
이전 단계와 같이 로그하는 것을 모두 포함합니다.
`Artifact`는 `Run`의 입력과 출력 모두입니다!

새로운 `job_type`, `preprocess-data`를 사용하여 이것이 이전 작업과 다른 종류의 작업임을 명확히 합니다.


```python
def preprocess_and_log(steps):

    with wandb.init(project="artifacts-example", job_type="preprocess-data") as run:

        processed_data = wandb.Artifact(
            "mnist-preprocess", type="dataset",
            description="전처리된 MNIST 데이터세트",
            metadata=steps)
         
        # ✔️ 사용할 artifact를 선언합니다
        raw_data_artifact = run.use_artifact('mnist-raw:latest')

        # 📥 필요한 경우, artifact를 다운로드합니다
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

여기서 주목할 점은 `preprocess` 단계가
`preprocessed_data`의 `metadata`로 저장된다는 것입니다.

실험을 재현 가능하게 만들려고 한다면,
많은 메타데이터를 캡처하는 것이 좋은 생각입니다!

또한, 우리의 데이터세트가 "`large artifact`"임에도 불구하고,
`download` 단계는 1초 미만으로 완료됩니다.

자세한 내용은 아래 마크다운 셀을 확장하세요.


```python
steps = {"normalize": True,
         "expand_dims": True}

preprocess_and_log(steps)
```

### ✔️ `run.use_artifact`

이 단계들은 더 단순합니다. 소비자는 `Artifact`의 `name`과 조금 더 많은 것을 알아야 합니다.

그 "조금 더"는 당신이 원하는 `Artifact`의 특정 버전의 `alias`입니다.

기본적으로, 마지막으로 업로드된 버전은 `latest`로 태그됩니다.
그렇지 않으면, `v0`/`v1` 등으로 이전 버전을 선택하거나,
`best`나 `jit-script`과 같은 자체 별칭을 제공할 수 있습니다.
[Docker Hub](https://hub.docker.com/) 태그처럼,
별칭은 이름과 `:`로 구분되므로 우리가 원하는 `Artifact`는 `mnist-raw:latest`입니다.

> **👍의 규칙**: 별칭을 짧고 달

# 3️⃣ 모델 로그하기

이것으로 `Artifact`의 API가 어떻게 작동하는지 보는 것으로 충분하지만, 워크플로를 개선할 수 있는 방법을 보기 위해 파이프라인의 끝까지 이 예제를 따라가 보겠습니다.

여기 첫 번째 셀은 PyTorch에서 매우 단순한 ConvNet인 DNN `모델`을 구축합니다.

우리는 `모델`을 초기화하는 것부터 시작할 것이며, 학습은 하지 않을 것입니다.
그렇게 함으로써, 다른 모든 것을 일정하게 유지하면서 학습을 반복할 수 있습니다.

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

        fc_input_dims = floor((input_shape[1] - kernel_sizes[0] + 1) / pool_sizes[0]) # 1번 레이어 출력 크기
        fc_input_dims = floor((fc_input_dims - kernel_sizes[-1] + 1) / pool_sizes[-1]) # 2번 레이어 출력 크기
        fc_input_dims = fc_input_dims*fc_input_dims*hidden_layer_sizes[-1] # 3번 레이어 출력 크기

        self.fc = nn.Linear(fc_input_dims, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc(x)
        return x
```

여기서 우리는 W&B를 사용하여 실행을 추적하고, 따라서 [`wandb.config`](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-config/Configs_in_W%26B.ipynb) 객체를 사용하여 모든 하이퍼파라미터를 저장합니다.

`config` 객체의 `dict` 버전은 매우 유용한 `메타데이터` 조각이므로 반드시 포함시켜야 합니다!

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
        # ➕ 아티팩트에 파일 추가하는 또 다른 방법
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

### ➕ `artifact.add_file`

`new_file`을 작성하고 동시에 `Artifact`에 추가하는 대신,
데이터세트 로깅 예제에서처럼,
한 단계에서 파일을 작성(여기서는 `torch.save`)
그리고 다음 단계에서 `Artifact`에 추가할 수도 있습니다.

> **👍의 규칙**: 중복을 방지하기 위해 가능하면 `new_file`을 사용하세요.

# 4️⃣ 로깅된 모델 아티팩트 사용하기

`데이터세트`에 `use_artifact`를 호출할 수 있듯이,
`initialized_model`에 그것을 호출하여 다른 `Run`에서 사용할 수 있습니다.

이번에는 `모델`을 `학습`해 보겠습니다.

자세한 내용은 [PyTorch와 W&B 통합](http://wandb.me/pytorch-colab)에 대한 Colab을 확인하세요.

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

        # 에포크마다 검증 세트에서 모델 평가
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
            test_loss += F.cross_entropy(output, target, reduction='sum')  # 배치 손실 합산
            pred = output.argmax(dim=1, keepdim=True)  # 최대 로그 확률의 인덱스를 얻음
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

이번에는 두 개의 별도의 `Artifact` 생성 `Run`을 실행할 것입니다.

첫 번째가 `모델` 학습을 마치면,
`두 번째`는 `학습된 모델` `아티팩트`를 소비하여 `test_dataset`에서 성능을 `평가`합니다.

또한, 네트워크가 가장 혼란스러워하는 32개의 예시를 찾아냅니다 --
`categorical_crossentropy`가 가장 높은 예시입니다.

이것은 데이터세트와 모델의 문제를 진단하는 좋은 방법입니다!

```python
def evaluate(model, test_loader):
    """
    ## 학습된 모델 평가
    """

    loss, accuracy = test(model, test_loader)
    highest_losses, hardest_examples, true_labels, predictions = get_hardest_k_examples(model, test_loader.dataset)

    return loss, accuracy, highest_losses, hardest_examples, true_labels, predictions

def get_hardest_k_examples(model, testing_set, k=32):
    model.eval()

    loader = DataLoader(testing_set, 1, shuffle=False)

    # 데이터세트의 각 항목에 대한 손실과 예측값을 얻음
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

이 로깅 함수들은 새로운 `아티팩트` 기능을 추가하지 않으므로, 그것들에 대해서는 언급하지 않을 것입니다:
우리는 단지 `아티팩트`를 `사용`하고, `다운로드`하고,
`로그`합니다.

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

`아티팩트`의 `유형`을 변경했다는 것을 알아차렸을 것입니다:
이 `Run`들은 `데이터세트`가 아닌 `모델`을 사용했습니다.
`모델`을 생성하는 `Run`들은 Artifacts 페이지의 그래프 보기에서 `데이터세트`를 생성하는 것과 분리됩니다.

확인해보세요! 이전처럼, Run 페이지로 이동한 다음,
왼쪽 사이드바에서 "Artifacts" 탭을 선택하고,
`아티팩트`를 선택한 다음 "Graph View" 탭을 클릭합니다.

### 💣 폭발 그래프

"Explode"라고 표시된 버튼을 주목했을 것입니다. 그것을 클릭하지 마세요, 그렇게 하면 W&B 본사에 있는 겸손한 저자의 책상 아래에 작은 폭탄이 터질 것입니다!

농담입니다. 그것은 훨씬 더 온화한 방식으로 그래프를 "폭발"시킵니다:
`아티팩트`와 `Run`이 `유형`의 수준이 아니라 개별 인스턴스의 수준에서 분리됩니다:
노드는 `데이터세트`와 `load-data`가 아니라 `데이터세트:mnist-raw:v1`과 `load-data:sunny-smoke-1` 등입니다.

이것은 당신의 파이프라인에 대한 완전한 통찰력을 제공하며,
로그된 메트릭, 메타데이터 등이 모두 당신의 손끝에 있습니다 --
당신이 W&B와 함께 로그하기로 선택한 것에 의해서만 제한됩니다.

# 다음은?
다음 튜토리얼에서는 모델의 변경 사항을 소통하고 모델 개발 수명 주기를 W&B 모델로 관리하는 방법을 배우게 됩니다:

## 👉 [모델 개발 수명 주기 추적](models)