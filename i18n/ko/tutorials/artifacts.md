
# 모델과 데이터셋 추적하기

[**여기에서 Colab 노트북으로 시도해 보세요 →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Pipeline_Versioning_with_W&B_Artifacts.ipynb)

이 노트북에서는 W&B 아티팩트를 사용하여 ML 실험 파이프라인을 추적하는 방법을 보여드립니다.

### [비디오 튜토리얼](http://tiny.cc/wb-artifacts-video)을 따라 해보세요!

### 🤔 아티팩트는 무엇이고 왜 중요한가요?

"아티팩트"는 그리스 [암포라 🏺](https://en.wikipedia.org/wiki/Amphora)와 같이 생성된 오브젝트입니다 -- 프로세스의 출력물입니다.
ML에서 가장 중요한 아티팩트는 _데이터셋_과 _모델_입니다.

그리고 [코로나도의 십자가](https://indianajones.fandom.com/wiki/Cross_of_Coronado)처럼, 이 중요한 아티팩트는 박물관에 속합니다!
즉, 카탈로그화되어 조직화되어야 합니다
그래서 당신, 당신의 팀, 그리고 ML 커뮤니티 전체가 그것들로부터 배울 수 있습니다.
결국, 트레이닝을 추적하지 않는 이들은 그것을 반복하게 됩니다.

우리의 아티팩트 API를 사용하여, 당신은 W&B `Run`의 출력물로 `아티팩트`를 로그하거나 `Run`에 입력으로 `아티팩트`를 사용할 수 있습니다, 이 다이어그램에서처럼,
여기서 트레이닝 run은 데이터셋을 입력으로 받아 모델을 생성합니다.
 
 ![](https://gblobscdn.gitbook.com/assets%2F-Lqya5RvLedGEWPhtkjU%2F-M94QAXA-oJmE6q07_iT%2F-M94QJCXLeePzH1p_fW1%2Fsimple%20artifact%20diagram%202.png?alt=media&token=94bc438a-bd3b-414d-a4e4-aa4f6f359f21)

하나의 run이 다른 run의 출력을 입력으로 사용할 수 있기 때문에, 아티팩트와 Run은 방향성 그래프 -- 실제로는 이분 [DAG](https://en.wikipedia.org/wiki/Directed_acyclic_graph)! -- 를 형성하며, 노드는 `아티팩트`와 `Run`에 대한 것이고
화살표는 `Run`이 소비하거나 생성하는 `아티팩트`를 연결합니다.

# 0️⃣ 설치 및 가져오기

아티팩트는 우리의 Python 라이브러리의 일부이며, 버전 `0.9.2`부터 시작됩니다.

대부분의 ML Python 스택과 마찬가지로, `pip`를 통해 사용할 수 있습니다.


```python
# wandb 버전 0.9.2+와 호환됩니다
!pip install wandb -qqq
!apt install tree
```


```python
import os
import wandb
```

# 1️⃣ 데이터셋 로그하기

먼저, 몇 가지 아티팩트를 정의해 보겠습니다.

이 예제는 PyTorch의
["기본 MNIST 예제"](https://github.com/pytorch/examples/tree/master/mnist/)를 기반으로 하지만,
[TensorFlow](http://wandb.me/artifacts-colab), 다른 프레임워크에서도 마찬가지로 쉽게 수행할 수 있으며,
순수 Python에서도 가능합니다.

우리는 `데이터셋`으로 시작합니다:
- 파라미터를 선택하기 위한 `train` 세트,
- 하이퍼파라미터를 선택하기 위한 `validation` 세트,
- 최종 모델을 평가하기 위한 `test` 세트

아래 첫 번째 셀은 이 세 가지 데이터셋을 정의합니다.


```python
import random 

import torch
import torchvision
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm

# 결정적 동작을 보장합니다
torch.backends.cudnn.deterministic = True
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# 장치 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 데이터 파라미터
num_classes = 10
input_shape = (1, 28, 28)

# MNIST 미러 목록에서 느린 미러 제거
torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                      if not mirror.startswith("http://yann.lecun.com")]

def load(train_size=50_000):
    """
    # 데이터를 로드합니다
    """

    # 데이터, 트레인과 테스트 세트로 분할됩니다
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

이는 이 예제에서 반복되는 패턴을 설정합니다:
데이터를 생성하는 코드 주위로 데이터를 로그하는 코드를 래핑합니다.
이 경우, `load`ing 데이터의 코드는
로그하고 `load_and_log`하는 데이터의 코드와 분리됩니다.

이것은 좋은 관행입니다!

이 데이터셋을 아티팩트로 로그하려면,
우리는 단지
1. `wandb.init`으로 `Run`을 생성해야 합니다 (L4),
2. 데이터셋에 대한 `아티팩트`를 생성해야 합니다 (L10), 그리고
3. 관련된 `파일`들을 저장하고 로그해야 합니다 (L20, L23).

아래 코드 셀의 예제를 확인한 다음, 더 자세한 내용을 알아보기 위해 이후 섹션을 확장하세요.


```python
def load_and_log():

    # 🚀 run을 시작하고, 이를 라벨링할 타입과 그것이 속할 프로젝트를 지정합니다
    with wandb.init(project="artifacts-example", job_type="load-data") as run:
        
        datasets = load()  # 데이터셋을 로드하는 별도의 코드
        names = ["training", "validation", "test"]

        # 🏺 우리의 아티팩트를 생성합니다
        raw_data = wandb.Artifact(
            "mnist-raw", type="dataset",
            description="Raw MNIST 데이터셋, 트레인/밸/테스트로 분할됨",
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


`Artifact`를 생성할 `Run`을 만들 때, 이것이 어떤 `프로젝트`에 속하는지를 명시해야 합니다.

당신의 워크플로우에 따라,
프로젝트는 `car-that-drives-itself`처럼 크거나 `iterative-architecture-experiment-117`처럼 작을 수 있습니다.

> **👍 규칙**: 가능하다면, `Artifact`를 공유하는 모든 `Run`을 단일 프로젝트 내에 유지하세요. 이것은 사물을 단순하게 유지하지만, 걱정하지 마세요 -- `아티팩트`는 프로젝트 간에 이동 가능합니다!

모든 다양한 종류의 작업을 추적하기 위해,
`Run`을 만들 때 `job_type`을 제공하는 것이 유용합니다.
이것은 당신의 아티팩트 그래프를 깔끔하게 유지하는 데 도움이 됩니다.

> **👍 규칙**: `job_type`은 설명적이어야 하며 파이프라인의 단일 단계에 해당해야 합니다. 여기서는 데이터를 `load`하는 것과 데이터를 `preprocess`하는 것을 구분합니다.

### 🏺 `wandb.Artifact`


무언가를 `아티팩트`로 로그하려면, 먼저 `아티팩트` 오브젝트를 만들어야 합니다.

모든 `아티팩트`에는 `이름`이 있습니다 -- 그것이 첫 번째 인수가 설정하는 것입니다.

> **👍 규칙**: `이름`은 설명적이지만 기억하기 쉽고 입력하기 쉬워야 합니다 --
우리는 하이픈으로 구분된 이름을 사용하는 것을 좋아하며 코드에서 변수 이름에 해당합니다.

또한 `유형`이 있습니다. `Run`의 `job_type`과 마찬가지로,
이것은 `Run`과 `아티팩트`의 그래프를 구성하는 데 사용됩니다.

> **👍 규칙**: `유형`은 단순해야 합니다:
`dataset`이나 `model`처럼
`mnist-data-YYYYMMDD`보다는 간단해야 합니다.

`설명`과 일부 `메타데이터`를 사전 형태로 첨부할 수도 있습니다.
`메타데이터`는 JSON으로 직렬화될 수 있어야 합니다.

> **👍 규칙**: `메타데이터`는 가능한 한 설명적이어야 합니다.

### 🐣 `artifact.new_file` 및 ✍️ `run.log_artifact`

`아티팩트` 오브젝트를 만든 후, 그것에 파일을 추가해야 합니다.

그렇습니다: _파일들_이라고 복수형으로 말했습니다.
`아티팩트`는 디렉토리처럼 구조화되어 있으며,
파일과 하위 디렉토리가 있습니다.

> **👍 규칙**: 의미가 있을 때마다, `아티팩트`의 내용을 여러 파일로 분할하세요. 이것은 확장할 때 도움이 됩니다!

`new_file` 메소드를 사용하여
파일을 동시에 작성하고 `아티팩트`에 첨부합니다.
아래에서는 `add_file` 메소드를 사용할 것입니다.
이 두 단계를 분리합니다.

모든 파일을 추가한 후, [wandb.ai](https://wandb.ai)에 `log_artifact` 해야 합니다.

출력에 몇 가지 URL이 나타났음을 알 수 있습니다,
`Run` 페이지 URL을 포함합니다.
그것은 `Run`의 결과를 볼 수 있는 곳입니다,
로그된 모든 `아티팩트`를 포함합니다.

아래에서는 Run 페이지의 다른 구성 요소를 더 잘 활용하는 몇 가지 예제를 볼 것입니다.

# 2️⃣ 로그된 데이터셋 아티팩트 사용하기

W&B의 `아티팩트`는 박물관의 아티팩트와 달리,
저장뿐만 아니라 _사용_되도록 설계되었습니다.

그것이 어떤 모습인지 살펴보겠습니다.

아래 셀은 원시 데이터셋을 입력으로 받아
`preprocess`된 데이터셋을 생성하는 파이프라인 단계를 정의합니다:
`정규화`되고 올바르게 형태가 지정됩니다.

다시 한번 `wandb`와 인터페이스하는 코드와 `preprocess`의 핵심 코드를 분리했음을 알 수 있습니다.


```python
def preprocess(dataset, normalize=True, expand_dims=True):
    """
    ## 데이터 준비
    """
    x, y = dataset.tensors

    if normalize:
        # 이미지를 [0, 1] 범위로 스케일링합니다
        x = x.type(torch.float32) / 255

    if expand_dims:
        # 이미지가 (1, 28, 28) 형태를 가지도록 합니다
        x = torch.unsqueeze(x, 1)
    
    return TensorDataset(x, y)
```

이제 `wandb.Artifact` 로깅으로 이 `preprocess` 단계를 계측하는 코드입니다.

아래 예제에서는 `아티팩트`를 `사용`하는 것이 새롭습니다,
그리고 `로그`하는 것은 마지막 단계와 같습니다.
`아티팩트`는 `Run`의 입력과 출력 모두입니다!

새로운 `job_type`, `preprocess-data`를 사용하여
이것이 이전 것과 다른 종류의 작업임을 명확히 합니다.


```python
def preprocess_and_log(steps):

    with wandb.init(project="artifacts-example", job_type="preprocess-data") as run:

        processed_data = wandb.Artifact(
            "mnist-preprocess", type="dataset",
            description="전처리된 MNIST 데이터셋",
            metadata=steps)
         
        # ✔️ 사용할 아티팩트를 선언합니다
        raw_data_artifact = run.use_artifact('mnist-raw:latest')

        # 📥 필요하다면, 아티팩트를 다운로드합니다
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

여기서 주목해야 할 것은 `preprocess` 단계의 `steps`
가 `preprocessed_data`의 `메타데이터`로 저장된다는 것입니다.

실험을 재현 가능하게 만들려면,
많은 메타데이터를 캡처하는 것이 좋은 생각입니다!

또한, 우리의 데이터셋이 "`대규

# 4️⃣ 로그된 모델 아티팩트 사용하기

`데이터셋`에 `use_artifact`를 호출할 수 있듯이, 우리는 `initialized_model`에도 그것을 호출하여 다른 `Run`에서 사용할 수 있습니다.

이번에는 `모델`을 `트레이닝`해봅시다.

더 자세한 내용은 우리의 Colab에서 확인하세요.
[PyTorch와 함께 W&B를 구성하는 방법](http://wandb.me/pytorch-colab)에 대해.

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
                print('트레이닝 에포크: {} [{}/{} ({:.0%})]\t손실: {:.6f}'.format(
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
            pred = output.argmax(dim=1, keepdim=True)  # 최대 로그-확률의 인덱스를 가져옴
            correct += pred.eq(target.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)
    
    return test_loss, accuracy


def train_log(loss, example_ct, epoch):
    loss = float(loss)

    # 마법이 일어나는 곳
    wandb.log({"epoch": epoch, "train/loss": loss}, step=example_ct)
    print(f"예시 " + str(example_ct).zfill(5) + f"개 후 손실: {loss:.3f}")
    

def test_log(loss, accuracy, example_ct, epoch):
    loss = float(loss)
    accuracy = float(accuracy)

    # 마법이 일어나는 곳
    wandb.log({"epoch": epoch, "validation/loss": loss, "validation/accuracy": accuracy}, step=example_ct)
    print(f"예시 " + str(example_ct).zfill(5) + f"개 후 손실/정확도: {loss:.3f}/{accuracy:.3f}")
```

이번에는 두 개의 별도의 `Artifact`을 생성하는 `Run`을 실행합니다.

첫 번째가 `모델` 트레이닝을 마치면,
`두 번째`는 `테스트 데이터셋`에서 `trained-model` `Artifact`의 성능을 `평가`함으로써 그것을 사용합니다.

또한, 네트워크가 가장 혼란스러워하는 32개의 예시들 -- `categorical_crossentropy`가 가장 높은 것들을 찾아냅니다.

이것은 데이터셋과 모델에 대한 문제를 진단하는 좋은 방법입니다!

```python
def evaluate(model, test_loader):
    """
    ## 트레이닝된 모델 평가하기
    """

    loss, accuracy = test(model, test_loader)
    highest_losses, hardest_examples, true_labels, predictions = get_hardest_k_examples(model, test_loader.dataset)

    return loss, accuracy, highest_losses, hardest_examples, true_labels, predictions

def get_hardest_k_examples(model, testing_set, k=32):
    model.eval()

    loader = DataLoader(testing_set, 1, shuffle=False)

    # 데이터셋의 각 항목에 대한 손실과 예측값을 가져옴
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

이 로깅 함수들은 새로운 `Artifact` 기능을 추가하지 않으므로, 우리는 그것들에 대해 언급하지 않을 것입니다:
우리는 단지 `Artifact`을 `사용`하고, `다운로드`하고,
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
            description="트레이닝된 NN 모델",
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

### 🔁 그래프 뷰

`Artifact`의 `type`을 변경했음을 주목하세요:
이 `Run`들은 `데이터셋`이 아닌 `모델`을 사용했습니다.
`모델`을 생성하는 `Run`들은 Artifacts 페이지의 그래프 뷰에서 `데이터셋`을 생성하는 것들과 분리됩니다.

확인해보세요! 이전처럼, Run 페이지로 가서,
왼쪽 사이드바에서 "Artifacts" 탭을 선택하고,
`Artifact`을 선택한 다음,
"Graph View" 탭을 클릭하십시오.

### 💣 터진 그래프

"Explode"라고 표시된 버튼을 주목했을 겁니다. 그것을 클릭하지 마세요, 왜냐하면 그것은 W&B 본사에 있는 겸손한 저자의 책상 아래에 작은 폭탄을 설치할 것이기 때문이죠!

농담입니다. 그것은 그래프를 훨씬 더 부드러운 방법으로 "폭발"시킵니다:
`Artifact`과 `Run`이 `type`의 수준에서 분리되어,
노드들은 `데이터셋`과 `load-data`가 아니라, `데이터셋:mnist-raw:v1`과 `load-data:sunny-smoke-1` 등이 됩니다.

이것은 당신의 파이프라인에 대한 완전한 통찰을 제공합니다,
로그된 메트릭, 메타데이터 등이
모두 당신의 손끝에 있습니다 --
당신이 우리와 함께 로그하기로 선택한 것에 의해서만 제한됩니다.

# 다음은 무엇인가요?
다음 튜토리얼에서는 W&B 모델로 모델 변경 사항을 소통하고 모델 개발 생명주기를 관리하는 방법을 배우게 됩니다:

## 👉 [모델 개발 생명주기 추적하기](models)