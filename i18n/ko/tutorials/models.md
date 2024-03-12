
# 모델 등록하기

[**여기에서 Colab 노트북으로 시도해보세요 →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-model-registry/Model_Registry_E2E.ipynb)

모델 레지스트리는 조직 전반에서 진행 중인 모든 모델 작업과 관련 아티팩트를 보관하고 정리하는 중앙 장소입니다:
- 모델 체크포인트 관리
- 풍부한 모델 카드로 모델 문서화
- 사용/배포되는 모든 모델의 이력 유지
- 모델의 깔끔한 인계 및 단계 관리 용이
- 다양한 모델 작업 태그 및 정리
- 모델 진행 상황에 대한 자동 알림 설정

이 튜토리얼은 간단한 이미지 분류 작업에 대한 모델 개발 라이프사이클을 추적하는 방법을 안내합니다.

### 🛠️ `wandb` 설치
 
```bash
!pip install -q wandb onnx pytorch-lightning
```

## W&B에 로그인
- `wandb login` 또는 `wandb.login()`을 사용하여 명시적으로 로그인할 수 있습니다(아래 참조).
- 또는 환경 변수를 설정할 수 있습니다. W&B 로깅의 행동을 변경할 수 있는 몇 가지 환경 변수가 있습니다. 가장 중요한 것은:
    - `WANDB_API_KEY` - 프로필 아래 "설정" 섹션에서 찾을 수 있습니다.
    - `WANDB_BASE_URL` - W&B 서버의 URL입니다.
- W&B 앱에서 "프로필" -> "설정"에서 API 토큰 찾기

![api_token](https://drive.google.com/uc?export=view&id=1Xn7hnn0rfPu_EW0A_-32oCXqDmpA0-kx) 

```notebook
!wandb login
```

:::note
[W&B 서버](..//guides/hosting/intro.md) 배포(전용 클라우드 또는 자체 관리)에 연결할 때는 --relogin 및 --host 옵션을 사용하세요:

```notebook
!wandb login --relogin --host=http://your-shared-local-host.com
```

필요한 경우 배포 관리자에게 호스트 이름을 문의하세요.
:::

## 데이터 및 모델 체크포인트를 아티팩트로 로깅하기  
W&B 아티팩트를 사용하면 임의의 직렬화 데이터(예: 데이터셋, 모델 체크포인트, 평가 결과)를 추적하고 버전관리할 수 있습니다. 아티팩트를 생성할 때, 이름과 타입을 지정하며, 그 아티팩트는 영구적으로 실험 기록 시스템에 연결됩니다. 기본 데이터가 변경되고 그 데이터 자산을 다시 로깅하면, W&B는 내용의 체크섬을 통해 자동으로 새 버전을 생성합니다. W&B 아티팩트는 공유되는 비구조화된 파일 시스템 위에 있는 가벼운 추상화 계층으로 생각할 수 있습니다.

### 아티팩트의 구조 

`Artifact` 클래스는 W&B 아티팩트 레지스트리의 항목에 해당합니다. 아티팩트는
* 이름
* 타입
* 메타데이터
* 설명
* 파일, 파일 디렉토리, 또는 참조를 가집니다.

사용 예:
```python
run = wandb.init(project="my-project")
artifact = wandb.Artifact(name="my_artifact", type="data")
artifact.add_file("/path/to/my/file.txt")
run.log_artifact(artifact)
run.finish()
```

이 튜토리얼에서 우리가 할 첫 번째 일은 트레이닝 데이터셋을 다운로드하고 트레이닝 작업에서 사용할 아티팩트로 로깅하는 것입니다.

```python
# @title W&B 프로젝트와 엔티티 입력

# 폼 변수
PROJECT_NAME = "model-registry-tutorial"  # @param {type:"string"}
ENTITY = None  # @param {type:"string"}

# "TINY", "SMALL", "MEDIUM", 또는 "LARGE" 중 하나를 선택하여
# 이 세 데이터셋 중 하나를 선택하세요
# TINY 데이터셋: 100개 이미지, 30MB
# SMALL 데이터셋: 1000개 이미지, 312MB
# MEDIUM 데이터셋: 5000개 이미지, 1.5GB
# LARGE 데이터셋: 12,000개 이미지, 3.6GB

SIZE = "TINY"

if SIZE == "TINY":
    src_url = "https://storage.googleapis.com/wandb_datasets/nature_100.zip"
    src_zip = "nature_100.zip"
    DATA_SRC = "nature_100"
    IMAGES_PER_LABEL = 10
    BALANCED_SPLITS = {"train": 8, "val": 1, "test": 1}
elif SIZE == "SMALL":
    src_url = "https://storage.googleapis.com/wandb_datasets/nature_1K.zip"
    src_zip = "nature_1K.zip"
    DATA_SRC = "nature_1K"
    IMAGES_PER_LABEL = 100
    BALANCED_SPLITS = {"train": 80, "val": 10, "test": 10}
elif SIZE == "MEDIUM":
    src_url = "https://storage.googleapis.com/wandb_datasets/nature_12K.zip"
    src_zip = "nature_12K.zip"
    DATA_SRC = "inaturalist_12K/train"  # (기술적으로 10K 이미지만의 서브셋)
    IMAGES_PER_LABEL = 500
    BALANCED_SPLITS = {"train": 400, "val": 50, "test": 50}
elif SIZE == "LARGE":
    src_url = "https://storage.googleapis.com/wandb_datasets/nature_12K.zip"
    src_zip = "nature_12K.zip"
    DATA_SRC = "inaturalist_12K/train"  # (기술적으로 10K 이미지만의 서브셋)
    IMAGES_PER_LABEL = 1000
    BALANCED_SPLITS = {"train": 800, "val": 100, "test": 100}
```


```notebook
%%capture
!curl -SL $src_url > $src_zip
!unzip $src_zip
```


```python
import wandb
import pandas as pd
import os

with wandb.init(project=PROJECT_NAME, entity=ENTITY, job_type="log_datasets") as run:
    img_paths = []
    for root, dirs, files in os.walk("nature_100", topdown=False):
        for name in files:
            img_path = os.path.join(root, name)
            label = img_path.split("/")[1]
            img_paths.append([img_path, label])

    index_df = pd.DataFrame(columns=["image_path", "label"], data=img_paths)
    index_df.to_csv("index.csv", index=False)

    train_art = wandb.Artifact(
        name="Nature_100",
        type="raw_images",
        description="10개 클래스, 클래스 당 10개 이미지를 가진 자연 이미지 데이터셋",
    )
    train_art.add_dir("nature_100")

    # 각 이미지의 라벨을 나타내는 csv도 추가
    train_art.add_file("index.csv")
    wandb.log_artifact(train_art)
```

### 아티팩트 이름과 에일리어스를 사용하여 데이터 자산을 쉽게 인계하고 추상화
- 데이터셋이나 모델의 `name:alias` 조합을 간단히 참조함으로써 워크플로우의 구성 요소를 표준화할 수 있습니다.
- 예를 들어, W&B 아티팩트 이름과 에일리어스를 인수로 받아 적절하게 로드하는 PyTorch `Dataset`이나 `DataModule`을 구축할 수 있습니다.

이제 이 데이터셋과 연관된 모든 메타데이터, 이를 사용하는 W&B 런, 그리고 상위 및 하위 아티팩트 전체 계보를 볼 수 있습니다!

![api_token](https://drive.google.com/uc?export=view&id=1fEEddXMkabgcgusja0g8zMz8whlP2Y5P) 

```python
from torchvision import transforms
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from skimage import io, transform
from torchvision import transforms, utils, models
import math


class NatureDataset(Dataset):
    def __init__(
        self,
        wandb_run,
        artifact_name_alias="Nature_100:latest",
        local_target_dir="Nature_100:latest",
        transform=None,
    ):
        self.local_target_dir = local_target_dir
        self.transform = transform

        # 로컬로 아티팩트를 다운로드하여 메모리에 로드
        art = wandb_run.use_artifact(artifact_name_alias)
        path_at = art.download(root=self.local_target_dir)

        self.ref_df = pd.read_csv(os.path.join(self.local_target_dir, "index.csv"))
        self.class_names = self.ref_df.iloc[:, 1].unique().tolist()
        self.idx_to_class = {k: v for k, v in enumerate(self.class_names)}
        self.class_to_idx = {v: k for k, v in enumerate(self.class_names)}

    def __len__(self):
        return len(self.ref_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.ref_df.iloc[idx, 0]

        image = io.imread(img_path)
        label = self.ref_df.iloc[idx, 1]
        label = torch.tensor(self.class_to_idx[label], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label


class NatureDatasetModule(pl.LightningDataModule):
    def __init__(
        self,
        wandb_run,
        artifact_name_alias: str = "Nature_100:latest",
        local_target_dir: str = "Nature_100:latest",
        batch_size: int = 16,
        input_size: int = 224,
        seed: int = 42,
    ):
        super().__init__()
        self.wandb_run = wandb_run
        self.artifact_name_alias = artifact_name_alias
        self.local_target_dir = local_target_dir
        self.batch_size = batch_size
        self.input_size = input_size
        self.seed = seed

    def setup(self, stage=None):
        self.nature_dataset = NatureDataset(
            wandb_run=self.wandb_run,
            artifact_name_alias=self.artifact_name_alias,
            local_target_dir=self.local_target_dir,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.CenterCrop(self.input_size),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            ),
        )

        nature_length = len(self.nature_dataset)
        train_size = math.floor(0.8 * nature_length)
        val_size = math.floor(0.2 * nature_length)
        self.nature_train, self.nature_val = random_split(
            self.nature_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed),
        )
        return self

    def train_dataloader(self):
        return DataLoader(self.nature_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.nature_val, batch_size=self.batch_size)

    def predict_dataloader(self):
        pass

    def teardown(self, stage: str):
        pass
```

## 모델 트레이닝

### 모델 클래스 및 검증 함수 작성 

```python
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import onnx


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # 이 if 문에서 설정될 변수들을 초기화합니다. 각각의
    # 변수는 모델별로 특정됩니다.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """Resnet18"""
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """Alexnet"""
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """VGG11_bn"""
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """Squeezenet"""
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = torch.nn.Conv2d(
            512, num_classes, kernel_size=(1, 1), stride=(1, 1)
        )
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """Densenet"""
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = torch.nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


class NaturePyTorchModule(torch.nn.Module):
    def __init__(self, model_name, num_classes=10, feature_extract=True, lr=0.01):
        """모델 매개변수를 정의하는 데 사용되는 메소드"""
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.feature_extract = feature_extract
        self.lr = lr
        self.model, self.input_size = initialize_model(
            model_name=self.model_name,
            num_classes=self.num_classes,
            feature_extract=True,
        )

    def forward(self, x):
        """추론 입력 -> 출력을 위해 사용되는 메소드"""
        x = self.model(x)

        return x


def evaluate_model(model, eval_data, idx_to_class, class_names, epoch_ndx):
    device = torch.device("cpu")
    model.eval()
    test_loss = 0
    correct = 0
    preds = []
    actual = []

    val_table = wandb.Table(columns=["pred", "actual", "image"])

    with torch.no_grad():
        for data, target in eval_data:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # 배치 손실을 합산
            pred = output.argmax(
                dim=1, keepdim=True
            )  # 최대 로그 확률의 인덱스를 가져옴
            preds += list(pred.flatten().tolist())
            actual += target.numpy().tolist()
            correct += pred.eq(target.view_as(pred)).sum().item()

            for idx, img in enumerate(data):
                img = img.numpy().transpose(1, 2, 0)
                pred_class = idx_to_class[pred.numpy()[idx][0]]
                target_class = idx_to_class[target.numpy()[idx]]
                val_table.add_data(pred_class, target_class, wandb.Image(img))

    test_loss /= len(eval_data.dataset)
    accuracy = 100.0 * correct / len(eval_data.dataset)
    conf_mat = wandb.plot.confusion_matrix(
        y_true=actual, preds=preds, class_names=class_names
    )
    return test_loss, accuracy, preds, val_table, conf_mat
```

### 트레이닝 루프 추적
트레이닝 중에는 모델을 시간이 지남에 따라 체크포인트로 저장하는

### "링킹"이란 무엇인가요?
레지스트리에 링크할 때, 그 등록된 모델의 새로운 버전을 생성하는데, 이는 프로젝트에 있는 아티팩트 버전을 가리키는 포인터일 뿐입니다. W&B가 프로젝트 내의 아티팩트 버전 관리와 등록된 모델의 버전 관리를 분리하는 이유가 있습니다. 모델 아티팩트 버전을 링킹하는 프로세스는 등록된 모델 작업 아래에서 그 아티팩트 버전을 "북마킹"하는 것과 동등합니다.

일반적으로 연구/실험 과정에서 연구자들은 수백, 아니 수천 개의 모델 체크포인트 아티팩트를 생성하지만, 실제로 "빛을 보는" 것은 하나 또는 두 개뿐입니다. 이러한 체크포인트들을 별도의 버전 관리된 레지스트리에 링킹하는 프로세스는 모델 개발 측면과 모델 배포/사용 측면의 워크플로우를 구분하는 데 도움이 됩니다. 모델의 전 세계적으로 이해되는 버전/에일리어스는 연구 및 개발에서 생성되는 모든 실험적 버전들로부터 오염되지 않아야 하며, 따라서 등록된 모델의 버전은 모델 체크포인트 로깅과 달리 새로운 "북마크된" 모델에 따라 증가해야 합니다.

## 모든 모델을 위한 중앙 집중식 허브 생성
- 등록된 모델에 모델 카드, 태그, 슬랙 알림 추가
- 모델이 다양한 단계를 거치면서 에일리어스 변경
- 모델 문서화 및 리그레션 리포트를 위해 모델 레지스트리를 리포트에 포함시킵니다. 이 리포트를 [예시](https://api.wandb.ai/links/wandb-smle/r82bj9at)로 참조하세요.
![model registry](https://drive.google.com/uc?export=view&id=1lKPgaw-Ak4WK_91aBMcLvUMJL6pDQpgO)

### 새로운 모델이 레지스트리에 링크될 때 슬랙 알림 설정

![model registry](https://drive.google.com/uc?export=view&id=1RsWCa6maJYD5y34gQ0nwWiKSWUCqcjT9)

## 등록된 모델 사용
이제 `name:alias`를 참조하여 API를 통해 등록된 모델을 사용할 수 있습니다. 모델 사용자는 엔지니어, 연구자 또는 CI/CD 프로세스이며, 테스트를 거치거나 프로덕션으로 이동해야 하는 모델들이 "빛을 보아야 하는" 모든 모델을 위한 중앙 허브로 모델 레지스트리로 이동할 수 있습니다.

```notebook
%%wandb -h 600

run = wandb.init(project=PROJECT_NAME, entity=ENTITY, job_type='inference')
artifact = run.use_artifact(f'{ENTITY}/model-registry/Model Registry Tutorial:staging', type='model')
artifact_dir = artifact.download()
wandb.finish()
```

# 다음 단계는?
다음 튜토리얼에서는 대규모 언어 모델을 반복하고 W&B 프롬프트를 사용하여 디버깅하는 방법을 배우게 됩니다:

## 👉 [LLMs 반복](prompts)