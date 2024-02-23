
# 모델 등록하기

[**Colab 노트북에서 시도해보기 →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-model-registry/Model_Registry_E2E.ipynb)

모델 레지스트리는 기관 전반에 걸쳐 작업 중인 모든 모델 작업과 관련 아티팩트를 집중적으로 저장하고 조직하는 중앙 장소입니다:
- 모델 체크포인트 관리
- 풍부한 모델 카드로 모델 문서화
- 사용/배포되는 모든 모델의 기록 유지
- 모델의 깨끗한 인계 및 단계 관리 용이
- 다양한 모델 작업 태그 및 조직
- 모델 진행 시 자동 알림 설정

이 튜토리얼은 간단한 이미지 분류 작업에 대한 모델 개발 라이프사이클을 추적하는 방법을 안내합니다.

### 🛠️ `wandb` 설치하기
 
```bash
!pip install -q wandb onnx pytorch-lightning
```

## W&B 로그인하기
- `wandb login` 또는 `wandb.login()`을 사용하여 명시적으로 로그인할 수 있습니다(아래 참조)
- 또는 환경 변수를 설정할 수 있습니다. W&B 로깅의 동작을 변경할 수 있는 여러 환경 변수가 있습니다. 가장 중요한 변수는 다음과 같습니다:
    - `WANDB_API_KEY` - 프로필 아래 "설정" 섹션에서 찾을 수 있습니다
    - `WANDB_BASE_URL` - W&B 서버의 URL입니다
- W&B 앱에서 "프로필" -> "설정"에서 API 토큰 찾기

![api_token](https://drive.google.com/uc?export=view&id=1Xn7hnn0rfPu_EW0A_-32oCXqDmpA0-kx) 

```notebook
!wandb login
```

:::note
**데디케이티드 클라우드** 또는 **자체 관리**인 [W&B 서버](..//guides/hosting/intro.md) 배포에 연결할 때는 --relogin 및 --host 옵션을 사용하세요:

```notebook
!wandb login --relogin --host=http://your-shared-local-host.com
```

필요한 경우 배포 관리자에게 호스트 이름을 문의하세요.
:::

## 데이터와 모델 체크포인트를 아티팩트로 로깅하기
W&B 아티팩트를 사용하면 임의의 직렬화된 데이터(예: 데이터세트, 모델 체크포인트, 평가 결과)를 추적하고 버전 관리할 수 있습니다. 아티팩트를 생성할 때 이름과 유형을 지정하고, 해당 아티팩트는 영원히 실험 기록 시스템에 연결됩니다. 기본 데이터가 변경되고 해당 데이터 자산을 다시 로깅하면, W&B는 내용의 체크섬을 통해 자동으로 새 버전을 생성합니다. W&B 아티팩트는 공유되지 않은 구조화된 파일 시스템 위에 있는 가벼운 추상화 계층으로 생각할 수 있습니다.

### 아티팩트의 해부학

`Artifact` 클래스는 W&B 아티팩트 레지스트리의 항목에 해당합니다. 아티팩트는
* 이름
* 유형
* 메타데이터
* 설명
* 파일, 파일 디렉터리, 또는 참조를 가집니다

예제 사용법:
```python
run = wandb.init(project="my-project")
artifact = wandb.Artifact(name="my_artifact", type="data")
artifact.add_file("/path/to/my/file.txt")
run.log_artifact(artifact)
run.finish()
```

이 튜토리얼에서는 첫 번째로 학습 데이터세트를 다운로드하고 학습 작업에서 하류로 사용할 아티팩트로 로깅합니다.

```python
# @title W&B 프로젝트와 엔티티 입력

# 폼 변수
PROJECT_NAME = "model-registry-tutorial"  # @param {type:"string"}
ENTITY = None  # @param {type:"string"}

# 다음 세 가지 데이터세트 중 하나를 선택하려면 SIZE를 "TINY", "SMALL", "MEDIUM", 또는 "LARGE"로 설정하세요
# TINY 데이터세트: 100개의 이미지, 30MB
# SMALL 데이터세트: 1000개의 이미지, 312MB
# MEDIUM 데이터세트: 5000개의 이미지, 1.5GB
# LARGE 데이터세트: 12,000개의 이미지, 3.6GB

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
    DATA_SRC = "inaturalist_12K/train"  # (실제로는 10K 이미지만 포함된 서브세트)
    IMAGES_PER_LABEL = 500
    BALANCED_SPLITS = {"train": 400, "val": 50, "test": 50}
elif SIZE == "LARGE":
    src_url = "https://storage.googleapis.com/wandb_datasets/nature_12K.zip"
    src_zip = "nature_12K.zip"
    DATA_SRC = "inaturalist_12K/train"  # (실제로는 10K 이미지만 포함된 서브세트)
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
        description="10 클래스당 10개 이미지를 포함하는 자연 이미지 데이터세트",
    )
    train_art.add_dir("nature_100")

    # 각 이미지의 레이블을 나타내는 csv도 추가
    train_art.add_file("index.csv")
    wandb.log_artifact(train_art)
```

### 아티팩트 이름과 별칭을 사용하여 데이터 자산을 쉽게 인계하고 추상화하기
- 데이터세트나 모델의 `이름:별칭` 조합을 단순히 참조함으로써 워크플로의 구성 요소를 더 표준화할 수 있습니다
- 예를 들어, W&B 아티팩트 이름과 별칭을 인수로 받아 적절히 로드하는 PyTorch `Dataset` 또는 `DataModule`을 구축할 수 있습니다

이제 이 데이터세트와 이를 사용하는 W&B 실행, 그리고 상류 및 하류 아티팩트의 전체 계보와 관련된 모든 메타데이터를 확인할 수 있습니다!

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

        # 아티팩트를 로컬로 가져와 메모리에 로드
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

## 모델 학습

### 모델 클래스와 검증 함수 작성하기

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
    # 이 if 문에서 설정될 변수들을 초기화합니다. 각 변수는 모델별로 특화되어 있습니다.
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
        print("잘못된 모델 이름, 종료...")
        exit()

    return model_ft, input_size


class NaturePyTorchModule(torch.nn.Module):
    def __init__(self, model_name, num_classes=10, feature_extract=True, lr=0.01):
        """모델 파라미터를 정의하는 메서드"""
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
        """추론을 위한 메서드 input -> output"""
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
            ).item()  # 배치 손실 합계
            pred = output.argmax(
                dim=1, keepdim=True
            )  # 최대 로그-확률의 인덱스를 얻음
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

### 학습 루프 추적하기
학습 중에는 모델을 시간에 따라 체크포인트하는 것이 모범 사례입니다. 따라서 학습이 중단되거나 인스턴스가 충돌하는 경우 중단된 지점부터 다시 시작할 수 있습니다. 아티팩트 로깅을 사용하면 W&B에서 모든 체크포인트를 추적하고 원하는 메타데이터(직렬화 형식, 클래스 레이블 등)를 첨부할 수 있습니다. 그러면 체크포인트를 사용해야 하는 사람이 사용 방법을 알 수 있습니다. 어떤 형태의 모델을 아티팩트로 로깅할 때는 아티팩트의 `type`을 `model`로 설정해야 합니다.

```python
run = wandb.init(
    project=PROJECT_NAME,
    entity=ENTITY,
    job_type="training",
    config={
        "model_type": "squeezenet",
        "lr": 1.0,
        "gamma": 0.75,
        "batch_size": 16,
        "epochs": 5,
    },
)

model = NaturePyTorchModule(wandb.config["model_type"])
wandb.watch(model)

wandb.config["input_size"] = 224

nature_module = NatureDatasetModule(
    wandb_run=run,
    artifact_name_alias="Nature_100:latest",
    local_target_dir="Nature_100:latest",
    batch_size=wandb.config["batch_size"],
    input_size=wandb.config["input_size"],
)
nature_module.setup()

# 모델 학습
learning_rate = wandb.config["lr"]
gamma = wandb.config["gamma"]
epochs = wandb.config["epochs"]

device = torch.device("cpu")
optimizer = optim.Adadelta(model.parameters(), lr=wandb.config["lr"])
scheduler = StepLR(optimizer, step_size=1, gamma=wandb.config["gamma"])

best_loss = float("inf")
best_model = None

for epoch_ndx in range(epochs):
    model.train()
    for batch_ndx, batch in enumerate(nature_module.train_dataloader()):
        data, target = batch[0].to("cpu"), batch[1].to("cpu")
        optimizer.zero_grad()
        preds = model(data)
        loss = F.nll_loss(preds, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        ### 메트릭 로깅 ###
        wandb.log(
            {
                "train/epoch_ndx": epoch_ndx,
                "train/batch_ndx": batch_ndx,
                "train/train_loss": loss,
                "train/learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

    ### 각 에포크의 끝에서 평가 ###
    model.eval()
    test_loss, accuracy, preds, val_table, conf_mat = evaluate_model(
        model,
        nature_module.val_dataloader(),
        nature_module.nature_dataset.idx_to_class,
        nature_module.nature_dataset.class_names,
        epoch_ndx,
    )

    is_best = test_loss < best_loss

    wandb.log(
        {
            "eval/test_loss": test_loss,
            "eval/accuracy": accuracy,
            "eval/conf_mat": conf_mat,
            "eval/val_table": val_table,
        }
    )

    ### 모델 가중치 체크포인트 ###
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    torch.onnx.export(
        model,  # 실행되는 모델
        x,  # 모델 입력 (또는 여러 입력의 경우 튜플)
        "model.onnx",  # 모델을 저장할 위치 (파일 또는 파일과 유사한 객체)
        export_params=True,  # 모델 파일 내에 훈련된 파라미터 가중치를 저장
        opset_version=10,  # 모델을 내보낼 ONNX 버전
        do_constant_folding=True,  # 최적화를 위해 상수 폴딩을 실행할지 여부
        input_names=["input"],  # 모델의 입력 이름
        output_names=["output"],  # 모델의 출력 이름
        dynamic_axes={
            "input": {0: "batch_size"},  # 가변 길이 축
            "output": {0: "batch_size"},
        },
    )

    art = wandb.Artifact(
        f"nature-{wandb.run.id}",
        type="model",
        metadata={
            "format": "onnx",
            "num_classes": len(nature_module.nature_dataset.class_names),
            "model_type": wandb.config["model_type"],
            "model_input_size": wandb.config["input_size"],
            "index_to_class": nature_module.nature_dataset.idx_to_class,
        },
    )

    art.add_file("model.onnx")

    ### 시간이 지남에 따라 최고의 체크포인트를 추적하기 위해 별칭 추가
    wandb.log_artifact(art, aliases=["best", "latest"] if is_best else None)
    if is_best:
        best_model = art
```

### 프로젝트의 모든 모델 체크포인트를 한 곳에서 관리하세요.

![api_token](https://drive.google.com/uc?export=view&id=1z7nXRgqHTPYjfR1SoP-CkezyxklbAZlM)

### 참고: W&B 오프라인 동기화
학습 과정 중 어떤 이유로든 네트워크 통신이 끊긴 경우, 언제든지 `wandb sync`를 사용하여 진행 상황을 동기화할 수 있습니다.

W&B sdk는 모든 로그된 데이터를 로컬 `wandb` 디렉터리에 캐시하고 `wandb sync`를 호출하면 로컬 상태를 웹 앱과 동기화합니다.

## 모델 레지스트리
실험 중 여러 실행에서 많은 체크포인트를 로깅한 후, 이제 워크플로의 다음 단계(예: 테스트, 배포)로 최고의 체크포인트를 넘겨줄 시간입니다.

모델 레지스트리는 개별 W&B 프로젝트 위에 존재하는 중앙 페이지입니다. **등록된 모델**을 보유한 포트폴리오는 개별 W&B 프로젝트에서 살아 있는 가치 있는 체크포인트에 대한 "링크"를 저장합니다.

모델 레지스트리는 모든 모델 작업에 대한 최고의 체크포인트를 보관하는 중앙 장소를 제공합니다. 로깅한 모든 `model` 아티팩트는 **등록된 모델**에 "링크"될 수 있습니다.

### UI를 통한 **등록된 모델** 생성 및 링크

#### 1. 팀 페이지로 이동하여 `Model Registry`를 선택하여 팀의 모델 레지스트리에 엑세스하세요.

![model registry](https://drive.google.com/uc?export=view&id=1ZtJwBsFWPTm4Sg5w8vHhRpvDSeQPwsKw)

#### 2. 새로운 등록된 모델을 생성하세요.

![model registry](https://drive.google.com/uc?export=view&id=1RuayTZHNE0LJCxt1t0l6-2zjwiV4aDXe)

#### 3. 모든 모델 체크포인트를 보유한 프로젝트의 아티팩트 탭으로 이동하세요.

![model registry](https://drive.google.com/uc?export=view&id=1LfTLrRNpBBPaUb_RmBIE7fWFMG0h3e0E)

#### 4. 원하는 모델 아티팩트 버전에 대해 "Link to Registry"를 클릭하세요.

### **API**를 통한 등록된 모델 생성 및 링크
`wandb.run.link_artifact`를 호출하고 아티팩트 개체와 **등록된 모델**의 이름을 전달하여 [모델을 API를 통해 링크](https://docs.wandb.ai/guides/models)할 수 있습니다. 여기에는 추가하고자 하는 별칭도 포함됩니다. **등록된 모델**은 W&B에서 엔티티(팀) 범위이므로 팀의 멤버만 해당 엔티티의 **등록된 모델**을 볼 수 있고 엑세스할 수 있습니다. api를 통해 등록된 모델 이름을 `<entity>/model-registry/<registered-model-name>`으로 표시합니다. 등록된 모델이 없는 경우 자동으로 생성됩니다.

```python
if ENTITY:
    wandb.run.link_artifact(
        best_model,
        f"{ENTITY}/model-registry/Model Registry Tutorial",
        aliases=["staging"],
    )
else:
    print("Must indicate entity where Registered Model will exist")
wandb.finish()
```

### "링크"란 무엇인가요?
레지스트리에 링크하면 해당 프로젝트에서 살아 있는 아티팩트 버전을 가리키는 등록된 모델의 새 버전이 생성됩니다. W&B가 프로젝트 내의 아티팩트 버전 관리와 등록된 모델의 버전 관리를 분리하는 이유가 있습니다. 모델 아티팩트 버전을 링크하는 프로세스는 등록된 모델 작업 아래에서 해당 아티팩트 버전을 "북마킹"하는 것과 같습니다.

일반적으로 R&D/실험 중에 연구자들은 수백, 아니면 수천 개의 모델 체크포인트 아티팩트를 생성하지만, 실제로 "빛을 보는" 것은 한두 개에 불과합니다. 이러한 체크포인트를 별도의 버전 관리 레지스트리에 링크하는 과정은 모델 개발 측면과 모델 배포/소비 측면의 워크플로를 구분하는 데 도움이 됩니다. 모델의 전 세계적으로 이해되는 버전/별칭은 R&D에서 생성되는 모든 실험적 버전으로부터 오염되지 않아야 하므로 등록된 모델의 버전은 모델 체크포인트 로깅 대신 새로 "북마크된" 모델에 따라 증가해야 합니다.

## 모든 모델을 위한 중앙 집중식 허브 생성
- 등록된 모델에 모델 카드, 태그, 슬랙 알림 추가
- 모델이 다른 단계를 거치면서 별칭 변경
- 모델 문서와 회귀 리포트를 위해 모델 레지스트리를 리포트에 포함시키세요. 이 [예시](https://api.wandb.ai/links/wandb-smle/r82bj9at) 리포트를 확인하세요.
![model registry](https://drive.google.com/uc?export=view&id=1lKPgaw-Ak4WK_91aBMcLvUMJL6pDQpgO)

### 레지스트리에 새 모델이 링크될 때 슬랙 알림 설정

![model registry](https://drive.google.com/uc?export=view&id=1RsWCa6maJYD5y34gQ0nwWiKSWUCqcjT9)

## 등록된 모델 사용
이제 해당 `name:alias`를 참조하여 API를 통해 등록된 모델을 사용할 수 있습니다. 모델 소비자, 엔지니어, 연구자 또는 CI/CD 프로세스가 되었든, 모델 레지스트리는 테스트를 거치거나 프로덕션으로 이동해야 하는 모든 모델의 중앙 허브로 사용될 수 있습니다.

```notebook
%%wandb -h 600

run = wandb.init(project=PROJECT_NAME, entity=ENTITY, job_type='inference')
artifact = run.use_artifact(f'{ENTITY}/model-registry/Model Registry Tutorial:staging', type='model')
artifact_dir = artifact.download()
wandb.finish()
```

# 다음 단계는?
다음 튜토리얼에서는 W&B Prompts를 사용하여 대규모 언어 모델을 반복하고 디버깅하는 방법을 배우게 됩니다:

## 👉 [LLMs 반복하기](prompts)