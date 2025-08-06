---
title: MONAI를 활용한 3D 뇌종양 분할
menu:
  tutorials:
    identifier: ko-tutorials-integration-tutorials-monai_3d_segmentation
    parent: integration-tutorials
weight: 10
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/monai/3d_brain_tumor_segmentation.ipynb" >}}

이 튜토리얼에서는 [MONAI](https://github.com/Project-MONAI/MONAI)를 활용하여 다중 레이블 3D 뇌종양 세그멘테이션 트레이닝 워크플로우를 구성하고, [W&B](https://wandb.ai/site)의 실험 추적 및 데이터 시각화 기능을 사용하는 방법을 살펴봅니다. 이 튜토리얼에서는 다음과 같은 기능을 다룹니다.

1. W&B Run을 초기화하고 실험과 연관된 모든 config 정보를 동기화하여 재현성을 높입니다.
2. MONAI transform API:
    1. dictionary 형식 데이터에 대한 MONAI Transform 활용.
    2. MONAI `transforms` API에 따라 새로운 transform 정의하는 방법.
    3. 데이터 증강을 위한 intensity 랜덤 조정 방식.
3. 데이터 로딩 및 시각화:
    1. `Nifti` 이미지 및 메타데이터 불러오기, 이미지 리스트 로딩 및 스택 쌓기.
    2. IO 및 transform 캐싱을 통해 학습 및 검증 속도 가속화.
    3. `wandb.Table`과 W&B에서 인터랙티브 세그멘테이션 오버레이를 사용하여 데이터 시각화.
4. 3D `SegResNet` 모델 트레이닝
    1. MONAI의 `networks`, `losses`, `metrics` API 활용.
    2. PyTorch 트레이닝 루프를 통해 3D `SegResNet` 모델 학습.
    3. W&B를 사용한 트레이닝 실험 추적.
    4. 모델 체크포인트를 W&B model artifact로 기록 및 버전 관리.
5. `wandb.Table`과 W&B 인터랙티브 세그멘테이션 오버레이로 검증 데이터셋의 예측값 시각화 및 비교.

## 환경 설정 및 설치

먼저, MONAI와 W&B의 최신 버전을 설치합니다.

```python
!python -c "import monai" || pip install -q -U "monai[nibabel, tqdm]"
!python -c "import wandb" || pip install -q -U wandb
```

```python
import os

import numpy as np
from tqdm.auto import tqdm
import wandb

from monai.apps import DecathlonDataset
from monai.data import DataLoader, decollate_batch
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
)
from monai.utils import set_determinism

import torch
```

그리고 나서, Colab 인스턴스에서 W&B를 사용할 수 있도록 인증을 진행합니다.

```python
wandb.login()
```

## W&B Run 초기화

새로운 W&B Run을 시작하여 실험을 추적하세요. 올바른 config 시스템 사용은 재현성 있는 기계학습 실험을 위한 권장 모범 사례입니다. 모든 실험에서 하이퍼파라미터를 W&B로 트래킹할 수 있습니다.

```python
with wandb.init(project="monai-brain-tumor-segmentation") as run:

    config = run.config
    config.seed = 0
    config.roi_size = [224, 224, 144]
    config.batch_size = 1
    config.num_workers = 4
    config.max_train_images_visualized = 20
    config.max_val_images_visualized = 20
    config.dice_loss_smoothen_numerator = 0
    config.dice_loss_smoothen_denominator = 1e-5
    config.dice_loss_squared_prediction = True
    config.dice_loss_target_onehot = False
    config.dice_loss_apply_sigmoid = True
    config.initial_learning_rate = 1e-4
    config.weight_decay = 1e-5
    config.max_train_epochs = 50
    config.validation_intervals = 1
    config.dataset_dir = "./dataset/"
    config.checkpoint_dir = "./checkpoints"
    config.inference_roi_size = (128, 128, 64)
    config.max_prediction_images_visualized = 20
```

재현성 있는 학습을 위해 랜덤 seed도 세팅해줍니다.

```python
set_determinism(seed=config.seed)

# 디렉터리 생성
os.makedirs(config.dataset_dir, exist_ok=True)
os.makedirs(config.checkpoint_dir, exist_ok=True)
```

## 데이터 로딩 및 변환

여기서는 `monai.transforms` API를 활용해 다중 클래스 라벨을 one-hot 형태의 다중 레이블 세그멘테이션 태스크로 변환하는 커스텀 transform을 생성합니다.

```python
class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    brats 클래스에 따라 라벨을 멀티 채널로 변환합니다:
    라벨 1은 peritumoral edema
    라벨 2는 GD-enhancing tumor
    라벨 3은 necrotic 및 non-enhancing tumor core
    가능한 클래스: TC (Tumor core), WT (Whole tumor), ET (Enhancing tumor)

    참고: https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # TC를 만들기 위해 라벨 2와 3을 합칩니다
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # WT를 만들기 위해 라벨 1, 2, 3을 합칩니다
            result.append(
                torch.logical_or(
                    torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                )
            )
            # 라벨 2는 ET입니다
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d
```

다음으로, 트레이닝과 검증 데이터셋 각각에 대해 transform을 설정합니다.

```python
train_transform = Compose(
    [
        # Nifti 이미지 4개를 로드해서 스택으로 쌓음
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        RandSpatialCropd(
            keys=["image", "label"], roi_size=config.roi_size, random_size=False
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ]
)
val_transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
)
```

### 데이터셋

이 실험에 사용된 데이터셋은 http://medicaldecathlon.com/ 에서 제공됩니다. 이 데이터셋은 다중 모달/다중 기관의 MRI 데이터(FLAIR, T1w, T1gd, T2w)를 사용해 Gliomas, necrotic/active tumor, oedema를 분할합니다. 총 750개의 4D 볼륨(트레이닝 484개 + 테스트 266개)으로 구성되어 있습니다.

`DecathlonDataset`을 사용하면 데이터셋을 자동으로 다운로드 및 압축 해제할 수 있습니다. 이 클래스는 MONAI의 `CacheDataset`을 상속하며, `cache_num=N`으로 학습 데이터를 N개 캐시하거나 검증 데이터는 메모리 상황에 따라 모든 아이템을 캐시할 수 있습니다.

```python
train_dataset = DecathlonDataset(
    root_dir=config.dataset_dir,
    task="Task01_BrainTumour",
    transform=val_transform,
    section="training",
    download=True,
    cache_rate=0.0,
    num_workers=4,
)
val_dataset = DecathlonDataset(
    root_dir=config.dataset_dir,
    task="Task01_BrainTumour",
    transform=val_transform,
    section="validation",
    download=False,
    cache_rate=0.0,
    num_workers=4,
)
```

{{% alert %}}
**참고:** 학습 데이터셋에 `train_transform`을 바로 적용하는 대신, 트레이닝/검증 데이터셋 모두에 `val_transform`을 먼저 적용하세요. 이는 학습 이전에 데이터셋의 양 split에서 샘플 시각화를 진행하기 위함입니다.
{{% /alert %}}

### 데이터셋 시각화

W&B에서는 이미지, 비디오, 오디오 등 다양한 미디어 로그 및 시각화를 지원합니다. 여러분의 결과를 탐색하고 다양한 run, 모델, 데이터셋을 시각적으로 비교할 수 있습니다. [세그멘테이션 마스크 오버레이 시스템]({{< relref path="/guides/models/track/log/media/#image-overlays-in-tables" lang="ko" >}})을 활용하여 데이터 볼륨을 시각화할 수 있습니다. [테이블]({{< relref path="/guides/models/tables/" lang="ko" >}})에 세그멘테이션 마스크를 기록하려면 각 행에 대해 `wandb.Image` 오브젝트를 제공해야 합니다.

아래는 예시 코드입니다:

```python
table = wandb.Table(columns=["ID", "Image"])

for id, img, label in zip(ids, images, labels):
    mask_img = wandb.Image(
        img,
        masks={
            "prediction": {"mask_data": label, "class_labels": class_labels}
            # ...
        },
    )

    table.add_data(id, img)

run.log({"Table": table})
```

다음은 샘플 이미지, 라벨, `wandb.Table` 오브젝트와 메타데이터를 받아 W&B 대시보드에 업로드할 테이블의 행을 채우는 유틸리티 함수 예시입니다.

```python
def log_data_samples_into_tables(
    sample_image: np.array,
    sample_label: np.array,
    split: str = None,
    data_idx: int = None,
    table: wandb.Table = None,
):
    num_channels, _, _, num_slices = sample_image.shape
    with tqdm(total=num_slices, leave=False) as progress_bar:
        for slice_idx in range(num_slices):
            ground_truth_wandb_images = []
            for channel_idx in range(num_channels):
                ground_truth_wandb_images.append(
                    masks = {
                        "ground-truth/Tumor-Core": {
                            "mask_data": sample_label[0, :, :, slice_idx],
                            "class_labels": {0: "background", 1: "Tumor Core"},
                        },
                        "ground-truth/Whole-Tumor": {
                            "mask_data": sample_label[1, :, :, slice_idx] * 2,
                            "class_labels": {0: "background", 2: "Whole Tumor"},
                        },
                        "ground-truth/Enhancing-Tumor": {
                            "mask_data": sample_label[2, :, :, slice_idx] * 3,
                            "class_labels": {0: "background", 3: "Enhancing Tumor"},
                        },
                    }
                    wandb.Image(
                        sample_image[channel_idx, :, :, slice_idx],
                        masks=masks,
                    )
                )
            table.add_data(split, data_idx, slice_idx, *ground_truth_wandb_images)
            progress_bar.update(1)
    return table
```

다음으로, 어떤 column으로 구성된 테이블 오브젝트를 사용할지 정의합니다.

```python
table = wandb.Table(
    columns=[
        "Split",
        "Data Index",
        "Slice Index",
        "Image-Channel-0",
        "Image-Channel-1",
        "Image-Channel-2",
        "Image-Channel-3",
    ]
)
```

그다음, 각각 `train_dataset`과 `val_dataset`을 반복하여 데이터 샘플 시각화를 생성하고 테이블 행을 채워 대시보드에 로그합니다.

```python
# train_dataset 시각화 생성
max_samples = (
    min(config.max_train_images_visualized, len(train_dataset))
    if config.max_train_images_visualized > 0
    else len(train_dataset)
)
progress_bar = tqdm(
    enumerate(train_dataset[:max_samples]),
    total=max_samples,
    desc="Generating Train Dataset Visualizations:",
)
for data_idx, sample in progress_bar:
    sample_image = sample["image"].detach().cpu().numpy()
    sample_label = sample["label"].detach().cpu().numpy()
    table = log_data_samples_into_tables(
        sample_image,
        sample_label,
        split="train",
        data_idx=data_idx,
        table=table,
    )

# val_dataset 시각화 생성
max_samples = (
    min(config.max_val_images_visualized, len(val_dataset))
    if config.max_val_images_visualized > 0
    else len(val_dataset)
)
progress_bar = tqdm(
    enumerate(val_dataset[:max_samples]),
    total=max_samples,
    desc="Generating Validation Dataset Visualizations:",
)
for data_idx, sample in progress_bar:
    sample_image = sample["image"].detach().cpu().numpy()
    sample_label = sample["label"].detach().cpu().numpy()
    table = log_data_samples_into_tables(
        sample_image,
        sample_label,
        split="val",
        data_idx=data_idx,
        table=table,
    )

# 테이블을 대시보드에 로그
run.log({"Tumor-Segmentation-Data": table})
```

이 데이터는 W&B 대시보드에서 인터랙티브 테이블 형태로 확인할 수 있습니다. 각 row에서는 데이터 볼륨의 특정 채널/슬라이스에 세그멘테이션 마스크가 오버레이되는 모습을 확인할 수 있습니다. [Weave 쿼리]({{< relref path="/guides/weave" lang="ko" >}})를 작성하여 테이블 내 행을 필터링하여 한 row만 집중할 수도 있습니다.

| {{< img src="/images/tutorials/monai/viz-1.gif" alt="Logged table data" >}} | 
|:--:| 
| **로그된 테이블 데이터 예시** |

이미지를 열면, 각 세그멘테이션 마스크를 인터랙티브 오버레이로 직접 확인할 수 있습니다.

| {{< img src="/images/tutorials/monai/viz-2.gif" alt="Segmentation maps" >}} | 
|:--:| 
| **시각화된 세그멘테이션 맵 예시** |

{{% alert %}}
**참고:** 데이터셋의 라벨은 클래스 간 오버랩이 없는 마스크로 구성되어 있습니다. 오버레이는 각 라벨을 별도의 마스크로 기록합니다.
{{% /alert %}}

### 데이터 로딩

이제 PyTorch DataLoader를 사용하여 데이터셋에서 데이터를 불러옵니다. DataLoader를 생성하기 전, `train_dataset`의 `transform`을 `train_transform`으로 설정하여 학습 데이터 전처리와 변환을 적용합니다.

```python
# 학습 데이터셋에 train_transform 적용
train_dataset.transform = train_transform

# train_loader 생성
train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers,
)

# val_loader 생성
val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers,
)
```

## 모델, Loss, 옵티마이저 생성

이 튜토리얼은 [3D MRI brain tumor segmentation using auto-encoder regularization](https://arxiv.org/pdf/1810.11654.pdf) 논문을 기반으로 `SegResNet` 모델을 생성합니다. `SegResNet` 모델은 PyTorch Module로, `monai.networks` API 및 옵티마이저, 러닝레이트 스케줄러와 함께 구현되어 있습니다.

```python
device = torch.device("cuda:0")

# 모델 생성
model = SegResNet(
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=16,
    in_channels=4,
    out_channels=3,
    dropout_prob=0.2,
).to(device)

# 옵티마이저 생성
optimizer = torch.optim.Adam(
    model.parameters(),
    config.initial_learning_rate,
    weight_decay=config.weight_decay,
)

# 러닝레이트 스케줄러 생성
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=config.max_train_epochs
)
```

`monai.losses` API로 멀티레이블 `DiceLoss`를 정의하고, `monai.metrics` API에서 Dice metric을 정의합니다.

```python
loss_function = DiceLoss(
    smooth_nr=config.dice_loss_smoothen_numerator,
    smooth_dr=config.dice_loss_smoothen_denominator,
    squared_pred=config.dice_loss_squared_prediction,
    to_onehot_y=config.dice_loss_target_onehot,
    sigmoid=config.dice_loss_apply_sigmoid,
)

dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

# 자동 mixed-precision을 적용해 학습 가속
scaler = torch.cuda.amp.GradScaler()
torch.backends.cudnn.benchmark = True
```

검증 과정이나 학습 후 모델을 실행할 때 사용할 혼합정밀 추론용 작은 유틸리티 함수를 정의합니다.

```python
def inference(model, input):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(240, 240, 160),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    with torch.cuda.amp.autocast():
        return _compute(input)
```

## 트레이닝 & 검증

트레이닝 전, 나중에 `run.log()`로 기록할 메트릭 속성을 정의합니다. 학습 및 검증 실험을 추적하기 위함입니다.

```python
run.define_metric("epoch/epoch_step")
run.define_metric("epoch/*", step_metric="epoch/epoch_step")
run.define_metric("batch/batch_step")
run.define_metric("batch/*", step_metric="batch/batch_step")
run.define_metric("validation/validation_step")
run.define_metric("validation/*", step_metric="validation/validation_step")

batch_step = 0
validation_step = 0
metric_values = []
metric_values_tumor_core = []
metric_values_whole_tumor = []
metric_values_enhanced_tumor = []
```

### 표준 PyTorch 트레이닝 루프 실행

```python
with wandb.init(
    project="monai-brain-tumor-segmentation",
    config=config,
    job_type="train",
    reinit=True,
) as run:

    # W&B Artifact 오브젝트 정의
    artifact = wandb.Artifact(
        name=f"{run.id}-checkpoint", type="model"
    )

    epoch_progress_bar = tqdm(range(config.max_train_epochs), desc="Training:")

    for epoch in epoch_progress_bar:
        model.train()
        epoch_loss = 0

        total_batch_steps = len(train_dataset) // train_loader.batch_size
        batch_progress_bar = tqdm(train_loader, total=total_batch_steps, leave=False)
        
        # 트레이닝 스텝
        for batch_data in batch_progress_bar:
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            batch_progress_bar.set_description(f"train_loss: {loss.item():.4f}:")
            ## 배치별 트레이닝 loss를 W&B에 로그
            run.log({"batch/batch_step": batch_step, "batch/train_loss": loss.item()})
            batch_step += 1

        lr_scheduler.step()
        epoch_loss /= total_batch_steps
        ## 에폭별 트레이닝 loss 및 러닝레이트를 W&B에 로그
        run.log(
            {
                "epoch/epoch_step": epoch,
                "epoch/mean_train_loss": epoch_loss,
                "epoch/learning_rate": lr_scheduler.get_last_lr()[0],
            }
        )
        epoch_progress_bar.set_description(f"Training: train_loss: {epoch_loss:.4f}:")

        # 검증 및 모델 체크포인트 스텝
        if (epoch + 1) % config.validation_intervals == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    val_outputs = inference(model, val_inputs)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    dice_metric_batch(y_pred=val_outputs, y=val_labels)

                metric_values.append(dice_metric.aggregate().item())
                metric_batch = dice_metric_batch.aggregate()
                metric_values_tumor_core.append(metric_batch[0].item())
                metric_values_whole_tumor.append(metric_batch[1].item())
                metric_values_enhanced_tumor.append(metric_batch[2].item())
                dice_metric.reset()
                dice_metric_batch.reset()

                checkpoint_path = os.path.join(config.checkpoint_dir, "model.pth")
                torch.save(model.state_dict(), checkpoint_path)
                
                # 모델 체크포인트를 W&B artifact로 기록 및 버전 관리
                artifact.add_file(local_path=checkpoint_path)
                run.log_artifact(artifact, aliases=[f"epoch_{epoch}"])

                # 검증 메트릭을 W&B 대시보드에 로그
                run.log(
                    {
                        "validation/validation_step": validation_step,
                        "validation/mean_dice": metric_values[-1],
                        "validation/mean_dice_tumor_core": metric_values_tumor_core[-1],
                        "validation/mean_dice_whole_tumor": metric_values_whole_tumor[-1],
                        "validation/mean_dice_enhanced_tumor": metric_values_enhanced_tumor[-1],
                    }
                )
                validation_step += 1


    # 이 artifact의 로그가 끝날 때까지 대기
    artifact.wait()
```

코드에서 `wandb.log`를 사용하면, 트레이닝과 검증 과정 중 발생하는 모든 메트릭뿐만 아니라 시스템 메트릭(CPU, GPU 등)도 W&B 대시보드에서 함께 추적할 수 있습니다.

| {{< img src="/images/tutorials/monai/viz-3.gif" alt="Training and validation tracking" >}} | 
|:--:| 
| **W&B에서 트레이닝 및 검증 과정 추적 예시** |

W&B Run 대시보드의 artifacts 탭에서 트레이닝 중 기록된 여러 모델 체크포인트 artifact 버전에 엑세스할 수 있습니다.

| {{< img src="/images/tutorials/monai/viz-4.gif" alt="Model checkpoints logging" >}} | 
|:--:| 
| **W&B에서 모델 체크포인트 기록 및 버전 관리 예시** |

## 추론(Inference)

artifacts 인터페이스를 활용하여, artifact 버전 중 성능이 가장 좋은(이 예제에서는 에폭별 평균 트레이닝 로스가 최적) 모델 체크포인트를 선택할 수 있습니다. 또한 artifact의 전체 계보를 확인하며 원하는 버전을 직접 사용할 수 있습니다.

| {{< img src="/images/tutorials/monai/viz-5.gif" alt="Model artifact tracking" >}} | 
|:--:| 
| **W&B에서 모델 artifact 추적 예시** |

에폭별 평균 트레이닝 loss가 가장 낮은 모델 artifact 버전을 가져와 체크포인트 state dictionary를 모델에 로드합니다.

```python
run = wandb.init(
    project="monai-brain-tumor-segmentation",
    job_type="inference",
    reinit=True,
)
model_artifact = run.use_artifact(
    "geekyrakshit/monai-brain-tumor-segmentation/d5ex6n4a-checkpoint:v49",
    type="model",
)
model_artifact_dir = model_artifact.download()
model.load_state_dict(torch.load(os.path.join(model_artifact_dir, "model.pth")))
model.eval()
```

### 예측값 시각화 및 그라운드 트루스 레이블과 비교

사전학습된 모델의 예측과 실제 세그멘테이션 마스크(그라운드 트루스)를 인터랙티브하게 비교 시각화하는 함수 예시입니다.

```python
def log_predictions_into_tables(
    sample_image: np.array,
    sample_label: np.array,
    predicted_label: np.array,
    split: str = None,
    data_idx: int = None,
    table: wandb.Table = None,
):
    num_channels, _, _, num_slices = sample_image.shape
    with tqdm(total=num_slices, leave=False) as progress_bar:
        for slice_idx in range(num_slices):
            wandb_images = []
            for channel_idx in range(num_channels):
                wandb_images += [
                    wandb.Image(
                        sample_image[channel_idx, :, :, slice_idx],
                        masks={
                            "ground-truth/Tumor-Core": {
                                "mask_data": sample_label[0, :, :, slice_idx],
                                "class_labels": {0: "background", 1: "Tumor Core"},
                            },
                            "prediction/Tumor-Core": {
                                "mask_data": predicted_label[0, :, :, slice_idx] * 2,
                                "class_labels": {0: "background", 2: "Tumor Core"},
                            },
                        },
                    ),
                    wandb.Image(
                        sample_image[channel_idx, :, :, slice_idx],
                        masks={
                            "ground-truth/Whole-Tumor": {
                                "mask_data": sample_label[1, :, :, slice_idx],
                                "class_labels": {0: "background", 1: "Whole Tumor"},
                            },
                            "prediction/Whole-Tumor": {
                                "mask_data": predicted_label[1, :, :, slice_idx] * 2,
                                "class_labels": {0: "background", 2: "Whole Tumor"},
                            },
                        },
                    ),
                    wandb.Image(
                        sample_image[channel_idx, :, :, slice_idx],
                        masks={
                            "ground-truth/Enhancing-Tumor": {
                                "mask_data": sample_label[2, :, :, slice_idx],
                                "class_labels": {0: "background", 1: "Enhancing Tumor"},
                            },
                            "prediction/Enhancing-Tumor": {
                                "mask_data": predicted_label[2, :, :, slice_idx] * 2,
                                "class_labels": {0: "background", 2: "Enhancing Tumor"},
                            },
                        },
                    ),
                ]
            table.add_data(split, data_idx, slice_idx, *wandb_images)
            progress_bar.update(1)
    return table
```

예측 결과를 prediction 테이블에 기록합니다.

```python
run = wandb.init(
    project="monai-brain-tumor-segmentation",
    job_type="inference",
    reinit=True,
)
# prediction 테이블 생성
prediction_table = wandb.Table(
    columns=[
        "Split",
        "Data Index",
        "Slice Index",
        "Image-Channel-0/Tumor-Core",
        "Image-Channel-1/Tumor-Core",
        "Image-Channel-2/Tumor-Core",
        "Image-Channel-3/Tumor-Core",
        "Image-Channel-0/Whole-Tumor",
        "Image-Channel-1/Whole-Tumor",
        "Image-Channel-2/Whole-Tumor",
        "Image-Channel-3/Whole-Tumor",
        "Image-Channel-0/Enhancing-Tumor",
        "Image-Channel-1/Enhancing-Tumor",
        "Image-Channel-2/Enhancing-Tumor",
        "Image-Channel-3/Enhancing-Tumor",
    ]
)

# inference 및 시각화 수행
with torch.no_grad():
    config.max_prediction_images_visualized
    max_samples = (
        min(config.max_prediction_images_visualized, len(val_dataset))
        if config.max_prediction_images_visualized > 0
        else len(val_dataset)
    )
    progress_bar = tqdm(
        enumerate(val_dataset[:max_samples]),
        total=max_samples,
        desc="Generating Predictions:",
    )
    for data_idx, sample in progress_bar:
        val_input = sample["image"].unsqueeze(0).to(device)
        val_output = inference(model, val_input)
        val_output = post_trans(val_output[0])
        prediction_table = log_predictions_into_tables(
            sample_image=sample["image"].cpu().numpy(),
            sample_label=sample["label"].cpu().numpy(),
            predicted_label=val_output.cpu().numpy(),
            data_idx=data_idx,
            split="validation",
            table=prediction_table,
        )

    run.log({"Predictions/Tumor-Segmentation-Data": prediction_table})


# 실험 종료
run.finish()
```

인터랙티브 세그멘테이션 마스크 오버레이를 사용해 클래스별 예측 마스크와 그라운드 트루스 레이블을 분석/비교할 수 있습니다.

| {{< img src="/images/tutorials/monai/viz-6.gif" alt="Predictions and ground-truth" >}} | 
|:--:| 
| **W&B에서 예측값과 그라운드 트루스 시각화 예시** |

## 참고 및 추가 자료

* [MONAI Tutorial: Brain tumor 3D segmentation with MONAI](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb)
* [WandB Report: Brain Tumor Segmentation using MONAI and WandB](https://wandb.ai/geekyrakshit/brain-tumor-segmentation/reports/Brain-Tumor-Segmentation-using-MONAI-and-WandB---Vmlldzo0MjUzODIw)
