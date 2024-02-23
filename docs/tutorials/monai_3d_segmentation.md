---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# MONAI를 사용한 3D 뇌종양 분할

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/main/colabs/monai/3d_brain_tumor_segmentation.ipynb"></CTAButtons>

이 튜토리얼은 [MONAI](https://github.com/Project-MONAI/MONAI)를 사용하여 다중 라벨 3D 뇌종양 분할 작업의 학습 워크플로를 구축하는 방법과 [Weights & Biases](https://wandb.ai/site)의 실험 추적 및 데이터 시각화 기능을 사용하는 방법을 보여줍니다. 튜토리얼에는 다음 기능이 포함됩니다:

1. Weights & Biases 실행을 초기화하고 재현성을 위해 실행과 관련된 모든 설정을 동기화합니다.
2. MONAI 변환 API:
    1. 사전 형식 데이터를 위한 MONAI 변환.
    2. MONAI `transforms` API에 따라 새로운 변환을 정의하는 방법.
    3. 데이터 증강을 위해 무작위로 강도를 조절하는 방법.
3. 데이터 로딩 및 시각화:
    1. 메타데이터와 함께 `Nifti` 이미지를 로드하고, 이미지 목록을 로드하여 쌓습니다.
    2. 학습 및 검증을 가속화하기 위해 IO 및 변환을 캐시합니다.
    3. `wandb.Table`과 Weights & Biases에서 대화형 분할 오버레이를 사용하여 데이터를 시각화합니다.
4. 3D `SegResNet` 모델 학습
    1. MONAI의 `networks`, `losses`, `metrics` API 사용.
    2. PyTorch 학습 루프를 사용하여 3D `SegResNet` 모델 학습.
    3. Weights & Biases를 사용하여 학습 실험 추적.
    4. Weights & Biases에 모델 체크포인트를 로그하고 버전 관리를 위해 모델 아티팩트로 기록.
5. `wandb.Table`과 Weights & Biases에서 대화형 분할 오버레이를 사용하여 검증 데이터셋의 예측값을 시각화하고 비교합니다.

## 🌴 설정 및 설치

먼저 MONAI와 Weights and Biases의 최신 버전을 설치합니다.

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

그런 다음 Colab 인스턴스를 W&B에 인증합니다.

```python
wandb.login()
```

## 🌳 W&B 실행 초기화

새 W&B 실행을 시작하여 실험 추적을 시작합니다.

```python
wandb.init(project="monai-brain-tumor-segmentation")
```

재현 가능한 머신러닝을 위한 적절한 설정 시스템 사용이 권장되는 모범 사례입니다. W&B를 사용하여 모든 실험의 하이퍼파라미터를 추적할 수 있습니다.

```python
config = wandb.config
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

결정적인 학습을 활성화하거나 끄기 위해 모듈에 대한 랜덤 시드도 설정해야 합니다.

```python
set_determinism(seed=config.seed)

# 디렉토리 생성
os.makedirs(config.dataset_dir, exist_ok=True)
os.makedirs(config.checkpoint_dir, exist_ok=True)
```

## 💿 데이터 로딩 및 변환

여기에서는 `monai.transforms` API를 사용하여 다중 클래스 라벨을 다중 라벨 분할 작업의 원-핫 형식으로 변환하는 사용자 정의 변환을 생성합니다.

```python
class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Brats 클래스에 기반한 다중 채널로 라벨을 변환:
    라벨 1은 주변부 부종
    라벨 2는 GD 강화 종양
    라벨 3은 괴사 및 비강화 종양 코어
    가능한 클래스는 TC (종양 코어), WT (전체 종양)
    및 ET (강화 종양)입니다.

    참조: https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # 라벨 2와 라벨 3을 병합하여 TC 구성
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # 라벨 1, 2, 3을 병합하여 WT 구성
            result.append(
                torch.logical_or(
                    torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                )
            )
            # 라벨 2는 ET
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d
```

다음으로, 각각 학습 및 검증 데이터셋을 위한 변환을 설정합니다.

```python
train_transform = Compose(
    [
        # 4개의 Nifti 이미지를 로드하고 함께 쌓습니다
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

### 🍁 데이터셋

이 실험에서 사용된 데이터셋은 http://medicaldecathlon.com/에서 온 것입니다. 이는 글리오마, 괴사/활성 종양 및 부종을 분할하기 위해 다중 모달 다중 사이트 MRI 데이터(FLAIR, T1w, T1gd, T2w)를 사용합니다. 데이터셋은 750개의 4D 볼륨(484 학습 + 266 테스트)으로 구성됩니다.

`DecathlonDataset`을 사용하여 데이터셋을 자동으로 다운로드하고 추출합니다. 이는 MONAI `CacheDataset`을 상속받아 `cache_num=N`을 설정하여 학습용으로 `N` 항목을 캐시하고 메모리 크기에 따라 검증용으로 모든 항목을 캐시할 수 있습니다.

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

:::info
**참고:** 학습 전에 학습 및 검증 데이터셋의 분할에서 샘플을 시각화하기 때문에 `train_dataset`에 `train_transform`을 적용하는 대신 `val_transform`을 학습 및 검증 데이터셋에 모두 적용합니다.
:::

### 📸 데이터셋 시각화

Weights & Biases는 이미지, 비디오, 오디오 등을 지원합니다. 결과를 탐색하고 실행, 모델 및 데이터셋을 시각적으로 비교하기 위해 풍부한 미디어를 로그할 수 있습니다. [분할 마스크 오버레이 시스템](https://docs.wandb.ai/guides/track/log/media#image-overlays-in-tables)을 사용하여 데이터 볼륨을 시각화합니다. [tables](https://docs.wandb.ai/guides/tables)에서 분할 마스크를 로그하려면 각 행의 테이블에 대해 `wandb.Image` 객체를 제공해야 합니다.

아래의 의사코드에서 예제를 제공합니다:

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

wandb.log({"Table": table})
```

샘플 이미지, 라벨, `wandb.Table` 객체 및 일부 관련 메타데이터를 가져와 Weights & Biases 대시보드에 로그될 테이블의 행을 채우는 간단한 유틸리티 함수를 작성합니다.

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

다음으로 `wandb.Table` 객체와 데이터 시각화로 채울 테이블의 열이 무엇인지 정의합니다.

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

그런 다음 `train_dataset` 및 `val_dataset`을 각각 반복하여 데이터 샘플의 시각화를 생성하고 대시보드에 로그할 테이블의 행을 채웁니다.

```python
# train_dataset에 대한 시각화 생성
max_samples = (
    min(config.max_train_images_visualized, len(train_dataset))
    if config.max_train_images_visualized > 0
    else len(train_dataset)
)
progress_bar = tqdm(
    enumerate(train_dataset[:max_samples]),
    total=max_samples,
    desc="학습 데이터셋 시각화 생성 중:",
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

# val_dataset에 대한 시각화 생성
max_samples = (
    min(config.max_val_images_visualized, len(val_dataset))
    if config.max_val_images_visualized > 0
    else len(val_dataset)
)
progress_bar = tqdm(
    enumerate(val_dataset[:max_samples]),
    total=max_samples,
    desc="검증 데이터셋 시각화 생성 중:",
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

# 대시보드에 테이블 로그
wandb.log({"Tumor-Segmentation-Data": table})
```

데이터는 대화형 테이블 형식으로 W&B 대시보드에 나타납니다. 한 데이터 볼륨의 특정 슬라이스의 각 채널을 각각의 분할 마스크와 함께 오버레이하여 각 행에서 볼 수 있습니다. [Weave 쿼리](https://docs.wandb.ai/guides/weave)를 작성하여 테이블에서 데이터를 필터링하고 특정 행에 초점을 맞출 수 있습니다.

| ![로그된 테이블 데이터의 예시.](@site/static/images/tutorials/monai/viz-1.gif) | 
|:--:| 
| **로그된 테이블 데이터의 예시.** |

이미지를 열어 대화형 오버레이를 사용하여 각 분할 마스크와 상호 작용하는 방법을 확인하세요.

| ![시각화된 분할 지도

## 🚝 학습 및 검증

학습 전에 나중에 `wandb.log()`로 로깅될 메트릭 속성을 정의하여 학습 및 검증 실험을 추적합니다.

```python
wandb.define_metric("epoch/epoch_step")
wandb.define_metric("epoch/*", step_metric="epoch/epoch_step")
wandb.define_metric("batch/batch_step")
wandb.define_metric("batch/*", step_metric="batch/batch_step")
wandb.define_metric("validation/validation_step")
wandb.define_metric("validation/*", step_metric="validation/validation_step")

batch_step = 0
validation_step = 0
metric_values = []
metric_values_tumor_core = []
metric_values_whole_tumor = []
metric_values_enhanced_tumor = []
```

### 🍭 표준 PyTorch 학습 루프 실행하기

```python
# W&B 아티팩트 개체 정의하기
artifact = wandb.Artifact(
    name=f"{wandb.run.id}-체크포인트", type="model"
)

epoch_progress_bar = tqdm(range(config.max_train_epochs), desc="학습 중:")

for epoch in epoch_progress_bar:
    model.train()
    epoch_loss = 0

    total_batch_steps = len(train_dataset) // train_loader.batch_size
    batch_progress_bar = tqdm(train_loader, total=total_batch_steps, leave=False)
    
    # 학습 단계
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
        ## W&B에 배치 별 학습 손실 로깅
        wandb.log({"batch/batch_step": batch_step, "batch/train_loss": loss.item()})
        batch_step += 1

    lr_scheduler.step()
    epoch_loss /= total_batch_steps
    ## W&B에 배치 별 학습 손실 및 학습률 로깅
    wandb.log(
        {
            "epoch/epoch_step": epoch,
            "epoch/mean_train_loss": epoch_loss,
            "epoch/learning_rate": lr_scheduler.get_last_lr()[0],
        }
    )
    epoch_progress_bar.set_description(f"학습 중: train_loss: {epoch_loss:.4f}:")

    # 검증 및 모델 체크포인팅 단계
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
            
            # W&B 아티팩트를 사용하여 체크포인트 로깅 및 버전 관리
            artifact.add_file(local_path=checkpoint_path)
            wandb.log_artifact(artifact, aliases=[f"epoch_{epoch}"])

            # W&B 대시보드에 검증 메트릭 로깅
            wandb.log(
                {
                    "validation/validation_step": validation_step,
                    "validation/mean_dice": metric_values[-1],
                    "validation/mean_dice_tumor_core": metric_values_tumor_core[-1],
                    "validation/mean_dice_whole_tumor": metric_values_whole_tumor[-1],
                    "validation/mean_dice_enhanced_tumor": metric_values_enhanced_tumor[-1],
                }
            )
            validation_step += 1


# 이 아티팩트 로깅이 완료될 때까지 기다리기
artifact.wait()
```

`wandb.log`를 코드에 적용함으로써 학습 및 검증 프로세스와 관련된 모든 메트릭을 추적할 뿐만 아니라 시스템 메트릭(이 경우 우리의 CPU와 GPU)도 W&B 대시보드에서 추적할 수 있습니다.

| ![W&B에서 학습 및 검증 프로세스 추적 예시.](@site/static/images/tutorials/monai/viz-3.gif) | 
|:--:| 
| **W&B에서 학습 및 검증 프로세스 추적 예시.** |

W&B 실행 대시보드의 아티팩트 탭으로 이동하여 학습 중 로깅된 모델 체크포인트 아티팩트의 다양한 버전에 엑세스하세요.

| ![W&B에서 모델 체크포인트 로깅 및 버전 관리 예시.](@site/static/images/tutorials/monai/viz-4.gif) | 
|:--:| 
| **W&B에서 모델 체크포인트 로깅 및 버전 관리 예시.** |

## 🔱 추론

아티팩트 인터페이스를 사용하여 최고의 모델 체크포인트 버전, 이 경우 평균 에포크별 학습 손실을 선택할 수 있습니다. 또한 아티팩트의 전체 계보를 탐색하고 필요한 버전을 사용할 수 있습니다.

| ![W&B에서 모델 아티팩트 추적 예시.](@site/static/images/tutorials/monai/viz-5.gif) | 
|:--:| 
| **W&B에서 모델 아티팩트 추적 예시.** |

최고의 에포크별 평균 학습 손실을 가진 모델 아티팩트 버전을 가져와서 체크포인트 상태 사전을 모델에 로드하세요.

```python
model_artifact = wandb.use_artifact(
    "geekyrakshit/monai-brain-tumor-segmentation/d5ex6n4a-체크포인트:v49",
    type="model",
)
model_artifact_dir = model_artifact.download()
model.load_state_dict(torch.load(os.path.join(model_artifact_dir, "model.pth")))
model.eval()
```

### 📸 예측 시각화 및 실제값 라벨과 비교하기

인터랙티브 분할 마스크 오버레이를 사용하여 pre-trained 모델의 예측을 시각화하고 해당 실제 분할 마스크와 비교하는 또 다른 유틸리티 함수를 생성합니다.

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

예측 결과를 예측 테이블에 로깅합니다.

```python
# 예측 테이블 생성
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

# 추론 및 시각화 수행
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
        desc="예측 생성 중:",
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

    wandb.log({"Predictions/Tumor-Segmentation-Data": prediction_table})


# 실험 종료
wandb.finish()
```

인터랙티브 분할 마스크 오버레이를 사용하여 각 클래스에 대한 예측 분할 마스크와 실제 라벨을 분석하고 비교합니다.

| ![W&B에서 예측 및 실제값 시각화 예시.](@site/static/images/tutorials/monai/viz-6.gif) | 
|:--:| 
| **W&B에서 예측 및 실제값 시각화 예시.** |

## 감사의 말 및 추가 자료

* [MONAI 튜토리얼: MONAI를 사용한 뇌종양 3D 분할](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb)
* [WandB 리포트: MONAI와 WandB를 사용한 뇌종양 분할](https://wandb.ai/geekyrakshit/brain-tumor-segmentation/reports/Brain-Tumor-Segmentation-using-MONAI-and-WandB---Vmlldzo0MjUzODIw)