import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';


# 3D脳腫瘍セグメンテーション with MONAI

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/main/colabs/monai/3d_brain_tumor_segmentation.ipynb"></CTAButtons>

このチュートリアルは、[MONAI](https://github.com/Project-MONAI/MONAI) を使用してマルチラベルの3D脳腫瘍セグメンテーションタスクのトレーニングワークフローを構築し、[Weights & Biases](https://wandb.ai/site) の実験管理とデータ可視化機能を利用する方法を示します。チュートリアルには以下の機能が含まれています:

1. Weights & Biases の run を初期化し、再現性のために run に関連するすべての設定を同期します。
2. MONAI トランスフォーム API:
    1. 辞書形式のデータのための MONAI トランスフォーム。
    2. MONAI `transforms` API に従って新しいトランスフォームを定義する方法。
    3. データ増強のために強度をランダムに調整する方法。
3. データの読み込みと可視化:
    1. メタデータを含む `Nifti` 画像を読み込み、画像リストをロードしてスタックします。
    2. トレーニングと検証を高速化するために、IOとトランスフォームをキャッシュします。
    3. Weights & Biases で `wandb.Table` を使用してデータを可視化し、インタラクティブなセグメンテーションオーバーレイを行います。
4. 3D `SegResNet` モデルのトレーニング
    1. MONAI の `networks`, `losses`, `metrics` API を使用します。
    2. PyTorch のトレーニングループを使用して 3D `SegResNet` モデルをトレーニングします。
    3. Weights & Biases を使用して実験を追跡します。
    4. モデルチェックポイントをモデルArtifactsとしてログおよびバージョン管理します。
5. Weights & Biases で `wandb.Table` を使用して検証データセットの予測結果を可視化および比較します。

## 🌴 環境セットアップとインストール

まず、最新バージョンの MONAI と Weights & Biases をインストールします。

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

次に、Colabインスタンスを認証してW&Bを使用します。

```python
wandb.login()
```

## 🌳 W&B Run の初期化

新しい W&B run を開始して、実験の追跡を開始します。

```python
wandb.init(project="monai-brain-tumor-segmentation")
```

適切な設定システムの使用は、再現可能な機械学習のための推奨されるベストプラクティスです。W&Bを使用して、すべての実験のハイパーパラメーターを追跡できます。

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

また、ランダムシードを設定し、決定的トレーニングを有効または無効にします。

```python
set_determinism(seed=config.seed)

# ディレクトリを作成
os.makedirs(config.dataset_dir, exist_ok=True)
os.makedirs(config.checkpoint_dir, exist_ok=True)
```

## 💿 データの読み込みと変換

ここでは、`monai.transforms` API を使用して、マルチクラスラベルをワンホット形式でマルチラベルセグメンテーションタスクに変換するカスタムトランスフォームを作成します。

```python
class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    脳腫瘍クラスに基づいてラベルをマルチチャネルに変換:
    label 1 は腫瘍周辺の浮腫
    label 2 は増強された腫瘍
    label 3 は壊死しているまたは非増強された腫瘍核
    可能なクラスは TC (腫瘍核), WT (全腫瘍)
    または ET (増強された腫瘍)。

    参考: https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # label 2 と label 3 をマージして TC を構築
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # label 1, 2, 3 をマージして WT を構築
            result.append(
                torch.logical_or(
                    torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                )
            )
            # label 2 は ET
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d
```

次に、トレーニングと検証のデータセットそれぞれに対してトランスフォームを設定します。

```python
train_transform = Compose(
    [
        # 4つのNifti画像を読み込み、一緒にスタックします
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

### 🍁 データセット

この実験で使用するデータセットは http://medicaldecathlon.com/ から取得されたものです。FLAIR、T1w、T1gd、T2w などのマルチモーダルなマルチサイト MRI データを使用して、非増強/活動腫瘍、壊死部位をセグメント化します。データセットは合計750 の4Dボリューム（トレーニング 484 + テスト 266）で構成されています。

`DecathlonDataset` を使用して、データセットを自動的にダウンロードして解凍します。これは MONAI の `CacheDataset` を継承しており、`cache_num=N` をセットしてトレーニング用にN個のアイテムをキャッシュし、検証用にはメモリサイズに応じてデフォルトの引数を使用してすべてのアイテムをキャッシュできます。

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
**Note:** `train_transform` を `train_dataset` に適用する代わりに、データセットの両スプリットからサンプルをビジュアライズする前に、検証データセットとトレーニングデータセットの両方に `val_transform` を適用します。
:::

### 📸 データセットの可視化

Weights & Biases は画像、ビデオ、音声などをサポートしています。結果を視覚的に比較しながらリッチメディアをログすることで、run、モデル、データセットを探求できます。[セグメンテーションマスクオーバーレイシステム](https://docs.wandb.ai/guides/track/log/media#image-overlays-in-tables) を使用してデータボリュームを可視化します。Segmentationマスクを [tables](https://docs.wandb.ai/guides/tables) にログするには、テーブルの各行に `wandb.Image` オブジェクトを提供する必要があります。

例は以下の疑似コードに示されています:

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

次に、サンプル画像、ラベル、`wandb.Table` オブジェクト、および関連メタデータを受け取り、Weights & Biases のダッシュボードにログされるテーブルの行を埋める簡単なユーティリティ関数を作成します。

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

次に、`wandb.Table` オブジェクトとその列を定義し、データ可視化で行を埋められるようにします。

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

それから、`train_dataset` と `val_dataset` それぞれをループして、データサンプルの可視化を生成し、テーブルの行を埋めてダッシュボードにログを作成します。

```python
# train_dataset の可視化を生成
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

# val_dataset の可視化を生成
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

# テーブルをダッシュボードにログ
wandb.log({"Tumor-Segmentation-Data": table})
```

データは W&B ダッシュボード上でインタラクティブな表形式で表示されます。各チャネルの特定のスライスを確認し、それぞれの行にセグメンテーションマスクがオーバーレイされたものを見ることができます。[Weave queries](https://docs.wandb.ai/guides/weave) を使用してテーブル内のデータをフィルタリングし、特定の行に焦点を当てることができます。

| ![An example of logged table data.](@site/static/images/tutorials/monai/viz-1.gif) | 
|:--:| 
| **テーブルデータのログ例です。** |

画像を開いて、それぞれのセグメンテーションマスクとインタラクティブにオーバーレイする方法を確認します。

| ![An example of visualized segmentation maps.](@site/static/images/tutorials/monai/viz-2.gif) | 
|:--:| 
| **セグメンテーションマップの可視化例です。** |

:::info
**Note:** データセットのラベルは、クラス間でオーバーラップしないマスクで構成されています。オーバーレイは、ラベルをオーバーレイ内の別のマスクとしてログします。
:::

### 🛫 データの読み込み

データセットからデータを読み込むための PyTorch DataLoaders を作成します。DataLoaders を作成する前に、`train_dataset` に `train_transform` を設定し、トレーニング用にデータを前処理し変換します。

```python
# train_transforms をトレーニングデータセットに適用
train_dataset.transform = train_transform

# train_loader を作成
train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers,
)

# val_loader を作成
val_loader = DataLoader(
    val_dataset,
    batch_size