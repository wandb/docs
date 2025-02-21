---
title: 3D brain tumor segmentation with MONAI
menu:
  tutorials:
    identifier: ja-tutorials-integration-tutorials-monai_3d_segmentation
    parent: integration-tutorials
weight: 10
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/monai/3d_brain_tumor_segmentation.ipynb" >}}

このチュートリアルでは、[MONAI](https://github.com/Project-MONAI/MONAI) を使用して多ラベル 3D 脳腫瘍セグメンテーションタスクのトレーニングワークフローを構築し、[Weights & Biases](https://wandb.ai/site) の実験管理およびデータ可視化機能を利用する方法を示します。チュートリアルには次の機能が含まれています：

1. Weights & Biases の run を初期化し、再現性のために run に関連付けられたすべての設定を同期します。
2. MONAI トランスフォーム API:
    1. 辞書形式のデータ用の MONAI トランスフォーム。
    2. MONAI `transforms` API に従って新しいトランスフォームを定義する方法。
    3. データ拡張のためにランダムに強度を調整する方法。
3. データの読み込みと可視化:
    1. メタデータを持つ `Nifti` 画像を読み込み、画像のリストを読み込んでスタックします。
    2. トレーニングと検証を加速するためのキャッシュ IO とトランスフォーム。
    3. `wandb.Table` と Weights & Biases 上のインタラクティブセグメンテーションオーバーレイを使用してデータを可視化します。
4. 3D `SegResNet` モデルのトレーニング
    1. MONAI の `networks`、`losses`、`metrics` API を使用。
    2. PyTorch トレーニングループを使用して 3D `SegResNet` モデルをトレーニング。
    3. Weights & Biases を使用してトレーニング実験を追跡。
    4. Weights & Biases 上でモデルアーティファクトとしてモデルチェックポイントをログとバージョン管理。
5. `wandb.Table` と Weights & Biases 上のインタラクティブセグメンテーションオーバーレイを使用して検証データセットの予測を可視化および比較。

## セットアップとインストール

まず、MONAI と Weights & Biases の最新バージョンをインストールします。

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

次に、W&B を使用するために Colab インスタンスを認証します。

```python
wandb.login()
```

## W&B Run を初期化

トレーニング実験を追跡するために新しい W&B run を開始します。

```python
wandb.init(project="monai-brain-tumor-segmentation")
```

再現可能な機械学習のためには、適切な設定システムを使用することをお勧めします。W&B を使用して、各実験のハイパーパラメーターを追跡できます。

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

また、モジュールのランダムシードを設定して決定的なトレーニングをオンまたはオフにする必要があります。

```python
set_determinism(seed=config.seed)

# ディレクトリを作成
os.makedirs(config.dataset_dir, exist_ok=True)
os.makedirs(config.checkpoint_dir, exist_ok=True)
```

## データ読み込みと変換

ここでは、`monai.transforms` API を使用して、マルチクラスラベルをワンホット形式のマルチラベルのセグメンテーションタスクに変換するカスタムトランスフォームを作成します。

```python
class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    brats クラスに基づいてラベルをマルチチャンネルに変換:
    ラベル 1 は腫瘍周囲浮腫
    ラベル 2 は GD 増強腫瘍
    ラベル 3 は壊死および非増強性腫瘍コア
    可能なクラスは TC (腫瘍コア)、WT (全腫瘍)、
    ET (増強腫瘍)。

    参照: https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # TC を構成するためにラベル 2 とラベル 3 をマージ
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # WT を構成するためにラベル 1、2 と 3 をマージ
            result.append(
                torch.logical_or(
                    torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                )
            )
            # ラベル 2 は ET
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d
```

次に、トレーニングデータセットと検証データセットにそれぞれのトランスフォームを設定します。

```python
train_transform = Compose(
    [
        # 4 つの Nifti 画像を読み込んで一緒にスタックする
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

### データセット

この実験に使用されるデータセットは http://medicaldecathlon.com/ から取得します。多様なモーダリティ、多施設の MRI データ (FLAIR, T1w, T1gd, T2w) を使用して、グリオーマ、壊死/活動性腫瘍、浮腫をセグメント化します。このデータセットは 750 の 4D ボリューム (484 トレーニング + 266 テスティング) で構成されています。

`DecathlonDataset` を使用してデータセットを自動的にダウンロードおよび抽出します。MONAI の `CacheDataset` を継承しており、メモリサイズに応じてトレーニング用に `cache_num=N` を設定して N 個のアイテムをキャッシュし、検証用にデフォルトの引数を使用してすべてのアイテムをキャッシュできます。

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
**注意:** `train_dataset` に `train_transform` を適用する代わりに、トレーニングと検証のデータセットの両方に `val_transform` を適用します。これは、トレーニングの前にデータセットのスプリットからサンプルを可視化するためです。
{{% /alert %}}

### データセットの可視化

Weights & Biases は画像、ビデオ、オーディオなどをサポートしています。豊富なメディアをログすることで結果を探索し、run、model、およびデータセットを視覚的に比較できます。[セグメンテーションマスクオーバーレイシステム]({{< relref path="/guides/models/track/log/media/#image-overlays-in-tables" lang="ja" >}})を使用してデータボリュームを可視化します。[テーブル]({{< relref path="/guides/core/tables/" lang="ja" >}})にセグメンテーションマスクをログするには、`wandb.Image` オブジェクトをテーブルの各行に提供する必要があります。

以下は疑似コードでの例です：

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

次に、サンプル画像、ラベル、`wandb.Table` オブジェクト、関連するメタデータを受け取り、Weights & Biases ダッシュボードにログされるテーブルの行を埋める簡単なユーティリティ関数を書きます。

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
                            "class_labels": {0: "背景", 1: "腫瘍コア"},
                        },
                        "ground-truth/Whole-Tumor": {
                            "mask_data": sample_label[1, :, :, slice_idx] * 2,
                            "class_labels": {0: "背景", 2: "全腫瘍"},
                        },
                        "ground-truth/Enhancing-Tumor": {
                            "mask_data": sample_label[2, :, :, slice_idx] * 3,
                            "class_labels": {0: "背景", 3: "増強腫瘍"},
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

次に、`wandb.Table` オブジェクトを定義し、データ可視化で埋めるべき列を決定します。

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

次に、`train_dataset` と `val_dataset` にそれぞれループを実行して、データサンプルの可視化を生成し、ダッシュボードにログするテーブルの行を埋めます。

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

データは W&B ダッシュボード上でインタラクティブな表形式で表示されます。データボリュームの特定のスライスの各チャンネルと、それぞれのセグメンテーションマスクが各行にオーバーレイされた様子が確認できます。[Weave queries]({{< relref path="/guides/weave" lang="ja" >}}) を書いてテーブル上のデータをフィルターし、特定の行に集中することができます。

| {{< img src="/images/tutorials/monai/viz-1.gif" alt="An example of logged table data." >}} | 
|:--:| 
| **ログされたテーブルデータの例。** |

画像を開いて、それぞれのセグメンテーションマスクとどのようにインタラクションできるか確認してください。

| {{< img src="/images/tutorials/monai/viz-2.gif" alt="An example of visualized segmentation maps." >}} | 
|:--:| 
| **視覚化されたセグメンテーションマップの例。** |

{{% alert %}}
**注意:** データセットのラベルは、クラス全体で非重複のマスクで構成されています。オーバーレイは、它それぞれのマスクをオーバーレイします。
{{% /alert %}}

### データのロード

データセットからデータを読み込むための PyTorch DataLoaders を作成します。DataLoaders を作成する前に、トレーニング用データに `train_transform` を適用して、トレーニング用のデータを前処理および変換します。

```python
# トレーニングデータセットにトレーニングトランスフォームを適用
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
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers,
)
```

## モデル、損失関数、オプティマイザーの作成

このチュートリアルでは、論文 [3D MRI brain tumor segmentation using auto-encoder regularization](https://arxiv.org/pdf/1810.11654.pdf) を参考に `SegResNet` モデルを作成します。`SegResNet` モデルは、`monai.networks` API の一部として PyTorch モジュールとして実装されており、オプティマイザーと学習率スケジューラも備えています。

```python
device = torch.device("cuda:0")

# モデルを作成
model = SegResNet(
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=16,
    in_channels=4,
    out_channels=3,
    dropout_prob=0.2,
).to(device)

# オプティマイザー作成
optimizer = torch.optim.Adam(
    model.parameters(),
    config.initial_learning_rate,
    weight_decay=config.weight_decay,
)

# 学習率スケジューラ作成
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=config.max_train_epochs
)
```

`monai.losses` API を使用してマルチラベルの `DiceLoss` と、`monai.metrics` API を使用して対応するダイスメトリクスを定義します。

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

# 自動混合精度を使用してトレーニングを加速
scaler = torch.cuda.amp.GradScaler()
torch.backends.cudnn.benchmark = True
```

混合精度推論のための小さなユーティリティ関数を定義します。これはトレーニングプロセスの検証ステップおよびトレーニング後にモデルを実行する場合に役立ちます。

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

## トレーニングと検証

トレーニング前に、`wandb.log()` とともに後でログに記録するためのメトリクスのプロパティを定義し、トレーニングと検証の実験を追跡します。

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

### 標準的な PyTorch トレーニングループの実行

```python
# W&B Artifact オブジェクトを定義
artifact = wandb.Artifact(
    name=f"{wandb.run.id}-checkpoint", type="model"
)

epoch_progress_bar = tqdm(range(config.max_train_epochs), desc="Training:")

for epoch in epoch_progress_bar:
    model.train()
    epoch_loss = 0

    total_batch_steps = len(train_dataset) // train_loader.batch_size
    batch_progress_bar = tqdm(train_loader, total=total_batch_steps, leave=False)
    
    # トレーニングステップ
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
        ## バッチごとにトレーニング損失を W&B にログ
        wandb.log({"batch/batch_step": batch_step, "batch/train_loss": loss.item()})
        batch_step += 1

    lr_scheduler.step()
    epoch_loss /= total_batch_steps
    ## エポックごとにトレーニング損失と学習率を W&B にログ
    wandb.log(
        {
            "epoch/epoch_step": epoch,
            "epoch/mean_train_loss": epoch_loss,
            "epoch/learning_rate": lr_scheduler.get_last_lr()[0],
        }
    )
    epoch_progress_bar.set_description(f"Training: train_loss: {epoch_loss:.4f}:")

    # 検証とモデルのチェックポイントステップ
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
            
            # W&B アーティファクトを使用して、モデルチェックポイントをログおよびバージョン管理
            artifact.add_file(local_path=checkpoint_path)
            wandb.log_artifact(artifact, aliases=[f"epoch_{epoch}"])

            # W&B ダッシュボードに検証メトリクスをログ
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


# このアーティファクトがログを完了するまで待ちます
artifact.wait()
```

コードを `wandb.log` で計装することで、トレーニングと検証の過程に関連するすべてのメトリクスを追跡できるだけでなく、W&B ダッシュボードでのすべてのシステムメトリクス (この場合は CPU と GPU) も追跡できます。

| {{< img src="/images/tutorials/monai/viz-3.gif" alt="An example of training and validation process tracking on W&B." >}} | 
|:--:| 
| **W&B におけるトレーニングと検証プロセス追跡の例。** |

W&B の run ダッシュボードのアーティファクトタブに移動して、トレーニング中にログされたモデルチェックポイントアーティファクトの異なるバージョンにアクセスします。

| {{< img src="/images/tutorials/monai/viz-4.gif" alt="An example of model checkpoints logging and versioning on W&B." >}} | 
|:--:| 
| **W&B におけるモデルチェックポイントのログとバージョン管理の例。** |

## 推論

アーティファクトインターフェースを使用することで、アーティファクトのバージョンを選択して最良のモデルチェックポイントを判断できます。この場合、エポックごとの平均トレーニング損失を使用します。また、アーティファクトの全体的なリネージを探索し、必要なバージョンを使用できます。

| {{< img src="/images/tutorials/monai/viz-5.gif" alt="An example of model artifact tracking on W&B." >}} | 
|:--:| 
| **W&B におけるモデルアーティファクト追跡の例。** |

エポックごとの平均トレーニング損失が最良であるモデルアーティファクトのバージョンをフェッチし、チェックポイント状態辞書をモデルにロードします。

```python
model_artifact = wandb.use_artifact(
    "geekyrakshit/monai-brain-tumor-segmentation/d5ex6n4a-checkpoint:v49",
    type="model",
)
model_artifact_dir = model_artifact.download()
model.load_state_dict(torch.load(os.path.join(model_artifact_dir, "model.pth")))
model.eval()
```

### 予測の可視化およびグラウンドトゥルースラベルとの比較

事前学習モデルの予測を可視化し、インタラクティブなセグメンテーションマスクオーバーレイを使用して対応するグラウンドトゥルースセグメンテーションマスクと比較する別のユーティリティ関数を作成します。

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
                                "class_labels": {0: "背景", 1: "腫瘍コア"},
                            },
                            "prediction/Tumor-Core": {
                                "mask_data": predicted_label[0, :, :, slice_idx] * 2,
                                "class_labels": {0: "背景", 2: "腫瘍コア"},
                            },
                        },
                    ),
                    wandb.Image(
                        sample_image[channel_idx, :, :, slice_idx],
                        masks={
                            "ground-truth/Whole-Tumor": {
                                "mask_data": sample_label[1, :, :, slice_idx],
                                "class_labels": {0: "背景", 1: "全腫瘍"},
                            },
                            "prediction/Whole-Tumor": {
                                "mask_data": predicted_label[1, :, :, slice_idx] * 2,
                                "class_labels": {0: "背景", 2: "全腫瘍"},
                            },
                        },
                    ),
                    wandb.Image(
                        sample_image[channel_idx, :, :, slice_idx],
                        masks={
                            "ground-truth/Enhancing-Tumor": {
                                "mask_data": sample_label[2, :, :, slice_idx],
                                "class_labels": {0: "背景", 1: "増強腫瘍"},
                            },
                            "prediction/Enhancing-Tumor": {
                                "mask_data": predicted_label[2, :, :, slice_idx] * 2,
                                "class_labels": {0: "背景", 2: "増強腫瘍"},
                            },
                        },
                    ),
                ]
            table.add_data(split, data_idx, slice_idx, *wandb_images)
            progress_bar.update(1)
    return table
```

予測結果を予測テーブルにログします。

```python
# 予測テーブルを作成
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

# 推論と可視化を実行
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

    wandb.log({"Predictions/Tumor-Segmentation-Data": prediction_table})


# 実験を終了
wandb.finish()
```

インタラクティブなセグメンテーションマスクオーバーレイを使用して、各クラスの予測されたセグメンテーションマスクとグラウンドトゥルースラベルを分析および比較します。

| {{< img src="/images/tutorials/monai/viz-6.gif" alt="An example of predictions and ground-truth visualization on W&B." >}} | 
|:--:| 
| **W&B における予測とグラウンドトゥルースの可視化の例。** |

## 謝辞とさらなるリソース

* [MONAI チュートリアル: MONAI を使用した脳腫瘍 3D セグメンテーション](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb)
* [WandB レポート: MONAI と WandB を使用した脳腫瘍セグメンテーション](https://wandb.ai/geekyrakshit/brain-tumor-segmentation/reports/Brain-Tumor-Segmentation-using-MONAI-and-WandB---Vmlldzo0MjUzODIw)
```