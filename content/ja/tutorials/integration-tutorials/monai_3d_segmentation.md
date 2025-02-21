---
title: 3D brain tumor segmentation with MONAI
menu:
  tutorials:
    identifier: ja-tutorials-integration-tutorials-monai_3d_segmentation
    parent: integration-tutorials
weight: 10
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/monai/3d_brain_tumor_segmentation.ipynb" >}}

このチュートリアルでは、[MONAI](https://github.com/Project-MONAI/MONAI) を使用して、マルチラベル 3D 脳腫瘍セグメンテーションタスクのトレーニング ワークフローを構築し、[Weights & Biases](https://wandb.ai/site) の 実験管理 および Data Visualization 機能を使用する方法を説明します。このチュートリアルには、以下の機能が含まれています。

1. Weights & Biases の run を初期化し、再現性のために run に関連付けられたすべての構成を同期します。
2. MONAI transform API:
    1. 辞書形式のデータに対する MONAI Transforms 。
    2. MONAI `transforms` API に従って新しい transform を定義する方法。
    3. Data Augmentation のために強度をランダムに調整する方法。
3. データのロードと可視化:
    1. メタデータを含む `Nifti` 画像のロード、画像のリストのロードとスタック。
    2. トレーニングと検証を高速化するためのキャッシュ IO と transform 。
    3. `wandb.Table` と Weights & Biases 上のインタラクティブなセグメンテーションオーバーレイを使用したデータの可視化。
4. 3D `SegResNet` モデルのトレーニング
    1. MONAI の `networks` 、 `losses` 、および `metrics` API の使用。
    2. PyTorch トレーニングループを使用した 3D `SegResNet` モデルのトレーニング。
    3. Weights & Biases を使用したトレーニング実験の追跡。
    4. Weights & Biases 上のモデル Artifacts としてのモデルチェックポイントのログとバージョニング。
5. `wandb.Table` と Weights & Biases 上のインタラクティブなセグメンテーションオーバーレイを使用した、検証データセットでの予測の可視化と比較。

## セットアップとインストール

まず、MONAI と Weights & Biases の両方の最新バージョンをインストールします。

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

## W&B Run の初期化

新しい W&B run を開始して、実験の追跡を開始します。

```python
wandb.init(project="monai-brain-tumor-segmentation")
```

適切な構成システムの使用は、再現性のある Machine Learning の推奨されるベストプラクティスです。W&B を使用して、すべての実験のハイパーパラメーターを追跡できます。

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

また、決定論的トレーニングを有効または無効にするために、モジュールの乱数シードを設定する必要があります。

```python
set_determinism(seed=config.seed)

# ディレクトリを作成
os.makedirs(config.dataset_dir, exist_ok=True)
os.makedirs(config.checkpoint_dir, exist_ok=True)
```

## データのロードと変換

ここでは、 `monai.transforms` API を使用して、マルチクラスラベルを one-hot 形式のマルチラベルセグメンテーションタスクに変換するカスタム transform を作成します。

```python
class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    brats クラスに基づいてラベルをマルチチャンネルに変換します:
    ラベル 1 はperitumoral edema
    ラベル 2 はGD-enhancing tumor
    ラベル 3 はnecotic and non-enhancing tumor core
    可能なクラスはTC (Tumor core)、WT (Whole tumor)
    およびET (Enhancing tumor) です。

    参考: https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # ラベル 2 とラベル 3 をマージして TC を構築
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # ラベル 1、2、3 をマージして WT を構築
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

次に、トレーニングデータセットと検証データセットの transform をそれぞれ設定します。

```python
train_transform = Compose(
    [
        # 4 つの Nifti 画像をロードして一緒にスタック
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

この実験で使用されるデータセットは、http://medicaldecathlon.com/ から提供されています。これは、Gliomas、壊死/活動性腫瘍、および浮腫をセグメント化するために、マルチモーダルマルチサイト MRI データ (FLAIR、T1w、T1gd、T2w) を使用します。データセットは、750 個の 4D ボリューム (484 トレーニング + 266 テスト) で構成されています。

`DecathlonDataset` を使用して、データセットを自動的にダウンロードして抽出します。これは MONAI `CacheDataset` を継承しており、 `cache_num=N` を設定してトレーニング用に `N` 個のアイテムをキャッシュし、メモリサイズに応じて、デフォルトの引数を使用して検証用にすべてのアイテムをキャッシュできます。

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
**注:** `train_transform` を `train_dataset` に適用する代わりに、トレーニングデータセットと検証データセットの両方に `val_transform` を適用します。これは、トレーニングの前に、データセットの分割の両方からサンプルを可視化するためです。
{{% /alert %}}

### データセットの可視化

Weights & Biases は、画像、ビデオ、オーディオなどをサポートしています。リッチメディアを記録して結果を調査し、run 、Models 、および Datasets を視覚的に比較できます。[セグメンテーションマスクオーバーレイシステム]({{< relref path="/guides/models/track/log/media/#image-overlays-in-tables" lang="ja" >}})を使用して、データボリュームを可視化します。[Tables ]({{< relref path="/guides/core/tables/" lang="ja" >}})にセグメンテーションマスクを記録するには、テーブルの各行に `wandb.Image` オブジェクトを指定する必要があります。

以下に擬似コードの例を示します。

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

次に、サンプル画像、ラベル、 `wandb.Table` オブジェクト、および関連するメタデータを受け取り、Weights & Biases ダッシュボードに記録されるテーブルの行に入力する簡単なユーティリティ関数を作成します。

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

次に、 `wandb.Table` オブジェクトと、データ可視化を入力できるように、それが構成する列を定義します。

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

次に、 `train_dataset` と `val_dataset` をそれぞれループして、データサンプルの可視化を生成し、ダッシュボードに記録するテーブルの行に入力します。

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
    desc="トレーニングデータセットの可視化を生成中:",
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
    desc="検証データセットの可視化を生成中:",
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

# テーブルをダッシュボードに記録
wandb.log({"Tumor-Segmentation-Data": table})
```

データは、インタラクティブな表形式で W&B ダッシュボードに表示されます。各行にそれぞれのセグメンテーションマスクがオーバーレイされたデータボリュームから、特定の スライスの 各チャネルを確認できます。[Weave クエリ]({{< relref path="/guides/weave" lang="ja" >}})を作成して、テーブルのデータをフィルタリングし、特定の行に焦点を当てることができます。

| {{< img src="/images/tutorials/monai/viz-1.gif" alt="ログに記録されたテーブルデータの例。" >}} | 
|:--:| 
| **ログに記録されたテーブルデータの例。** |

画像を開いて、インタラクティブなオーバーレイを使用して各セグメンテーションマスクを操作する方法を確認してください。

| {{< img src="/images/tutorials/monai/viz-2.gif" alt="可視化されたセグメンテーションマップの例。" >}} | 
|:--:| 
| **可視化されたセグメンテーションマップの例。** |

{{% alert %}}
**注:** データセット内のラベルは、クラス間で重複しないマスクで構成されています。オーバーレイは、ラベルをオーバーレイ内の個別のマスクとして記録します。
{{% /alert %}}

### データのロード

データセットからデータをロードするために、PyTorch DataLoaders を作成します。DataLoaders を作成する前に、トレーニングのためにデータを前処理および変換するために、 `train_dataset` の `transform` を `train_transform` に設定します。

```python
# トレーニングデータセットに train_transforms を適用
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

## モデル、損失、およびオプティマイザーの作成

このチュートリアルでは、論文 [自動エンコーダー正則化を使用した 3D MRI 脳腫瘍セグメンテーション](https://arxiv.org/pdf/1810.11654.pdf) に基づいて `SegResNet` モデルを作成します。 `monai.networks` API の一部として PyTorch モジュールとして実装されている `SegResNet` モデル、および オプティマイザー と学習率スケジューラー。

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

# オプティマイザーを作成
optimizer = torch.optim.Adam(
    model.parameters(),
    config.initial_learning_rate,
    weight_decay=config.weight_decay,
)

# 学習率スケジューラーを作成
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=config.max_train_epochs
)
```

`monai.losses` API を使用して損失をマルチラベル `DiceLoss` として定義し、 `monai.metrics` API を使用して対応する Dice メトリクスを定義します。

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

# 自動混合精度を使用してトレーニングを高速化
scaler = torch.cuda.amp.GradScaler()
torch.backends.cudnn.benchmark = True
```

混合精度推論のための小さなユーティリティを定義します。これは、トレーニングプロセスの検証ステップ中、およびトレーニング後にモデルを実行する場合に役立ちます。

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

トレーニングの前に、トレーニング実験と検証実験を追跡するために、後で `wandb.log()` で記録されるメトリックプロパティを定義します。

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

epoch_progress_bar = tqdm(range(config.max_train_epochs), desc="トレーニング:")

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
        ## バッチごとのトレーニング損失を W&B に記録
        wandb.log({"batch/batch_step": batch_step, "batch/train_loss": loss.item()})
        batch_step += 1

    lr_scheduler.step()
    epoch_loss /= total_batch_steps
    ## バッチごとのトレーニング損失と学習率を W&B に記録
    wandb.log(
        {
            "epoch/epoch_step": epoch,
            "epoch/mean_train_loss": epoch_loss,
            "epoch/learning_rate": lr_scheduler.get_last_lr()[0],
        }
    )
    epoch_progress_bar.set_description(f"トレーニング: train_loss: {epoch_loss:.4f}:")

    # 検証とモデルチェックポイントステップ
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
            
            # W&B Artifacts を使用してモデルチェックポイントを記録およびバージョン管理
            artifact.add_file(local_path=checkpoint_path)
            wandb.log_artifact(artifact, aliases=[f"epoch_{epoch}"])

            # 検証メトリクスを W&B ダッシュボードに記録
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


# この Artifact がログ記録を終了するまで待機
artifact.wait()
```

`wandb.log` でコードを計測すると、トレーニングおよび検証プロセスに関連付けられたすべてのメトリクスだけでなく、W&B ダッシュボード上のすべてのシステムメトリクス (この場合は CPU および GPU ) を追跡できます。

| {{< img src="/images/tutorials/monai/viz-3.gif" alt="W&B でのトレーニングおよび検証プロセスの追跡の例。" >}} | 
|:--:| 
| **W&B でのトレーニングおよび検証プロセスの追跡の例。** |

W&B run ダッシュボードの Artifacts タブに移動して、トレーニング中に記録されたモデルチェックポイント Artifacts のさまざまなバージョンにアクセスします。

| {{< img src="/images/tutorials/monai/viz-4.gif" alt="W&B でのモデルチェックポイントのログ記録とバージョニングの例。" >}} | 
|:--:| 
| **W&B でのモデルチェックポイントのログ記録とバージョニングの例。** |

## 推論

Artifacts インターフェースを使用すると、Artifact のどのバージョンが最適なモデルチェックポイントであるかを選択できます。この場合、エポックごとの平均トレーニング損失です。Artifact の系統全体を調べて、必要なバージョンを使用することもできます。

| {{< img src="/images/tutorials/monai/viz-5.gif" alt="W&B でのモデル Artifact の追跡の例。" >}} | 
|:--:| 
| **W&B でのモデル Artifact の追跡の例。** |

エポックごとの平均トレーニング損失が最適なモデル Artifact のバージョンを取得し、チェックポイント状態ディクショナリをモデルにロードします。

```python
model_artifact = wandb.use_artifact(
    "geekyrakshit/monai-brain-tumor-segmentation/d5ex6n4a-checkpoint:v49",
    type="model",
)
model_artifact_dir = model_artifact.download()
model.load_state_dict(torch.load(os.path.join(model_artifact_dir, "model.pth")))
model.eval()
```

### 予測の可視化と正解ラベルとの比較

別のユーティリティ関数を作成して、事前トレーニング済みモデルの予測を可視化し、インタラクティブなセグメンテーションマスクオーバーレイを使用して、対応する正解セグメンテーションマスクと比較します。

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

予測結果を予測テーブルに記録します。

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

# 推論と可視化の実行
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
        desc="予測を生成中:",
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

インタラクティブなセグメンテーションマスクオーバーレイを使用して、各クラスの予測されたセグメンテーションマスクと正解ラベルを分析および比較します。

| {{< img src="/images/tutorials/monai/viz-6.gif" alt="W&B での予測と正解の可視化の例。" >}} | 
|:--:| 
| **W&B での予測と正解の可視化の例。** |

## 謝辞とその他のリソース

* [MONAI チュートリアル: MONAI を使用した脳腫瘍の 3D セグメンテーション](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb)
* [WandB レポート: MONAI と WandB を使用した脳腫瘍のセグメンテーション](https://wandb.ai/geekyrakshit/brain-tumor-segmentation/reports/Brain-Tumor-Segmentation-using-MONAI-and-WandB---Vmlldzo0MjUzODIw)
