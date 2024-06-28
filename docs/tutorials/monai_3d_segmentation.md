import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';


# 3D brain tumor segmentation with MONAI

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/main/colabs/monai/3d_brain_tumor_segmentation.ipynb"></CTAButtons>

このチュートリアルでは、[MONAI](https://github.com/Project-MONAI/MONAI) を使用してマルチラベルの3D脳腫瘍セグメンテーションタスクのトレーニングワークフローを構築し、[Weights & Biases](https://wandb.ai/site) の実験管理とデータ可視化機能を使用する方法を示します。チュートリアルには次の機能が含まれています。

1. Weights & Biases の run を初期化し、再現性のために run に関連するすべての設定を同期します。
2. MONAI transform API:
    1. 辞書形式データのための MONAI Transforms。
    2. MONAI `transforms` API に従って新しい transform を定義する方法。
    3. データ拡張のために強度をランダムに調整する方法。
3. データのロードと可視化:
    1. メタデータ付き `Nifti` 画像をロードし、画像リストをロードしてそれらをスタックします。
    2. トレーニングと検証を加速するためのキャッシュ IO と transforms。
    3. `wandb.Table` を使用してデータを可視化し、Weights & Biases 上でインタラクティブなセグメンテーションオーバーレイを行います。
4. 3D `SegResNet` モデルのトレーニング:
    1. MONAI の `networks`、`losses`、`metrics` APIs を使用。
    2. PyTorch トレーニングループを使用して 3D `SegResNet` モデルをトレーニング。
    3. Weights & Biases を使用してトレーニング実験を追跡します。
    4. モデルのチェックポイントを Weights & Biases のアーティファクトとしてログおよびバージョン管理します。
5. `wandb.Table` および Weights & Biases 上でインタラクティブなセグメンテーションオーバーレイを使用して、検証データセット上の予測を可視化および比較します。

## 🌴 Set 🌻アップとインストール 🌳🌳🌳🌳🌳🌳🌳
最初に、最新バージョンの MONAI と Weights & Biases をインストールします。

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

次に、Colab インスタンスを認証して W&B を使用できるようにします。

```python
wandb.login()
```

## 🌳 W&B Run の初期化

新しい W&B run を開始して実験を追跡します。

```python
wandb.init(project="monai-brain-tumor-segmentation")
```

適切な config システムを使用することは、再現性のある機械学習のための推奨ベストプラクティスです。W&B を使用して各実験のハイパーパラメーターを追跡できます。

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

モジュールのランダムシードを設定して、決定的トレーニングを有効または無効にする必要もあります。

```python
set_determinism(seed=config.seed)

# ディレクトリを作成
os.makedirs(config.dataset_dir, exist_ok=True)
os.makedirs(config.checkpoint_dir, exist_ok=True)
```

## 💿 データのロードと変換

ここでは、`monai.transforms` API を使用して、マルチクラスラベルを one-hot 形式のマルチラベルセグメンテーションタスクに変換するカスタムトランスフォームを作成します。

```python
class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    brats クラスに基づいてラベルをマルチチャンネルに変換します：
    ラベル 1 は周囲の浮腫
    ラベル 2 は GD 増強腫瘍
    ラベル 3 は壊死と非増強腫瘍コア
    クラスは TC (腫瘍コア)、WT (全腫瘍)、ET (増強腫瘍) です。

    参考: https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # ラベル 2 とラベル 3 を結合して TC を構築
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # ラベル 1, 2 と 3 を結合して WT を構築
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

次に、トレーニングおよび検証データセットのための transform を設定します。

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

### 🍁 データセット

この実験に使用するデータセットは http://medicaldecathlon.com/ から取得されます。これは、FLAIR、T1w、T1gd、T2w のマルチモーダル・マルチサイト MRI データを使用して、グリオーマ、壊死/活動中の腫瘍、および浮腫をセグメント化します。データセットは 750 の 4D ボリューム (484 トレーニング + 266 テスト) で構成されています。

`DecathlonDataset` を使用してデータセットを自動的にダウンロードおよび抽出します。 これは MONAI `CacheDataset` を継承しており、メモリサイズに応じて `cache_num=N` を設定してトレーニングのために `N` 個のアイテムをキャッシュし、検証用にすべてのアイテムをキャッシュするデフォルト設定を利用できます。

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
**Note:**  `train_dataset` に `train_transform` を適用する代わりに、トレーニングデータと検証データの両方に `val_transform` を適用します。これは、トレーニング前にデータセットの両方のサンプルを可視化するためです。
:::

### 📸 データセットの可視化

Weights & Biases は画像、ビデオ、オーディオなどをサポートしています。リッチメディアをログに記録して結果を探索し、run、model、および dataset を視覚的に比較できます。データボリュームを視覚化するための[セグメンテーションマスクオーバーレイシステム](https://docs.wandb.ai/guides/track/log/media#image-overlays-in-tables) を使用します。[tables](https://docs.wandb.ai/guides/tables) にセグメンテーションマスクをログに記録するには、テーブルの各行に `wandb.Image` オブジェクトを提供する必要があります。

擬似コードの例を以下に示します:

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

次に、サンプル画像、ラベル、`wandb.Table` オブジェクト、およびいくつかの関連するメタデータを受け取り、Weights & Biases ダッシュボードにログするテーブルの行を埋めるユーティリティ関数を書きます。

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

次に、`wandb.Table` オブジェクトとその列を定義して、データ可視化で埋めることができるようにします。

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

次に、それぞれの `train_dataset` と `val_dataset` をループして、データサンプルの可視化を生成し、テーブルの行を埋めてダッシュボードにログします。

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

データが W&B のダッシュボードにインタラクティブな表形式で表示されます。データボリュームの特定のスライスの各チャンネルに対応する行で、それぞれのセグメンテーションマスクが重ねられた状態を確認できます。 [Weave クエリ](https://docs.wandb.ai/guides/weave) を記述して、表のデータをフィルタリングし、特定の行に注目できます。

| ![An example of logged table data.](@site/static/images/tutorials/monai/viz-1.gif) | 
|:--:| 
| **An example of logged table data.** |

画像を開いて、インタラクティブなオーバーレイを使用して各セグメンテーションマスクをどのように操作できるかを確認してください。

| ![An example of visualized segmentation maps.](@site/static/images/tutorials/monai/viz-2.gif) | 
|:--:| 
| **An example of visualized segmentation maps.* |

:::info
**Note:** このデータセットのラベルはクラス間で非重複マスクを含みます。オーバーレイはラベルを個別のマスクとしてログに記録します。
:::

### 🛫 データのロード

データセットからデータをロードするための PyTorch DataLoader を作成します。DataLoader を作成する前に、 `train_transform` を ` train_dataset` に設定して、トレーニングのためにデータを前処理および変換します。

```python
# トレーニングデータセットに train_transforms を適用
train_dataset.transform = train_transform

# train_loader を作成
train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
   

## 🤖 Creating the Model, Loss, and Optimizer

このチュートリアルでは、[3D MRI brain tumor segmentation using auto-encoder regularization](https://arxiv.org/pdf/1810.11654.pdf) という論文に基づいて `SegResNet` モデルを作成します。この `SegResNet` モデルは、`monai.networks` API の一部として PyTorch Module として実装されており、オプティマイザーや学習率スケジューラーも含まれています。

```python
device = torch.device("cuda:0")

# model を作成
model = SegResNet(
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=16,
    in_channels=4,
    out_channels=3,
    dropout_prob=0.2,
).to(device)

# optimizer を作成
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

`monai.losses` API を使用してマルチラベルの `DiceLoss` を損失関数として定義し、対応するダイスメトリクスを `monai.metrics` API を使って定義します。

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

混合精度推論のための小さなユーティリティを定義します。これはトレーニングプロセスの検証ステップおよび訓練後にモデルを実行する際に役立ちます。

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

## 🚝 Training and Validation

トレーニングの前に、`wandb.log()` を使用して、トレーニングおよび検証実験のトラッキングのために後でログに記録するメトリクスプロパティを定義します。

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

### 🍭 Execute Standard PyTorch Training Loop

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
        # バッチ単位のトレーニング損失をW&Bにログ
        wandb.log({"batch/batch_step": batch_step, "batch/train_loss": loss.item()})
        batch_step += 1

    lr_scheduler.step()
    epoch_loss /= total_batch_steps
    # エポック単位のトレーニング損失と学習率をW&Bにログ
    wandb.log(
        {
            "epoch/epoch_step": epoch,
            "epoch/mean_train_loss": epoch_loss,
            "epoch/learning_rate": lr_scheduler.get_last_lr()[0],
        }
    )
    epoch_progress_bar.set_description(f"Training: train_loss: {epoch_loss:.4f}:")

    # 検証およびモデルのチェックポイントステップ
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
            
            # モデルチェックポイントをW&BのArtifactを使用してログおよびバージョン管理
            artifact.add_file(local_path=checkpoint_path)
            wandb.log_artifact(artifact, aliases=[f"epoch_{epoch}"])

            # 検証メトリクスをW&Bダッシュボードにログ
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


# このArtifactのログが完了するのを待つ
artifact.wait()
```

`wandb.log` を利用することで、トレーニングおよび検証プロセスに関連するすべてのメトリクスだけでなく、W&Bダッシュボード上のすべてのシステムメトリクス（この場合、私たちのCPUとGPU）もトラッキングすることができます。

| ![An example of training and validation process tracking on W&B.](@site/static/images/tutorials/monai/viz-3.gif) | 
|:--:| 
| **An example of training and validation process tracking on W&B.** |

トレーニング中にログに記録されるモデルチェックポイントの異なるバージョンにアクセスするには、W&B run ダッシュボードのアーティファクトタブに移動します。

| ![An example of model checkpoints logging and versioning on W&B.](@site/static/images/tutorials/monai/viz-4.gif) | 
|:--:| 
| **An example of model checkpoints logging and versioning on W&B.** |

## 🔱 Inference

Artifactsインターフェースを使用して、アーティファクトのどのバージョンが最高のモデルチェックポイントであるかを選択できます。この場合、エポック単位の平均トレーニング損失がこれに当たります。また、アーティファクトの全リネージを探索し、必要なバージョンを使用することもできます。

| ![An example of model artifact tracking on W&B.](@site/static/images/tutorials/monai/viz-5.gif) | 
|:--:| 
| **An example of model artifact tracking on W&B.** |

エポック単位の平均トレーニング損失が最も良いモデルアーティファクトのバージョンを取得し、チェックポイントのステート辞書をモデルにロードします。

```python
model_artifact = wandb.use_artifact(
    "geekyrakshit/monai-brain-tumor-segmentation/d5ex6n4a-checkpoint:v49",
    type="model",
)
model_artifact_dir = model_artifact.download()
model.load_state_dict(torch.load(os.path.join(model_artifact_dir, "model.pth")))
model.eval()
```

### 📸 Visualizing Predictions and Comparing with the Ground Truth Labels

学習済みモデルの予測結果を可視化し、対応する正解セグメンテーションマスクと比較するためのユーティリティ関数を作成します。

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

# 推論および可視化を実行
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

インタラクティブなセグメンテーションマスクオーバーレイを使用して、各クラスの予測セグメンテーションマスクと正解ラベルを分析および比較します。

| ![An example of predictions and ground-truth visualization on W&B.](@site/static/images/tutorials/monai/viz-6.gif) | 
|:--:| 
| **An example of predictions and ground-truth visualization on W&B.** |

## Acknowledgements and more resources

* [MONAI Tutorial: Brain tumor 3D segmentation with MONAI](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb)
* [WandB Report: Brain Tumor Segmentation using MONAI and WandB](https://wandb.ai/geekyrakshit/brain-tumor-segmentation/reports/Brain-Tumor-Segmentation-using-MONAI-and-WandB---Vmlldzo0MjUzODIw)