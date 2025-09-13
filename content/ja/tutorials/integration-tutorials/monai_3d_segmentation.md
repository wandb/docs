---
title: MONAI を用いた 3D 脳腫瘍セグメンテーション
menu:
  tutorials:
    identifier: ja-tutorials-integration-tutorials-monai_3d_segmentation
    parent: integration-tutorials
weight: 10
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/monai/3d_brain_tumor_segmentation.ipynb" >}}

このチュートリアルでは、[MONAI](https://github.com/Project-MONAI/MONAI) を使って多ラベル 3D 脳腫瘍セグメンテーションのトレーニング ワークフローを構築し、[W&B](https://wandb.ai/site) の 実験管理 と データ可視化 機能を活用する方法を紹介します。チュートリアルでは次の内容を扱います:

1. W&B Run を初期化し、再現性のために run に関連するすべての config を同期する。
2. MONAI transform API:
    1. 辞書形式のデータに対する MONAI Transforms。
    2. MONAI の `transforms` API に従って新しい transform を定義する方法。
    3. データ拡張のために強度をランダムに調整する方法。
3. データの読み込みと可視化:
    1. メタデータ付きの `Nifti` 画像を読み込み、画像リストを読み込んでスタックする。
    2. IO と transform をキャッシュしてトレーニングと検証を高速化する。
    3. `wandb.Table` と W&B 上のインタラクティブなセグメンテーションオーバーレイでデータを可視化する。
4. 3D `SegResNet` モデルのトレーニング
    1. MONAI の `networks`、`losses`、`metrics` API を使用する。
    2. PyTorch のトレーニングループで 3D `SegResNet` モデルを学習する。
    3. W&B でトレーニングの experiment をトラッキングする。
    4. モデルのチェックポイントを W&B 上で model artifacts としてログし、バージョン管理する。
5. `wandb.Table` と W&B のインタラクティブなセグメンテーションオーバーレイで、検証データセット上の予測を可視化して比較する。

## Setup and Installation

まず、MONAI と W&B の最新バージョンをインストールします。

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

次に、Colab インスタンスを認証して W&B を使えるようにします。

```python
wandb.login()
```

## Initialize a W&B Run

W&B Run を作成して experiment のトラッキングを開始します。適切な config システムを用いることは、機械学習の再現性を高めるための推奨ベストプラクティスです。W&B を使えば、各実験のハイパーパラメーターをトラッキングできます。

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

決定論的なトレーニングを有効化または無効化できるよう、各モジュールの乱数シードも設定します。

```python
set_determinism(seed=config.seed)

# ディレクトリを作成
os.makedirs(config.dataset_dir, exist_ok=True)
os.makedirs(config.checkpoint_dir, exist_ok=True)
```

## Data Loading and Transformation

ここでは、`monai.transforms` API を使って、マルチクラスのラベルをワンホットのマルチラベル セグメンテーションに変換するカスタム transform を作成します。

```python
class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    ラベルを brats のクラスに基づくマルチチャネルに変換します:
    ラベル 1 は腫瘍周辺の浮腫
    ラベル 2 は 造影増強腫瘍
    ラベル 3 は 壊死および非増強腫瘍コア
    利用可能なクラスは TC (Tumor core), WT (Whole tumor),
    ET (Enhancing tumor) です。

    参考: https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # ラベル 2 とラベル 3 を結合して TC を作る
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # ラベル 1, 2, 3 を結合して WT を作る
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

次に、トレーニング用と検証用のデータセットに対する transform をそれぞれ設定します。

```python
train_transform = Compose(
    [
        # 4 つの Nifti 画像を読み込み、スタックする
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

### The Dataset

この experiment で使用するデータセットは http://medicaldecathlon.com/ から取得します。多施設・多モダリティの MRI データ (FLAIR, T1w, T1gd, T2w) を用いて、グリオーマ、壊死/活動性腫瘍、浮腫をセグメンテーションします。データセットは 750 個の 4D ボリューム (トレーニング 484 + テスト 266) で構成されています。

`DecathlonDataset` を使うと、データセットのダウンロードと展開を自動化できます。これは MONAI の `CacheDataset` を継承しており、メモリ容量に応じて学習では `cache_num=N` で `N` 件をキャッシュし、検証ではデフォルト引数で全件をキャッシュする設定が可能です。

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
**Note:** `train_dataset` に `train_transform` を適用する代わりに、学習用・検証用の両方のデータセットに `val_transform` を適用してください。これは、トレーニング前にデータセットの両分割からサンプルを可視化するためです。
{{% /alert %}}

### Visualizing the Dataset

W&B は画像、動画、音声などに対応しています。リッチなメディアをログして結果を探索し、Runs、Models、Datasets を見比べることができます。データボリュームの可視化には [segmentation mask overlay system]({{< relref path="/guides/models/track/log/media/#image-overlays-in-tables" lang="ja" >}}) を使います。[tables]({{< relref path="/guides/models/tables/" lang="ja" >}}) にセグメンテーションマスクをログするには、テーブルの各行に対して `wandb.Image` オブジェクトを用意する必要があります。

以下に擬似コードの例を示します:

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

続いて、サンプル画像とラベル、`wandb.Table` オブジェクト、および関連メタデータを受け取り、W&B のダッシュボードにログするテーブルの行を埋める簡単なユーティリティ関数を書きます。

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

次に、`wandb.Table` オブジェクトを定義し、どの列を持たせるかを指定して、データの可視化で行を埋められるようにします。

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

その後、`train_dataset` と `val_dataset` をそれぞれループして、データサンプルの可視化を生成し、ダッシュボードにログするテーブルの行を埋めます。

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
run.log({"Tumor-Segmentation-Data": table})
```

データは W&B のダッシュボードにインタラクティブな表形式で表示されます。各行で、あるボリュームの特定スライスに対し、各チャネル画像の上に対応するセグメンテーションマスクがオーバーレイされています。テーブルのデータをフィルタし、特定の行にフォーカスするには [Weave queries]({{< relref path="/guides/weave" lang="ja" >}}) を記述できます。

| {{< img src="/images/tutorials/monai/viz-1.gif" alt="ログしたテーブルデータ" >}} | 
|:--:| 
| **ログしたテーブルデータの例。** |

画像を開き、インタラクティブなオーバーレイを使って各セグメンテーションマスクをどのように操作できるかを確認してみましょう。

| {{< img src="/images/tutorials/monai/viz-2.gif" alt="セグメンテーションマップ" >}} | 
|:--:| 
| **セグメンテーションマップの可視化例。* |

{{% alert %}}
**Note:** このデータセットのラベルは、クラス間で重なりのないマスクで構成されています。オーバーレイでは、ラベルを別々のマスクとしてログします。
{{% /alert %}}

### Loading the Data

データセットからデータを読み込むための PyTorch の DataLoader を作成します。DataLoader を作成する前に、学習用の前処理と transform を行うために `train_dataset` の `transform` を `train_transform` に設定します。

```python
# 学習用データセットに train_transforms を適用
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

## Creating the Model, Loss, and Optimizer

このチュートリアルでは、論文 [3D MRI brain tumor segmentation using auto-encoder regularization](https://arxiv.org/pdf/1810.11654.pdf) に基づく `SegResNet` モデルを作成します。`SegResNet` モデルは `monai.networks` API の一部として PyTorch の Module で実装されており、オプティマイザーと学習率スケジューラも使用します。

```python
device = torch.device("cuda:0")

# モデルの作成
model = SegResNet(
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=16,
    in_channels=4,
    out_channels=3,
    dropout_prob=0.2,
).to(device)

# オプティマイザーの作成
optimizer = torch.optim.Adam(
    model.parameters(),
    config.initial_learning_rate,
    weight_decay=config.weight_decay,
)

# 学習率スケジューラの作成
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=config.max_train_epochs
)
```

損失関数は `monai.losses` API のマルチラベル `DiceLoss` を用い、対応する Dice メトリクスは `monai.metrics` API を用いて定義します。

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

# 自動混合精度でトレーニングを高速化
scaler = torch.cuda.amp.GradScaler()
torch.backends.cudnn.benchmark = True
```

混合精度推論のための小さなユーティリティを定義します。これはトレーニングの検証ステップや、学習後にモデルを実行する際に役立ちます。

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

## Training and Validation

トレーニング前に、トレーニングおよび検証の experiment をトラッキングするために、後で `run.log()` でログするメトリクスのプロパティを定義しておきます。

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

### Execute Standard PyTorch Training Loop

```python
with wandb.init(
    project="monai-brain-tumor-segmentation",
    config=config,
    job_type="train",
    reinit=True,
) as run:

    # W&B Artifact オブジェクトを定義
    artifact = wandb.Artifact(
        name=f"{run.id}-checkpoint", type="model"
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
            ## バッチごとのトレーニング損失を W&B にログ
            run.log({"batch/batch_step": batch_step, "batch/train_loss": loss.item()})
            batch_step += 1

        lr_scheduler.step()
        epoch_loss /= total_batch_steps
        ## エポック平均のトレーニング損失と学習率を W&B にログ
        run.log(
            {
                "epoch/epoch_step": epoch,
                "epoch/mean_train_loss": epoch_loss,
                "epoch/learning_rate": lr_scheduler.get_last_lr()[0],
            }
        )
        epoch_progress_bar.set_description(f"Training: train_loss: {epoch_loss:.4f}:")

        # 検証とモデルのチェックポイント保存
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
                
                # W&B artifacts を使ってモデルのチェックポイントをログ & バージョン管理
                artifact.add_file(local_path=checkpoint_path)
                run.log_artifact(artifact, aliases=[f"epoch_{epoch}"])

                # 検証メトリクスを W&B ダッシュボードにログ
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


    # この artifact のログ完了を待機
    artifact.wait()
```

コードに `wandb.log` を組み込むと、トレーニングおよび検証プロセスに関係するあらゆるメトリクスをトラッキングできるだけでなく、W&B のダッシュボードでシステムメトリクス (この例では CPU と GPU) も自動的に記録されます。

| {{< img src="/images/tutorials/monai/viz-3.gif" alt="トレーニングと検証のトラッキング" >}} | 
|:--:| 
| **W&B 上でのトレーニングと検証プロセスのトラッキング例。** |

W&B の Run ダッシュボードで artifacts タブに移動し、トレーニング中にログされたモデルチェックポイント artifacts のさまざまなバージョンに アクセス します。

| {{< img src="/images/tutorials/monai/viz-4.gif" alt="モデルチェックポイントのログ" >}} | 
|:--:| 
| **W&B 上でのモデルチェックポイントのログとバージョン管理の例。** |

## Inference

artifacts インターフェースを使って、どの artifact のバージョンが最良のモデルチェックポイントか (この例ではエポックごとの平均トレーニング損失) を選べます。artifact のリネージ全体を探索し、必要なバージョンを使うこともできます。

| {{< img src="/images/tutorials/monai/viz-5.gif" alt="モデル artifact のトラッキング" >}} | 
|:--:| 
| **W&B 上でのモデル artifact トラッキングの例。** |

エポックごとの平均トレーニング損失が最良のモデル artifact のバージョンを取得し、チェックポイントの state dict をモデルに読み込みます。

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

### Visualizing Predictions and Comparing with the Ground Truth Labels

学習済みモデルの予測を可視化し、対応する正解セグメンテーションマスクと、インタラクティブなセグメンテーションマスクのオーバーレイで比較するためのユーティリティ関数を作成します。

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

予測結果を prediction テーブルにログします。

```python
run = wandb.init(
    project="monai-brain-tumor-segmentation",
    job_type="inference",
    reinit=True,
)
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

    run.log({"Predictions/Tumor-Segmentation-Data": prediction_table})


# 実験を終了
run.finish()
```

インタラクティブなセグメンテーションマスクのオーバーレイを使って、各クラスの予測マスクと正解ラベルを分析・比較します。

| {{< img src="/images/tutorials/monai/viz-6.gif" alt="予測と正解" >}} | 
|:--:| 
| **W&B 上での予測と正解の可視化例。** |

## Acknowledgements and more resources

* [MONAI Tutorial: Brain tumor 3D segmentation with MONAI](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb)
* [WandB Report: Brain Tumor Segmentation using MONAI and WandB](https://wandb.ai/geekyrakshit/brain-tumor-segmentation/reports/Brain-Tumor-Segmentation-using-MONAI-and-WandB---Vmlldzo0MjUzODIw)