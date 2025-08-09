---
title: MONAI を使用した 3D 脳腫瘍セグメンテーション
menu:
  tutorials:
    identifier: ja-tutorials-integration-tutorials-monai_3d_segmentation
    parent: integration-tutorials
weight: 10
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/monai/3d_brain_tumor_segmentation.ipynb" >}}

このチュートリアルでは、[MONAI](https://github.com/Project-MONAI/MONAI) を活用して、多ラベル3D脳腫瘍セグメンテーションタスクのトレーニングワークフローを構築し、[W&B](https://wandb.ai/site) の実験管理やデータ可視化機能を利用する方法を解説します。チュートリアルでは、以下の内容を扱います。

1. W&B Run を初期化し、再現性のために run に紐づく全ての config を同期します。
2. MONAI の transform API の利用例:
   1. 辞書形式データの MONAI Transforms 利用例
   2. MONAI の `transforms` API に従った新しい transform の定義方法
   3. データ拡張のため、強度をランダムに調整する方法
3. データのロードと可視化:
   1. `Nifti` 画像とメタデータの読み込み、複数画像のロードとスタック
   2. IOとtransformのキャッシュによるトレーニング・検証の高速化
   3. `wandb.Table` と W&B のインタラクティブなセグメンテーションオーバーレイによるデータ可視化
4. 3D `SegResNet` モデルのトレーニング
   1. MONAI の `networks`, `losses`, `metrics` API 利用例
   2. PyTorch トレーニングループによる 3D `SegResNet` モデルの学習
   3. W&B を活用したトレーニング実験のトラッキング
   4. モデルチェックポイントを W&B の model artifact として記録・バージョン管理
5. `wandb.Table` とインタラクティブなセグメンテーションオーバーレイを用いた検証用データセットでの予測可視化と比較

## セットアップとインストール

まず、MONAI と W&B の最新版をインストールします。

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

続いて、Colab インスタンスから W&B の利用認証を行います。

```python
wandb.login()
```

## W&B Run の初期化

新しい W&B Run を開始し、実験のトラッキングを開始します。再現性のある機械学習には適切な config システムの利用が推奨されています。W&B を使えば、各実験ごとのハイパーパラメーターも記録できます。

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

また、ランダムシードを固定し、モジュールの決定論的な挙動を有効・無効化できます。

```python
set_determinism(seed=config.seed)

# ディレクトリの作成
os.makedirs(config.dataset_dir, exist_ok=True)
os.makedirs(config.checkpoint_dir, exist_ok=True)
```

## データのロードと変換

ここでは、`monai.transforms` API を使い、マルチクラスラベルを one-hot 形式のマルチラベルセグメンテーションタスクへ変換するカスタム transform を作成します。

```python
class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    brats クラスに基づいてラベルをマルチチャンネル化します:
    label 1: 浮腫（peritumoral edema）
    label 2: 増強腫瘍（GD-enhancing tumor）
    label 3: 壊死/非増強腫瘍中核（necrotic/non-enhancing tumor core）
    TC（腫瘍コア）・WT（全腫瘍）・ET（増強腫瘍）クラスを構成可能

    参考: https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # ラベル2とラベル3を結合してTCを作成
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # ラベル1,2,3を結合してWTを作成
            result.append(
                torch.logical_or(
                    torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                )
            )
            # ラベル2はET
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d
```

次に、トレーニング用・検証用データセットそれぞれに適用する transform を定義します。

```python
train_transform = Compose(
    [
        # 4つの Nifti 画像を読み込み、結合
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

### データセットについて

この実験で使用するデータセットは http://medicaldecathlon.com/ から取得できます。複数モダリティ・複数施設のMRIデータ（FLAIR, T1w, T1gd, T2w）を用いてグリオーマの腫瘍・壊死部・浮腫領域をセグメント化します。データセットは 750 の4Dボリューム（学習用484 + テスト用266）です。

`DecathlonDataset` を用いることで、データセットを自動でダウンロード・展開できます。これは MONAI の `CacheDataset` を継承しているため、`cache_num=N` でN個をキャッシュさせたり、検証用にはデフォルト引数を利用し全部キャッシュするなど、メモリ状況に応じて制御できます。

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
**注意:** `train_transform` を `train_dataset` に適用するのではなく、トレーニングと検証どちらのデータセットにも `val_transform` を使ってください。これは、実際のトレーニングに入る前に双方のスプリットデータのサンプルを可視化するためです。
{{% /alert %}}

### データセットの可視化

W&B では画像、動画、音声など様々なリッチメディアで記録できます。これにより、結果の探索やrun・モデル・データセット間の視覚的な比較が可能です。[セグメンテーションマスクのオーバーレイ機能]({{< relref path="/guides/models/track/log/media/#image-overlays-in-tables" lang="ja" >}}) を用いて、データボリュームを直接可視化できます。[テーブル]({{< relref path="/guides/models/tables/" lang="ja" >}})でセグメンテーションマスクを記録する場合、各行ごとに `wandb.Image` オブジェクトを作成してください。

以下は疑似コードの例です。

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

次に、サンプル画像とラベル、`wandb.Table` オブジェクト、および関連メタデータを受け取り、W&B ダッシュボードに記録するテーブルの行を埋めるユーティリティ関数を定義します。

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

次に、`wandb.Table` オブジェクトと、そのカラム定義を記述します。これで可視化データをテーブルへ追加できます。

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

続いて、`train_dataset` と `val_dataset` のサンプル画像群をループし、可視化を生成し、ダッシュボードへ記録するテーブルへ行を追加します。

```python
# train_dataset 用の可視化を生成
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

# val_dataset 用の可視化を生成
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

# テーブルをダッシュボードに記録
run.log({"Tumor-Segmentation-Data": table})
```

このデータは W&B ダッシュボード上でインタラクティブな表形式で表示されます。各行ごとに各チャネルの特定スライス画像と、そのセグメンテーションマスクがオーバーレイされています。テーブルデータを [Weave クエリ]({{< relref path="/guides/weave" lang="ja" >}}) でフィルタして、1行だけに着目することも可能です。

| {{< img src="/images/tutorials/monai/viz-1.gif" alt="Logged table data" >}} | 
|:--:| 
| **記録されたテーブルデータの一例** |

画像を開くと、各セグメンテーションマスクをインタラクティブなオーバーレイで操作できます。

| {{< img src="/images/tutorials/monai/viz-2.gif" alt="Segmentation maps" >}} | 
|:--:| 
| **可視化されたセグメンテーションマップの例** |

{{% alert %}}
**注意:** データセット内のラベルはクラス間で重なりがないようになっています。オーバーレイでは各クラスのマスクが個別マスクとして記録されます。
{{% /alert %}}

### データのロード

データロード用の PyTorch DataLoader を作成します。事前に `train_dataset` の transform を `train_transform` に切り替えて、トレーニング用の前処理・変換を適用しておきます。

```python
# train_transforms を学習データセットに適用
train_dataset.transform = train_transform

# train_loader 作成
train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers,
)

# val_loader 作成
val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers,
)
```

## モデル、損失関数、オプティマイザーの作成

本チュートリアルでは論文 [3D MRI brain tumor segmentation using auto-encoder regularization](https://arxiv.org/pdf/1810.11654.pdf) を元にした `SegResNet` モデルを構築します。`SegResNet` モデルは `monai.networks` API で提供されている PyTorch モジュールで、オプティマイザーや学習率スケジューラーも一緒に定義します。

```python
device = torch.device("cuda:0")

# モデル作成
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

# 学習率スケジューラー
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=config.max_train_epochs
)
```

損失関数には `monai.losses` API の multi-label `DiceLoss`、評価メトリクスには `monai.metrics` API の各種 dice metrics を定義します。

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

# 自動混合精度による高速化
scaler = torch.cuda.amp.GradScaler()
torch.backends.cudnn.benchmark = True
```

混合精度推論用の小さなユーティリティも定義します。これは検証ステップや学習後の推論で利用します。

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

トレーニング前に、`run.log()` と連携させるための各種 metric を定義し、トレーニング・検証過程をトラッキングできるようにします。

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

### 標準的な PyTorch トレーニングループの実行

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
            ## バッチごとのトレーニング損失を W&B へ記録
            run.log({"batch/batch_step": batch_step, "batch/train_loss": loss.item()})
            batch_step += 1

        lr_scheduler.step()
        epoch_loss /= total_batch_steps
        ## エポックごとのトレーニング損失と学習率を W&B へ記録
        run.log(
            {
                "epoch/epoch_step": epoch,
                "epoch/mean_train_loss": epoch_loss,
                "epoch/learning_rate": lr_scheduler.get_last_lr()[0],
            }
        )
        epoch_progress_bar.set_description(f"Training: train_loss: {epoch_loss:.4f}:")

        # 検証・モデルチェックポイント保存ステップ
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
                
                # モデルチェックポイントを W&B artifact で記録・バージョン管理
                artifact.add_file(local_path=checkpoint_path)
                run.log_artifact(artifact, aliases=[f"epoch_{epoch}"])

                # 検証メトリクスを W&B ダッシュボードへ記録
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


    # artifact 記録完了まで待機
    artifact.wait()
```

`wandb.log` の活用で、トレーニング・検証進行中の全メトリクスだけでなく、システムメトリクス（ここでは CPU や GPU 情報）も W&B ダッシュボードに自動記録されます。

| {{< img src="/images/tutorials/monai/viz-3.gif" alt="Training and validation tracking" >}} | 
|:--:| 
| **トレーニング・検証プロセスの W&B トラッキング例** |

W&B Run ダッシュボードの Artifacts タブから、トレーニング中に記録された各モデルチェックポイントのバージョン管理一覧も閲覧できます。

| {{< img src="/images/tutorials/monai/viz-4.gif" alt="Model checkpoints logging" >}} | 
|:--:| 
| **W&B におけるモデルチェックポイントのロギングとバージョン管理例** |

## 推論

Artifacts インターフェースから、最も性能の良かったモデル（ここでは平均エポック毎トレーニング損失が最小値のもの）を手軽に選択可能です。artifact の全リネージも探索でき、必要なバージョンを利用できます。

| {{< img src="/images/tutorials/monai/viz-5.gif" alt="Model artifact tracking" >}} | 
|:--:| 
| **W&B 上のモデル artifact トラッキング例** |

最も良い平均エポック損失で記録されたモデル artifact バージョンを取得し、チェックポイントファイルをモデルにロードします。

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

### 予測結果の可視化・正解ラベルとの比較

学習済みモデルの予測結果を可視化し、該当する正解セグメンテーションマスクとインタラクティブに比較するユーティリティ関数も用意します。

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

予測テーブルに予測結果を記録します。

```python
run = wandb.init(
    project="monai-brain-tumor-segmentation",
    job_type="inference",
    reinit=True,
)
# 予測用テーブルを作成
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


# 実験終了
run.finish()
```

インタラクティブなセグメンテーションマスクオーバーレイを活用して、各クラスごとに予測マスクと正解ラベルを分析・比較できます。

| {{< img src="/images/tutorials/monai/viz-6.gif" alt="Predictions and ground-truth" >}} | 
|:--:| 
| **W&B 上での予測結果と正解ラベルの可視化例** |

## 謝辞・参考リソース

* [MONAI チュートリアル: MONAI での脳腫瘍3Dセグメンテーション](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb)
* [WandB レポート: Brain Tumor Segmentation using MONAI and WandB](https://wandb.ai/geekyrakshit/brain-tumor-segmentation/reports/Brain-Tumor-Segmentation-using-MONAI-and-WandB---Vmlldzo0MjUzODIw)
