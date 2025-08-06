---
title: MONAI を使った 3D 脳腫瘍セグメンテーション
menu:
  tutorials:
    identifier: monai_3d_segmentation
    parent: integration-tutorials
weight: 10
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/monai/3d_brain_tumor_segmentation.ipynb" >}}

このチュートリアルでは、[MONAI](https://github.com/Project-MONAI/MONAI) を用いた多ラベル3D脳腫瘍セグメンテーション課題のトレーニングワークフロー構築方法と、[W&B](https://wandb.ai/site) の実験管理やデータ可視化機能の利用方法を解説します。主な内容は以下の通りです。

1. W&B Run を初期化し、run に関連するすべての設定を同期して再現性を確保
2. MONAI の transform API:
    1. 辞書形式データ用の MONAI Transform
    2. MONAI `transforms` API に沿った新しい transform の定義方法
    3. データ拡張のための強度調整（ランダム化）
3. データの読み込みと可視化:
    1. `Nifti` 画像のメタデータ付き読み込み、画像リストの読み込みおよびスタック
    2. IOキャッシュとtransformの利用によるトレーニング・バリデーションの高速化
    3. `wandb.Table` および W&B 上のインタラクティブなセグメンテーションオーバーレイでのデータ可視化
4. 3D `SegResNet` モデルのトレーニング
    1. MONAI の `networks`, `losses`, `metrics` API の活用
    2. PyTorch トレーニングループでの3D `SegResNet` モデルの学習
    3. W&B での実験トラッキング
    4. モデルチェックポイントを model artifact として W&B にログ & バージョン管理
5. W&B の `wandb.Table` とインタラクティブなオーバーレイを使ったバリデーションデータセット上での予測結果の可視化・比較

## セットアップとインストール

最初に、MONAI と W&B の最新版をインストールします。

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

続いて、Colab インスタンスを W&B に認証します。

```python
wandb.login()
```

## W&B Run の初期化

W&B Run を新たにスタートし、実験の管理を開始します。適切な config システムの活用は機械学習の再現性確保において推奨されます。W&B なら各実験のハイパーパラメーターもトラッキング可能です。

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

ランダムシードも設定して、決定論的なトレーニングを有効または無効にできます。

```python
set_determinism(seed=config.seed)

# ディレクトリ作成
os.makedirs(config.dataset_dir, exist_ok=True)
os.makedirs(config.checkpoint_dir, exist_ok=True)
```

## データの読み込みと変換

ここでは `monai.transforms` API を使い、複数クラスのラベルを one-hot 形式の多ラベルセグメンテーションタスクに変換する独自 transform を作成します。

```python
class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Brats クラスに基づきラベルをマルチチャンネルに変換:
    ラベル 1: 周辺浮腫
    ラベル 2: 増強腫瘍
    ラベル 3: 壊死/非増強腫瘍核
    クラスは TC (腫瘍核), WT (全腫瘍), ET (増強腫瘍)

    参考: https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # ラベル2と3をまとめてTCを構築
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # ラベル1,2,3をまとめてWTを構築
            result.append(
                torch.logical_or(
                    torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                )
            )
            # ラベル2がET
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d
```

次に、トレーニングデータ・バリデーションデータセット用にtransformをそれぞれ定義します。

```python
train_transform = Compose(
    [
        # 4つのNifti画像をロードしてスタック
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

本実験で使用するデータセットは http://medicaldecathlon.com/ から取得します。複数モダリティ(MRI: FLAIR, T1w, T1gd, T2w)・複数施設に跨るデータで、グリオーマ、壊死・活動性腫瘍、浮腫のセグメンテーション用です。750個の4Dボリューム (484 Training、266 Testing) が含まれています。

`DecathlonDataset` を使えば、自動的にデータセットをダウンロード・展開可能です。`CacheDataset` を継承しているため、`cache_num=N` でトレーニング時のN件キャッシュや、バリデーション時はデフォルトで全件キャッシュできます（メモリ搭載量に応じて調整）。

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
**Note:** `train_transform` を `train_dataset` に適用する代わりに `val_transform` をトレーニング・バリデーション両方のデータセットに適用してください。なぜなら、学習開始前に両データセットからサンプルを可視化するためです。
{{% /alert %}}

### データセットの可視化

W&B では画像・動画・音声など様々なリッチメディアのログに対応しています。結果を視覚的に比較し、run・model・dataset ごとに詳細な分析が可能です。[セグメンテーションマスクオーバーレイシステム]({{< relref "/guides/models/track/log/media/#image-overlays-in-tables" >}}) を活用すると、データボリュームも可視化できます。[tables]({{< relref "/guides/models/tables/" >}}) にセグメンテーションマスクを記録するには、各行に対して `wandb.Image` オブジェクトを渡してください。

以下はサンプルコードです。

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

次に、サンプル画像・ラベル・`wandb.Table` オブジェクトおよび付随するメタデータを受け取り、W&B ダッシュボード用テーブルの各行を生成するユーティリティ関数を作成します。

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

続いて、`wandb.Table` オブジェクトを定義し、可視化するデータに対応するカラムを指定します。

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

それから、それぞれ `train_dataset` と `val_dataset` をループしデータサンプルの可視化画像を生成、テーブルに行を追加してダッシュボードに記録します。

```python
# train_dataset の可視化画像を生成
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

# val_dataset の可視化画像を生成
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

# テーブルをダッシュボードへログ
run.log({"Tumor-Segmentation-Data": table})
```

このデータは W&B ダッシュボード上のインタラクティブなテーブルフォーマットで閲覧できます。あるデータボリュームのスライスごとに各チャネルの画像と対応するセグメンテーションマスクが各行で重ねて表示されます。[Weave クエリ]({{< relref "/guides/weave" >}}) を使えば、行ごとにフィルタリングも可能です。

| {{< img src="/images/tutorials/monai/viz-1.gif" alt="Logged table data" >}} | 
|:--:| 
| **ログされた table データの例。** |

画像を開くと、各セグメンテーションマスクをインタラクティブに切り替えて確認できます。

| {{< img src="/images/tutorials/monai/viz-2.gif" alt="Segmentation maps" >}} | 
|:--:| 
| **セグメンテーションマップ可視化の例** |

{{% alert %}}
**Note:** このデータセットのラベルは各クラスが重複しないマスクとなっています。オーバーレイでは各クラスのラベルを個別マスクとして記録しています。
{{% /alert %}}

### データのロード

データセットからデータをロードする PyTorch DataLoader を作成します。DataLoader 作成前に `train_dataset` の `transform` を `train_transform` に切り替え、トレーニング用に前処理・変換を行います。

```python
# train_transforms をトレーニングデータセットへ適用
train_dataset.transform = train_transform

# train_loader の作成
train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers,
)

# val_loader の作成
val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers,
)
```

## モデル・損失関数・オプティマイザーの作成

本チュートリアルでは論文 [3D MRI brain tumor segmentation using auto-encoder regularization](https://arxiv.org/pdf/1810.11654.pdf) を参考に `SegResNet` モデルを作成します。`SegResNet` モデルは `monai.networks` API にて PyTorch Module として実装されており、オプティマイザーと学習率スケジューラも設定できます。

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

損失関数はマルチラベル用の `DiceLoss` を `monai.losses` API で、対応するダイスメトリクスは `monai.metrics` API で作成します。

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

# 自動混合精度の活用で学習を高速化
scaler = torch.cuda.amp.GradScaler()
torch.backends.cudnn.benchmark = True
```

バリデーション工程や学習後の推論で使える混合精度推論用のユーティリティ関数を定義します。

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

## トレーニングとバリデーション

トレーニング前に、`run.log()` でロギングするメトリクスのプロパティを定義します。これにより、トレーニング・バリデーション工程のモニタリングが容易になります。

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

### PyTorch 標準トレーニングループの実行

```python
with wandb.init(
    project="monai-brain-tumor-segmentation",
    config=config,
    job_type="train",
    reinit=True,
) as run:

    # W&B Artifact オブジェクトの定義
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
            ## W&Bへバッチ単位の学習損失を記録
            run.log({"batch/batch_step": batch_step, "batch/train_loss": loss.item()})
            batch_step += 1

        lr_scheduler.step()
        epoch_loss /= total_batch_steps
        ## W&Bへエポック単位の学習損失・学習率を記録
        run.log(
            {
                "epoch/epoch_step": epoch,
                "epoch/mean_train_loss": epoch_loss,
                "epoch/learning_rate": lr_scheduler.get_last_lr()[0],
            }
        )
        epoch_progress_bar.set_description(f"Training: train_loss: {epoch_loss:.4f}:")

        # バリデーションおよびモデルチェックポイント
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
                
                # W&B Artifactsでモデルチェックポイントをログ＆バージョン管理
                artifact.add_file(local_path=checkpoint_path)
                run.log_artifact(artifact, aliases=[f"epoch_{epoch}"])

                # バリデーションメトリクスをW&Bダッシュボードに記録
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


    # アーティファクトのロギング完了待ち
    artifact.wait()
```

`wandb.log` を活用することで、トレーニングおよびバリデーション過程で発生する全メトリクスはもちろん、システムメトリクス（CPUやGPU使用率等）もW&Bダッシュボード上で一元管理できます。

| {{< img src="/images/tutorials/monai/viz-3.gif" alt="Training and validation tracking" >}} | 
|:--:| 
| **W&Bでのトレーニング・バリデーション過程トラッキング例。** |

W&B run の artifacts タブから、トレーニング中に記録された各モデルチェックポイントのバージョン違いにアクセスできます。

| {{< img src="/images/tutorials/monai/viz-4.gif" alt="Model checkpoints logging" >}} | 
|:--:| 
| **W&Bにおけるモデルチェックポイントのログ・バージョン管理例。** |

## 推論

Artifacts インターフェースを利用して、訓練時に記録されたモデルチェックポイントから最適なバージョン（例えば、エポックごとの平均トレーニング損失が最小のもの）を選択できます。アーティファクト全体のリネージも可視化可能で、必要なバージョンを取得できます。

| {{< img src="/images/tutorials/monai/viz-5.gif" alt="Model artifact tracking" >}} | 
|:--:| 
| **W&Bでのモデル artifact トラッキング例。** |

エポックごとの平均トレーニング損失が最良のモデルartifactバージョンを取得し、そのチェックポイントの state dict をモデルに読み込みます。

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

### 推論結果の可視化と正解ラベルとの比較

学習済みモデルの予測値を可視化し、対応する正解セグメンテーションマスクと比較するためのユーティリティ関数を作成します。インタラクティブなオーバーレイを使ってマスクを比較しましょう。

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

推論結果を prediction table にログします。

```python
run = wandb.init(
    project="monai-brain-tumor-segmentation",
    job_type="inference",
    reinit=True,
)
# prediction table の作成
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

インタラクティブなセグメンテーションマスクオーバーレイを用いることで、各クラスごとに予測結果と正解ラベルを詳細に比較できます。

| {{< img src="/images/tutorials/monai/viz-6.gif" alt="Predictions and ground-truth" >}} | 
|:--:| 
| **W&Bの予測結果＆正解ラベル可視化例** |

## 謝辞・参考リソース

* [MONAI チュートリアル: Brain tumor 3D segmentation with MONAI](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb)
* [WandB Report: Brain Tumor Segmentation using MONAI and WandB](https://wandb.ai/geekyrakshit/brain-tumor-segmentation/reports/Brain-Tumor-Segmentation-using-MONAI-and-WandB---Vmlldzo0MjUzODIw)
