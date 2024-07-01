# PyTorch Lightning

[**Try in a Colab Notebook here →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Image_Classification_using_PyTorch_Lightning.ipynb)

PyTorch Lightningを使用して画像分類パイプラインを構築します。コードの可読性と再現性を高めるために、この[スタイルガイド](https://lightning.ai/docs/pytorch/stable/starter/style_guide.html)に従います。このガイドの素晴らしい説明は[こちら](https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY)で確認できます。

## PyTorch LightningとW&Bのセットアップ

このチュートリアルでは、PyTorch Lightning（明らかですね！）とWeights and Biasesが必要です。

```
!pip install lightning -q
# weights and biasesをインストール
!pip install wandb -qU
```

以下のインポートが必要です。

```
import lightning.pytorch as pl
# お気に入りの機械学習トラッキングツール
from lightning.pytorch.loggers import WandbLogger

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader

from torchmetrics import Accuracy

from torchvision import transforms
from torchvision.datasets import CIFAR10

import wandb
```

次に、wandbアカウントにログインする必要があります。

```
wandb.login()
```

## 🔧 DataModule - 私たちにふさわしいデータパイプライン

DataModuleは、データ関連のフックをLightningModuleから分離する方法です。これにより、データセットに依存しないモデルを開発できます。

データパイプラインを1つの共有可能で再利用可能なクラスに整理します。datamoduleは、PyTorchでのデータ処理に関する5つのステップをカプセル化します：
- ダウンロード / トークン化 / 処理。
- クリーンアップと（必要なら）ディスクへの保存。
- Dataset内で読み込み。
- 変換を適用（回転、トークン化など）。
- DataLoader内にラップ。

datamoduleについての詳細は[こちら](https://lightning.ai/docs/pytorch/stable/data/datamodule.html)で確認できます。では、Cifar-10データセットのdatamoduleを構築しましょう。

```
class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir: str = './'):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.num_classes = 10
    
    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)
    
    def setup(self, stage=None):
        # DataLoaderで使用するtrain/valデータセットを割り当て
        if stage == 'fit' or stage is None:
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])

        # DataLoaderで使用するテストデータセットを割り当て
        if stage == 'test' or stage is None:
            self.cifar_test = CIFAR10(self.data_dir, train=False, transform=self.transform)
    
    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size)
```

## 📱 Callbacks

コールバックは、プロジェクト間で再利用可能な自己完結型のプログラムです。PyTorch Lightningにはいくつかの[組み込みコールバック](https://lightning.ai/docs/pytorch/latest/extensions/callbacks.html#built-in-callbacks)があり、これはよく使用されます。
PyTorch Lightningのコールバックについての詳細は[こちら](https://lightning.ai/docs/pytorch/latest/extensions/callbacks.html)で確認できます。

### 組み込みコールバック

このチュートリアルでは、[Early Stopping](https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.callbacks.EarlyStopping.html#lightning.callbacks.EarlyStopping)と[Model Checkpoint](https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint)の組み込みコールバックを使用します。これらは`Trainer`に渡すことができます。

### カスタムコールバック
カスタムKerasのコールバックに慣れている方は、PyTorchパイプラインでも同様のことができる能力が付加価値となるでしょう。

画像分類を実行するため、特定のサンプル画像に対するモデルの予測を視覚化する能力は役立ちます。これをコールバックの形式で行うことは、モデルを早期段階でデバッグするのに有効です。

```
class ImagePredictionLogger(pl.callbacks.Callback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # テンソルをCPUに移動
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # モデルの予測を取得
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        # 画像をwandb Imageとしてログ
        trainer.logger.experiment.log({
            "examples":[wandb.Image(x, caption=f"Pred:{pred}, Label:{y}") 
                           for x, pred, y in zip(val_imgs[:self.num_samples], 
                                                 preds[:self.num_samples], 
                                                 val_labels[:self.num_samples])]
            })
        
```

## 🎺 LightningModule - システムを定義する

LightningModuleはシステムを定義するのであり、モデルではありません。ここでのシステムは、すべての研究コードを1つのクラスにまとめて自己完結型にします。`LightningModule`はPyTorchコードを5つのセクションに整理します：
- 計算（`__init__`）。
- トレーニングループ（`training_step`）
- 検証ループ（`validation_step`）
- テストループ（`test_step`）
- オプティマイザー（`configure_optimizers`）

これにより、簡単に共有可能なデータセットに依存しないモデルを作成できます。それでは、Cifar-10分類のシステムを構築しましょう。

```
class LitModel(pl.LightningModule):
    def __init__(self, input_shape, num_classes, learning_rate=2e-4):
        super().__init__()
        
        # ハイパーパラメーターをログ
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1)

        self.pool1 = torch.nn.MaxPool2d(2)
        self.pool2 = torch.nn.MaxPool2d(2)
        
        n_sizes = self._get_conv_output(input_shape)

        self.fc1 = nn.Linear(n_sizes, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)

    # 畳み込みブロックからLinear層に移動する出力テンソルのサイズを返します。
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._forward_features(input) 
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size
        
    # 畳み込みブロックからの特徴テンソルを返します。
    def _forward_features(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        return x
    
    # 推論時に使用されます
    def forward(self, x):
       x = self._forward_features(x)
       x = x.view(x.size(0), -1)
       x = F.relu(self.fc1(x))
       x = F.relu(self.fc2(x))
       x = F.log_softmax(self.fc3(x), dim=1)
       
       return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        
        # トレーニングメトリクス
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # 検証メトリクス
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        
        # テストメトリクス
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

```

## 🚋 トレーニングと評価

`DataModule`を使用してデータパイプラインを、`LightningModule`を使用してモデルアーキテクチャ＋トレーニングループを整理したので、PyTorch Lightning `Trainer`はその他すべてを自動化してくれます。

Trainerは以下を自動化します：
- エポックとバッチの反復
- `optimizer.step()`, `backward`, `zero_grad()` の呼び出し
- `.eval()` の呼び出し、grads の有効化/無効化
- 重みの保存と読み込み
- Weights and Biasesのログ
- マルチGPUトレーニングのサポート
- TPUのサポート
- 16ビットトレーニングのサポート

```
dm = CIFAR10DataModule(batch_size=32)
# x_dataloaderにアクセスするには、prepare_dataとsetupを呼び出す必要があります。
dm.prepare_data()
dm.setup()

# 画像予測をログするためにカスタムImagePredictionLoggerコールバックが必要とするサンプル
val_samples = next(iter(dm.val_dataloader()))
val_imgs, val_labels = val_samples[0], val_samples[1]
val_imgs.shape, val_labels.shape
```

```
model = LitModel((3, 32, 32), dm.num_classes)

# wandbロガーの初期化
wandb_logger = WandbLogger(project='wandb-lightning', job_type='train')

# コールバックの初期化
early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss")
checkpoint_callback = pl.callbacks.ModelCheckpoint()

# トレーナーの初期化
trainer = pl.Trainer(max_epochs=2,
                     logger=wandb_logger,
                     callbacks=[early_stop_callback,
                                ImagePredictionLogger(val_samples),
                                checkpoint_callback],
                     )

# モデルをトレーニングする ⚡🚅⚡
trainer.fit(model, dm)

# ホールドアウトテストセットでモデルを評価 ⚡⚡
trainer.test(dataloaders=dm.test_dataloader())

# wandbランを終了
wandb.finish()
```

## 最終的な考え
私はTensorFlow/Kerasエコシステムから来ており、PyTorchはエレガントなフレームワークでありながらやや圧倒されると感じています。ただし、個人的な経験です。PyTorch Lightningを探求している中で、私をPyTorchから遠ざけていたほぼすべての理由が解決されていることに気付きました。ここに私の興奮をまとめたものを紹介します：
- 以前: 従来のPyTorchモデルの定義はあちこちに分散していました。`model.py`スクリプトにモデルがあり、トレーニングループは`train.py`ファイルにありました。パイプラインを理解するためにかなり往復する必要がありました。
- 今: `LightningModule`はシステムとして機能し、モデルとともに`training_step`や`validation_step`などが定義されています。これにより、モジュール化され、共有可能です。
- 以前: TensorFlow/Kerasの最高の部分は、入力データパイプラインです。彼らのデータセットカタログは豊富で成長しています。PyTorchのデータパイプラインは以前は最も苦痛なポイントでした。通常のPyTorchコードでは、データのダウンロード、クリーニング、準備が多くのファイルに散らばっていました。
- 今: DataModuleはデータパイプラインを1つの共有可能で再利用可能なクラスに整理します。これは、`train_dataloader`、`val_dataloader`、`test_dataloader`、および対応する変換とデータ処理/ダウンロードステップのコレクションです。
- 以前: Kerasでは`model.fit`を呼び出してモデルをトレーニングし、`model.predict`で推論を実行し、`model.evaluate`でテストデータに対する評価が提供されました。PyTorchではこれは当てはまりません。通常は別々の`train.py`と`test.py`ファイルが見つかるでしょう。
- 今: `LightningModule`を使うことで、`Trainer`がすべてを自動化します。`trainer.fit`と`trainer.test`を呼び出すだけで、モデルのトレーニングと評価ができます。
- 以前: TensorFlowはTPUを愛していますが、PyTorchは…。
- 今: PyTorch Lightningを使うことで、同じモデルを複数のGPUやTPUでトレーニングするのが非常に簡単です。すごい！
- 以前: 私はコールバックの大ファンであり、カスタムコールバックを書くのが好きです。Early Stoppingのような些細なことですら従来のPyTorchでは議論の対象でした。
- 今: PyTorch Lightningを使うことで、Early StoppingとModel Checkpointingが簡単に行えます。カスタムコールバックも書けます。

## 🎨 結論とリソース

このレポートが役立つことを願っています。コードをいじって、お好きなデータセットで画像分類器をトレーニングすることをお勧めします。

