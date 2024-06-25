
# PyTorch Lightning

[**Try in a Colab Notebook here →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Image_Classification_using_PyTorch_Lightning.ipynb)

PyTorch Lightningを使用して画像分類パイプラインを構築します。この[スタイルガイド](https://lightning.ai/docs/pytorch/stable/starter/style_guide.html)に従って、コードの可読性と再現性を向上させます。詳細な説明は[こちら](https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY)で確認できます。

## PyTorch LightningとW&Bの設定

このチュートリアルでは、PyTorch Lightning（明らかですよね！）とWeights and Biasesが必要です。

```
!pip install lightning -q
# weights and biasesのインストール
!pip install wandb -qU
```

これらのインポートが必要です。

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

## 🔧 DataModule - 必要なデータパイプライン

DataModulesは、LightningModuleからデータ関連のフックを分離する方法であり、データセットに依存しないモデルを開発できます。

データパイプラインを共有可能で再利用可能な1つのクラスにまとめます。datamoduleは、PyTorchにおけるデータプロセッシングに関わる5つのステップをカプセル化します：
- ダウンロード / トークナイズ / プロセス。
- クリーンアップ（場合によってはディスクに保存）。
- Dataset内にロード。
- 変換を適用（回転、トークナイズなど）。
- DataLoader内にラップ。

datamodulesについて詳しくは[こちら](https://lightning.ai/docs/pytorch/stable/data/datamodule.html)をご覧ください。Cifar-10 Datasetのためにdatamoduleを構築しましょう。

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
        # dataloaderで使用するためのtrain/valデータセットを割り当て
        if stage == 'fit' or stage is None:
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])

        # dataloaderで使用するためのtestデータセットを割り当て
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

コールバックは、複数のプロジェクトで再利用可能な自己完結型のプログラムです。PyTorch Lightningにはいくつかの[組み込みのコールバック](https://lightning.ai/docs/pytorch/latest/extensions/callbacks.html#built-in-callbacks)が用意されています。PyTorch Lightningのコールバックについて詳しくは[こちら](https://lightning.ai/docs/pytorch/latest/extensions/callbacks.html)をご覧ください。

### 組み込みのコールバック

このチュートリアルでは、[Early Stopping](https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.callbacks.EarlyStopping.html#lightning.callbacks.EarlyStopping)と[Model Checkpoint](https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint)の組み込みのコールバックを使用します。これらは`Trainer`に渡すことができます。

### カスタムコールバック

Custom Kerasのコールバックに慣れているなら、PyTorchパイプラインでも同様のことができるのはうれしい驚きです。

画像分類を行うため、画像のいくつかのサンプルでモデルの予測を視覚化する能力が役立ちます。これをコールバックの形で提供することで、モデルの初期段階でデバッグを行いやすくなります。

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

## 🎺 LightningModule - システムの定義

LightningModuleはシステムを定義し、モデルを定義しません。ここでのシステムは、すべての研究コードを1つのクラスにまとめて自己完結型にします。`LightningModule`はPyTorchコードを5つのセクションに整理します：
- 計算（`__init__`）。
- トレーニングループ（`training_step`）
- 検証ループ（`validation_step`）
- テストループ（`test_step`）
- オプティマイザー（`configure_optimizers`）

これにより、簡単に共有できるデータセットに依存しないモデルを構築できます。Cifar-10分類のためのシステムを構築しましょう。

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

    # convブロックからLinear層に入る出力テンソルのサイズを返す
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._forward_features(input) 
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size
        
    # convブロックからの特徴テンソルを返す
    def _forward_features(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        return x
    
    # 推論時に使用される
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
        
        # 検証メトリクス
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

データパイプラインを`DataModule`で整理し、モデルアーキテクチャ＋トレーニングループを`LightningModule`で整理したので、PyTorch Lightningの`Trainer`が残りのすべてを自動化してくれます。

Trainerは以下を自動化します：
- エポックとバッチの繰り返し
- `optimizer.step()`、`backward`、`zero_grad`の呼び出し
- `.eval()`の呼び出し、グラデーションの有効化/無効化
- 重みの保存と読み込み
- Weights and Biasesのログ
- マルチGPUトレーニングのサポート
- TPUサポート
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

# Trainerの初期化
trainer = pl.Trainer(max_epochs=2,
                     logger=wandb_logger,
                     callbacks=[early_stop_callback,
                                ImagePredictionLogger(val_samples),
                                checkpoint_callback],
                     )

# モデルのトレーニング ⚡🚅⚡
trainer.fit(model, dm)

# ホールドアウトテストセットでのモデルの評価 ⚡⚡
trainer.test(dataloaders=dm.test_dataloader())

# wandb runの終了
wandb.finish()
```

## 最終的な考察

私はTensorFlow/Kerasのエコシステムから来ましたが、PyTorchは優れたフレームワークでありながら少し圧倒される存在でした。しかし、PyTorch Lightningを探索する中で、PyTorchから避けていた理由のほとんどが解消されたことに気付きました。以下は私の興奮の簡単な要約です：
- 以前：従来のPyTorchモデルの定義は散らばっていました。モデルは`model.py`スクリプトに記載され、トレーニングループは`train.py`ファイルに記載されていました。パイプラインを理解するために多くの見直しが必要でした。
- 今：`LightningModule`はモデルを`training_step`、`validation_step`などと一緒に定義するシステムとして機能します。これでモジュール化され、共有が簡単になりました。
- 以前：TensorFlow/Kerasの最大の利点は入力データパイプラインです。彼らのデータセットカタログは豊富で成長を続けています。従来のPyTorchコードでは、データのダウンロード/クリーンアップ/準備は通常多くのファイルに分散していました。
- 今：DataModuleはデータパイプラインを1つの共有可能で再利用可能なクラスに整理します。`train_dataloader`、`val_dataloader`、`test_dataloader`に対応する変換とデータプロセッシング/ダウンロードステップが簡単にまとめられています。
- 以前：Kerasでは`model.fit`を呼び出してモデルをトレーニングし、`model.predict`を使用して推論を実行できました。`model.evaluate`はテストデータのシンプルな評価を提供していましたが、これはPyTorchではそうではありませんでした。通常、`train.py`と`test.py`ファイルが別々に存在します。
- 今：`LightningModule`があるため、`Trainer`がすべてを自動化します。`trainer.fit`と`trainer.test`を呼び出すだけでモデルをトレーニングおよび評価できます。
- 以前：TensorFlowはTPUを愛し、PyTorchは...まあ！ 
- 今：PyTorch Lightningを使えば、同じモデルを複数のGPUやTPUで簡単にトレーニングできます。素晴らしい！
- 以前：私はコールバックの大ファンでカスタムコールバックを書くことを好みます。Early Stoppingのような簡単なものでさえ、従来のPyTorchでは議論の的になっていました。
- 今：PyTorch Lightningでは、Early StoppingとModel Checkpointingの使用がとても簡単です。カスタムコールバックを書くこともできます。

## 🎨 結論とリソース

このレポートが役立つことを願っています。コードを試して、自分の選んだデータセットで画像分類器をトレーニングすることをお勧めします。

PyTorch Lightningについてもっと学ぶためのリソースはこちらです：
- [Step-by-step walk-through](https://lightning.ai/docs/pytorch/latest/starter/introduction.html) - これは公式のチュートリアルの一つです。彼らのドキュメントは非常に良く書かれており、学習リソースとして強くお勧めします。
- [Use Pytorch Lightning with Weights & Biases](https://wandb.me/lightning) - これはW&BをPyTorch Lightningと一緒に使用する方法を学ぶためのクイックコラボです。