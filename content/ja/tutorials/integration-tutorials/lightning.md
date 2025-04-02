---
title: PyTorch Lightning
menu:
  tutorials:
    identifier: ja-tutorials-integration-tutorials-lightning
    parent: integration-tutorials
weight: 2
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Image_Classification_using_PyTorch_Lightning.ipynb" >}}
PyTorch Lightning を使用して画像分類のパイプラインを構築します。コードの可読性と再現性を高めるために、こちらの [スタイル ガイド ](https://lightning.ai/docs/pytorch/stable/starter/style_guide.html)に従います。これに関するわかりやすい説明は、[こちら](https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY)にあります。

## PyTorch Lightning と W&B のセットアップ

このチュートリアルでは、PyTorch Lightning と Weights & Biases が必要です。

```shell
pip install lightning -q
pip install wandb -qU
```

```python
import lightning.pytorch as pl

# お気に入りの 機械学習 トラッキング ツール
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

次に、wandb アカウントにログインする必要があります。

```
wandb.login()
```

## DataModule - 価値のあるデータ パイプライン

DataModule は、データ関連のフックを LightningModule から分離して、データセットに依存しないモデルを開発できるようにする方法です。

データパイプラインを 1 つの共有可能で再利用可能なクラスにまとめます。 datamodule は、PyTorch でのデータ処理に関わる次の 5 つのステップをカプセル化します。
- ダウンロード/トークン化/プロセッシング。
- クリーンアップして、（場合によっては）ディスクに保存します。
- データセット内にロードします。
- 変換（回転、トークン化など）を適用します。
- DataLoader 内にラップします。

datamodule の詳細については、[こちら](https://lightning.ai/docs/pytorch/stable/data/datamodule.html)をご覧ください。 Cifar-10 データセット用の datamodule を構築してみましょう。

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
        # dataloader で使用するトレーニング/検証データセットを割り当てます
        if stage == 'fit' or stage is None:
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])

        # dataloader で使用するテストデータセットを割り当てます
        if stage == 'test' or stage is None:
            self.cifar_test = CIFAR10(self.data_dir, train=False, transform=self.transform)
    
    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size)
```

## コールバック

callback は、プロジェクト間で再利用できる自己完結型のプログラムです。 PyTorch Lightning には、定期的に使用されるいくつかの [組み込み callbacks](https://lightning.ai/docs/pytorch/latest/extensions/callbacks.html#built-in-callbacks)が付属しています。
PyTorch Lightning の callbacks の詳細については、[こちら](https://lightning.ai/docs/pytorch/latest/extensions/callbacks.html)をご覧ください。

### 組み込み Callbacks

このチュートリアルでは、組み込みの [Early Stopping](https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.callbacks.EarlyStopping.html#lightning.callbacks.EarlyStopping) と [Model Checkpoint](https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint) の callbacks を使用します。 これらは `Trainer` に渡すことができます。

### カスタム Callbacks
カスタム Keras callback に慣れている場合は、PyTorch パイプラインで同じことができる機能は、まさに嬉しいおまけです。

画像分類を実行しているので、モデルの予測をいくつかの画像のサンプルで視覚化できると役立ちます。 これを callback の形式にすることで、初期段階でモデルをデバッグできます。

```
class ImagePredictionLogger(pl.callbacks.Callback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # テンソルを CPU に取り込みます
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # モデルの予測を取得します
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        # 画像を wandb Image として記録します
        trainer.logger.experiment.log({
            "examples":[wandb.Image(x, caption=f"Pred:{pred}, Label:{y}") 
                           for x, pred, y in zip(val_imgs[:self.num_samples], 
                                                 preds[:self.num_samples], 
                                                 val_labels[:self.num_samples])]
            })
        
```

## LightningModule - システムの定義

LightningModule は、モデルではなくシステムを定義します。 ここでは、システムはすべての研究コードを 1 つのクラスにグループ化して、自己完結型にします。 `LightningModule` は、PyTorch コードを次の 5 つのセクションに整理します。
- 計算 (`__init__`)。
- トレーニング ループ (`training_step`)
- 検証ループ (`validation_step`)
- テスト ループ (`test_step`)
- オプティマイザー (`configure_optimizers`)

したがって、簡単に共有できるデータセットに依存しないモデルを構築できます。 Cifar-10 分類用のシステムを構築してみましょう。

```
class LitModel(pl.LightningModule):
    def __init__(self, input_shape, num_classes, learning_rate=2e-4):
        super().__init__()
        
        # ハイパーパラメータを記録します
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

    # conv ブロックから Linear レイヤーに入る出力テンソルのサイズを返します。
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._forward_features(input) 
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size
        
    # conv ブロックから特徴テンソルを返します
    def _forward_features(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        return x
    
    # 推論中に使用されます
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
        
        # トレーニング メトリクス
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

## トレーニングと評価

`DataModule` を使用してデータパイプラインを整理し、`LightningModule` を使用してモデル アーキテクチャ + トレーニング ループを整理したので、PyTorch Lightning `Trainer` がそれ以外のすべてを自動化します。

Trainer は以下を自動化します。
- エポックとバッチの反復
- `optimizer.step()`、`backward`、`zero_grad()` の呼び出し
- `.eval()` の呼び出し、grads の有効化/無効化
- 重みの保存とロード
- Weights & Biases ロギング
- 複数 GPU トレーニングのサポート
- TPU サポート
- 16 ビット トレーニングのサポート

```
dm = CIFAR10DataModule(batch_size=32)
# x_dataloader にアクセスするには、prepare_data と setup を呼び出す必要があります。
dm.prepare_data()
dm.setup()

# 画像予測を記録するためにカスタム ImagePredictionLogger callback で必要なサンプル。
val_samples = next(iter(dm.val_dataloader()))
val_imgs, val_labels = val_samples[0], val_samples[1]
val_imgs.shape, val_labels.shape
```

```
model = LitModel((3, 32, 32), dm.num_classes)

# wandb logger を初期化します
wandb_logger = WandbLogger(project='wandb-lightning', job_type='train')

# Callbacks を初期化します
early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss")
checkpoint_callback = pl.callbacks.ModelCheckpoint()

# trainer を初期化します
trainer = pl.Trainer(max_epochs=2,
                     logger=wandb_logger,
                     callbacks=[early_stop_callback,
                                ImagePredictionLogger(val_samples),
                                checkpoint_callback],
                     )

# モデルをトレーニングします
trainer.fit(model, dm)

# ⚡⚡ 保持されたテスト セットでモデルを評価します
trainer.test(dataloaders=dm.test_dataloader())

# wandb run を閉じます
wandb.finish()
```

## 最終的な考え

私は TensorFlow/Keras エコシステム出身で、PyTorch はエレガントなフレームワークですが、少し圧倒されると感じています。 これはあくまで私の個人的な経験です。 PyTorch Lightning を調べているうちに、PyTorch から遠ざかっていた理由のほとんどが解消されていることに気づきました。 私が興奮している点の簡単なまとめを以下に示します。
- 以前: 従来の PyTorch モデルの定義は、あちこちに散らばっていました。 モデルは `model.py` スクリプトに、トレーニング ループは `train.py` ファイルに記述されていました。 パイプラインを理解するには、何度も見返す必要がありました。
- 現在: `LightningModule` は、モデルが `training_step`、`validation_step` などと共に定義されているシステムとして機能します。 これで、モジュール式になり、共有できるようになりました。
- 以前: TensorFlow/Keras の最も優れている点は、入力データ パイプラインです。 データセット カタログは豊富で、成長を続けています。 PyTorch のデータ パイプラインは、これまでで最大の難点でした。 通常の PyTorch コードでは、データのダウンロード/クリーンアップ/準備は通常、多くのファイルに分散しています。
- 現在: DataModule は、データ パイプラインを 1 つの共有可能で再利用可能なクラスにまとめます。 これは単に、`train_dataloader`、`val_dataloader`(s)、`test_dataloader`(s) のコレクションであり、必要な変換とデータ プロセッシング/ダウンロードの手順が付属しています。
- 以前: Keras を使用すると、`model.fit` を呼び出してモデルをトレーニングし、`model.predict` を呼び出して推論を実行できます。 `model.evaluate` は、テストデータに対する古き良き単純な評価を提供しました。 これは PyTorch には当てはまりません。 通常、個別の `train.py` ファイルと `test.py` ファイルが見つかります。
- 現在: `LightningModule` が導入されたことで、`Trainer` がすべてを自動化します。 モデルをトレーニングおよび評価するには、`trainer.fit` と `trainer.test` を呼び出すだけで済みます。
- 以前: TensorFlow は TPU が大好きですが、PyTorch は...
- 現在: PyTorch Lightning を使用すると、複数の GPU や TPU 上でも同じモデルを簡単にトレーニングできます。
- 以前: 私は Callbacks の大ファンで、カスタム callbacks を記述することを好みます。 Early Stopping のように些細なことでも、従来の PyTorch では議論の対象となっていました。
- 現在: PyTorch Lightning を使用すると、Early Stopping と Model Checkpointing を簡単に使用できます。 カスタム callbacks を記述することもできます。

## 🎨 まとめとリソース

このレポートがお役に立てば幸いです。 コードを試して、選択したデータセットを使用して画像分類器をトレーニングすることをお勧めします。

PyTorch Lightning の詳細については、次のリソースをご覧ください。
- [ステップごとのウォークスルー](https://lightning.ai/docs/pytorch/latest/starter/introduction.html) - これは公式チュートリアルの 1 つです。 ドキュメントは非常によく書かれており、優れた学習リソースとして強くお勧めします。
- [Weights & Biases で Pytorch Lightning を使用する](https://wandb.me/lightning) - これは、W&B を PyTorch Lightning で使用する方法の詳細を学ぶために実行できる簡単な colab です。
