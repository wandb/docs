---
title: PyTorch Lightning
menu:
  tutorials:
    identifier: lightning
    parent: integration-tutorials
weight: 2
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Image_Classification_using_PyTorch_Lightning.ipynb" >}}
PyTorch Lightning を使って画像分類のパイプラインを構築します。この [スタイルガイド](https://lightning.ai/docs/pytorch/stable/starter/style_guide.html) に従い、 コードの可読性と再現性 を高めます。この詳しい解説は [こちら](https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY) でも確認できます。

## PyTorch Lightning と W&B のセットアップ

このチュートリアルでは、PyTorch Lightning と W&B が必要です。

```shell
pip install lightning -q
pip install wandb -qU
```

```python
import lightning.pytorch as pl

# あなたのお気に入りの機械学習トラッキングツール
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

次に、W&B アカウントにログインします。

```
wandb.login()
```

## DataModule - 本当に欲しかったデータパイプライン

DataModule とは、データに関するフックを LightningModule から切り離し、データセット非依存なモデル開発を実現する方法です。

データパイプライン全体を、1つの使い回しやすいクラスとして整理できます。datamodule は PyTorch でのデータ処理に必要な5つのステップをカプセル化します:
- ダウンロード / トークナイズ / 前処理
- クレンジングと（必要なら）ディスクへの保存
- Dataset 内に読み込み
- 変換の適用（回転・トークナイズなど）
- DataLoader でラップ

datamodule については[こちら](https://lightning.ai/docs/pytorch/stable/data/datamodule.html)で詳しく学べます。Cifar-10 データセット向けに datamodule を実装してみましょう。

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
        # dataloader 用の train/val データセットを割り当てる
        if stage == 'fit' or stage is None:
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])

        # dataloader 用の test データセットを割り当てる
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

コールバックとは、プロジェクト間で再利用できる独立したプログラムです。PyTorch Lightning にはよく使われる [組み込みのコールバック](https://lightning.ai/docs/pytorch/latest/extensions/callbacks.html#built-in-callbacks) がいくつか用意されています。
PyTorch Lightning のコールバックの詳細は [こちら](https://lightning.ai/docs/pytorch/latest/extensions/callbacks.html) をご覧ください。

### 組み込みコールバック

このチュートリアルでは、[Early Stopping](https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.callbacks.EarlyStopping.html#lightning.callbacks.EarlyStopping) と [Model Checkpoint](https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint) の組み込みコールバックを利用します。これらは `Trainer` に渡すことができます。

### カスタムコールバック
Keras のカスタムコールバックに慣れている方なら、PyTorch でも同じことができるのは嬉しいポイントです。

画像分類を行う中で、いくつかの画像サンプルに対するモデルの予測を可視化できるのはとても役立ちます。コールバックとして実装すれば、モデリング初期段階でのデバッグにも便利です。

```
class ImagePredictionLogger(pl.callbacks.Callback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # テンソルを CPU に移動
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # モデルによる予測を取得
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        # 画像を wandb Image としてログする
        trainer.logger.experiment.log({
            "examples":[wandb.Image(x, caption=f"Pred:{pred}, Label:{y}") 
                           for x, pred, y in zip(val_imgs[:self.num_samples], 
                                                 preds[:self.num_samples], 
                                                 val_labels[:self.num_samples])]
            })
        
```

## LightningModule - システムを定義する

LightningModule は「モデル」ではなく「システム」を定義します。ここでいうシステムとは、すべての研究コードを 1 つのクラスに集約し、自己完結型にしたものです。`LightningModule` では PyTorch のコードを5つのセクションに整理します:
- 計算（`__init__`）
- トレーニングループ（`training_step`）
- 検証ループ（`validation_step`）
- テストループ（`test_step`）
- オプティマイザー（`configure_optimizers`）

このようにして、データセット非依存なモデルを簡単に共有できる形で作れます。Cifar-10 用分類システムを構築してみましょう。

```
class LitModel(pl.LightningModule):
    def __init__(self, input_shape, num_classes, learning_rate=2e-4):
        super().__init__()
        
        # ハイパーパラメータをログ
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

    # conv ブロックから Linear 層へ渡す出力テンソルのサイズを返す
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._forward_features(input) 
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size
        
    # conv ブロックから特徴テンソルを返す
    def _forward_features(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        return x
    
    # 推論時に使われる
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

## トレーニングと評価

これで、`DataModule` でデータパイプライン、`LightningModule` でモデルアーキテクチャとトレーニングループを整理できました。あとは PyTorch Lightning の `Trainer` がその後の処理を自動化してくれます。

Trainer で自動化されること:
- エポック、バッチの繰り返し
- `optimizer.step()`、`backward`、`zero_grad()` の呼び出し
- `.eval()` の呼び出しと勾配の有効／無効化
- 重みの保存と読み込み
- W&B へのログ記録
- マルチ GPU トレーニングのサポート
- TPU サポート
- 16 ビット精度トレーニングのサポート

```
dm = CIFAR10DataModule(batch_size=32)
# x_dataloaderにアクセスするにはprepare_dataとsetupの呼び出しが必要です
dm.prepare_data()
dm.setup()

# ImagePredictionLogger コールバックで画像予測のログを取るためにサンプルを準備
val_samples = next(iter(dm.val_dataloader()))
val_imgs, val_labels = val_samples[0], val_samples[1]
val_imgs.shape, val_labels.shape
```

```
model = LitModel((3, 32, 32), dm.num_classes)

# W&B ロガーを初期化
wandb_logger = WandbLogger(project='wandb-lightning', job_type='train')

# コールバックを初期化
early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss")
checkpoint_callback = pl.callbacks.ModelCheckpoint()

# Trainer を初期化
trainer = pl.Trainer(max_epochs=2,
                     logger=wandb_logger,
                     callbacks=[early_stop_callback,
                                ImagePredictionLogger(val_samples),
                                checkpoint_callback],
                     )

# モデルのトレーニング 
trainer.fit(model, dm)

# テストセットでモデルを評価 ⚡⚡
trainer.test(dataloaders=dm.test_dataloader())

# W&B run をクローズ
run.finish()
```

## 最後に
私は TensorFlow/Keras エコシステムの出身で、PyTorch はすごくエレガントだけどちょっと難しく感じていました。でも PyTorch Lightning を触ってみて、PyTorch から遠ざかっていた理由のほとんどが Lightning で解決されていることに気づきました。私なりの感動ポイントをまとめます:
- 当時: 従来の PyTorch ではモデル定義があちこちに分散していて、`model.py` にモデル、`train.py` にトレーニングループ、パイプラインを理解するにはあちこち見直す必要がありました。
- 現在: `LightningModule` は、モデル定義と `training_step`、`validation_step` などの処理が一つのシステムとしてまとまり、モジュール化・共有もしやすくなりました。
- 当時: TensorFlow/Keras のデータパイプラインは本当に優秀。PyTorch のデータパイプラインは大きな課題で、ダウンロードや前処理が様々なファイルに分散していました。
- 現在: DataModule がパイプラインを1つの再利用可能クラスに整理。`train_dataloader`、`val_dataloader`、`test_dataloader` 呼び出しや変換、ダウンロード等もここに集約できます。
- 当時: Keras なら `model.fit` で学習、`model.predict` で推論、`model.evaluate` で評価が一発。PyTorch では `train.py` と `test.py` が分かれるパターンがほとんどでした。
- 現在: `LightningModule` 採用後は `Trainer` が全自動化。`trainer.fit`、`trainer.test` を呼び出すだけです。
- 当時: TensorFlow は TPU が得意、PyTorch は... という感じでした。
- 現在: PyTorch Lightning なら GPU 複数台や TPU でも簡単に並列学習できます。
- 当時: コールバックが大好きで、カスタムコールバックも書きたい派。でも PyTorch で Early Stopping すら議論になるほど大変でした。
- 現在: PyTorch Lightning で Early Stopping や Model Checkpoint も一瞬。もちろんカスタムコールバックも簡単に書けます。

## 🎨 まとめ・関連リソース

このレポートが皆さんのお役に立てば幸いです。ぜひコードを色々と触って、お好きなデータセットで画像分類モデルを訓練してみてください。

PyTorch Lightning をもっと知りたい方のためのリソース：
- [Step-by-step walk-through](https://lightning.ai/docs/pytorch/latest/starter/introduction.html): 公式チュートリアルのひとつです。ドキュメントも非常に分かりやすいので、学習リソースとしてとてもおすすめです。
- [Use Pytorch Lightning with W&B](https://wandb.me/lightning): PyTorch Lightning と W&B の連携方法を学べる Colab です。手軽に動かしてみてください。