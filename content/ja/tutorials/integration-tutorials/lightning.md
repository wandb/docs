---
title: PyTorch Lightning
menu:
  tutorials:
    identifier: ja-tutorials-integration-tutorials-lightning
    parent: integration-tutorials
weight: 2
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Image_Classification_using_PyTorch_Lightning.ipynb" >}}
PyTorch Lightning を使って 画像分類 の パイプライン を構築します。コードの可読性と 再現性 を高めるため、この [スタイルガイド](https://lightning.ai/docs/pytorch/stable/starter/style_guide.html) に従います。わかりやすい解説は [こちら](https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY) にもあります。

## PyTorch Lightning と W&B のセットアップ

このチュートリアルでは PyTorch Lightning と W&B が必要です。

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

次に、wandb アカウントにログインします。

```
wandb.login()
```

## DataModule - 望んでいた データ パイプライン

DataModule は、データ関連の フック を LightningModule から切り離し、データセット に依存しない モデル を開発できるようにする仕組みです。

データ パイプラインを、共有可能で再利用可能な 1 つのクラスにまとめます。DataModule は、PyTorch における データ プロセッシング の 5 つのステップをカプセル化します:
- ダウンロード / トークナイズ / 前処理
- クリーンアップし、（必要に応じて）ディスクに保存
- Dataset に読み込む
- 変換を適用（回転、トークナイズ など）
- DataLoader でラップする

DataModule についての詳細は [こちら](https://lightning.ai/docs/pytorch/stable/data/datamodule.html)。ここでは CIFAR-10 データセット 向けの DataModule を作ってみましょう。


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
        # dataloader で使う学習/検証データセットを割り当て
        if stage == 'fit' or stage is None:
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])

        # dataloader で使うテスト データセットを割り当て
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

コールバック は、プロジェクト をまたいで再利用できる自己完結型のプログラムです。PyTorch Lightning にはよく使われる [ビルトインのコールバック](https://lightning.ai/docs/pytorch/latest/extensions/callbacks.html#built-in-callbacks) が用意されています。
PyTorch Lightning のコールバックについての詳細は [こちら](https://lightning.ai/docs/pytorch/latest/extensions/callbacks.html)。

### 組み込みコールバック

このチュートリアルでは、[Early Stopping](https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.callbacks.EarlyStopping.html#lightning.callbacks.EarlyStopping) と [Model Checkpoint](https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint) のビルトイン コールバックを使います。これらは `Trainer` に渡せます。


### カスタム コールバック
Keras のカスタム コールバックに馴染みがあれば、PyTorch の パイプライン でも同じことができるのはまさに嬉しいおまけです。

画像分類 を行うので、いくつかの画像サンプルに対する モデル の 予測 を可視化できると便利です。これをコールバックとして実装すると、初期段階でのデバッグに役立ちます。 


```
class ImagePredictionLogger(pl.callbacks.Callback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # テンソルを CPU に移す
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # モデルの 予測 を取得
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        # 画像を wandb Image として ログ する
        trainer.logger.experiment.log({
            "examples":[wandb.Image(x, caption=f"Pred:{pred}, Label:{y}") 
                           for x, pred, y in zip(val_imgs[:self.num_samples], 
                                                 preds[:self.num_samples], 
                                                 val_labels[:self.num_samples])]
            })
        
```

## LightningModule - システムを定義する

LightningModule が定義するのは モデル ではなく「システム」です。ここでいうシステムとは、すべての 研究 用コードを 1 つのクラスにまとめ、自己完結させたものを指します。`LightningModule` はあなたの PyTorch コードを次の 5 つのセクションに整理します:
- 計算（`__init__`）
- 学習ループ（`training_step`）
- 検証ループ（`validation_step`）
- テスト ループ（`test_step`）
- オプティマイザー（`configure_optimizers`）

これにより、データセット に依存しない モデル を簡単に共有できます。では、CIFAR-10 の分類タスク向けにシステムを作りましょう。


```
class LitModel(pl.LightningModule):
    def __init__(self, input_shape, num_classes, learning_rate=2e-4):
        super().__init__()
        
        # ハイパーパラメーター を ログ
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

    # conv ブロックから Linear 層に入る出力テンソルのサイズを返す
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._forward_features(input) 
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size
        
    # conv ブロックからの特徴テンソルを返す
    def _forward_features(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        return x
    
    # 推論時に使用
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
        
        # 学習時の メトリクス
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # 検証 メトリクス
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        
        # 検証 メトリクス
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

```

## トレーニング と 評価

`DataModule` で データ パイプライン を、`LightningModule` で モデル の アーキテクチャー と トレーニング ループを整理したので、あとは PyTorch Lightning の `Trainer` がすべて自動化してくれます。

Trainer が自動化すること:
- エポック と バッチ の反復
- `optimizer.step()`, `backward`, `zero_grad()` の呼び出し
- `.eval()` の呼び出し、勾配の有効化/無効化
- 重みの保存と読み込み
- W&B への ログ
- マルチ GPU トレーニング のサポート
- TPU サポート
- 16-bit トレーニング サポート


```
dm = CIFAR10DataModule(batch_size=32)
# x_dataloader にアクセスするには prepare_data と setup を呼び出す必要があります。
dm.prepare_data()
dm.setup()

# 画像の予測を ログ するために、カスタム ImagePredictionLogger コールバックが必要とするサンプル
val_samples = next(iter(dm.val_dataloader()))
val_imgs, val_labels = val_samples[0], val_samples[1]
val_imgs.shape, val_labels.shape
```


```
model = LitModel((3, 32, 32), dm.num_classes)

# wandb ロガーを初期化
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

# モデルを トレーニング 
trainer.fit(model, dm)

# 保持しておいた テストセット で モデルを評価 ⚡⚡
trainer.test(dataloaders=dm.test_dataloader())

# wandb の run をクローズ
run.finish()
```

## さいごに
私は TensorFlow/Keras の エコシステム 出身で、PyTorch はエレガントな フレームワーク だと感じつつも少し圧倒されてきました（あくまで個人の感想です）。PyTorch Lightning を触ってみて、PyTorch を敬遠していた理由のほとんどが解消されていると気づきました。ワクワクポイントを手短にまとめます:
- 当時: 典型的な PyTorch の モデル 定義は散らばりがちでした。`model.py` に モデル、`train.py` に トレーニング ループという具合で、パイプライン の把握に行ったり来たりが必要でした。 
- 今は: `LightningModule` がシステムとして機能し、`training_step` や `validation_step` などと一緒に モデル を定義できます。モジュール化され、共有もしやすくなりました。
- 当時: TensorFlow/Keras の良さは入力 データ パイプライン にあり、データセット カタログも充実していました。PyTorch の データ パイプライン は最大の痛点で、通常の PyTorch コードではデータのダウンロード/クリーンアップ/前処理が多くのファイルに散らばっていました。 
- 今は: DataModule が データ パイプライン を 1 つの共有・再利用可能なクラスに整理します。`train_dataloader`、`val_dataloader`、`test_dataloader` と、必要な変換や データ の処理/ダウンロード手順の集合です。
- 当時: Keras では `model.fit` で学習、`model.predict` で推論、`model.evaluate` でテスト データのシンプルな評価ができましたが、PyTorch ではそうはいきません。たいてい `train.py` と `test.py` が別々に存在します。 
- 今は: `LightningModule` があるので、`Trainer` がすべてを自動化します。`trainer.fit` と `trainer.test` を呼べば学習と評価ができます。
- 当時: TensorFlow は TPU が大好き、PyTorch は… 
- 今は: PyTorch Lightning なら、同じ モデル を複数 GPU や TPU でも簡単に学習できます。
- 当時: 私は コールバック の大ファンで、カスタム コールバックを書くのが好きです。Early Stopping のような些細なことでも、従来の PyTorch では議論の的になりました。 
- 今は: PyTorch Lightning では Early Stopping と Model Checkpointing がとても簡単。カスタム コールバックも書けます。 

## 🎨 まとめ と リソース

このレポートが役立つことを願っています。ぜひコードをいじって、お好きな データセット で 画像分類器 をトレーニング してみてください。 

PyTorch Lightning をさらに学ぶためのリソース:
- [ステップバイステップの解説](https://lightning.ai/docs/pytorch/latest/starter/introduction.html): 公式チュートリアルのひとつです。ドキュメントがとてもよく書かれていて、強くおすすめします。
- [Use Pytorch Lightning with W&B](https://wandb.me/lightning): W&B と PyTorch Lightning の使い方を素早く学べる Colab です。