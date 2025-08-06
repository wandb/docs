---
title: PyTorch Lightning
menu:
  tutorials:
    identifier: ja-tutorials-integration-tutorials-lightning
    parent: integration-tutorials
weight: 2
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Image_Classification_using_PyTorch_Lightning.ipynb" >}}
PyTorch Lightning を使って画像分類パイプラインを構築します。コードの可読性と再現性を高めるために、[こちらのスタイルガイド](https://lightning.ai/docs/pytorch/stable/starter/style_guide.html)に従います。さらに詳しい説明は[こちら](https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY)でご覧いただけます。

## PyTorch Lightning と W&B のセットアップ

このチュートリアルでは、PyTorch Lightning と W&B が必要です。

```shell
pip install lightning -q
pip install wandb -qU
```

```python
import lightning.pytorch as pl

# お気に入りの機械学習トラッキング ツール
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

次に、wandb アカウントにログインしてください。

```
wandb.login()
```

## DataModule ー 理想的なデータパイプライン

DataModule は、データに関連するフックを LightningModule から切り離せる仕組みです。これにより、データセットに依存しないモデル開発が可能になります。

データパイプライン全体を 1 つのクラスにまとめて再利用可能にします。DataModule は、PyTorch のデータプロセッシング手順（5つ）をカプセル化します。
- ダウンロード / トークナイズ / プロセス
- クレンジングして、必要ならディスクに保存
- Dataset へロード
- 変換処理の適用（回転、トークン化など）
- DataLoader でラップ

DataModule について詳しくは[こちら](https://lightning.ai/docs/pytorch/stable/data/datamodule.html)をご覧ください。Cifar-10 データセット向けに DataModule を作ってみましょう。


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
        # dataloader で利用する train/val のデータセットを設定
        if stage == 'fit' or stage is None:
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])

        # dataloader で利用する test データセットを設定
        if stage == 'test' or stage is None:
            self.cifar_test = CIFAR10(self.data_dir, train=False, transform=self.transform)
    
    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size)
```

## コールバック（Callback）

コールバックは、プロジェクト間で再利用できる自己完結型のプログラムです。PyTorch Lightning には、[ビルトイン・コールバック](https://lightning.ai/docs/pytorch/latest/extensions/callbacks.html#built-in-callbacks)がいくつか用意されています。
PyTorch Lightning におけるコールバックの詳細は[こちら](https://lightning.ai/docs/pytorch/latest/extensions/callbacks.html)。

### ビルトイン コールバック

このチュートリアルでは、[Early Stopping](https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.callbacks.EarlyStopping.html#lightning.callbacks.EarlyStopping) と [Model Checkpoint](https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint) のビルトインコールバックを利用します。これらは `Trainer` に渡して使用できます。


### カスタムコールバック
カスタム Keras コールバックに慣れていれば、PyTorch パイプラインでも同じことができるのは嬉しいポイントです。

今回は画像分類なので、画像サンプルごとにモデルの予測を可視化できると便利です。これをコールバックとして組み込むことで、初期段階からモデルのデバッグがしやすくなります。


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
        # モデルの予測を取得
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        # 画像を wandb の Image としてログ
        trainer.logger.experiment.log({
            "examples":[wandb.Image(x, caption=f"Pred:{pred}, Label:{y}") 
                           for x, pred, y in zip(val_imgs[:self.num_samples], 
                                                 preds[:self.num_samples], 
                                                 val_labels[:self.num_samples])]
            })
        
```

## LightningModule ー システム定義

LightningModule は「システム」を定義するもので、単なるモデルではありません。この「システム」は、研究用コードすべてを 1クラス内にまとめることで自己完結型になります。`LightningModule` は PyTorch コードを以下の 5 つのセクションに整理します。
- 計算処理（`__init__`）
- トレーニングループ（`training_step`）
- 検証ループ（`validation_step`）
- テストループ（`test_step`）
- オプティマイザー（`configure_optimizers`）

このようにして、データセットに依存しないモデルを簡単に共有可能です。Cifar-10 分類システムを作りましょう。


```
class LitModel(pl.LightningModule):
    def __init__(self, input_shape, num_classes, learning_rate=2e-4):
        super().__init__()
        
        # ハイパーパラメータのログ
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

    # Conv ブロックから Linear 層に渡す出力テンソルのサイズを返す
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._forward_features(input) 
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size
        
    # Conv ブロックから出る特徴テンソルを返す
    def _forward_features(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        return x
    
    # 推論時に利用
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

## トレーニングと評価

`DataModule` でデータパイプラインを整理し、アーキテクチャとトレーニングループを `LightningModule` で実装できたので、PyTorch Lightning の `Trainer` が残りの処理を自動化してくれます。

Trainer が自動化する主な処理は以下です。
- エポック・バッチのイテレーション
- `optimizer.step()`, `backward`, `zero_grad()` の呼び出し
- `.eval()` や勾配有効/無効化の切替
- 重みの保存・読み込み
- W&B へのログ
- 複数 GPU でのトレーニング対応
- TPU サポート
- 16ビット訓練対応


```
dm = CIFAR10DataModule(batch_size=32)
# x_dataloader にアクセスするには prepare_data と setup が必要
dm.prepare_data()
dm.setup()

# カスタム ImagePredictionLogger コールバック用サンプルの取得
val_samples = next(iter(dm.val_dataloader()))
val_imgs, val_labels = val_samples[0], val_samples[1]
val_imgs.shape, val_labels.shape
```


```
model = LitModel((3, 32, 32), dm.num_classes)

# wandb logger 初期化
wandb_logger = WandbLogger(project='wandb-lightning', job_type='train')

# コールバックの初期化
early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss")
checkpoint_callback = pl.callbacks.ModelCheckpoint()

# Trainer 初期化
trainer = pl.Trainer(max_epochs=2,
                     logger=wandb_logger,
                     callbacks=[early_stop_callback,
                                ImagePredictionLogger(val_samples),
                                checkpoint_callback],
                     )

# モデル学習 
trainer.fit(model, dm)

# 保持したテストセットで評価 ⚡⚡
trainer.test(dataloaders=dm.test_dataloader())

# wandb run の終了
run.finish()
```

## まとめに寄せて
私は TensorFlow/Keras のエコシステムから来たため、PyTorch は素晴らしくも少し敷居が高いと感じていました（個人的な感想です）。ですが PyTorch Lightning を試してみると、PyTorch を避けていた理由のほとんどが解消されていることに気付けました。ここで私が感じた違いをまとめます。
- 当時: 従来の PyTorch でモデル定義は `model.py`、トレーニングループは `train.py` とファイルが分散し、パイプライン全体の把握が難しかった。
- 現在: `LightningModule` にモデル定義と `training_step`、`validation_step` などシステムが一括管理され、モジュール化・共有が容易。
- 当時: TensorFlow/Keras のデータパイプラインは非常に充実していた一方、PyTorch の data pipeline は最も大きな課題でした。データダウンロードや前処理が各所に分散していました。
- 現在: DataModule のおかげでパイプライン全体を 1 クラスにまとめて共有・再利用可能。`train_dataloader`・`val_dataloader`・`test_dataloader` および必要な処理を統合可能。
- 当時: Keras なら `model.fit` で学習も `model.predict` ですぐ推論、`model.evaluate` ですぐ評価できましたが、PyTorch はだいたいトレーニング用・テスト用スクリプトを別々に作っていました。
- 現在: `LightningModule` と Trainer のおかげで、`trainer.fit` でトレーニング、`trainer.test` で評価を簡単に実行できるようになりました。
- 当時: TensorFlow は TPU が得意、PyTorch は…
- 現在: PyTorch Lightning なら複数 GPU や TPU でのトレーニングも簡単。
- 当時: コールバック機能のファンとして Early Stopping ひとつ取っても従来 PyTorch では面倒でした。
- 現在: PyTorch Lightning なら Early Stopping や Model Checkpoint もシンプル、カスタムコールバックも記述可能。

## 🎨 結論とリソース

このレポートが皆さんのお役に立てば幸いです。ぜひコードを触って、お好きなデータセットで画像分類モデルをトレーニングしてみてください。

PyTorch Lightning をさらに学びたい方はこちらもおすすめです。
- [ステップバイステップ解説](https://lightning.ai/docs/pytorch/latest/starter/introduction.html): 公式チュートリアルのひとつ。とても分かりやすいので初心者に特におすすめです。
- [W&B で Pytorch Lightning を使う](https://wandb.me/lightning): W&B と PyTorch Lightning を連携する方法を学びたい方におすすめの colab です。