


# PyTorch Lightning

[**こちらのColabノートブックで試す →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Image_Classification_using_PyTorch_Lightning.ipynb)

PyTorch Lightning を使って画像分類の開発フローを構築します。コードの読みやすさと再現性を高めるために、この [スタイルガイド](https://lightning.ai/docs/pytorch/stable/starter/style_guide.html) に従います。詳しい説明は[こちら](https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY)からご覧いただけます。

## PyTorch LightningとW&Bのセットアップ

このチュートリアルでは、PyTorch Lightning（当然ですね！）とWeights & Biasesが必要です。

```
!pip install lightning -q
# weights and biasesをインストール
!pip install wandb -qU
```

これらのインポートが必要になります。

```
import lightning.pytorch as pl
# お気に入りの機械学習追跡ツール
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

DataModuleは、データ関連のフックをLightningModuleから切り離す方法で、データセットに依存しないモデルを開発できます。

データパイプラインを1つの共有可能で再利用可能なクラスに整理します。DataModuleは、PyTorchでのデータ処理に関する5つのステップをカプセル化します。
- ダウンロード / トークナイズ / 処理。
- クリーンアップして（必要なら）ディスクに保存。
- データセット内にロード。
- 変換を適用（回転、トークナイズなど）。
- DataLoader内にラップ。

DataModuleについてもっと詳しくは[こちら](https://lightning.ai/docs/pytorch/stable/data/datamodule.html)をご覧ください。Cifar-10データセットのDataModuleを作成しましょう。

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
        # データローダーで使用するためにtrain/valデータセットを割り当てる
        if stage == 'fit' or stage is None:
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])

        # データローダーで使用するためにtestデータセットを割り当てる
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

コールバックとは、プロジェクト間で再利用できる自己完結型のプログラムです。PyTorch Lightningには、よく使用されるいくつかの[ビルトインコールバック](https://lightning.ai/docs/pytorch/latest/extensions/callbacks.html#built-in-callbacks)があります。
PyTorch Lightningのコールバックについてもっと詳しくは[こちら](https://lightning.ai/docs/pytorch/latest/extensions/callbacks.html)をご覧ください。

### ビルトインコールバック

このチュートリアルでは、[Early Stopping](https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.callbacks.EarlyStopping.html#lightning.callbacks.EarlyStopping)と[Model Checkpoint](https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint)のビルトインコールバックを使用します。これらは`Trainer`に渡すことができます。

### カスタムコールバック
カスタムKerasコールバックに慣れているなら、同じことをPyTorchパイプラインで行える能力は非常に嬉しいことでしょう。

画像分類を行っているため、モデルの予測をいくつかのサンプル画像で視覚化する能力が役立ちます。このようなコールバックは早期段階でモデルをデバッグするのに役立ちます。

```
class ImagePredictionLogger(pl.callbacks.Callback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # テンソルをCPUに持ってくる
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # モデルの予測を取得
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        # 画像を wandb Image としてログ
        trainer.logger.experiment.log({
            "examples":[wandb.Image(x, caption=f"Pred:{pred}, Label:{y}") 
                           for x, pred, y in zip(val_imgs[:self.num_samples], 
                                                 preds[:self.num_samples], 
                                                 val_labels[:self.num_samples])]
            })
        
```

## 🎺 LightningModule - システムの定義

LightningModuleはシステムを定義し、モデルを定義するわけではありません。ここでのシステムとは、すべての研究コードを単一のクラスにまとめて自己完結型にするものです。`LightningModule`はあなたのPyTorchコードを5つのセクションに整理します：
- 計算（`__init__`）。
- 訓練ループ（`training_step`）
- 検証ループ（`validation_step`）
- テストループ（`test_step`）
- オプティマイザー（`configure_optimizers`）

このようにして、簡単に共有できるデータセットに依存しないモデルを構築できるようになります。Cifar-10分類のシステムを構築しましょう。

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

    # ConvブロックからLinear層に入る出力テンソルのサイズを返します
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._forward_features(input) 
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size
        
    # Convブロックからの特徴テンソルを返します
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
        
        # 訓練メトリクス
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

## 🚋 訓練と評価

`DataModule` を使ってデータパイプラインを整理し、`LightningModule` を使ってモデルのアーキテクチャと訓練ループを整理したので、PyTorch Lightning の `Trainer` が他のすべてを自動化してくれます。

Trainer の自動化内容:
- エポックとバッチの反復
- `optimizer.step()`、`backward`、`zero_grad()` の呼び出し
- `.eval()` の呼び出し、グラデーションの有効/無効化
- 重みの保存と読み込み
- Weights and Biases のログ
- マルチGPU トレーニングのサポート
- TPU サポート
- 16-bit トレーニングのサポート

```
dm = CIFAR10DataModule(batch_size=32)
# x_dataloader にアクセスするには、prepare_data と setup を呼び出す必要があります。
dm.prepare_data()
dm.setup()

# 画像予測ロガーコールバックに必要なサンプルをログするためのサンプル
val_samples = next(iter(dm.val_dataloader()))
val_imgs, val_labels = val_samples[0], val_samples[1]
val_imgs.shape, val_labels.shape
```

```
model = LitModel((3, 32, 32), dm.num_classes)

# wandbロガーを初期化
wandb_logger = WandbLogger(project='wandb-lightning', job_type='train')

# コールバックを初期化
early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss")
checkpoint_callback = pl.callbacks.ModelCheckpoint()

# トレーナーを初期化
trainer = pl.Trainer(max_epochs=2,
                     logger=wandb_logger,
                     callbacks=[early_stop_callback,
                                ImagePredictionLogger(val_samples),
                                checkpoint_callback],
                     )

# モデルをトレーニング ⚡🚅⚡
trainer.fit(model, dm)

# 保持されたテストセットでモデルを評価 ⚡⚡
trainer.test(dataloaders=dm.test_dataloader())

# wandb runを終了
wandb.finish()
```

## 最後に
私はTensorFlow/Kerasエコシステムから来ており、PyTorchはエレガントなフレームワークですが、少し圧倒されることがあります。これはあくまで私の個人的な経験ですが。PyTorch Lightningを探索する中で、PyTorchから離れていた理由のほとんどが解決されていることに気付きました。ここに私の興奮を簡単にまとめます：
- 以前: 従来のPyTorchモデルの定義は非常に分散していました。モデルが`model.py`スクリプトにあり、トレーニングループが`train.py`ファイルにあるような感じで、開発フローを理解するために頻繁に行き来することが多かったです。
- 今: `LightningModule` はシステムとして機能し、モデルと `training_step`、`validation_step` などが定義されています。これでモジュール化され、共有可能です。
- 以前: TensorFlow/Kerasの一番の魅力は入力データパイプラインでした。彼らのデータセットカタログは豊富で成長しています。PyTorchのデータパイプラインは最大の難点でした。通常のPyTorchコードでは、データのダウンロード／クリーニング／準備が多くのファイルに分散していました。
- 今: DataModuleはデータパイプラインを一つの共有可能で再利用可能なクラスにまとめます。これは、`train_dataloader`、`val_dataloader`(s)、`test_dataloader`(s) とそれに対応する変換とデータ処理／ダウンロードステップを含む単なるコレクションです。
- 以前: Kerasでは`model.fit`を呼び出してモデルを訓練し、`model.predict`を呼び出して推論を行い、`model.evaluate`を使用してテストデータで簡単な評価が可能でした。PyTorchではこれは当てはまりません。通常、`train.py`と`test.py`という別々のファイルが存在します。
- 今: `LightningModule`が整っていれば、`Trainer`がすべてを自動化します。ただ`trainer.fit`や`trainer.test`を呼び出せばモデルをトレーニングおよび評価できます。
- 以前: TensorFlowはTPUが大好き、PyTorchは...。
- 今: PyTorch Lightningがあれば、同じモデルを複数のGPUやTPUで訓練することが非常に簡単です。驚き！
- 以前: コールバックの大ファンであり、カスタムコールバックを書くのが好きです。Early Stoppingのような些細なことさえ、従来のPyTorchでは議論の的でした。
- 今: PyTorch LightningでEarly StoppingやModel Checkpointingを使うのは簡単です。カスタムコールバックも書けます。

## 🎨 結論とリソース

このレポートが役立つことを願っています。コードを使って画像分類器をトレーニングしてみてください。

PyTorch Lightningについてもっと学ぶためのリソースはこちら：
- [ステップバイステップのチュートリアル](https://lightning.ai/docs/pytorch/latest/starter/introduction.html) - これは公式チュートリアルの一つです。彼らのドキュメントは非常によく書かれており、学習リソースとして強くお勧めします。
- [Weights & BiasesでPyTorch Lightningを使う](https://wandb.me/lightning) - W&BをPyTorch Lightningで使う方法を学ぶためのクイックコラボです。