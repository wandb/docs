
# PyTorch Lightning

[**Colabノートブックで試す →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Image_Classification_using_PyTorch_Lightning.ipynb)

PyTorch Lightningを使用して画像分類開発フローを構築します。この[スタイルガイド](https://lightning.ai/docs/pytorch/stable/starter/style_guide.html)に従い、コードの可読性と再現性を向上させます。この説明の詳細は[こちら](https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY)で確認できます。

## PyTorch LightningとW&Bのセットアップ

このチュートリアルでは、PyTorch Lightning（当然ですね！）とWeights & Biasesが必要です。

```
!pip install lightning -q
# weights and biases をインストール
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

次に、あなたのwandbアカウントにログインする必要があります。

```
wandb.login()
```

## 🔧 DataModule - 理想のデータパイプライン

DataModulesは、データ関連のフックをLightningModuleから分離し、データセットに依存しないモデルを開発できるようにする手法です。

データパイプラインを共有可能で再利用可能なクラスにまとめています。DataModuleはPyTorchにおけるデータ処理の5つのステップをカプセル化します:
- データのダウンロード/トークナイズ/処理
- データのクリーニングと必要に応じてディスクへの保存
- Dataset内へのロード
- 変換の適用（回転、トークナイズなど）
- DataLoader内でのラップ

DataModulesについての詳細は[こちら](https://lightning.ai/docs/pytorch/stable/data/datamodule.html)で学べます。Cifar-10データセットのDataModuleを構築しましょう。

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
        # DataLoaderで使用するtrain/valデータセットを割り当てる
        if stage == 'fit' or stage is None:
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])

        # DataLoaderで使用するtestデータセットを割り当てる
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

Callbackは、Projects間で再利用可能な自己完結型のプログラムです。PyTorch Lightningには、いくつかの[組み込みのコールバック](https://lightning.ai/docs/pytorch/latest/extensions/callbacks.html#built-in-callbacks)があります。
PyTorch Lightningにおけるコールバックの詳細は[こちら](https://lightning.ai/docs/pytorch/latest/extensions/callbacks.html)で学べます。

### 組み込みのコールバック

このチュートリアルでは、[Early Stopping](https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.callbacks.EarlyStopping.html#lightning.callbacks.EarlyStopping)と[Model Checkpoint](https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint)の組み込みコールバックを使用します。これらは`Trainer`に渡すことができます。

### カスタムコールバック
カスタムKerasコールバックに慣れている場合、PyTorchの開発フローでも同じことができる能力は大きな利点です。

画像分類を行っているため、いくつかの画像サンプル上でモデルの予測を視覚化する能力は役立ちます。これをコールバックとして実装することで、初期段階でモデルをデバッグするのに役立ちます。

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

LightningModuleはシステムを定義しますが、それはモデルではありません。システムでは、すべての研究用コードを一つのクラスにまとめ、自己完結型にします。`LightningModule`はPyTorchコードを5つのセクションに整理します:
- 計算（`__init__`）
- トレーニングループ（`training_step`）
- 検証ループ（`validation_step`）
- テストループ（`test_step`）
- オプティマイザー（`configure_optimizers`）

これにより、容易に共有可能なデータセットに依存しないモデルを構築できます。Cifar-10分類のためのシステムを構築しましょう。

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

    # ConvブロックからLinearレイヤーに送られる出力テンソルのサイズを返します。
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._forward_features(input) 
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size
        
    # Convブロックからの特徴テンソルを返します。
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

データパイプラインを`DataModule`で、モデルのアーキテクチャーとトレーニングループを`LightningModule`で整理したので、PyTorch Lightningの`Trainer`が残りのすべてを自動化してくれます。

`Trainer`は以下を自動化します:
- エポックとバッチの繰り返し
- `optimizer.step()`、`backward`、`zero_grad()`の呼び出し
- `.eval()`の呼び出し、グラデーションの有効化/無効化
- 重みの保存とロード
- Weights & Biasesのログ
- マルチGPUトレーニングのサポート
- TPUのサポート
- 16-bitトレーニングのサポート

```
dm = CIFAR10DataModule(batch_size=32)
# x_dataloaderにアクセスするには、prepare_dataとsetupを呼び出す必要があります。
dm.prepare_data()
dm.setup()

# 画像予測をログするためのカスタムImagePredictionLoggerコールバックに必要なサンプル。
val_samples = next(iter(dm.val_dataloader()))
val_imgs, val_labels = val_samples[0], val_samples[1]
val_imgs.shape, val_labels.shape
```

```
model = LitModel((3, 32, 32), dm.num_classes)

# wandb loggerを初期化
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

# モデルのトレーニング ⚡🚅⚡
trainer.fit(model, dm)

# テストセットでモデルを評価 ⚡⚡
trainer.test(dataloaders=dm.test_dataloader())

# wandb runを終了
wandb.finish()
```

## 最終考察

私はTensorFlow/Kerasエコシステムから来ており、PyTorchはエレガントなフレームワークであるにもかかわらず、少し圧倒されることがあります。ただし、これは個人的な経験です。PyTorch Lightningを探索しているうちに、私がPyTorchから遠ざけていたほとんどの理由が解消されていることに気づきました。ここで、私が感じた興奮の概要を示します:
- 昔は: 従来のPyTorchモデルの定義は非常に分かりづらかった。modelを定義するのが`model.py`スクリプト、トレーニングループが`train.py`ファイルに分かれていて、開発フローを理解するために行ったり来たりする必要がありました。
- 今は: `LightningModule`はシステムとして機能し、モデルと共に`training_step`や`validation_step`などが定義されており、モジュール化され共有が容易です。
- 昔は: TensorFlow/Kerasの最大の利点は、入力データパイプラインにありました。カタログが豊富で増え続けています。PyTorchのデータパイプラインは最大の課題でした。通常のPyTorchコードでは、データのダウンロード、クリーニング、準備が多くのファイルに散在しています。
- 今は: DataModuleはデータパイプラインを一つの共有可能で再利用可能なクラスに整理してあります。`train_dataloader`、`val_dataloader`(s)、`test_dataloader`(s)に対応するデータ処理/ダウンロードステップが揃っています。
- 昔は: Kerasでは`model.fit`を呼び出してモデルをトレーニングし、`model.predict`で推論を実行し、`model.evaluate`でテストデータを使った評価が簡単でしたが、PyTorchではそうではありません。通常、`train.py`と`test.py`の別ファイルが必要でした。
- 今は: `LightningModule`があるので、`Trainer`がすべてを自動化してくれます。`trainer.fit`と`trainer.test`を呼び出すだけで、モデルのトレーニングと評価ができます。
- 昔は: TensorFlowはTPUを愛していますが、PyTorchは……？
- 今は: PyTorch Lightningを使うと、同じモデルを複数のGPUやTPUで簡単にトレーニングできるようになりました。驚きです。
- 昔は: コールバックファンである私は、カスタムコールバックを書くのが好きで、Early Stoppingのような些細なことまでもが従来のPyTorchでは議論の対象でした。
- 今は: PyTorch Lightningを使えば、Early StoppingやModel Checkpointの利用が簡単であり、カスタムコールバックを書くことも可能です。

## 🎨 結論とリソース

このレポートが役立つことを願っています。コードをいじって、好みのデータセットで画像分類器をトレーニングしてみてください。

PyTorch Lightningについて学ぶためのリソースはこちらです:
- [ステップバイステップのウォークスルー](https://lightning.ai/docs/pytorch/latest/starter/introduction.html) - これは公式チュートリアルの一つです。ドキュメントが非常に良く書かれており、学習リソースとして強くお勧めします。
- [Weights & BiasesとPyTorch Lightningを使用する](https://wandb.me/lightning) - これはクイックなColabノートブックで、W&BとPyTorch Lightningの使い方について詳しく学べます。