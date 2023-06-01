---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# PyTorch Lightning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://wandb.me/lightning)

PyTorch Lightningは、PyTorchコードを整理し、分散トレーニングや16ビット精度などの高度な機能を簡単に追加するための軽量なラッパーを提供します。W&Bは、ML実験のログを取るための軽量なラッパーを提供します。しかし、両方を自分で組み合わせる必要はありません。Weights & Biasesは、PyTorch Lightningライブラリに[**`WandbLogger`**](https://pytorch-lightning.readthedocs.io/en/stable/extensions/generated/pytorch\_lightning.loggers.WandbLogger.html#pytorch\_lightning.loggers.WandbLogger)を介して直接組み込まれています。

## ⚡ たった二行で素早く始めましょう。

```python
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

wandb_logger = WandbLogger()
trainer = Trainer(logger=wandb_logger)
```

![どこからでもアクセス可能なインタラクティブなダッシュボードなど！](@site/static/images/integrations/n6P7K4M.gif)

## wandbにサインアップし、ログインする

a) 無料アカウントに[**サインアップ**](https://wandb.ai/site)する

b) `wandb`ライブラリをPipインストール

c) トレーニングスクリプトでログインするには、www.wandb.aiでアカウントにサインインしてから、[**Authorizeページ**](https://wandb.ai/authorize)で**APIキーを見つけてください。**

もし、Weights and Biasesを初めて使う場合は、[**クイックスタート**](../../quickstart.md)をチェックしてみてください。
<Tabs
  defaultValue="cli"
  values={[
    {label: 'コマンドライン', value: 'cli'},
    {label: 'ノートブック', value: 'notebook'},
  ]}>
  <TabItem value="cli">

```python
pip install wandb

wandb login
```

</TabItem>
  <TabItem value="notebook">

```python
!pip install wandb

import wandb
wandb.login()
```

  </TabItem>
</Tabs>

## PyTorch Lightning の `WandbLogger` を使う方法

PyTorch Lightning には、メトリクスやモデルの重み、メディアなどをシームレスにログに記録できる [**`WandbLogger`**](https://pytorch-lightning.readthedocs.io/en/latest/extensions/generated/pytorch\_lightning.loggers.WandbLogger.html?highlight=wandblogger) クラスがあります。WandbLogger をインスタンス化し、Lightning の `Trainer` に渡すだけです。
```
wandb_logger = WandbLogger()
trainer = Trainer(logger=wandb_logger)
```

### Logger 引数

以下は、WandbLoggerでよく使われるパラメータの一部です。完全なリストと説明は、PyTorch Lightningの[**`WandbLogger`ドキュメント**](https://pytorch-lightning.readthedocs.io/en/latest/extensions/generated/pytorch\_lightning.loggers.WandbLogger.html?highlight=wandblogger)をご覧ください。

| パラメータ  | 説明                                                               |
| ----------- | ------------------------------------------------------------------ |
| `project`   | wandb プロジェクトにログを送る                                    |
| `name`      | wandb runに名前を付ける                                            |
| `log_model` | `log_model="all"` ですべてのモデルをログするか、`log_model=True` でトレーニング終了時にログする |
| `save_dir`  | データが保存されるパス                                             |

### LightningModule のハイパーパラメーターをログに記録する

```python
class LitModule(LightningModule):
    def __init__(self, *args, **kwarg):
        self.save_hyperparameters()
```

### さらなるconfigパラメータをログに記録する

```python
# 1つのパラメーターを追加する
wandb_logger.experiment.config["key"] = value
# 複数のパラメータを追加する
wandb_logger.experiment.config.update({key1: val1, key2: val2})

# wandbモジュールを直接使う
wandb.config["キー"] = 値
wandb.config.update()
```

### 勾配、パラメータヒストグラム、モデルトポロジーを記録する

モデルオブジェクトを `wandblogger.watch()` に渡すことで、トレーニング中のモデルの勾配やパラメータを監視できます。詳細は PyTorch Lightning の [**`WandbLogger` ドキュメント**](https://pytorch-lightning.readthedocs.io/en/latest/extensions/generated/pytorch\_lightning.loggers.WandbLogger.html?highlight=wandblogger) をご覧ください。

### メトリクスを記録する

`WandbLogger` を使っている場合、`LightningModule` 内の `self.log('my_metric_name', metric_value)` を呼び出すことで、W&Bにメトリクスを記録できます。これはあなたの `training_step` や __`validation_step` メソッド内で行うことができます。

以下のコードスニペットは、メトリクスと `LightningModule` のハイパーパラメータを記録するように `LightningModule` を定義する方法を示しています。この例では、[`torchmetrics`](https://github.com/PyTorchLightning/metrics) ライブラリを使ってメトリクスを計算します。

```python
import torch
from torch.nn import Linear, CrossEntropyLoss, functional as F
from torch.optim import Adam
from torchmetrics.functional import accuracy
from pytorch_lightning import LightningModule

class My_LitModule(LightningModule):

    def __init__(self, n_classes=10, n_layer_1=128, n_layer_2=256, lr=1e-3):
        '''モデルのパラメータを定義するメソッド'''
        super().__init__()

# mnist画像は(1, 28, 28) (チャンネル, 幅, 高さ）です。
        self.layer_1 = Linear(28 * 28, n_layer_1)
        self.layer_2 = Linear(n_layer_1, n_layer_2)
        self.layer_3 = Linear(n_layer_2, n_classes)

        self.loss = CrossEntropyLoss()
        self.lr = lr

        # ハイパーパラメーターをself.hparamsに保存する (W&Bによって自動ロギングされます)
        self.save_hyperparameters()

    def forward(self, x):
        '''推論用の入力から出力へのメソッド'''
        
        # (b, 1, 28, 28) -> (b, 1*28*28)
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)

        # 3回(linear + relu)をやりましょう
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

    def training_step(self, batch, batch_idx):
        '''単一バッチからの損失を返す必要があります'''
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # 損失とメトリックを記録
        self.log('train_loss', loss)
        self.log('train_accuracy', acc)
        return loss
def validation_step(self, batch, batch_idx):
        '''メトリクスのログ記録に使用されます'''
        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # 損失とメトリクスをログに記録
        self.log('val_loss', loss)
        self.log('val_accuracy', acc)
        return preds

    def configure_optimizers(self):
        '''モデルのオプティマイザを定義します'''
        return Adam(self.parameters(), lr=self.lr)

    def _get_preds_loss_accuracy(self, batch):
        '''トレーニング/検証/テストのステップが似ているため、便利な関数です'''
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.loss(logits, y)
        acc = accuracy(preds, y)
        return preds, loss, acc
```

### メトリックの最小値/最大値をログに記録

wandbの[`define_metric`](https://docs.wandb.ai/ref/python/run#define\_metric)関数を使用することで、W&Bのサマリーメトリックに表示されるメトリックの最小値、最大値、平均値、または最適値を定義できます。`define_metric`が使われていない場合、ログに記録された最後の値がサマリーメトリックに表示されます。[`define_metric`](https://docs.wandb.ai/ref/python/run#define\_metric)のリファレンスドキュメントと[ガイド](https://docs.wandb.ai/guides/track/log#customize-axes-and-summaries-with-define\_metric)を参照してください。

W&Bのサマリーメトリックで最大検証精度を追跡するように指示するには、`wandb.define_metric`を1回呼び出すだけです。例えば、トレーニングの開始時に以下のように呼び出すことができます。

```python
class My_LitModule(LightningModule):
    ...

def validation_step(self, batch, batch_idx):
        if trainer.global_step == 0: 
            wandb.define_metric('val_accuracy', summary='max')
        
        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # ロスと指標をログする
        self.log('val_loss', loss)
        self.log('val_accuracy', acc)
        return preds
```

### モデルのチェックポイント作成

W&Bにカスタムチェックポイントを設定するには、PyTorch Lightningの [`ModelCheckpoint`](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch\_lightning.callbacks.ModelCheckpoint.html#pytorch\_lightning.callbacks.ModelCheckpoint) を`WandbLogger` のlog_model 引数で使用します。

```python
# `val_accuracy`が増加する場合にのみモデルをログする
wandb_logger = WandbLogger(log_model="all")
checkpoint_callback = ModelCheckpoint(monitor="val_accuracy", mode="max")
trainer = Trainer(logger=wandb_logger, callbacks=[checkpoint_callback])
```

最新と最良のエイリアスが自動的に設定されているため、W&Bのアーティファクトから簡単にモデルチェックポイントを取得できます。

```python
# アーティファクトパネルで参照を取得できます
# "VERSION"はバージョン（例："v2"）またはエイリアス（"latest"または"best"）であることができます
checkpoint_reference = "USER/PROJECT/MODEL-RUN_ID:VERSION"


# チェックポイントをローカルにダウンロード（すでにキャッシュされていない場合）
run = wandb.init(project="MNIST")
artifact = run.use_artifact(checkpoint_reference, type="model")
artifact_dir = artifact.download()

# チェックポイントの読み込み
model = LitModule.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")
```

### 画像、テキスト、その他のログ

`WandbLogger`には、メディアをログするための`log_image`、`log_text`、`log_table`メソッドがあります。

また、Audio、Molecules、Point Clouds、3Dオブジェクトなどの他のメディアタイプをログするために、直接`wandb.log`や`trainer.logger.experiment.log`を呼び出すこともできます。

<Tabs
  defaultValue="images"
  values={[
    {label: 'Log Images', value: 'images'},
    {label: 'Log Text', value: 'text'},
    {label: 'Log Tables', value: 'tables'},
  ]}>
  <TabItem value="images">

```python
# using tensors, numpy arrays or PIL images
wandb_logger.log_image(key="samples", images=[img1, img2])

# adding captions
wandb_logger.log_image(key="samples", images=[img1, img2], caption=["tree", "person"])

# using file path
wandb_logger.log_image(key="samples", images=["img_1.jpg", "img_2.jpg"])

# using .log in the trainer
trainer.logger.experiment.log({
    "samples": [wandb.Image(img, caption=caption) 
    for (img, caption) in my_images]
})
```
  </TabItem>
  <TabItem value="text">

```python
# data should be a list of lists
columns = ["input", "label", "prediction"]
my_data = [["cheese", "english", "english"], ["fromage", "french", "spanish"]]

# using columns and data
wandb_logger.log_text(key="my_samples", columns=columns, data=my_data)

# using a pandas DataFrame
wandb_logger.log_text(key="my_samples", dataframe=my_dataframe)
```

  </TabItem>
  <TabItem value="tables">

```python
# log a W&B Table that has a text caption, an image and audio
columns = ["caption", "image", "sound"]

# data should be a list of lists
my_data = [["cheese", wandb.Image(img_1), wandb.Audio(snd_1)], 
        ["wine", wandb.Image(img_2), wandb.Audio(snd_2)]]

# log the Table
wandb_logger.log_table(key="my_samples", columns=columns, data=data)
```

  </TabItem>
</Tabs>


LightningのCallbacksシステムを使用して、WandbLoggerを介してWeights & Biasesにログを記録するタイミングを制御できます。この例では、検証画像のサンプルと予測をログに記録しています。

```python
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

class LogPredictionSamplesCallback(Callback):

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """検証バッチが終了したときに呼び出されます。"""

# `outputs`は`LightningModule.validation_step`から来ています
        # これは、この場合、モデルの予測に対応します
        
        # 1つ目のバッチから20個のサンプル画像の予測をログに記録しましょう
        if batch_idx == 0:
            n = 20
            x, y = batch
            images = [img for img in x[:n]]
            captions = [f'正解: {y_i} - 予測: {y_pred}' 
                for y_i, y_pred in zip(y[:n], outputs[:n])]
            
            
            # オプション1：`WandbLogger.log_image`を使って画像をログに記録する
            wandb_logger.log_image(
                key='sample_images', 
                images=images, 
                caption=captions)


            # オプション2：画像と予測をW&Bテーブルとしてログに記録する
            columns = ['image', 'ground truth', 'prediction']
            data = [[wandb.Image(x_i), y_i, y_pred] f
                or x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))]
            wandb_logger.log_table(
                key='sample_table',
                columns=columns,
                data=data)            
...

trainer = pl.Trainer(
    ...
    callbacks=[LogPredictionSamplesCallback()]
)
```
### LightningとW&Bを使って複数のGPUを使用する方法は？

PyTorch Lightningは、DDPインターフェースを介して、複数のGPUをサポートしています。ただし、PyTorch Lightningの設計では、GPUのインスタンス化方法に注意が必要です。

Lightningは、トレーニングループ内の各GPU（またはランク）が、同じ初期条件で正確に同じ方法でインスタンス化されていることを前提としています。ただし、ランク0プロセスのみが`wandb.run`オブジェクトにアクセスでき、ランクが0でないプロセスの場合は`wandb.run = None`です。このことで、ランクが0でないプロセスが失敗する可能性があります。このような状況は、ランク0プロセスが既にクラッシュしたランク0以外のプロセスに参加していないため、「**デッドロック**」に陥る可能性があります。

このため、トレーニングコードを設定する方法に注意が必要です。おすすめの方法は、コードが`wandb.run`オブジェクトから独立しているように設定することです。

```python
class MNISTClassifier(pl.LightningModule):
    def __init__(self):
        super(MNISTClassifier, self).__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
```

```
model = MNISTClassifier()
    wandb_logger = WandbLogger(project = "<project_name>")
    callbacks = [
        ModelCheckpoint(
            dirpath = "checkpoints",
            every_n_train_steps=100,
        ),
    ]
    trainer = pl.Trainer(
        max_epochs = 3, 
        gpus = 2, 
        logger = wandb_logger, 
        strategy="ddp", 
        callbacks=callbacks
    ) 
    trainer.fit(model, train_loader, val_loader)
```

## インタラクティブな例をチェックしよう！

私たちのチュートリアルColab [こちら](https://wandb.me/lit-colab)でビデオチュートリアルに沿って進めることができます。

<!-- {% embed url="https://www.youtube.com/watch?v=hUXQm46TAKc" %} -->

## よくある質問

### W&BはLightningとどのように統合されていますか？
コアの統合は、[Lightning `loggers` API](https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html)をベースにしており、フレームワークに依存しない方法でログコードの多くを記述できます。`Logger`は[Lightning `Trainer`](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html)に渡され、APIの豊富な[フック・コールバックシステム](https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html)に基づいてトリガーされます。これにより、研究コードとエンジニアリング、ログコードがうまく分離されます。



### 追加コードなしで何をログに残しますか？



モデルのチェックポイントをW&Bに保存し、閲覧や今後のrunでの使用のためにダウンロードすることができます。また、GPU使用量やネットワークI/Oなどの[システムメトリクス](../app/features/system-metrics.md)、ハードウェアやOS情報などの環境情報、[コードの状態](../app/features/panels/code.md)（gitコミットや差分パッチ、ノートブックの内容やセッション履歴を含む）、および標準出力に出力される内容をすべてキャプチャします。



### トレーニング設定で`wandb.run`を使用する必要がある場合はどうすればいいですか？



自分でアクセスする必要がある変数のスコープを基本的に拡張する必要があります。言い換えれば、すべてのプロセスで初期条件が同じであることを確認することです。



```python

if os.environ.get("LOCAL_RANK", None) is None:

    os.environ["WANDB_DIR"] = wandb.run.dir

```



そして、`os.environ["WANDB_DIR"]`を使用してモデルチェックポイントのディレクトリを設定できます。この方法で、`wandb.run.dir`はゼロ以外のランクのプロセスにも使用できます。