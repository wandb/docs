---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# PyTorch Lightning

[**Try in a Colab Notebook here →**](https://wandb.me/lightning)

PyTorch Lightningは、PyTorchコードを整理し、分散トレーニングや16ビット精度などの高度な機能を簡単に追加するための軽量ラッパーを提供します。W&Bは、機械学習の実験をログに記録するための軽量ラッパーを提供します。しかし、これらを自分で組み合わせる必要はありません。Weights & Biasesは、[**`WandbLogger`**](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)を通じて、直接PyTorch Lightningライブラリに組み込まれています。

## ⚡ 数行で始める

<Tabs
  defaultValue="pytorch"
  values={[
    {label: "Pytorch Logger", value: "pytorch"},
    {label: "Fabric Logger", value: "fabric"},
]}>

<TabItem value="pytorch">

```python
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer

wandb_logger = WandbLogger(log_model="all")
trainer = Trainer(logger=wandb_logger)
```

:::info
**wandb.log()の使用:** `WandbLogger`は、Trainerの`global_step`を使用してW&Bにログを記録します。コード内で直接`wandb.log`を追加で呼び出している場合、`wandb.log()`の`step`引数は**使用しないでください**。代わりに、他のメトリクスと同様にTrainerの`global_step`をログに記録してください。

`wandb.log({"accuracy":0.99, "trainer/global_step": step})`
:::

</TabItem>

<TabItem value="fabric">

```python
import lightning as L
from wandb.integration.lightning.fabric import WandbLogger

wandb_logger = WandbLogger(log_model="all")
fabric = L.Fabric(loggers=[wandb_logger])
fabric.launch()
fabric.log_dict({"important_metric": important_metric})
```

</TabItem>

</Tabs>

![Interactive dashboards accessible anywhere, and more!](@site/static/images/integrations/n6P7K4M.gif)

## wandbにサインアップとログイン

a) [**Sign up**](https://wandb.ai/site) for a free account

b) `wandb`ライブラリをPipでインストール

c) トレーニングスクリプトでログインするには、www.wandb.aiでアカウントにサインインしている必要があります。その後、[**Authorize page**](https://wandb.ai/authorize)でAPIキーを見つけます。

Weights and Biasesを初めて使用する場合は、[**クイックスタート**](../../quickstart.md)をチェックすることをお勧めします。

<Tabs
  defaultValue="cli"
  values={[
    {label: 'Command Line', value: 'cli'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="cli">

```bash
pip install wandb

wandb login
```

</TabItem>
  <TabItem value="notebook">

```notebook
!pip install wandb

import wandb
wandb.login()
```

  </TabItem>
</Tabs>

## PyTorch Lightningの`WandbLogger`の使用

PyTorch Lightningには、メトリクス、モデルの重み、メディアなどをシームレスにログ記録できる複数の`WandbLogger`クラス ([**`Pytorch`**](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)) ([**`Fabric`**](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)) があります。単にWandbLoggerをインスタンス化し、Lightningの`Trainer`や`Fabric`に渡すだけです。

```
wandb_logger = WandbLogger()
```

<Tabs
  defaultValue="pytorch"
  values={[
    {label: "Pytorch Logger", value: "pytorch"},
    {label: "Fabric Logger", value: "fabric"},
]}>

<TabItem value="pytorch">

```
trainer = Trainer(logger=wandb_logger)
```

</TabItem>

<TabItem value="fabric">

```
fabric = L.Fabric(loggers=[wandb_logger])
fabric.launch()
fabric.log_dict({
    "important_metric": important_metric
})
```

</TabItem>

</Tabs>

### ロガーの引数

以下はWandbLoggerで最も使用される引数の一部です。詳細なリストと説明はPyTorch Lightningをご参照ください。

- ([**`Pytorch`**](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb))
- ([**`Fabric`**](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb))

| Parameter   | Description                                                                   |
| ----------- | ----------------------------------------------------------------------------- |
| `project`   | 記録するwandb Projectを定義                                                   |
| `name`      | wandb runに名前を付ける                                                       |
| `log_model` | `log_model="all"`の場合は全モデル、`log_model=True`の場合はトレーニングの最後にログに記録 |
| `save_dir`  | データが保存されるパス                                                     |

### ハイパーパラメーターのログ

<Tabs
  defaultValue="pytorch"
  values={[
    {label: "Pytorch Logger", value: "pytorch"},
    {label: "Fabric Logger", value: "fabric"},
]}>

<TabItem value="pytorch">

```python
class LitModule(LightningModule):
    def __init__(self, *args, **kwarg):
        self.save_hyperparameters()
```

</TabItem>

<TabItem value="fabric">

```python
wandb_logger.log_hyperparams(
    {
        "hyperparameter_1": hyperparameter_1,
        "hyperparameter_2": hyperparameter_2,
    }
)
```

</TabItem>

</Tabs>

### 追加の設定パラメータのログ

```python
# add one parameter
wandb_logger.experiment.config["key"] = value

# add multiple parameters
wandb_logger.experiment.config.update({key1: val1, key2: val2})

# use directly wandb module
wandb.config["key"] = value
wandb.config.update()
```

### 勾配、パラメータのヒストグラム、モデルトポロジのログ

トレーニング中のモデルの勾配とパラメータを監視するために、`wandblogger.watch()`にモデルオブジェクトを渡すことができます。PyTorch Lightningの`WandbLogger`ドキュメントを参照してください。

### メトリクスのログ

<Tabs
  defaultValue="pytorch"
  values={[
    {label: "Pytorch Logger", value: "pytorch"},
    {label: "Fabric Logger", value: "fabric"},
]}>

<TabItem value="pytorch">

メトリクスをW&Bにログするためには、`training_step`や`validation_step`メソッド内で`self.log('my_metric_name', metric_value)`を呼び出します。

以下のコードスニペットは、メトリクスと`LightningModule`のハイパーパラメーターをログする`LightningModule`を定義する方法を示しています。この例では、メトリクスを計算するために[`torchmetrics`](https://github.com/PyTorchLightning/metrics)ライブラリを使用します。

```python
import torch
from torch.nn import Linear, CrossEntropyLoss, functional as F
from torch.optim import Adam
from torchmetrics.functional import accuracy
from lightning.pytorch import LightningModule


class My_LitModule(LightningModule):
    def __init__(self, n_classes=10, n_layer_1=128, n_layer_2=256, lr=1e-3):
        """モデルパラメータを定義するためのメソッド"""
        super().__init__()

        # mnist画像は（1, 28, 28）（チャンネル、幅、高さ）
        self.layer_1 = Linear(28 * 28, n_layer_1)
        self.layer_2 = Linear(n_layer_1, n_layer_2)
        self.layer_3 = Linear(n_layer_2, n_classes)

        self.loss = CrossEntropyLoss()
        self.lr = lr

        # ハイパーパラメーターをself.hparamsに保存（W&Bによる自動ログ）
        self.save_hyperparameters()

    def forward(self, x):
        """推論のためのメソッド input -> output"""

        # (b, 1, 28, 28) -> (b, 1*28*28)
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)

        # 3回の(linear + relu)を実行
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

    def training_step(self, batch, batch_idx):
        """一つのバッチから損失を返す必要がある"""
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # 損失とメトリクスをログ
        self.log("train_loss", loss)
        self.log("train_accuracy", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        """メトリクスをログするために使用"""
        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # 損失とメトリクスをログ
        self.log("val_loss", loss)
        self.log("val_accuracy", acc)
        return preds

    def configure_optimizers(self):
        """モデルのオプティマイザーを定義"""
        return Adam(self.parameters(), lr=self.lr)

    def _get_preds_loss_accuracy(self, batch):
        """トレイン/バリデーション/テストステップが似ているための便利な関数"""
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.loss(logits, y)
        acc = accuracy(preds, y)
        return preds, loss, acc
```

</TabItem>

<TabItem value="fabric">

```python
import lightning as L
import torch
import torchvision as tv
from wandb.integration.lightning.fabric import WandbLogger
import wandb

fabric = L.Fabric(loggers=[wandb_logger])
fabric.launch()

model = tv.models.resnet18()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
model, optimizer = fabric.setup(model, optimizer)

train_dataloader = fabric.setup_dataloaders(
    torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
)

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
        fabric.log_dict({"loss": loss})
```

</TabItem>

</Tabs>

### メトリクスの最小値/最大値をログ

wandbの[`define_metric`](https://docs.wandb.ai/ref/python/run#define\_metric)関数を使用すると、W&Bのサマリーメトリクスにそのメトリクスの最小値、最大値、平均値、または最良値を表示するかどうかを定義できます。 `define_metric`が使用されていない場合、最後にログに記録された値がサマリーメトリクスに表示されます。`define_metric`の[リファレンスドキュメントはこちら](https://docs.wandb.ai/ref/python/run#define\_metric)、および[ガイドはこちら](https://docs.wandb.ai/guides/track/log#customize-axes-and-summaries-with-define\_metric)を参照してください。

W&Bのサマリーメトリクスで最大検証精度を追跡するように指示するには、`wandb.define_metric`を一度呼び出すだけで済みます。たとえば、トレーニングの開始時に呼び出すことができます。

<Tabs
  defaultValue="pytorch"
  values={[
    {label: "Pytorch Logger", value: "pytorch"},
    {label: "Fabric Logger", value: "fabric"},
]}>

<TabItem value="pytorch">

```python
class My_LitModule(LightningModule):
    ...

    def validation_step(self, batch, batch_idx):
        if trainer.global_step == 0:
            wandb.define_metric("val_accuracy", summary="max")

        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # 損失とメトリクスをログ
        self.log("val_loss", loss)
        self.log("val_accuracy", acc)
        return preds
```

</TabItem>

<TabItem value="fabric">

```python
wandb.define_metric("val_accuracy", summary="max")
fabric = L.Fabric(loggers=[wandb_logger])
fabric.launch()
fabric.log_dict({"val_accuracy": val_accuracy})
```

</TabItem>

</Tabs>

### モデルのチェックポイント

モデルのチェックポイントをW&Bの[Artifacts](https://docs.wandb.ai/guides/data-and-model-versioning)として保存するには、Lightningの[`ModelCheckpoint`](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch\_lightning.callbacks.ModelCheckpoint.html#pytorch\_lightning.callbacks.ModelCheckpoint)コールバックを使用し、`WandbLogger`で`log_model`引数を設定します。

```python
# `val_accuracy`が増加した場合にのみモデルをログ
wandb_logger = WandbLogger(log_model="all")
checkpoint_callback = ModelCheckpoint(monitor="val_accuracy", mode="max")
```

<Tabs
  defaultValue="pytorch"
  values={[
    {label: "Pytorch Logger", value: "pytorch"},
    {label: "Fabric Logger", value: "fabric"},
]}>

<TabItem value="pytorch">

```python
trainer = Trainer(logger=wandb_logger, callbacks=[checkpoint_callback])
```

</TabItem>

<TabItem value="fabric">

```python
fabric = L.Fabric(loggers=[wandb_logger], callbacks=[checkpoint_callback])
```

</TabItem>

</Tabs>

_最新_ および _最良_ のエイリアスは、自動的に設定され、W&Bの[Artifact](https://docs.wandb.ai/guides/data-and-model-versioning)からモデルのチェックポイントを簡単に取得できます。

```python
# artifacts パネルでリファレンスを取得可能
# "VERSION"はバージョン（例: "v2"）またはエイリアス（"latest" または "best"）が指定可能
checkpoint_reference = "USER/PROJECT/MODEL-RUN_ID:VERSION"
```

<Tabs
  defaultValue="logger"
  values={[
    {label: "Via Logger", value: "logger"},
    {label: "Via wandb", value: "wandb"},
]}>

<TabItem value="logger">

```python
# ローカルにチェックポイントをダウンロード（キャッシュされていない場合）
wandb_logger.download_artifact(checkpoint_reference, artifact_type="model")
```

</TabItem>

<TabItem value="wandb">

```python
# ローカルにチェックポイントをダウンロード（キャッシュされていない場合）
run = wandb.init(project="MNIST")
artifact = run.use_artifact(checkpoint_reference, type="model")
artifact_dir = artifact.download()
```

</TabItem>

</Tabs>

<Tabs
  defaultValue="pytorch"
  values={[
    {label: "Pytorch Logger", value: "pytorch"},
    {label: "Fabric Logger", value: "fabric"},
]}>

<TabItem value="pytorch">

```python
# チェックポイントを読み込む
model = LitModule.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")
```

</TabItem>

<TabItem value="fabric">

```python
# 生のチェックポイントを取得
full_checkpoint = fabric.load(Path(artifact_dir) / "model.ckpt")

model.load_state_dict(full_checkpoint["model"])
optimizer.load_state_dict(full_checkpoint["optimizer"])
```

</TabItem>

</Tabs>

ログしたモデルのチェックポイントは、[W&B Artifacts](https://docs.wandb.ai/guides/artifacts) UIを通じて確認できます。完全なモデルリネージも含まれています（UI内のモデルチェックポイントの例は[こちら](https://wandb.ai/wandb/arttest/artifacts/model/iv3_trained/5334ab69740f9dda4fed/lineage?_gl=1*yyql5q*_ga*MTQxOTYyNzExOS4xNjg0NDYyNzk1*_ga_JH1SJHJQXJ*MTY5MjMwNzI2Mi4yNjkuMS4xNjkyMzA5NjM2LjM3LjAuMA..)）。

最良のモデルチェックポイントをブックマークし、チーム全体で集中管理するために、チェックポイントを [W&B Model Registry](https://docs.wandb.ai/guides/models) にリンクできます。

ここでは、タスクごとに最良のモデルを整理し、モデルのライフサイクルを管理し、MLライフサイクル全体での追跡と監査を容易にし、Webhookやジョブを使用して下流のアクションを[自動化](https://docs.wandb.ai/guides/models/automation)します。

### 画像、テキストなどのログを作成する

`WandbLogger`には、メディアをログするための`log_image`、`log_text`、`log_table`メソッドがあります。

また、`wandb.log`や`trainer.logger.experiment.log`を直接呼び出して、オーディオや分子、ポイントクラウド、3Dオブジェクトなどのメディアタイプをログすることもできます。

<Tabs
  defaultValue="images"
  values={[
    {label: 'Log Images', value: 'images'},
    {label: 'Log Text', value: 'text'},
    {label: 'Log Tables', value: 'tables'},
  ]}>
  <TabItem value="images">

```python
# テンソル、numpy配列またはPIL画像を使用
wandb_logger.log_image(key="samples", images=[img1, img2])

# キャプションを追加
wandb_logger.log_image(key="samples", images=[img1, img2], caption=["tree", "person"])

# ファイルパスを使用
wandb_logger.log_image(key="samples", images=["img_1.jpg", "img_2.jpg"])

# trainer内で.logを使用
trainer.logger.experiment.log(
    {"samples": [wandb.Image(img, caption=caption) for (img, caption) in my_images]},
    step=current_trainer_global_step,
)
```
  </TabItem>
  <TabItem value="text">

```python
# データはリストのリストである必要があります
columns = ["input", "label", "prediction"]
my_data = [["cheese", "english", "english"], ["fromage", "french", "spanish"]]

# カラムとデータを使用
wandb_logger.log_text(key="my_samples", columns=columns, data=my_data)

# pandas DataFrameを使用
wandb_logger.log_text(key="my_samples", dataframe=my_dataframe)
```

  </TabItem>
  <TabItem value="tables">

```python
# テキストキャプション、画像、オーディオを含むW&B Tableをログ
columns = ["caption", "image", "sound"]

# データはリストのリストである必要があります
my_data = [
    ["cheese", wandb.Image(img_1), wandb.Audio(snd_1)],
    ["wine", wandb.Image(img_2), wandb.Audio(snd_2)],
]

# Tableをログ
wandb_logger.log_table(key="my_samples", columns=columns, data=data)
```

  </TabItem>
</Tabs>


LightningのCallbackシステムを使用して、WandbLogger経由でWeights & Biasesにログを作成するタイミングを制御できます。この例では、検証画像と予測のサンプルをログしています。

```python
import torch
import wandb
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

# or
# from wandb.integration.lightning.fabric import WandbLogger


class LogPredictionSamplesCallback(Callback):
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """検証バッチが終了したときに呼び出されます。"""

        # `outputs`は`LightningModule.validation_step`から来ます
        # この場合はモデルの予測に相当します

        # 最初のバッチから20個のサンプル画像予測をログします
        if batch_idx == 0:
            n = 20
            x, y = batch
            images = [img for img in x[:n]]
            captions = [
                f"Ground Truth: {y_i} - Prediction: {y_pred}"
                for y_i, y_pred in zip(y[:n], outputs[:n])
            ]

            # オプション1: `WandbLogger.log_image`で画像をログ
            wandb_logger.log_image(key="sample_images", images=images, caption=captions)

            # オプション2: 画像と予測をW&B Tableとしてログ
            columns = ["image", "ground truth", "prediction"]
            data = [
                [wandb.Image(x_i), y_i, y_pred] or x_i,
                y_i,
                y_pred in list(zip(x[:n], y[:n], outputs[:n])),
            ]
            wandb_logger.log_table(key="sample_table", columns=columns, data=data)


trainer = pl.Trainer(callbacks=[LogPredictionSamplesCallback()])
```

### LightningとW&Bで複数のGPUを使用する方法？

PyTorch Lightningは、そのDDPインターフェースを通じてマルチGPUをサポートしています。ただし、PyTorch Lightningの設計では、GPUのインスタンス化方法に注意が必要です。

Lightningは、トレーニングループ内の各GPU（またはランク）が同じ初期条件で完全に同じ方法でインスタンス化される必要があると仮定しています。しかし、ランク0のプロセスのみが`wandb.run`オブジェクトにアクセスでき、ランク0以外のプロセスでは`wandb.run = None`になります。このため、ランク0以外のプロセスが失敗する可能性があります。このような状況では、ランク0のプロセスがランク0以外のプロセスが参加するのを待つため、**デッドロック**に陥る可能性があります。

この理由から、トレーニングコードの設定方法には注意が必要です。推奨される方法は、コードが`wandb.run`オブジェクトに依存しないようにすることです。

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

        self.log("train/loss", loss)
        return {"train_loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)

        self.log("val/loss", loss)
        return {"val_loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


def main():
    # すべてのランダムシードを同じ値に設定します
    # これは分散トレーニング環境では重要です
    # 各ランクは独自の初期ウェイトセットを取得します
    # これらが一致しない場合、勾配も一致しないため、
    # トレーニングが収束しない可能性があります
    pl.seed_everything(1)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    model = MNISTClassifier()
    wandb_logger = WandbLogger(project="<project_name>")
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints",
            every_n_train_steps=100,
        ),
    ]
    trainer = pl.Trainer(
        max_epochs=3, gpus=2, logger=wandb_logger, strategy="ddp", callbacks=callbacks
    )
    trainer.fit(model, train_loader, val_loader)
```

## インタラクティブな例をチェック！

ビデオチュートリアルで、チュートリアルコラボに従うことができます。[こちら](https://wandb.me/lit-colab)

## よくある質問

### W&BはLightningとどのように統合されますか？

コアインテグレーションは[Lightning `loggers` API](https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html)に基づいています。これにより、多くのログコードをフレームワークに依存しない方法で記述できます。`Logger`は[Lightning `Trainer`](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html)に渡され、APIの豊富な[フックとコールバックシステム](https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html)に基づいてトリガーされます。これにより、研究コードがエンジニアリングおよびログコードから適切に分離されます。

### 追加のコードなしでインテグレーションは何をログしますか？

W&Bにモデルのチェックポイントを保存します。これにより、将来のRunsに使用するためにそれらを表示またはダウンロードできます。また、GPU使用率やネットワークI/Oなどの[システムメトリクス](../app/features/system-metrics.md)や、ハードウェアやOS情報などの環境情報、gitコミットと差分パッチ、ノートブックの内容やセッション履歴を含む[コードの状態](../app/features/panels/code.md)、および標準出力に印刷されたものすべてをキャプチャします。

### トレーニングセットアップで`wandb.run`を使用する必要がある場合はどうすればよいですか？

アクセスする必要がある変数のスコープを拡張する必要があります。つまり、初期条件がすべてのプロセスで同じであることを確認することです。

```python
if os.environ.get("LOCAL_RANK", None) is None:
    os.environ["WANDB_DIR"] = wandb.run.dir
```

次に、モデルのチェックポイントディレクトリを設定するために`os.environ["WANDB_DIR"]`を使用できます。これにより、非ゼロランクのプロセスでも`wandb.run.dir`を使用できるようになります。