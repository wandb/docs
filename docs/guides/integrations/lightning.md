---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# PyTorch Lightning

[**こちらでColabノートブックを試す →**](https://wandb.me/lightning)

PyTorch Lightningは、PyTorchコードを整理し、分散トレーニングや16ビット精度などの高度な機能を簡単に追加できる軽量のラッパーを提供します。W&Bは、ML実験をログに記録するための軽量のラッパーを提供しますが、両者を組み合わせる必要はありません。Weights & Biasesは、[**`WandbLogger`**](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb) を介して、直接PyTorch Lightningライブラリに組み込まれています。

## ⚡ 少ない行数でライトニングファストに開始

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
**wandb.log()の使用:** `WandbLogger`はTrainerの`global_step`を使用してW&Bにログを記録します。コード内で直接`wandb.log`に追加の呼び出しを行う場合は、`wandb.log()`で`step`引数を使用し **ないでください** 。

代わりに、他のメトリクスと同様にTrainerの`global_step`をログに記録してください。このように：

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

![インタラクティブなダッシュボードにどこからでもアクセス可能で、それ以上の機能も！](@site/static/images/integrations/n6P7K4M.gif)

## wandbにサインアップとログイン

a) [**サインアップ**](https://wandb.ai/site) 無料アカウントを作成

b) `wandb`ライブラリをPipインストール

c) トレーニングスクリプトにログインするには、www.wandb.aiでアカウントにサインインしている必要があり、そこでAPIキーが見つかります。 [**Authorizeページ**](https://wandb.ai/authorize) でAPIキーを確認できます。

初めてWeights & Biasesを使用する場合は、[**クイックスタート**](../../quickstart.md) を確認することをお勧めします。

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

## PyTorch Lightningの`WandbLogger`の使用方法

PyTorch Lightningには、メトリクス、モデル重み、メディアなどをシームレスにログ記録できる複数の`WandbLogger` ( [**`Pytorch`**](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb) [**`Fabric`**](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb) ) クラスがあります。WandbLoggerをインスタンス化して、Lightningの`Trainer`または`Fabric`に渡すだけです。

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

### ロガー引数

以下は、WandbLoggerで最もよく使用されるパラメータの一部です。完全なリストと説明については、PyTorch Lightningを参照してください。

- ( [**`Pytorch`**](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb) )
- ( [**`Fabric`**](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb) )

| パラメータ     | 説明                                                                             |
| ------------- | ----------------------------------------------------------------------------- |
| `project`     | ログを記録するwandbプロジェクトを定義                                           |
| `name`        | wandb runに名前を付ける                                                        |
| `log_model`   | `log_model="all"`の場合すべてのモデルを記録するか、`log_model=True`の場合トレーニング終了時に記録 |
| `save_dir`    | データが保存されるパス                                                         |

### ハイパーパラメータをログする

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

### 追加の設定パラメータをログする

```python
# パラメータを1つ追加
wandb_logger.experiment.config["key"] = value

# 複数のパラメータを追加
wandb_logger.experiment.config.update({key1: val1, key2: val2})

# 直接wandbモジュールを使用
wandb.config["key"] = value
wandb.config.update()
```

### 勾配、パラメータヒストグラム、モデルトポロジーをログする

トレーニング中にモデルの勾配やパラメータを監視するために、モデルオブジェクトを`wandblogger.watch()`に渡すことができます。PyTorch Lightningの`WandbLogger`ドキュメントを参照してください。

### メトリクスをログする

<Tabs
  defaultValue="pytorch"
  values={[
    {label: "Pytorch Logger", value: "pytorch"},
    {label: "Fabric Logger", value: "fabric"},
]}>

<TabItem value="pytorch">

`WandbLogger`を使用するとき、`LightiningModule`内の`self.log('my_metric_name', metric_vale)`を呼び出してメトリクスをW&Bにログ記録できます。これは例えば、`training_step`または`validation_step`メソッド内で行います。

以下のコードスニペットは、メトリクスと`LightningModule`のハイパーパラメータをログするために、`LightningModule`を定義する方法を示しています。この例では、メトリクスを計算するために[`torchmetrics`](https://github.com/PyTorchLightning/metrics)ライブラリを使用します。

```python
import torch
from torch.nn import Linear, CrossEntropyLoss, functional as F
from torch.optim import Adam
from torchmetrics.functional import accuracy
from lightning.pytorch import LightningModule


class My_LitModule(LightningModule):
    def __init__(self, n_classes=10, n_layer_1=128, n_layer_2=256, lr=1e-3):
        """モデルパラメータを定義するメソッド"""
        super().__init__()

        # mnist画像は(1, 28, 28) (チャンネル, 幅, 高さ)
        self.layer_1 = Linear(28 * 28, n_layer_1)
        self.layer_2 = Linear(n_layer_1, n_layer_2)
        self.layer_3 = Linear(n_layer_2, n_classes)

        self.loss = CrossEntropyLoss()
        self.lr = lr

        # ハイパーパラメータをself.hparamsに保存 (W&Bによって自動的にログされる)
        self.save_hyperparameters()

    def forward(self, x):
        """推論入力 -> 出力に使用するメソッド"""

        # (b, 1, 28, 28) -> (b, 1*28*28)
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)

        # 3x (線形 + ReLU) を行う
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

    def training_step(self, batch, batch_idx):
        """単一バッチからのロスを返す必要がある"""
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # ロスとメトリクスをログする
        self.log("train_loss", loss)
        self.log("train_accuracy", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        """メトリクスをログするために使用"""
        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # ロスとメトリクスをログする
        self.log("val_loss", loss)
        self.log("val_accuracy", acc)
        return preds

    def configure_optimizers(self):
        """モデルオプティマイザーを定義"""
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

### メトリクスの最小/最大値をログする

wandbの[`define_metric`](https://docs.wandb.ai/ref/python/run#define\_metric)関数を使用して、特定のメトリクスの最小、最大、平均、最適値をW&Bサマリーメトリクスに表示するかどうかを定義できます。`define_metric` が使用されていない場合、最後にログされた値がサマリーメトリクスに表示されます。詳細は [reference docs here](https://docs.wandb.ai/ref/python/run#define\_metric) と [ガイド](https://docs.wandb.ai/guides/track/log#customize-axes-and-summaries-with-define\_metric) を参照してください。

トレーニングの開始時に一度だけ`wandb.define_metric`を呼び出すことで、W&Bサマリーメトリクスにおける最大の検証精度を追跡するようW&Bに指示できます。以下のように呼び出します：

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

        # ロスとメトリクスをログする
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

モデルのチェックポイントをW&B [Artifacts](https://docs.wandb.ai/guides/data-and-model-versioning)に保存するために、Lightningの[`ModelCheckpoint`](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch\_lightning.callbacks.ModelCheckpoint.html#pytorch\_lightning.callbacks.ModelCheckpoint) コールバックを使用し、`WandbLogger`の`log_model`引数を設定します：

```python
# `val_accuracy`が増加した場合のみモデルをログする
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

最新および最適なエイリアスは、W&Bの[Artifact](https://docs.wandb.ai/guides/data-and-model-versioning)からモデルのチェックポイントを簡単に取得できるように自動的に設定されます：

```python
# エイリアスはArtifactsパネルで参照できます
# "VERSION"はバージョン (例: "v2") またはエイリアス ("最新" あるいは "最適") です
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
# チェックポイントをローカルにダウンロード (キャッシュされていない場合)
wandb_logger.download_artifact(checkpoint_reference, artifact_type="model")
```

</TabItem>

<TabItem value="wandb">

```python
# チェックポイントをローカルにダウンロード (キャッシュされていない場合)
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
# 生のチェックポイントをリクエスト
full_checkpoint = fabric.load(Path(artifact_dir) / "model.ckpt")

model.load_state_dict(full_checkpoint["model"])
optimizer.load_state_dict(full_checkpoint["optimizer"])
```

</TabItem>

</Tabs>

ログしたモデルのチェックポイントは、[W&B Artifacts](https://docs.wandb.ai/guides/artifacts) UIを通じて表示可能で、完全なモデルのリネージが含まれます。UIでのモデルのチェックポイントの例はこちらで確認できます [here](https://wandb.ai/wandb/arttest/artifacts/model/iv3_trained/5334ab69740f9dda4fed/lineage?_gl=1*yyql5q*_ga*MTQxOTYyNzExOS4xNjg0NDYyNzk1*_ga_JH1SJHJQXJ*MTY5MjMwNzI2Mi4yNjkuMS4xNjkyMzA5NjM2LjM3LjAuMA..)。

最高のモデルのチェックポイントをブックマークし、それをチーム全体で集中管理するために、それらをW&B [Model Registry](https://docs.wandb.ai/guides/models) にリンクすることができます。ここでは、最適なモデルをタスクごとに整理し、モデルライフサイクルを管理し、MLライフサイクル全体での追跡と監査を容易にし、webhooksやジョブでのアクションを[自動化](https://docs.wandb.ai/guides/models/automation) できます。

### 画像、テキスト、およびその他のログ

`WandbLogger`は、メディア用の`log_image`、`log_text`、および`log_table`メソッドを備えています。 オーディオ、分子、ポイントクラウド、3Dオブジェクトなどの他のメディアタイプをログに記録するために、`wandb.log`や`trainer.logger.experiment.log`を直接呼び出すこともできます。

<Tabs
  defaultValue="images"
  values={[
    {label: 'Log Images', value: 'images'},
    {label: 'Log Text', value: 'text'},
    {label: 'Log Tables', value: 'tables'},
  ]}>
  <TabItem value="images">

```python
# テンソル、numpy配列、またはPIL画像を使用する
wandb_logger.log_image(key="samples", images=[img1, img2])

# キャプションを追加
wandb_logger.log_image(key="samples", images=[img1, img2], caption=["tree", "person"])

# ファイルパスを使用する
wandb_logger.log_image(key="samples", images=["img_1.jpg", "img_2.jpg"])

# トレーナーで.logを使用
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

# 列とデータを使用する
wandb_logger.log_text(key="my_samples", columns=columns, data=my_data)

# pandasデータフレームを使用する
wandb_logger.log_text(key="my_samples", dataframe=my_dataframe)
```

  </TabItem>
  <TabItem value="tables">

```python
# テキストキャプション、画像、音声が含まれているW&Bテーブルをログする
columns = ["caption", "image", "sound"]

# データはリストのリストである必要があります
my_data = [
    ["cheese", wandb.Image(img_1), wandb.Audio(snd_1)],
    ["wine", wandb.Image(img_2), wandb.Audio(snd_2)],
]

# テーブルをログする
wandb_logger.log_table(key="my_samples", columns=columns, data=data)
```

  </TabItem>
</Tabs>

Lightningのコールバックシステムを使用して、WandbLogger経由でWeights & Biasesにログするタイミングを制御できます。以下の例では、検証画像と予測のサンプルをログします：

```python
import torch
import wandb
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

# または
# from wandb.integration.lightning.fabric import WandbLogger

class LogPredictionSamplesCallback(Callback):
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """検証バッチ終了時に呼び出される"""

        # `outputs`は`LightningModule.validation_step`からのもので
        # これはこの場合のモデル予測に対応します

        # 最初のバッチから20枚のサンプル画像予測をログします
        if batch_idx == 0:
            n = 20
            x, y = batch
            images = [img for img in x[:n]]
            captions = [
                f"Ground Truth: {y_i} - Prediction: {y_pred}"
                for y_i, y_pred in zip(y[:n], outputs[:n])
            ]

            # オプション1: `WandbLogger.log_image`を使用して画像をログする
            wandb_logger.log_image(key="sample_images", images=images, caption=captions)

            # オプション2: 画像と予測をW&Bテーブルとしてログする
            columns = ["image", "ground truth", "prediction"]
            data = [
                [wandb.Image(x_i), y_i, y_pred] for (x_i, y_i, y_pred) in list(zip(x[:n], y[:n], outputs[:n]))
            ]
            wandb_logger.log_table(key="sample_table", columns=columns, data=data)

trainer = pl.Trainer(callbacks=[LogPredictionSamplesCallback()])
```

### LightningとW&Bを使用して複数のGPUを使用する方法は？

PyTorch Lightningは、DDPインターフェースを介してマルチGPUをサポートしています。ただし、PyTorch Lightningの設計では、GPU（またはランク）を同じ初期条件で正確にインスタンス化する必要があります。しかし、ランク0プロセスのみが`wandb.run`オブジェクトにアクセスでき、非ゼロランクプロセスには`wandb.run = None`が与えられます。これは、非ゼロプロセスが失敗する可能性があり、この状況ではランク0プロセスが非ゼロプロセスの参加を待っている間にデッドロックが発生する可能性があります。

このため、トレーニングコードの設定方法に注意する必要があります。他のプロセスと同じ初期条件で独立して設定することをお勧めします。

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
    # すべての乱数シードを同じ値に設定
    # これは分散トレーニング設定で重要です
    # 各ランクは独自のセットの初期重みを取得します。
    # それらが一致しない場合、勾配も一致せず、収束しないトレーニングに繋がる可能性があります。
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

チュートリアル動画に沿って進み、[こちらのチュートリアルcolab](https://wandb.me/lit-colab)を見て下さい。

## よくある質問

### W&BはLightningとどのように統合されていますか？

コアの統合は、[Lightning `loggers` API](https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html)に基づいており、多くのログ記録コードをフレームワークに依存しない方法で記述できます。`Logger`は[Lightning `Trainer`](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html)に渡され、そのAPIの豊富な[hook-and-callback system](https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html)に基づいてトリガーされます。これにより、リサーチコードがエンジニアリングおよびログ記録コードと明確に分離されます。

### 追加コードなしで統合ログには何が含まれますか？

モデルのチェックポイントをW&Bに保存し、将来のrunで使用するために表示またはダウンロードできます。また、[GPUの使用状況やネットワークI/O](../app/features/system-metrics.md)などのシステムメトリクス、ハードウェアやOS情報などの環境情報、[コードの状態](../app/features/panels/code.md)（gitコミットと差分パッチ、ノートブックの内容とセッション履歴を含む）、および標準出力に出力されるものすべてをキャプチャします。

### トレーニングセットアップでどうしても`wandb.run`を使用する必要がある場合はどうすれば良いですか？

アクセスする必要がある変数のスコープを自分で拡張する必要があります。言い換えると、すべてのプロセスで同じ初期条件を確保することです。

```python
if os.environ.get("LOCAL_RANK", None) is None:
    os.environ["WANDB_DIR"] = wandb.run.dir
```

