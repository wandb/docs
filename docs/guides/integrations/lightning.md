---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# PyTorch Lightning

[**コラボノートブックで試す →**](https://wandb.me/lightning)

PyTorch Lightning は、PyTorch コードを整理し、分散トレーニングや16ビット精度などの高度な機能を簡単に追加するための軽量ラッパーを提供します。W&B は、あなたの機械学習実験を記録するための軽量ラッパーを提供しますが、両者を自分で組み合わせる必要はありません。Weights & Biases は PyTorch Lightning ライブラリに直接組み込まれており、[**`WandbLogger`**](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb) 経由で利用できます。

## ⚡ 数行で素早く始めることができます。

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
**wandb.log() の使用について:** `WandbLogger` は Trainer の `global_step` を使用して W&B にログを記録します。コード内で直接 `wandb.log` を追加で呼び出す場合、`wandb.log()` 内で `step` 引数を使用しないでください。

代わりに、次のように Trainer の `global_step` を他のメトリクスと同様にログに記録します：

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

![どこからでもアクセス可能なインタラクティブダッシュボードとその他の機能](@site/static/images/integrations/n6P7K4M.gif)

## wandb へのサインアップとログイン

a) 無料アカウントに[**サインアップ**](https://wandb.ai/site)します。

b) `wandb` ライブラリを Pip でインストールします。

c) トレーニングスクリプトでログインするには、www.wandb.ai でアカウントにサインインしている必要があります。その後、[**Authorizeページ**](https://wandb.ai/authorize) で API キーを見つけることができます。

Weights & Biases を初めて使用する場合、[**クイックスタート**](../../quickstart.md) を確認してください。

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

## PyTorch Lightning の `WandbLogger` の使用

PyTorch Lightning には複数の `WandbLogger` ([**`Pytorch`**](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)) ([**`Fabric`**](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)) クラスがあり、メトリクス、モデルの重み、メディアなどをシームレスにログできます。単に WandbLogger をインスタンス化し、Lightning の `Trainer` または `Fabric` に渡します。

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

### Logger の引数

以下に WandbLogger で最もよく使用されるパラメータを示します。詳細と説明は PyTorch Lightning のドキュメントを参照してください。

- ([**`Pytorch`**](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb))
- ([**`Fabric`**](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb))

| パラメータ   | 説明                                                                   |
| ----------- | ----------------------------------------------------------------------- |
| `project`   | ログを記録する wandb プロジェクトを定義                                 |
| `name`      | wandb run に名前を付ける                                                |
| `log_model` | `log_model="all"` の場合は全てのモデルを、`log_model=True` の場合はトレーニング終了時にモデルをログします |
| `save_dir`  | データを保存するパス                                                    |

### ハイパーパラメータのログ

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

### 追加のコンフィグパラメータをログに記録

```python
# パラメータを1つ追加
wandb_logger.experiment.config["key"] = value

# 複数のパラメータを追加
wandb_logger.experiment.config.update({key1: val1, key2: val2})

# 直接 wandb モジュールを使用
wandb.config["key"] = value
wandb.config.update()
```

### 勾配、パラメータのヒストグラム、モデルのトポロジーをログ

モデルオブジェクトを `wandblogger.watch()` に渡すことで、トレーニング中にモデルの勾配とパラメータを監視できます。PyTorch Lightning の `WandbLogger` ドキュメントを参照してください。

### メトリクスのログ

<Tabs
  defaultValue="pytorch"
  values={[
    {label: "Pytorch Logger", value: "pytorch"},
    {label: "Fabric Logger", value: "fabric"},
]}>

<TabItem value="pytorch">

`training_step` や `validation_step` メソッド内で `self.log('my_metric_name', metric_vale)` と呼び出すことで、`LightningModule` 内でメトリクスを W&B にログアップすることができます。

以下のコードスニペットは、メトリクスと `LightningModule` のハイパーパラメータをログする方法を示しています。この例では、メトリクスを計算するために [`torchmetrics`](https://github.com/PyTorchLightning/metrics) ライブラリを使用します。

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

        # MNIST 画像は (1, 28, 28) (チャンネル, 幅, 高さ)
        self.layer_1 = Linear(28 * 28, n_layer_1)
        self.layer_2 = Linear(n_layer_1, n_layer_2)
        self.layer_3 = Linear(n_layer_2, n_classes)

        self.loss = CrossEntropyLoss()
        self.lr = lr

        # ハイパーパラメータを self.hparams に保存（W&B によって自動的にログされます）
        self.save_hyperparameters()

    def forward(self, x):
        """推論入力 -> 出力に使用されるメソッド"""

        # (b, 1, 28, 28) -> (b, 1*28*28)
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)

        # 3 x (線形 + ReLU) を行う
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

    def training_step(self, batch, batch_idx):
        """1つのバッチからのロスを返す必要があります"""
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # ロスとメトリクスをログに記録
        self.log("train_loss", loss)
        self.log("train_accuracy", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        """メトリクスをログするために使用されます"""
        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # ロスとメトリクスをログに記録
        self.log("val_loss", loss)
        self.log("val_accuracy", acc)
        return preds

    def configure_optimizers(self):
        """モデルオプティマイザーを定義する"""
        return Adam(self.parameters(), lr=self.lr)

    def _get_preds_loss_accuracy(self, batch):
        """train/valid/test ステップが似ているための便利な関数"""
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

### メトリクスの最小/最大をログ

wandb の [`define_metric`](https://docs.wandb.ai/ref/python/run#define\_metric) 関数を使用して、メトリクスの最小、最大、平均または最良の値を W&B サマリーメトリクスに表示するかどうかを定義できます。`define_metric` が使用されていない場合、最後にログされた値がサマリーメトリクスに表示されます。`define_metric` の[参考文書はこちら](https://docs.wandb.ai/ref/python/run#define\_metric)、および[ガイドはこちら](https://docs.wandb.ai/guides/track/log#customize-axes-and-summaries-with-define\_metric) を参照ください。

W&B サマリーメトリクスで最大検証精度をトラッキングするためには、トレーニング開始時に次のように `wandb.define_metric` を1回呼ぶだけです：

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

        # ロスとメトリクスをログ
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

### モデルチェックポイント

モデルチェックポイントを W&B [Artifacts](https://docs.wandb.ai/guides/data-and-model-versioning) として保存するには、Lightning の [`ModelCheckpoint`](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch\_lightning.callbacks.ModelCheckpoint.html#pytorch\_lightning.callbacks.ModelCheckpoint) コールバックを使用し、`WandbLogger` の `log_model` 引数を設定します：

```python
# `val_accuracy` が増加した場合のみモデルをログ
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

最新と最良のエイリアスが自動的に設定され、W&B [Artifact](https://docs.wandb.ai/guides/data-and-model-versioning) からモデルチェックポイントを簡単に取得できます：

```python
# artifacts パネルで参照を取得できます
# "VERSION" はバージョン（例：「v2」）またはエイリアス（例：「latest」または「best」）のいずれかです
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
# チェックポイントをローカルにダウンロード（既にキャッシュされていない場合）
wandb_logger.download_artifact(checkpoint_reference, artifact_type="model")
```

</TabItem>

<TabItem value="wandb">

```python
# チェックポイントをローカルにダウンロード（既にキャッシュされていない場合）
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
# チェックポイントをロード
model = LitModule.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")
```

</TabItem>

<TabItem value="fabric">

```python
# 生のチェックポイントを要求
full_checkpoint = fabric.load(Path(artifact_dir) / "model.ckpt")

model.load_state_dict(full_checkpoint["model"])
optimizer.load_state_dict(full_checkpoint["optimizer"])
```

</TabItem>

</Tabs>

ログしたモデルチェックポイントは W&B [Artifacts](https://docs.wandb.ai/guides/artifacts) UI を通じて表示・ダウンロードでき、完全なモデルリネージを含みます（UI 内のモデルチェックポイントの例 [こちら](https://wandb.ai/wandb/arttest/artifacts/model/iv3_trained/5334ab69740f9dda4fed/lineage?_gl=1*yyql5q*_ga*MTQxOTYyNzExOS4xNjg0NDYyNzk1*_ga_JH1SJHJQXJ*MTY5MjMwNzI2Mi4yNjkuMS4xNjkyMzA5NjM2LjM3LjAuMA..)）。

最良のモデルチェックポイントをブックマークしチーム全体で中央管理するために、[W&B Model Registry](https://docs.wandb.ai/guides/models) にリンクできます。

ここでは、タスク別に最良のモデルを整理し、モデルのライフサイクルを管理し、MLライフサイクル全体を通じて簡単なトラッキングと監査を実現し、Webhookやジョブを使用して[自動化](https://docs.wandb.ai/guides/models/automation)することができます。

### 画像、テキストなどをログ

`WandbLogger` には `log_image`、`log_text`、`log_table` メソッドがあり、メディアをログすることができます。

また、他のメディアタイプ（音声、分子、ポイントクラウド、3Dオブジェクトなど）をログするために、直接 `wandb.log` または `trainer.logger.experiment.log` を呼び出すこともできます。

<Tabs
  defaultValue="images"
  values={[
    {label: '画像のログ', value: 'images'},
    {label: 'テキストのログ', value: 'text'},
    {label: 'テーブルのログ', value: 'tables'},
  ]}>
  <TabItem value="images">

```python
# テンソル、numpy配列、またはPIL画像を使用
wandb_logger.log_image(key="samples", images=[img1, img2])

# キャプションを追加
wandb_logger.log_image(key="samples", images=[img1, img2], caption=["tree", "person"])

# ファイルパスを使用
wandb_logger.log_image(key="samples", images=["img_1.jpg", "img_2.jpg"])

# トレーナーで .log を使用
trainer.logger.experiment.log(
    {"samples": [wandb.Image(img, caption=caption) for (img, caption) in my_images]},
    step=current_trainer_global_step,
)
```
  </TabItem>
  <TabItem value="text">

```python
# data はリストのリストである必要があります
columns = ["input", "label", "prediction"]
my_data = [["cheese", "english", "english"], ["fromage", "french", "spanish"]]

# 列とデータを使用
wandb_logger.log_text(key="my_samples", columns=columns, data=my_data)

# pandas DataFrame を使用
wandb_logger.log_text(key="my_samples", dataframe=my_dataframe)
```

  </TabItem>
  <TabItem value="tables">

```python
# テキストキャプション、画像、音声を持つW&Bテーブルをログ
columns = ["caption", "image", "sound"]

# data はリストのリストである必要があります
my_data = [
    ["cheese", wandb.Image(img_1), wandb.Audio(snd_1)],
    ["wine", wandb.Image(img_2), wandb.Audio(snd_2)],
]

# テーブルのログ
wandb_logger.log_table(key="my_samples", columns=columns, data=data)
```

  </TabItem>
</Tabs>

Lightning の Callbacks システムを使用して、WandbLogger を介して Weights & Biases にログを記録するタイミングを制御できます。この例では、検証画像と予測のサンプルをログに記録します：

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
        """検証バッチが終了したときに呼び出されます。"""

        # `outputs` は `LightningModule.validation_step` からのもので、
        # この場合は私たちのモデルの予測に相当します。

        # 最初のバッチから20サンプル画像の予測をログ
        if batch_idx == 0:
            n = 20
            x, y = batch
            images = [img for img in x[:n]]
            captions = [
                f"Ground Truth: {y_i} - Prediction: {y_pred}"
                for y_i, y_pred in zip(y[:n], outputs[:n])
            ]

            # オプション1: `WandbLogger.log_image` で画像をログ
            wandb_logger.log_image(key="sample_images", images=images, caption=captions)

            # オプション2: 画像と予測をW&Bテーブルとしてログ
            columns = ["image", "ground truth", "prediction"]
            data = [
                [wandb.Image(x_i), y_i, y_pred] or x_i,
                y_i,
                y_pred in list(zip(x[:n], y[:n], outputs[:n])),
            ]
            wandb_logger.log_table(key="sample_table", columns=columns, data=data)


trainer = pl.Trainer(callbacks=[LogPredictionSamplesCallback()])
```

### LightningとW&Bを使用して複数のGPUをどのように使うか？

PyTorch Lightning には DDP インターフェースを通じたマルチGPUサポートがあります。ただし、PyTorch Lightning の設計には、GPU（またはランク）のトレーニングループが全く同じ初期条件でインスタンス化されることを前提としています。しかし、ランク0プロセスだけが `wandb.run` オブジェクトにアクセスでき、ランク0以外のプロセスは `wandb.run = None` になります。このような状況は、ランク0プロセスが他のランクのプロセスを待ち続けるため、デッドロックに陥る可能性があります。

したがって、トレーニングコードのセットアップに注意が必要です。推奨される方法は、コードを `wandb.run` オブジェクトに依存しないように設定することです。

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
    # すべてのランダムシードを同じ値にする設定
    # これは分散トレーニング環境で重要です
    # 各ランクは独自の初期重みを取得します。
    # それらが一致しないと、勾配も一致せずに
    # トレーニングが収束しない可能性があります。
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

## インタラクティブな例を確認！

ビデオチュートリアルに沿って進めるか、[こちら](https://wandb.me/lit-colab) のチュートリアルコラボで進めることができます。

## よくある質問

### W&B はどのように Lightning に統合されますか？

コア統合は[Lightning `loggers` API](https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html) に基づいており、多くのログ記録コードをフレームワークに依存しない方法で記述することができます。`Logger` は [Lightning `Trainer`](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html) に渡され、その API の豊富な[フックおよびコールバックシステム](https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html) に基づいてトリガーされます。これにより、研究コードがエンジニアリングおよびログ記録コードから分離されます。

### 追加のコードなく統合は何をログしますか？

モデルのチェックポイントをW&Bに保存し、将来のランで使用するために表示またはダウンロードできます。また、[システムメトリクス](../app/features/system-metrics.md)（GPU使用率やネットワークI/Oなど）、ハードウェアやOS情報などの環境情報、[コード状態](../app/features/panels/code.md)（gitコミット情報やdiffパッチ、ノートブックの内容やセッション履歴を含む）、標準出力に印刷される全てをキャプチャします。

### トレーニングセットアップで `wandb.run` をどうしても使用する必要がある場合はどうすればよいですか？

必要な変数のスコープを自分で拡大することになります。つまり、全てのプロセスで同じ初期条件を設定することを確認してください。

```python
if os.environ.get("LOCAL_RANK", None) is None:
    os.environ["WANDB_DIR"] = wandb.run.dir
```

