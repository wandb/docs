---
title: PyTorch Lightning
menu:
  default:
    identifier: ja-guides-integrations-lightning
    parent: integrations
weight: 340
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Optimize_PyTorch_Lightning_models_with_Weights_%26_Biases.ipynb" >}}

PyTorch Lightning は、PyTorch のコードを整理し、分散トレーニングや 16 ビット精度などの高度な機能を簡単に追加できる軽量ラッパーを提供します。W&B は、ML 実験をログ記録するための軽量ラッパーを提供します。Weights & Biases は、[**`WandbLogger`**](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb) を介して PyTorch Lightning ライブラリに直接組み込まれているため、この 2 つを自分で組み合わせる必要はありません。

## Lightning との統合

{{< tabpane text=true >}}
{{% tab header="PyTorch Logger" value="pytorch" %}}

```python
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer

wandb_logger = WandbLogger(log_model="all")
trainer = Trainer(logger=wandb_logger)
```

{{% alert %}}
**wandb.log() の使用:** `WandbLogger` は、Trainer の `global_step` を使用して W&B にログを記録します。コード内で `wandb.log` を直接追加で呼び出す場合は、`wandb.log()` で `step` 引数を使用**しないでください**。

代わりに、他のメトリクスと同様に、Trainer の `global_step` をログに記録します。

```python
wandb.log({"accuracy":0.99, "trainer/global_step": step})
```
{{% /alert %}}

{{% /tab %}}

{{% tab header="Fabric Logger" value="fabric" %}}

```python
import lightning as L
from wandb.integration.lightning.fabric import WandbLogger

wandb_logger = WandbLogger(log_model="all")
fabric = L.Fabric(loggers=[wandb_logger])
fabric.launch()
fabric.log_dict({"important_metric": important_metric})
```

{{% /tab %}}

{{< /tabpane >}}

{{< img src="/images/integrations/n6P7K4M.gif" alt="Interactive dashboards accessible anywhere, and more!" >}}

### サインアップして APIキー を作成

APIキー は、W&B に対してマシンを認証します。APIキー は、ユーザープロフィールから生成できます。

{{% alert %}}
より効率的なアプローチとして、[https://wandb.ai/authorize](https://wandb.ai/authorize) に直接アクセスして APIキー を生成できます。表示された APIキー をコピーして、パスワードマネージャーなどの安全な場所に保存します。
{{% /alert %}}

1. 右上隅にあるユーザープロフィールアイコンをクリックします。
2. **ユーザー設定**を選択し、**APIキー**セクションまでスクロールします。
3. **表示**をクリックします。表示された APIキー をコピーします。APIキー を非表示にするには、ページをリロードします。

### `wandb` ライブラリをインストールしてログインする

ローカルに `wandb` ライブラリをインストールしてログインするには:

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) を APIキー に設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

2. `wandb` ライブラリをインストールしてログインします。

    ```shell
    pip install wandb

    wandb login
    ```

{{% /tab %}}

{{% tab header="Python" value="python" %}}

```bash
pip install wandb
```
```python
import wandb
wandb.login()
```

{{% /tab %}}

{{% tab header="Python notebook" value="notebook" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

## PyTorch Lightning の `WandbLogger` を使用する

PyTorch Lightning には、メトリクスやモデルの重み、メディアなどをログに記録するための複数の `WandbLogger` クラスがあります。

- [`PyTorch`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)
- [`Fabric`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)

Lightning と統合するには、WandbLogger をインスタンス化し、Lightning の `Trainer` または `Fabric` に渡します。

{{< tabpane text=true >}}

{{% tab header="PyTorch Logger" value="pytorch" %}}

```python
trainer = Trainer(logger=wandb_logger)
```

{{% /tab %}}

{{% tab header="Fabric Logger" value="fabric" %}}

```python
fabric = L.Fabric(loggers=[wandb_logger])
fabric.launch()
fabric.log_dict({
    "important_metric": important_metric
})
```

{{% /tab %}}

{{< /tabpane >}}

### 一般的なロガーの引数

以下は、WandbLogger で最もよく使用されるパラメーターの一部です。すべてのロガー引数の詳細については、PyTorch Lightning のドキュメントを確認してください。

- [`PyTorch`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)
- [`Fabric`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)

| パラメータ   | 説明                                                                   |
| ----------- | ----------------------------------------------------------------------------- |
| `project`   | ログを記録する wandb Project を定義します。                                           |
| `name`      | wandb run に名前を付けます。                                                 |
| `log_model` | `log_model="all"` の場合はすべてのモデルをログに記録し、`log_model=True` の場合はトレーニングの最後にログに記録します。 |
| `save_dir`  | データが保存されるパス                                                      |

## ハイパーパラメーター をログに記録する

{{< tabpane text=true >}}

{{% tab header="PyTorch Logger" value="pytorch" %}}

```python
class LitModule(LightningModule):
    def __init__(self, *args, **kwarg):
        self.save_hyperparameters()
```

{{% /tab %}}

{{% tab header="Fabric Logger" value="fabric" %}}

```python
wandb_logger.log_hyperparams(
    {
        "hyperparameter_1": hyperparameter_1,
        "hyperparameter_2": hyperparameter_2,
    }
)
```

{{% /tab %}}

{{< /tabpane >}}

## 追加の構成パラメーター をログに記録する

```python
# パラメータ を 1 つ追加
wandb_logger.experiment.config["key"] = value

# 複数の パラメータ を追加
wandb_logger.experiment.config.update({key1: val1, key2: val2})

# wandb モジュールを直接使用
wandb.config["key"] = value
wandb.config.update()
```

## 勾配、 パラメータ のヒストグラム、およびモデルのトポロジをログに記録する

モデルオブジェクトを `wandblogger.watch()` に渡して、トレーニング中にモデルの勾配と パラメータ を監視できます。PyTorch Lightning `WandbLogger` ドキュメントを参照してください。

## メトリクス をログに記録する

{{< tabpane text=true >}}

{{% tab header="PyTorch Logger" value="pytorch" %}}

`WandbLogger` を使用すると、`LightningModule` 内 (例: `training_step` メソッドまたは `validation_step` メソッド) で `self.log('my_metric_name', metric_vale)` を呼び出すことで、メトリクス を W&B にログ記録できます。

以下のコードスニペットは、メトリクス と `LightningModule` の ハイパーパラメーター をログ記録するように `LightningModule` を定義する方法を示しています。この例では、[`torchmetrics`](https://github.com/PyTorchLightning/metrics) ライブラリを使用してメトリクス を計算します。

```python
import torch
from torch.nn import Linear, CrossEntropyLoss, functional as F
from torch.optim import Adam
from torchmetrics.functional import accuracy
from lightning.pytorch import LightningModule


class My_LitModule(LightningModule):
    def __init__(self, n_classes=10, n_layer_1=128, n_layer_2=256, lr=1e-3):
        """モデル パラメータ を定義するために使用されるメソッド"""
        super().__init__()

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = Linear(28 * 28, n_layer_1)
        self.layer_2 = Linear(n_layer_1, n_layer_2)
        self.layer_3 = Linear(n_layer_2, n_classes)

        self.loss = CrossEntropyLoss()
        self.lr = lr

        # ハイパー パラメータ を self.hparams に保存 (W&B によって自動的にログ記録)
        self.save_hyperparameters()

    def forward(self, x):
        """推論 input -> output に使用されるメソッド"""

        # (b, 1, 28, 28) -> (b, 1*28*28)
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)

        # 3 x (linear + relu) を実行しましょう
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

    def training_step(self, batch, batch_idx):
        """単一のバッチから損失を返す必要があります"""
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # 損失とメトリックをログに記録
        self.log("train_loss", loss)
        self.log("train_accuracy", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        """メトリクス のログ記録に使用"""
        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # 損失とメトリックをログに記録
        self.log("val_loss", loss)
        self.log("val_accuracy", acc)
        return preds

    def configure_optimizers(self):
        """モデル オプティマイザー を定義します"""
        return Adam(self.parameters(), lr=self.lr)

    def _get_preds_loss_accuracy(self, batch):
        """train/valid/test ステップが類似しているため、便利な関数"""
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.loss(logits, y)
        acc = accuracy(preds, y)
        return preds, loss, acc
```

{{% /tab %}}

{{% tab header="Fabric Logger" value="fabric" %}}

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

{{% /tab %}}

{{< /tabpane >}}

## メトリクス の最小値/最大値をログに記録する

wandb の [`define_metric`]({{< relref path="/ref/python/run#define_metric" lang="ja" >}}) 関数を使用すると、W&B サマリーメトリックに、そのメトリックの最小値、最大値、平均値、または最適値を表示するかどうかを定義できます。`define`_`metric` _ が使用されていない場合、最後にログに記録された値がサマリーメトリックに表示されます。`define_metric` の[リファレンスドキュメントはこちら]({{< relref path="/ref/python/run#define_metric" lang="ja" >}})、[ガイドはこちら]({{< relref path="/guides/models/track/log/customize-logging-axes" lang="ja" >}})をご覧ください。

W&B に W&B サマリーメトリックの最大検証精度を追跡するように指示するには、トレーニングの開始時に 1 回だけ `wandb.define_metric` を呼び出します。

{{< tabpane text=true >}}
{{% tab header="PyTorch Logger" value="pytorch" %}}

```python
class My_LitModule(LightningModule):
    ...

    def validation_step(self, batch, batch_idx):
        if trainer.global_step == 0:
            wandb.define_metric("val_accuracy", summary="max")

        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # 損失とメトリックをログに記録
        self.log("val_loss", loss)
        self.log("val_accuracy", acc)
        return preds
```

{{% /tab %}}
{{% tab header="Fabric Logger" value="fabric" %}}

```python
wandb.define_metric("val_accuracy", summary="max")
fabric = L.Fabric(loggers=[wandb_logger])
fabric.launch()
fabric.log_dict({"val_accuracy": val_accuracy})
```

{{% /tab %}}
{{< /tabpane >}}

## モデル をチェックポイントする

モデル のチェックポイントを W&B [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) として保存するには、Lightning の [`ModelCheckpoint`](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint) コールバックを使用し、`WandbLogger` で `log_model` 引数を設定します。

{{< tabpane text=true >}}

{{% tab header="PyTorch Logger" value="pytorch" %}}

```python
trainer = Trainer(logger=wandb_logger, callbacks=[checkpoint_callback])
```

{{% /tab %}}

{{% tab header="Fabric Logger" value="fabric" %}}

```python
fabric = L.Fabric(loggers=[wandb_logger], callbacks=[checkpoint_callback])
```

{{% /tab %}}

{{< /tabpane >}}

_latest_ および _best_ エイリアスは、W&B [Artifact]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) からモデル のチェックポイントを簡単に取得できるように自動的に設定されます。

```python
# 参照は Artifacts パネルで取得できます
# "VERSION" は、バージョン ("v2" など) またはエイリアス ("latest" または "best") にすることができます
checkpoint_reference = "USER/PROJECT/MODEL-RUN_ID:VERSION"
```

{{< tabpane text=true >}}
{{% tab header="Logger 経由" value="logger" %}}

```python
# チェックポイントをローカルにダウンロードします (まだキャッシュされていない場合)
wandb_logger.download_artifact(checkpoint_reference, artifact_type="model")
```

{{% /tab %}}

{{% tab header="wandb 経由" value="wandb" %}}

```python
# チェックポイントをローカルにダウンロードします (まだキャッシュされていない場合)
run = wandb.init(project="MNIST")
artifact = run.use_artifact(checkpoint_reference, type="model")
artifact_dir = artifact.download()
```

{{% /tab %}}
{{< /tabpane >}}

{{< tabpane text=true >}}
{{% tab header="PyTorch Logger" value="pytorch" %}}

```python
# チェックポイントを読み込む
model = LitModule.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")
```

{{% /tab %}}

{{% tab header="Fabric Logger" value="fabric" %}}

```python
# 未加工のチェックポイントを要求する
full_checkpoint = fabric.load(Path(artifact_dir) / "model.ckpt")

model.load_state_dict(full_checkpoint["model"])
optimizer.load_state_dict(full_checkpoint["optimizer"])
```

{{% /tab %}}
{{< /tabpane >}}

ログに記録するモデル のチェックポイントは、[W&B Artifacts]({{< relref path="/guides/core/artifacts" lang="ja" >}}) UI で表示でき、完全なモデル リネージが含まれています (UI のサンプル モデル のチェックポイントを[こちら](https://wandb.ai/wandb/arttest/artifacts/model/iv3_trained/5334ab69740f9dda4fed/lineage?_gl=1*yyql5q*_ga*MTQxOTYyNzExOS4xNjg0NDYyNzk1*_ga_JH1SJHJQXJ*MTY5MjMwNzI2Mi4yNjkuMS4xNjkyMzA5NjM2LjM3LjAuMA..) で参照してください)。

最適なモデル のチェックポイントをブックマークし、チーム全体で一元化するには、[W&B モデルレジストリ]({{< relref path="/guides/models" lang="ja" >}}) にリンクできます。

ここでは、タスクごとに最適なモデル を整理したり、モデル のライフサイクルを管理したり、ML ライフサイクル全体で簡単な追跡と監査を容易にしたり、Webhook またはジョブでダウンストリームアクションを[自動化]({{< relref path="/guides/core/automations/" lang="ja" >}}) したりできます。

## 画像、テキストなどをログに記録する

`WandbLogger` には、メディアをログ記録するための `log_image`、`log_text`、および `log_table` メソッドがあります。

また、`wandb.log` または `trainer.logger.experiment.log` を直接呼び出して、オーディオ、分子、点群、3D オブジェクトなどの他のメディアタイプをログに記録することもできます。

{{< tabpane text=true >}}

{{% tab header="画像のログ記録" value="images" %}}

```python
# テンソル、numpy 配列、または PIL 画像を使用
wandb_logger.log_image(key="samples", images=[img1, img2])

# キャプションを追加
wandb_logger.log_image(key="samples", images=[img1, img2], caption=["tree", "person"])

# ファイルパスを使用
wandb_logger.log_image(key="samples", images=["img_1.jpg", "img_2.jpg"])

# trainer で .log を使用
trainer.logger.experiment.log(
    {"samples": [wandb.Image(img, caption=caption) for (img, caption) in my_images]},
    step=current_trainer_global_step,
)
```

{{% /tab %}}

{{% tab header="テキストのログ記録" value="text" %}}

```python
# データはリストのリストである必要があります
columns = ["input", "label", "prediction"]
my_data = [["cheese", "english", "english"], ["fromage", "french", "spanish"]]

# 列とデータを使用
wandb_logger.log_text(key="my_samples", columns=columns, data=my_data)

# pandas DataFrame を使用
wandb_logger.log_text(key="my_samples", dataframe=my_dataframe)
```

{{% /tab %}}

{{% tab header="テーブル のログ記録" value="tables" %}}

```python
# テキスト キャプション、画像、オーディオを含む W&B テーブル をログに記録
columns = ["caption", "image", "sound"]

# データはリストのリストである必要があります
my_data = [
    ["cheese", wandb.Image(img_1), wandb.Audio(snd_1)],
    ["wine", wandb.Image(img_2), wandb.Audio(snd_2)],
]

# テーブル をログに記録
wandb_logger.log_table(key="my_samples", columns=columns, data=data)
```

{{% /tab %}}

{{< /tabpane >}}

Lightning のコールバックシステムを使用すると、この例では検証画像と予測のサンプルをログに記録します。WandbLogger 経由で Weights & Biases にログを記録するタイミングを制御できます。

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

        # `outputs` は `LightningModule.validation_step` から取得されます
        # これは、この場合のモデル予測に対応します

        # 最初のバッチから 20 個のサンプル画像予測をログに記録しましょう
        if batch_idx == 0:
            n = 20
            x, y = batch
            images = [img for img in x[:n]]
            captions = [
                f"Ground Truth: {y_i} - Prediction: {y_pred}"
                for y_i, y_pred in zip(y[:n], outputs[:n])
            ]

            # オプション 1: `WandbLogger.log_image` で画像をログに記録する
            wandb_logger.log_image(key="sample_images", images=images, caption=captions)

            # オプション 2: 画像と予測を W&B テーブル としてログに記録する
            columns = ["image", "ground truth", "prediction"]
            data = [
                [wandb.Image(x_i), y_i, y_pred] or x_i,
                y_i,
                y_pred in list(zip(x[:n], y[:n], outputs[:n])),
            ]
            wandb_logger.log_table(key="sample_table", columns=columns, data=data)


trainer = pl.Trainer(callbacks=[LogPredictionSamplesCallback()])
```

## Lightning と W&B で複数の GPU を使用する

PyTorch Lightning には、DDP インターフェイスによるマルチ GPU サポートがあります。ただし、PyTorch Lightning の設計では、GPU のインスタンス化方法に注意する必要があります。

Lightning は、トレーニングループの各 GPU (またはランク) が、まったく同じ方法 (同じ初期条件) でインスタンス化される必要があると想定しています。ただし、ランク 0 プロセスのみが `wandb.run` オブジェクトにアクセスでき、ゼロ以外のランク プロセスの場合: `wandb.run = None`。これにより、ゼロ以外のプロセスが失敗する可能性があります。このような状況は、ランク 0 プロセスがゼロ以外のランク プロセスが参加するのを待機するため、**デッドロック**状態になる可能性があります。

このため、トレーニングコードのセットアップ方法には注意してください。トレーニングコードを `wandb.run` オブジェクトから独立させることをお勧めします。

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
    # すべての乱数シードを同じ値に設定します。
    # これは、分散トレーニング環境では重要です。
    # 各ランクは、独自の初期重みセットを取得します。
    # 一致しない場合、勾配も一致しません。
    # 収束しない可能性のあるトレーニングにつながります。
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

## 例

Colab のビデオチュートリアルを[こちら](https://wandb.me/lit-colab)でフォローできます。

## よくある質問

### W&B は Lightning とどのように統合されますか?

コア統合は、[Lightning `loggers` API](https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html) に基づいており、フレームワークに依存しない方法で多くのログ記録コードを記述できます。`Logger` は、[Lightning `Trainer`](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html) に渡され、その API の豊富な[フックアンドコールバックシステム](https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html) に基づいてトリガーされます。これにより、研究コードがエンジニアリングコードおよびログ記録コードから適切に分離されます。

### 追加のコードなしで統合は何をログに記録しますか?

モデル のチェックポイントを W&B に保存します。そこで表示したり、ダウンロードして将来の run で使用したりできます。[システムメトリクス]({{< relref path="/guides/models/app/settings-page/system-metrics.md" lang="ja" >}})(GPU 使用率やネットワーク I/O など)、ハードウェアや OS 情報などの環境情報、[コード の状態]({{< relref path="/guides/models/app/features/panels/code.md" lang="ja" >}})(git コミットと差分パッチ、ノートブック の内容、セッション履歴を含む)、および標準出力に出力されるものをキャプチャします。

### トレーニングのセットアップで `wandb.run` を使用する必要がある場合はどうすればよいですか?

アクセスする必要がある変数のスコープを自分で拡張する必要があります。つまり、初期条件がすべてのプロセスで同じであることを確認してください。

```python
if os.environ.get("LOCAL_RANK", None) is None:
    os.environ["WANDB_DIR"] = wandb.run.dir
```

そうである場合は、`os.environ["WANDB_DIR"]` を使用してモデル のチェックポイントディレクトリーを設定できます。これにより、ゼロ以外のランクプロセスは `wandb.run.dir` にアクセスできます。
```