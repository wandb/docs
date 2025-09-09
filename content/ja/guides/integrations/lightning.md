---
title: PyTorch Lightning
menu:
  default:
    identifier: ja-guides-integrations-lightning
    parent: integrations
weight: 340
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Optimize_PyTorch_Lightning_models_with_Weights_%26_Biases.ipynb" >}}

PyTorch Lightning は、PyTorch のコードを整理し、分散トレーニングや 16 ビット精度などの高度な機能を簡単に追加するための軽量なラッパーを提供します。W&B は、ML の実験を手軽にログできる軽量なラッパーを提供します。さらに、自分で両者を組み合わせる必要はありません。W&B は [`WandbLogger`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb) を通じて PyTorch Lightning ライブラリに直接組み込まれています。

## Lightning と連携する

{{< tabpane text=true >}}
{{% tab header="PyTorch ロガー" value="pytorch" %}}

```python
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer

wandb_logger = WandbLogger(log_model="all")
trainer = Trainer(logger=wandb_logger)
```

{{% alert %}}
**wandb.log() の使用:** `WandbLogger` は Trainer の `global_step` を使って W&B にログを送ります。コード内で直接 `wandb.log` を追加で呼ぶ場合は、`wandb.log()` の `step` 引数は使わないでください。

代わりに、他のメトリクスと同様に Trainer の `global_step` をログしてください:

```python
wandb.log({"accuracy":0.99, "trainer/global_step": step})
```
{{% /alert %}}

{{% /tab %}}

{{% tab header="Fabric ロガー" value="fabric" %}}

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

{{< img src="/images/integrations/n6P7K4M.gif" alt="インタラクティブなダッシュボード" >}}

### サインアップして API キーを作成する

API キーは、あなたのマシンを W&B に認証するためのものです。API キーはユーザープロフィールから生成できます。

{{% alert %}}
より簡単な方法として、[W&B authorization page](https://wandb.ai/authorize) に直接アクセスして API キーを生成できます。表示された API キーをコピーし、パスワードマネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上のユーザープロフィールアイコンをクリックします。
1. **User Settings** を選び、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックして表示された API キーをコピーします。API キーを隠すにはページを再読み込みします。

### `wandb` ライブラリをインストールしてログインする

ローカルに `wandb` ライブラリをインストールしてログインするには:

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) に API キーを設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb` ライブラリをインストールしてログインします。



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

{{% tab header="Python ノートブック" value="notebook" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}


## PyTorch Lightning の `WandbLogger` を使う

PyTorch Lightning には複数の `WandbLogger` クラスがあり、メトリクスやモデルの重み、メディアなどをログできます。

- [`PyTorch`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)
- [`Fabric`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)

Lightning と統合するには、`WandbLogger` をインスタンス化して Lightning の `Trainer` または `Fabric` に渡します。

{{< tabpane text=true >}}

{{% tab header="PyTorch ロガー" value="pytorch" %}}

```python
trainer = Trainer(logger=wandb_logger)
```

{{% /tab %}}

{{% tab header="Fabric ロガー" value="fabric" %}}

```python
fabric = L.Fabric(loggers=[wandb_logger])
fabric.launch()
fabric.log_dict({
    "important_metric": important_metric
})
```

{{% /tab %}}

{{< /tabpane >}}


### よく使うロガー引数

以下は `WandbLogger` でよく使われるパラメータの一部です。すべての引数の詳細は PyTorch Lightning のドキュメントを参照してください。

- [`PyTorch`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)
- [`Fabric`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)

| パラメータ | 説明 |
| --- | --- |
| `project` | ログ先の wandb Project を指定します |
| `name` | wandb の run に名前を付けます |
| `log_model` | `log_model="all"` ならすべてのモデルを、`log_model=True` ならトレーニング終了時にモデルをログします |
| `save_dir` | データを保存するパス |

## ハイパーパラメーターをログする

{{< tabpane text=true >}}

{{% tab header="PyTorch ロガー" value="pytorch" %}}

```python
class LitModule(LightningModule):
    def __init__(self, *args, **kwarg):
        self.save_hyperparameters()
```

{{% /tab %}}

{{% tab header="Fabric ロガー" value="fabric" %}}

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

## 追加の config パラメータをログする

```python
# パラメータを 1 つ追加
wandb_logger.experiment.config["key"] = value

# 複数のパラメータを追加
wandb_logger.experiment.config.update({key1: val1, key2: val2})

# wandb モジュールを直接使う
wandb.config["key"] = value
wandb.config.update()
```

## 勾配、パラメータのヒストグラム、モデルのトポロジをログする

`wandblogger.watch()` にモデルオブジェクトを渡すと、トレーニング中のモデルの勾配やパラメータを監視できます。詳しくは PyTorch Lightning の `WandbLogger` ドキュメントを参照してください。

## メトリクスをログする

{{< tabpane text=true >}}

{{% tab header="PyTorch ロガー" value="pytorch" %}}

`WandbLogger` を使う場合、`LightningModule` 内（`training_step` や `validation_step` など）で `self.log('my_metric_name', metric_vale)` を呼び出してメトリクスを W&B にログできます。

以下のコードスニペットでは、メトリクスや `LightningModule` のハイパーパラメーターをログするための `LightningModule` の定義例を示します。この例ではメトリクス計算に [`torchmetrics`](https://github.com/PyTorchLightning/metrics) ライブラリを使用しています。

```python
import torch
from torch.nn import Linear, CrossEntropyLoss, functional as F
from torch.optim import Adam
from torchmetrics.functional import accuracy
from lightning.pytorch import LightningModule


class My_LitModule(LightningModule):
    def __init__(self, n_classes=10, n_layer_1=128, n_layer_2=256, lr=1e-3):
        """モデルのパラメータを定義するメソッド"""
        super().__init__()

        # MNIST の画像は (1, 28, 28)（チャンネル、幅、高さ）
        self.layer_1 = Linear(28 * 28, n_layer_1)
        self.layer_2 = Linear(n_layer_1, n_layer_2)
        self.layer_3 = Linear(n_layer_2, n_classes)

        self.loss = CrossEntropyLoss()
        self.lr = lr

        # ハイパーパラメーターを self.hparams に保存（W&B が自動でログ）
        self.save_hyperparameters()

    def forward(self, x):
        """推論に使われるメソッド input -> output"""

        # (b, 1, 28, 28) -> (b, 1*28*28)
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)

        # 3 回（linear + relu）を適用
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

    def training_step(self, batch, batch_idx):
        """1 バッチ分の loss を返す必要がある"""
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # loss とメトリクスをログ
        self.log("train_loss", loss)
        self.log("train_accuracy", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        """メトリクスをログする処理"""
        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # loss とメトリクスをログ
        self.log("val_loss", loss)
        self.log("val_accuracy", acc)
        return preds

    def configure_optimizers(self):
        """モデルのオプティマイザーを定義"""
        return Adam(self.parameters(), lr=self.lr)

    def _get_preds_loss_accuracy(self, batch):
        """train/valid/test ステップが似ているためのユーティリティ"""
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.loss(logits, y)
        acc = accuracy(preds, y)
        return preds, loss, acc
```

{{% /tab %}}

{{% tab header="Fabric ロガー" value="fabric" %}}

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

## メトリクスの最小値/最大値をログする

wandb の [`define_metric`]({{< relref path="/ref/python/sdk/classes/run.md#define_metric" lang="ja" >}}) 関数を使うと、W&B のサマリメトリクスに最小値、最大値、平均、ベストのどれを表示するかを定義できます。`define_metric` を使わない場合は、最後にログされた値がサマリメトリクスに表示されます。`define_metric` の[リファレンスはこちら]({{< relref path="/ref/python/sdk/classes/run.md#define_metric" lang="ja" >}})、[ガイドはこちら]({{< relref path="/guides/models/track/log/customize-logging-axes" lang="ja" >}})を参照してください。

W&B のサマリメトリクスで検証精度の最大値を追跡させるには、トレーニングの最初に 1 回だけ `wandb.define_metric` を呼びます:

{{< tabpane text=true >}}
{{% tab header="PyTorch ロガー" value="pytorch" %}}

```python
class My_LitModule(LightningModule):
    ...

    def validation_step(self, batch, batch_idx):
        if trainer.global_step == 0:
            wandb.define_metric("val_accuracy", summary="max")

        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log("val_loss", loss)
        self.log("val_accuracy", acc)
        return preds
```

{{% /tab %}}
{{% tab header="Fabric ロガー" value="fabric" %}}

```python
wandb.define_metric("val_accuracy", summary="max")
fabric = L.Fabric(loggers=[wandb_logger])
fabric.launch()
fabric.log_dict({"val_accuracy": val_accuracy})
```

{{% /tab %}}
{{< /tabpane >}}

## モデルをチェックポイントする

モデルのチェックポイントを W&B の [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) として保存するには、Lightning の [`ModelCheckpoint`](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint) コールバックを使用し、`WandbLogger` の `log_model` 引数を設定します。

{{< tabpane text=true >}}

{{% tab header="PyTorch ロガー" value="pytorch" %}}

```python
trainer = Trainer(logger=wandb_logger, callbacks=[checkpoint_callback])
```

{{% /tab %}}

{{% tab header="Fabric ロガー" value="fabric" %}}

```python
fabric = L.Fabric(loggers=[wandb_logger], callbacks=[checkpoint_callback])
```

{{% /tab %}}

{{< /tabpane >}}

W&B の [Artifact]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) からモデルのチェックポイントを簡単に取得できるよう、 _latest_ と _best_ のエイリアスが自動で設定されます:

```python
# 参照は Artifacts パネルで取得できます
# "VERSION" は（例: "v2"）のような version か、"latest" や "best" のようなエイリアスを指定できます
checkpoint_reference = "USER/PROJECT/MODEL-RUN_ID:VERSION"
```

{{< tabpane text=true >}}
{{% tab header="Logger 経由" value="logger" %}}

```python
# チェックポイントをローカルにダウンロード（未キャッシュの場合）
wandb_logger.download_artifact(checkpoint_reference, artifact_type="model")
```

{{% /tab %}}

{{% tab header="wandb 経由" value="wandb" %}}

```python
# チェックポイントをローカルにダウンロード（未キャッシュの場合）
run = wandb.init(project="MNIST")
artifact = run.use_artifact(checkpoint_reference, type="model")
artifact_dir = artifact.download()
```

{{% /tab %}}
{{< /tabpane >}}

{{< tabpane text=true >}}
{{% tab header="PyTorch ロガー" value="pytorch" %}}

```python
# チェックポイントを読み込む
model = LitModule.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")
```

{{% /tab %}}

{{% tab header="Fabric ロガー" value="fabric" %}}

```python
# 生のチェックポイントを取得
full_checkpoint = fabric.load(Path(artifact_dir) / "model.ckpt")

model.load_state_dict(full_checkpoint["model"])
optimizer.load_state_dict(full_checkpoint["optimizer"])
```

{{% /tab %}}
{{< /tabpane >}}

ログしたモデルのチェックポイントは [W&B Artifacts]({{< relref path="/guides/core/artifacts" lang="ja" >}}) の UI から確認でき、完全なモデル リネージも含まれます（UI 上のモデルチェックポイントの例は[こちら](https://wandb.ai/wandb/arttest/artifacts/model/iv3_trained/5334ab69740f9dda4fed/lineage?_gl=1*yyql5q*_ga*MTQxOTYyNzExOS4xNjg0NDYyNzk1*_ga_JH1SJHJQXJ*MTY5MjMwNzI2Mi4yNjkuMS4xNjkyMzA5NjM2LjM3LjAuMA..)）。

チーム全体でベストなモデルのチェックポイントをブックマークして一元管理するには、[W&B Model Registry]({{< relref path="/guides/models" lang="ja" >}}) にリンクできます。

ここでは、タスクごとにベストモデルを整理し、モデルのライフサイクルを管理し、ML ライフサイクル全体での追跡や監査を容易にし、webhook や job を使って下流のアクションを[自動化]({{< relref path="/guides/core/automations/" lang="ja" >}})できます。 

## 画像、テキストなどをログする

`WandbLogger` には、メディアをログするための `log_image`、`log_text`、`log_table` メソッドがあります。

また、Audio、Molecule、Point Cloud、3D オブジェクトなど他のメディアタイプは、`wandb.log` または `trainer.logger.experiment.log` を直接呼び出してログすることもできます。

{{< tabpane text=true >}}

{{% tab header="画像をログ" value="images" %}}

```python
# テンソル、NumPy 配列、PIL 画像を使用
wandb_logger.log_image(key="samples", images=[img1, img2])

# キャプションを追加
wandb_logger.log_image(key="samples", images=[img1, img2], caption=["tree", "person"])

# ファイルパスを使用
wandb_logger.log_image(key="samples", images=["img_1.jpg", "img_2.jpg"])

# Trainer で .log を使う
trainer.logger.experiment.log(
    {"samples": [wandb.Image(img, caption=caption) for (img, caption) in my_images]},
    step=current_trainer_global_step,
)
```

{{% /tab %}}

{{% tab header="テキストをログ" value="text" %}}

```python
# data はリストのリストである必要があります
columns = ["input", "label", "prediction"]
my_data = [["cheese", "english", "english"], ["fromage", "french", "spanish"]]

# columns と data を使用
wandb_logger.log_text(key="my_samples", columns=columns, data=my_data)

# pandas DataFrame を使用
wandb_logger.log_text(key="my_samples", dataframe=my_dataframe)
```

{{% /tab %}}

{{% tab header="Tables をログ" value="tables" %}}

```python
# テキストのキャプション、画像、音声を含む W&B Table をログ
columns = ["caption", "image", "sound"]

# data はリストのリストである必要があります
my_data = [
    ["cheese", wandb.Image(img_1), wandb.Audio(snd_1)],
    ["wine", wandb.Image(img_2), wandb.Audio(snd_2)],
]

# Table をログ
wandb_logger.log_table(key="my_samples", columns=columns, data=data)
```

{{% /tab %}}

{{< /tabpane >}}

Lightning の Callback システムを使えば、`WandbLogger` 経由で W&B にいつログするかを制御できます。次の例では、検証画像と予測のサンプルをログします:


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
        """検証バッチが終了したときに呼ばれます。"""

        # `outputs` は `LightningModule.validation_step` の戻り値です
        # この例ではモデルの予測に相当します

        # 最初のバッチから 20 枚の画像予測サンプルをログしましょう
        if batch_idx == 0:
            n = 20
            x, y = batch
            images = [img for img in x[:n]]
            captions = [
                f"Ground Truth: {y_i} - Prediction: {y_pred}"
                for y_i, y_pred in zip(y[:n], outputs[:n])
            ]

            # オプション 1: `WandbLogger.log_image` で画像をログ
            wandb_logger.log_image(key="sample_images", images=images, caption=captions)

            # オプション 2: 画像と予測を W&B Table としてログ
            columns = ["image", "ground truth", "prediction"]
            data = [
                [wandb.Image(x_i), y_i, y_pred] or x_i,
                y_i,
                y_pred in list(zip(x[:n], y[:n], outputs[:n])),
            ]
            wandb_logger.log_table(key="sample_table", columns=columns, data=data)


trainer = pl.Trainer(callbacks=[LogPredictionSamplesCallback()])
```

## Lightning と W&B で複数 GPU を使う

PyTorch Lightning は DDP インターフェースを通じてマルチ GPU をサポートしています。ただし、PyTorch Lightning の設計上、GPU の初期化方法には注意が必要です。

Lightning は、トレーニングループ内の各 GPU（または Rank）がまったく同じ方法、つまり同じ初期条件でインスタンス化されることを前提としています。しかし、`wandb.run` オブジェクトにアクセスできるのは rank 0 のプロセスだけで、0 以外の rank のプロセスでは `wandb.run = None` になります。これにより、0 以外のプロセスが失敗する可能性があります。この状況では、rank 0 のプロセスはすでにクラッシュした 0 以外の rank のプロセスが合流するのを待ち続けるため、**デッドロック** に陥ることがあります。

このため、トレーニングコードのセットアップには注意してください。推奨の方法は、コードを `wandb.run` オブジェクトに依存しないように構成することです。

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
    # 分散トレーニングでは重要です。
    # 各 rank は独自の初期重みを受け取ります。
    # もし一致しなければ、勾配も一致せず、
    # 収束しない学習につながる可能性があります。
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

[ビデオチュートリアル（Colab ノートブック付き）](https://wandb.me/lit-colab)に沿って学習できます。

## よくある質問

### W&B は Lightning とどう統合されていますか？

中核の統合は [Lightning の `loggers` API](https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html) に基づいており、フレームワーク非依存の書き方で多くのログコードを書けます。`Logger` は [Lightning の `Trainer`](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html) に渡され、この API の豊富な[フックとコールバックの仕組み](https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html)に基づいてトリガーされます。これにより、研究用のコードとエンジニアリングやログのコードをきれいに分離できます。

### 追加のコードなしで、統合は何をログしますか？

モデルのチェックポイントを W&B に保存し、将来の run で閲覧やダウンロードができます。また、[システムメトリクス]({{< relref path="/ref/system-metrics.md" lang="ja" >}})（GPU 使用率やネットワーク I/O など）、環境情報（ハードウェアや OS 情報など）、[コードの状態]({{< relref path="/guides/models/app/features/panels/code.md" lang="ja" >}})（git のコミットや diff パッチ、ノートブックの内容やセッション履歴を含む）、標準出力に出力されたすべてを自動で取得します。

### トレーニングのセットアップで `wandb.run` を使う必要がある場合はどうすればよいですか？

必要な変数に自分でアクセスできるよう、そのスコープを広げてください。言い換えると、すべてのプロセスで初期条件が同じになるようにしてください。

```python
if os.environ.get("LOCAL_RANK", None) is None:
    os.environ["WANDB_DIR"] = wandb.run.dir
```

条件が揃っていれば、`os.environ["WANDB_DIR"]` を使ってモデルのチェックポイント保存先ディレクトリを設定できます。こうしておけば、0 以外の rank のプロセスでも `wandb.run.dir` にアクセスできます。