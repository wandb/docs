---
title: PyTorch Lightning
menu:
  default:
    identifier: ja-guides-integrations-lightning
    parent: integrations
weight: 340
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Optimize_PyTorch_Lightning_models_with_Weights_%26_Biases.ipynb" >}}

PyTorch Lightning は、PyTorch コードを整理し、分散トレーニングや 16 ビット精度のような高度な機能を簡単に追加するための軽量ラッパーを提供します。 W&B は、あなたの ML 実験を記録するための軽量ラッパーを提供します。しかし、自分でそれらを組み合わせる必要はありません。Weights & Biases は、[**`WandbLogger`**](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb) を介して PyTorch Lightning ライブラリに直接組み込まれています。

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
**wandb.log() を使用する際の注意点:** `WandbLogger` は Trainer の `global_step` を使用して W&B にログを記録します。コード内で直接 `wandb.log` を追加で呼び出す場合、`wandb.log()` の `step` 引数を使用しないでください。

代わりに、Trainer の `global_step` を他のメトリクスと同様に記録してください：

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

{{< img src="/images/integrations/n6P7K4M.gif" alt="どこからでもアクセスできるインタラクティブなダッシュボード、他にも！" >}}

### サインアップして APIキーを作成する

APIキー は、あなたのマシンを W&B に認証するためのものです。あなたのユーザープロフィールから APIキー を生成できます。

{{% alert %}}
よりスムーズなアプローチとして、[https://wandb.ai/authorize](https://wandb.ai/authorize) に直接アクセスして APIキー を生成することができます。表示された APIキー を安全な場所（パスワードマネージャーなど）に保存してください。
{{% /alert %}}

1. 右上のユーザープロフィールアイコンをクリックします。
1. **User Settings** を選択し、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックします。表示された APIキー をコピーします。APIキー を非表示にするには、ページをリロードします。

### `wandb` ライブラリをインストールしてログインする

ローカルに `wandb` ライブラリをインストールしてログインする方法：

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. あなたの APIキー に `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) を設定します。

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

{{% tab header="Python notebook" value="notebook" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}


## PyTorch Lightning の `WandbLogger` を使用する

PyTorch Lightning には、メトリクスやモデルの重み、メディアなどを記録するための複数の `WandbLogger` クラスがあります。

- [`PyTorch`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)
- [`Fabric`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)

Lightning と統合するには、WandbLogger をインスタンス化し、Lightning の `Trainer` または `Fabric` に渡してください。

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


### よく使用されるロガーの引数

以下に、WandbLogger でよく使用されるパラメータを示します。すべてのロガー引数の詳細については PyTorch Lightning のドキュメントを確認してください。

- [`PyTorch`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)
- [`Fabric`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)

| Parameter   | Description                                                                   |
| ----------- | ----------------------------------------------------------------------------- |
| `project`   | 記録する wandb Project を定義します                                           |
| `name`      | あなたの wandb run に名前を付けます                                            |
| `log_model` | `log_model="all"` の場合はすべてのモデルを記録し、`log_model=True` の場合はトレーニングの最後に記録します |
| `save_dir`  | データが保存されるパス                                                        |

## ハイパーパラメーターを記録する

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

## 追加の設定パラメータを記録する

```python
# パラメータを1つ追加する
wandb_logger.experiment.config["key"] = value

# 複数のパラメータを追加する
wandb_logger.experiment.config.update({key1: val1, key2: val2})

# 直接 wandb モジュールを使用する
wandb.config["key"] = value
wandb.config.update()
```

## 勾配、パラメータヒストグラム、モデルトポロジーを記録する

モデルのオブジェクトを `wandblogger.watch()` に渡すことで、トレーニング中のモデルの勾配とパラメータを監視できます。PyTorch Lightning の `WandbLogger` ドキュメントを参照してください。

## メトリクスを記録する

{{< tabpane text=true >}}

{{% tab header="PyTorch Logger" value="pytorch" %}}

`WandbLogger` を使用しているときは、`LightningModule` 内で `self.log('my_metric_name', metric_value)` を呼び出すことで W&B にメトリクスを記録できます。たとえば、`training_step` や `validation_step` メソッド内でこれを行います。

以下のコードスニペットは、メトリクスとハイパーパラメーターを記録するための `LightningModule` を定義する方法を示しています。この例では、[`torchmetrics`](https://github.com/PyTorchLightning/metrics) ライブラリを使用してメトリクスを計算します。

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

        # mnist 画像は (1, 28, 28) (チャンネル、幅、高さ) です
        self.layer_1 = Linear(28 * 28, n_layer_1)
        self.layer_2 = Linear(n_layer_1, n_layer_2)
        self.layer_3 = Linear(n_layer_2, n_classes)

        self.loss = CrossEntropyLoss()
        self.lr = lr

        # ハイパーパラメーターを self.hparams に保存します (W&B によって自動でログされます)
        self.save_hyperparameters()

    def forward(self, x):
        """推論のための入力 -> 出力 メソッド"""

        # (b, 1, 28, 28) -> (b, 1*28*28)
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)

        # 3 * (線形 + ReLU)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

    def training_step(self, batch, batch_idx):
        """1つのバッチからの損失を返す必要があります"""
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # 損失とメトリクスを記録する
        self.log("train_loss", loss)
        self.log("train_accuracy", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        """メトリクスを記録するために使用されます"""
        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # 損失とメトリクスを記録する
        self.log("val_loss", loss)
        self.log("val_accuracy", acc)
        return preds

    def configure_optimizers(self):
        """モデルオプティマイザーを定義します"""
        return Adam(self.parameters(), lr=self.lr)

    def _get_preds_loss_accuracy(self, batch):
        """train/valid/test ステップが類似しているための便利な機能"""
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

## メトリクスの最小/最大値を記録する

wandb の [`define_metric`]({{< relref path="/ref/python/run#define_metric" lang="ja" >}}) 関数を使用して、W&B の要約メトリクスがそのメトリクスの最小、最大、平均、または最良の値を表示するかどうかを定義できます。`define_metric` が使用されていない場合、最後に記録された値が要約メトリクスに表示されます。詳細な `define_metric` の [ドキュメントはこちら]({{< relref path="/ref/python/run#define_metric" lang="ja" >}}) と [ガイドはこちら]({{< relref path="/guides/models/track/log/customize-logging-axes" lang="ja" >}}) を参照してください。

W&B の要約メトリクスで最大の検証精度を追跡するよう W&B に指示するには、トレーニングの開始時に一度だけ `wandb.define_metric` を呼び出します：

{{< tabpane text=true >}}
{{% tab header="PyTorch Logger" value="pytorch" %}}

```python
class My_LitModule(LightningModule):
    ...

    def validation_step(self, batch, batch_idx):
        if trainer.global_step == 0:
            wandb.define_metric("val_accuracy", summary="max")

        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # 損失とメトリクスを記録する
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

## モデルチェックポイントを作成する

モデルのチェックポイントを W&B の [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) として保存するには、Lightning の [`ModelCheckpoint`](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint) コールバックを使用し、`WandbLogger` の `log_model` 引数を設定します。

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

_最新_ 及び _最良_ のエイリアスは、W&B の [Artifact]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) からモデルのチェックポイントを簡単に取得できるように自動的に設定されます：

```python
# アーティファクトパネルでリファレンスを取得できます
# "VERSION" はバージョン (例: "v2") またはエイリアス ("latest" または "best") です
checkpoint_reference = "USER/PROJECT/MODEL-RUN_ID:VERSION"
```

{{< tabpane text=true >}}
{{% tab header="Via Logger" value="logger" %}}

```python
# チェックポイントをローカルにダウンロードする（既にキャッシュされていない場合）
wandb_logger.download_artifact(checkpoint_reference, artifact_type="model")
```

{{% /tab %}}

{{% tab header="Via wandb" value="wandb" %}}

```python
# チェックポイントをローカルにダウンロードする（既にキャッシュされていない場合）
run = wandb.init(project="MNIST")
artifact = run.use_artifact(checkpoint_reference, type="model")
artifact_dir = artifact.download()
```

{{% /tab %}}
{{< /tabpane >}}

{{< tabpane text=true >}}
{{% tab header="PyTorch Logger" value="pytorch" %}}

```python
# チェックポイントをロードする
model = LitModule.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")
```

{{% /tab %}}

{{% tab header="Fabric Logger" value="fabric" %}}

```python
# 生のチェックポイントをリクエストする
full_checkpoint = fabric.load(Path(artifact_dir) / "model.ckpt")

model.load_state_dict(full_checkpoint["model"])
optimizer.load_state_dict(full_checkpoint["optimizer"])
```

{{% /tab %}}
{{< /tabpane >}}

記録されたモデルのチェックポイントは [W&B Artifacts]({{< relref path="/guides/core/artifacts" lang="ja" >}}) UI を通じて表示可能で、完全なモデルリネージも含まれます（UIでのモデルチェックポイントの例はこちら (https://wandb.ai/wandb/arttest/artifacts/model/iv3_trained/5334ab69740f9dda4fed/lineage?_gl=1*yyql5q*_ga*MTQxOTYyNzExOS4xNjg0NDYyNzk1*_ga_JH1SJHJQXJ*MTY5MjMwNzI2Mi4yNjkuMS4xNjkyMzA5NjM2LjM3LjAuMA..))。

最良のモデルチェックポイントをブックマークし、チーム全体でそれらを一元化するために、[W&B Model Registry]({{< relref path="/guides/models" lang="ja" >}}) にリンクすることができます。

これにより、タスクごとに最良のモデルを整理し、モデルのライフサイクルを管理し、MLライフサイクル全体で簡単な追跡と監査を可能にし、Webhooksやジョブでのダウンストリームアクションを[自動化]({{< relref path="/guides/core/automations/" lang="ja" >}})することができます。

## 画像やテキストなどを記録する

`WandbLogger` は、メディアを記録するための `log_image`、`log_text`、`log_table` メソッドを持っています。

他にも、音声、分子、ポイントクラウド、3Dオブジェクトなどのメディアタイプを記録するために、直接的に `wandb.log` や `trainer.logger.experiment.log` を呼び出すことができます。

{{< tabpane text=true >}}

{{% tab header="Log Images" value="images" %}}

```python
# テンソル、NumPy 配列、または PIL 画像を使用
wandb_logger.log_image(key="samples", images=[img1, img2])

# キャプションを追加
wandb_logger.log_image(key="samples", images=[img1, img2], caption=["tree", "person"])

# ファイルパスを使用
wandb_logger.log_image(key="samples", images=["img_1.jpg", "img_2.jpg"])

# トレーナで .log を使用
trainer.logger.experiment.log(
    {"samples": [wandb.Image(img, caption=caption) for (img, caption) in my_images]},
    step=current_trainer_global_step,
)
```

{{% /tab %}}

{{% tab header="Log Text" value="text" %}}

```python
# データはリストのリストであるべきです
columns = ["input", "label", "prediction"]
my_data = [["cheese", "english", "english"], ["fromage", "french", "spanish"]]

# カラムとデータを使用
wandb_logger.log_text(key="my_samples", columns=columns, data=my_data)

# pandas データフレームを使用
wandb_logger.log_text(key="my_samples", dataframe=my_dataframe)
```

{{% /tab %}}

{{% tab header="Log Tables" value="tables" %}}

```python
# テキストキャプション、画像、およびオーディオを含む W&B テーブルを記録
columns = ["caption", "image", "sound"]

# データはリストのリストであるべきです
my_data = [
    ["cheese", wandb.Image(img_1), wandb.Audio(snd_1)],
    ["wine", wandb.Image(img_2), wandb.Audio(snd_2)],
]

# テーブルを記録
wandb_logger.log_table(key="my_samples", columns=columns, data=data)
```

{{% /tab %}}

{{< /tabpane >}}

Lightning のコールバックシステムを使用して、WandbLogger を介して Weights & Biases にログを記録するタイミングを制御することができます。この例では、検証画像と予測のサンプルをログします：

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
        """検証バッチの終了時に呼び出されます。"""

        # `outputs` は `LightningModule.validation_step` からのもので、今回はモデルの予測に相当します

        # 最初のバッチから20のサンプル画像予測をログします
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

            # オプション2: 画像と予測をW&B テーブルとしてログ
            columns = ["image", "ground truth", "prediction"]
            data = [
                [wandb.Image(x_i), y_i, y_pred] or x_i,
                y_i,
                y_pred in list(zip(x[:n], y[:n], outputs[:n])),
            ]
            wandb_logger.log_table(key="sample_table", columns=columns, data=data)


trainer = pl.Trainer(callbacks=[LogPredictionSamplesCallback()])
```

## 複数の GPU を使用して Lightning と W&B を使用する

PyTorch Lightning は DDP インターフェースを通じてマルチGPUをサポートしています。ただし、PyTorch Lightning のデザインは GPU をインスタンス化する際に注意が必要です。

Lightning は、トレーニングループ内の各 GPU (またはランク) がまったく同じ方法で、同じ初期条件でインスタンス化されなければならないと仮定しています。ただし、ランク0のプロセスだけが `wandb.run` オブジェクトに アクセスでき、非ゼロランクのプロセスには `wandb.run = None` となります。これが原因で、非ゼロプロセスが失敗する可能性があります。このような状況になると、ランク0のプロセスが非ゼロランクのプロセスに参加を待つことになり、既にクラッシュしてしまうため、**デッドロック**に陥る可能性があります。

このため、トレーニングコードのセットアップに注意する必要があります。推奨される方法は、コードを `wandb.run` オブジェクトに依存しないようにすることです。

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
    # 同じ値にランダムシードをすべて設定します。
    # これは分散トレーニングの設定で重要です。
    # 各ランクは自身の初期重みセットを取得します。
    # 一致しない場合、勾配も一致せず、
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



## 例

Colab のビデオチュートリアルに従うことができます。[こちら](https://wandb.me/lit-colab) をクリックしてください。

## よくある質問 (FAQ)

### W&B は Lightning とどのように統合されていますか？

コアなインテグレーションは、[Lightning `loggers` API](https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html) に基づいており、ログのコードをフレームワークに依存しない方法で多く書かせることができます。`Logger` は [Lightning `Trainer`](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html) に渡され、この API の豊富な [フックとコールバックシステム](https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html) に基づいてトリガーされます。これにより、研究コードがエンジニアリングやログのコードと完全に分離されます。

### 追加のコードなしでインテグレーションがログする内容は？

モデルのチェックポイントを W&B に保存し、今後のRunsで使用するために閲覧またはダウンロードできるようにします。また、GPU使用量やネットワークI/Oなどの[システムメトリクス]({{< relref path="/ref/system-metrics.md" lang="ja" >}})、ハードウェア情報やOS情報などの環境情報、gitコミットやdiffパッチ、ノートブックコンテンツやセッション履歴を含む[コードの状態]({{< relref path="/guides/models/app/features/panels/code.md" lang="ja" >}})、標準出力に印刷されるものをキャプチャします。

### トレーニングセットアップで `wandb.run` を使用する必要がある場合はどうすればいいですか？

アクセスが必要な変数のスコープを自分で拡張する必要があります。言い換えれば、初期条件がすべてのプロセスで同じであることを確認してください。

```python
if os.environ.get("LOCAL_RANK", None) is None:
    os.environ["WANDB_DIR"] = wandb.run.dir
```

条件が同じならば、`os.environ["WANDB_DIR"]` を使用してモデルのチェックポイントディレクトリをセットアップできます。これにより、非ゼロランクプロセスでも `wandb.run.dir` にアクセスできます。