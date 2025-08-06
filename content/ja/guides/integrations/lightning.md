---
title: PyTorch Lightning
menu:
  default:
    identifier: lightning
    parent: integrations
weight: 340
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Optimize_PyTorch_Lightning_models_with_Weights_%26_Biases.ipynb" >}}

PyTorch Lightning は、PyTorch コードを整理し、分散トレーニングや16ビット精度などの先進機能を簡単に追加できる軽量なラッパーを提供します。W&B は、機械学習実験のログ管理に便利な軽量ラッパーを提供します。両者を自分で組み合わせる必要はありません：W&B は PyTorch Lightning ライブラリに [`WandbLogger`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb) を通じて直接統合されています。

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
**wandb.log() の利用時の注意:** `WandbLogger` は Trainer の `global_step` を使って W&B にログを送信します。もし直接 `wandb.log` を追加で使う場合、`wandb.log()` の `step` 引数は **使わないでください**。

代わりに、Trainer の `global_step` と同じようにログしてください。

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

{{< img src="/images/integrations/n6P7K4M.gif" alt="インタラクティブなダッシュボード" >}}

### サインアップして APIキー を作成

APIキー は、マシンを W&B に認証するためのものです。APIキー はユーザープロフィールから発行できます。

{{% alert %}}
より簡単な方法として、[W&B 認証ページ](https://wandb.ai/authorize) から APIキー を生成できます。表示される APIキー をコピーし、パスワードマネージャーなど安全な場所に保存してください。
{{% /alert %}}

1. 右上のユーザープロフィールアイコンをクリックします。
1. **User Settings** を選択し、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックし、表示された APIキー をコピーします。APIキー を非表示にする場合はページをリロードしてください。

### `wandb` ライブラリのインストール＆ログイン

`wandb` ライブラリをローカルにインストールし、ログインするには:

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref "/guides/models/track/environment-variables.md" >}}) に APIキー を設定してください。

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


## PyTorch Lightning の `WandbLogger` を利用する

PyTorch Lightning では、メトリクスやモデル重み、メディアなどを記録できる複数の `WandbLogger` クラスが用意されています。

- [`PyTorch`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)
- [`Fabric`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)

Lightning との統合は、`WandbLogger` をインスタンス化し、Lightning の `Trainer` または `Fabric` に渡します。

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


### よく使われる logger の引数

`WandbLogger` でよく使われる主なパラメータです。詳細は PyTorch Lightning のドキュメントを参照してください。

- [`PyTorch`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)
- [`Fabric`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)

| パラメータ   | 説明                                                            |
| ----------- | --------------------------------------------------------------- |
| `project`   | どの wandb Project にログを送るかを設定                         |
| `name`      | wandb run に名前をつける                                        |
| `log_model` | `log_model="all"` で全てのモデルを記録、`log_model=True` で学習終了時のみ |
| `save_dir`  | データの保存先ディレクトリのパス                                |

## ハイパーパラメータのログ

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

## 追加のコンフィグパラメータをログする

```python
# 1つのパラメータを追加
wandb_logger.experiment.config["key"] = value

# 複数のパラメータを追加
wandb_logger.experiment.config.update({key1: val1, key2: val2})

# 直接 wandb モジュールを使う
wandb.config["key"] = value
wandb.config.update()
```

## 勾配・パラメータヒストグラム・モデル構造のログ

モデルの勾配やパラメータをトレーニング中に監視したい場合は、`wandblogger.watch()` にモデルオブジェクトを渡してください。詳細は PyTorch Lightning の `WandbLogger` ドキュメントをご参照ください。

## メトリクスのログ

{{< tabpane text=true >}}

{{% tab header="PyTorch Logger" value="pytorch" %}}

`WandbLogger` 利用時は、`LightningModule` 内で `self.log('my_metric_name', metric_vale)` を呼ぶことで W&B にメトリクスを記録できます（例えば `training_step` や `validation_step` メソッド内）。

下記のコードスニペットは、`LightningModule` でメトリクスやハイパーパラメータをどうログするかを示しています。この例ではメトリクス計算に [`torchmetrics`](https://github.com/PyTorchLightning/metrics) ライブラリを使っています。

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

        # mnist 画像サイズは (1, 28, 28) (チャンネル、幅、高さ)
        self.layer_1 = Linear(28 * 28, n_layer_1)
        self.layer_2 = Linear(n_layer_1, n_layer_2)
        self.layer_3 = Linear(n_layer_2, n_classes)

        self.loss = CrossEntropyLoss()
        self.lr = lr

        # ハイパーパラメータを self.hparams に保存（W&Bが自動ログします）
        self.save_hyperparameters()

    def forward(self, x):
        """推論用 入力 -> 出力のメソッド"""

        # (b, 1, 28, 28) -> (b, 1*28*28)
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)

        # 3 回 (線形 + relu)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

    def training_step(self, batch, batch_idx):
        """1バッチ分の loss を返す必要がある"""
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # loss とメトリクスをログ
        self.log("train_loss", loss)
        self.log("train_accuracy", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        """メトリクスのログ用"""
        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # loss とメトリクスをログ
        self.log("val_loss", loss)
        self.log("val_accuracy", acc)
        return preds

    def configure_optimizers(self):
        """モデルのオプティマイザーを定義"""
        return Adam(self.parameters(), lr=self.lr)

    def _get_preds_loss_accuracy(self, batch):
        """学習・検証・テストステップで共通の関数"""
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

## メトリクスの min/max を記録

wandb の [`define_metric`]({{< relref "/ref/python/sdk/classes/run.md#define_metric" >}}) を使うと、W&B のサマリーメトリクスとして min、max、mean、best のいずれかを表示できます。`define_metric` を使わなかった場合は、最後にログされた値が summary メトリクスとして表示されます。詳しくは [`define_metric`リファレンス]({{< relref "/ref/python/sdk/classes/run.md#define_metric" >}}) および [ガイド]({{< relref "/guides/models/track/log/customize-logging-axes" >}}) もご参照ください。

例えば、W&B summary に「最大検証精度」を記録したい場合は、学習の最初に `wandb.define_metric` を 1回だけ呼びます。

{{< tabpane text=true >}}
{{% tab header="PyTorch Logger" value="pytorch" %}}

```python
class My_LitModule(LightningModule):
    ...

    def validation_step(self, batch, batch_idx):
        if trainer.global_step == 0:
            wandb.define_metric("val_accuracy", summary="max")

        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # loss とメトリクスをログ
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

## モデルのチェックポイントを保存する

モデルのチェックポイントを W&B の [Artifacts]({{< relref "/guides/core/artifacts/" >}}) として保存するには、Lightning の [`ModelCheckpoint`](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint) コールバックを使い、`WandbLogger` の `log_model` 引数を設定します。

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

_最新_ と _ベスト_ エイリアスは自動で設定されるため、W&B [Artifact]({{< relref "/guides/core/artifacts/" >}}) から簡単にモデルチェックポイントを取得できます。

```python
# artifacts パネルで参照が取得可能
# "VERSION" にはバージョン（例: "v2"）またはエイリアス（"latest" や "best"）を指定可能
checkpoint_reference = "USER/PROJECT/MODEL-RUN_ID:VERSION"
```

{{< tabpane text=true >}}
{{% tab header="Logger 経由で取得" value="logger" %}}

```python
# チェックポイントをローカルにダウンロード（まだキャッシュされていない場合）
wandb_logger.download_artifact(checkpoint_reference, artifact_type="model")
```

{{% /tab %}}

{{% tab header="wandb 経由で取得" value="wandb" %}}

```python
# チェックポイントをローカルにダウンロード（まだキャッシュされていない場合）
run = wandb.init(project="MNIST")
artifact = run.use_artifact(checkpoint_reference, type="model")
artifact_dir = artifact.download()
```

{{% /tab %}}
{{< /tabpane >}}

{{< tabpane text=true >}}
{{% tab header="PyTorch Logger" value="pytorch" %}}

```python
# チェックポイントのロード
model = LitModule.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")
```

{{% /tab %}}

{{% tab header="Fabric Logger" value="fabric" %}}

```python
# 生のチェックポイントをリクエスト
full_checkpoint = fabric.load(Path(artifact_dir) / "model.ckpt")

model.load_state_dict(full_checkpoint["model"])
optimizer.load_state_dict(full_checkpoint["optimizer"])
```

{{% /tab %}}
{{< /tabpane >}}

記録したモデルのチェックポイントは [W&B Artifacts]({{< relref "/guides/core/artifacts" >}}) UI から確認でき、完全なモデルリネージも含まれます（[UIの例はこちら](https://wandb.ai/wandb/arttest/artifacts/model/iv3_trained/5334ab69740f9dda4fed/lineage?_gl=1*yyql5q*_ga*MTQxOTYyNzExOS4xNjg0NDYyNzk1*_ga_JH1SJHJQXJ*MTY5MjMwNzI2Mi4yNjkuMS4xNjkyMzA5NjM2LjM3LjAuMA..)）。

チーム内でベストなモデルチェックポイントをブックマークし一元管理するには、[W&B Model Registry]({{< relref "/guides/models" >}}) にリンクできます。

ここではタスクごとに優秀なモデルを整理し、ライフサイクル管理や ML 開発全体での追跡・監査を容易にし、webhook やジョブによる [自動化]({{< relref "/guides/core/automations/" >}}) も可能です。

## 画像・テキスト等のメディアの記録

`WandbLogger` には画像の `log_image`、テキストの `log_text`、テーブルの `log_table` といったメディア用ロギングメソッドが用意されています。

また `wandb.log` や `trainer.logger.experiment.log` からも、Audio、分子構造、点群、3Dオブジェクトなど、その他の様々なメディアタイプを記録できます。

{{< tabpane text=true >}}

{{% tab header="画像のログ" value="images" %}}

```python
# Tensor, numpy配列, PIL画像のいずれもOK
wandb_logger.log_image(key="samples", images=[img1, img2])

# キャプションを追加
wandb_logger.log_image(key="samples", images=[img1, img2], caption=["tree", "person"])

# ファイルパスで画像指定
wandb_logger.log_image(key="samples", images=["img_1.jpg", "img_2.jpg"])

# trainer の .log を利用した場合
trainer.logger.experiment.log(
    {"samples": [wandb.Image(img, caption=caption) for (img, caption) in my_images]},
    step=current_trainer_global_step,
)
```

{{% /tab %}}

{{% tab header="テキストのログ" value="text" %}}

```python
# データはリストのリスト形式
columns = ["input", "label", "prediction"]
my_data = [["cheese", "english", "english"], ["fromage", "french", "spanish"]]

# columns と data で指定
wandb_logger.log_text(key="my_samples", columns=columns, data=my_data)

# pandas DataFrame をそのまま渡す場合
wandb_logger.log_text(key="my_samples", dataframe=my_dataframe)
```

{{% /tab %}}

{{% tab header="テーブルのログ" value="tables" %}}

```python
# テキストキャプション・画像・音声を含む W&B Table を記録
columns = ["caption", "image", "sound"]

# データはリストのリスト
my_data = [
    ["cheese", wandb.Image(img_1), wandb.Audio(snd_1)],
    ["wine", wandb.Image(img_2), wandb.Audio(snd_2)],
]

# Table のログ
wandb_logger.log_table(key="my_samples", columns=columns, data=data)
```

{{% /tab %}}

{{< /tabpane >}}

W&B へのログタイミングを細かく制御したいときは、Lightning の Callback 機構を利用できます。例えば検証時に画像と予測をログする場合は次のようになります。

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
        """バリデーションバッチ終了時に呼ばれる。"""

        # `outputs` は `LightningModule.validation_step` の返り値
        # 今回はモデル予測

        # 最初のバッチだけ画像予測サンプル20個をログ
        if batch_idx == 0:
            n = 20
            x, y = batch
            images = [img for img in x[:n]]
            captions = [
                f"Ground Truth: {y_i} - Prediction: {y_pred}"
                for y_i, y_pred in zip(y[:n], outputs[:n])
            ]

            # 方法1: WandbLogger.log_image で画像ログ
            wandb_logger.log_image(key="sample_images", images=images, caption=captions)

            # 方法2: 画像と予測を W&B Table で記録
            columns = ["image", "ground truth", "prediction"]
            data = [
                [wandb.Image(x_i), y_i, y_pred] or x_i,
                y_i,
                y_pred in list(zip(x[:n], y[:n], outputs[:n])),
            ]
            wandb_logger.log_table(key="sample_table", columns=columns, data=data)


trainer = pl.Trainer(callbacks=[LogPredictionSamplesCallback()])
```

## LightningとW&BのマルチGPU利用

PyTorch Lightning は DDP（分散データ並列）インターフェースでマルチGPUをサポートします。ただし、PyTorch Lightning の設計上、すべてのGPU（または各 Rank）で同じ初期条件でインスタンス化する必要があります。しかし Rank 0 プロセスだけが `wandb.run` オブジェクトにアクセスでき、Rank 0 以外では `wandb.run = None` になっています。このため Rank 0 以外が失敗し、Rank 0 が待ち状態（**デッドロック**）になることがあります。

このため、`wandb.run` オブジェクトに依存しないコーディング構成にすることをおすすめします。

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
    # すべてのランダムシードを統一
    # 分散トレーニングでは各Rankで同じ初期重みになることが重要
    # 合わないと勾配も合わず、収束しないことがある
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



## 事例

[ビデオチュートリアルとColabノートブック](https://wandb.me/lit-colab) で手順を確認できます。

## よくあるご質問

### W&B は Lightning とどのように統合されていますか？

コア統合は [Lightning `loggers` API](https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html) を利用しています。これにより、多くのロギングコードをフレームワーク非依存に記述可能です。`Logger` は [Lightning `Trainer`](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html) に渡され、その API の強力な [フック・コールバックシステム](https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html) を通じて動きます。これにより研究コードとエンジニアリング・ロギングコードをきれいに分離できます。

### 追加実装なしでログされる内容は？

モデルのチェックポイントは W&B に保存され、後から閲覧やダウンロードが可能です。そのほか、[システムメトリクス]({{< relref "/guides/models/app/settings-page/system-metrics.md" >}})（GPU使用率やネットワークI/O など）、ハードウェアやOS情報などの環境情報、[コードの状態]({{< relref "/guides/models/app/features/panels/code.md" >}})（git commit やパッチ、ノートブック、セッション履歴など）、標準出力内容 も記録されます。

### トレーニングの中で `wandb.run` を使いたい場合は？

必要な変数のスコープを自分で拡張してください。つまり、すべてのプロセスで初期条件が同じであることを自分で保証してください。

```python
if os.environ.get("LOCAL_RANK", None) is None:
    os.environ["WANDB_DIR"] = wandb.run.dir
```

このようにすると、`os.environ["WANDB_DIR"]` を使ってチェックポイント保存先ディレクトリを設定できます。これでどのプロセス（Rank 0 以外も）でも `wandb.run.dir` にアクセス可能です。