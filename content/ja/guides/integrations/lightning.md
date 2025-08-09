---
title: PyTorch Lightning
menu:
  default:
    identifier: ja-guides-integrations-lightning
    parent: integrations
weight: 340
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Optimize_PyTorch_Lightning_models_with_Weights_%26_Biases.ipynb" >}}

PyTorch Lightning は PyTorch コードの整理や、分散トレーニング・16ビット精度などの高度な機能を簡単に追加できる軽量なラッパーを提供します。W&B もまた、あなたの機械学習実験のログを記録する軽量なラッパーです。これらを自分で組み合わせる必要はありません。W&B は、[`WandbLogger`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb) を介して PyTorch Lightning ライブラリに直接組み込まれています。

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
**wandb.log() の利用について:** `WandbLogger` は Trainer の `global_step` を使って W&B にログを送ります。もしコード内で直接 `wandb.log` を追加で呼ぶ場合は、`wandb.log()` の `step` 引数を **指定しないでください**。

かわりに、他のメトリクスと同じく Trainer の `global_step` をログに入れましょう:

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

### サインアップしてAPIキーを作成

APIキーは、あなたのマシンを W&B へ認証するためのものです。APIキーはユーザープロフィールから発行できます。

{{% alert %}}
よりスムーズな方法として、[W&B認証ページ](https://wandb.ai/authorize)に直接アクセスしてAPIキーを発行できます。表示された API キーをコピーして、パスワードマネージャなど安全な場所に保存してください。
{{% /alert %}}

1. 右上のユーザープロフィールアイコンをクリックします。
1. **User Settings** を選択し、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックします。表示された API キーをコピーしてください。API キーを再度隠したい場合は、ページをリロードしてください。

### `wandb` ライブラリのインストールとログイン

`wandb` ライブラリをローカルにインストールし、ログインする方法:

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) に API キーを設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb` ライブラリをインストールし、ログインします。

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


## PyTorch Lightning の `WandbLogger` を使う

PyTorch Lightning には、メトリクスやモデルの重み、メディアなどをログするための複数の `WandbLogger` クラスが用意されています。

- [`PyTorch`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)
- [`Fabric`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)

Lightning と連携するには、`WandbLogger` をインスタンス化し、Lightning の `Trainer` または `Fabric` に渡します。

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


### よく使われるロガー引数

`WandbLogger` でよく利用されるパラメータを紹介します。すべての引数の詳細は PyTorch Lightning のドキュメントを参照してください。

- [`PyTorch`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)
- [`Fabric`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)

| パラメータ      | 説明                                                                                 |
| -------------- | ----------------------------------------------------------------------------------- |
| `project`      | ログを保存する wandb Project を指定                                                  |
| `name`         | wandb run に名前をつける                                                             |
| `log_model`    | `log_model="all"` で全モデル、`log_model=True` でトレーニング終了時のみ記録             |
| `save_dir`     | データを保存するパス                                                                  |

## ハイパーパラメーターのログ

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

## 他の設定パラメータの追加ログ

```python
# 1つのパラメータを追加
wandb_logger.experiment.config["key"] = value

# 複数のパラメータを追加
wandb_logger.experiment.config.update({key1: val1, key2: val2})

# wandb モジュールを直接利用
wandb.config["key"] = value
wandb.config.update()
```

## 勾配、パラメータのヒストグラム、モデルトポロジーのログ

`wandblogger.watch()` にモデルオブジェクトを渡すことで、トレーニング中の勾配やパラメータの監視が可能です。詳細は PyTorch Lightning の `WandbLogger` ドキュメントを参照してください。

## メトリクスの記録

{{< tabpane text=true >}}

{{% tab header="PyTorch Logger" value="pytorch" %}}

`LightningModule` 内、たとえば `training_step` や `validation_step` メソッド内で `self.log('my_metric_name', metric_vale)` を呼び出すことで、W&B にメトリクスを記録できます。

以下のコードスニペットは、メトリクスやハイパーパラメーターをログするための `LightningModule` の定義例です。この例ではメトリクス計算に [`torchmetrics`](https://github.com/PyTorchLightning/metrics) ライブラリを利用しています。

```python
import torch
from torch.nn import Linear, CrossEntropyLoss, functional as F
from torch.optim import Adam
from torchmetrics.functional import accuracy
from lightning.pytorch import LightningModule


class My_LitModule(LightningModule):
    def __init__(self, n_classes=10, n_layer_1=128, n_layer_2=256, lr=1e-3):
        """モデルのパラメータ定義に使うメソッド"""
        super().__init__()

        # mnist画像は (1, 28, 28) (チャンネル, 幅, 高さ)
        self.layer_1 = Linear(28 * 28, n_layer_1)
        self.layer_2 = Linear(n_layer_1, n_layer_2)
        self.layer_3 = Linear(n_layer_2, n_classes)

        self.loss = CrossEntropyLoss()
        self.lr = lr

        # ハイパーパラメーターを self.hparams に保存（W&B が自動でログ）
        self.save_hyperparameters()

    def forward(self, x):
        """推論用入力 -> 出力のメソッド"""

        # (b, 1, 28, 28) -> (b, 1*28*28)
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)

        # 3回 (linear + relu)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

    def training_step(self, batch, batch_idx):
        """1バッチ分の損失を返す"""
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # loss とメトリクスを記録
        self.log("train_loss", loss)
        self.log("train_accuracy", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        """メトリクスの記録に使用"""
        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # loss とメトリクスを記録
        self.log("val_loss", loss)
        self.log("val_accuracy", acc)
        return preds

    def configure_optimizers(self):
        """オプティマイザーの定義"""
        return Adam(self.parameters(), lr=self.lr)

    def _get_preds_loss_accuracy(self, batch):
        """train/valid/test ステップで共通して使う補助関数"""
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

## 指標の min/max を記録する

wandb の [`define_metric`]({{< relref path="/ref/python/sdk/classes/run.md#define_metric" lang="ja" >}}) 関数を使うと、W&B のサマリー指標に最小・最大・平均・ベスト値などを表示するか指定できます。`define_metric` を使わない場合、最後にログした値がサマリー指標として表示されます。詳細は [`define_metric` リファレンスドキュメント]({{< relref path="/ref/python/sdk/classes/run.md#define_metric" lang="ja" >}}) や [ガイド]({{< relref path="/guides/models/track/log/customize-logging-axes" lang="ja" >}}) をご覧ください。

W&B のサマリーで最大検証精度を記録したい場合は、トレーニング開始時に一度だけ `wandb.define_metric` を呼びます:

{{< tabpane text=true >}}
{{% tab header="PyTorch Logger" value="pytorch" %}}

```python
class My_LitModule(LightningModule):
    ...

    def validation_step(self, batch, batch_idx):
        if trainer.global_step == 0:
            wandb.define_metric("val_accuracy", summary="max")

        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # loss とメトリクスを記録
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

## モデルのチェックポイントを保存

モデルのチェックポイントを W&B の [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) として保存するには、Lightning の [`ModelCheckpoint`](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint) コールバックを使い、`WandbLogger` の `log_model` 引数を設定します。

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

_artifacts_ パネルから簡単にモデルチェックポイントを取得できるよう、_latest_ と _best_ エイリアスが自動で設定されます（W&B [Artifact]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) を参照）。

```python
# artifacts パネルから参照情報を取得可能
# "VERSION" はバージョン（例: "v2"）またはエイリアス（"latest" や "best"）を指定
checkpoint_reference = "USER/PROJECT/MODEL-RUN_ID:VERSION"
```

{{< tabpane text=true >}}
{{% tab header="Via Logger" value="logger" %}}

```python
# チェックポイントをローカルにダウンロード（未キャッシュの場合のみ）
wandb_logger.download_artifact(checkpoint_reference, artifact_type="model")
```

{{% /tab %}}

{{% tab header="Via wandb" value="wandb" %}}

```python
# チェックポイントをローカルにダウンロード（未キャッシュの場合のみ）
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

記録したモデルチェックポイントは [W&B Artifacts]({{< relref path="/guides/core/artifacts" lang="ja" >}}) のUIで参照でき、完全なモデルリネージ情報も含まれます（[例はこちら](https://wandb.ai/wandb/arttest/artifacts/model/iv3_trained/5334ab69740f9dda4fed/lineage?_gl=1*yyql5q*_ga*MTQxOTYyNzExOS4xNjg0NDYyNzk1*_ga_JH1SJHJQXJ*MTY5MjMwNzI2Mi4yNjkuMS4xNjkyMzA5NjM2LjM3LjAuMA..)）。

ベストなモデルチェックポイントをブックマークしチーム全体で管理したい場合は、[W&B Model Registry]({{< relref path="/guides/models" lang="ja" >}}) にリンクできます。

ここでは、タスクごとにベストなモデルを整理し、モデルライフサイクルを管理し、MLライフサイクル全体の追跡や監査を簡単にし、[オートメーション]({{< relref path="/guides/core/automations/" lang="ja" >}}) でWebhookやジョブによる下流処理も可能です。

## 画像・テキストなど各種メディアの記録

`WandbLogger` には `log_image`, `log_text`, `log_table` などメディア記録用のメソッドがあります。

また、`wandb.log` もしくは `trainer.logger.experiment.log` を直接呼ぶことで、Audio, 分子, ポイントクラウド, 3Dオブジェクトなど様々なメディア種のログが行えます。

{{< tabpane text=true >}}

{{% tab header="画像のログ" value="images" %}}

```python
# テンソル, numpy配列, PIL画像を使う場合
wandb_logger.log_image(key="samples", images=[img1, img2])

# キャプションの追加
wandb_logger.log_image(key="samples", images=[img1, img2], caption=["tree", "person"])

# ファイルパス指定
wandb_logger.log_image(key="samples", images=["img_1.jpg", "img_2.jpg"])

# trainer から .log を使う場合
trainer.logger.experiment.log(
    {"samples": [wandb.Image(img, caption=caption) for (img, caption) in my_images]},
    step=current_trainer_global_step,
)
```

{{% /tab %}}

{{% tab header="テキストのログ" value="text" %}}

```python
# data はリストのリスト
columns = ["input", "label", "prediction"]
my_data = [["cheese", "english", "english"], ["fromage", "french", "spanish"]]

# columns と data を使う場合
wandb_logger.log_text(key="my_samples", columns=columns, data=my_data)

# pandas DataFrame を使う場合
wandb_logger.log_text(key="my_samples", dataframe=my_dataframe)
```

{{% /tab %}}

{{% tab header="テーブルのログ" value="tables" %}}

```python
# テキストキャプション・画像・音声を含む W&B Table のログ
columns = ["caption", "image", "sound"]

# data はリストのリスト
my_data = [
    ["cheese", wandb.Image(img_1), wandb.Audio(snd_1)],
    ["wine", wandb.Image(img_2), wandb.Audio(snd_2)],
]

# Table の記録
wandb_logger.log_table(key="my_samples", columns=columns, data=data)
```

{{% /tab %}}

{{< /tabpane >}}

Lightning の Callback システムを使って、`WandbLogger` で W&B にログを記録するタイミングをコントロールできます。以下は検証画像と予測のサンプルをログする例です:

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
        """検証バッチ終了時に呼ばれます"""

        # `outputs` は `LightningModule.validation_step` の返り値で、この例では予測値

        # 最初のバッチからサンプル画像予測を20件ログ
        if batch_idx == 0:
            n = 20
            x, y = batch
            images = [img for img in x[:n]]
            captions = [
                f"Ground Truth: {y_i} - Prediction: {y_pred}"
                for y_i, y_pred in zip(y[:n], outputs[:n])
            ]

            # オプション1: `WandbLogger.log_image` を使う
            wandb_logger.log_image(key="sample_images", images=images, caption=captions)

            # オプション2: 画像と予測を W&B Table でログ
            columns = ["image", "ground truth", "prediction"]
            data = [
                [wandb.Image(x_i), y_i, y_pred] or x_i,
                y_i,
                y_pred in list(zip(x[:n], y[:n], outputs[:n])),
            ]
            wandb_logger.log_table(key="sample_table", columns=columns, data=data)


trainer = pl.Trainer(callbacks=[LogPredictionSamplesCallback()])
```

## Lightning と W&B で複数GPUを使う

PyTorch Lightning では、DDP インターフェースを通じてマルチGPUがサポートされています。ただし Lightning の設計上、各GPU（ランク）で同一の初期条件でインスタンス化する必要があります。一方で、唯一ランク0プロセスのみが `wandb.run` オブジェクトにアクセスできます。ランク0以外のプロセスでは `wandb.run = None` となり、これによってクラッシュする場合があります。

このような状況ではランク0プロセスが他のプロセスを待ち続けて **デッドロック** になるので、`wandb.run` オブジェクトに依存しない形でコードを記述してください。

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
    # 分散トレーニング環境では重要です
    # 各ランクで異なる初期重みだと勾配が合わず収束しません
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

[Colab ノートブック付きの動画チュートリアル](https://wandb.me/lit-colab) でさらに学べます。

## よくある質問

### W&B は Lightning とどう統合されていますか？

コアの統合は [Lightning `loggers` API](https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html) をベースにしています。このAPIを使うことで、フレームワークに依存しない形で多くのログ記述が可能です。`Logger` は [Lightning `Trainer`](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html) に渡され、その豊富な [フック&コールバックシステム](https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html) を通じて呼び出されます。これにより研究コードとエンジニアリングやロギング部分をきれいに分離して保てます。

### 追加コードなしでは何が記録されますか？

モデルのチェックポイントを W&B に保存します。後で参照したり他のRunでダウンロードしたりできます。[システムメトリクス]({{< relref path="/guides/models/app/settings-page/system-metrics.md" lang="ja" >}})（GPU使用率やネットワークI/O）、ハードウェアやOSなどの環境情報、[コードの状態]({{< relref path="/guides/models/app/features/panels/code.md" lang="ja" >}})（gitコミット履歴、ノートブックやセッション履歴）、標準出力に表示される内容などもキャプチャされます。

### トレーニングセットアップで `wandb.run` を使いたい場合は？

必要な変数のスコープを自分で拡張してください。すなわち、すべてのプロセスで初期条件が一致していることを確認しましょう。

```python
if os.environ.get("LOCAL_RANK", None) is None:
    os.environ["WANDB_DIR"] = wandb.run.dir
```

この場合、`os.environ["WANDB_DIR"]` を使ってモデルチェックポイントのディレクトリを設定できます。これによりランク0以外のプロセスでも `wandb.run.dir` にアクセスできるようになります。