---
title: PyTorch Lightning
menu:
  default:
    identifier: ja-guides-integrations-lightning
    parent: integrations
weight: 340
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Optimize_PyTorch_Lightning_models_with_Weights_%26_Biases.ipynb" >}}

PyTorch Lightning は、PyTorch のコードを整理し、分散トレーニングや16ビット精度などの高度な機能を簡単に追加するための軽量なラッパーを提供します。W&B は、ML 実験をログするための軽量なラッパーを提供します。しかし、自分でこの2つを組み合わせる必要はありません。Weights & Biases は、[**`WandbLogger`**](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb) を通じて、PyTorch Lightning ライブラリに直接組み込まれています。

## Lightning とのインテグレーション

{{< tabpane text=true >}}
{{% tab header="PyTorch Logger" value="pytorch" %}}

```python
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer

wandb_logger = WandbLogger(log_model="all")
trainer = Trainer(logger=wandb_logger)
```

{{% alert %}}
**wandb.log() の使用:** `WandbLogger` は、Trainer の `global_step` を使用して W&B にログを送信します。コード内で `wandb.log` を直接呼び出す場合、`wandb.log()` の `step` 引数を使用しないでください。

代わりに、Trainer の `global_step` を他のメトリクスと同様にログしてください。

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

{{< img src="/images/integrations/n6P7K4M.gif" alt="どこでもアクセスできるインタラクティブなダッシュボードなど！" >}}

### サインアップと API キーの作成

APIキーは、あなたのマシンを W&B に認証します。ユーザープロフィールから APIキーを生成できます。

{{% alert %}}
よりスムーズな方法として、[https://wandb.ai/authorize](https://wandb.ai/authorize) に直接アクセスして APIキー を生成できます。表示されたAPIキーをコピーし、パスワードマネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上のユーザープロフィールアイコンをクリックします。
2. **User Settings** を選択し、**API Keys** セクションまでスクロールします。
3. **Reveal** をクリックし、表示されたAPIキーをコピーします。APIキーを非表示にするには、ページを再読み込みします。

### `wandb` ライブラリをインストールしてログイン

`wandb` ライブラリをローカルにインストールしてログインするには：

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) をあなたの APIキーに設定します。

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

PyTorch Lightning には、メトリクスやモデルの重み、メディアなどをログするための複数の `WandbLogger` クラスがあります。

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

### 一般的なロガー引数

以下は、WandbLogger で最も使用されるパラメーターのいくつかです。すべてのロガー引数の詳細については、PyTorch Lightning のドキュメントを確認してください。

- [`PyTorch`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)
- [`Fabric`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)

| パラメーター | 説明                                                                 |
| ----------- | ------------------------------------------------------------------ |
| `project`   | ログを記録する wandb Project を定義                                     |
| `name`      | wandb run に名前を付ける                                               |
| `log_model` | `log_model="all"` の場合はすべてのモデルをログし、`log_model=True` の場合はトレーニング終了時にログする |
| `save_dir`  | データが保存されるパス                                                        |

## ハイパーパラメーターをログに記録する

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

## 追加の設定パラメーターをログに記録する

```python
# パラメータを一つ追加する
wandb_logger.experiment.config["key"] = value

# 複数のパラメータを追加する
wandb_logger.experiment.config.update({key1: val1, key2: val2})

# wandb モジュールを直接使用する
wandb.config["key"] = value
wandb.config.update()
```

## 勾配、パラメータヒストグラム、モデルトポロジーをログに記録する

モデルオブジェクトを `wandblogger.watch()` に渡すことで、トレーニング中のモデルの勾配とパラメータを監視できます。PyTorch Lightning の `WandbLogger` ドキュメントを参照してください。

## メトリクスをログに記録する

{{< tabpane text=true >}}

{{% tab header="PyTorch Logger" value="pytorch" %}}

`WandbLogger` を使用する際に、`self.log('my_metric_name', metric_value)` を `LightningModule` 内で呼び出すことで、メトリクスを W&B にログできます。たとえば、`training_step` または `validation_step メソッド` 内で実行します。

以下のコードスニペットは、メトリクスと `LightningModule` ハイパーパラメーターをログするための `LightningModule` を定義する方法を示しています。この例では、メトリクスを計算するために [`torchmetrics`](https://github.com/PyTorchLightning/metrics) ライブラリを使用しています。

```python
import torch
from torch.nn import Linear, CrossEntropyLoss, functional as F
from torch.optim import Adam
from torchmetrics.functional import accuracy
from lightning.pytorch import LightningModule


class My_LitModule(LightningModule):
    def __init__(self, n_classes=10, n_layer_1=128, n_layer_2=256, lr=1e-3):
        """モデルパラメーターを定義するためのメソッド"""
        super().__init__()

        # mnist 画像は (1, 28, 28) (チャネル、幅、高さ)
        self.layer_1 = Linear(28 * 28, n_layer_1)
        self.layer_2 = Linear(n_layer_1, n_layer_2)
        self.layer_3 = Linear(n_layer_2, n_classes)

        self.loss = CrossEntropyLoss()
        self.lr = lr

        # ハイパーパラメーターを self.hparams に保存 (W&B に自動的にログされる)
        self.save_hyperparameters()

    def forward(self, x):
        """推論用の入力 -> 出力のメソッド"""

        # (b, 1, 28, 28) -> (b, 1*28*28)
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)

        # 3 x (線形+リレー) を行う
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

    def training_step(self, batch, batch_idx):
        """単一のバッチから損失を返す必要あり"""
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # ロスとメトリクスをログする
        self.log("train_loss", loss)
        self.log("train_accuracy", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        """メトリクスのログ記録に使用される"""
        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # ロスとメトリクスをログする
        self.log("val_loss", loss)
        self.log("val_accuracy", acc)
        return preds

    def configure_optimizers(self):
        """モデルオプティマイザーを定義する"""
        return Adam(self.parameters(), lr=self.lr)

    def _get_preds_loss_accuracy(self, batch):
        """train/valid/test ステップが類似しているための便利な関数"""
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

## メトリクスの最小値/最大値をログに記録する

wandbの[`define_metric`]({{< relref path="/ref/python/run#define_metric" lang="ja" >}})関数を使用して、W&Bサマリーメトリクスに、そのメトリクスの最小値、最大値、平均、または最良値を表示するかどうかを定義できます。 `define_metric` が使用されていない場合、ログされた最後の値がサマリーメトリクスに表示されます。 `define_metric` [リファレンスドキュメント]({{< relref path="/ref/python/run#define_metric" lang="ja" >}}) を参照し、[ガイド]({{< relref path="/guides/models/track/log/customize-logging-axes" lang="ja" >}}) を確認してください。

W&Bサマリーメトリクスで最大の検証精度を追跡するようにW&Bに指示するには、トレーニングの開始時に一度だけ `wandb.define_metric` を呼び出します。

{{< tabpane text=true >}}
{{% tab header="PyTorch Logger" value="pytorch" %}}

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

## モデルをチェックポイントする

モデルのチェックポイントを W&B [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) として保存するには、Lightning [`ModelCheckpoint`](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint) コールバックを使用し、`WandbLogger` で `log_model` 引数を設定します。

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

_最新_ と _最良_ のエイリアスは自動的に設定され、W&B [Artifact]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) からモデルのチェックポイントを簡単に取得できます：

```python
# アーティファクトパネルで参照を取得できる
# "VERSION" はバージョン (例: "v2") またはエイリアス ("latest" または "best") であり得ます
checkpoint_reference = "USER/PROJECT/MODEL-RUN_ID:VERSION"
```

{{< tabpane text=true >}}
{{% tab header="ロガー経由" value="logger" %}}

```python
# チェックポイントをローカルにダウンロード (まだキャッシュされていない場合)
wandb_logger.download_artifact(checkpoint_reference, artifact_type="model")
```

{{% /tab %}}

{{% tab header="wandb 経由" value="wandb" %}}

```python
# チェックポイントをローカルにダウンロード (まだキャッシュされていない場合)
run = wandb.init(project="MNIST")
artifact = run.use_artifact(checkpoint_reference, type="model")
artifact_dir = artifact.download()
```

{{% /tab %}}
{{< /tabpane >}}

{{< tabpane text=true >}}
{{% tab header="PyTorch Logger" value="pytorch" %}}

```python
# チェックポイントをロード
model = LitModule.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")
```

{{% /tab %}}

{{% tab header="Fabric Logger" value="fabric" %}}

```python
# 生のチェックポイントを要求
full_checkpoint = fabric.load(Path(artifact_dir) / "model.ckpt")

model.load_state_dict(full_checkpoint["model"])
optimizer.load_state_dict(full_checkpoint["optimizer"])
```

{{% /tab %}}
{{< /tabpane >}}

ログしたモデルのチェックポイントは、[W&B Artifacts]({{< relref path="/guides/core/artifacts" lang="ja" >}}) の UI を通じて確認でき、完全なモデルリネージを含みます（UI でのモデルチェックポイントの例はこちらで確認できます: [here](https://wandb.ai/wandb/arttest/artifacts/model/iv3_trained/5334ab69740f9dda4fed/lineage?_gl=1*yyql5q*_ga*MTQxOTYyNzExOS4xNjg0NDYyNzk1*_ga_JH1SJHJQXJ*MTY5MjMwNzI2Mi4yNjkuMS4xNjkyMzA5NjM2LjM3LjAuMA..))。

最高のモデルチェックポイントをまとめ、チーム間で整理するには、[W&B Model Registry]({{< relref path="/guides/models" lang="ja" >}}) にリンクすることができます。

ここで、タスクごとに最適なモデルを整理し、モデルのライフサイクルを管理し、ML ライフサイクル全体での容易なトラッキングと監査を促進し、Webhook やジョブを使用して[自動化]({{< relref path="/guides/models/automations/project-scoped-automations/#create-a-webhook-automation" lang="ja" >}})を下流のアクションに実行できます。

## 画像、テキスト、およびその他をログに記録する

`WandbLogger` には、メディアをログするための `log_image`、`log_text`、および `log_table` メソッドがあります。

また、Audio、Molecules、Point Clouds、3D Objects などの他のメディアタイプをログするために、`wandb.log` または `trainer.logger.experiment.log` を直接呼び出すこともできます。

{{< tabpane text=true >}}

{{% tab header="画像をログ" value="images" %}}

```python
# テンソル、numpy 配列、または PIL 画像を使用
wandb_logger.log_image(key="samples", images=[img1, img2])

# キャプションを追加
wandb_logger.log_image(key="samples", images=[img1, img2], caption=["tree", "person"])

# ファイルパスを使用
wandb_logger.log_image(key="samples", images=["img_1.jpg", "img_2.jpg"])

# トレーナー内で .log を使用
trainer.logger.experiment.log(
    {"samples": [wandb.Image(img, caption=caption) for (img, caption) in my_images]},
    step=current_trainer_global_step,
)
```

{{% /tab %}}

{{% tab header="テキストをログ" value="text" %}}

```python
# データはリストのリストである必要があります
columns = ["input", "label", "prediction"]
my_data = [["cheese", "english", "english"], ["fromage", "french", "spanish"]]

# 列とデータを使用
wandb_logger.log_text(key="my_samples", columns=columns, data=my_data)

# Pandas DataFrame を使用
wandb_logger.log_text(key="my_samples", dataframe=my_dataframe)
```

{{% /tab %}}

{{% tab header="テーブルをログ" value="tables" %}}

```python
# 注釈、画像、オーディオを持つ W&B テーブルをログ
columns = ["caption", "image", "sound"]

# データはリストのリストである必要があります
my_data = [
    ["cheese", wandb.Image(img_1), wandb.Audio(snd_1)],
    ["wine", wandb.Image(img_2), wandb.Audio(snd_2)],
]

# テーブルをログ
wandb_logger.log_table(key="my_samples", columns=columns, data=data)
```

{{% /tab %}}

{{< /tabpane >}}

Lightning のコールバックシステムを使用して、WandbLogger 経由で Weights & Biases にログするタイミングを制御できます。この例では、検証画像と予測のサンプルをログに記録します:

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
        """バリデーションバッチの終了時に呼び出されます。"""

        # `outputs` は `LightningModule.validation_step` から得られ、
        # この場合はモデルの予測に対応します。

        # 20 サンプルの画像予測を最初のバッチからログします
        if batch_idx == 0:
            n = 20
            x, y = batch
            images = [img for img in x[:n]]
            captions = [
                f"Ground Truth: {y_i} - Prediction: {y_pred}"
                for y_i, y_pred in zip(y[:n], outputs[:n])
            ]

            # オプション 1: `WandbLogger.log_image` で画像をログする
            wandb_logger.log_image(key="sample_images", images=images, caption=captions)

            # オプション 2: W&B Table として画像と予測をログする
            columns = ["image", "ground truth", "prediction"]
            data = [
                [wandb.Image(x_i), y_i, y_pred] or x_i,
                y_i,
                y_pred in list(zip(x[:n], y[:n], outputs[:n])),
            ]
            wandb_logger.log_table(key="sample_table", columns=columns, data=data)

trainer = pl.Trainer(callbacks=[LogPredictionSamplesCallback()])
```

## Lightning と W&B を使用して複数の GPU を使用する

PyTorch Lightning には、DDP インターフェースを通じた Multi-GPU サポートがあります。ただし、PyTorch Lightning の設計では、GPU のインスタンス化に注意が必要です。

Lightning は、トレーニングループ内の各 GPU (または Rank) が、同じ初期条件で正確に同じ方法でインスタンス化される必要があると想定しています。ただし、rank 0 プロセスのみが `wandb.run` オブジェクトへのアクセスを取得し、非ゼロランクプロセスでは: `wandb.run = None` です。これにより、非ゼロプロセスが失敗する可能性があります。このような状況は、rank 0 プロセスがすでにクラッシュした非ゼロランクプロセスの参加を待機するため、**デッドロック** に陥る可能性があります。

この理由から、トレーニングコードのセットアップに注意が必要です。`wandb.run` オブジェクトに依存しないコードセットアップをお勧めします。

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
    # すべてのランダムシードを同じ値に設定します。
    # 分散トレーニング環境ではこれは重要です。
    # 各ランクは独自の初期重みのセットを受け取ります。
    # それらが一致しない場合、勾配も一致しません。
    # これにより、収収束しないトレーニングになる可能性があります。
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

Colab のビデオチュートリアルでフォローすることができます: [here](https://wandb.me/lit-colab)。

## よくある質問

### W&B は Lightning とどのようにインテグレートしますか？

コアのインテグレーションは、[Lightning `loggers` API](https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html) に基づいており、フレームワークに依存しない方法で多くのログコードを記述できます。`Logger` は [Lightning `Trainer`](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html) に渡され、この API の豊富な [フックとコールバックシステム](https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html) に基づいてトリガーされます。これにより、研究コードをエンジニアリングとログコードから良好に分離します。

### W&B は追加コードなしで何をログするのでしょうか？

モデルのチェックポイントを W&B に保存し、将来の Runs で使用するためにこれを参照したりダウンロードできるようにします。また、[システムメトリクス]({{< relref path="/guides/models/app/settings-page/system-metrics.md" lang="ja" >}})、GPU 使用率やネットワーク I/O、ハードウェアや OS 情報などの環境情報、[コード状態]({{< relref path="/guides/models/app/features/panels/code.md" lang="ja" >}})（git コミットや差分パッチ、ノートブックの内容とセッション履歴を含む）、および標準出力に印刷されるものすべてをキャプチャします。

### トレーニングセットアップで `wandb.run` を使用する必要がある場合はどうしますか？

アクセスする必要がある変数のスコープを自分で広げる必要があります。つまり、初期条件がすべてのプロセスで同じであることを確認してください。

```python
if os.environ.get("LOCAL_RANK", None) is None:
    os.environ["WANDB_DIR"] = wandb.run.dir
```

初期条件が一致していれば、`os.environ["WANDB_DIR"]` を使用してモデルチェックポイントディレクトリーを設定できます。この方法で、非ゼロランクのプロセスでも `wandb.run.dir` にアクセスできます。