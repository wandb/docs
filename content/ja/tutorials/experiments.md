---
title: 実験をトラッキングする
menu:
  tutorials:
    identifier: ja-tutorials-experiments
weight: 1
---

{{< cta-button 
    colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_&_Biases.ipynb" 
>}}

[W&B](https://wandb.ai/site) を使うことで、機械学習の実験管理、モデルのチェックポイント管理、チームとのコラボレーションなどが簡単に行えます。

このノートブックでは、シンプルな PyTorch モデルを使って実験を作成し、管理する方法を学びます。ノートブックの最後には、チームのメンバーと共有・カスタマイズできるインタラクティブなプロジェクトダッシュボードを手に入れることができます。[ダッシュボード例はこちら](https://wandb.ai/wandb/wandb_example)。

## 前提条件

W&B の Python SDK をインストールし、ログインします:

```shell
!pip install wandb -qU
```

```python
# W&B アカウントにログイン
import wandb
import random
import math

# wandb の新しいバックエンドを一時的に有効化
wandb.require("core")
```

```python
wandb.login()
```

## W&B を使って機械学習実験をシミュレーション・管理する

機械学習実験を作成し、管理・可視化します。手順は以下の通りです:

1. [run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) を初期化し、追跡したいハイパーパラメーターを渡します。
2. トレーニングループ内で、精度や損失などのメトリクスをログに記録します。

```python
import wandb
import random

project="basic-intro"
config = {
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10,
}

with wandb.init(project=project, config=config) as run:
  # このブロックは、トレーニングループでメトリクスを記録する例です
  epochs = 10
  offset = random.random() / 5
  for epoch in range(2, epochs):
      acc = 1 - 2 ** -epoch - random.random() / epoch - offset
      loss = 2 ** -epoch + random.random() / epoch + offset

      # 2️. スクリプトから W&B にメトリクスをログする
      run.log({"acc": acc, "loss": loss})    
```

W&B プロジェクトで自分のモデルがどのように学習できたかを参照しましょう。前のセルで出力された URL リンクをコピー＆貼り付けすると、自分のモデルの挙動を示すグラフ付きダッシュボードのある W&B プロジェクトページが開きます。

下の画像はダッシュボードの一例です:

{{< img src="/images/tutorials/experiments-1.png" alt="W&B experiment tracking dashboard" >}}

このように、W&B を疑似的な機械学習トレーニングループに組み込む方法が分かりました。次は、PyTorch の基本的なニューラルネットワークを使って実際の実験を管理してみましょう。下記のコードでは、モデルのチェックポイントも W&B へアップロードでき、組織の他のチームと共有できます。

## PyTorch を使って機械学習実験を管理する

次のコードセルでは、シンプルな MNIST 分類器を定義し、トレーニングします。トレーニングの途中で W&B が URL を出力するので、プロジェクトページへのリンクをクリックすれば、リアルタイムでメトリクスの推移などを確認できます。

W&B の run では、自動的に [メトリクス]({{< relref path="/guides/models/track/runs/#workspace-tab" lang="ja" >}})、  
システム情報、  
[ハイパーパラメーター]({{< relref path="/guides/models/track/runs/#overview-tab" lang="ja" >}})、  
[ターミナル出力]({{< relref path="/guides/models/track/runs/#logs-tab" lang="ja" >}}) などが記録され、  
さらに、モデルの入力と出力を含む [インタラクティブなテーブル]({{< relref path="/guides/models/tables/" lang="ja" >}}) も見ることができます。

### PyTorch Dataloader のセットアップ

このセルでは、モデルをトレーニングするための便利な関数を定義します。これらの関数自体は W&B 専用ではありませんので詳細は割愛します。詳しくは PyTorch のドキュメントで [forward, backward トレーニングループ](https://pytorch.org/tutorials/beginner/nn_tutorial.html)、  
[PyTorch DataLoader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) の使い方、  
[`torch.nn.Sequential` クラス](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html) でのモデル定義方法をご参照ください。

```python
import wandb
import torch, torchvision
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision.transforms as T

MNIST.mirrors = [
    mirror for mirror in MNIST.mirrors if "http://yann.lecun.com/" not in mirror
]

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_dataloader(is_train, batch_size, slice=5):
    "トレーニング用 DataLoader を取得"
    full_dataset = MNIST(
        root=".", train=is_train, transform=T.ToTensor(), download=True
    )
    sub_dataset = torch.utils.data.Subset(
        full_dataset, indices=range(0, len(full_dataset), slice)
    )
    loader = torch.utils.data.DataLoader(
        dataset=sub_dataset,
        batch_size=batch_size,
        shuffle=True if is_train else False,
        pin_memory=True,
        num_workers=2,
    )
    return loader


def get_model(dropout):
    "シンプルなモデル"
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, 10),
    ).to(device)
    return model


def validate_model(model, valid_dl, loss_func, log_images=False, batch_idx=0):
    "バリデーションデータセットでモデルの性能を評価し、wandb.Table をログします"
    model.eval()
    val_loss = 0.0
    with torch.inference_mode():
        correct = 0
        for i, (images, labels) in enumerate(valid_dl):
            images, labels = images.to(device), labels.to(device)

            # Forward pass ➡
            outputs = model(images)
            val_loss += loss_func(outputs, labels) * labels.size(0)

            # 正解数を計算・加算
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            # 1バッチ分の画像をダッシュボードにログ（常に同一のbatch_idx）
            if i == batch_idx and log_images:
                log_image_table(images, predicted, labels, outputs.softmax(dim=1))
    return val_loss / len(valid_dl.dataset), correct / len(valid_dl.dataset)
```

### 予測値と正解値の比較テーブルを作成

このセルは W&B 特有なので詳しく解説します。

ここでは `log_image_table` という関数を定義します。この関数は W&B Table オブジェクトを生成し、各画像に対してモデルの予測結果を W&B に記録できるようにします（必須ではありませんがおすすめです）。

より具体的には、それぞれの行が、モデルへの入力画像、その予測値と実際の値（ラベル）を表示する構成になります。

```python
def log_image_table(images, predicted, labels, probs):
    "wandb.Table で (img, pred, target, scores) をログ"
    # 画像、ラベル、予測値を記録する wandb Table を作成
    table = wandb.Table(
        columns=["image", "pred", "target"] + [f"score_{i}" for i in range(10)]
    )
    for img, pred, targ, prob in zip(
        images.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")
    ):
        table.add_data(wandb.Image(img[0].numpy() * 255), pred, targ, *prob.numpy())

    with wandb.init() as run:
        run.log({"predictions_table": table}, commit=False)
```

### モデルのトレーニングとチェックポイントのアップロード

下記のコードでモデルをトレーニングし、チェックポイントを W&B プロジェクトに保存します。保存したチェックポイントはトレーニング中のモデル評価にいつでも活用できます。

W&B なら、保存したモデルやチェックポイントも組織内の他のメンバーと簡単に共有可能です。チーム以外のメンバーとモデルやチェックポイントを共有する方法は [W&B Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) をご覧ください。

```python
import wandb

config = {
    "epochs": 5,
    "batch_size": 128,
    "lr": 1e-3,
    "dropout": random.uniform(0.01, 0.80),
}

project = "pytorch-intro"

# wandb run の初期化
with wandb.init(project=project, config=config) as run:

    # 必要なら config を取得
    config = run.config

    # データ取得
    train_dl = get_dataloader(is_train=True, batch_size=config.batch_size)
    valid_dl = get_dataloader(is_train=False, batch_size=2 * config.batch_size)
    n_steps_per_epoch = math.ceil(len(train_dl.dataset) / config.batch_size)

    # シンプルな MLP モデル
    model = get_model(config.dropout)

    # 損失関数とオプティマイザーの定義
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # トレーニング
    example_ct = 0
    step_ct = 0
    for epoch in range(config.epochs):
        model.train()
        for step, (images, labels) in enumerate(train_dl):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            train_loss = loss_func(outputs, labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            example_ct += len(images)
            metrics = {
                "train/train_loss": train_loss,
                "train/epoch": (step + 1 + (n_steps_per_epoch * epoch))
                / n_steps_per_epoch,
                "train/example_ct": example_ct,
            }

            if step + 1 < n_steps_per_epoch:
                # トレーニング中のメトリクスを wandb に記録
                run.log(metrics)

            step_ct += 1

        val_loss, accuracy = validate_model(
            model, valid_dl, loss_func, log_images=(epoch == (config.epochs - 1))
        )

        # バリデーション・トレーニングのメトリクスもログ
        val_metrics = {"val/val_loss": val_loss, "val/val_accuracy": accuracy}
        run.log({**metrics, **val_metrics})

        # モデルのチェックポイントを wandb に保存
        torch.save(model, "my_model.pt")
        run.log_model(
            "./my_model.pt",
            "my_mnist_model",
            aliases=[f"epoch-{epoch+1}_dropout-{round(run.config.dropout, 4)}"],
        )

        print(
            f"Epoch: {epoch+1}, Train Loss: {train_loss:.3f}, Valid Loss: {val_loss:3f}, Accuracy: {accuracy:.2f}"
        )

    # テストセットがある場合はサマリーメトリクスとしてログ
    run.summary["test_accuracy"] = 0.8
```

これで、W&B を使って初めてのモデルのトレーニングが完了しました。上記のリンクをクリックし、メトリクスや保存したモデルのチェックポイントを W&B App UI の Artifacts タブで確認しましょう。

## （オプション）W&B アラートをセットアップ

[W&B Alerts]({{< relref path="/guides/models/track/runs/alert/" lang="ja" >}}) を使えば、Python コードから Slack やメールにアラートを送信することができます。

初めて Slack やメールアラートをコード経由で送信する際は、次の2ステップを行います:

1) W&B の [ユーザー設定](https://wandb.ai/settings) で Alerts を有効化  
2) コードに `run.alert()` を追加します。たとえば:

```python
run.alert(title="Low accuracy", text=f"Accuracy is below the acceptable threshold")
```

下記セルは、`run.alert()` の最小限の利用例を示します

```python
import wandb

# wandb run を開始
with wandb.init(project="pytorch-intro") as run:

    # モデルトレーニングループをシミュレート
    acc_threshold = 0.3
    for training_step in range(1000):

        # 精度にランダムな数値を生成
        accuracy = round(random.random() + random.random(), 3)
        print(f"Accuracy is: {accuracy}, {acc_threshold}")

        # 精度を wandb に記録
        run.log({"Accuracy": accuracy})

        # 精度がしきい値を下回ったら W&B アラートを発火し run を停止
        if accuracy <= acc_threshold:
            # wandb アラートの送信
            run.alert(
                title="Low Accuracy",
                text=f"Accuracy {accuracy} at step {training_step} is below the acceptable threshold, {acc_threshold}",
            )
            print("Alert triggered")
            break
```

さらに詳しい使い方は [W&B Alerts の概要]({{< relref path="/guides/models/track/runs/alert" lang="ja" >}}) をご覧ください。

## 次のステップ

次のチュートリアルでは、W&B Sweeps を使ったハイパーパラメーター最適化を学びます:  
[PyTorch でハイパーパラメータースイープ](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb)
