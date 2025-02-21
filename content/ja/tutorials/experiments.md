---
title: Track experiments
menu:
  tutorials:
    identifier: ja-tutorials-experiments
weight: 1
---

{{< cta-button
    colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_&_Biases.ipynb"
>}}

[W&B](https://wandb.ai/site) を使用して、 機械学習 の 実験管理 、 モデル の チェックポイント、 チーム とのコラボレーションなどを行いましょう。

この notebook では、簡単な PyTorch モデル を使用して 機械学習 の 実験 を作成および追跡します。この notebook の終わりまでに、チーム の他のメンバーと共有およびカスタマイズできるインタラクティブな プロジェクト の ダッシュボード が作成されます。 [ダッシュボード の例はこちら](https://wandb.ai/wandb/wandb_example)をご覧ください。

## 前提条件

W&B Python SDK をインストールしてログインします。

```shell
!pip install wandb -qU
```

```python
# W&B アカウントにログイン
import wandb
import random
import math

# Use wandb-core, temporary for wandb's new backend
wandb.require("core")
```

```python
wandb.login()
```

## W&B で 機械学習 の 実験 をシミュレーションおよび追跡する

機械学習 の 実験 を作成、追跡、および視覚化します。これを行うには:

1. [W&B run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) を初期化し、追跡する ハイパーパラメータ を渡します。
2. トレーニング ループ 内で、精度や損失などの メトリクス を ログ に記録します。

```
import random
import math

# 5 つのシミュレーションされた 実験 を起動
total_runs = 5
for run in range(total_runs):
  # 1️. この スクリプト を追跡するために新しい run を開始
  wandb.init(
      # この run が ログ に記録される プロジェクト を設定
      project="basic-intro",
      # run 名を渡します (そうでない場合は、sunshine-lollypop-10 のようにランダムに割り当てられます)
      name=f"experiment_{run}",
      # ハイパーパラメータ と run メタデータ を追跡
      config={
      "learning_rate": 0.02,
      "architecture": "CNN",
      "dataset": "CIFAR-100",
      "epochs": 10,
      })

  # この簡単なブロックは、 メトリクス を ログ に記録する トレーニング ループ をシミュレートします
  epochs = 10
  offset = random.random() / 5
  for epoch in range(2, epochs):
      acc = 1 - 2 ** -epoch - random.random() / epoch - offset
      loss = 2 ** -epoch + random.random() / epoch + offset

      # 2️. スクリプト から W&B に メトリクス を ログ に記録
      wandb.log({"acc": acc, "loss": loss})

  # run を完了としてマーク
  wandb.finish()
```

W&B プロジェクト で 機械学習 がどのように実行されたかを確認します。前のセルから出力された URL リンクをコピーして貼り付けます。URL は、グラフ が表示される ダッシュボード を含む W&B プロジェクト にリダイレクトされます

次の図は、 ダッシュボード がどのように見えるかを示しています。

{{< img src="/images/tutorials/experiments-1.png" alt="" >}}

W&B を 疑似 機械学習 トレーニング ループ に統合する方法がわかったので、基本的な PyTorch ニューラルネットワーク を使用して 機械学習 の 実験 を追跡しましょう。次の コード は、 モデル の チェックポイント を W&B にアップロードし、組織内の他の チーム と共有することもできます。

## Pytorch を使用して 機械学習 の 実験 を追跡する

次の コード セル は、単純な MNIST 分類器を定義してトレーニングします。トレーニング 中に、W&B が URL を出力するのが表示されます。 プロジェクト ページ リンクをクリックして、結果が W&B プロジェクト にライブ ストリーミングされるのを確認します。

W&B run は、[メトリクス]({{< relref path="/guides/models/track/runs/#workspace-tab" lang="ja" >}})、
[システム 情報]({{< relref path="/guides/models/track/runs/#system-tab" lang="ja" >}})、
[ハイパーパラメータ]({{< relref path="/guides/models/track/runs/#overview-tab" lang="ja" >}})、
[ターミナル 出力]({{< relref path="/guides/models/track/runs/#logs-tab" lang="ja" >}}) を自動的に ログ に記録し、
[インタラクティブな テーブル]({{< relref path="/guides/core/tables/" lang="ja" >}}) が表示されます。
モデル の 入力と出力があります。

### PyTorch Dataloader をセットアップする

次のセルは、 機械学習 モデル をトレーニングするために必要な便利な関数を定義します。関数自体は W&B に固有ではないため、ここでは詳細には説明しません。[`torch.nn.Sequential` Class](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html) を使用して [forward pass と backward pass のトレーニング ループ](https://pytorch.org/tutorials/beginner/nn_tutorial.html) を定義する方法、[PyTorch DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) を使用してトレーニング 用の データを読み込む方法、PyTorch モデル を定義する方法の詳細については、PyTorch のドキュメントを参照してください。

```python
# @title
import torch, torchvision
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision.transforms as T

MNIST.mirrors = [
    mirror for mirror in MNIST.mirrors if "http://yann.lecun.com/" not in mirror
]

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_dataloader(is_train, batch_size, slice=5):
    "Get a training dataloader"
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
    "A simple model"
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
    "Compute performance of the model on the validation dataset and log a wandb.Table"
    model.eval()
    val_loss = 0.0
    with torch.inference_mode():
        correct = 0
        for i, (images, labels) in enumerate(valid_dl):
            images, labels = images.to(device), labels.to(device)

            # Forward pass ➡
            outputs = model(images)
            val_loss += loss_func(outputs, labels) * labels.size(0)

            # Compute accuracy and accumulate
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            # Log one batch of images to the dashboard, always same batch_idx.
            if i == batch_idx and log_images:
                log_image_table(images, predicted, labels, outputs.softmax(dim=1))
    return val_loss / len(valid_dl.dataset), correct / len(valid_dl.dataset)
```

### 予測 値と 真 の 値を比較するテーブルを作成する

次のセルは W&B に固有であるため、見ていきましょう。

セルでは、`log_image_table` という関数を定義します。厳密に言うとオプションですが、この関数は W&B Table オブジェクトを作成します。テーブル オブジェクトを使用して、 モデル が各画像に対して予測した内容を示す テーブル を作成します。

より具体的には、各行は、 モデル に供給された画像と、予測 値および実際の値 (ラベル) で構成されます。

```python
def log_image_table(images, predicted, labels, probs):
    "Log a wandb.Table with (img, pred, target, scores)"
    # 画像、ラベル、および予測 を ログ に記録するための wandb Table を作成します
    table = wandb.Table(
        columns=["image", "pred", "target"] + [f"score_{i}" for i in range(10)]
    )
    for img, pred, targ, prob in zip(
        images.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")
    ):
        table.add_data(wandb.Image(img[0].numpy() * 255), pred, targ, *prob.numpy())
    wandb.log({"predictions_table": table}, commit=False)
```

### モデル をトレーニングして チェックポイント をアップロードする

次の コード は、 モデル の チェックポイント をトレーニングし、 プロジェクト に保存します。通常どおり モデル の チェックポイント を使用して、トレーニング 中の モデル のパフォーマンスを評価します。

W&B を使用すると、保存した モデル と モデル の チェックポイント を チーム または組織の他のメンバーと簡単に共有できます。 モデル と モデル の チェックポイント を チーム 外のメンバーと共有する方法については、[W&B Registry]({{< relref path="/guides/models/registry/" lang="ja" >}}) を参照してください。

```python
# Launch 3 experiments, trying different dropout rates
for _ in range(3):
    # initialise a wandb run
    wandb.init(
        project="pytorch-intro",
        config={
            "epochs": 5,
            "batch_size": 128,
            "lr": 1e-3,
            "dropout": random.uniform(0.01, 0.80),
        },
    )

    # Copy your config
    config = wandb.config

    # Get the data
    train_dl = get_dataloader(is_train=True, batch_size=config.batch_size)
    valid_dl = get_dataloader(is_train=False, batch_size=2 * config.batch_size)
    n_steps_per_epoch = math.ceil(len(train_dl.dataset) / config.batch_size)

    # A simple MLP model
    model = get_model(config.dropout)

    # Make the loss and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Training
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
                # Log train metrics to wandb
                wandb.log(metrics)

            step_ct += 1

        val_loss, accuracy = validate_model(
            model, valid_dl, loss_func, log_images=(epoch == (config.epochs - 1))
        )

        # Log train and validation metrics to wandb
        val_metrics = {"val/val_loss": val_loss, "val/val_accuracy": accuracy}
        wandb.log({**metrics, **val_metrics})

        # Save the model checkpoint to wandb
        torch.save(model, "my_model.pt")
        wandb.log_model(
            "./my_model.pt",
            "my_mnist_model",
            aliases=[f"epoch-{epoch+1}_dropout-{round(wandb.config.dropout, 4)}"],
        )

        print(
            f"Epoch: {epoch+1}, Train Loss: {train_loss:.3f}, Valid Loss: {val_loss:3f}, Accuracy: {accuracy:.2f}"
        )

    # If you had a test set, this is how you could log it as a Summary metric
    wandb.summary["test_accuracy"] = 0.8

    # Close your wandb run
    wandb.finish()
```

これで、W&B を使用して最初の モデル をトレーニングしました。上記のいずれかのリンクをクリックして、 メトリクス を確認し、W&B App UI の Artifacts タブに保存された モデル の チェックポイント を表示します。

## (オプション) W&B Alert をセットアップする

Python コード から Slack または メール に アラート を送信する [W&B Alerts]({{< relref path="/guides/models/track/runs/alert/" lang="ja" >}}) を作成します。

コード からトリガーされる Slack または メール アラート を初めて送信する場合は、次の 2 つの手順に従います。

1) W&B [ユーザー 設定](https://wandb.ai/settings)で Alerts をオンにします
2) `wandb.alert()` を コード に追加します。次に例を示します。

```python
wandb.alert(title="Low accuracy", text=f"Accuracy is below the acceptable threshold")
```

次のセルは、`wandb.alert` の使用方法を示す最小限の例を示しています。

```python
# Start a wandb run
wandb.init(project="pytorch-intro")

# Simulating a model training loop
acc_threshold = 0.3
for training_step in range(1000):

    # Generate a random number for accuracy
    accuracy = round(random.random() + random.random(), 3)
    print(f"Accuracy is: {accuracy}, {acc_threshold}")

    # Log accuracy to wandb
    wandb.log({"Accuracy": accuracy})

    # If the accuracy is below the threshold, fire a W&B Alert and stop the run
    if accuracy <= acc_threshold:
        # Send the wandb Alert
        wandb.alert(
            title="Low Accuracy",
            text=f"Accuracy {accuracy} at step {training_step} is below the acceptable theshold, {acc_threshold}",
        )
        print("Alert triggered")
        break

# Mark the run as finished (useful in Jupyter notebooks)
wandb.finish()
```

[W&B Alerts の完全なドキュメントはこちら]({{< relref path="/guides/models/track/runs/alert" lang="ja" >}})にあります。

## 次のステップ
次のチュートリアルでは、W&B Sweeps を使用して ハイパーパラメータ 最適化を行う方法を学習します。
[PyTorch を使用した ハイパーパラメータ sweep](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb)
