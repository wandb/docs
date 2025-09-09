---
title: 実験を追跡する
menu:
  tutorials:
    identifier: ja-tutorials-experiments
weight: 1
---

{{< cta-button 
    colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_&_Biases.ipynb" 
>}}

[W&B](https://wandb.ai/site) を使うと、機械学習の 実験管理、モデル チェックポイントの保存、チームでのコラボレーションなどが行えます。 

このノートブックでは、シンプルな PyTorch モデルを使って機械学習の 実験を作成し、追跡します。ノートブックを終える頃には、チームのメンバーと共有・カスタマイズできるインタラクティブな W&B Project ダッシュボードが手に入ります。[サンプルのダッシュボードはこちら](https://wandb.ai/wandb/wandb_example).

## 前提条件

W&B の Python SDK をインストールしてログインします:


```shell
!pip install wandb -qU
```


```python
# W&B アカウントにログイン
import wandb
import random
import math

# wandb の新しいバックエンド用に一時的に wandb-core を使用
wandb.require("core")
```


```python
wandb.login()
```

## W&B で機械学習の 実験をシミュレートして追跡する

機械学習の 実験を作成・追跡・可視化します。手順は次のとおりです:

1. [run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) を初期化し、追跡したい ハイパーパラメーター を渡します。
2. トレーニングループ内で、精度や損失などの メトリクス をログします。



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
  # このブロックはメトリクスをログするトレーニングループをシミュレートします
  epochs = 10
  offset = random.random() / 5
  for epoch in range(2, epochs):
      acc = 1 - 2 ** -epoch - random.random() / epoch - offset
      loss = 2 ** -epoch + random.random() / epoch + offset

      # 2. スクリプトから W&B へメトリクスをログします
      run.log({"acc": acc, "loss": loss})    
```

W&B Project で機械学習の実行結果を確認しましょう。前のセルに出力された URL をコピー＆ペーストします。URL は、モデルの挙動を示すグラフを含むダッシュボードがある W&B Project へリダイレクトします。 

以下はダッシュボードの例です:

{{< img src="/images/tutorials/experiments-1.png" alt="W&B 実験管理ダッシュボード" >}}

擬似的な機械学習のトレーニングループに W&B を組み込む方法がわかったので、基本的な PyTorch ニューラルネットワークで 実験を追跡してみましょう。次のコードは、後で組織内の他のチームと共有できるよう、モデルの チェックポイント も W&B にアップロードします。

## PyTorch で機械学習の 実験を追跡する

次のコードセルでは、シンプルな MNIST 分類器を定義して学習します。トレーニング中、W&B が URL を出力します。Project ページのリンクをクリックすると、結果がリアルタイムで W&B Project にストリーミングされる様子を確認できます。

W&B の run は自動的に [メトリクス]({{< relref path="/guides/models/track/runs/#workspace-tab" lang="ja" >}})、システム情報、[ハイパーパラメーター]({{< relref path="/guides/models/track/runs/#overview-tab" lang="ja" >}})、[ターミナル出力]({{< relref path="/guides/models/track/runs/#logs-tab" lang="ja" >}}) をログし、モデルの入力と出力を含む [インタラクティブなテーブル]({{< relref path="/guides/models/tables/" lang="ja" >}}) が表示されます。 

### PyTorch DataLoader をセットアップする
次のセルでは、機械学習の モデルを学習するために必要ないくつかの便利な関数を定義します。これらの関数は W&B 固有ではないため、ここでは詳しくは扱いません。詳細は PyTorch のドキュメントを参照してください。具体的には、[forward および backward のトレーニングループ](https://pytorch.org/tutorials/beginner/nn_tutorial.html)、トレーニング用に データ を読み込むための [PyTorch DataLoader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) の使い方、[`torch.nn.Sequential` クラス](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html) を使って PyTorch の モデル を定義する方法などです。 


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

            # 精度を計算して累積します
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            # ダッシュボードに 1 バッチ分の画像をログします。常に同じ batch_idx を使用します。
            if i == batch_idx and log_images:
                log_image_table(images, predicted, labels, outputs.softmax(dim=1))
    return val_loss / len(valid_dl.dataset), correct / len(valid_dl.dataset)
```

### 予測値と正解値を比較するテーブルを作成する

次のセルは W&B 特有の処理なので、説明します。

このセルでは `log_image_table` という関数を定義します。技術的には任意ですが、この関数は W&B の Table オブジェクトを作成します。このテーブル オブジェクトを使って、各画像に対するモデルの予測を表示するテーブルを作成します。 

具体的には、各行にはモデルに入力した画像、予測値、実際の 値（ラベル）が含まれます。 


```python
def log_image_table(images, predicted, labels, probs):
    "Log a wandb.Table with (img, pred, target, scores)"
    # 画像・ラベル・予測を記録するための wandb Table を作成します
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

### モデルを学習して チェックポイント をアップロードする

次のコードは モデルを学習し、Project に モデルの チェックポイント を保存します。通常どおり、トレーニング中のモデルの挙動を評価するために チェックポイント を活用してください。 

また W&B なら、保存した モデルや モデルの チェックポイント をチームや組織の他のメンバーと簡単に共有できます。チーム外のメンバーと共有する方法は [W&B Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) を参照してください。


```python
import wandb

config = {
    "epochs": 5,
    "batch_size": 128,
    "lr": 1e-3,
    "dropout": random.uniform(0.01, 0.80),
}

project = "pytorch-intro"

# wandb の run を初期化します
with wandb.init(project=project, config=config) as run:

    # 必要に応じて config をコピーします
    config = run.config

    # データを取得します
    train_dl = get_dataloader(is_train=True, batch_size=config.batch_size)
    valid_dl = get_dataloader(is_train=False, batch_size=2 * config.batch_size)
    n_steps_per_epoch = math.ceil(len(train_dl.dataset) / config.batch_size)

    # シンプルな MLP モデル
    model = get_model(config.dropout)

    # 損失関数と オプティマイザー を用意します
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
                # 学習のメトリクスを wandb にログします
                run.log(metrics)

            step_ct += 1

        val_loss, accuracy = validate_model(
            model, valid_dl, loss_func, log_images=(epoch == (config.epochs - 1))
        )

        # 学習および検証のメトリクスを wandb にログします
        val_metrics = {"val/val_loss": val_loss, "val/val_accuracy": accuracy}
        run.log({**metrics, **val_metrics})

        # モデルの チェックポイント を wandb に保存します
        torch.save(model, "my_model.pt")
        run.log_model(
            "./my_model.pt",
            "my_mnist_model",
            aliases=[f"epoch-{epoch+1}_dropout-{round(run.config.dropout, 4)}"],
        )

        print(
            f"Epoch: {epoch+1}, Train Loss: {train_loss:.3f}, Valid Loss: {val_loss:3f}, Accuracy: {accuracy:.2f}"
        )

    # テストセット がある場合は、次のように Summary メトリクスとしてログできます
    run.summary["test_accuracy"] = 0.8
```

これで W&B を使って最初の モデル を学習できました。上のいずれかのリンクをクリックすると、メトリクスを確認でき、W&B App の UI にある Artifacts タブで保存済みの モデル チェックポイントも確認できます。

## （任意）W&B Alert を設定する

[W&B Alerts]({{< relref path="/guides/models/track/runs/alert/" lang="ja" >}}) を作成して、Python コードから Slack またはメールに通知を送信します。 

コードからトリガーされる Slack またはメールのアラートを初めて送るときは、次の 2 ステップが必要です:

1) W&B の [User Settings](https://wandb.ai/settings) で Alerts を有効にする
2) コードに `run.alert()` を追加する。例:

```python
run.alert(title="Low accuracy", text=f"Accuracy is below the acceptable threshold")
```

以下のセルは、`run.alert()` の使い方を確認できる最小例です。


```python
import wandb

# wandb の run を開始します
with wandb.init(project="pytorch-intro") as run:

    # モデルのトレーニングループをシミュレートします
    acc_threshold = 0.3
    for training_step in range(1000):

        # 精度の疑似値を生成します
        accuracy = round(random.random() + random.random(), 3)
        print(f"Accuracy is: {accuracy}, {acc_threshold}")

        # 精度を wandb にログします
        run.log({"Accuracy": accuracy})

        # 精度がしきい値を下回ったら W&B Alert を送信し、run を停止します
        if accuracy <= acc_threshold:
            # wandb の Alert を送信します
            run.alert(
                title="Low Accuracy",
                text=f"Accuracy {accuracy} at step {training_step} is below the acceptable threshold, {acc_threshold}",
            )
            print("Alert triggered")
            break
```

詳細は [W&B Alerts の概要]({{< relref path="/guides/models/track/runs/alert" lang="ja" >}}) を参照してください。

## 次のステップ
次のチュートリアルでは、W&B Sweeps を使った ハイパーパラメーター 最適化を学びます:
[PyTorch を使ったハイパーパラメーター Sweeps](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb)