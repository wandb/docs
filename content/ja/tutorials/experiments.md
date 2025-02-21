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

[W&B](https://wandb.ai/site) を使用して、機械学習の実験管理、モデルチェックポイント管理、チームとのコラボレーションなどを行いましょう。

このノートブックでは、簡単な PyTorch モデルを使って機械学習の実験を作成し、記録します。ノートブックの終わりには、他のチームメンバーと共有し、カスタマイズできるインタラクティブなプロジェクトダッシュボードを持つことができるでしょう。[こちらでダッシュボードの例をご覧ください](https://wandb.ai/wandb/wandb_example)。

## 必要条件

W&B Python SDK をインストールし、ログインしてください。

```shell
!pip install wandb -qU
```

```python
# W&B アカウントにログイン
import wandb
import random
import math

# 一時的に wandb の新しいバックエンドを使用します
wandb.require("core")
```

```python
wandb.login()
```

## W&B を使って機械学習実験をシミュレートし追跡する

機械学習の実験を作成、追跡、可視化します。これを行うために：

1. [W&B run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) を初期化し、追跡したいハイパーパラメーターを渡します。
2. トレーニングループ内で、精度や損失などのメトリクスをログします。

```
import random
import math

# 5 つのシミュレートされた実験を実行します
total_runs = 5
for run in range(total_runs):
  # 1️. このスクリプトを追跡するための新しい run を開始します
  wandb.init(
      # この run がログされるプロジェクトを設定します
      project="basic-intro",
      # run 名を渡します（指定がない場合はランダムに割り当てられます）
      name=f"experiment_{run}",
      # ハイパーパラメーターと run のメタデータを追跡します
      config={
      "learning_rate": 0.02,
      "architecture": "CNN",
      "dataset": "CIFAR-100",
      "epochs": 10,
      })

  # この単純なブロックは、メトリクスをログするトレーニングループをシミュレートします
  epochs = 10
  offset = random.random() / 5
  for epoch in range(2, epochs):
      acc = 1 - 2 ** -epoch - random.random() / epoch - offset
      loss = 2 ** -epoch + random.random() / epoch + offset

      # 2️. スクリプトから W&B にメトリクスをログします
      wandb.log({"acc": acc, "loss": loss})

  # run を完了としてマークします
  wandb.finish()
```

W&B プロジェクトで機械学習のパフォーマンスを確認します。前のセルから出力された URL リンクをコピーして貼り付けます。URL は、グラフが表示される W&B プロジェクトのダッシュボードにリダイレクトされます。

次の画像は、ダッシュボードの例を示しています：

{{< img src="/images/tutorials/experiments-1.png" alt="" >}}

W&B を疑似機械学習トレーニングループに統合する方法を理解したので、基本的な PyTorch ニューラルネットワークを使った機械学習実験を追跡してみましょう。以下のコードは、モデルのチェックポイントも W&B にアップロードし、組織内の他のチームと共有できるようにします。

## PyTorch を使った機械学習実験の追跡

以下のコードセルは、シンプルな MNIST 分類器を定義し、トレーニングします。トレーニング中に W&B が URL を出力します。プロジェクトページのリンクをクリックすると、ワンドバ・プロジェクトに結果がライブで表示されます。

W&B は自動的に [メトリクス]({{< relref path="/guides/models/track/runs/#workspace-tab" lang="ja" >}})、[システム情報]({{< relref path="/guides/models/track/runs/#system-tab" lang="ja" >}})、[ハイパーパラメーター]({{< relref path="/guides/models/track/runs/#overview-tab" lang="ja" >}})、[ターミナル出力]({{< relref path="/guides/models/track/runs/#logs-tab" lang="ja" >}}) をログし、モデルの入力と出力を表示する [インタラクティブテーブル]({{< relref path="/guides/core/tables/" lang="ja" >}}) を見ることができます。

### PyTorch Dataloader のセットアップ

以下のセルは、機械学習モデルをトレーニングするために必要な便利な関数を定義しています。これらの関数自体は W&B に特化したものではないので、ここでは詳しく説明しません。トレーニングループの [forward, backward pass](https://pytorch.org/tutorials/beginner/nn_tutorial.html) の定義方法や、PyTorch DataLoaders を用いてトレーニングデータをロードする方法の詳細については、PyTorch のドキュメントをご覧ください。PyTorch モデルを定義する際の [`torch.nn.Sequential` クラス](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html) の使用方法についても同様です。

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
    "トレーニングデータローダーを取得する"
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
    "バリデーションデータセット上でのモデルのパフォーマンスを計算し、wandb.Table をログする"
    model.eval()
    val_loss = 0.0
    with torch.inference_mode():
        correct = 0
        for i, (images, labels) in enumerate(valid_dl):
            images, labels = images.to(device), labels.to(device)

            # Forward pass ➡
            outputs = model(images)
            val_loss += loss_func(outputs, labels) * labels.size(0)

            # 精度を計算し、累積
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            # ダッシュボードに1 バッチの画像をログ、常に同じ batch_idx
            if i == batch_idx and log_images:
                log_image_table(images, predicted, labels, outputs.softmax(dim=1))
    return val_loss / len(valid_dl.dataset), correct / len(valid_dl.dataset)
```

### 予測値と実際の値を比較するテーブルを作成する

以下のセルは W&B 独自のものですので、詳しく見ていきましょう。

このセルでは `log_image_table` という関数を定義しています。技術的には任意ですが、この関数は W&B の Table オブジェクトを作成します。この Table オブジェクトを使用して、各画像に対してモデルが予測した内容を示すテーブルを作成します。

より具体的には、各行には、モデルに入力された画像、予測された値、および実際の値（ラベル）が含まれます。

```python
def log_image_table(images, predicted, labels, probs):
    "画像、予測、目標、スコアで wandb.Table をログする"
    # 画像、ラベル、予測をログする wandb Table を作成
    table = wandb.Table(
        columns=["image", "pred", "target"] + [f"score_{i}" for i in range(10)]
    )
    for img, pred, targ, prob in zip(
        images.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")
    ):
        table.add_data(wandb.Image(img[0].numpy() * 255), pred, targ, *prob.numpy())
    wandb.log({"predictions_table": table}, commit=False)
```

### モデルをトレーニングし、チェックポイントをアップロードする

以下のコードはモデルをトレーニングし、モデルのチェックポイントをプロジェクトに保存します。トレーニング中にモデルのパフォーマンスを評価するために、通常行うようにモデルチェックポイントを使用します。

W&B は、保存したモデルやモデルチェックポイントをチームや組織の他のメンバーと簡単に共有できるようにします。チーム外のメンバーとモデルやモデルチェックポイントを共有する方法については、[W&B Registry]({{< relref path="/guides/models/registry/" lang="ja" >}})を参照してください。

```python
# 3 つの実験を開始し、異なるドロップアウト率を試します
for _ in range(3):
    # wandb run を初期化
    wandb.init(
        project="pytorch-intro",
        config={
            "epochs": 5,
            "batch_size": 128,
            "lr": 1e-3,
            "dropout": random.uniform(0.01, 0.80),
        },
    )

    # コンフィグをコピー
    config = wandb.config

    # データを取得
    train_dl = get_dataloader(is_train=True, batch_size=config.batch_size)
    valid_dl = get_dataloader(is_train=False, batch_size=2 * config.batch_size)
    n_steps_per_epoch = math.ceil(len(train_dl.dataset) / config.batch_size)

    # シンプルな MLP モデル
    model = get_model(config.dropout)

    # ロスとオプティマイザを作成
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
                # wandb へのトレインメトリクスをログ
                wandb.log(metrics)

            step_ct += 1

        val_loss, accuracy = validate_model(
            model, valid_dl, loss_func, log_images=(epoch == (config.epochs - 1))
        )

        # wandb へのトレインとバリデーションメトリクスをログ
        val_metrics = {"val/val_loss": val_loss, "val/val_accuracy": accuracy}
        wandb.log({**metrics, **val_metrics})

        # モデルのチェックポイントを wandb に保存
        torch.save(model, "my_model.pt")
        wandb.log_model(
            "./my_model.pt",
            "my_mnist_model",
            aliases=[f"epoch-{epoch+1}_dropout-{round(wandb.config.dropout, 4)}"],
        )

        print(
            f"Epoch: {epoch+1}, Train Loss: {train_loss:.3f}, Valid Loss: {val_loss:3f}, Accuracy: {accuracy:.2f}"
        )

    # テストセットがあれば、これを使ってサマリーメトリクスとしてログできます
    wandb.summary["test_accuracy"] = 0.8

    # wandb run を終了
    wandb.finish()
```

これで W&B を使用して最初のモデルをトレーニングしました。上記のリンクをクリックして、メトリクスを確認し、保存されたモデルチェックポイントを W&B アプリ UI のアーティファクトタブで確認してください。

## (任意) W&B アラートを設定する

[W&B Alerts]({{< relref path="/guides/models/track/runs/alert/" lang="ja" >}}) を作成して、Python コードから Slack や email にアラートを送信します。

コードから Slack やメールアラートを送信したい場合に従うべき最初の 2 つのステップ：

1) W&B の[ユーザー設定](https://wandb.ai/settings)でアラートをオンにする
2) コードに `wandb.alert()` を追加する。例：

```python
wandb.alert(title="Low accuracy", text=f"Accuracy is below the acceptable threshold")
```

以下のセルは `wandb.alert` の使用方法の最小限の例を示しています

```python
# wandb run を開始
wandb.init(project="pytorch-intro")

# モデルのトレーニングループをシミュレート
acc_threshold = 0.3
for training_step in range(1000):

    # 精度のランダムな数値を生成
    accuracy = round(random.random() + random.random(), 3)
    print(f"Accuracy is: {accuracy}, {acc_threshold}")

    # 精度を wandb にログ
    wandb.log({"Accuracy": accuracy})

    # 精度が閾値を下回った場合、W&B アラートを発行し、run を停止する
    if accuracy <= acc_threshold:
        # wandb アラートを送信
        wandb.alert(
            title="Low Accuracy",
            text=f"Accuracy {accuracy} at step {training_step} is below the acceptable theshold, {acc_threshold}",
        )
        print("Alert triggered")
        break

# run を完了としてマーク (Jupyter notebooks で便利)
wandb.finish()
```

[W&B Alerts の詳細なドキュメントはこちらです]({{< relref path="/guides/models/track/runs/alert" lang="ja" >}})。

## 次のステップ
次のチュートリアルでは、W&B Sweeps を使用したハイパーパラメータの最適化を学びます：
[PyTorch を使用したハイパーパラメータ Sweeps](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb)