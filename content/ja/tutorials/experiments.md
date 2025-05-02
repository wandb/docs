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

[W&B](https://wandb.ai/site) を使って機械学習の実験管理、モデルのチェックポイント、チームとの共同作業などを行いましょう。

このノートブックでは、簡単な PyTorch モデルを使用して機械学習の実験を作成し、追跡します。ノートブックの最後には、チームの他のメンバーと共有してカスタマイズ可能なインタラクティブなプロジェクトダッシュボードを持つことになるでしょう。[ここで例のダッシュボードを閲覧](https://wandb.ai/wandb/wandb_example)できます。

## 前提条件

W&B Python SDK をインストールしてログインします:

```shell
!pip install wandb -qU
```

```python
# W&B アカウントにログイン
import wandb
import random
import math

# wandb の新しいバックエンド用に wandb-core を使用
wandb.require("core")
```

```python
wandb.login()
```

## W&B を使用して機械学習の実験をシミュレーションし、追跡する

機械学習の実験を作成、追跡、視覚化します。これを行うには:

1. [W&B run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) を初期化し、追跡したいハイパーパラメーターを渡します。
2. トレーニングループ内で、精度や損失などのメトリクスをログに記録します。

```
import random
import math

# シミュレートされた実験を 5 回実行します
total_runs = 5
for run in range(total_runs):
  # 1️. このスクリプトを追跡するための新しい run を開始します
  wandb.init(
      # この run がログされるプロジェクトを設定
      project="basic-intro",
      # run 名を渡します（そうでなければ sunshine-lollypop-10 のようにランダムに割り当てられます）
      name=f"experiment_{run}",
      # ハイパーパラメーターと run メタデータを追跡
      config={
      "learning_rate": 0.02,
      "architecture": "CNN",
      "dataset": "CIFAR-100",
      "epochs": 10,
      })

  # この簡単なブロックは、メトリクスをログに記録するトレーニングループをシミュレーションします
  epochs = 10
  offset = random.random() / 5
  for epoch in range(2, epochs):
      acc = 1 - 2 ** -epoch - random.random() / epoch - offset
      loss = 2 ** -epoch + random.random() / epoch + offset

      # 2️. スクリプトから W&B にメトリクスをログ
      wandb.log({"acc": acc, "loss": loss})

  # run を終了としてマーク
  wandb.finish()
```

W&B プロジェクトでの機械学習のパフォーマンスを確認します。前のセルから出力される URL リンクをコピーして貼り付けてください。その URL は、グラフを表示するダッシュボードを含む W&B プロジェクトにリダイレクトされます。

以下の画像は、ダッシュボードがどのように見えるかを示しています。

{{< img src="/images/tutorials/experiments-1.png" alt="" >}}

W&B を疑似的な機械学習トレーニングループに統合する方法を理解したので、基本的な PyTorch ニューラルネットワークを使用して機械学習の実験を追跡してみましょう。以下のコードは、他の組織内チームと共有するために W&B にモデルのチェックポイントをアップロードすることもできます。

## Pytorch を使用して機械学習の実験を追跡する

以下のコードセルは、簡単な MNIST クラス分類器を定義しトレーニングします。トレーニング中は、W&B が URL を表示します。プロジェクトページリンクをクリックして、W&B プロジェクトでリアルタイムに結果を確認してください。

W&B run では自動で [メトリクス]({{< relref path="/guides/models/track/runs/#workspace-tab" lang="ja" >}})、[システム情報]({{< relref path="/guides/models/track/runs/#system-tab" lang="ja" >}})、[ハイパーパラメーター]({{< relref path="/guides/models/track/runs/#overview-tab" lang="ja" >}})、[ターミナル出力]({{< relref path="/guides/models/track/runs/#logs-tab" lang="ja" >}}) をログし、モデルの入力と出力を含む [インタラクティブテーブル]({{< relref path="/guides/models/tables/" lang="ja" >}}) が表示されます。

### PyTorch Dataloader をセットアップする
次のセルでは、機械学習モデルをトレーニングするために必要な便利な関数を定義します。これらの関数は W&B に特化したものではないため、ここでは詳しくは説明しません。詳細については、[forward および backward training loop](https://pytorch.org/tutorials/beginner/nn_tutorial.html) の定義方法、トレーニングデータをロードするための [PyTorch DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) の使用方法、および [`torch.nn.Sequential` クラス](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html) を使用して PyTorch モデルを定義する方法について、PyTorch のドキュメントを参照してください。

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
    "簡単なモデル"
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
    "検証データセットでのモデルの性能を計算し wandb.Table をログ"
    model.eval()
    val_loss = 0.0
    with torch.inference_mode():
        correct = 0
        for i, (images, labels) in enumerate(valid_dl):
            images, labels = images.to(device), labels.to(device)

            # フォワードパス ➡
            outputs = model(images)
            val_loss += loss_func(outputs, labels) * labels.size(0)

            # 精度を計算して累積する
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            # ダッシュボードに1バッチの画像を常に同じbatch_idxでログ
            if i == batch_idx and log_images:
                log_image_table(images, predicted, labels, outputs.softmax(dim=1))
    return val_loss / len(valid_dl.dataset), correct / len(valid_dl.dataset)
```

### 予測値と真の値を比較するためのテーブルを作成する

以下のセルは W&B に固有のものなので、一緒に見ていきましょう。

このセルでは `log_image_table`という関数を定義しています。技術的にはオプションですが、この関数は W&B Table オブジェクトを作成します。このテーブルオブジェクトを使用して、各画像に対してモデルがどのように予測したかを示すテーブルを作成します。

具体的には、それぞれの行にはモデルに入力された画像、予測された値、そして実際の値 (ラベル) が含まれます。

```python
def log_image_table(images, predicted, labels, probs):
    "wandb.Table を (img, pred, target, scores) としてログ"
    # 画像、ラベル、および予測を記録するための wandb テーブルを作成
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

以下のコードは、プロジェクトにモデルのチェックポイントを保存します。トレーニング中にモデルのパフォーマンスを評価するために、通常どおりモデルのチェックポイントを使用します。

W&B は、保存したモデルやモデルのチェックポイントをチームや組織の他のメンバーと容易に共有することも可能です。チーム外のメンバーとモデルやモデルのチェックポイントを共有する方法は、[W&B Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) をご覧ください。

```python
# 3 つの異なるドロップアウト率を試して 3 つの実験を開始する
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

    # config をコピー
    config = wandb.config

    # データを取得
    train_dl = get_dataloader(is_train=True, batch_size=config.batch_size)
    valid_dl = get_dataloader(is_train=False, batch_size=2 * config.batch_size)
    n_steps_per_epoch = math.ceil(len(train_dl.dataset) / config.batch_size)

    # 簡単な MLP モデル
    model = get_model(config.dropout)

    # 損失とオプティマイザーを作成
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
                # トレーニングメトリクスを wandb にログ
                wandb.log(metrics)

            step_ct += 1

        val_loss, accuracy = validate_model(
            model, valid_dl, loss_func, log_images=(epoch == (config.epochs - 1))
        )

        # トレーニングと検証のメトリクスを wandb にログ
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

    # テストセットがあった場合、以下のようにして Summary メトリクスとしてログすることができます
    wandb.summary["test_accuracy"] = 0.8

    # wandb run を閉じる
    wandb.finish()
```

これで W&B を使用して最初のモデルをトレーニングしました。上記のリンクをクリックしてメトリクスを確認し、W&B App UI の Artifacts タブで保存したモデルのチェックポイントを確認してください。

## (オプション) W&B アラートを設定する

[W&B アラート]({{< relref path="/guides/models/track/runs/alert/" lang="ja" >}}) を作成して、Python コードからあなたの Slack やメールにアラートを送信します。

コードから発生する 1 回目の Slack またはメールのアラートに対して実施する 2 つの手順は次のとおりです:

1) W&B の[ユーザー設定](https://wandb.ai/settings)でアラートをオンにする
2) `wandb.alert()` をコードに追加します。例:

```python
wandb.alert(title="低精度", text=f"精度が許容範囲を下回りました")
```

以下のセルでは、`wandb.alert` の使い方を見るための最小限の例を示しています。

```python
# wandb run を開始
wandb.init(project="pytorch-intro")

# モデルトレーニングループをシミュレーション
acc_threshold = 0.3
for training_step in range(1000):

    # 精度のランダムな数値を生成
    accuracy = round(random.random() + random.random(), 3)
    print(f"Accuracy is: {accuracy}, {acc_threshold}")

    # 精度を wandb にログ
    wandb.log({"精度": accuracy})

    # 精度がしきい値を下回った場合、W&B アラートを発動し run を停止する
    if accuracy <= acc_threshold:
        # wandb アラートを送信
        wandb.alert(
            title="低精度",
            text=f"精度 {accuracy} はステップ {training_step} で許容しきい値 {acc_threshold} を下回っています",
        )
        print("アラートが発動されました")
        break

# run を終了としてマーク（Jupyter ノートブックで便利）
wandb.finish()
```

[W&B アラートの完全なドキュメントはこちらで見つけることができます]({{< relref path="/guides/models/track/runs/alert" lang="ja" >}})。

## 次のステップ
次のチュートリアルでは、W&B Sweeps を使用したハイパーパラメーター最適化について学びます:
[PyTorchを使ったハイパーパラメータースイープ](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb)