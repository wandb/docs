---
title: 実験をトラッキングする
menu:
  tutorials:
    identifier: experiments
weight: 1
---

{{< cta-button 
    colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_&_Biases.ipynb" 
>}}

[W&B](https://wandb.ai/site) を使って機械学習の実験管理、モデルのチェックポイント保存、チームでのコラボレーションなどを行いましょう。

このノートブックでは、シンプルな PyTorch モデルを使って機械学習の実験を作成・管理していきます。終わる頃には、チームメンバーと共有・カスタマイズできるインタラクティブなプロジェクトダッシュボードが完成します。[例のダッシュボードはこちら](https://wandb.ai/wandb/wandb_example)。

## 前提条件

W&B Python SDK をインストールし、ログインします：


```shell
!pip install wandb -qU
```


```python
# W&B アカウントにログイン
import wandb
import random
import math

# wandb の新しいバックエンド用に一時的に wandb-core を使用します
wandb.require("core")
```


```python
wandb.login()
```

## W&B で機械学習実験をシミュレート・トラッキング

機械学習実験を作成・追跡・可視化します。手順は以下の通りです：

1. [run]({{< relref "/guides/models/track/runs/" >}}) を初期化し、追跡したいハイパーパラメータを渡します。
2. トレーニングループ内で、精度や損失などのメトリクスをログします。



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
  # このブロックは、トレーニングループでメトリクスをログする例です
  epochs = 10
  offset = random.random() / 5
  for epoch in range(2, epochs):
      acc = 1 - 2 ** -epoch - random.random() / epoch - offset
      loss = 2 ** -epoch + random.random() / epoch + offset

      # 2️．スクリプトからメトリクスをW&Bにログする
      run.log({"acc": acc, "loss": loss})    
```


W&B プロジェクトで、あなたの機械学習実験がどのようになったか確認してみましょう。前のセルから表示される URL をコピー＆ペーストしてください。そのURLからは、モデルのパフォーマンスを示すグラフなどが掲載された W&B プロジェクトのダッシュボードに移動します。

以下の画像は、ダッシュボードの一例です：

{{< img src="/images/tutorials/experiments-1.png" alt="W&B experiment tracking dashboard" >}}

W&B を擬似的な機械学習トレーニングループに組み込む方法が分かったので、今度はベーシックな PyTorch ニューラルネットワークを使って本格的な実験を管理してみましょう。以下のコードは、モデルのチェックポイントも W&B にアップロードします。これにより、組織内の他チームとも簡単に共有できます。

## PyTorch を使って機械学習実験をトラッキング

次のコードセルでは、シンプルな MNIST 分類器を定義・トレーニングします。トレーニング中、W&Bから URL が表示されます。そのプロジェクトページのリンクをクリックすることで、あなたの結果がリアルタイムで W&B プロジェクトに流れていく様子を確認できます。

W&B の run では自動的に [メトリクス]({{< relref "/guides/models/track/runs/#workspace-tab" >}})、  
システム情報、  
[ハイパーパラメータ]({{< relref "/guides/models/track/runs/#overview-tab" >}})、  
[ターミナル出力]({{< relref "/guides/models/track/runs/#logs-tab" >}}) などがログされ、  
モデルの入力・出力を含む [インタラクティブなテーブル]({{< relref "/guides/models/tables/" >}}) も確認できます。

### PyTorch DataLoader の準備
次のセルは、機械学習モデルをトレーニングするのに必要な便利関数を定義しています。これらの関数自体はW&B独自のものではなく、ここで詳細には説明しません。 [forward と backward のトレーニングループ](https://pytorch.org/tutorials/beginner/nn_tutorial.html)の定義方法、[PyTorch DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) の使い方、[`torch.nn.Sequential` クラス](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html) の使い方など、PyTorch の公式ドキュメントもご参照ください。


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
    "トレーニング用データローダーを取得"
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
    "検証データセットでモデルの性能を評価し、wandb.Table をログ"
    model.eval()
    val_loss = 0.0
    with torch.inference_mode():
        correct = 0
        for i, (images, labels) in enumerate(valid_dl):
            images, labels = images.to(device), labels.to(device)

            # フォワードパス ➡
            outputs = model(images)
            val_loss += loss_func(outputs, labels) * labels.size(0)

            # 精度を計算・集計
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            # 一部のバッチ画像をダッシュボードにログ（毎回同じ batch_idx）
            if i == batch_idx and log_images:
                log_image_table(images, predicted, labels, outputs.softmax(dim=1))
    return val_loss / len(valid_dl.dataset), correct / len(valid_dl.dataset)
```

### 予測値 vs 正解ラベルを比較するテーブルの作成

このセルは W&B 独自なので、詳しく見ておきましょう。

ここでは `log_image_table` という関数を定義しています。これは、W&B の Table オブジェクトを作成するもので、各画像ごとにモデルがどう予測したかを一覧表示するためのテーブルです。

より詳しく言うと、各行にはモデルに入力した画像、その予測値、実際の値（ラベル）が載ります。 


```python
def log_image_table(images, predicted, labels, probs):
    "wandb.Table に (img, pred, target, scores) を記録"
    # 画像・ラベル・予測値をまとめて Table にログ
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

### モデルを学習し、チェックポイントをアップロードする

以下のコードでモデルを学習し、W&B プロジェクトにチェックポイントを保存します。モデルのチェックポイントは、通常通りトレーニング中のモデル性能評価に活用できます。

W&B なら保存したモデルやチェックポイントをチームや組織の他メンバーと簡単に共有可能です。チーム外の人と共有したい場合など、より詳細は [W&B Registry]({{< relref "/guides/core/registry/" >}}) をご覧ください。


```python
import wandb

config = {
    "epochs": 5,
    "batch_size": 128,
    "lr": 1e-3,
    "dropout": random.uniform(0.01, 0.80),
}

project = "pytorch-intro"

# wandb run を初期化
with wandb.init(project=project, config=config) as run:

    # 必要なら config をコピー
    config = run.config

    # データを取得
    train_dl = get_dataloader(is_train=True, batch_size=config.batch_size)
    valid_dl = get_dataloader(is_train=False, batch_size=2 * config.batch_size)
    n_steps_per_epoch = math.ceil(len(train_dl.dataset) / config.batch_size)

    # シンプルな MLP モデル
    model = get_model(config.dropout)

    # 損失関数・オプティマイザー
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
                run.log(metrics)

            step_ct += 1

        val_loss, accuracy = validate_model(
            model, valid_dl, loss_func, log_images=(epoch == (config.epochs - 1))
        )

        # トレーニング・バリデーションのメトリクスを wandb にログ
        val_metrics = {"val/val_loss": val_loss, "val/val_accuracy": accuracy}
        run.log({**metrics, **val_metrics})

        # モデルチェックポイントを wandb に保存
        torch.save(model, "my_model.pt")
        run.log_model(
            "./my_model.pt",
            "my_mnist_model",
            aliases=[f"epoch-{epoch+1}_dropout-{round(run.config.dropout, 4)}"],
        )

        print(
            f"Epoch: {epoch+1}, Train Loss: {train_loss:.3f}, Valid Loss: {val_loss:3f}, Accuracy: {accuracy:.2f}"
        )

    # テストセットがある場合は、Summary メトリクスとしてログ
    run.summary["test_accuracy"] = 0.8
```

これで、W&B を使った最初のモデルのトレーニングができました。上記リンクのいずれかをクリックして、あなたのメトリクスや保存したモデルチェックポイントが Artifacts タブにアップされているのを W&B アプリで確かめてみましょう。

## （オプション）W&B Alert の設定

[W&B Alerts]({{< relref "/guides/models/track/runs/alert/" >}}) を作成すると、Python コードから Slack やメールなどへ通知を送信できます。

Slack やメールで通知を受けたい場合、最初に以下2ステップを実施してください：

1) W&B の [ユーザー設定](https://wandb.ai/settings) で Alerts を有効化
2) コード内で `run.alert()` を追加、 例：

```python
run.alert(title="Low accuracy", text=f"Accuracy is below the acceptable threshold")
```

簡単な例は以下の通りです：


```python
import wandb

# wandb run 開始
with wandb.init(project="pytorch-intro") as run:

    # モデル学習ループのシミュレーション
    acc_threshold = 0.3
    for training_step in range(1000):

        # 精度にランダムな数値を生成
        accuracy = round(random.random() + random.random(), 3)
        print(f"Accuracy is: {accuracy}, {acc_threshold}")

        # 精度を wandb にログ
        run.log({"Accuracy": accuracy})

        # 精度がしきい値未満なら、W&B Alert を送信し run を停止
        if accuracy <= acc_threshold:
            # wandb Alert を送信
            run.alert(
                title="Low Accuracy",
                text=f"Accuracy {accuracy} at step {training_step} is below the acceptable threshold, {acc_threshold}",
            )
            print("Alert triggered")
            break
```

詳しくは[W&B Alerts概要]({{< relref "/guides/models/track/runs/alert" >}}) をご覧ください。

## 次のステップ
次のチュートリアルでは、W&B Sweeps を使ったハイパーパラメータ最適化について学びます：  
[PyTorch でのハイパーパラメータスイープ](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb)
