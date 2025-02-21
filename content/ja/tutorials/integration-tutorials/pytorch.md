---
title: PyTorch
menu:
  tutorials:
    identifier: ja-tutorials-integration-tutorials-pytorch
    parent: integration-tutorials
weight: 1
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb" >}}

機械学習の 実験管理 、データセットの バージョン管理 、プロジェクトのコラボレーションには、[Weights & Biases](https://wandb.com) をご利用ください。

{{< img src="/images/tutorials/huggingface-why.png" alt="" >}}

## このノートブックの内容

このノートブックでは、Weights & Biases を PyTorch コードと統合して、パイプラインに 実験管理 を追加する方法を紹介します。

{{< img src="/images/tutorials/pytorch.png" alt="" >}}

```python
# ライブラリのインポート
import wandb

# 新しい実験を開始
wandb.init(project="new-sota-model")

# config でハイパーパラメータの辞書を取得
wandb.config = {"learning_rate": 0.001, "epochs": 100, "batch_size": 128}

# モデルとデータの設定
model, dataloader = get_model(), get_data()

# オプション: 勾配を追跡
wandb.watch(model)

for batch in dataloader:
  metrics = model.training_step()
  # トレーニングループ内でメトリクスをログに記録して、モデルのパフォーマンスを可視化します。
  wandb.log(metrics)

# オプション: 最後にモデルを保存
model.to_onnx()
wandb.save("model.onnx")
```

[動画チュートリアル](http://wandb.me/pytorch-video) もご覧ください。

**注意**: _Step_ で始まるセクションは、既存のパイプラインに W&B を統合するために必要なものです。残りの部分は、データのロードとモデルの定義を行います。

## インストール、インポート、ログイン

```python
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm.auto import tqdm

# 決定論的な振る舞いを保証
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# デバイスの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# MNIST ミラーのリストから低速なミラーを削除
torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                      if not mirror.startswith("http://yann.lecun.com")]
```

### 0️⃣ Step 0: W&B のインストール

まず、ライブラリを取得する必要があります。
`wandb` は `pip` を使用して簡単にインストールできます。

```python
!pip install wandb onnx -Uq
```

### 1️⃣ Step 1: W&B のインポートとログイン

データを当社のウェブサービスに記録するには、ログインする必要があります。

W&B を初めて使用する場合は、表示されるリンクから無料アカウントにサインアップする必要があります。

```
import wandb

wandb.login()
```

## 実験とパイプラインの定義

### `wandb.init` を使用してメタデータとハイパーパラメータを追跡

プログラムで最初に行うことは、実験を定義することです。
ハイパーパラメータは何ですか？この run に関連付けられているメタデータは何ですか？

この情報を `config` 辞書 （または同様の オブジェクト ）に保存し、必要に応じてアクセスするのは、非常に一般的な ワークフロー です。

この例では、いくつかのハイパーパラメータのみを変化させ、残りを手動でコーディングしています。
ただし、モデルの任意の部分を `config` に含めることができます。

また、いくつかのメタデータも含まれています。MNIST データセットと畳み込み アーキテクチャ を使用しています。後で同じ プロジェクト で CIFAR 上の完全に接続された アーキテクチャ を使用する場合、これは run を分離するのに役立ちます。

```python
config = dict(
    epochs=5,
    classes=10,
    kernels=[16, 32],
    batch_size=128,
    learning_rate=0.005,
    dataset="MNIST",
    architecture="CNN")
```

次に、モデルトレーニングでは非常に一般的な、全体のパイプラインを定義しましょう。

1. まず、モデル、関連するデータ、 オプティマイザー を `make` し、
2. 次に、それに応じてモデルを `train` し、最後に
3. `test` して、トレーニングの成果を確認します。

これらの関数は以下で実装します。

```python
def model_pipeline(hyperparameters):

    # wandb に開始するように指示
    with wandb.init(project="pytorch-demo", config=hyperparameters):
      # wandb.config からすべての HP にアクセスして、ログが実行と一致するようにします。
      config = wandb.config

      # モデル、データ、および最適化問題を作成
      model, train_loader, test_loader, criterion, optimizer = make(config)
      print(model)

      # それらを使用してモデルをトレーニング
      train(model, train_loader, criterion, optimizer, config)

      # 最終的なパフォーマンスをテスト
      test(model, test_loader)

    return model
```

標準的なパイプラインとの唯一の違いは、すべてが `wandb.init` のコンテキスト内で発生することです。
この関数を呼び出すと、コードと当社の サーバー 間の通信回線が確立されます。

`config` 辞書 を `wandb.init` に渡すと、そのすべての情報がすぐにログに記録されるため、実験で使用するように設定したハイパーパラメータ値を常に把握できます。

選択およびログに記録した値が常にモデルで使用される値であることを確認するために、 オブジェクト の `wandb.config` コピーを使用することをお勧めします。
いくつかの例については、以下の `make` の定義を確認してください。

> *注記*: 当社は、コードがクラッシュしないように、（巨大な海の怪物 が データセンター を攻撃した場合など）当社の側の問題を処理するために、別の プロセス でコードを実行するように注意しています。
クラーケン が深海に戻ったときなど、問題が解決されたら、`wandb sync` でデータをログに記録できます。

```python
def make(config):
    # データを作成
    train, test = get_data(train=True), get_data(train=False)
    train_loader = make_loader(train, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)

    # モデルを作成
    model = ConvNet(config.kernels, config.classes).to(device)

    # 損失とオプティマイザーを作成
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, test_loader, criterion, optimizer
```

### データロードとモデルの定義

次に、データのロード方法とモデルの外観を指定する必要があります。

この部分は非常に重要ですが、`wandb` がなくても同じであるため、ここでは詳しく説明しません。

```python
def get_data(slice=5, train=True):
    full_dataset = torchvision.datasets.MNIST(root=".",
                                              train=train, 
                                              transform=transforms.ToTensor(),
                                              download=True)
    #  [::slice] でのスライスと同等
    sub_dataset = torch.utils.data.Subset(
      full_dataset, indices=range(0, len(full_dataset), slice))
    
    return sub_dataset


def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size, 
                                         shuffle=True,
                                         pin_memory=True, num_workers=2)
    return loader
```

通常、モデルの定義は楽しい部分です。

ただし、`wandb` では何も変わらないため、標準の ConvNet アーキテクチャ を使用します。

これを試して実験することを恐れないでください。すべての 結果 は [wandb.ai](https://wandb.ai) に記録されます。

```python
# 従来の畳み込みニューラルネットワーク

class ConvNet(nn.Module):
    def __init__(self, kernels, classes=10):
        super(ConvNet, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, kernels[0], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, kernels[1], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * kernels[-1], classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
```

### トレーニングロジックの定義

`model_pipeline` に進み、`train` の方法を指定します。

ここでは、2 つの `wandb` 関数、`watch` と `log` が関係します。

## `wandb.watch` で勾配を追跡し、`wandb.log` でその他すべてを追跡

`wandb.watch` は、トレーニングの `log_freq` ステップごとに、モデルの勾配と パラメータ をログに記録します。

トレーニングを開始する前に呼び出すだけです。

残りのトレーニングコードは同じままです。
エポック と バッチ を反復処理し、 forward pass と backward pass を実行して、`オプティマイザー` を適用します。

```python
def train(model, loader, criterion, optimizer, config):
    # モデルの動作（勾配、重みなど）を wandb に監視させます。
    wandb.watch(model, criterion, log="all", log_freq=10)

    # トレーニングを実行し、wandb で追跡
    total_batches = len(loader) * config.epochs
    example_ct = 0  # 表示されたサンプル数
    batch_ct = 0
    for epoch in tqdm(range(config.epochs)):
        for _, (images, labels) in enumerate(loader):

            loss = train_batch(images, labels, model, optimizer, criterion)
            example_ct +=  len(images)
            batch_ct += 1

            # 25 番目のバッチごとにメトリクスをレポート
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)


def train_batch(images, labels, model, optimizer, criterion):
    images, labels = images.to(device), labels.to(device)
    
    # Forward pass ➡
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # オプティマイザーでステップ
    optimizer.step()

    return loss
```

唯一の違いはロギングコードにあります。
以前は、 ターミナル に出力することで メトリクス を報告していたかもしれませんが、
現在は同じ情報を `wandb.log` に渡します。

`wandb.log` は、キー として文字列を持つ 辞書 を想定しています。
これらの文字列は、値 を構成するログに記録される オブジェクト を識別します。
オプションで、トレーニングのどの `step` にいるかをログに記録することもできます。

> *注記*: バッチサイズ 間で簡単に比較できるように、モデルが表示した サンプル 数を使用するのが好きですが、 raw ステップ または バッチ カウントを使用できます。トレーニング run が長い場合は、`エポック` でログに記録することもできます。

```python
def train_log(loss, example_ct, epoch):
    # 魔法が起こる場所
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")
```

### テストロジックの定義

モデルのトレーニングが完了したら、それをテストします。
おそらく、本番環境からの新しいデータに対して実行するか、手動でキュレーションされた サンプル に適用します。

## （オプション）`wandb.save` の呼び出し

これは、モデルの アーキテクチャ と最終的な パラメータ を ディスク に保存する絶好の機会でもあります。
最大限の互換性を実現するために、[Open Neural Network eXchange (ONNX) 形式](https://onnx.ai/) でモデルを `export` します。

その ファイル名 を `wandb.save` に渡すと、モデル パラメータ が W&B の サーバー に保存されます。どの `.h5` または `.pb` がどのトレーニング run に対応するかを追跡する必要はもうありません。

モデル の保存、 バージョン管理 、および配布のためのより高度な `wandb` 機能については、[Artifacts ツール](https://www.wandb.com/artifacts) を確認してください。

```python
def test(model, test_loader):
    model.eval()

    # いくつかのテスト サンプル でモデルを実行
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Accuracy of the model on the {total} " +
              f"test images: {correct / total:%}")
        
        wandb.log({"test_accuracy": correct / total})

    # モデルを交換可能な ONNX 形式で保存
    torch.onnx.export(model, images, "model.onnx")
    wandb.save("model.onnx")
```

### wandb.ai でトレーニングを実行し、メトリクス をライブで監視する

パイプライン全体を定義し、
W&B コードのいくつかの行を挿入したので、
完全に追跡された実験を実行する準備ができました。

ドキュメント 、
プロジェクト 内のすべての run を整理する プロジェクト ページ 、および
この run の 結果 が保存される Run ページ へのリンクをいくつか報告します。

Run ページ に移動し、次の タブ を確認してください。

1. **Charts**: トレーニング全体でモデルの勾配、 パラメータ 値、および 損失 がログに記録されます。
2. **System**: ディスク I/O 使用率、 CPU および GPU メトリクス （温度が急上昇するのを見てください🔥）など、さまざまな システム メトリクス が含まれています。
3. **Logs**: トレーニング中に標準出力にプッシュされたもののコピーがあります。
4. **Files**: トレーニングが完了すると、`model.onnx` をクリックして、[Netron モデル ビューアー](https://github.com/lutzroeder/netron) で ネットワーク を表示できます。

`with wandb.init` ブロック が終了すると、run が終了すると、
セル出力に 結果 の概要も出力されます。

```python
# パイプラインを使用してモデルを構築、トレーニング、分析
model = model_pipeline(config)
```

### Sweeps でハイパーパラメータ をテストする

この例では、ハイパーパラメータ の単一のセットのみを調べました。
ただし、ほとんどの ML ワークフロー の重要な部分は、
多数のハイパーパラメータ を反復処理することです。

Weights & Biases Sweeps を使用して、ハイパーパラメータ のテストを自動化し、可能なモデル と 最適化戦略 の 空間 を探索できます。

## [W&B Sweeps を使用した PyTorch でのハイパーパラメータ 最適化を確認してください](http://wandb.me/sweeps-colab)

Weights & Biases でハイパーパラメータ sweep を実行するのは非常に簡単です。3 つの簡単な手順があります。

1. **sweep の定義**: 検索する パラメータ 、検索戦略、最適化 メトリクス などを指定する 辞書 または [YAML ファイル]({{< relref path="/guides/models/sweeps/define-sweep-configuration" lang="ja" >}}) を作成することで、これを行います。

2. **sweep の初期化**:
`sweep_id = wandb.sweep(sweep_config)`

3. **sweep agent の実行**:
`wandb.agent(sweep_id, function=train)`

ハイパーパラメータ sweep の実行はこれで終わりです。

{{< img src="/images/tutorials/pytorch-2.png" alt="" >}}

## サンプル ギャラリー

W&B で追跡および 可視化 された プロジェクト の サンプル については、[ギャラリー →](https://app.wandb.ai/gallery) を参照してください。

## 高度な設定
1. [環境変数]({{< relref path="/guides/hosting/env-vars/" lang="ja" >}}): マネージド クラスター でトレーニングを実行できるように、環境変数に APIキー を設定します。
2. [オフラインモード]({{< relref path="/support/run_wandb_offline.md" lang="ja" >}}): `dryrun` モードを使用してオフラインでトレーニングし、後で 結果 を同期します。
3. [オンプレミス]({{< relref path="/guides/hosting/hosting-options/self-managed" lang="ja" >}}): プライベートクラウド または 独自の インフラストラクチャー 内のエアギャップ サーバー に W&B をインストールします。当社には、学者から エンタープライズ チーム まで、すべての人のためのローカル インストール があります。
4. [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}): チューニング 用の軽量 ツール を使用して、ハイパーパラメータ 検索を迅速に設定します。
