---
title: PyTorch
menu:
  tutorials:
    identifier: ja-tutorials-integration-tutorials-pytorch
    parent: integration-tutorials
weight: 1
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb" >}}

[Weights & Biases](https://wandb.com) を使用して、機械学習の実験管理、データセットのバージョン管理、およびプロジェクトのコラボレーションを行います。

{{< img src="/images/tutorials/huggingface-why.png" alt="" >}}

## このノートブックでカバーする内容

PyTorch のコードに Weights & Biases を統合して、実験管理をパイプラインに追加する方法を示します。

{{< img src="/images/tutorials/pytorch.png" alt="" >}}

```python
# ライブラリをインポート
import wandb

# 新しい実験を開始
wandb.init(project="new-sota-model")

# ハイパーパラメータの辞書を config でキャプチャ
wandb.config = {"learning_rate": 0.001, "epochs": 100, "batch_size": 128}

# モデルとデータを設定
model, dataloader = get_model(), get_data()

# オプション: 勾配を追跡
wandb.watch(model)

for batch in dataloader:
  metrics = model.training_step()
  # トレーニングループ内でメトリクスをログしてモデルの性能を視覚化
  wandb.log(metrics)

# オプション: モデルを最後に保存
model.to_onnx()
wandb.save("model.onnx")
```

[ビデオチュートリアル](http://wandb.me/pytorch-video)に沿って進めましょう。

**注意**: _Step_ で始まるセクションは、既存のパイプラインに W&B を統合するために必要な部分です。それ以外はデータを読み込み、モデルを定義するだけです。

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

# MNIST ミラーのリストから遅いミラーを削除
torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                      if not mirror.startswith("http://yann.lecun.com")]
```

### ステップ 0: W&B をインストール

まずは、このライブラリを入手する必要があります。
`wandb` は `pip` を使用して簡単にインストールできます。

```python
!pip install wandb onnx -Uq
```

### ステップ 1: W&B をインポートしてログイン

データをウェブサービスにログするためには、
ログインが必要です。

W&B を初めて使用する場合は、
表示されるリンクから無料アカウントに登録する必要があります。

```
import wandb

wandb.login()
```

## 実験とパイプラインを定義

### `wandb.init` でメタデータとハイパーパラメータを追跡

プログラム上、最初に行うのは実験を定義することです。
ハイパーパラメータとは何か？この run に関連するメタデータは何か？

この情報を `config` 辞書（または類似オブジェクト）に保存し、
必要に応じてアクセスするのは非常に一般的なワークフローです。

この例では、数少ないハイパーパラメータのみを変動させ、
残りを手動でコーディングします。
しかし、あなたのモデルのどの部分も `config` の一部にすることができます。

また、いくつかのメタデータも含めます：私たちは MNIST データセットと畳み込み
ニューラルネットワークを使用しているので、
同じプロジェクトで、たとえば全結合ニューラルネットワークで CIFAR を扱う場合、
この情報が run を分ける助けになります。

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

さて、全体のパイプラインを定義しましょう。
これは非常に典型的なモデルトレーニングの手順です：

1. まず、モデル、関連するデータ、オプティマイザーを `make` し、
2. それに基づいてモデルを `train` し、最後に
3. トレーニングがどのように行われたかを確認するために `test` します。

これらの関数を以下で実装します。

```python
def model_pipeline(hyperparameters):

    # wandb に開始を伝えます
    with wandb.init(project="pytorch-demo", config=hyperparameters):
      # すべての HP を wandb.config 経由でアクセスし、ログが実行と一致するようにします。
      config = wandb.config

      # モデル、データ、最適化問題を作成します
      model, train_loader, test_loader, criterion, optimizer = make(config)
      print(model)

      # それらを使用してモデルをトレーニングします
      train(model, train_loader, criterion, optimizer, config)

      # 最終的なパフォーマンスをテストします
      test(model, test_loader)

    return model
```

ここでの標準パイプラインとの唯一の違いは、
すべてが `wandb.init` のコンテキスト内で行われる点です。
この関数を呼び出すことで、
コードとサーバーとの間に通信のラインが確立されます。

`config` 辞書を `wandb.init` に渡すことで、
すべての情報が即座にログされるため、
実験に使用するハイパーパラメータの値を常に把握できます。

選んでログした値が常にモデルに使用されることを保証するために、
`wandb.config` のオブジェクトコピーを使用することをお勧めします。
以下で `make` の定義をチェックして、いくつかの例を見てください。

> *サイドノート*: コードを別々のプロセスで実行するようにします、
私たちの側で問題があっても
（たとえば、巨大な海の怪物がデータセンターを攻撃した場合でも）
コードはクラッシュしません。
問題が解決したら、クラーケンが深海に戻る頃に、
`wandb sync` でデータをログできます。

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

### データのロードとモデルを定義

次に、データがどのようにロードされるか、およびモデルの見た目を指定する必要があります。

これは非常に重要な部分ですが、`wandb` なしで通常行われるのと何も違いませんので、
ここで詳しくは説明しません。

```python
def get_data(slice=5, train=True):
    full_dataset = torchvision.datasets.MNIST(root=".",
                                              train=train, 
                                              transform=transforms.ToTensor(),
                                              download=True)
    #  [::slice] でスライスするのと同等 
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

モデルの定義は通常楽しい部分です。

しかし `wandb` でも何も変わらないので、
標準的な ConvNet アーキテクチャーにとどまりましょう。

この部分をいじっていくつかの実験を試みるのを恐れないでください --
あなたの結果はすべて [wandb.ai](https://wandb.ai) にログされます。

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

### トレーニングロジックを定義

`model_pipeline` を進めると、`train` の指定を行う時がきます。

ここでは `wandb` の 2 つの関数、`watch` と `log` が活躍します。

## `wandb.watch` で勾配を追跡し、他のすべてを `wandb.log` で管理

`wandb.watch` はトレーニングの `log_freq` ステップごとに
モデルの勾配とパラメータをログします。

トレーニングを開始する前にこれを呼び出すだけで済みます。

トレーニングコードの残りは前と変わらず、
エポックとバッチを繰り返し、
forward pass と backward pass を実行し、
`optimizer` を適用します。

```python
def train(model, loader, criterion, optimizer, config):
    # モデルを追跡している wandb に情報を伝えます: 勾配、重み、その他。
    wandb.watch(model, criterion, log="all", log_freq=10)

    # トレーニングを実行し wandb で追跡
    total_batches = len(loader) * config.epochs
    example_ct = 0  # これまでに見た例の数
    batch_ct = 0
    for epoch in tqdm(range(config.epochs)):
        for _, (images, labels) in enumerate(loader):

            loss = train_batch(images, labels, model, optimizer, criterion)
            example_ct +=  len(images)
            batch_ct += 1

            # 25 バッチおきにメトリクスを報告
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

唯一の違いは、記録のコードです：
以前ターミナルにメトリクスを報告していたところを、
`wandb.log` に同じ情報を渡します。

`wandb.log` は文字列をキーとする辞書を期待します。
これらの文字列は、ログされるオブジェクトを識別し、値を構成します。
また、オプションでトレーニングのどの `step` にいるかをログできます。

> *サイドノート*: 私はモデルが見た例の数を使用するのが好きです、
これはバッチサイズを超えて比較しやすくなるためですが、
生のステップやバッチカウントを使用することもできます。
長いトレーニングランの場合、`epoch` 単位でログすることも理にかなっているかもしれません。

```python
def train_log(loss, example_ct, epoch):
    # マジックが起こるところ
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")
```

### テストロジックを定義

モデルのトレーニングが完了したら、テストしてみましょう：
本番環境から新たなデータに対して実行したり、
手作業で準備した例に適用したりします。



## (オプション) `wandb.save` を呼び出す

この時点で、モデルのアーキテクチャと最終パラメータをディスクに保存するのも良い時です。
最大の互換性を考慮して、モデルを
[Open Neural Network eXchange (ONNX) フォーマット](https://onnx.ai/) でエクスポートします。

そのファイル名を `wandb.save` に渡すことで、モデルのパラメータが W&B のサーバーに保存されます：
もはやどの `.h5` や `.pb` がどのトレーニングrunに対応しているのかを見失うことはありません。

モデルの保存、バージョン管理、配布のための、高度な `wandb` 機能については、
[Artifacts ツール](https://www.wandb.com/artifacts)をご覧ください。

```python
def test(model, test_loader):
    model.eval()

    # いくつかのテスト例にモデルを実行
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

    # 交換可能な ONNX フォーマットでモデルを保存
    torch.onnx.export(model, images, "model.onnx")
    wandb.save("model.onnx")
```

### トレーニングを実行し、wandb.ai でライブメトリクスを確認

さて、全体のパイプラインを定義し、数行の W&B コードを追加したら、
完全に追跡された実験を実行する準備が整いました。

いくつかのリンクを報告します：
ドキュメンテーション、
プロジェクトページ、これにはプロジェクトのすべての run が整理されています。
この run の結果が保存されるランページ。

ランページに移動して、これらのタブを確認：

1. **Charts**、トレーニング中にモデルの勾配、パラメーター値、損失がログされます
2. **System**、ここにはディスク I/O 応答率、CPU および GPU メトリクス（温度上昇も監視）、その他のさまざまなシステムメトリクスが含まれます
3. **Logs**、トレーニング中に標準出力にプッシュされたもののコピーがあります
4. **Files**、トレーニングが完了すると、`model.onnx` をクリックして [Netron モデルビューア](https://github.com/lutzroeder/netron) でネットワークを表示できます。

`with wandb.init` ブロック終了時にランが終了すると、
結果の要約もセルの出力で印刷されます。

```python
# パイプラインでモデルを構築、トレーニング、分析
model = model_pipeline(config)
```

### ハイパーパラメータをスイープでテスト

この例では、ハイパーパラメータのセットを 1 つしか見ていません。
しかし、ほとんどの ML ワークフローの重要な部分は、
多くのハイパーパラメータを反復することです。

Weights & Biases Sweeps を使用してハイパーパラメータのテストを自動化し、モデルと最適化戦略の空間を探索できます。

## [W&B Sweeps を使用した PyTorch におけるハイパーパラメータ最適化をチェック](http://wandb.me/sweeps-colab)

Weights & Biases を使用したハイパーパラメータ探索です。非常に簡単です。たった3つの簡単なステップがあります：

1. **スイープを定義する**: これは、検索するパラメータ、検索戦略、最適化メトリクスなどを指定する辞書または[YAML ファイル]({{< relref path="/guides/models/sweeps/define-sweep-configuration" lang="ja" >}})を作成することで行います。

2. **スイープを初期化する**: 
`sweep_id = wandb.sweep(sweep_config)`

3. **スイープエージェントを実行する**: 
`wandb.agent(sweep_id, function=train)`

これだけでハイパーパラメータ探索が実行できます。

{{< img src="/images/tutorials/pytorch-2.png" alt="" >}}

## Example Gallery

W&B で追跡および視覚化されたプロジェクトの例を[ギャラリー →](https://app.wandb.ai/gallery)でご覧ください。

## Advanced Setup
1. [環境変数]({{< relref path="/guides/hosting/env-vars/" lang="ja" >}}): 環境変数に API キーを設定して、管理されたクラスターでトレーニングを実行できます。
2. [オフラインモード]({{< relref path="/support/kb-articles/run_wandb_offline.md" lang="ja" >}}): `dryrun` モードを使用してオフラインでトレーニングし、後で結果を同期。
3. [オンプレ]({{< relref path="/guides/hosting/hosting-options/self-managed" lang="ja" >}}): プライベートクラウドまたはエアギャップされたサーバーに W&B をインストールします。学術機関から企業チームまで、すべての人に向けたローカルインストールを提供。
4. [スイープ]({{< relref path="/guides/models/sweeps/" lang="ja" >}}): ハイパーパラメータ探索を迅速に設定できる、チューニングのための軽量ツール。