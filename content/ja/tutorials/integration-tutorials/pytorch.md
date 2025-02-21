---
title: PyTorch
menu:
  tutorials:
    identifier: ja-tutorials-integration-tutorials-pytorch
    parent: integration-tutorials
weight: 1
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb" >}}

[Weights & Biases](https://wandb.com) を使って機械学習の実験管理、データセット・バージョン管理、プロジェクトの共同作業を行いましょう。

{{< img src="/images/tutorials/huggingface-why.png" alt="" >}}

## このノートブックがカバーする内容

PyTorch コードと Weights & Biases を統合して、実験管理をパイプラインに追加する方法を紹介します。

{{< img src="/images/tutorials/pytorch.png" alt="" >}}

```python
# ライブラリをインポートする
import wandb

# 新しい実験を開始する
wandb.init(project="new-sota-model")

# 構成でハイパーパラメーターの辞書をキャプチャする
wandb.config = {"learning_rate": 0.001, "epochs": 100, "batch_size": 128}

# モデルとデータを設定する
model, dataloader = get_model(), get_data()

# オプション: 勾配を追跡する
wandb.watch(model)

for batch in dataloader:
  metrics = model.training_step()
  # トレーニングループ内でメトリクスをログに記録し、モデルのパフォーマンスを視覚化する
  wandb.log(metrics)

# オプション: 最後にモデルを保存する
model.to_onnx()
wandb.save("model.onnx")
```

[ビデオチュートリアル](http://wandb.me/pytorch-video)に沿って進めてください。

**注**: _Step_ で始まるセクションは、既存のパイプラインに W&B を統合するために必要な全てです。残りはデータを読み込んでモデルを定義するだけです。

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

# 決定論的な振る舞いを保証する
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# デバイスの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# MNIST ミラーリストから遅いミラーを削除する
torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                      if not mirror.startswith("http://yann.lecun.com")]
```

### 0️⃣ Step 0: W&B をインストールする

始めるには、まずライブラリを手に入れる必要があります。
`wandb` は `pip` で簡単にインストールできます。

```python
!pip install wandb onnx -Uq
```

### 1️⃣ Step 1: W&B をインポートしてログインする

データをウェブサービスにログするためには、ログインする必要があります。

これが初めての方は、リンク先で無料アカウントを登録する必要があります。

```
import wandb

wandb.login()
```

## 実験とパイプラインの定義

### `wandb.init` を使ってメタデータとハイパーパラメーターを追跡する

プログラム的には、まず実験を定義します：
どのハイパーパラメーターがあるのか？この run に関連付けられるメタデータは何か？

この情報を `config` という辞書（あるいは類似のオブジェクト）に保存し、
必要に応じてアクセスするのはかなり一般的なワークフローです。

この例では、いくつかのハイパーパラメーターのみを変え、
残りは手動でコーディングしています。
しかし、モデルのどの部分も `config` の一部にすることができます。

また、いくつかのメタデータも含めています：使用しているデータセットは MNIST で、アーキテクチャーは畳み込みアーキテクチャーです。同じプロジェクトで、例えば完全に接続されたアーキテクチャーを CIFAR で扱う場合、これが runs を分ける助けとなります。

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

では、全体のパイプラインを定義しましょう。
モデル・トレーニングにはかなり一般的です：

1. 最初にモデル、関連データ、およびオプティマイザーを `make` し、
2. 次にモデルを `train` し、
3. 最後にトレーニングの成果を `test` します。

以下にこれらの関数を実装します。

```python
def model_pipeline(hyperparameters):

    # wandb に開始を通知する
    with wandb.init(project="pytorch-demo", config=hyperparameters):
      # すべての HP に wandb.config 経由でアクセスし、ログを実行と一致させます。
      config = wandb.config

      # モデル、データ、および最適化問題を作成する
      model, train_loader, test_loader, criterion, optimizer = make(config)
      print(model)

      # モデルをトレーニングする
      train(model, train_loader, criterion, optimizer, config)

      # 最終的なパフォーマンスをテストする
      test(model, test_loader)

    return model
```

ここでの唯一の違いは、すべてが `wandb.init` のコンテキスト内で発生することです。
この関数を呼び出すことで、コードとサーバー間の通信が設定されます。

`wandb.init` に `config` 辞書を渡すことで、
その情報がすぐにログにつけられ、実験に使用するハイパーパラメーターの値を
常に知ることができます。

選択してログした値が常にモデルで使用されることを保証するために、
オブジェクトの `wandb.config` コピーを使用することをお勧めします。
以下の `make` の定義でいくつかの例を確認してください。

> *サイドノート*: コードを別のプロセスで実行することで、
  もしもこちら側に何か問題が起きた場合
  （例えば巨大な海の怪物がデータセンターを襲った場合など）でも
  コードがクラッシュしないように注意を払っています。
  問題が解決したら、例えば怪物が深海に戻ったら、
  `wandb sync` でデータをログすることができます。

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

### データのロードとモデルの定義

次に、データのロード方法とモデルの見た目を指定する必要があります。

この部分は非常に重要ですが、
`wandb` を使うかどうかで変わることはありませんので、
ここでは深入りしません。

```python
def get_data(slice=5, train=True):
    full_dataset = torchvision.datasets.MNIST(root=".",
                                              train=train, 
                                              transform=transforms.ToTensor(),
                                              download=True)
    #  [::slice] と同等
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

でも、`wandb` とは何も変わらないので、
ここでは標準的な ConvNet アーキテクチャーに固執します。

いくつかの実験を試したり、これをいじくり回したりすることを
恐れないでください —
すべての結果は [wandb.ai](https://wandb.ai) でログされます。

```python
# 通常のニューラルネットワークと畳み込みニューラルネットワーク

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

`model_pipeline` の進行に伴い、`train` を指定する時が来ました。

ここで役に立つ `wandb` の2つの関数は、`watch` と `log` です。

## `wandb.watch` で勾配を追跡し、他の全てを `wandb.log` で記録

`wandb.watch` はトレーニングの各 `log_freq` ステップごとに
モデルの勾配とパラメーターを記録します。

トレーニングを始める前に、その関数を呼び出すだけです。

その後のトレーニングコードは同じままです：
エポックとバッチを繰り返し、
forward pass と backward pass を行い、
`オプティマイザー` を適用します。

```python
def train(model, loader, criterion, optimizer, config):
    # モデルが行っていること（勾配、重みなど）を wandb がウォッチするように指示します。
    wandb.watch(model, criterion, log="all", log_freq=10)

    # トレーニングを実行し、wandb でトラックする
    total_batches = len(loader) * config.epochs
    example_ct = 0  # 見た例の数
    batch_ct = 0
    for epoch in tqdm(range(config.epochs)):
        for _, (images, labels) in enumerate(loader):

            loss = train_batch(images, labels, model, optimizer, criterion)
            example_ct +=  len(images)
            batch_ct += 1

            # 25 番目のバッチごとにメトリクスを報告する
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

    # オプティマイザーでステップを適用する
    optimizer.step()

    return loss
```

唯一の違いはログコードです：
以前はメトリクスをターミナルに出力して報告していたところを、
今では同じ情報を `wandb.log` に渡します。

`wandb.log` は文字列をキーとする辞書を期待しています。
これらの文字列は記録されるオブジェクトを識別し、値を構成します。
どのサイクルのトレーニングステップにいるかを
オプションでログすることもできます。

> *サイドノート*: モデルが見た例の数を使用するのが好きです、
  バッチサイズ間での比較が容易になるためです。
  しかし、ステップ数やバッチ数を使用することもできます。
  長時間のトレーニングでは、`epoch` ごとにログすることも理にかなっています。

```python
def train_log(loss, example_ct, epoch):
    # マジックが発生する場所
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")
```

### テストロジックの定義

モデルのトレーニングが終了したら、テストしたいと考えます：
例えば、本番のデータで一部の新しいデータをモデルに対して実行するか、
手作りの例に適用するかです。

## (オプション) `wandb.save` を呼び出す

これはモデルのアーキテクチャーと最終パラメーターをディスクに保存するのに
適したタイミングでもあります。
最大限の互換性を確保するために、モデルを
[Open Neural Network eXchange (ONNX) フォーマット](https://onnx.ai/) で `エクスポート` します。

そのファイル名を `wandb.save` に渡すことで、モデル パラメーターが W&B のサーバーに保存されることが保証されます：
トレーニング実行に対応する `.h5` や `.pb` を紛失することはもうありません。

モデルを保存、バージョン管理し、配布するためのより高度な `wandb` 機能については、[Artifacts ツール](https://www.wandb.com/artifacts)をご覧ください。

```python
def test(model, test_loader):
    model.eval()

    # 一部のテスト例でモデルを実行する
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

    # モデルを交換可能な ONNX フォーマットで保存する
    torch.onnx.export(model, images, "model.onnx")
    wandb.save("model.onnx")
```

### トレーニングを実行して wandb.ai でリアルタイムでメトリクスを確認する

全体のパイプラインを定義し、
W&B コードを数行追加した今、
完全に追跡された実験を実行する準備が整いました。

以下のリンクをいくつか報告します：
ドキュメント、
プロジェクトページは、プロジェクト内のすべての run を整理し、
この run の結果が保存される Run ページです。

Run ページに移動して、次のタブを確認してください：

1. **Charts** — トレーニング中にモデルの勾配、パラメーター値、および損失がログされています
2. **System** — ディスク I/O 使用率、CPU および GPU メトリクス（この温度が爆発するのを見てみましょう 🔥）など、様々なシステムのメトリクスが含まれています
3. **Logs** — トレーニング中に標準出力に送られたもののコピーが含まれています
4. **Files** — トレーニングが完了したら、`model.onnx` をクリックして [Netron モデル ビューアー](https://github.com/lutzroeder/netron) でネットワークを表示できます。

`wandb.init` ブロックが終了すると、
セル出力に結果の概要も表示されます。

```python
# パイプラインでモデルを構築、トレーニング、および分析する
model = model_pipeline(config)
```

### ハイパーパラメーターをテストするための Sweeps

この例ではハイパーパラメーターの 1 セットを確認しました。
でも、多くの ML ワークフローの重要な部分は、
いくつかのパラメーターを対応することです。

Weights & Biases Sweeps を使用してハイパーパラメーターのテストを自動化し、
可能なモデルと最適化戦略のスペースを探索できます。

## [W&B Sweeps を使用した PyTorch でのハイパーパラメーターの最適化](http://wandb.me/sweeps-colab) を確認してください

Weights & Biases を使用したハイパーパラメーター探索の実行は非常に簡単です。たった3つのステップだけ：

1. **探索を定義する**:
   検索するパラメーター、検索戦略、最適化メトリクスなどを指定する辞書または [YAML ファイル]({{< relref path="/guides/models/sweeps/define-sweep-configuration" lang="ja" >}}) を作成します。

2. **探索を初期化する**:
   `sweep_id = wandb.sweep(sweep_config)`

3. **エージェントを実行する**:
   `wandb.agent(sweep_id, function=train)`

これが、ハイパーパラメーター探索を実行するためのすべてです。

{{< img src="/images/tutorials/pytorch-2.png" alt="" >}}

## プロジェクトギャラリー

W&B で追跡および視覚化されたプロジェクトの例を [ギャラリーで](https://app.wandb.ai/gallery) 見てみましょう。

## 高度なセットアップ
1. [環境変数]({{< relref path="/guides/hosting/env-vars/" lang="ja" >}}): API キーを環境変数に設定して、管理されたクラスターでトレーニングを実行できるようにします。
2. [オフラインモード]({{< relref path="/support/run_wandb_offline.md" lang="ja" >}}): `dryrun` モードを使用してオフライントレーニングを行い、結果を後で同期します。
3. [オンプレ]({{< relref path="/guides/hosting/hosting-options/self-managed" lang="ja" >}}): プライベートクラウドやエアギャップされたサーバーに W&B をインストールし、自身のインフラストラクチャーで使用します。アカデミックから企業チームまで、誰でも対応するためのローカルインストールを提供しています。
4. [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}): 軽量のツールを使ってハイパーパラメーター探索を迅速に設定します。