
# PyTorch
[Weights & Biases](https://wandb.com) を使って機械学習の実験管理、データセットのバージョン管理、プロジェクトのコラボレーションを行いましょう。

[**こちらの Colab Notebook でお試しください →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb)

<div><img /></div>

<img src="https://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />

<div><img /></div>

## このノートブックで扱う内容:

Weights & Biases を PyTorch コードに統合して、実験管理をパイプラインに追加する方法を紹介します。

## 結果として得られるインタラクティブな W&B ダッシュボードは次のようになります:
![](https://i.imgur.com/z8TK2Et.png)

## 擬似コードで行うことは次の通りです:
```python
# ライブラリのインポート
import wandb

# 新しい実験を開始
wandb.init(project="new-sota-model")

# ハイパーパラメータの辞書を設定
wandb.config = {"learning_rate": 0.001, "epochs": 100, "batch_size": 128}

# モデルとデータのセットアップ
model, dataloader = get_model(), get_data()

# オプション: 勾配の追跡
wandb.watch(model)

for batch in dataloader:
  metrics = model.training_step()
  # トレーニングループ内でメトリクスをログしてモデルのパフォーマンスを可視化
  wandb.log(metrics)

# オプション: 最後にモデルを保存
model.to_onnx()
wandb.save("model.onnx")
```

## [ビデオチュートリアル](http://wandb.me/pytorch-video) に沿って進めましょう！
**注記**: 「_Step_」から始まるセクションは、既存のパイプラインに W&B を統合するために必要なすべての内容です。それ以外はデータの読み込みとモデルの定義です。

# 🚀 インストール、インポート、ログイン

```python
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm.auto import tqdm

# 確定的な振る舞いを確保
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# デバイスの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# MNISTミラーリストから遅いミラーを削除
torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                      if not mirror.startswith("http://yann.lecun.com")]
```

### 0️⃣ ステップ0: W&Bのインストール

まず、ライブラリを取得する必要があります。
`wandb` は `pip` を使って簡単にインストールできます。

```python
!pip install wandb onnx -Uq
```

### 1️⃣ ステップ1: W&Bのインポートとログイン

データをウェブサービスにログするためには、
ログインする必要があります。

初めて W&B を使用する場合は、
表示されたリンクから無料アカウントにサインアップしてください。

```
import wandb

wandb.login()
```

# 👩‍🔬 実験とパイプラインの定義

## 2️⃣ ステップ2: `wandb.init` でメタデータとハイパーパラメータを追跡

プログラム的には、最初に実験を定義します:
ハイパーパラメータは何ですか? このrunに関連するメタデータは何ですか?

この情報を `config` 辞書 (または類似のオブジェクト) に保存し、
必要に応じてアクセスするワークフローが一般的です。

この例では、いくつかのハイパーパラメータだけを変更可能にし、
残りは手動で設定しています。
しかし、モデるのどの部分でも `config` の一部にすることができます!

また、メタデータも含めています: MNIST データセットと畳み込みアーキテクチャを使用しています。同じプロジェクトで、将来的に例えばCIFARの全結合アーキテクチャーを扱う場合、これがrunを分けるのに役立ちます。

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

次に、モデルトレーニングの一般的なパイプラインを定義します:

1. まずモデルと関連するデータとオプティマイザーを `make` し、
2. モデルを適宜 `train` し、
3. 最後にトレーニングがどうだったかを確認するために `test` します。

これらの関数は以下で実装します。

```python
def model_pipeline(hyperparameters):

    # wandbに始めるように指示
    with wandb.init(project="pytorch-demo", config=hyperparameters):
      # すべてのHPにwandb.config経由でアクセスし、ログが実行に一致するようにする
      config = wandb.config

      # モデル、データ、および最適化問題を作成
      model, train_loader, test_loader, criterion, optimizer = make(config)
      print(model)

      # モデルをトレーニング
      train(model, train_loader, criterion, optimizer, config)

      # 最終パフォーマンスをテスト
      test(model, test_loader)

    return model
```

標準的なパイプラインと異なるのは、
すべて `wandb.init` のコンテキスト内で実行される点です。
この関数を呼び出すことで、コードとサーバー間の通信が確立されます。

`config` 辞書を `wandb.init` に渡すことで、
すべての情報が即座にログされます。
これにより、実験に使用するハイパーパラメータの値を常に把握できます。

選択した値をログした後でモデルがそれらの値を確実に使用するために、
`wandb.config` コピーを使用するのをお勧めします。
以下の `make` の定義を見ていくつかの例を確認してください。

> *副注*: コードを別々のプロセスで実行するようにしているため、
（たとえば、巨大な海の怪物がデータセンターを攻撃しても）
コードがクラッシュしないようにしています。
問題が解決されたら（例: クラーケンが深海に戻る）
`wandb sync` でデータをログできます。

```python
def make(config):
    # データの作成
    train, test = get_data(train=True), get_data(train=False)
    train_loader = make_loader(train, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)

    # モデルの作成
    model = ConvNet(config.kernels, config.classes).to(device)

    # 損失とオプティマイザーの作成
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, test_loader, criterion, optimizer
```

# 📡 データの読み込みとモデルの定義

次に、データのロード方法とモデルの構成を指定する必要があります。

この部分は非常に重要ですが、`wandb` がない場合と特に変わりませんので、
詳しくは説明しません。

```python
def get_data(slice=5, train=True):
    full_dataset = torchvision.datasets.MNIST(root=".",
                                              train=train, 
                                              transform=transforms.ToTensor(),
                                              download=True)
    # [::slice] でスライスするのと同等
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

モデルの定義は通常、一番楽しい部分です！

しかし `wandb` を使うことで何も変わりません。
だから、標準的なConvNetアーキテクチャーを使いましょう。

これをいじったり、実験したりすることを恐れないでください--
すべての結果が [wandb.ai](https://wandb.ai) にログされます！

```python
# 従来型および畳み込みニューラルネットワーク

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

# 👟 トレーニング ロジックの定義

`model_pipeline`の次は、`train` の方法を指定する時間です。

ここで使用する `wandb` の関数は2つです: `watch` と `log`.

### 3️⃣ ステップ3. 勾配は `wandb.watch` で、その他のすべては `wandb.log` で追跡

`wandb.watch` は、トレーニングの各ステップでモデルの勾配とパラメータをログします。

トレーニングを開始する前にこれを呼び出すだけです。

それ以外のトレーニングコードは同じままです:
エポックとバッチを繰り返し、
forward pass と backward pass を実行し、
`optimizer` を適用します。

```python
def train(model, loader, criterion, optimizer, config):
    # モデルが何をするか見守るようにwandbに指示: 勾配、重みなど
    wandb.watch(model, criterion, log="all", log_freq=10)

    # トレーニングの実行とwandbによる追跡
    total_batches = len(loader) * config.epochs
    example_ct = 0  # 見た例の数
    batch_ct = 0
    for epoch in tqdm(range(config.epochs)):
        for _, (images, labels) in enumerate(loader):

            loss = train_batch(images, labels, model, optimizer, criterion)
            example_ct +=  len(images)
            batch_ct += 1

            # 25バッチごとにメトリクスを報告
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

    # オプティマイザーでステップ実行
    optimizer.step()

    return loss
```

ここで違うのはログコードです:
以前はメトリクスをターミナルに表示していたかもしれませんが、
今は同じ情報を `wandb.log` に渡します。

`wandb.log` は、文字列をキーとする辞書を期待します。
これらの文字列は、ログされるオブジェクトを識別し、値となります。
また、トレーニングのどの `step` にいるかもオプションでログできます。

> *副注*: モデルが見た例の数を使用するのが好きです、
これによりバッチサイズ間での比較が簡単になるからです。
ただし、生のステップやバッチカウントを使用することもできます。
長期のトレーニングでは、`epoch` でログするのも理にかなっています。

```python
def train_log(loss, example_ct, epoch):
    # 魔法が起こる場所
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")
```

# 🧪 テスト ロジックの定義

モデルのトレーニングが終了したら、テストを行います:
実際のデータで実行するかもしれませんし、
手作りの「難しい例」に適用するかもしれません。

#### 4️⃣ 任意ステップ4: `wandb.save` を呼び出す

このタイミングでモデルのアーキテクチャーと最終パラメーターを保存するのも良い時期です。
最大の互換性のために、モデルを
[Open Neural Network eXchange (ONNX) フォーマット](https://onnx.ai/)で `export`します。

そのファイル名を `wandb.save` に渡すことで、モデルパラメータが W&B のサーバーに保存されます: どの `.h5` や `.pb` がどのトレーニングrunに対応するのかを見失うことがなくなります！

モデルの保存、バージョン管理、および配布のためのより高度な `wandb` 機能については、[Artifactsツール](https://www.wandb.com/artifacts)をチェックしてください。

```python
def test(model, test_loader):
    model.eval()

    # モデルをテストデータに対して実行
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

    # 交換可能なONNXフォーマットでモデルを保存
    torch.onnx.export(model, images, "model.onnx")
    wandb.save("model.onnx")
```

# 🏃‍♀️ トレーニングを実行し、 wandb.ai でメトリクスをライブで確認!

さあ、パイプライン全体を定義し、
W&Bコードを数行追加したので、
完全にトラッキングされた実験を実行する準備が整いました。

これからいくつかのリンクを報告します:
ドキュメンテーション、
プロジェクトページ、ここには同一プロジェクトのすべてのrunが整理されています。
Runページ、ここにこのrunの結果が保管されます。

Runページに移動し、次のタブをチェックしてください:

1. **Charts**, モデルの勾配、パラメータ値、およびトレーニング中の損失がログされます
2. **System**, ディスクI/O使用率、CPUとGPUメトリクス (🔥温度の上昇に注目) など、さまざまなシステムメトリクスを含みます
3. **Logs**, トレーニング中に標準出力に送り込まれたもののコピーが表示されます
4. **Files**, トレーニングが完了したら、`model.onnx` をクリックして[Netron モデルビューア](https://github.com/lutzroeder/netron) でネットワークを見ることができます。

run が終了すると (つまり、`with wandb.init` ブロックが終了すると)、
セルの出力に結果のまとめも表示されます。

```python
# パイプラインでモデルを構築、トレーニング、解析
model = model_pipeline(config)
```

# 🧹ハイパーパラメータを使用したSweepsのテスト

この例では単一のハイパーパラメータセットのみを見てきました。
しかし、多くのMLワークフローの重要な部分は、複数のハイパーパラメータを反復することです。

Weights & Biases のSweepsを利用して、ハイパーパラメータのテストを自動化し、可能なモデルと最適

# 🤓 Advanced Setup
1. [Environment variables](https://docs.wandb.com/library/environment-variables): APIキーを環境変数に設定して、管理されたクラスターでトレーニングを実行できます。
2. [Offline mode](https://docs.wandb.com/library/technical-faq#can-i-run-wandb-offline): `dryrun`モードを使用してオフライントレーニングを行い、後で結果を同期します。
3. [On-prem](https://docs.wandb.com/self-hosted): W&Bをプライベートクラウドまたはエアギャップサーバーにインストールします。我々は学術関係者から企業チームまで、すべての方に向けてローカルインストールを提供しています。
4. [Sweeps](https://docs.wandb.com/sweeps): 軽量なチューニングツールを使って、ハイパーパラメータ検索を迅速に設定します。