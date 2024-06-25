
# PyTorch
[Weights & Biases](https://wandb.com) を使って機械学習の実験管理、データセットのバージョン管理、プロジェクトのコラボレーションを行いましょう。

[**こちらのColabノートブックで試してみてください →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb)

<div><img /></div>

<img src="https://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />

<div><img /></div>

## このノートブックでカバーする内容:

PyTorchコードにWeights & Biasesを統合して、実験管理をパイプラインに追加する方法を紹介します。

## 結果として得られるインタラクティブなW&Bダッシュボードは次のようになります:
![](https://i.imgur.com/z8TK2Et.png)

## 擬似コードでは、以下のことを行います:
```python
# ライブラリのインポート
import wandb

# 新しい実験の開始
wandb.init(project="new-sota-model")

# ハイパーパラメーターの辞書を設定
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

## [ビデオチュートリアル](http://wandb.me/pytorch-video) と一緒に進めましょう！
**注**: _Step_ から始まるセクションは、既存のパイプラインにW&Bを統合するために必要な部分です。残りはデータの読み込みとモデルの定義です。

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

# 決定論的な振る舞いを保証
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# デバイスの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# MNISTのミラーリストから遅いミラーを削除
torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                      if not mirror.startswith("http://yann.lecun.com")]
```

### 0️⃣ ステップ 0: W&Bのインストール

始めるには、ライブラリを入手する必要があります。
`wandb` は `pip` を使って簡単にインストールできます。

```python
!pip install wandb onnx -Uq
```

### 1️⃣ ステップ 1: W&Bのインポートとログイン

データをウェブサービスにログするためには、ログインする必要があります。

初めてW&Bを使用する場合は、表示されるリンクで無料アカウントにサインアップしてください。

```
import wandb

wandb.login()
```

# 👩‍🔬 実験とパイプラインの定義

## 2️⃣ ステップ 2: `wandb.init` を使ってメタデータとハイパーパラメーターを記録

プログラム的には、最初に実験を定義します: ハイパーパラメーターは何か? このrunに関連するメタデータは何か?

多くの場合、この情報を `config` 辞書 (または類似のオブジェクト) に格納し、必要に応じてアクセスするのが一般的です。

この例では、いくつかのハイパーパラメーターだけを変動させ、残りは手動で設定しています。
しかし、モデルの任意の部分が `config` の一部になる可能性があります。

また、メタデータとして、MNISTデータセットと畳み込みアーキテクチャーを使用していることも含めています。後で同じプロジェクトで全結合アーキテクチャーを使用する場合、これがrunを分ける手助けとなります。

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

さて、モデルトレーニングのための典型的なパイプラインを定義します:

1. まず、モデルと関連データとオプティマイザーを `make` し、
2. 次にモデルを `train` し、
3. 最後にトレーニングがどれだけうまくいったかを確認するために `test` します。

これらの関数は以下で実装します。

```python
def model_pipeline(hyperparameters):

    # wandbを使って開始を通知
    with wandb.init(project="pytorch-demo", config=hyperparameters):
      # ログが実行と一致するように、すべてのHPにwandb.configを通じてアクセスします！
      config = wandb.config

      # モデル、データ、および最適化の問題を構築
      model, train_loader, test_loader, criterion, optimizer = make(config)
      print(model)

      # それらを使ってモデルをトレーニング
      train(model, train_loader, criterion, optimizer, config)

      # 最終パフォーマンスをテスト
      test(model, test_loader)

    return model
```

ここで標準のパイプラインとの唯一の違いは、すべてが `wandb.init` のコンテキスト内で行われることです。
この関数を呼び出すことで、コードとサーバー間の通信が確立されます。

`wandb.init` に `config` 辞書を渡すことで、すぐにこの情報がログされます。そのため、常に使用するハイパーパラメーター値がわかります。

設定した値とログされた値がモデルで常に使用されるように、`wandb.config` コピーを使用することをお勧めします。
以下の `make` の定義を確認していくつかの例を見てください。

> *サイドノート*: コードが独立したプロセスで実行されるようにしていますので、私たちの側で問題が発生しても（例: 巨大なシーモンスターがデータセンターを攻撃した場合）、コードがクラッシュすることはありません。
問題が解決したら（例: クラーケンが深海に戻った場合）`wandb sync` でデータをログできます。

```python
def make(config):
    # データの作成
    train, test = get_data(train=True), get_data(train=False)
    train_loader = make_loader(train, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)

    # モデルの作成
    model = ConvNet(config.kernels, config.classes).to(device)

    # 損失関数とオプティマイザーの作成
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, test_loader, criterion, optimizer
```

# 📡 データ読み込みとモデルの定義

次に、データの読み込みとモデルの構造を指定する必要があります。

これは非常に重要な部分ですが、
`wandb` を使用しないときと同じですので、詳細には触れません。

```python
def get_data(slice=5, train=True):
    full_dataset = torchvision.datasets.MNIST(root=".",
                                              train=train, 
                                              transform=transforms.ToTensor(),
                                              download=True)
    # [::slice]でスライスするのと同等
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

モデルの定義は通常、楽しい部分です！

しかし、`wandb` では何も変わらないので、標準的なConvNetアーキテクチャーを使います。

これをいじっていくつかの実験を試してみることを恐れないでください --
すべての結果は[wandb.ai](https://wandb.ai)にログされます！

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

# 👟 トレーニングロジックの定義

`model_pipeline` の次はトレーニングの指定です。

ここでは2つの `wandb` 関数が使用されます: `watch` と `log`。

### 3️⃣ ステップ 3. `wandb.watch` を使って勾配を、その他のすべてを `wandb.log` で追跡

`wandb.watch` はトレーニングの各 `log_freq` ステップでモデルの勾配とパラメーターをログします。

トレーニングを開始する前にこれを呼び出すだけです。

残りのトレーニングコードは同じままです:
エポックとバッチを繰り返し、
forward pass と backward pass を実行し、
`optimizer` を適用します。

```python
def train(model, loader, criterion, optimizer, config):
    # モデルの活動を監視するようにwandbに指示: 勾配、重みなど！
    wandb.watch(model, criterion, log="all", log_freq=10)

    # トレーニングを実行し、wandbで追跡
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
    
    # フォワードパス ➡
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # バックワードパス ⬅
    optimizer.zero_grad()
    loss.backward()

    # オプティマイザーでステップ
    optimizer.step()

    return loss
```

唯一違う点はログコードです:
従来ターミナルにメトリクスを報告していたところ、
今は同じ情報を `wandb.log` に渡します。

`wandb.log` は、キーとして文字列を持つ辞書を期待しています。
これらの文字列はロギングされるオブジェクトを識別し、それが値となります。
トレーニングのどの `step` にいるかもオプションでログできます。

> *サイドノート*: 私はモデルが見た例の数を使うのが好きです、これはバッチサイズを超えて比較しやすくなるからです。
でも、ステップ数やバッチカウントを使うこともできます。長いトレーニングでは `epoch`ごとにログすることも意味があるかもしれません。

```python
def train_log(loss, example_ct, epoch):
    # マジックが起こるところ
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")
```

# 🧪 テストロジックの定義

モデルのトレーニングが終わったら、テストを実行したいです:
プロダクションからの新しいデータに対して実行するかもしれませんし、
手作業でキュレーションした「難しい例」に対して適用するかもしれません。

#### 4️⃣ オプション ステップ 4: `wandb.save` を呼び出す

これはまた、モデルのアーキテクチャーと最終パラメーターをディスクに保存する絶好の機会です。
最大限の互換性を持たせるために、
[Open Neural Network eXchange (ONNX) フォーマット](https://onnx.ai/) でモデルを `export` します。

そのファイル名を `wandb.save` に渡すと、モデルのパラメーターがW&Bのサーバーに保存されます: どの `.h5` や `.pb` がどのトレーニングrunに対応するか見失うことはありません！

W&Bのモデルの保存、バージョン管理、配布のための高度な機能については、[Artifactsツール](https://www.wandb.com/artifacts) をご覧ください。

```python
def test(model, test_loader):
    model.eval()

    # テスト例に対してモデルを実行
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

# 🏃‍♀️ トレーニングを実行して、wandb.aiでメトリクスをライブで見ましょう！

これでパイプライン全体を定義し、W&Bコードの数行を挿入したので、
完全に追跡された実験を実行する準備が整いました。

いくつかのリンクを提供します:
ドキュメント,
プロジェクトページは、プロジェクト内のすべてのrunを整理し、
runページには、このrunの結果が保存されます。

runページに移動して、次のタブを確認してください:

1. **Charts**: トレーニング中にモデルの勾配、パラメーター値、および損失がログされます。
2. **System**: ディスクI/O使用率、CPUとGPUの各メトリクス（温度が急上昇🔥）など、さまざまなシステムメトリクスが含まれます。
3. **Logs**: トレーニング中に標準出力に送信されたもののコピーがあります。
4. **Files**: トレーニングが完了したら、`model.onnx` をクリックして[Netronモデルビューアー](https://github.com/lutzroeder/netron)でネットワークを表示できます。

runが終了したら
(つまり `wandb.init` ブロックが終了したら)、
セル出力に結果の概要を印刷します。

```
# パイプラインでモデルを構築、トレーニング、分析
model = model_pipeline(config)
```

# 🧹 ハイパーパラメーターをSweepsでテスト

この例では、単一のハイパーパラメーターセットしか見ていません。
しかし、ほとんどのMLワークフローの重要な部分は、
多数のハイパーパラメーターを繰り返します。

Weights & Biases Swe