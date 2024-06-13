


# PyTorch
[Weights & Biases](https://wandb.com) を使用して機械学習の実験トラッキング、データセットのバージョン管理、およびプロジェクトのコラボレーションを行います。

[**Colabノートブックで試す →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb)

<div><img /></div>

<img src="https://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />

<div><img /></div>

## このノートブックでカバーすること:

PyTorchコードとWeights & Biasesを統合してパイプラインに実験トラッキングを追加する方法を示します。

## 結果のインタラクティブなW&Bダッシュボードは以下のようになります:
![](https://i.imgur.com/z8TK2Et.png)

## 擬似コードで行うこと:
```python
# ライブラリをインポート
import wandb

# 新しい実験を開始
wandb.init(project="new-sota-model")

# ハイパーパラメーターの辞書を設定
wandb.config = {"learning_rate": 0.001, "epochs": 100, "batch_size": 128}

# モデルとデータを設定
model, dataloader = get_model(), get_data()

# オプション: 勾配をトラッキング
wandb.watch(model)

for batch in dataloader:
  metrics = model.training_step()
  # トレーニングループ内でメトリクスをログしてモデルのパフォーマンスを視覚化
  wandb.log(metrics)

# オプション: 最後にモデルを保存
model.to_onnx()
wandb.save("model.onnx")
```

## [ビデオチュートリアル](http://wandb.me/pytorch-video)を見ながら進めましょう！
**注**: _Step_ で始まるセクションは、既存のパイプラインにW&Bを統合するために必要なすべてです。それ以外の部分はデータを読み込み、モデルを定義するだけです。

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

# MNISTミラーのリストから遅いミラーを削除
torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                      if not mirror.startswith("http://yann.lecun.com")]
```

### 0️⃣ Step 0: W&Bをインストール

始めるためには、このライブラリを取得する必要があります。
`wandb` は `pip` を使用して簡単にインストールできます。

```python
!pip install wandb onnx -Uq
```

### 1️⃣ Step 1: W&Bをインポートしてログイン

データをウェブサービスにログするために、ログインが必要です。

W&Bを初めて使用する場合は、表示されるリンクで無料アカウントにサインアップする必要があります。

```
import wandb

wandb.login()
```

# 👩‍🔬 実験とパイプラインの定義

## 2️⃣ Step 2: `wandb.init` を使用してメタデータとハイパーパラメーターをトラッキング

プログラム的には、最初に実験を定義します:
ハイパーパラメーターは何か？このrunに関連付けられるメタデータは何か？

この情報を `config` 辞書 (または類似のオブジェクト) に格納し、必要に応じてアクティブにすることは非常に一般的なワークフローです。

この例では、いくつかのハイパーパラメーターのみを変化させ、残りは手動でコーディングしています。
しかし、モデルの任意の部分を `config` に含めることができます！

メタデータもいくつか含めています: MNISTデータセットと畳み込みアーキテクチャを使用しています。
後に同じプロジェクトで完全に接続されたアーキテクチャを使用する場合、これがrunを分離するのに役立ちます。

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

では、全体のパイプラインを定義します。
これはモデルトレーニングのための非常に一般的なものです:

1. まずモデルと関連データおよびオプティマイザーを `make` し、
2. モデルをトレーニングし、
3. 最後にそれをテストします。

これらの関数は以下で実装します。

```python
def model_pipeline(hyperparameters):

    # wandbに開始を指示
    with wandb.init(project="pytorch-demo", config=hyperparameters):
      # wandb.config 経由ですべてのHPにアクセスし、ログと実行が一致するようにする
      config = wandb.config

      # モデル、データ、および最適化問題を作成
      model, train_loader, test_loader, criterion, optimizer = make(config)
      print(model)

      # トレーニングのために使用
      train(model, train_loader, criterion, optimizer, config)

      # 最終的なパフォーマンスをテスト
      test(model, test_loader)

    return model
```

ここで標準パイプラインと異なるのは、
すべてが `wandb.init` の文脈内で発生することです。
この関数を呼び出すことで、コードとサーバーの間の通信ラインを設定します。

`wandb.init` に `config` 辞書を渡すことで、
実験で使用するハイパーパラメーター値が常に記録されるため、
設定した値を常に把握できるようになります。

選択してログした値が常にモデルで使用されることを保証するために、
`wandb.config` のコピーを使用することをお勧めします。
以下の `make` の定義を確認して、いくつかの例を見てみましょう。

> *サイドノート*: コードを別々のプロセスで実行するように注意を払っています。
そのため、私たち側の問題（例：巨大な海の怪物がデータセンターを攻撃）は、
あなたのコードをクラッシュさせません。
問題が解決したら（例：クラーケンが深海に戻る）、
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

次に、データの読み込み方法とモデルの形状を指定する必要があります。

この部分は非常に重要ですが、
`wandb` がなくても同じ方法で行うため、ここでは詳しく説明しません。

```python
def get_data(slice=5, train=True):
    full_dataset = torchvision.datasets.MNIST(root=".",
                                              train=train, 
                                              transform=transforms.ToTensor(),
                                              download=True)
    # [::slice] と同等にスライス
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

モデルを定義することは通常楽しい部分です！

しかし、`wandb` を使用しても何も変わりません。
したがって、標準的なConvNetアーキテクチャに固執します。

これに関して実験を試したりするのを恐れないでください --
すべての結果は[wandb.ai](https://wandb.ai) にログされます！

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

`model_pipeline` を進める中で、次に `train` の方法を指定します。

ここで `wandb` の2つの関数が重要です: `watch` と `log` です。

### 3️⃣ Step 3. `wandb.watch` で勾配をトラッキングし、その他すべてを `wandb.log` でトラッキング

`wandb.watch` は、トレーニングの各 `log_freq` ステップごとにモデルの勾配とパラメーターをログします。

トレーニングを開始する前にこれを呼び出すだけです。

残りのトレーニングコードは同じです:
エポックとバッチを繰り返し、
forwardおよびbackwardパスを実行し、
オプティマイザーを適用します。

```python
def train(model, loader, criterion, optimizer, config):
    # モデルに何が起こるかを監視するようwandbに指示: 勾配、重みなど
    wandb.watch(model, criterion, log="all", log_freq=10)

    # トレーニングを実行し、wandbでトラッキング
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

    # オプティマイザーでステップ
    optimizer.step()

    return loss
```

唯一の違いはログコードです:
以前は端末にメトリクスを報告していましたが、
今は同じ情報を `wandb.log` に渡します。

`wandb.log` は文字列をキーとする辞書を期待します。
これらの文字列はログされるオブジェクトを識別し、
値として渡されます。
トレーニングのどのステップにいるかをオプションでログすることもできます。

> *サイドノート*: 私はモデルが見た例の数を使用するのが好きです。
これはバッチサイズ全体で比較を容易にするためです。
しかし、ステップ数やバッチ数の生の値を使用することもできます。
より長いトレーニングランでは、`epoch` でログするのが理にかなっていることもあります。

```python
def train_log(loss, example_ct, epoch):
    # マジックが起こる場所
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")
```

# 🧪 テストロジックの定義

モデルのトレーニングが完了したら、テストしたい：
生産から新しいデータに対して実行するかもしれませんし、
手でキュレーションされた「難しい例」に適用するかもしれません。

#### 4️⃣ オプションのStep 4: `wandb.save` を呼び出す

これはまた、モデルのアーキテクチャを保存し、
最終的なパラメーターをディスクに保存する絶好のタイミングです。
最大の互換性を確保するために、
モデルを[Open Neural Network eXchange (ONNX) フォーマット](https://onnx.ai/)でエクスポートします。

そのファイル名を `wandb.save` に渡すことで、モデルパラメーターがW&Bのサーバーに保存されます:
どの `.h5` や `.pb` がどのトレーニングランに対応しているのか見失うことはありません！

より高度な`wandb`の機能については、モデルの保存、バージョン管理、および配布のための
[Artifacts ツール](https://www.wandb.com/artifacts)をご覧ください。

```python
def test(model, test_loader):
    model.eval()

    # いくつかのテスト例でモデルを実行
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

# 🏃‍♀️ トレーニングを実行し、wandb.aiでメトリクスをライブで確認！

パイプライン全体を定義し、
わずかなW&Bのコードを追加したので、
完全にトラッキングされた実験を実行する準備ができました。

以下のリンクをいくつか報告します:
ドキュメント、
プロジェクトページ（同一プロジェクト内のすべてのrunを整理）、
このrunの結果が保存されるRunページ。

Runページに移動して、次のタブを確認しましょう:

1. **チャート**: トレーニング中にモデルのグラデーション、パラメーター値、損失がログされます
2. **システム**: ディスクI/O利用率、CPUおよびGPUのメトリクス（温度の上昇を見守りましょう 🔥）など、さまざまなシステムメトリクスが含まれます
3. **ログ**: トレーニング中に標準出力にプッシュされたすべてのコピーが含まれます
4. **ファイル**: トレーニングが完了したら、`model.onnx` をクリックして[Netronモデルビューア](https://github.com/lutzroeder/netron)でネットワークを表示できます。

runが終了すると
（つまり、`wandb.init` ブロックが終了されると）、
セル出力に結果の概要も表示します。

```
# パイプラインでモデルを構築、トレーニング、および分析
model = model_pipeline(config)
```

# 🧹 ハイパーパラメーターのテストをSweepsで行う

この例では単一のハイパーパラメーターセットのみを検討しました。
しかし、ほとんどの