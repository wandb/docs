
# PyTorch
[Weights & Biases](https://wandb.com)を使用して、機械学習の実験トラッキング、データセットのバージョン管理、プロジェクトのコラボレーションを行いましょう。

[**こちらのColab Notebookでお試しください →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb)

<div><img /></div>

<img src="https://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />

<div><img /></div>

## このノートブックで扱う内容:

PyTorchコードにWeights & Biasesを統合して、実験トラッキングを開発フローに追加する方法を紹介します。

## 生成されるインタラクティブなW&Bダッシュボードの例:
![](https://i.imgur.com/z8TK2Et.png)

## 擬似コードで行うこと:
```python
# ライブラリをインポート
import wandb

# 新しい実験を開始
wandb.init(project="new-sota-model")

# ハイパーパラメーターの辞書をconfigで取得
wandb.config = {"learning_rate": 0.001, "epochs": 100, "batch_size": 128}

# モデルとデータを設定
model, dataloader = get_model(), get_data()

# オプション：勾配をトラック
wandb.watch(model)

for batch in dataloader:
  metrics = model.training_step()
  # トレーニングループ内でメトリクスをログしてモデルの性能を可視化
  wandb.log(metrics)

# オプション：最後にモデルを保存
model.to_onnx()
wandb.save("model.onnx")
```

## [ビデオチュートリアル](http://wandb.me/pytorch-video)に沿って学ぼう！
**注意**: _Step_で始まるセクションは、既存の開発フローにW&Bを統合するのに必要な部分です。残りはデータのロードとモデルの定義に関するものです。

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

# 確定的な挙動を保証
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

まず最初に、ライブラリを取得します。
`wandb`は`pip`を使って簡単にインストールできます。

```python
!pip install wandb onnx -Uq
```

### 1️⃣ ステップ1: W&Bをインポートしてログイン

データをウェブサービスにログするためには、ログインが必要です。

初めてW&Bを使用する場合は、表示されるリンクから無料アカウントを作成する必要があります。

```
import wandb

wandb.login()
```

# 👩‍🔬 実験と開発フローの定義

## 2️⃣ ステップ2: `wandb.init`でメタデータとハイパーパラメーターをトラッキング

最初に行うのは、実験の定義です。
ハイパーパラメーターは何か？このrunに関連付けられるメタデータは何か？

これらの情報を`config`辞書（または同様のオブジェクト）に保存し、必要に応じてアクセスするのが一般的なワークフローです。

この例では、いくつかのハイパーパラメーターだけを変動させて、他の部分はハードコードしています。
しかし、モデルの任意の部分を`config`に含めることができます！

また、いくつかのメタデータを含めます：MNISTデータセットを使用しており、アーキテクチャは畳み込み（CNN）です。
今後同じプロジェクトで完全連結アーキテクチャなどを使用する場合、runを分けるのに役立ちます。

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

次に、モデルトレーニングの一般的な開発フローを定義します：

1. モデルと関連データ、オプティマイザーをまず作成し、
2. モデルをトレーニングして、
3. 最後にその性能をテストします。

以下でこれらの関数を実装します。

```python
def model_pipeline(hyperparameters):

    # wandbに開始を知らせる
    with wandb.init(project="pytorch-demo", config=hyperparameters):
      # 設定をwandb.configから取得し、ログと実行内容を一致させる
      config = wandb.config

      # モデル、データ、最適化問題を作成
      model, train_loader, test_loader, criterion, optimizer = make(config)
      print(model)

      # モデルをトレーニング
      train(model, train_loader, criterion, optimizer, config)

      # 最終性能をテスト
      test(model, test_loader)

    return model
```

ここでの唯一の違いは、すべてが`wandb.init`のコンテキスト内で行われることです。
この関数を呼び出すと、コードとW&Bサーバー間の通信が確立されます。

`config`辞書を`wandb.init`に渡すと、その情報が即座にログされるので、
実験で使用するハイパーパラメーターの値を常に把握できます。

モデルで使用される値が常に選択されたものとログされたものであることを確認するために、
`wandb.config`のコピーを使用することをお勧めします。
以下の`make`関数の定義をチェックしていくつかの例を見てください。

> *サイドノート*: コードは別のプロセスで実行されるため、当社のサーバーに問題が発生した際（例：巨大な海の怪物がデータセンターを襲った時）でも、コードがクラッシュしません。
問題が解決したら（例：クラーケンが深海に戻ったら）、`wandb sync`でデータをログできます。

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

# 📡 データのロードとモデルの定義

次に、データのロード方法とモデルの構造を定義します。

この部分は非常に重要ですが、
`wandb`を使わない場合と何も変わらないので、多くを説明しません。

```python
def get_data(slice=5, train=True):
    full_dataset = torchvision.datasets.MNIST(root=".",
                                              train=train, 
                                              transform=transforms.ToTensor(),
                                              download=True)
    #  [::slice]でスライスしたのと同じ
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

モデルの定義は通常楽しい部分です！

`wandb`を使っても何も変わらないので、標準のConvNetアーキテクチャにこだわります。

これをいじっていくつかの実験を試みることを恐れないでください——すべての結果が[wandb.ai](https://wandb.ai)にログされるので！

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

# 👟 トレーニングロジックの定義

次に進んで、 `model_pipeline` の中で `train` を定義します。

ここでは `wandb` の2つの関数 `watch` と `log` が役立ちます。

### 3️⃣ ステップ3. `wandb.watch`で勾配をトラッキングし、他のすべてを`wandb.log`で記録

`wandb.watch`はトレーニングの各`log_freq`ステップごとにモデルの勾配とパラメーターをログします。

トレーニングを開始する前にこれを呼び出すだけで十分です。

残りのトレーニングコードは同じです：
エポックとバッチごとに繰り返し、
forwardパスとbackwardパスを実行し、
`オプティマイザー`を適用します。

```python
def train(model, loader, criterion, optimizer, config):
    # モデルの動作を監視するようにwandbに指示：勾配、重みなど
    wandb.watch(model, criterion, log="all", log_freq=10)

    # トレーニングを実行し、wandbでトラック
    total_batches = len(loader) * config.epochs
    example_ct = 0  # 訓練データの数
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
    
    # forwardパス ➡
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # backwardパス ⬅
    optimizer.zero_grad()
    loss.backward()

    # オプティマイザーによるステップ
    optimizer.step()

    return loss
```

違いがあるのはログ処理の部分だけです：
以前はメトリクスを端末に出力していたかもしれませんが、今は同じ情報を`wandb.log`に渡します。

`wandb.log`は辞書を期待します。この辞書には文字列をキーとして持ち、ログされるオブジェクトが値になります。
オプションでトレーニングの`ステップ`もログすることができます。

> *サイドノート*: 私はモデルが見たデータの数を使って比較しやすくしているのですが、生ステップやバッチカウントを使っても構いません。長時間のトレーニングでは、`エポック`でログするのも良いかもしれません。

```python
def train_log(loss, example_ct, epoch):
    # ここで魔法が起きる
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")
```

# 🧪 テストロジックの定義

モデルのトレーニングが終わった後、テストします：
プロダクションからの新しいデータで実行するか、手作業で選定した「難しい例」に適用します。

#### 4️⃣ オプションステップ 4: `wandb.save`を呼び出す

これはまた、モデルのアーキテクチャと最終的なパラメーターをディスクに保存する良い機会です。
最大限の互換性を持たせるため、モデルを[Open Neural Network eXchange (ONNX)フォーマット](https://onnx.ai/)でエクスポートします。

そのファイル名を`wandb.save`に渡すことで、モデルパラメーターがW&Bのサーバーに保存されます：どの`.h5`や `.pb`がどのトレーニングrunに対応するかを見失うことはありません！

モデルの保存、バージョン管理、および配布のためのより高度な`wandb`機能については、[Artifacts tools](https://www.wandb.com/artifacts)をご覧ください。

```python
def test(model, test_loader):
    model.eval()

    # テストデータ上でモデルを実行
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

    # モデルをONNXフォーマットで保存
    torch.onnx.export(model, images, "model.onnx")
    wandb.save("model.onnx")
```

# 🏃‍♀️ トレーニングを実行して、wandb.aiでメトリクスをリアルタイムで確認しよう！

これで開発フロー全体が定義され、数行のW&Bコードが追加されたので、
完全にトラックされた実験を実行する準備が整いました。

以下のリンクをいくつか報告します：
ドキュメント、
プロジェクトページ（プロジェクト内の全てのrunを整理するページ）、
runページ（このrunの結果が保存される場所）。

以下のタブを確認するためにrunページに移動してください：

1. **Charts**: トレーニング中にモデルの勾配、パラメーター値、損失がログされる
2. **System**: ディスクI/O利用率、CPUおよびGPUメトリクス（温度上昇を見張れ🔥）、その他多くのシステムメトリクスを含む
3. **Logs**: トレーニング中に標準出力に送られたもののコピーがここにある
4. **Files**: トレーニング完了後モデル `model.onnx` をクリックして[Netron model viewer](https://github.com/lutzroeder/netron)でネットワークを表示できる

runが完了（つまり`with wandb.init`ブロックが終了）したら、セルの出力に結果の要約も表示します。

```
# 開発フローでモデルを作成、トレーニング、分析する
model = model_pipeline(config)
```

# 🧹 ハイパーパラメーターをスイープでテスト

この例では単一のハイパーパラメーターセットだけを見ました。
しかし、多くの機械学習ワークフローの重要な部分は、ハイパーパラメーターの複数回の繰り返しです。

Weights & Biases Sweepsを使用してハイパーパラメーターのテストを自動化し、可能なモデルと最適化戦略の空間を探索できます。

## [PyTorchでのハイパーパラメーター最適化をW&B Sweepsでチェックしよう →