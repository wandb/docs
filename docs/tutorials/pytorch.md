# PyTorch
[Weights & Biases](https://wandb.com) を使用して、機械学習の実験管理、データセットのバージョン管理、プロジェクトのコラボレーションを行いましょう。

[**Colab Notebookで試してみる →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb)

<div><img /></div>

<img src="https://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />

<div><img /></div>

## このノートブックでカバーする内容:

Weights & BiasesをPyTorchのコードに統合して、実験管理機能をパイプラインに追加する方法を紹介します。

## 結果として得られるインタラクティブなW&Bダッシュボードは以下のようになります:
![](https://i.imgur.com/z8TK2Et.png)

## 擬似コードで行うことは以下の通りです:
```python
# ライブラリのインポート
import wandb

# 新しい実験の開始
wandb.init(project="new-sota-model")

# 　 ハイパーパラメータの辞書を設定
wandb.config = {"learning_rate": 0.001, "epochs": 100, "batch_size": 128}

# モデルとデータの設定
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

## [ビデオチュートリアル](http://wandb.me/pytorch-video)と一緒に進めましょう!
**注意**: _Step_ で始まるセクションは、既存のパイプラインにW&Bを統合するために必要なものです。それ以外はデータのロードとモデルの定義のみです。

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

# 確定的な振る舞いを保証
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# デバイスの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 遅いミラーをMNISTミラーのリストから除去
torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                      if not mirror.startswith("http://yann.lecun.com")]
```

### 0️⃣ ステップ 0: W&Bのインストール

始めるには、ライブラリを取得する必要があります。
`wandb` は `pip` を使って簡単にインストールできます。

```python
!pip install wandb onnx -Uq
```

### 1️⃣ ステップ 1: W&Bをインポートしてログイン

データをウェブサービスにログするためには、ログインが必要です。

初めてW&Bを使用する場合は、表示されるリンクから無料アカウントにサインアップしてください。

```
import wandb

wandb.login()
```

# 👩‍🔬 実験とパイプラインの定義

## 2️⃣ ステップ 2: `wandb.init`でメタデータとハイパーパラメータを追跡

最初に行うことは実験の定義です:
ハイパーパラメータは何ですか? このrunに関連するメタデータは何ですか?

これらの情報を `config` 辞書（またはそれに似たオブジェクト）に保存し、必要に応じてアクセスするのが一般的なワークフローです。

この例では、いくつかのハイパーパラメータだけを変更し、残りは手動でコードに埋め込んでいます。
しかし、モデルの任意の部分を `config` に含めることができます!

また、メタデータも含めています: MNISTデータセットと畳み込みアーキテクチャを使用しています。同じプロジェクトで全結合アーキテクチャを使用する場合、これによりrunを区別するのに役立ちます。

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

次に、モデルのトレーニングのための一般的なパイプラインを定義しましょう:

1. まずモデル、関連データ、およびオプティマイザーを `make` し、
2. それに応じてモデルを `train` し、
3. 最後に `test` してトレーニングの成果を確認します。

これらの関数は以下で実装します。

```python
def model_pipeline(hyperparameters):

    # wandbに開始を知らせる
    with wandb.init(project="pytorch-demo", config=hyperparameters):
      # wandb.configを通じてすべてのハイパーパラメータにアクセスし、ログが実行と一致するようにする！
      config = wandb.config

      # モデル、データ、およびオプティマイザーの設定
      model, train_loader, test_loader, criterion, optimizer = make(config)
      print(model)

      # モデルをトレーニングする
      train(model, train_loader, criterion, optimizer, config)

      # モデルの最終パフォーマンスをテストする
      test(model, test_loader)

    return model
```

この標準的なパイプラインとの唯一の違いは、すべてが `wandb.init` のコンテキスト内で行われることです。
この関数を呼び出すことで、コードとサーバー間の通信が設定されます。

`config` 辞書を `wandb.init` に渡すことで、その情報がすぐにログされます。
これにより、実験に使用するハイパーパラメータの値を常に把握できます。

選択しログした値が常にモデルで使用されるようにするため、`wandb.config` のコピーを使用することをお勧めします。
以下の `make` の定義を確認して、いくつかの例を見てみましょう。

> *サイドノート*: コードを別々のプロセスで実行するようにしています。
これにより、サーバー側の問題（例: 巨大な海のモンスターがデータセンターを攻撃する）でコードがクラッシュしません。
問題が解決したら（例: クラーケンが深海に戻る）、`wandb sync` でデータをログできます。

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

# 📡 データ読み込みとモデルの定義

次に、データがどのように読み込まれるか、モデルがどのように見えるかを指定する必要があります。

この部分は非常に重要ですが、`wandb` がない場合と何ら変わりませんので、詳細には触れません。

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

モデルを定義するのは普通楽しい部分です!

しかし `wandb` では何も変わりませんので、
標準的なConvNetアーキテクチャを使用します。

これで実験を試してみてください -- すべての結果が[wandb.ai](https://wandb.ai)にログされます!

```python
# 標準的な畳み込みニューラルネットワーク

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

`model_pipeline` を続けて、次はトレーニング方法を指定します。

ここで使用する2つの`wandb`関数は、`watch` と `log` です。

### 3️⃣ ステップ 3. 勾配を`wandb.watch`で、他のすべてを`wandb.log`で追跡

`wandb.watch`は、モデルの勾配とパラメータをトレーニングの各`log_freq`ステップでログします。

トレーニングを開始する前に呼び出すだけです。

残りのトレーニングコードは前と同じです:
エポックとバッチを反復し、
forward passとbackward passを実行し、
オプティマイザーを適用します。

```python
def train(model, loader, criterion, optimizer, config):
    # モデルで起こること（勾配、重みなど）をwandbで監視する
    wandb.watch(model, criterion, log="all", log_freq=10)

    # トレーニングを実行しwandbで追跡
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

唯一の違いは、ログコードです:
以前はメトリクスを端末に出力していたかもしれませんが、
今は同じ情報を`wandb.log`に渡します。

`wandb.log`は、文字列をキーとして持つ辞書を期待しています。
これらの文字列はログされるオブジェクトを識別し、そのオブジェクトが値を構成します。
また、トレーニングのどの`ステップ`にいるかをオプションでログすることもできます。

> *サイドノート*: モデルが見た例の数を使用するのが好きです、
バッチサイズ全体で比較が容易になるためです。
ただし、生のステップやバッチ数を使用することもできます。長時間のトレーニングの場合、`エポック`ごとにログを取るのも理にかなっています。

```python
def train_log(loss, example_ct, epoch):
    # 魔法が起こるところ
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")
```

# 🧪 テストロジックの定義

モデルのトレーニングが完了したら、テストを行います:
新しいデータで動作させるか、あるいは手作業で選んだ「難しい例」に適用します。

#### 4️⃣ オプション ステップ 4: `wandb.save`の呼び出し

これはモデルのアーキテクチャと最終パラメータをディスクに保存するための偉大な機会です。
最大の互換性のために、モデルを[Open Neural Network eXchange (ONNX)形式](https://onnx.ai/)でエクスポートします。

そのファイル名を`wandb.save`に渡すことで、モデルのパラメータがW&Bのサーバーに保存されます:
どの`.h5`や`.pb`がどのトレーニングrunに対応するかを忘れることはもうありません!

保存、バージョン管理、分配のための高度な`wandb`機能については、
私たちの[Artifactsツール](https://www.wandb.com/artifacts)をチェックしてください。

```python
def test(model, test_loader):
    model.eval()

    # モデルをいくつかのテスト例で実行
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

    # モデルを交換可能なONNX形式で保存
    torch.onnx.export(model, images, "model.onnx")
    wandb.save("model.onnx")
```

# 🏃‍♀️ トレーニングを実行して、wandb.aiでメトリクスをリアルタイムで確認しましょう!

これで全パイプラインが定義され、
W&Bコードを数行追加したので、
完全に追跡された実験の実行に準備が整いました。

いくつかのリンクを報告します:
ドキュメント、
プロジェクトページ（プロジェクト内のすべてのrunを整理）、
runページ（このrunの結果が保存されます）。

runページに移動して、以下のタブを確認してください:

1. **Charts**: モデルの勾配、パラメータ値、損失がトレーニング中にログされます
2. **System**: ディスクI/O使用率、CPUおよびGPUメトリクス（温度上昇🔥）などのさまざまなシステムメトリクスが含まれます
3. **Logs**: トレーニング中に標準出力に送られたもののコピー
4. **Files**: トレーニング完了後、`model.onnx`をクリックして[Netronモデルビューア](https://github.com/lutzroeder/netron)でネットワークを表示できます。

runが終了したら（`with wandb.init`ブロックが終了します）、結果の概要もセル出力に表示されます。

```
# パイプラインでモデルを構築、トレーニング、解析
model = model_pipeline(config)
```

# 🧹 ハイパーパラメータをスイープでテスト

この例では、単一のハイパーパラメータセットについてのみ見ました。
しかし、多くの機械学習ワークフローの重要な部分は、複数のハイパーパラメータを反復することです。

Weights & BiasesのSweepsを使用すると、ハイパーパラメータテストを自動化し、
可能なモデルや最適化戦略の空間を探索できます。

## [W&B Sweepsを使用したPyTorchのハイパーパラメータ最適化をチェック →](http://wandb.me/sweeps-colab)

Weights & Biasesスイープを実行するのは非常に簡単です。以下の3つのステップだけです：

1. **スイープの定義**: 検索するパラメータ、検索戦略、最適化メトリクスなどを指定する辞書や[YAMLファイル](https://docs.wandb.com/library/sweeps/configuration)を作成します。

2. **スイープの初期化**:
`sweep_id = wandb.sweep(sweep_config)`

3. **スイープエージェントの実行**:
`wandb.agent(sweep_id, function=train)`

お見事！これでハイパーパラメータスイープを実行する方法がすべてわかりました！
<img src="https://imgur.com/UiQKg0L.png" alt="Weights & Biases" />

# 🖼️ サンプルギャラリー

W&Bを使って追跡および視覚化されたプロジェクトの例を[ギャラリーで確認 →](https://app.wandb.ai/gallery)

