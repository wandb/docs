---
title: モデルとデータセットをトラッキングする
menu:
  tutorials:
    identifier: ja-tutorials-artifacts
weight: 4
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Pipeline_Versioning_with_W&B_Artifacts.ipynb" >}}
このノートブックでは、W&B Artifacts を使って ML 実験パイプラインをトラッキングする方法を紹介します。

[ビデオチュートリアル](https://tiny.cc/wb-artifacts-video) もぜひご覧ください。

## アーティファクト（Artifacts）について

artifact（アーティファクト）は、ギリシャの[アンフォラ](https://en.wikipedia.org/wiki/Amphora)のように、何かしらプロセスのアウトプットとして生産された「オブジェクト」です。
ML で最も重要なアーティファクトは _dataset_ と _model_ です。

そして[クロス・オブ・コロナド](https://indianajones.fandom.com/wiki/Cross_of_Coronado)のように、これらの重要なアーティファクトはまるで博物館に納めるべき貴重なものです。
つまり、きちんとカタログ化し整理しておくことで、あなた自身はもちろん、チームやMLコミュニティ全体の学びにつながります。
トレーニング結果をトラッキングしない人は、同じことを繰り返してしまうかもしれません。

W&B Artifacts API を使えば、W&B の `Run` の出力として `Artifact` を記録・ログしたり、逆に `Run` の入力として `Artifact` を使うことができます。下図のように、トレーニング run が dataset を入力として受け取り、model を出力しています。

{{< img src="/images/tutorials/artifacts-diagram.png" alt="Artifacts ワークフロー図" >}}

ある run の出力を別の run の入力に使えるため、`Artifact` と `Run` を組み合わせると有向グラフ（[DAG](https://en.wikipedia.org/wiki/Directed_acyclic_graph)）ができます。
ノードは `Artifact` と `Run`、矢印は `Run` が `Artifact` を消費または生成する関係を示します。

## Artifacts でモデルやデータセットをトラッキングする

### インストールとインポート

Artifacts は、Python ライブラリ（バージョン `0.9.2` 以降）の機能のひとつです。

一般的な ML の Python スタック同様、`pip` でインストールできます。

```python
# wandb バージョン 0.9.2+ に対応
!pip install wandb -qqq
!apt install tree
```

```python
import os
import wandb
```

### データセットをログする

まずは、いくつかの Artifacts を定義しましょう。

この例は PyTorch の
["Basic MNIST Example"](https://github.com/pytorch/examples/tree/master/mnist/)
をもとにしていますが、[TensorFlow](https://wandb.me/artifacts-colab) や他のフレームワーク、あるいは純粋な Python でも同じように実施できます。

はじめに `Dataset` を準備します。
- 学習用セット（`train`）：パラメータの選択用
- 検証用セット（`validation`）：ハイパーパラメータの選択用
- テストセット（`test`）：最終モデルの評価用

下のセルでこれら3つのデータセットを定義しています。

```python
import random 

import torch
import torchvision
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm

# 挙動を決定論的にする
torch.backends.cudnn.deterministic = True
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# デバイス設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# データ用パラメータ
num_classes = 10
input_shape = (1, 28, 28)

# 遅いMNISTミラーを削除
torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                      if not mirror.startswith("http://yann.lecun.com")]

def load(train_size=50_000):
    """
    # データをロード
    """

    # 学習データとテストデータをロード
    train = torchvision.datasets.MNIST("./", train=True, download=True)
    test = torchvision.datasets.MNIST("./", train=False, download=True)
    (x_train, y_train), (x_test, y_test) = (train.data, train.targets), (test.data, test.targets)

    # 検証セットの分割（ハイパーパラメータチューニング用）
    x_train, x_val = x_train[:train_size], x_train[train_size:]
    y_train, y_val = y_train[:train_size], y_train[train_size:]

    training_set = TensorDataset(x_train, y_train)
    validation_set = TensorDataset(x_val, y_val)
    test_set = TensorDataset(x_test, y_test)

    datasets = [training_set, validation_set, test_set]

    return datasets
```

この例では、データを Artifacts としてログするコードと、データ自体を生成するコードを分けて実装しています。
この場合、`load` でのデータロードと、`load_and_log` でのログ作成が別れているのが分かります。

これは推奨されるよいパターンです。

これらのデータセットを Artifacts としてログするためには
1. `wandb.init()` で `Run` を作成（L4）
2. データセット用の `Artifact` を作成（L10）
3. 関連する `file` を保存・ログする（L20, L23）

下のセルがその例で、このあとに詳細な解説パートがあります。

```python
def load_and_log():

    # Runを開始。typeでラベル付け、projectも指定
    with wandb.init(project="artifacts-example", job_type="load-data") as run:
        
        datasets = load()  # データセットのロード部分のみ分離
        names = ["training", "validation", "test"]

        # 🏺 Artifact を作成
        raw_data = wandb.Artifact(
            "mnist-raw", type="dataset",
            description="Raw MNIST dataset, split into train/val/test",
            metadata={"source": "torchvision.datasets.MNIST",
                      "sizes": [len(dataset) for dataset in datasets]})

        for name, data in zip(names, datasets):
            # 🐣 Artifactに新しいファイルを登録し、中身を書き込む
            with raw_data.new_file(name + ".pt", mode="wb") as file:
                x, y = data.tensors
                torch.save((x, y), file)

        # ✍️ W&Bに Artifact を保存
        run.log_artifact(raw_data)

load_and_log()
```

#### `wandb.init()`

これから `Artifact` を生成する `Run` を作るときは、
どの `project` に属するのか明示する必要があります。

ワークフローによって、
project の粒度は「car-that-drives-itself」レベルだったり
「iterative-architecture-experiment-117」程度だったりします。

> **ベストプラクティス**: できれば共通のアーティファクトを共有するすべての `Run` は一つの project にまとめておきましょう。この方がシンプルですが、`Artifact` 自体は project をまたいで使えます。

様々な jobs を管理したい場合は
`job_type` を渡すのがおすすめです。
Artifacts のグラフを見やすく整理できます。

> **ベストプラクティス**: `job_type` はパイプラインの一つのステップを明示しましょう。ここでは `load` データと `preprocess` データを区別。

#### `wandb.Artifact`

何かを `Artifact` としてログするには、まず `Artifact` オブジェクトを生成します。

すべての `Artifact` には `name` が必要です（第1引数）。

> **ベストプラクティス**: `name` は説明的かつ簡潔に。ハイフン区切りで、コード内の変数名と関連づけるのもおすすめです。

`type` も必要です。これは `Run` の `job_type` 同様、Artifacts・Runs のグラフ整理に使われます。

> **ベストプラクティス**: `type` はシンプルに。たとえば `mnist-data-YYYYMMDD` より `dataset` や `model` など。

さらに、`description` や `metadata`（辞書型, JSON化可能）も付与できます。

> **ベストプラクティス**: `metadata` はできるだけ詳しく記述するのがおすすめ。

#### `artifact.new_file` と `run.log_artifact`

`Artifact` オブジェクトを作成したら、そこにファイルを追加しましょう。

アーティファクトには _ファイル_（複数形!）を入れられます。
ディレクトリのような構造にし、ファイルやサブディレクトリも作れます。

> **ベストプラクティス**: 可能なら Artifact の中身はファイルに分割しましょう。大規模化した際に管理がしやすくなります。

ファイルの書き込みとArtifactへの紐付けは `new_file` メソッドで同時にできます。
あとで紹介する `add_file` メソッドは書き込みと紐付けを別々にする方法です。

すべてファイルを追加したら `log_artifact` で [wandb.ai](https://wandb.ai) に記録しましょう。

出力にはいくつかの URL が含まれ、その中に Run ページのものも出ます。
そこから `Run` の成果やログされた `Artifact` を確認できます。

Run ページでできる他のことは、以下でまた紹介します。

### ログ済みデータセット Artifact を使う

W&B の `Artifact` は博物館の展示品と違い、「使われる」ためにあります。

そのイメージを具体的に見てみましょう。

下記のセルでは、生のデータセットを入力として受け取り、`preprocess` されたデータセット（正規化や形状の調整済み）を作るステップを用意しています。

ここでもロジック本体（`preprocess`）と wandb 連携部分を分けています。

```python
def preprocess(dataset, normalize=True, expand_dims=True):
    """
    ## データの準備
    """
    x, y = dataset.tensors

    if normalize:
        # 画素値を [0, 1] 範囲にスケーリング
        x = x.type(torch.float32) / 255

    if expand_dims:
        # 画像形状を (1, 28, 28) に変換
        x = torch.unsqueeze(x, 1)
    
    return TensorDataset(x, y)
```

次はこの `preprocess` ステップを `wandb.Artifact` と連携して実行するコードです。

注目すべき点は、例の中で `Artifact` を「使う（use）」ことと「ログする（log）」こと両方が行われている点です。
`Artifact` は `Run` の入力にも出力にもなります。

また、`job_type` を `preprocess-data` とすることで、この処理ステップが他と区別できるようになっています。

```python
def preprocess_and_log(steps):

    with wandb.init(project="artifacts-example", job_type="preprocess-data") as run:

        processed_data = wandb.Artifact(
            "mnist-preprocess", type="dataset",
            description="Preprocessed MNIST dataset",
            metadata=steps)
         
        # ✔️ 使用するartifactを宣言
        raw_data_artifact = run.use_artifact('mnist-raw:latest')

        # 📥 必要に応じてダウンロード
        raw_dataset = raw_data_artifact.download()
        
        for split in ["training", "validation", "test"]:
            raw_split = read(raw_dataset, split)
            processed_dataset = preprocess(raw_split, **steps)

            with processed_data.new_file(split + ".pt", mode="wb") as file:
                x, y = processed_dataset.tensors
                torch.save((x, y), file)

        run.log_artifact(processed_data)


def read(data_dir, split):
    filename = split + ".pt"
    x, y = torch.load(os.path.join(data_dir, filename))

    return TensorDataset(x, y)
```

ここでひとつ注目したいのは、前処理の `steps` を metadata として `preprocessed_data` に保存している点です。

実験を再現可能にしたい場合、豊富な metadata の記録がとても役立ちます。

また、この例では「大きな artifact」ですが、`download` 処理も一瞬で終わることが多いです。

詳細は以下のマークダウンセルを参照してください。

```python
steps = {"normalize": True,
         "expand_dims": True}

preprocess_and_log(steps)
```

#### `run.use_artifact()`

artifact の利用はシンプルです。利用者は `Artifact` の `name` と、もう一つだけ識別情報が必要です。

そのもう一つが「`alias`」です。これは指定したバージョンを選ぶために使います。

デフォルトでは、最後にアップロードされたバージョンに `latest` が付いています。
それ以外に `v0` や `v1`、また独自の `best` や `jit-script` など好きなエイリアスも使えます。
[Docker Hub](https://hub.docker.com/) のタグと同じ感じですね。

エイリアスは名前の後ろに `:` でつなげる形です。
つまり `mnist-raw:latest` という感じで指定します。

> **ベストプラクティス**: エイリアスはシンプルに。`latest` や `best` など、わかりやすい `alias` を活用しましょう。

#### `artifact.download`

`download` を使うのにメモリが心配になるかもしれません。

でも大丈夫。実際にダウンロードする前に、同じバージョンが既にローカルにあるか確認しています。
これは [トレント](https://en.wikipedia.org/wiki/Torrent_file) や [`git` バージョン管理](https://blog.thoughtram.io/git/2014/11/18/the-anatomy-of-a-git-commit.html)と同じく「ハッシュ値」の仕組みを利用しています。

Artifact を作成・ログすると、作業ディレクトリ内の `artifacts` フォルダにサブディレクトリが増えていきます。
各 `Artifact` ごとにディレクトリがあります。
`!tree artifacts` で中身を見てみましょう。

```python
!tree artifacts
```

#### Artifacts ページ

`Artifact` をログし使ったら、Run ページの Artifacts タブを見てみましょう。

wandb の出力から Run ページ URL にアクセスし、
左サイドバーの「Artifacts」タブを選びます（データベースアイコンで、ホッケーパック3枚分を積み重ねたようなマークです）。

**Input Artifacts** または **Output Artifacts** のテーブルで行をクリックし、
上部のタブ（**Overview**, **Metadata** など）を切り替えると、その `Artifact` に記録された全情報が確認できます。

特におすすめなのが **Graph View**。
デフォルトでは
`Artifact` の `type` と `Run` の `job_type` で2種類のノードとなったグラフが描かれ、
矢印が消費・生成関係を示します。

### モデルのログ

ここまでで `Artifact` API の使い方は理解できたと思いますが、
もう少し進めてパイプラインの最後まで見てみましょう。
`Artifact` がどのように ML ワークフローを改善するか体感できます。

まずはシンプルな PyTorch コンボリューションネット（ConvNet）モデルを作ります。

まずは `model` を初期化するだけで、まだトレーニングしません。
これにより、同じ条件で何度でもトレーニングが簡単になります。

```python
from math import floor

import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, hidden_layer_sizes=[32, 64],
                  kernel_sizes=[3],
                  activation="ReLU",
                  pool_sizes=[2],
                  dropout=0.5,
                  num_classes=num_classes,
                  input_shape=input_shape):
      
        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential(
              nn.Conv2d(in_channels=input_shape[0], out_channels=hidden_layer_sizes[0], kernel_size=kernel_sizes[0]),
              getattr(nn, activation)(),
              nn.MaxPool2d(kernel_size=pool_sizes[0])
        )
        self.layer2 = nn.Sequential(
              nn.Conv2d(in_channels=hidden_layer_sizes[0], out_channels=hidden_layer_sizes[-1], kernel_size=kernel_sizes[-1]),
              getattr(nn, activation)(),
              nn.MaxPool2d(kernel_size=pool_sizes[-1])
        )
        self.layer3 = nn.Sequential(
              nn.Flatten(),
              nn.Dropout(dropout)
        )

        fc_input_dims = floor((input_shape[1] - kernel_sizes[0] + 1) / pool_sizes[0]) # layer 1 output size
        fc_input_dims = floor((fc_input_dims - kernel_sizes[-1] + 1) / pool_sizes[-1]) # layer 2 output size
        fc_input_dims = fc_input_dims*fc_input_dims*hidden_layer_sizes[-1] # layer 3 output size

        self.fc = nn.Linear(fc_input_dims, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc(x)
        return x
```

ここでは wandb で run をトラッキングするため、
[`run.config`](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-config/Configs_in_W%26B.ipynb)
を使ってハイパーパラメータをすべて保存しています。

その `config` オブジェクトの `dict` 版は metadata の記録としてとても有用なので、ぜひ含めましょう。

```python
def build_model_and_log(config):
    with wandb.init(project="artifacts-example", job_type="initialize", config=config) as run:
        config = run.config
        
        model = ConvNet(**config)

        model_artifact = wandb.Artifact(
            "convnet", type="model",
            description="Simple AlexNet style CNN",
            metadata=dict(config))

        torch.save(model.state_dict(), "initialized_model.pth")
        # ➕ Artifact にファイルを追加する別の方法
        model_artifact.add_file("initialized_model.pth")

        run.save("initialized_model.pth")

        run.log_artifact(model_artifact)

model_config = {"hidden_layer_sizes": [32, 64],
                "kernel_sizes": [3],
                "activation": "ReLU",
                "pool_sizes": [2],
                "dropout": 0.5,
                "num_classes": 10}

build_model_and_log(model_config)
```

#### `artifact.add_file()`

データセットロギングの例のように `new_file` で書き込みと追加を同時にする代わりに、
一度ファイルを書き出してから（ここでは `torch.save`）、
あとで `add` することも可能です。

> **ベストプラクティス**: 可能なかぎり重複を避けるため `new_file` の利用を推奨します。

#### ログしたモデル Artifact を使う

データセットと同じように、`initialized_model` に対して `use_artifact` を使い、
他の `Run` で利用できます。

ここでは `model` の `train`（トレーニング）を行います。

詳しくは
[PyTorch と W&B の連携 Colab](https://wandb.me/pytorch-colab)
もご覧ください。

```python
import wandb
import torch.nn.functional as F

def train(model, train_loader, valid_loader, config):
    optimizer = getattr(torch.optim, config.optimizer)(model.parameters())
    model.train()
    example_ct = 0
    for epoch in range(config.epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            example_ct += len(data)

            if batch_idx % config.batch_log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0%})]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    batch_idx / len(train_loader), loss.item()))
                
                train_log(loss, example_ct, epoch)

        # 各エポックで検証セットを評価
        loss, accuracy = test(model, valid_loader)  
        test_log(loss, accuracy, example_ct, epoch)

    
def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum')  # バッチごとの損失を合計
            pred = output.argmax(dim=1, keepdim=True)  # 最大値予測
            correct += pred.eq(target.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)
    
    return test_loss, accuracy


def train_log(loss, example_ct, epoch):
    loss = float(loss)

    # ログの中心となる部分
    with wandb.init(project="artifacts-example", job_type="train") as run:
        run.log({"epoch": epoch, "train/loss": loss}, step=example_ct)
        print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")
    

def test_log(loss, accuracy, example_ct, epoch):
    loss = float(loss)
    accuracy = float(accuracy)

    # ログの中心となる部分
    with wandb.init() as run:
        run.log({"epoch": epoch, "validation/loss": loss, "validation/accuracy": accuracy}, step=example_ct)
        print(f"Loss/accuracy after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}/{accuracy:.3f}")
```

今回は2つの `Artifact` を生成する `Run` を使います。

最初の run で `model` を `train` したら、
次の run で `trained-model` Artifact を使い
`test_dataset` で性能を評価します。

また、ネットワークが最も「困惑」した32個のサンプル、つまり `categorical_crossentropy` が最も高い例を抽出します。

こういった難しい例をチェックすることは、データセットやモデルの問題を見つける良い方法です。

```python
def evaluate(model, test_loader):
    """
    ## 学習済みモデルの評価
    """

    loss, accuracy = test(model, test_loader)
    highest_losses, hardest_examples, true_labels, predictions = get_hardest_k_examples(model, test_loader.dataset)

    return loss, accuracy, highest_losses, hardest_examples, true_labels, predictions

def get_hardest_k_examples(model, testing_set, k=32):
    model.eval()

    loader = DataLoader(testing_set, 1, shuffle=False)

    # 各サンプルごとに損失と予測値を取得
    losses = None
    predictions = None
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            
            if losses is None:
                losses = loss.view((1, 1))
                predictions = pred
            else:
                losses = torch.cat((losses, loss.view((1, 1))), 0)
                predictions = torch.cat((predictions, pred), 0)

    argsort_loss = torch.argsort(losses, dim=0)

    highest_k_losses = losses[argsort_loss[-k:]]
    hardest_k_examples = testing_set[argsort_loss[-k:]][0]
    true_labels = testing_set[argsort_loss[-k:]][1]
    predicted_labels = predictions[argsort_loss[-k:]]

    return highest_k_losses, hardest_k_examples, true_labels, predicted_labels
```

このロギング関数群自体は、Artifact の新機能ではなく、
これまで通り `use`、`download`、`log` する流れです。

```python
from torch.utils.data import DataLoader

def train_and_log(config):

    with wandb.init(project="artifacts-example", job_type="train", config=config) as run:
        config = run.config

        data = run.use_artifact('mnist-preprocess:latest')
        data_dir = data.download()

        training_dataset =  read(data_dir, "training")
        validation_dataset = read(data_dir, "validation")

        train_loader = DataLoader(training_dataset, batch_size=config.batch_size)
        validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size)
        
        model_artifact = run.use_artifact("convnet:latest")
        model_dir = model_artifact.download()
        model_path = os.path.join(model_dir, "initialized_model.pth")
        model_config = model_artifact.metadata
        config.update(model_config)

        model = ConvNet(**model_config)
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
 
        train(model, train_loader, validation_loader, config)

        model_artifact = wandb.Artifact(
            "trained-model", type="model",
            description="Trained NN model",
            metadata=dict(model_config))

        torch.save(model.state_dict(), "trained_model.pth")
        model_artifact.add_file("trained_model.pth")
        run.save("trained_model.pth")

        run.log_artifact(model_artifact)

    return model

    
def evaluate_and_log(config=None):
    
    with wandb.init(project="artifacts-example", job_type="report", config=config) as run:
        data = run.use_artifact('mnist-preprocess:latest')
        data_dir = data.download()
        testing_set = read(data_dir, "test")

        test_loader = torch.utils.data.DataLoader(testing_set, batch_size=128, shuffle=False)

        model_artifact = run.use_artifact("trained-model:latest")
        model_dir = model_artifact.download()
        model_path = os.path.join(model_dir, "trained_model.pth")
        model_config = model_artifact.metadata

        model = ConvNet(**model_config)
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        loss, accuracy, highest_losses, hardest_examples, true_labels, preds = evaluate(model, test_loader)

        run.summary.update({"loss": loss, "accuracy": accuracy})

        run.log({"high-loss-examples":
            [wandb.Image(hard_example, caption=str(int(pred)) + "," +  str(int(label)))
             for hard_example, pred, label in zip(hardest_examples, preds, true_labels)]})
```

```python
train_config = {"batch_size": 128,
                "epochs": 5,
                "batch_log_interval": 25,
                "optimizer": "Adam"}

model = train_and_log(train_config)
evaluate_and_log()
```