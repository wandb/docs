---
title: モデルとデータセットのトラッキング
menu:
  tutorials:
    identifier: artifacts
weight: 4
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Pipeline_Versioning_with_W&B_Artifacts.ipynb" >}}
このノートブックでは、W&B Artifacts を利用して ML 実験のパイプラインをトラッキングする方法を紹介します。

[ビデオチュートリアル](https://tiny.cc/wb-artifacts-video)も参考にしてください。

## artifacts について

artifact は、ギリシャの[アンフォラ壺](https://en.wikipedia.org/wiki/Amphora)のような「生み出されたオブジェクト（生成物）」であり、何らかのプロセスの出力です。
ML の領域では、もっとも重要な artifact は _データセット_ と _モデル_ です。

そして、[コロナドの十字架](https://indianajones.fandom.com/wiki/Cross_of_Coronado)のように、こうした大事な artifact は「博物館」に収める価値があります。
つまり、きちんとカタログ化・整理して、
あなた自身・チーム、そして ML コミュニティ全体がそこから学べる状態にしましょう。
トレーニングの履歴をトラッキングしない者は、同じ失敗を繰り返す運命にあるのです。

W&B の Artifacts API を使えば、`Artifact` を W&B の `Run` の出力として記録したり、逆に `Artifact` を `Run` の入力として利用できます。たとえば、トレーニングの run がデータセットを入力としてモデルを出力する場合です。

{{< img src="/images/tutorials/artifacts-diagram.png" alt="Artifacts workflow diagram" >}}

1 つの run がほかの run の出力を入力として利用することもできるので、`Artifact` と `Run` は有向グラフ（バイパーティト [DAG](https://en.wikipedia.org/wiki/Directed_acyclic_graph)）のように結び付きます。ノードは `Artifact` と `Run` で構成され、矢印が `Run` とそれが扱う（消費・生成する）`Artifact` をつなぎます。

## artifacts を使ってモデルやデータセットを管理

### インストール & インポート

Artifacts は、バージョン `0.9.2` 以降の Python ライブラリに含まれています。

ML Python スタックの多くと同じく、`pip` でインストールできます。


```python
# wandb バージョン 0.9.2 以降に対応
!pip install wandb -qqq
!apt install tree
```


```python
import os
import wandb
```

### データセットの記録

まずは Artifacts の定義から始めましょう。

この例は、PyTorch の
["Basic MNIST Example"](https://github.com/pytorch/examples/tree/master/mnist/)
に基づいていますが、[TensorFlow](https://wandb.me/artifacts-colab) や他のフレームワーク、あるいは純粋な Python でも同じように実装できます。

`Dataset` を定義します：
- `train`用のセット（パラメータ調整用）
- `validation`用のセット（ハイパーパラメータ調整用）
- `test`用のセット（最終モデル評価用）

以下のセルで、これら 3 つのデータセットを定義します。


```python
import random 

import torch
import torchvision
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm

# 挙動を決定的にする
torch.backends.cudnn.deterministic = True
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# デバイス指定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# データ関連パラメータ
num_classes = 10
input_shape = (1, 28, 28)

# MNIST の遅いミラーサイトを除外
torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                      if not mirror.startswith("http://yann.lecun.com")]

def load(train_size=50_000):
    """
    # データの読み込み
    """

    # データを train/test に分割
    train = torchvision.datasets.MNIST("./", train=True, download=True)
    test = torchvision.datasets.MNIST("./", train=False, download=True)
    (x_train, y_train), (x_test, y_test) = (train.data, train.targets), (test.data, test.targets)

    # ハイパーパラメータ調整用に検証セットを分割
    x_train, x_val = x_train[:train_size], x_train[train_size:]
    y_train, y_val = y_train[:train_size], y_train[train_size:]

    training_set = TensorDataset(x_train, y_train)
    validation_set = TensorDataset(x_val, y_val)
    test_set = TensorDataset(x_test, y_test)

    datasets = [training_set, validation_set, test_set]

    return datasets
```

この例で繰り返し使うパターンを設定しています。
データを Artifact として記録するコードは、データ生成のコードをラップする形で書かれています。
この場合、データを読み込む `load` のコードと、それを記録する `load_and_log` のコードが分けられています。

これはおすすめのやり方です。

これらのデータセットを Artifacts として記録するには、
1. `wandb.init()` で `Run` を作成（L4）
2. データセット用の `Artifact` を作成（L10）
3. 関連する `file` を保存・記録（L20, L23）

以下のコード例を見て、その後の詳細説明セクションも参考にしてください。


```python
def load_and_log():

    # Run を開始。タイプとプロジェクトも指定
    with wandb.init(project="artifacts-example", job_type="load-data") as run:
        
        datasets = load()  # データセットの読み込みは別関数に分離
        names = ["training", "validation", "test"]

        # 🏺 Artifact の作成
        raw_data = wandb.Artifact(
            "mnist-raw", type="dataset",
            description="Raw MNIST dataset, split into train/val/test",
            metadata={"source": "torchvision.datasets.MNIST",
                      "sizes": [len(dataset) for dataset in datasets]})

        for name, data in zip(names, datasets):
            # 🐣 artifact にファイルを新規追加し、中身を書き込む
            with raw_data.new_file(name + ".pt", mode="wb") as file:
                x, y = data.tensors
                torch.save((x, y), file)

        # ✍️ Artifact を W&B に保存
        run.log_artifact(raw_data)

load_and_log()
```

#### `wandb.init()`

`Artifact` を生成する `Run` を作成する際は、それが属する `project` を指定する必要があります。

ワークフローによって、
プロジェクトの粒度は `car-that-drives-itself` のように大きくてもよいし、
`iterative-architecture-experiment-117` のように小さくても構いません。

> **ベストプラクティス**: 可能なら、同じ `Artifact` を共有する `Run` を単一プロジェクトにまとめましょう。シンプルに管理できます。なお、`Artifact` はプロジェクトを跨いで利用可能です。

様々なジョブ種別の管理には、`job_type` を指定するのもおすすめです。
Artifacts のグラフ構造をきれいに保てます。

> **ベストプラクティス**: `job_type` はできるだけ具体的にし、パイプラインの各ステップごとに分けましょう。ここでは、`load`（データ読み込み）と `preprocess`（前処理）を分割しています。

#### `wandb.Artifact`

何かを `Artifact` として記録したい場合、まずは `Artifact` オブジェクトを作成します。

すべての `Artifact` には `name` が必要で、これが第一引数です。

> **ベストプラクティス**: `name` は、分かりやすく、覚えやすく、タイプしやすいものにしましょう。ハイフン区切り＆コード内の変数名に合わせるのがおすすめです。

`type` も設定できます。これは、`Run` の `job_type` と同じく、`Artifact` グラフの整理に使われます。

> **ベストプラクティス**: `type` はできるだけシンプルに。`mnist-data-YYYYMMDD` より `dataset` や `model` がおすすめです。

`description` や `metadata`（辞書型）も添付可能です。
`metadata` は JSON シリアライズ可能なものにしましょう。

> **ベストプラクティス**: `metadata` には、できるだけ詳細な情報を書きましょう。

#### `artifact.new_file` と `run.log_artifact`

`Artifact` オブジェクトを作成したら、ファイルを追加します。

「ファイル *s*」と複数形なのがポイント。
`Artifact` はディレクトリのような階層構造を持っており、
ファイルやサブディレクトリを格納できます。

> **ベストプラクティス**: 内容を複数ファイルに分割できるなら、できるだけ分割しましょう。大規模化する時の助けになります。

`new_file` メソッドは、
ファイルの書き込みと artifact への添付を同時に行います。
後ほど `add_file` メソッド（書き込みと添付を分離）も紹介します。

すべてのファイル追加が終わったら、`log_artifact` で [wandb.ai](https://wandb.ai) に記録しましょう。

出力にいくつかの URL が現れます。
その中に Run ページの URL も含まれています。
ここで Run の結果や、
記録された `Artifact` を確認できます。

Run ページの他の要素を活用した例も、後ほど紹介します。

### 記録済み Dataset Artifact の利用

W&B の `Artifact` は、博物館の artifact と異なり、
「使われる」ことを前提としています。保存だけではありません。

具体例を見てみましょう。

下のセルでは、生のデータセットを受け取り、
正規化・形状加工した `preprocess` 済みのデータセットを生成するパイプラインステップを定義しています。

今回も、主要なロジック（`preprocess`）と、
`wandb` との連携部分は分けて実装しています。


```python
def preprocess(dataset, normalize=True, expand_dims=True):
    """
    ## データの前処理
    """
    x, y = dataset.tensors

    if normalize:
        # 画像を [0, 1] 範囲にスケーリング
        x = x.type(torch.float32) / 255

    if expand_dims:
        # 画像の形状を (1, 28, 28) に揃える
        x = torch.unsqueeze(x, 1)
    
    return TensorDataset(x, y)
```

次は、この `preprocess` ステップを `wandb.Artifact` で記録するためのコードです。

この例では、新たに `Artifact` を「使い」、そして「記録」しています。
`Artifact` は `Run` の入力にも出力にもなれる点が特徴です。

`job_type` も新しく `preprocess-data` として、前ステップとの差別化を図ります。


```python
def preprocess_and_log(steps):

    with wandb.init(project="artifacts-example", job_type="preprocess-data") as run:

        processed_data = wandb.Artifact(
            "mnist-preprocess", type="dataset",
            description="Preprocessed MNIST dataset",
            metadata=steps)
         
        # ✔️ 使用する artifact を宣言
        raw_data_artifact = run.use_artifact('mnist-raw:latest')

        # 📥 必要があれば artifact をダウンロード
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

ここで注目したいのは、前処理の `steps`（設定）が
`preprocessed_data` の `metadata` として一緒に保存されている点です。

実験を再現可能にしたい場合、
豊富なメタデータの記録はとても有効です。

また、今回のデータセットは「大きな artifact」ですが、
`download` ステップは 1 秒もかからず終わります。

詳細は、以下の markdown セルを開いて確認できます。


```python
steps = {"normalize": True,
         "expand_dims": True}

preprocess_and_log(steps)
```

#### `run.use_artifact()`

このステップはよりシンプルです。利用者は、`Artifact` の `name` と少しの追加情報を知っていれば十分です。

その「少しの追加」が `alias` です。
`Artifact` の特定のバージョンを指定します。

デフォルトでは、最新バージョンが `latest` です。
`v0` / `v1` のような番号や `best`, `jit-script` など独自のエイリアスも利用できます。
[Docker Hub](https://hub.docker.com/) のタグのように、
名前とエイリアスは `:` で区切ります。
本例なら `mnist-raw:latest` です。

> **ベストプラクティス**: エイリアスは短く分かりやすく。`latest` や `best` のような独自エイリアスは便利です。

#### `artifact.download`

`download` の呼び出しで、メモリ使用量が倍増するのでは？と心配するかもしれません。

ご安心ください。ダウンロード前に、対象バージョンがローカルに存在するかを確認しています。
これは [torrent](https://en.wikipedia.org/wiki/Torrent_file) や [`git`](https://blog.thoughtram.io/git/2014/11/18/the-anatomy-of-a-git-commit.html) のバージョン管理と同じくハッシュ技術を活用しています。

Artifact が作成・記録されると、
作業ディレクトリの `artifacts` フォルダに
サブディレクトリが増えていきます。
それぞれが一つの `Artifact` です。
中身は `!tree artifacts` で確認できます：


```python
!tree artifacts
```

#### Artifacts ページ

`Artifact` を記録し、利用した今、
Run ページの Artifacts タブを見てみましょう。

`wandb` の出力に表示された Run ページの URL にアクセスし、
左サイドバーの「Artifacts」タブ（データベースのアイコン、「ホッケーパック 3 つ重ね」に見える）を選択してください。

**Input Artifacts** または **Output Artifacts** のテーブルの行をクリックし、
**Overview** や **Metadata** のタブでその `Artifact` の全記録を確認できます。

特に **Graph View**（グラフ表示）が便利です。
デフォルトでは、
`Artifact` の `type` と
`Run` の `job_type` をノードとして矢印でつなぎ、
どの Run がどの Artifact を消費・生成したかが一目で分かります。

### モデルの記録

これで `Artifact` API の基本は押さえられましたが、
パイプライン最後まで例を進めて、
Artifacts が ML ワークフローにどう役立つかも紹介しましょう。

まずは PyTorch で DNN（ごくシンプルな ConvNet モデル）を構築します。

ここではモデルの初期化のみを行い、トレーニングはしません。
同じ条件で何度でもトレーニングできるようにしています。


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

        fc_input_dims = floor((input_shape[1] - kernel_sizes[0] + 1) / pool_sizes[0]) # layer 1 の出力サイズ
        fc_input_dims = floor((fc_input_dims - kernel_sizes[-1] + 1) / pool_sizes[-1]) # layer 2 の出力サイズ
        fc_input_dims = fc_input_dims*fc_input_dims*hidden_layer_sizes[-1] # layer 3 の出力サイズ

        self.fc = nn.Linear(fc_input_dims, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc(x)
        return x
```

今回は W&B で run をトラッキングするので、
すべてのハイパーパラメータは [`run.config`](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-config/Configs_in_W%26B.ipynb)
オブジェクトに保存します。

この `config` オブジェクト（辞書型）は、大変便利な `metadata` となるので、必ず一緒に記録しましょう。


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
        # ➕ Artifact へのファイル追加方法の別パターン
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

データセット記録の例では `new_file` を使ってファイル作成と artifact への追加を同時に行いましたが、
この方法では（ここでは `torch.save` の後）
ファイル書き込みと artifact への追加を分離しています。

> **ベストプラクティス**: 重複を防ぐため、可能な限り `new_file` を使いましょう。

#### 記録済みモデル Artifact の利用

データセットと同じように、`initialized_model` に対しても
`use_artifact` を活用し、他の `Run` で利用できます。

今度は、この `model` を `train` しましょう。

詳細は
[PyTorch での W&B 利用](https://wandb.me/pytorch-colab)
の Colab もご覧ください。


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

        # 各エポックごとに検証セットでモデルを評価
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
            test_loss += F.cross_entropy(output, target, reduction='sum')  # バッチ損失を合計
            pred = output.argmax(dim=1, keepdim=True)  # 最大値のインデックスを取得
            correct += pred.eq(target.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)
    
    return test_loss, accuracy


def train_log(loss, example_ct, epoch):
    loss = float(loss)

    # wandb へのログ記録
    with wandb.init(project="artifacts-example", job_type="train") as run:
        run.log({"epoch": epoch, "train/loss": loss}, step=example_ct)
        print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")
    

def test_log(loss, accuracy, example_ct, epoch):
    loss = float(loss)
    accuracy = float(accuracy)

    # wandb へのログ記録
    with wandb.init() as run:
        run.log({"epoch": epoch, "validation/loss": loss, "validation/accuracy": accuracy}, step=example_ct)
        print(f"Loss/accuracy after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}/{accuracy:.3f}")
```

今回は 2 回の `Artifact` 生成をともなう `Run` を行います。

最初の `model` の `train` が終わったら、
2 回目の run で `trained-model` の `Artifact` を使い、
テストデータセットで性能評価を行います。

また、ネットワークがもっとも苦手とする 32 個のサンプル（`categorical_crossentropy` が最大となるサンプル）を抽出します。

これは、データセットやモデルの問題点を発見するのに役立ちます。


```python
def evaluate(model, test_loader):
    """
    ## トレーニング済みモデルの評価
    """

    loss, accuracy = test(model, test_loader)
    highest_losses, hardest_examples, true_labels, predictions = get_hardest_k_examples(model, test_loader.dataset)

    return loss, accuracy, highest_losses, hardest_examples, true_labels, predictions

def get_hardest_k_examples(model, testing_set, k=32):
    model.eval()

    loader = DataLoader(testing_set, 1, shuffle=False)

    # 各サンプルごとの損失と予測を取得
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

これらのログ記録用関数には新しい Artifact 機能はありません。
`use`・`download`・`log` の基本操作だけです。


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