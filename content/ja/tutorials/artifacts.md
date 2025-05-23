---
title: モデルとデータセットをトラッキングする
menu:
  tutorials:
    identifier: ja-tutorials-artifacts
weight: 4
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Pipeline_Versioning_with_W&B_Artifacts.ipynb" >}} 
このノートブックでは、W&B Artifacts を使用して ML 実験パイプラインを追跡する方法を紹介します。

[ビデオチュートリアル](http://tiny.cc/wb-artifacts-video)とともに進めてください。

## アーティファクトについて

ギリシャの[アンフォラ](https://en.wikipedia.org/wiki/Amphora)のように、アーティファクトは生成されたオブジェクト、つまりプロセスの出力です。
MLでは、最も重要なアーティファクトは _datasets_ と _models_ です。

そして、[コロナドの十字架](https://indianajones.fandom.com/wiki/Cross_of_Coronado)のように、これらの重要なアーティファクトは博物館に保管されるべきです。
それは、あなたやあなたのチーム、そして広範な ML コミュニティがそれらから学べるように、カタログ化されて整理されるべきだということです。
結局のところ、トレーニングを追跡しない人々はそれを繰り返す運命にあります。

私たちの Artifacts API を使用することで、`Artifact` を W&B `Run` の出力としてログしたり、`Run` の入力として `Artifact` を使用したりできます。
この図では、トレーニング run がデータセットを取り込み、モデルを生成します。
 
{{< img src="/images/tutorials/artifacts-diagram.png" alt="" >}} 

1つの run が他の run の出力を入力として使用できるため、`Artifact` と `Run` は、`Artifact` と `Run` のノードを持ち、それを消費または生成する `Run` に接続する矢印を持つ、有向グラフ（二部 [DAG](https://en.wikipedia.org/wiki/Directed_acyclic_graph)）を形成します。

## アーティファクトを使用してモデルとデータセットを追跡する

### インストールとインポート

Artifacts は、バージョン `0.9.2` 以降の私たちの Python ライブラリの一部です。

ほとんどの ML Python スタックの部分と同様に、`pip` で利用可能です。

```python
# wandb バージョン 0.9.2+ 互換
!pip install wandb -qqq
!apt install tree
```

```python
import os
import wandb
```

### データセットをログする

まず、いくつかの Artifacts を定義しましょう。

この例は、PyTorch の
["Basic MNIST Example"](https://github.com/pytorch/examples/tree/master/mnist/)
を基にしていますが、[TensorFlow](http://wandb.me/artifacts-colab) や他のフレームワーク、
あるいは純粋な Python でも同様に行うことができます。

データセットを `train`、`validation`、`test` として次のように定義します。
- パラメータを選択するための `train` セット
- ハイパーパラメータを選択するための `validation` セット
- 最終モデルを評価するための `test` セット

以下の最初のセルでこれらの3つのデータセットを定義します。

```python
import random

import torch
import torchvision
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm

# 挙動を決定的に
torch.backends.cudnn.deterministic = True
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# デバイスの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# データパラメータ
num_classes = 10
input_shape = (1, 28, 28)

# 遅いミラーを MNIST ミラーリストから削除
torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                      if not mirror.startswith("http://yann.lecun.com")]

def load(train_size=50_000):
    """
    # データをロードする
    """

    # データ、train と test セットで分割
    train = torchvision.datasets.MNIST("./", train=True, download=True)
    test = torchvision.datasets.MNIST("./", train=False, download=True)
    (x_train, y_train), (x_test, y_test) = (train.data, train.targets), (test.data, test.targets)

    # ハイパーパラメータチューニングのために検証セットを分割
    x_train, x_val = x_train[:train_size], x_train[train_size:]
    y_train, y_val = y_train[:train_size], y_train[train_size:]

    training_set = TensorDataset(x_train, y_train)
    validation_set = TensorDataset(x_val, y_val)
    test_set = TensorDataset(x_test, y_test)

    datasets = [training_set, validation_set, test_set]

    return datasets
```

このコードでは、アーティファクトとしてデータをログするコードがデータを生成するコードにラップされているというパターンが示されています。
この場合、データを `load` するコードはデータを `load_and_log` するコードとは分かれています。

これは良い実践です。

これらのデータセットをアーティファクトとしてログするには、次の手順が必要です。
1. `wandb.init` で `Run` を作成する (L4)
2. データセットの `Artifact` を作成する (L10)
3. 関連する `file` を保存してログする (L20, L23)

以下のコードセルの例を確認し、その後のセクションで詳細を展開してください。

```python
def load_and_log():

    # 🚀 run を開始し、ラベルを付けてどのプロジェクトに属するか呼び出せるようにする
    with wandb.init(project="artifacts-example", job_type="load-data") as run:
        
        datasets = load()  # データセットをロードするコードを分ける
        names = ["training", "validation", "test"]

        # 🏺 アーティファクトを作成
        raw_data = wandb.Artifact(
            "mnist-raw", type="dataset",
            description="Raw MNIST dataset, split into train/val/test",
            metadata={"source": "torchvision.datasets.MNIST",
                      "sizes": [len(dataset) for dataset in datasets]})

        for name, data in zip(names, datasets):
            # 🐣 アーティファクトに新しいファイルを保存し、その内容に何かを書き込む
            with raw_data.new_file(name + ".pt", mode="wb") as file:
                x, y = data.tensors
                torch.save((x, y), file)

        # ✍️ W&B にアーティファクトを保存
        run.log_artifact(raw_data)

load_and_log()
```

#### `wandb.init`

`Artifact` を生成する `Run` を作成するとき、その `project` に属することを明示する必要があります。

ワークフローに応じて、
プロジェクトは `car-that-drives-itself` のように大きいものであったり、
`iterative-architecture-experiment-117` のように小さいものであったりします。

> **良い実践のルール**: 可能であれば、同じ `Artifact` を共有するすべての `Run` を単一のプロジェクト内に保持してください。これによりシンプルになりますが、心配しないでください -- `Artifact` はプロジェクト間で移動可能です。

実行する様々な種類のジョブを追跡するために、`Runs` を作成する際に `job_type` を提供することをお勧めします。
これにより、Artifacts のグラフが整然とした状態を保つことができます。

> **良い実践のルール**: `job_type` は説明的であり、単一のパイプラインのステップに対応するものであるべきです。ここでは、`preprocess` データと `load` データを分けて処理しています。

#### `wandb.Artifact`

何かを `Artifact` としてログするためには、最初に `Artifact` オブジェクトを作成する必要があります。

すべての `Artifact` には `name` があり -- これが第一引数で設定されます。

> **良い実践のルール**: `name` は説明的で、記憶しやすくタイプしやすいものであるべきです -- 私たちは、コード内の変数名に対応するハイフンで区切られた名前を使用するのが好きです。

また、それには `type` があります。`Run` の `job_type` と同様に、これも `Run` と `Artifact` のグラフを整理するために使用されます。

> **良い実践のルール**: `type` はシンプルであるべきです。
`mnist-data-YYYYMMDD` よりも `dataset` や `model` のように。

また、辞書として `description` と `metadata` をアタッチすることもできます。
`metadata` は JSON にシリアライズ可能である必要があります。

> **良い実践のルール**: `metadata` はできる限り詳しく記述するべきです。

#### `artifact.new_file` および `run.log_artifact`

一度 `Artifact` オブジェクトを作成したら、ファイルを追加する必要があります。

あなたはそれを正しく読みました。複数のファイル（`s`付き）です。
`Artifact` はディレクトリーのように構造化されており、
ファイルとサブディレクトリーを持っています。

> **良い実践のルール**: 可能な限り、`Artifact` の内容を複数のファイルに分割してください。スケールする時が来たときに役立ちます。

`new_file` メソッドを使用して、ファイルを書き込むと同時に`Artifact` にアタッチします。
次のセクションでは、これらの2つの手順を分ける `add_file` メソッドを使用します。

すべてのファイルを追加したら、`run.log_artifact` を使用して [wandb.ai](https://wandb.ai) にログする必要があります。

出力には `Run` ページの URL などのいくつかの URL が表示されることに気付くでしょう。
そこでは、ログされた `Artifact` を含む `Run` の結果を確認できます。

下記では、Run ページの他のコンポーネントをより活用する例をいくつか見ていきます。

### ログされたデータセットアーティファクトを使用する

W&B 内の `Artifact` は博物館内のアーティファクトとは異なり、
単に保存されるだけでなく _使用_ を想定しています。

それがどのように機能するか見てみましょう。

以下のセルでは、生のデータセットを取り込み、それを `preprocess` したデータセット（つまり `normalize` され正しい形に整形されたもの）を生成するパイプラインステップを定義しています。

また、コードの重要部分である `preprocess` を `wandb` とインターフェースするコードから分けています。

```python
def preprocess(dataset, normalize=True, expand_dims=True):
    """
    ## データを準備する
    """
    x, y = dataset.tensors

    if normalize:
        # 画像を[0, 1]の範囲にスケール
        x = x.type(torch.float32) / 255

    if expand_dims:
        # 画像が形状(1, 28, 28)を持つことを確認
        x = torch.unsqueeze(x, 1)
    
    return TensorDataset(x, y)
```

次に示すのは、この `preprocess` ステップを `wandb.Artifact` ログで装備するためのコードです。

以下の例では、`use` する `Artifact` が新たに登場し、`log` することは前のステップと同様です。
`Artifact` は、`Run` の入力と出力の両方です。

このステップが前のものとは異なる種類のジョブであることを明確にするために、新しい `job_type`、`preprocess-data` を使用します。

```python
def preprocess_and_log(steps):

    with wandb.init(project="artifacts-example", job_type="preprocess-data") as run:

        processed_data = wandb.Artifact(
            "mnist-preprocess", type="dataset",
            description="Preprocessed MNIST dataset",
            metadata=steps)
         
        # ✔️ どのアーティファクトを使用するかを宣言
        raw_data_artifact = run.use_artifact('mnist-raw:latest')

        # 📥 必要に応じてアーティファクトをダウンロード
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

ここで注意すべきことは、前処理の `ステップ` が `preprocessed_data` に `metadata` とともに保存されているということです。

実験を再現可能にしようとしている場合、多くのメタデータを収集することは良いアイデアです。

また、データセットは "`large artifact`" であるにもかかわらず、`download` ステップは1秒もかからずに完了します。

詳細は以下のマークダウンセルを展開してください。

```python
steps = {"normalize": True,
         "expand_dims": True}

preprocess_and_log(steps)
```

#### `run.use_artifact`

これらのステップはよりシンプルです。消費者はその `Artifact` の `name` を知っていればいいだけです。その "bit more" とは、欲しい `Artifact` の特定のバージョンの `alias` です。

デフォルトでは、最後にアップロードされたバージョンが `latest` としてタグ付けされます。
それ以外の場合は `v0` や `v1` などで古いバージョンを選択することもでき、
また `best` や `jit-script` のような独自の alias を提供することもできます。
Docker Hub タグのように、alias は名前から `:` で分離されるため、私たちの求める `Artifact` は `mnist-raw:latest` になります。

> **良い実践のルール**: alias を短く簡潔に保持します。
カスタム `alias` である `latest` や `best` のような `Artifact` を使用して、ある条件を満たすものを指定します。

#### `artifact.download`

さて、あなたは `download` 呼び出しについて心配しているかもしれません。
別のコピーをダウンロードすると、メモリの負担が2倍になるのではないでしょうか？

心配しないでください。実際に何かをダウンロードする前に、
正しいバージョンがローカルにあるかどうか確認します。
これは [torrenting](https://en.wikipedia.org/wiki/Torrent_file) や [`git` を使用したバージョン管理](https://blog.thoughtram.io/git/2014/11/18/the-anatomy-of-a-git-commit.html) と同じ技術、ハッシュ化を使用しています。

`Artifact` が作成されログされると、
作業ディレクトリ内の `artifacts` というフォルダに
各 `Artifact` 用のサブディレクトリが埋まり始めます。
その内容を `!tree artifacts` で確認してください。

```python
!tree artifacts
```

#### アーティファクトページ

`Artifact` をログし、使用したので、Run ページの Artifacts タブを見てみましょう。

`wandb` 出力からの Run ページの URL に移動し、
左サイドバーから "Artifacts" タブを選択します
（これはデータベースのアイコンで、
ホッケークラブを3つ重ねたようなものに見えるアイコンです）。

**Input Artifacts** テーブルや **Output Artifacts** テーブルのいずれかの行をクリックし、
その後に表示されるタブ (**Overview**, **Metadata**)
でログされた `Artifact` についての情報を確認してください。

私たちは特に **Graph View** を好んでいます。
デフォルトでは、`Artifact` の `type` と
`Run` の `job_type` を 2 つの種類のノードとして持ち、
消費と生成を表す矢印を持つグラフを表示します。

### モデルをログする

これで `Artifact` API の動作がわかりましたが、
ML ワークフローを改善するために `Artifact` がどのように役立つかを理解するために、
パイプラインの終わりまでこの例を追ってみましょう。

ここでの最初のセルは、PyTorch で DNN `model` を構築します -- 本当にシンプルな ConvNet です。

まず、`model` を初期化するだけで、トレーニングはしません。
そのため、すべての他の要素を一定に保ちながら複数のトレーニングを繰り返すことができます。

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

ここでは、W&B を使用して run を追跡しており、すべてのハイパーパラメーターを保存するために [`wandb.config`](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-config/Configs_in_W%26B.ipynb) オブジェクトを使用しています。

その `dict`ionary バージョンの `config` object は本当に便利な `metadata` のピースなので、必ずそれを含めてください。

```python
def build_model_and_log(config):
    with wandb.init(project="artifacts-example", job_type="initialize", config=config) as run:
        config = wandb.config
        
        model = ConvNet(**config)

        model_artifact = wandb.Artifact(
            "convnet", type="model",
            description="Simple AlexNet style CNN",
            metadata=dict(config))

        torch.save(model.state_dict(), "initialized_model.pth")
        # ➕ アーティファクトにファイルを追加する別の方法
        model_artifact.add_file("initialized_model.pth")

        wandb.save("initialized_model.pth")

        run.log_artifact(model_artifact)

model_config = {"hidden_layer_sizes": [32, 64],
                "kernel_sizes": [3],
                "activation": "ReLU",
                "pool_sizes": [2],
                "dropout": 0.5,
                "num_classes": 10}

build_model_and_log(model_config)
```

#### `artifact.add_file`

データセットのログ例のように、`new_file` を使用してファイルを書き込みアーティファクトに追加する代わりに、
ファイルを書き込んだ後に（ここでは `torch.save` により）それをアーティファクトに追加することもできます。

> **👍のルール**: 重複を避けるために、できるだけ `new_file` を使用してください。

#### ログされたモデルアーティファクトを使用する

私たちがデータセットに対して `use_artifact` を呼び出すことができたように、別の `Run` で `initialized_model` を使用することもできます。

この時は、`model` を `train` しましょう。

詳細については、
[instrumenting W&B with PyTorch](http://wandb.me/pytorch-colab) に関する Colab を参照してください。

```python
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

        # エポック毎に検証セットでモデルを評価
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
            pred = output.argmax(dim=1, keepdim=True)  # 最大のlog確率のインデックスを取得
            correct += pred.eq(target.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)
    
    return test_loss, accuracy


def train_log(loss, example_ct, epoch):
    loss = float(loss)

    # 魔法が起こる場所
    wandb.log({"epoch": epoch, "train/loss": loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")
    

def test_log(loss, accuracy, example_ct, epoch):
    loss = float(loss)
    accuracy = float(accuracy)

    # 魔法が起こる場所
    wandb.log({"epoch": epoch, "validation/loss": loss, "validation/accuracy": accuracy}, step=example_ct)
    print(f"Loss/accuracy after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}/{accuracy:.3f}")
```

今回は 2 つの別々の `Artifact` を生成する `Run` を実行します。

最初の `model` ファイルを `train` する `Run` が終了し次第、
`second` は `test_dataset` 上の評価性能を `evaluate` することにより
`trained-model` `Artifact` を消費します。

また、ネットワークが最も混乱する32の例、
つまり `categorical_crossentropy` が最高の例を引き出します。

これは、データセットとモデルの問題を診断するための良い方法です。

```python
def evaluate(model, test_loader):
    """
    ## トレーニングされたモデルを評価する
    """

    loss, accuracy = test(model, test_loader)
    highest_losses, hardest_examples, true_labels, predictions = get_hardest_k_examples(model, test_loader.dataset)

    return loss, accuracy, highest_losses, hardest_examples, true_labels, predictions

def get_hardest_k_examples(model, testing_set, k=32):
    model.eval()

    loader = DataLoader(testing_set, 1, shuffle=False)

    # データセット内の各アイテムに対する損失と予測を取得する
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

これらのログ関数は新しい `Artifact` 機能を追加することはないため、
コメントしません: これらは単に、
`Artifact` を `use` し、`download` し、
`log` しているだけです。

```python
from torch.utils.data import DataLoader

def train_and_log(config):

    with wandb.init(project="artifacts-example", job_type="train", config=config) as run:
        config = wandb.config

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
        wandb.save("trained_model.pth")

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

        wandb.log({"high-loss-examples":
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
