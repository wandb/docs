---
title: Track models and datasets
menu:
  tutorials:
    identifier: ja-tutorials-artifacts
weight: 4
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Pipeline_Versioning_with_W&B_Artifacts.ipynb" >}}
このノートブックでは、W&B Artifacts を使用して ML 実験のパイプラインを追跡する方法を紹介します。

[ビデオチュートリアル](http://tiny.cc/wb-artifacts-video)を見ながら進めてください。

## アーティファクトについて

アーティファクトは、ギリシャの[アンフォラ](https://en.wikipedia.org/wiki/Amphora)のように、プロセスの出力物であるオブジェクトです。  
MLにおいて最も重要なアーティファクトは、_datasets_ と _models_ です。

さらに、[コロナードの十字架](https://indianajones.fandom.com/wiki/Cross_of_Coronado)のように、これらの重要なアーティファクトは博物館に収めるべきです。つまり、カタログ化され、整理された状態で保存されるべきであり、あなたやチーム、ML コミュニティがそれから学べるようにする必要があります。トレーニングを追跡しない人々は、トレーニングを繰り返さざるを得ないのです。

Artifacts API を使用すると、W&B `Run` の出力として `Artifact` をログに記録したり、`Run` の入力として `Artifact` を使用したりできます。この図では、トレーニング run がデータセットを受け取り、モデルを生成する例を示しています。

{{< img src="/images/tutorials/artifacts-diagram.png" alt="" >}}

1つの run が別の run の出力を入力として使用することができるので、`Artifact` と `Run` は有向グラフを形成します ( `Artifact` と `Run` のためのノードを持つ二部 [DAG](https://en.wikipedia.org/wiki/Directed_acyclic_graph) で、`Run` を消費または生成する `Artifact` に接続する矢印があります)。

## モデルとデータセットを追跡するためにアーティファクトを使用する

### インストールとインポート

Artifacts はバージョン `0.9.2` からの Python ライブラリの一部です。

ML Python スタックの多くの部分と同様に、`pip` 経由で利用可能です。

```python
# wandb バージョン 0.9.2+ と互換性あり
!pip install wandb -qqq
!apt install tree
```

```python
import os
import wandb
```

### データセットをログする

まず、いくつかの Artifacts を定義しましょう。

この例は PyTorch の ["Basic MNIST Example"](https://github.com/pytorch/examples/tree/master/mnist/) に基づいていますが、[TensorFlow](http://wandb.me/artifacts-colab)、他のフレームワーク、または純粋な Python でも同様に可能です。

`Dataset` から始めましょう:
- パラメータを選択する `train`ing セット、
- ハイパーパラメータを選択する `validation` セット、
- 最終モデルを評価するための `test`ing セット

以下の最初のセルは、これら3つのデータセットを定義しています。

```python
import random 

import torch
import torchvision
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm

# 確定的な振る舞いを保証する
torch.backends.cudnn.deterministic = True
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# デバイス設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# データパラメータ
num_classes = 10
input_shape = (1, 28, 28)

# 遅いミラーを MNIST ミラーのリストから削除する
torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                      if not mirror.startswith("http://yann.lecun.com")]

def load(train_size=50_000):
    """
    # データをロードする
    """

    # データを、train と test セットに分割する
    train = torchvision.datasets.MNIST("./", train=True, download=True)
    test = torchvision.datasets.MNIST("./", train=False, download=True)
    (x_train, y_train), (x_test, y_test) = (train.data, train.targets), (test.data, test.targets)

    # ハイパーパラメータチューニングのために検証セットを分割する
    x_train, x_val = x_train[:train_size], x_train[train_size:]
    y_train, y_val = y_train[:train_size], y_train[train_size:]

    training_set = TensorDataset(x_train, y_train)
    validation_set = TensorDataset(x_val, y_val)
    test_set = TensorDataset(x_test, y_test)

    datasets = [training_set, validation_set, test_set]

    return datasets
```

この例で繰り返し見るパターンがここにあります。 データをアーティファクトとしてログするコードが、そのデータを生成するためのコードでラップされています。 この場合、データを `load` するコードが、データを `load_and_log` するコードから分離されています。

これは良い慣習です。

これらのデータセットをアーティファクトとしてログするために、次の手順を実行します。
1. `wandb.init` を使用して `Run` を作成する (L4)
2. データセットのための `Artifact` を作成する (L10)
3. 関連する `file` を保存してログに記録する (L20, L23)

以下のコードセルの例を確認し、その後に続くセクションを展開して詳細を確認してください。

```python
def load_and_log():

    # 🚀 run を開始し、それにタイプをラベル付けし、プロジェクトとして呼び出せるようにする
    with wandb.init(project="artifacts-example", job_type="load-data") as run:
        
        datasets = load()  # データセットをロードするためのコードを分割する
        names = ["training", "validation", "test"]

        # 🏺 Artifact を作成する
        raw_data = wandb.Artifact(
            "mnist-raw", type="dataset",
            description="Raw MNIST dataset, split into train/val/test",
            metadata={"source": "torchvision.datasets.MNIST",
                      "sizes": [len(dataset) for dataset in datasets]})

        for name, data in zip(names, datasets):
            # 🐣 アーティファクトに新しいファイルを格納し、その内容に何かを書き込む。
            with raw_data.new_file(name + ".pt", mode="wb") as file:
                x, y = data.tensors
                torch.save((x, y), file)

        # ✍️ アーティファクトを W&B に保存する。
        run.log_artifact(raw_data)

load_and_log()
```

#### `wandb.init`

`Artifact` を生成する `Run` を作成するとき、`project` がどの `project` に属しているかを明示する必要があります。

ワークフローによりますが、プロジェクトは `car-that-drives-itself` のように大きくても、`iterative-architecture-experiment-117` のように小さくても構いません。

> **Rule of 👍**: 可能であれば、`Artifact` を共有するすべての `Run` を単一のプロジェクト内に保持しましょう。これにより物事がシンプルになりますが、心配は無用です。`Artifact` はプロジェクト間でポータブルです。

さまざまなタイプのジョブを追跡するために、`Run` を作成する際に `job_type` を提供することが役立ちます。これにより、Artifact のグラフが整然と整備されます。

> **Rule of 👍**: `job_type` は記述的であり、パイプラインの単一のステップに対応するべきです。ここでは、データの `load` をデータの `preprocess` から分離しています。

#### `wandb.Artifact`

`Artifact` として何かをログに記録するには、最初に `Artifact` オブジェクトを作成する必要があります。

すべての `Artifact` には `name` があり、最初の引数で設定します。

> **Rule of 👍**: `name` は記述的であるべきですが、記憶しやすく打ち込みやすいものである必要があります。私たちはハイフンで区切られた名前を使用し、コード内の変数名に対応させるのが好きです。

`type` も持っています。 `Run` の `job_type` のように、これは `Run` と `Artifact` のグラフを整理するために使用されます。

> **Rule of 👍**: `type` はシンプルであるべきです。`dataset` や `model` のようにシンプルでなるべく一般的なものにしましょう。

また、`description` と、辞書として `metadata` を添付することもできます。`metadata` は JSON にシリアライズ可能である必要があります。

> **Rule of 👍**: `metadata` はできるだけ詳しくするべきです。

#### `artifact.new_file` と `run.log_artifact`

`Artifact` オブジェクトを作成した後、ファイルを追加する必要があります。

_ファイル_ と _複数形_ で言ったことに気づきましたか？  
`Artifact` はディレクトリのように構造化されており、ファイルやサブディレクトリを持つことができます。

> **Rule of 👍**: 可能な場合、`Artifact` の内容を複数のファイルに分割しましょう。これにより、スケールするときに役立ちます。

`new_file` メソッドを使用して、同時にファイルを書き込み、その `Artifact` に添付します。そして下記で、`add_file` メソッドを使用して、その2つのステップを分けます。

すべてのファイルを追加したら、`log_artifact` を [wandb.ai](https://wandb.ai) に実行しましょう。

出力にいくつかの URL が表示されたことに注目してください。その中には Run ページのものもあります。
そこでは `Run` の結果を確認できます。
次に、Run ページの他のコンポーネントを使用したいくつかの例を見ましょう。

### ログされたデータセット アーティファクトを使用する

博物館内のアーティファクトとは異なり、W&B の `Artifact` は単に保存されるだけでなく、「使用」するために設計されています。

その様子を見てみましょう。

以下のセルは、raw データセットを使用して、`preprocess`ed データセット: `normalize`され、正しく形状を整えたデータセットを生成するパイプラインステップを定義しています。

再び、`wandb` をインターフェースするコードから `preprocess` のコードの本体を分離していることに注意してください。

```python
def preprocess(dataset, normalize=True, expand_dims=True):
    """
    ## データの準備
    """
    x, y = dataset.tensors

    if normalize:
        # 画像を [0, 1] 範囲にスケーリングする
        x = x.type(torch.float32) / 255

    if expand_dims:
        # 画像が形状 (1, 28, 28) を持つことを確認する
        x = torch.unsqueeze(x, 1)
    
    return TensorDataset(x, y)
```

それでは、`wandb.Artifact` でログをとる`preprocess` ステップに関連するコードについて見ていきましょう。

以下の例では、`Artifact` を `使用` し、新しいステップとして `log` していることに注意してください。`Artifact` は `Run` の入力であり、出力でもあります。

新しい `job_type` 、`preprocess-data` を使用することで、これが前のものとは異なる種類のジョブであることを明確にします。

```python
def preprocess_and_log(steps):

    with wandb.init(project="artifacts-example", job_type="preprocess-data") as run:

        processed_data = wandb.Artifact(
            "mnist-preprocess", type="dataset",
            description="Preprocessed MNIST dataset",
            metadata=steps)
         
        # ✔️ 使用するアーティファクトを宣言する
        raw_data_artifact = run.use_artifact('mnist-raw:latest')

        # 📥 必要であれば、アーティファクトをダウンロードする
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

ここで気付くべきことの1つは、前処理の `steps` が `preprocessed_data` と共に `metadata` として保存されることです。

実験を再現可能にしたい場合、たくさんのメタデータを記録することは良い考えです。

また、私たちのデータセットが「`large artifact`」であるにもかかわらず、`download` ステップが1秒以内で実行されていることにも注目してください。

以下のマークダウンセルを展開して詳細を確認してください。

```python
steps = {"normalize": True,
         "expand_dims": True}

preprocess_and_log(steps)
```

#### `run.use_artifact`

これらのステップはより単純です。消費者は `Artifact` の `name` を知っているだけで十分です。

その「少し余計なもの」は、必要な `Artifact` の特定のバージョンの `alias` です。

デフォルトでは、最後にアップロードされたバージョンには `latest` がタグ付けされます。 それ以外の場合、 `v0`/`v1` などの数字で古いバージョンを選択したり、`best` や `jit-script` のような独自のエイリアスを提供することができます。 [Docker Hub](https://hub.docker.com/)のタグのように、エイリアスは名前と `:` で区切られるので、私たちが求める `Artifact` は `mnist-raw:latest` です。

> **Rule of 👍**: エイリアスは短く簡潔にしてください。`Artifact` があるプロパティを満たす時に最新または最良のカスタムエイリアスを使用してください。

#### `artifact.download`

`download` 呼び出しについて心配しているかもしれません。
別のコピーをダウンロードすると、メモリの負担が倍増しませんか？

心配いりません。 実際に何かをダウンロードする前に、適切なバージョンがローカルに存在するかどうかを確認します。
これには、[torrenting](https://en.wikipedia.org/wiki/Torrent_file) および [`git`を使用したバージョン管理](https://blog.thoughtram.io/git/2014/11/18/the-anatomy-of-a-git-commit.html)を根底に支える技術が使用されます：ハッシングです。

`Artifact` が作成・ログされると、作業ディレクトリ内に `artifacts` というフォルダが作成され始めます。
ここには `Artifact` ごとにサブディレクトリが1つずつ作成されます。
`!tree artifacts` でその内容を確認してみましょう：

```python
!tree artifacts
```

#### アーティファクトページ

`Artifact` をログし、使用した今、Run ページのアーティファクトタブをチェックしてみましょう。

`wandb` の出力から Run ページの URL へ移動し、左側のサイドバーから「Artifacts」タブを選択します（それはデータベースのアイコンであり、3つのホッケーパックが積み重なったように見えます）。

**Input Artifacts** テーブルまたは **Output Artifacts** テーブルの行をクリックし、項目 (**Overview**, **Metadata**) を確認して、ログされた `Artifact` について確認しましょう。

私たちは特に **Graph View** が気に入っています。
デフォルトでは、`Artifact` の `type` と `Run` の `job_type` を2種類のノードとするグラフを表示し、消費と生成を表す矢印を示しています。

### モデルのログ

次に、`パイプライン`の終わりまでこの例をたどり、`Artifact` がどのようにして ML ワークフローを改善できるかを見てみましょう。

最初のセルでは、PyTorch で DNN `model` を構築します。本当にシンプルな ConvNet です。

ここでは、`モデル`をトレーニングせずに初期化するだけです。そうすることで、他のすべてを保ったままトレーニングを繰り返すことができます。

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

ここでは、 W&B を使って run を追跡しており、 [`wandb.config`](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-config/Configs_in_W%26B.ipynb) オブジェクトを使用してすべてのハイパーパラメータを格納しています。

その `config` オブジェクトの辞書版は非常に有用な `メタデータ` ですので、必ず含めるようにしてください。

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

データセットのログ例のように、`new_file` を同時に書き込んで `Artifact` に追加する代わりに、ファイルを1つのステップで書き込む（ここでは `torch.save`）こともでき、その後に `Artifact` に `add` する2ステップに分けることもできます。

> **Rule of 👍**: 重複を防ぐために、可能であれば `new_file` を使用しましょう。

#### ログされたモデル アーティファクトを使用する

データセットに `use_artifact` を呼び出すことができるように、`initialized_model` より別の `Run` で使用することができます。

今回は、`モデル`を `train` してみましょう。

詳細については、Colab の [PyTorchを使った W&B 計測](http://wandb.me/pytorch-colab) をご覧ください。

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

        # 各エポックでモデルを検証セットで評価する
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
            test_loss += F.cross_entropy(output, target, reduction='sum')  # バッチの損失を合計する
            pred = output.argmax(dim=1, keepdim=True)  # 最大 log-probability のインデックスを取得する
            correct += pred.eq(target.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)
    
    return test_loss, accuracy


def train_log(loss, example_ct, epoch):
    loss = float(loss)

    # ここで魔法が起こる
    wandb.log({"epoch": epoch, "train/loss": loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")
    

def test_log(loss, accuracy, example_ct, epoch):
    loss = float(loss)
    accuracy = float(accuracy)

    # ここで魔法が起こる
    wandb.log({"epoch": epoch, "validation/loss": loss, "validation/accuracy": accuracy}, step=example_ct)
    print(f"Loss/accuracy after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}/{accuracy:.3f}")
```

今回は、2つの別々な `Artifact` を生成する `Run` を行います。

最初の `Run` が `トレーニング` を終えると、`2番目` の `Run` は `test_dataset` のパフォーマンスを `evaluate` することで `trained-model` `Artifact` を消費します。

また、ネットワークが最も混乱している32の例、つまり `categorical_crossentropy` が最も高い例を取り出します。

これは、データセットやモデルに問題があるかどうかを診断する良い方法です。

```python
def evaluate(model, test_loader):
    """
    ## 訓練したモデルを評価する
    """

    loss, accuracy = test(model, test_loader)
    highest_losses, hardest_examples, true_labels, predictions = get_hardest_k_examples(model, test_loader.dataset)

    return loss, accuracy, highest_losses, hardest_examples, true_labels, predictions

def get_hardest_k_examples(model, testing_set, k=32):
    model.eval()

    loader = DataLoader(testing_set, 1, shuffle=False)

    # データセット内の各アイテムの損失と予測を取得する
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

これらのログ関数は新しい `Artifact` 機能を追加しないため、それについてコメントしません。これらの関数は単に `use`、`download`、そして `log` を行っています。

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