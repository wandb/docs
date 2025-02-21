---
title: Track models and datasets
menu:
  tutorials:
    identifier: ja-tutorials-artifacts
weight: 4
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Pipeline_Versioning_with_W&B_Artifacts.ipynb" >}}
この notebook では、W&B Artifacts を使用して ML の実験パイプラインを追跡する方法を紹介します。

[ビデオチュートリアル](http://tiny.cc/wb-artifacts-video)をご覧ください。

## Artifacts について

Artifacts は、ギリシャの[アンフォラ](https://en.wikipedia.org/wiki/Amphora)のように、
プロセスのアウトプットである生成物です。
ML において、最も重要な artifacts は _データセット_ と _モデル_ です。

そして、[ココロナドの十字架](https://indianajones.fandom.com/wiki/Cross_of_Coronado)のように、これらの重要な artifacts は博物館に収蔵されるべきです。
つまり、あなた、あなたの Team、そして ML コミュニティ全体がそれらから学べるように、カタログ化され、整理されるべきです。
結局のところ、トレーニングを追跡しない者は、それを繰り返す運命にあるのです。

Artifacts API を使用すると、次の図のように、W&B `Run` の出力として `Artifact` をログしたり、`Run` の入力として `Artifact` を使用したりできます。
トレーニング run はデータセットを入力として受け取り、モデルを生成します。
 
 {{< img src="/images/tutorials/artifacts-diagram.png" alt="" >}}

1 つの run が別の run の出力を入力として使用できるため、`Artifact` と `Run` は、`Artifact` と `Run` のノードを持つ有向グラフ (二部[DAG](https://en.wikipedia.org/wiki/Directed_acyclic_graph))を形成します。
そして、`Run` をそれが消費または生成する `Artifact` に接続する矢印を持ちます。

## Artifacts を使用してモデルとデータセットを追跡する

### インストールとインポート

Artifacts は、バージョン `0.9.2` 以降の Python ライブラリの一部です。

ML Python スタックのほとんどの部分と同様に、`pip` 経由で利用できます。


```python
# Compatible with wandb version 0.9.2+
!pip install wandb -qqq
!apt install tree
```


```python
import os
import wandb
```

### データセットのログ

まず、いくつかの Artifacts を定義しましょう。

この例は、PyTorch の
["Basic MNIST Example"](https://github.com/pytorch/examples/tree/master/mnist/)に基づいています。
しかし、[TensorFlow](http://wandb.me/artifacts-colab)や他の framework、
または純粋な Python で同じように行うことができました。

`Dataset` から始めます。
- パラメータを選択するための `train`ing セット、
- ハイパーパラメータを選択するための `validation` セット、
- 最終モデルを評価するための `test`ing セット

以下の最初のセルは、これらの 3 つのデータセットを定義します。


```python
import random 

import torch
import torchvision
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data parameters
num_classes = 10
input_shape = (1, 28, 28)

# drop slow mirror from list of MNIST mirrors
torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                      if not mirror.startswith("http://yann.lecun.com")]

def load(train_size=50_000):
    """
    # Load the data
    """

    # the data, split between train and test sets
    train = torchvision.datasets.MNIST("./", train=True, download=True)
    test = torchvision.datasets.MNIST("./", train=False, download=True)
    (x_train, y_train), (x_test, y_test) = (train.data, train.targets), (test.data, test.targets)

    # split off a validation set for hyperparameter tuning
    x_train, x_val = x_train[:train_size], x_train[train_size:]
    y_train, y_val = y_train[:train_size], y_train[train_size:]

    training_set = TensorDataset(x_train, y_train)
    validation_set = TensorDataset(x_val, y_val)
    test_set = TensorDataset(x_test, y_test)

    datasets = [training_set, validation_set, test_set]

    return datasets
```

これは、この例で繰り返されるパターンを設定します。
データを Artifacts としてログするコードは、
そのデータを生成するコードの周りにラップされます。
この場合、データを `load` するコードは、
データを `load_and_log` するコードから分離されています。

これは良い習慣です。

これらのデータセットを Artifacts としてログするには、
次のことが必要です。
1. `wandb.init` で `Run` を作成し (L4)、
2. データセットの `Artifact` を作成し (L10)、
3. 関連する `file` を保存してログします (L20、L23)。

以下のコードセルで例を確認し、
その後で詳細についてセクションを展開してください。


```python
def load_and_log():

    # 🚀 run を開始します。タイプを指定してラベルを付け、プロジェクトを指定してホームと呼ぶことができます。
    with wandb.init(project="artifacts-example", job_type="load-data") as run:
        
        datasets = load()  # データセットをロードするための個別のコード
        names = ["training", "validation", "test"]

        # 🏺 Artifact を作成します
        raw_data = wandb.Artifact(
            "mnist-raw", type="dataset",
            description="Raw MNIST dataset, split into train/val/test",
            metadata={"source": "torchvision.datasets.MNIST",
                      "sizes": [len(dataset) for dataset in datasets]})

        for name, data in zip(names, datasets):
            # 🐣 Artifact に新しいファイルを保存し、そのコンテンツに何かを書き込みます。
            with raw_data.new_file(name + ".pt", mode="wb") as file:
                x, y = data.tensors
                torch.save((x, y), file)

        # ✍️ Artifact を W&B に保存します。
        run.log_artifact(raw_data)

load_and_log()
```

#### `wandb.init`


`Artifact` を生成する `Run` を作成するときは、
それが属する `project` を記述する必要があります。

workflow によっては、
project は `car-that-drives-itself` ほど大きくすることも、
`iterative-architecture-experiment-117` ほど小さくすることもできます。

> **経験則**: 可能であれば、`Artifact` を共有するすべての `Run` を
単一の project 内に保持してください。これにより、物事がシンプルになりますが、
心配しないでください -- `Artifact` は project 間で移植可能です。

実行する可能性のあるさまざまな種類の job をすべて追跡するために、
`Run` を作成するときに `job_type` を指定すると便利です。
これにより、Artifacts のグラフが整理されます。

> **経験則**: `job_type` は記述的であり、パイプラインの単一のステップに対応する必要があります。ここでは、データの `load` とデータの `preprocess` を分離します。

#### `wandb.Artifact`


何かを `Artifact` としてログするには、最初に `Artifact` オブジェクトを作成する必要があります。

すべての `Artifact` には `name` があります。これは最初の引数で設定されます。

> **経験則**: `name` は記述的であり、覚えやすく入力しやすいものである必要があります --
ハイフンで区切られ、コード内の変数名に対応する名前を使用することをお勧めします。

また、`type` もあります。`Run` の `job_type` と同様に、
これは `Run` と `Artifact` のグラフを整理するために使用されます。

> **経験則**: `type` はシンプルである必要があります。
`mnist-data-YYYYMMDD` よりも `dataset` や `model` に近いものです。

`description` といくつかの `metadata` を dictionary としてアタッチすることもできます。
`metadata` は JSON にシリアライズ可能である必要があります。

> **経験則**: `metadata` は可能な限り記述的である必要があります。

#### `artifact.new_file` と `run.log_artifact`

`Artifact` オブジェクトを作成したら、それにファイルを追加する必要があります。

そのとおりです。_ファイル_ です。
`Artifact` はディレクトリーのように構成されており、
ファイルとサブディレクトリーがあります。

> **経験則**: 可能な場合は常に、`Artifact` のコンテンツを
複数のファイルに分割してください。これは、スケーリングするときに役立ちます。

`new_file` メソッドを使用して、
ファイルを同時に書き込み、`Artifact` にアタッチします。
以下では、`add_file` メソッドを使用します。
これにより、これら 2 つのステップが分離されます。

すべてのファイルを追加したら、[wandb.ai](https://wandb.ai)に `log_artifact` する必要があります。

出力にいくつかの URL が表示されます。
Run ページへの URL も含まれています。
そこでは、ログに記録された `Artifact` を含む、`Run` の結果を表示できます。

以下では、Run ページの他のコンポーネントをより有効に活用する例をいくつか示します。

### ログに記録されたデータセット Artifact の使用

博物館の artifacts とは異なり、W&B の `Artifact` は、
保存するだけでなく、_使用_ するように設計されています。

それがどのようなものか見てみましょう。

以下のセルは、raw データセットを入力として受け取り、
それを使用して `preprocess` されたデータセット (
`normalize` され、正しく整形されている)を生成するパイプラインステップを定義します。

コードの中核である `preprocess` を
`wandb` とのインターフェースとなるコードから分離していることに再び注意してください。


```python
def preprocess(dataset, normalize=True, expand_dims=True):
    """
    ## Prepare the data
    """
    x, y = dataset.tensors

    if normalize:
        # Scale images to the [0, 1] range
        x = x.type(torch.float32) / 255

    if expand_dims:
        # Make sure images have shape (1, 28, 28)
        x = torch.unsqueeze(x, 1)
    
    return TensorDataset(x, y)
```

次に、この `preprocess` ステップを `wandb.Artifact` ロギングで計測するコードを示します。

以下の例では、`Artifact` を `use` し、
これは新しいことです。
そして、それを `log` します。
これは最後のステップと同じです。
`Artifact` は `Run` の入力と出力の両方です。

新しい `job_type` である `preprocess-data` を使用して、
これが前のジョブとは異なる種類のジョブであることを明確にします。


```python
def preprocess_and_log(steps):

    with wandb.init(project="artifacts-example", job_type="preprocess-data") as run:

        processed_data = wandb.Artifact(
            "mnist-preprocess", type="dataset",
            description="Preprocessed MNIST dataset",
            metadata=steps)
         
        # ✔️ 使用する Artifact を宣言します
        raw_data_artifact = run.use_artifact('mnist-raw:latest')

        # 📥 必要に応じて、Artifact をダウンロードします
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

ここで注意すべきことの 1 つは、preprocessing の `steps` が
`metadata` として `preprocessed_data` と共に保存されることです。

実験の再現性を高めようとしている場合は、
多くの metadata をキャプチャすることをお勧めします。

また、データセットが「`large artifact`」であっても、
`download` ステップは 1 秒以内に完了します。

詳細については、以下の markdown セルを展開してください。


```python
steps = {"normalize": True,
         "expand_dims": True}

preprocess_and_log(steps)
```

#### `run.use_artifact`

これらのステップはより簡単です。コンシューマーは、`Artifact` の `name` に加えて、もう少しだけ知る必要があります。

その「もう少し」は、必要な `Artifact` の特定のバージョンの `alias` です。

デフォルトでは、最後にアップロードされたバージョンには `latest` のタグが付けられます。
それ以外の場合は、`v0`/`v1` などの古いバージョンを選択するか、`best` や `jit-script` などの独自のエイリアスを指定できます。
[Docker Hub](https://hub.docker.com/) タグと同様に、
エイリアスは `:` で名前から区切られます。
したがって、必要な `Artifact` は `mnist-raw:latest` です。

> **経験則**: エイリアスは短く簡潔にしてください。
`Artifact` が何らかのプロパティを満たす場合は、`latest` や `best` などのカスタム `alias` を使用します。

#### `artifact.download`

ここで、`download` 呼び出しについて心配しているかもしれません。
別のコピーをダウンロードすると、メモリへの負担が 2 倍になるのではないでしょうか?

ご心配なく。実際に何かをダウンロードする前に、
適切なバージョンがローカルで利用可能かどうかを確認します。
これには、[torrent](https://en.wikipedia.org/wiki/Torrent_file) と [`git` によるバージョン管理](https://blog.thoughtram.io/git/2014/11/18/the-anatomy-of-a-git-commit.html) の基盤となるのと同じテクノロジーであるハッシュが使用されます。

`Artifact` が作成およびログされると、
作業ディレクトリーの `artifacts` という名前のフォルダがサブディレクトリーでいっぱいになり始めます。
各 `Artifact` に 1 つずつです。
`!tree artifacts` でその内容を確認してください。


```python
!tree artifacts
```

#### Artifacts ページ

`Artifact` をログして使用したので、
Run ページの [Artifacts] タブを確認しましょう。

`wandb` 出力から Run ページの URL に移動し、
左側のサイドバーから [Artifacts] タブを選択します。
(これはデータベースアイコンが付いたもので、
3 つのホッケーパックが互いに積み重ねられているように見えます)。

[**入力 Artifacts**] テーブルまたは [**出力 Artifacts**] テーブルの行をクリックし、
次にタブ ([**概要**]、[**Metadata**]) を確認して、
`Artifact` についてログに記録されたすべてを確認します。

特に [**グラフ表示**] が気に入っています。
デフォルトでは、グラフは
`Artifact` の `type` と
`Run` の `job_type` を 2 種類のノードとして表示し、
消費と生成を表す矢印を表示します。

### モデルのログ

これで `Artifact` の API がどのように機能するかを確認するのに十分ですが、
この例をパイプラインの最後まで実行して、
`Artifact` が ML workflow をどのように改善できるかを見てみましょう。

この最初のセルは、PyTorch で DNN `model` を構築します。これは非常に単純な ConvNet です。

最初は `model` を初期化するだけで、トレーニングは行いません。
そうすれば、他のすべてを一定に保ちながら、トレーニングを繰り返すことができます。


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

ここでは、W&B を使用して run を追跡しているため、
[`wandb.config`](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-config/Configs_in_W%26B.ipynb)
オブジェクトを使用してすべてのハイパーパラメータを保存しています。

その `config` オブジェクトの `dict`ionary バージョンは非常に役立つ `metadata` であるため、必ず含めてください。


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
        # ➕ Artifact にファイルを追加する別の方法
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


データセットのロギング例のように、`new_file` を同時に書き込んで `Artifact` に追加する代わりに、
1 つのステップでファイル (ここでは `torch.save`) を書き込み、
別のステップで `add` で `Artifact` に書き込むこともできます。

> **経験則**: 重複を防ぐために、可能な場合は `new_file` を使用してください。

#### ログに記録されたモデル Artifact の使用

`dataset` で `use_artifact` を呼び出すことができるのと同じように、
`initialized_model` で `use_artifact` を呼び出して、
別の `Run` で使用できます。

今回は、`model` を `train` してみましょう。

詳細については、
[PyTorch で W&B を計測する](http://wandb.me/pytorch-colab)に関する Colab を参照してください。


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

        # evaluate the model on the validation set at each epoch
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
            test_loss += F.cross_entropy(output, target, reduction='sum')  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)
    
    return test_loss, accuracy


def train_log(loss, example_ct, epoch):
    loss = float(loss)

    # where the magic happens
    wandb.log({"epoch": epoch, "train/loss": loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")
    

def test_log(loss, accuracy, example_ct, epoch):
    loss = float(loss)
    accuracy = float(accuracy)

    # where the magic happens
    wandb.log({"epoch": epoch, "validation/loss": loss, "validation/accuracy": accuracy}, step=example_ct)
    print(f"Loss/accuracy after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}/{accuracy:.3f}")
```

今回は、2 つの別々の `Artifact` を生成する `Run` を実行します。

最初の run が `model` の `train`ing を完了すると、
`2 番目` は `trained-model` `Artifact` を消費し、
`test_dataset` でそのパフォーマンスを `evaluate` します。

また、ネットワークが最も混乱する 32 個の例を抽出します。
これは、`categorical_crossentropy` が最も高い例です。

これは、データセットとモデルの問題を診断するのに適した方法です。


```python
def evaluate(model, test_loader):
    """
    ## Evaluate the trained model
    """

    loss, accuracy = test(model, test_loader)
    highest_losses, hardest_examples, true_labels, predictions = get_hardest_k_examples(model, test_loader.dataset)

    return loss, accuracy, highest_losses, hardest_examples, true_labels, predictions

def get_hardest_k_examples(model, testing_set, k=32):
    model.eval()

    loader = DataLoader(testing_set, 1, shuffle=False)

    # get the losses and predictions for each item in the dataset
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

これらのロギング関数は新しい `Artifact` 機能を追加しないため、
コメントしません。
`Artifact` を `use`ing、`download`ing、
および `log`ing しているだけです。


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