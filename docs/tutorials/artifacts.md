
# Track models and datasets

[**Try in a Colab Notebook here →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Pipeline_Versioning_with_W&B_Artifacts.ipynb)

このノートブックでは、W&B Artifactsを使用してML実験パイプラインを追跡する方法を紹介します。

### 一緒に進めましょう [ビデオチュートリアル](http://tiny.cc/wb-artifacts-video)!

### 🤔 Artifactsとは何か、そしてなぜ気にするべきなのか？

「アーティファクト」とは、ギリシャの[アンフォラ 🏺](https://en.wikipedia.org/wiki/Amphora)のように生成されたオブジェクト、つまりプロセスの出力物です。
MLにおいて最も重要なアーティファクトは、 _データセット_ と _モデル_ です。

そして、[コロナドの十字架](https://indianajones.fandom.com/wiki/Cross_of_Coronado)のように、これらの重要なアーティファクトは博物館に収めるべきです！
つまり、あなたやチーム、そして広範なMLコミュニティがそれらから学べるようにカタログ化し整理すべきなのです。
結局のところ、トレーニングを追跡しない者はそれを繰り返す運命にあります。

Artifacts APIを使用すると、W&B `Run` の出力として `Artifact` をログに記録したり、以下の図のように `Run` への入力として `Artifact` を使用することができます。

 ![](https://gblobscdn.gitbook.com/assets%2F-Lqya5RvLedGEWPhtkjU%2F-M94QAXA-oJmE6q07_iT%2F-M94QJCXLeePzH1p_fW1%2Fsimple%20artifact%20diagram%202.png?alt=media&token=94bc438a-bd3b-414d-a4e4-aa4f6f359f21)

1つのrunが別のrunの出力を入力として使用できるため、ArtifactsとRunsは、有向グラフ、実際には二部グラフ [DAG](https://en.wikipedia.org/wiki/Directed_acyclic_graph) を形成します。
`Artifact` と `Run` のノードがあり、消費または生成する `Run` への矢印で接続されています。

# 0️⃣ インストールとインポート

ArtifactsはPythonライブラリの一部であり、バージョン `0.9.2` から利用可能です。

ML Pythonスタックのほとんどの部分と同様に、`pip` でインストールできます。

```python
# wandbバージョン0.9.2以上と互換性があります
!pip install wandb -qqq
!apt install tree
```

```python
import os
import wandb
```

# 1️⃣ データセットのログ記録

まず、いくつかのArtifactsを定義しましょう。

この例はPyTorchの
["Basic MNIST Example"](https://github.com/pytorch/examples/tree/master/mnist/)
に基づいていますが、[TensorFlow](http://wandb.me/artifacts-colab) や他のフレームワーク、
または純粋なPythonでも簡単に行うことができます。

まず `データセット` を用意します:
- パラメータを選択するための `トレーニング`セット
- ハイパーパラメーターを選択するための `検証`セット
- 最終モデルを評価するための `テスト`セット

以下のセルでこれら3つのデータセットを定義します。

```python
import random 

import torch
import torchvision
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm

# 再現性のある振る舞いを保証
torch.backends.cudnn.deterministic = True
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# デバイスの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# データのパラメータ
num_classes = 10
input_shape = (1, 28, 28)

# MNISTのミラーリストから遅いミラーを削除
torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                      if not mirror.startswith("http://yann.lecun.com")]

def load(train_size=50_000):
    """
    # データのロード
    """

    # データをトレーニングセットとテストセットに分割
    train = torchvision.datasets.MNIST("./", train=True, download=True)
    test = torchvision.datasets.MNIST("./", train=False, download=True)
    (x_train, y_train), (x_test, y_test) = (train.data, train.targets), (test.data, test.targets)

    # ハイパーパラメータチューニング用に検証セットを分割
    x_train, x_val = x_train[:train_size], x_train[train_size:]
    y_train, y_val = y_train[:train_size], y_train[train_size:]

    training_set = TensorDataset(x_train, y_train)
    validation_set = TensorDataset(x_val, y_val)
    test_set = TensorDataset(x_test, y_test)

    datasets = [training_set, validation_set, test_set]

    return datasets
```

この例では、データをArtifactとしてログするコードをデータを生成するコードの周りにラップします。
この場合、`load`データ用のコードが`load_and_log`データ用のコードと分けられています。

これは良いプラクティスです！

これらのデータセットをArtifactsとしてログするには、
1. `wandb.init` で `Run` を作成 (L4)
2. データセットの `Artifact` を作成 (L10)
3. 関連 `file` を保存してログ (L20, L23)

以下のコードセル例を確認し、その後のセクションで詳細を確認します。

```python
def load_and_log():

    # 🚀 runを開始し、ラベル付けするためのタイプと、ホームとなるプロジェクトを指定
    with wandb.init(project="artifacts-example", job_type="load-data") as run:
        
        datasets = load()  # データセットをロードするための別のコード
        names = ["training", "validation", "test"]

        # 🏺 Artifactを作成
        raw_data = wandb.Artifact(
            "mnist-raw", type="dataset",
            description="Raw MNIST dataset, split into train/val/test",
            metadata={"source": "torchvision.datasets.MNIST",
                      "sizes": [len(dataset) for dataset in datasets]})

        for name, data in zip(names, datasets):
            # 🐣 Artifactに新しいファイルを保存し、その内容に何かを書き込む
            with raw_data.new_file(name + ".pt", mode="wb") as file:
                x, y = data.tensors
                torch.save((x, y), file)

        # ✍️ W&Bにartifactを保存
        run.log_artifact(raw_data)

load_and_log()
```

### 🚀 `wandb.init`

Artifactsを生成するrunを作成する際には、どの`project`に属しているかを指定する必要があります。

ワークフローによっては、プロジェクトは `car-that-drives-itself` のように大規模なものから `iterative-architecture-experiment-117` のように小規模なものまであります。

> **Rule of 👍**: `Artifact`を共有するすべての `Run` を単一のプロジェクト内に保つことができれば、シンプルに保つことができますが、心配いりません -- `Artifact`はプロジェクト間でも移動可能です！

行う可能性のあるさまざまな種類のジョブを追跡するために、`Runs`を作成するときに`job_type`を指定することが役立ちます。
これによりArtifactsのグラフが整然と保たれます。

> **Rule of 👍**: `job_type`は記述的であり、パイプラインの単一ステップに対応するべきです。ここでは、データの `load` とデータの `preprocess` を分けています。

### 🏺 `wandb.Artifact`

何かを `Artifact` としてログするためには、まず `Artifact` オブジェクトを作成する必要があります。

すべての `Artifact` は `name` を持ちます -- 最初の引数がこれを設定します。

> **Rule of 👍**: `name`は記述的でありながら覚えやすく入力しやすいものであるべきです -- ハイフンで区切られた名前を使用し、コード内の変数名に対応させることをお勧めします。

また、`type`も持ちます。これは `Run` の `job_type` と同様に、`Run` と `Artifact` のグラフを整理するために使用されます。

> **Rule of 👍**: `type`はシンプルであるべきです： `dataset` や `model` のように。 `mnist-data-YYYYMMDD` よりもシンプルに。

さらに、`description`と辞書形式の`metadata`を追加することもできます。 `metadata`はJSONにシリアライズ可能である必要があります。

> **Rule of 👍**: `metadata`は可能な限り詳細にするべきです。

### 🐣 `artifact.new_file` と ✍️ `run.log_artifact`

`Artifact`オブジェクトを作成したら、ファイルを追加する必要があります。

その通りです： _ファイル_（複数形）です。
`Artifact`はディレクトリのように構造化されており、ファイルやサブディレクトリを持ちます。

> **Rule of 👍**: 可能であれば、`Artifact`の内容を複数のファイルに分けます。拡張する際に役立ちます！

`new_file`メソッドを使用して、ファイルを書き込むと同時に `Artifact`に添付します。
次に、両方のステップを分ける `add_file` メソッドを使用します。

すべてのファイルを追加したら、`log_artifact` を使用して [wandb.ai](https://wandb.ai) にログします。

出力にいくつかのURLが表示され、RunページのURLも含まれます。
そこで `Run` の結果、ログされたすべての `Artifact` を確認できます。

以下に詳細を示す例を見てみましょう。

# 2️⃣ ログされたデータセットArtifactを使用する

W&Bの `Artifact`は博物館のアーティファクトとは異なり、 _使用するため_ に設計されています。

その具体的な使用方法を見てみましょう。

以下のセルでは、元のデータセットを取り込み、`preprocess`されたデータセットに変換するパイプラインステップを定義しています。
`normalize`され、正しく形状が整えられます。

ここでも、主要なコード部分 `preprocess` と、`wandb` とインターフェースするコードを分けています。

```python
def preprocess(dataset, normalize=True, expand_dims=True):
    """
    ## データの準備
    """
    x, y = dataset.tensors

    if normalize:
        # 画像を [0, 1] の範囲にスケーリング
        x = x.type(torch.float32) / 255

    if expand_dims:
        # 画像の形状を (1, 28, 28) にする
        x = torch.unsqueeze(x, 1)
    
    return TensorDataset(x, y)
```

次に、この `preprocess` ステップを `wandb.Artifact` のログ記録で補強するコードを示します。

以下の例では、新たに `Artifact` を `use` し（新しいステップ）、同様に `log` しています（前のステップと同じです）。
`Artifact` は `Run` の入力と出力の両方です！

新しい `job_type`、`preprocess-data` を使用して、これが以前のジョブとは異なる種類のジョブであることを明確にします。

```python
def preprocess_and_log(steps):

    with wandb.init(project="artifacts-example", job_type="preprocess-data") as run:

        processed_data = wandb.Artifact(
            "mnist-preprocess", type="dataset",
            description="Preprocessed MNIST dataset",
            metadata=steps)
         
        # ✔️ 使用するartifactを宣言
        raw_data_artifact = run.use_artifact('mnist-raw:latest')

        # 📥 必要に応じてartifactをダウンロード
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

ここで注目すべき点は、前処理の `steps` が `preprocessed_data` に `metadata` として保存されていることです。

実験を再現可能にしようとする場合、豊富なメタデータをキャプチャすることは良い考えです！

さらに、データセットが「`large artifact`」であっても、`download`ステップは1秒未満で完了します。

詳細は以下のマークダウンセルを展開してください。

```python
steps = {"normalize": True,
         "expand_dims": True}

preprocess_and_log(steps)
```

### ✔️ `run.use_artifact`

これらのステップはシンプルです。消費者は `Artifact` の `name` を知っていれば十分です。

その「少し」は、必要な `Artifact` の特定のバージョンの `alias` です。

デフォルトでは、最後にアップロードされたバージョンが `latest` とタグ付けされます。
それ以外の場合、`v0`/`v1` などで古いバージョンを選択したり、`best` や `jit-script` などの独自のエイリアスを提供できます。

Docker Hub](https://hub.docker.com/) のタグと同様に、エイリアスは `:` で名前と分離されます。
したがって、必要な `Artifact` は `mnist-raw:latest` です。

> **Rule of 👍**: エイリアスは短く簡潔に保ちます。特定のプロパティを満たす `Artifact` が必要な場合は、独自の `alias` 例えば `latest` や `best` を使用します。

### 📥 `artifact.download`

次に、`download` 呼び出しについて心配しているかもしれません。
もう一度コピーをダウンロードすると、メモリの負担が2倍になりますか？

心配しないでください。実際に何かをダウンロードする前に、正しいバージョンがローカルにあるかどうかを確認します。
これは [torrenting](https://en.wikipedia.org/wiki/Torrent_file) や [バージョン管理 `git`](https://blog.thoughtram.io/git/2014/11/18/the-anatomy-of-a-git-commit.html) に基づく技術を使用しています：ハッシュ化です。

`Artifact` が作成されログされると、作業ディレクトリ内に `artifacts` フォルダーが作成され、
各 `Artifact` ごとにサブディレクトリが埋まっていきます。
`!tree artifacts` を使ってその内容を確認します。

```python
!tree artifacts
```

### 🌐 [wandb.ai](https://wandb.ai) 上の Artifactsページ

`Artifact` をログし、使用したので、RunページのArtifactsタブを確認してみましょう。

`wandb` の出力からRunページのURLに移動し、
左サイドバーから「Artifacts」タブを選択します（これは
3つのホッケーパックが積み重なったようなデータベースのアイコンです）。

「入力Artifacts」テーブルまたは「出力Artifacts」テーブルのいずれかの行をクリックし、次に「概要」「メタデータ」タブを確認して、
`Artifact` についてログされたすべての情報を確認します。

特に「グラフビュー」が気に入っています。
デフォルトでは、`Artifact` の `type` と `Run` の `job_type` が2つのノードタイプとして表示され、
消費と生産を表す矢印が表示されます。

# 3️⃣ モデルをログする

これで `Artifact` の API がどのように機能するかを確認できましたが、MLワークフローがどのように `Artifact` によって改善されるかを見るために、この例をパイプラインの最後まで続けてみましょう。

この最初のセルでは、PyTorch でシンプルな ConvNet を用いた DNN `model` を構築します。

最初に `model` を初期化するだけで、トレーニングは行いません。
この方法により、他のすべてが一定に保たれる中でもトレーニングを繰り返すことができます。

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

ここでは、W&B を使用して run を追跡し、
[`wandb.config`](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-config/Configs_in_W%26B.ipynb)
オブジェクトを使用してハイパーパラメーターをすべて保存します。

`config` オブジェクトの `dict` 形式は非常に便利な `metadata` の一部であるため、必ず含めてください。

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
        # ➕ Artifactにファイルを追加する別の方法
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

### ➕ `artifact.add_file`

データセットのログ例のように、同時に `new_file` を書きそれを `Artifact` に追加するのではなく、
ファイルを1つのステップで書き込んで (`torch.save` など)、
その後にそれを `Artifact` に `add` することもできます。

> **👍の原則**: 重複を避けるために可能な限り `new_file` を使用してください。

# 4️⃣ ログされたモデル Artifact を使用する

データセット に `use_artifact` を呼び出せるように、
別の `Run` で `initialized_model` を使用するためにこれを呼び出せます。

今回は、`model` をトレーニングしましょう。

詳細は、
[instrumenting W&B with PyTorch](http://wandb.me/pytorch-colab)
の Colab をチェックしてください。

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

        # 各エポックで検証セットに対してモデルを評価する
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
            test_loss += F.cross_entropy(output, target, reduction='sum')  # バッチの損失を合計
            pred = output.argmax(dim=1, keepdim=True)  # 最大のログ確率のインデックスを取得
            correct += pred.eq(target.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)
    
    return test_loss, accuracy


def train_log(loss, example_ct, epoch):
    loss = float(loss)

    # マジックが起こる場所
    wandb.log({"epoch": epoch, "train/loss": loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")
    

def test_log(loss, accuracy, example_ct, epoch):
    loss = float(loss)
    accuracy = float(accuracy)

    # マジックが起こる場所
    wandb.log({"epoch": epoch, "validation/loss": loss, "validation/accuracy": accuracy}, step=example_ct)
    print(f"Loss/accuracy after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}/{accuracy:.3f}")
```

今回は、2つの別々の `Artifact` を生成する `Run`  を実行します。

最初の `Run` が `model` のトレーニングを終了すると、
`second` は `trained-model` `Artifact` を消費し、
そのパフォーマンスを `test_dataset` 上で `evaluate` します。

また、ネットワークが最も混乱する32の例を抽出します --
この例では、`categorical_crossentropy` が最も高くなります。

これはデータセットとモデルの問題を診断する良い方法です！

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

    # データセット内の各アイテムの損失と予測を取得
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

これらのログ関数は新しい `Artifact` 機能を追加するものではないため、
注釈はしません。やっているのはただ
`use`、`download`、
および `log` です。

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

### 🔁 グラフビュー

`Artifact` の `type` を変更したことに注意してください:
これらの `Run` は `dataset` ではなく `model` を使用しました。
`model` を生成する `Run` は、
Artifacts ページのグラフビューで `dataset` を生成する `Run` から分離されます。

確認してみてください！ いつもの通り、Run ページに移動し、
左のサイドバーから「Artifacts」タブを選び、
`Artifact` を選択してから「Graph View」タブをクリックします。

### 💣 グラフの爆発

「Explode」と書かれたボタンに気づくかもしれません。それはクリックしないでください。それをクリックすると、W&B HQの著者の机の下に小爆弾が仕掛けられることになります！

冗談です。グラフをずっと穏やかな方法で「爆発」させます:
`Artifact` と `Run` は単一のインスタンスレベルで分離されます。
ノードは `dataset` と `load-data` ではなく、`dataset:mnist-raw:v1` と `load-data:sunny-smoke-1` などとなります。

これは、パイプラインへの完全な洞察を提供し、
ログされたメトリクス、metadata などがすべて
手の届くところにあります --
ログする内容以外に制限はありません。

# 次に何をしますか？
次のチュートリアルでは、モデルの変更を通信し、W&B Models を使用してモデル開発ライフサイクルを管理する方法を学びます:

## 👉 [Track Model Development Lifecycle](models)