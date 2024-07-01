# Track models and datasets

[**Try in a Colab Notebook here →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Pipeline_Versioning_with_W&B_Artifacts.ipynb)

このノートブックでは、W&B Artifacts を使ってMLの実験パイプラインをトラッキングする方法を紹介します。

### 一緒に進めましょう！ [ビデオチュートリアル](http://tiny.cc/wb-artifacts-video)

### 🤔 Artifactsとは何か、そしてなぜ重要なのか？

「artifact」は、ギリシャの[アンフォラ 🏺](https://en.wikipedia.org/wiki/Amphora)のような作られたオブジェクトで、プロセスの出力物です。
MLでは、最も重要なartifactは _datasets_ と _models_ です。

そして、[コロナドの十字架](https://indianajones.fandom.com/wiki/Cross_of_Coronado)のように、これらの重要なartifactは博物館に保存されるべきです！
つまり、これらはカタログ化され、整理されるべきです。
あなたやチーム、そして広範なMLコミュニティがそれらから学べるようにするためです。
トレーニングをトラッキングしない者はそれを繰り返す運命にあるのです。

Artifacts APIを使用すると、`Artifact`をW&Bの`Run`の出力としてログに記録したり、`Run`の入力として`Artifact`を使用したりできます。このダイアグラムのように、トレーニングrunがデータセットを入力として取り、モデルを生成します。

![](https://gblobscdn.gitbook.com/assets%2F-Lqya5RvLedGEWPhtkjU%2F-M94QAXA-oJmE6q07_iT%2F-M94QJCXLeePzH1p_fW1%2Fsimple%20artifact%20diagram%202.png?alt=media&token=94bc438a-bd3b-414d-a4e4-aa4f6f359f21)

1つのrunが他のrunの出力を入力として使用できるため、ArtifactsとRunsは一緒に有向グラフ（実際には二部[ダイアグラム](https://en.wikipedia.org/wiki/Directed_acyclic_graph)）を形成します。ノードは`Artifact`と`Run`のものであり、それぞれ消費や生成される`Artifact`に対してRunsが矢印でつながっています。

# 0️⃣ インストールとインポート

Artifactsは、バージョン`0.9.2`からPythonライブラリの一部です。

他の多くのML Pythonスタックと同様に、`pip`経由で利用可能です。

```python
# wandb バージョン0.9.2以降に対応
!pip install wandb -qqq
!apt install tree
```

```python
import os
import wandb
```

# 1️⃣ データセットをログする

まず、いくつかのArtifactsを定義しましょう。

この例はPyTorchの
["Basic MNIST Example"](https://github.com/pytorch/examples/tree/master/mnist)
に基づいていますが、[TensorFlow](http://wandb.me/artifacts-colab)や他の任意のフレームワーク、または純粋なPythonでも簡単に実行できます。

まずは、`Dataset`を以下のように定義します：
- `train`セットはパラメータを選ぶため、
- `validation`セットはハイパーパラメータを選ぶため、
- `test`セットは最終モデルを評価するため

以下のセルでは、これら3つのデータセットを定義します。

```python
import random 

import torch
import torchvision
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm

# 決定的な振る舞いを保証
torch.backends.cudnn.deterministic = True
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# デバイスの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# データパラメータ
num_classes = 10
input_shape = (1, 28, 28)

# 遅いミラーをドロップ
torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                      if not mirror.startswith("http://yann.lecun.com")]

def load(train_size=50_000):
    """
    # データをロード
    """

    # データをトレーニングおよびテストセットに分割
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

このコードは、データを生成するコードの周りにデータをArtifactとしてログするコードがラップされています。
この場合、データを`load`するコードと、データを`load_and_log`するコードが分離されています。

これは良い実践です！

これらのデータセットをArtifactとしてログするには、
1. `wandb.init`で`Run`を作成 (L4)
2. データセット用の`Artifact`を作成 (L10)
3. 関連する`file`を保存してログ (L20, L23)

以下のコードセルの例を見て、その後のセクションを展開して詳細を確認してください。

```python
def load_and_log():

    # 🚀 runをスタートし、タイプとプロジェクトでラベル付けする
    with wandb.init(project="artifacts-example", job_type="load-data") as run:
        
        datasets = load()  # データセットのロードのための分離コード
        names = ["training", "validation", "test"]

        # 🏺 Artifactを作成
        raw_data = wandb.Artifact(
            "mnist-raw", type="dataset",
            description="生のMNISTデータセットをトレーニング/検証/テストに分割",
            metadata={"source": "torchvision.datasets.MNIST",
                      "sizes": [len(dataset) for dataset in datasets]})

        for name, data in zip(names, datasets):
            # 🐣 アーティファクトに新しいファイルを保存し、その内容に何かを書き込みます。
            with raw_data.new_file(name + ".pt", mode="wb") as file:
                x, y = data.tensors
                torch.save((x, y), file)

        # ✍️ アーティファクトをW&Bに保存します。
        run.log_artifact(raw_data)

load_and_log()
```

### 🚀 `wandb.init`

Artifactを生成する`Run`を作成する際、どの`project`に属するかを指定する必要があります。

ワークフローに応じて、プロジェクトは`car-that-drives-itself`のように大きいものから`iterative-architecture-experiment-117`のように小さいものまで様々です。

> **Rule of 👍**: すべての`Run`が共有する`Artifact`を1つのプロジェクト内に保持できる場合は、それをシンプルに保ちましょう。ただし、心配しないでください──`Artifact`はプロジェクト間で移動可能です！

様々なジョブを管理するために、`Run`を作成する際に`job_type`を提供するのが便利です。
これによりArtifactのグラフが整理されます。

> **Rule of 👍**: `job_type`は説明的であり、パイプラインの単一ステップに対応するようにしましょう。ここでは、データの`load`とデータの`preprocess`を分けています。

### 🏺 `wandb.Artifact`

何かを`Artifact`としてログするには、最初に`Artifact`オブジェクトを作成する必要があります。

すべての`Artifact`には`name`があります──これが最初の引数で設定されます。

> **Rule of 👍**: `name`は説明的で、覚えやすく入力しやすいものにしましょう──私たちはハイフンで区切られた名前をコード内の変数名に対応させるのが好きです。

また、`type`もあります。これは、`Run`の`job_type`と同様に`Run`と`Artifact`のグラフを整理するために使用されます。

> **Rule of 👍**: `type`はシンプルであるべきです：`dataset`や`model`のようなもの。

さらに、辞書形式の`description`と`metadata`を追加することもできます。
`metadata`はJSONにシリアライズ可能である必要があります。

> **Rule of 👍**: `metadata`は可能な限り詳細にします。

### 🐣 `artifact.new_file`と✍️ `run.log_artifact`

`Artifact`オブジェクトを作成したら、ファイルを追加する必要があります。

その通り、_ファイル_ の _s_ があります。
`Artifact`はディレクトリのように構造化されており、ファイルやサブディレクトリを持ちます。

> **Rule of 👍**: できる限り、`Artifact`の内容を複数のファイルに分割してください。これによりスケールする際に便利です！

`new_file`メソッドを使用してファイルを書くと同時に`Artifact`に添付することができます。
以下では、これらの2つのステップを分ける`add_file`メソッドを使用します。

すべてのファイルを追加したら、`log_artifact`を使用して[wandb.ai](https://wandb.ai)に保存する必要があります。

出力にはいくつかのURLが表示されますが、その中にはRunページのURLも含まれます。
そこでは、`Run`の結果を確認でき、ログされた`Artifact`も表示されます。

以下はその他のRunページコンポーネントをより活用する例です。

# 2️⃣ ログされたデータセットArtifactを使用する

W&Bの`Artifact`は、博物館のアーティファクトとは違い、単に保存されるだけでなく、_使用_されることを想定しています。

それがどのようなものか見てみましょう。

以下のセルでは、生のデータセットを取り込み、それを`preprocess`edデータセットとして出力するパイプラインステップを定義します。
これは`normalize`され、正しく整形されたデータです。

ここでも、`wandb`とインターフェイスするコードから、重要なコードである`preprocess`を分離しています。

```python
def preprocess(dataset, normalize=True, expand_dims=True):
    """
    ## データを準備する
    """
    x, y = dataset.tensors

    if normalize:
        # 画像を[0, 1]の範囲にスケールする
        x = x.type(torch.float32) / 255

    if expand_dims:
        # 画像が(1, 28, 28)の形状を持っていることを確認する
        x = torch.unsqueeze(x, 1)
    
    return TensorDataset(x, y)
```

次に、`preprocess`ステップを`wandb.Artifact`のログを使用して器用に扱うコードです。

以下の例では、`Artifact`を`use`すること（これは新しい）、そしてそれをログすること（これは前のステップと同じ）を行っています。
`Artifact`は`Run`の入力と出力の両方です！

新しい`job_type`である`preprocess-data`を使用して、これが前の`load`とは異なる種類のジョブであることを明確にします。

```python
def preprocess_and_log(steps):

    with wandb.init(project="artifacts-example", job_type="preprocess-data") as run:

        processed_data = wandb.Artifact(
            "mnist-preprocess", type="dataset",
            description="前処理済みのMNISTデータセット",
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

ここで気づくべき点は、前処理の`steps`が`preprocessed_data`の`metadata`として保存されることです。

実験を再現可能にしたい場合、たくさんのメタデータをキャプチャすることは良い考えです！

また、私たちのデータセットが「`large artifact`」であっても、`download`ステップは1秒未満で完了します。

詳細は以下のセルを展開して確認してください。

```python
steps = {"normalize": True,
         "expand_dims": True}

preprocess_and_log(steps)
```

### ✔️ `run.use_artifact`

これらのステップはシンプルです。消費者は単に`Artifact`の`name`を知る必要があります、ちょっとだけ他の情報も。

その「ちょっとだけ他の情報」とは、求める特定の`Artifact`バージョンの`alias`です。

デフォルトでは、最後にアップロードされたバージョンには`latest`がタグ付けされます。
それ以外の場合、`v0`や`v1`などの古いバージョンを選択することもできますし、
`best`や`jit-script`など自分のエイリアスを提供することもできます。
ちょうど[Docker Hub](https://hub.docker.com/)のタグのように、
エイリアスは名前から`:`で分離されています。

> **Rule of 👍**: エイリアスは短くシンプルに保ちましょう。
カスタムエイリアス`latest`や`best`を使用して、特定のプロパティを満たすArtifactを選びましょう。

### 📥 `artifact.download`

`download`呼び出しが心配かもしれません。
別のコピーをダウンロードすると、メモリの負担が倍増するのでは？

心配しないでください。実際に何かをダウンロードする前に、
適切なバージョンがローカルに存在するか確認します。
これには[トレント](https://en.wikipedia.org/wiki/Torrent_file)や[`git`によるバージョン管理](https://blog.thoughtram.io/git/2014/11/18/the-anatomy-of-a-git-commit.html)の技術が使われています。

Artifactが作成・ログされると、作業ディレクトリの中に`artifacts`というフォルダができ、
それぞれのArtifact用のサブディレクトリが埋め尽くされます。
以下のコマンドでその内容をチェックしてください：

```python
!tree artifacts
```

### 🌐 [wandb.ai](https://wandb.ai)のArtifactsページ

Artifactをログして使用したら、RunページのArtifactsタブをチェックしましょう。

`wandb`の出力に表示されるRunページURLに移動し、
左サイドバーから「Artifacts」タブを選択します
（これはデータベースアイコンのもので、ホッケーのパックが3つ積み重なったように見えます）。

"Input Artifacts"テーブルや"Output Artifacts"テーブルの行をクリックし、
タブ（"Overview"、"Metadata"）をチェックして、Artifactに関するすべてのログを確認してください。

特に「Graph View」が好きです。
デフォルトでは、`Artifact`のタイプと`Run`の`job_type`が2種類のノードで示され、
矢印は消費と生成を表します。

# 3️⃣ モデルをログする

これで`Artifact` APIの動作がわかりましたが、この例をパイプラインの終わりまで続けて、
`Artifact`がMLワークフローをどのように改善できるか見てみましょう。

この最初のセルは、PyTorchでDNN `model`を構築します──ごくシンプルなConvNetです。

まず、モデルを初期化し、トレーニングはしません。
これにより、他の部分を固定してトレーニングを繰り返すことができます。

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

ここでは、W&Bを使用してrunをトラッキングし、
[`wandb.config`](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-config/Configs_in_W%26B.ipynb)
オブジェクトを使用してすべてのハイパーパラメータを保存しています。

その`config`オブジェクトの`dict`バージョンは非常に便利なメタデータなので、必ず含めてください！

```python
def build_model_and_log(config):
    with wandb.init(project="artifacts-example", job_type="initialize", config=config) as run:
        config = wandb.config
        
        model = ConvNet(**config)

        model_artifact = wandb.Artifact(
            "convnet", type="model",
            description="シンプルなAlexNetスタイルのCNN",
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

データセットのログ例のように、`new_file`を一度に書き込み、`Artifact`に追加する代わりに、
ファイルを1ステップで書く（ここでは、`torch.save`）ことができ、その後で`Artifact`に追加できます。

> **Rule of 👍**: 重複を防ぐために、可能であれば`new_file`を使用しましょう。

# 4️⃣ ログされたモデルArtifactを使用する

データセットに対して`use_artifact`を呼び出せるように、
`initialized_model`に対しても同様に呼び出して、別の`Run`で使用できます。

今回はモデルを`train`しましょう。

詳細は、私たちのColabノートブック
[W&BとPyTorchを使ったインスツルメント](http://wandb.me/pytorch-colab)をご覧ください。

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

        # 各エポックで検証セット上のモデルを評価
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
            pred = output.argmax(dim=1, keepdim=True)  # 最大対数確率のインデックスを取得
            correct += pred.eq(target.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)
    
    return test_loss, accuracy


def train_log(loss, example_ct, epoch):
    loss = float(loss)

    # マジックが起きる場所
    wandb.log({"epoch": epoch, "train/loss": loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")
    

def test_log(loss, accuracy, example_ct, epoch):
    loss = float(loss)
    accuracy = float(accuracy)

    # マジックが起きる場所
    wandb.log({"epoch": epoch, "validation/loss": loss, "validation/accuracy": accuracy}, step=example_ct)
    print(f"Loss/accuracy after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}/{accuracy:.3f}")
```

今回は、Artifactを生成する2つの別々の`Run`を行います。

まず、最初のrunがモデルをトレーニングすると、
次のrunが`trained-model`Artifactを使用して、
テストデータセットでパフォーマンスを評価します。

また、ネットワークが最もうまくいかなかった32例──`categorical_crossentropy`が最も高い例──を抽出します。

これは、データセットおよびモデルの問題を診断するための良い方法です！

```python
def evaluate(model, test_loader):
    """
    ## 訓練済みモデルを評価
    """

    loss, accuracy = test(model, test_loader)
    highest_losses, hardest_examples, true_labels, predictions = get_hardest_k_examples(model, test_loader.dataset)

    return loss, accuracy, highest_losses, hardest_examples, true_labels, predictions

def get_hardest_k_examples(model, testing_set, k=32):
    model.eval()

    loader = DataLoader(testing_set, 1, shuffle=False)

    # データセット内の各項目の損失と予測を取得
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

これらのログ関数は、新しい`Artifact`機能を追加するわけではありません。
単に使用し、ダウンロードし、Artifactをログします。

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
            description="トレーニング済みNNモデル",
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

`Artifact`の`type`を変更しました：
これらの`Run`sは`dataset`ではなく`model`を使用しました。
グラフビューのArtifactsページでは、
`model`を生成する`Run`sは`dataset`を生成する`Run`sとは区別されます。

ぜひチェックしてみてください！以前と同じように、Runページにアクセスし、
左サイドバーから「Artifacts」タブを選び、
`Artifact`を選択して「Graph View」タブをクリックしてください。

### 💣 グラフの展開

「Explode」とラベル付けされたボタンがあるのに気づいたかもしれません。それをクリックしないでください、それは私の机の下に小さな爆弾をセットします…というのは冗談です。そのボタンをクリックすると、グラフがより優しく展開されます：
`Artifact`や`Run`sは、個々のインスタンスのレベルで分離されます、タイプではなく、例えば、
ノードは`dataset`や`load-data`ではなく、`dataset:mnist-raw:v1`や`load-data:sunny-smoke-1`などです。

これにより、ログされたメトリクス、メタデータなどすべてが手の届く範囲にあり、
パイプラインの全貌を完全に把握できます──あなたが私たちにログを選んだものだけが限界です。

# 次は何をする？

