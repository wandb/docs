


# Track models and datasets

[**Try in a Colab Notebook here →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Pipeline_Versioning_with_W&B_Artifacts.ipynb)

このノートブックでは、W&B Artifacts を使って機械学習の実験パイプラインを追跡する方法を紹介します。

### [動画チュートリアル](http://tiny.cc/wb-artifacts-video)に従ってください！

### 🤔 Artifacts とそれが重要な理由は？

「artifact」（アーティファクト）とは、ギリシャの [amphora 🏺](https://en.wikipedia.org/wiki/Amphora) のように、プロセスの結果として生み出されるオブジェクトです。機械学習において、最も重要なアーティファクトは _datasets_ と _models_ です。

そして、[Cross of Coronado](https://indianajones.fandom.com/wiki/Cross_of_Coronado) のように、これらの重要なアーティファクトは博物館に保存されるべきです！つまり、カタログ化され組織化されるべきです。
そうすれば、あなたやチーム、さらには機械学習コミュニティ全体がそれから学ぶことができます。
結局のところ、トレーニングを追跡しない人はそれを繰り返す宿命にあります。

Artifacts APIを使用すると、W&B `Run`の出力として`Artifact`をログに記録したり、`Run`の入力として`Artifact`を使用したりできます。この図のように、トレーニングの実行がデータセットを取り込み、モデルを生成します。
 
 ![](https://gblobscdn.gitbook.com/assets%2F-Lqya5RvLedGEWPhtkjU%2F-M94QAXA-oJmE6q07_iT%2F-M94QJCXLeePzH1p_fW1%2Fsimple%20artifact%20diagram%202.png?alt=media&token=94bc438a-bd3b-414d-a4e4-aa4f6f359f21)

1つのRunが別のRunの出力を入力として使用できるため、ArtifactsとRunsは有向グラフ -- 実際には二部グラフ [DAG](https://en.wikipedia.org/wiki/Directed_acyclic_graph) ！-- を形成し、`Artifact`と`Run`のノードと、消費または生成する`Artifact`への矢印で構成されます。

# 0️⃣ インストールとインポート

Artifactsはバージョン `0.9.2` からPythonライブラリの一部です。

通常の機械学習のPythonスタックの一部として、`pip`で利用可能です。

```python
# Compatible with wandb version 0.9.2+
!pip install wandb -qqq
!apt install tree
```

```python
import os
import wandb
```

# 1️⃣ データセットをログに記録

まず、いくつかのArtifactsを定義しましょう。

この例はPyTorchの ["Basic MNIST Example"](https://github.com/pytorch/examples/tree/master/mnist/) を元にしていますが、[TensorFlow](http://wandb.me/artifacts-colab) や他のどのフレームワークでも同様に実行できます。

最初に `Dataset`s を用意します：
- `train`ing セット（パラメータを選ぶため）
- `validation` セット（ハイパーパラメータを選ぶため）
- `test`ing セット（最終モデルを評価するため）

最初のセルでは、これら3つのデータセットを定義します。

```python
import random 

import torch
import torchvision
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm

# デターミニスティックな振る舞いを保証する
torch.backends.cudnn.deterministic = True
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# デバイスの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# データパラメータ
num_classes = 10
input_shape = (1, 28, 28)

# MNISTミラーのリストから遅いミラーを削除
torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                      if not mirror.startswith("http://yann.lecun.com")]

def load(train_size=50_000):
    """
    # データを読み込む
    """

    # データをトレーニングセットとテストセットに分割する
    train = torchvision.datasets.MNIST("./", train=True, download=True)
    test = torchvision.datasets.MNIST("./", train=False, download=True)
    (x_train, y_train), (x_test, y_test) = (train.data, train.targets), (test.data, test.targets)

    # ハイパーパラメータ調整のために検証セットを分割する
    x_train, x_val = x_train[:train_size], x_train[train_size:]
    y_train, y_val = y_train[:train_size], y_train[train_size:]

    training_set = TensorDataset(x_train, y_train)
    validation_set = TensorDataset(x_val, y_val)
    test_set = TensorDataset(x_test, y_test)

    datasets = [training_set, validation_set, test_set]

    return datasets
```

これは、この例で繰り返し見るパターンを設定します：
データをアーティファクトとしてログに記録するコードが、そのデータを生成するコードの周りにラップされています。この場合、データを`load`するコードはデータを`load_and_log`するコードから分離されています。

これは良い実践です！

これらのデータセットをArtifactsとしてログに記録するためには、次のことを行う必要があります：
1. `wandb.init`で`Run`を作成する（L4）
2. データセット用の`Artifact`を作成する（L10）
3. 関連する`file`sを保存してログに記録する（L20, L23）

下のコードセルの例を確認してから、その後のセクションを展開して詳細を確認してください。

```python
def load_and_log():

    # 🚀 Runを開始し、それにラベルを付けるタイプとプロジェクトを指定する
    with wandb.init(project="artifacts-example", job_type="load-data") as run:
        
        datasets = load()  # データセットを読み込むためのコードを分離
        names = ["training", "validation", "test"]

        # 🏺 Artifactを作成する
        raw_data = wandb.Artifact(
            "mnist-raw", type="dataset",
            description="Raw MNIST dataset, split into train/val/test",
            metadata={"source": "torchvision.datasets.MNIST",
                      "sizes": [len(dataset) for dataset in datasets]})

        for name, data in zip(names, datasets):
            # 🐣 Artifactに新しいファイルを格納し、その内容を書き込む
            with raw_data.new_file(name + ".pt", mode="wb") as file:
                x, y = data.tensors
                torch.save((x, y), file)

        # ✍️ ArtifactをW&Bに保存する
        run.log_artifact(raw_data)

load_and_log()
```

### 🚀 `wandb.init`

`Artifact`sを生成する`Run`を作成する際には、どのプロジェクトに属するかを記載する必要があります。

ワークフローによっては、プロジェクトは `car-that-drives-itself`のように大きいかもしれませんし、`iterative-architecture-experiment-117`のように小さいかもしれません。

> **👍ルール**: 可能であれば、`Artifact`sを共有するすべての`Run`sを単一のプロジェクト内に保持してください。これによりシンプルになりますが、心配しないでください - `Artifact`sはプロジェクト間で移動可能です！

実行する可能性のある異なる種類のジョブを追跡するために、`Runs`を作成するときに`job_type`を提供するのが役立ちます。これにより、Artifactsのグラフが整然と整理されます。

> **👍ルール**: `job_type`は記述的であり、パイプラインの単一のステップに対応するべきです。ここでは、`load`データの読み込みと`preprocess`データの前処理を分けています。

### 🏺 `wandb.Artifact`

何かを`Artifact`としてログに記録するには、まず`Artifact`オブジェクトを作成する必要があります。

すべての`Artifact`には`name`があります - これが最初の引数で設定されるものです。

> **👍ルール**: `name`は記述的で、覚えやすく入力しやすいものであるべきです --
私たちは、ハイフンで区切られ、コード内の変数名に対応する名前を使用するのが好きです。

また、`type`もあります。`Run`sの`job_type`sと同じように、`Run`sと`Artifact`sのグラフを整理するために使用されます。

> **👍ルール**: `type`はシンプルであるべきです：
`dataset`や`model`のようなもの
`mnist-data-YYYYMMDD`よりも。

辞書として、`description`といくつかの`metadata`を添付することもできます。`metadata`はJSONにシリアライズできる必要があります。

> **👍ルール**: `metadata`は可能な限り詳細にするべきです。

### 🐣 `artifact.new_file` と ✍️ `run.log_artifact`

`Artifact`オブジェクトを作成したら、ファイルを追加する必要があります。

ファイルと複数形で正しく読みました。
`Artifact`sはディレクトリー構造を持っており、
ファイルとサブディレクトリを持ちます。

> **👍ルール**: 可能であれば、1つの`Artifact`の内容を複数のファイルに分割することが理想です。これにより、スケールアウトが必要になった時にも役立ちます！

`new_file`メソッドを使用して、
ファイルを書き込むと同時にそれを`Artifact`に添付します。
以下では、`add_file`メソッドを使用し、
これらの2つのステップを分離します。

すべてのファイルを追加したら、`log_artifact`を使用して [wandb.ai](https://wandb.ai) に送信する必要があります。

出力の中にいくつかのURLが表示されることに気づくでしょう。この中にはRunページのものも含まれます。
そこでは、`Run`の結果を見ることができ、ログに記録された`Artifact`sも含まれます。

以下で、Runページの他のコンポーネントをよりよく活用する例を見ることができます。

# 2️⃣ ログに記録されたデータセットアーティファクトを使用する

`W&B`の`Artifact`sは、美術館のアーティファクトと異なり、ただ保管されるのではなく、_使用される_ために設計されています。

どのように見えるか見てみましょう。

以下のセルでは、未処理のデータセットを取り込み、
それを使用して`preprocess`されたデータセットを生成するパイプラインステップを定義しています：
`normalize`dや正しい形状に整形されたものです。

再び、`preprocess`ステップの主要部分のコードが
`wandb`とのインターフェースを取るコードから分離されていることに注目してください。

```python
def preprocess(dataset, normalize=True, expand_dims=True):
    """
    ## データの準備
    """
    x, y = dataset.tensors

    if normalize:
        # 画像を[0, 1]の範囲にスケールする
        x = x.type(torch.float32) / 255

    if expand_dims:
        # 画像が (1, 28, 28) という形状を持つようにする
        x = torch.unsqueeze(x, 1)
    
    return TensorDataset(x, y)
```

今度は、`wandb.Artifact` ログを使用した `preprocess` ステップを操作するコードです。

以下の例では、`Artifact`を`use`するだけでなく、
ログに記録する部分も同じです。
`Artifact`sは`Run`sの入力と出力の両方です！

新しい`job_type`、`preprocess-data`を使用します。これは前述のジョブとは異なる種類のジョブであることを明確にしています。

```python
def preprocess_and_log(steps):

    with wandb.init(project="artifacts-example", job_type="preprocess-data") as run:

        processed_data = wandb.Artifact(
            "mnist-preprocess", type="dataset",
            description="Preprocessed MNIST dataset",
            metadata=steps)
         
        # ✔️ 使用するartifactを宣言する
        raw_data_artifact = run.use_artifact('mnist-raw:latest')

        # 📥 必要に応じてartifactをダウンロードする
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

ここで注目すべきことの1つは、前処理の`steps`が`preprocessed_data`として`metadata`と一緒に保存されることです。

実験を再現可能にしようとする場合、多くのメタデータをキャプチャすることは良いアイデアです！

また、データセットが「`large artifact`」であっても、`download`ステップは1秒未満で行われます。

詳細については、以下のマークダウンセルを展開してください。

```python
steps = {"normalize": True,
         "expand_dims": True}

preprocess_and_log(steps)
```

### ✔️ `run.use_artifact`

これらのステップはよりシンプルです。消費者は`Artifact`の`name`と少しの追加情報を知っていればいいのです。

その「少しの追加情報」というのは、特定のバージョンの`Artifact`の`alias`です。

デフォルトでは、最後にアップロードされたバージョンが`latest`とタグ付けされます。
そうでなければ、`v0`/`v1`などの古いバージョンを選ぶことができます。
また、自分のエイリアス、例えば`best`や`jit-script`などを提供することもできます。
ちょうど[Docker Hub](https://hub.docker.com/)のタグのように、
エイリアスは名前から`:`で区切られます。
私たちが欲しい`Artifact`は`mnist-raw:latest`です。

> **👍ルール**: エイリアスは短く、シンプルに保つ。`Artifact`を満たす特定のプロパティを持つものを指定する場合には、カスタム`エイリアス`を使用する。

### 📥 `artifact.download`

Now, `download`呼び出しについて気になるかもしれません。
別のコピーをダウンロードすると、メモリ負担が2倍になりませんか？

心配はいりません。実際に何かをダウンロードする前に、
正しいバージョンがローカルで利用可能かどうかを確認します。
これは[Torrenting](https://en.wikipedia.org/wiki/Torrent_file)と [Gitのバージョン管理](https://blog.thoughtram.io/git/2014/11/18/the-anatomy-of-a-git-commit.html)の背後にある技術を使用します：ハッシュ化です。

`Artifact`sが作成され、ログに記録されると、
作業ディレクトリ内の`artifacts`というフォルダが満たされ始め、
各`Artifact`に1つのサブディレクトリが作られます。
その内容を`!tree artifacts`で確認してください。

```python
!tree artifacts
```

### 🌐 [wandb.ai](https://wandb.ai) のArtifactsページ

Artifactをログに記録し、使用したので、
RunページのArtifactsタブを確認しましょう。

`wandb`出力のRunページURLに移動し、
左側のサイドバーから「Artifacts」タブを選択します
（データベースアイコンのあるもので、
ホッケーのパックが3つ積み重なっているように見えます）。

「入力アーティファクト」テーブルの行をクリックするか、
「出力アーティファクト」テーブルの行をクリックし、
その後の「概要」タブ、「メタデータ」タブを確認して、
`Artifact`についてログに記録されたすべての内容を確認してください。

特