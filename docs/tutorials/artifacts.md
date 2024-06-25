
# Track models and datasets

[**Try in a Colab Notebook here →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Pipeline_Versioning_with_W&B_Artifacts.ipynb)

このノートブックでは、W&B Artifacts を使用してあなたのML実験のパイプラインを追跡する方法を紹介します。

### [ビデオチュートリアル](http://tiny.cc/wb-artifacts-video)を見て、一緒に学びましょう！

### 🤔 Artifacts とは何か、そしてなぜ気にする必要があるのか?

「アーティファクト」とは、ギリシャの[アンフォラ 🏺](https://en.wikipedia.org/wiki/Amphora) のように、プロセスの結果として生み出されるオブジェクトのことです。MLでは、最も重要なアーティファクトは _datasets_ と _models_ です。

そして、[コロナドの十字架](https://indianajones.fandom.com/wiki/Cross_of_Coronado)のように、これらの重要なアーティファクトは博物館に所蔵されるべきです！つまり、あなたやチーム、そしてMLコミュニティ全体がそれらから学べるよう、これらはカタログ化され、整理されるべきなのです。トレーニングを追跡しない人は、同じことを繰り返す運命にあるのです。

私たちの Artifacts API を使用すると、W&B の `Run` の出力として `Artifact` をログに記録したり、`Run` の入力として `Artifact` を使用したりできます。以下の図のように、トレーニングの run がデータセットを取り込み、モデルを生成します。

 ![](https://gblobscdn.gitbook.com/assets%2F-Lqya5RvLedGEWPhtkjU%2F-M94QAXA-oJmE6q07_iT%2F-M94QJCXLeePzH1p_fW1%2Fsimple%20artifact%20diagram%202.png?alt=media&token=94bc438a-bd3b-414d-a4e4-aa4f6f359f21)

one run が別の run の出力を入力として使用できるため、Artifacts と Runs は一緒に有向グラフ、実際には二部 [DAG](https://en.wikipedia.org/wiki/Directed_acyclic_graph) を形成します！ノードとして Artifact と Run があり、Run が消費または生成する Artifact に矢印で接続されます。

# 0️⃣ インストールとインポート

Artifacts は、バージョン `0.9.2` からの私たちの Python ライブラリの一部です。

ML の Python スタックのほとんどの部分と同様に、`pip` を通じて利用可能です。

```python
# Compatible with wandb version 0.9.2+
!pip install wandb -qqq
!apt install tree
```

```python
import os
import wandb
```

# 1️⃣ データセットをログする

まず、いくつかの Artifacts を定義しましょう。

この例は PyTorch の[「Basic MNIST Example」](https://github.com/pytorch/examples/tree/master/mnist) に基づいていますが、[TensorFlow](http://wandb.me/artifacts-colab) や他のフレームワーク、純粋な Python でも同様に実行できます。

データセットから始めます：
- パラメータを選択するための `train` セット
- ハイパーパラメータを選択するための `validation` セット
- 最終モデルを評価するための `test` セット

以下のセルでは、これら3つのデータセットを定義します。

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

    # データをトレインセットとテストセットに分割する
    train = torchvision.datasets.MNIST("./", train=True, download=True)
    test = torchvision.datasets.MNIST("./", train=False, download=True)
    (x_train, y_train), (x_test, y_test) = (train.data, train.targets), (test.data, test.targets)

    # ハイパーパラメータチューニング用に検証セットに分割
    x_train, x_val = x_train[:train_size], x_train[train_size:]
    y_train, y_val = y_train[:train_size], y_train[train_size:]

    training_set = TensorDataset(x_train, y_train)
    validation_set = TensorDataset(x_val, y_val)
    test_set = TensorDataset(x_test, y_test)

    datasets = [training_set, validation_set, test_set]

    return datasets
```

これは、この例で繰り返されるパターンを設定します：
データをアーティファクトとしてログするためのコードが、データを生成するコードの周りにラップされています。
この場合、データを`load`するコードが`load_and_log`するコードから分けられています。

これは良いプラクティスです！

これらのデータセットをアーティファクトとしてログするには、
1. `Run` を `wandb.init` で作成し (L4)
2. データセットの `Artifact` を作成し (L10)
3. 関連する `file` を保存し、ログに記録します (L20, L23)。

以下のコードセルを確認してから、さらに詳細な部分は後で展開します。

```python
def load_and_log():

    # 🚀 runを開始し、それにラベルをつけて、プロジェクトを指定します
    with wandb.init(project="artifacts-example", job_type="load-data") as run:
        
        datasets = load()  # データセットを読み込むコードを分けておく
        names = ["training", "validation", "test"]

        # 🏺 Artifact を作成
        raw_data = wandb.Artifact(
            "mnist-raw", type="dataset",
            description="Raw MNIST dataset, split into train/val/test",
            metadata={"source": "torchvision.datasets.MNIST",
                      "sizes": [len(dataset) for dataset in datasets]})

        for name, data in zip(names, datasets):
            # 🐣 新しいファイルをアーティファクトに保存し、内容を書き込みます。
            with raw_data.new_file(name + ".pt", mode="wb") as file:
                x, y = data.tensors
                torch.save((x, y), file)

        # ✍️ アーティファクトを W&B に保存します。
        run.log_artifact(raw_data)

load_and_log()
```

### 🚀 `wandb.init`

`Artifact`s を生成する `Run` を行うときには、それが属する `project` を明示する必要があります。

ワークフローに応じて、プロジェクトは `car-that-drives-itself` のように大きなものから `iterative-architecture-experiment-117` のように小さなものまでさまざまです。

> **👍 のルール**: Artifacts を共有するすべての `Run` を一つのプロジェクト内に保持できればシンプルです。ただし心配は無用です — Artifacts はプロジェクト間で移動可能です！

実行する様々なジョブを追跡するために `job_type` を指定することをお勧めします。これにより Artifacts のグラフが整理されます。

> **👍 のルール**: `job_type` は説明的であり、パイプラインの一つのステップに対応するべきです。ここでは、データの `load` と `preprocess` を分けています。

### 🏺 `wandb.Artifact`

何かを `Artifact` としてログするためには、まず `Artifact` オブジェクトを作成する必要があります。

すべての `Artifact` には `name` があります。これは第一引数で設定されます。

> **👍 のルール**: `name` は説明的であり覚えやすく、打ち込みやすいものにするべきです。私たちはハイフンで区切られた名前を使用するのが好きで、それはコード内の変数名に対応するものです。

また、`type` もあります。これは `Run` のための `job_type` のように、`Run` と `Artifact` のグラフを整理するために使用されます。

> **👍 のルール**: `type` はシンプルであるべきです： `dataset` や `model` のように。

さらに `description` といくつかの `metadata` (辞書として) を添付できます。`metadata` は JSON でシリアライズ可能である必要があります。

> **👍 のルール**: `metadata` はできるだけ具体的にするべきです。

### 🐣 `artifact.new_file` と ✍️ `run.log_artifact`

一度 `Artifact` オブジェクトを作成すると、そこにファイルを追加する必要があります。

複数の _ファイル_ を追加することができます。`Artifact` はディレクトリのように構造化されており、ファイルやサブディレクトリを持つことができます。

> **👍 のルール**: 可能な場合は、`Artifact` の内容を複数のファイルに分割しておくと、将来的にスケールする際に役立ちます。

`new_file` メソッドを使用して、同時にファイルを書き込み、それを `Artifact` に添付します。以下では、これら2つのステップを分ける `add_file` メソッドを使用します。

すべてのファイルを追加し終わったら、`log_artifact` で [wandb.ai](https://wandb.ai) にログを残します。

出力にいくつかのURLが表示されることに気づくでしょう。その中には Run のページのURLもあります。そのページでは、`Run` の結果やログに記録された `Artifact` を確認できます。

以下では `Run` ページの他のコンポーネントのより良い使用例を見ていきます。

# 2️⃣ ログされたデータセットアーティファクトを使用する

博物館のアーティファクトとは異なり、W&B の `Artifact` は _使う_ ために設計されています。ただ保管するだけではありません。

それがどのようになるか見てみましょう。

以下のセルは生のデータセットを使用し、それを `preprocess` して `normalize` されたデータセットを生成するパイプラインのステップを定義しています。

再度、コードの核となる部分を `preprocess` から`wandb` とインターフェースするコードに分けていることに注目してください。

```python
def preprocess(dataset, normalize=True, expand_dims=True):
    """
    ## データの準備
    """
    x, y = dataset.tensors

    if normalize:
        # 画像を [0, 1] の範囲にスケール
        x = x.type(torch.float32) / 255

    if expand_dims:
        # 画像が (1, 28, 28) の形であることを確認
        x = torch.unsqueeze(x, 1)
    
    return TensorDataset(x, y)
```

次に、`wandb.Artifact` ログを使用してこの `preprocess` ステップを記録するコードです。

以下の例では、新しいアーティファクトの `use` と、以前と同じ `log` を実行しています。アーティファクトは `Run` の入力および出力の両方で使用されます！

新しい `job_type`、`preprocess-data` を使用して、これが前のものとは異なる種類のジョブであることを明示しています。

```python
def preprocess_and_log(steps):

    with wandb.init(project="artifacts-example", job_type="preprocess-data") as run:

        processed_data = wandb.Artifact(
            "mnist-preprocess", type="dataset",
            description="Preprocessed MNIST dataset",
            metadata=steps)
         
        # ✔️ 使用するアーティファクトを宣言
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

この例では、`preprocessed_data` のステップが `metadata` として保存されています。

実験を再現可能にしようとしている場合、多くのメタデータを取得することは良い考えです！

さらに、データセットが「大きなアーティファクト」にもかかわらず、`download` ステップは1秒未満で完了します。

詳細を見るには、下のマークダウンセルを展開してください。

```python
steps = {"normalize": True,
         "expand_dims": True}

preprocess_and_log(steps)
```

### ✔️ `run.use_artifact`

これらのステップはシンプルです。コンシューマーは `Artifact` の `name` ともう少しだけを知っている必要があります。

その「もう少し」とは、使用したい特定バージョンの `Artifact` の `alias` です。

デフォルトでは、最後にアップロードされたバージョンには `latest` タグが付けられます。それ以外にも `v0`/`v1`などで古いバージョンを選択できたり、`best` や `jit-script` などの独自のエイリアスを指定できます。[Docker Hub](https://hub.docker.com/) タグのように、エイリアスは名前と `:` で区切られます。したがって、必要な `Artifact` は `mnist-raw:latest` です。

> **👍 のルール**: エイリアスは短くシンプルに保ちます。`alias` は `Artifact` が満たすプロパティに応じてカスタムエイリアスを使用します。

### 📥 `artifact.download`

`download` コールについて心配しているかもしれません。もう一つのコピーをダウンロードすると、メモリへの負担が倍増しませんか？

心配ありません。実際に何かをダウンロードする前に、正しいバージョンがローカルにあるか確認します。これは [torrenting](https://en.wikipedia.org/wiki/Torrent_file) や [地下鐸バージョン管理システム `git`](https://blog.thoughtram.io/git/2014/11/18/the-anatomy-of-a-git-commit.html) の下にある技術と同じです：ハッシュ化です。

`Artifact` が作成されログされると、作業ディレクトリに `artifacts` フォルダが現れ、各 `Artifact` のサブディレクトリが一つずつ埋まっていきます。以下のコマンドで内容を確認してください：`!tree artifacts`:

```python
!tree artifacts
```

### 🌐 [wandb.ai](https://wandb.ai) の Artifacts ページ

`Artifact` を記録して使用したので、Run ページの Artifacts タブを確認しましょう。

`wandb` 出力から Run ページの URL に移動し、左サイドバーから「Artifacts」タブを選択します。（これは、3つのホッケーのパックが積み重なっているようなデータベースアイコンです）。

「Input Artifacts」テーブルか「Output Artifacts」テーブルの行をクリックし、次に「Overview」か「Metadata」タブをチェックして、記録されたすべてを確認します。

特に「Graph View」がオススメです。デフォルトでは、`Artifact` の `type` と `Run` の `job_type` を2種類のノードとして表示し、矢印で消費と生成を表現します。

# 3️⃣ モデルのログ

これは API の使い方を理解するためには十分ですが、このパイプラインを最後まで追いかけ、`Artifact` がどのように ML ワークフローを改善できるかを見てみましょう。

最初のセルでは、非常にシンプルな ConvNet を持つ DNN モデルを PyTorch で構築します。

最初にモデル