
# モデルとデータセットのトラッキング

[**Colabノートブックで試す →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Pipeline_Versioning_with_W&B_Artifacts.ipynb)

このノートブックでは、W&B Artifactsを使用してML実験の開発フローをトラッキングする方法を紹介します。

### [ビデオチュートリアル](http://tiny.cc/wb-artifacts-video)に沿って進めてください！

### 🤔 Artifactsとは何で、なぜ重要なのか？

「artifact」は、ギリシャの[アンフォラ🏺](https://en.wikipedia.org/wiki/Amphora)のように、プロセスの出力物として生成されたオブジェクトです。MLにおいて最も重要なartifactは、_データセット_と_モデル_です。

そして、[コロナドの十字架](https://indianajones.fandom.com/wiki/Cross_of_Coronado)のように、これらの重要なartifactは博物館に保管されるべきです！つまり、これらはカタログ化され組織化されるべきで、あなたやあなたのチーム、そして広範なMLコミュニティがそれらから学べるようにすべきです。
トレーニングをトラックしない人は、それを繰り返す運命にあるのです。

私たちのArtifacts APIを使用すると、この図のように、W&Bの`Run`の出力として`Artifact`をログに記録したり、`Run`の入力として`Artifact`を使用したりできます。

 ![](https://gblobscdn.gitbook.com/assets%2F-Lqya5RvLedGEWPhtkjU%2F-M94QAXA-oJmE6q07_iT%2F-M94QJCXLeePzH1p_fW1%2Fsimple%20artifact%20diagram%202.png?alt=media&token=94bc438a-bd3b-414d-a4e4-aa4f6f359f21)

1つのrunが他のrunの出力を入力として使用できるため、ArtifactsとRunsは、`Artifact`と`Run`のノードを持ち、それらが消費または生成する`Artifact`に接続する矢印がある、実際には二部有向非巡回グラフ（[DAG](https://en.wikipedia.org/wiki/Directed_acyclic_graph)）を形成します。

# 0️⃣ インストールとインポート

Artifactsは、バージョン`0.9.2`以上のPythonライブラリの一部です。

多くのML Pythonスタックの一部のように、`pip`経由で利用できます。

```python
# wandbバージョン0.9.2+に対応
!pip install wandb -qqq
!apt install tree
```

```python
import os
import wandb
```

# 1️⃣ データセットのログ

まず、いくつかのArtifactsを定義しましょう。

この例は、PyTorchの[「Basic MNIST Example」](https://github.com/pytorch/examples/tree/master/mnist/)に基づいていますが、同じように[TensorFlow](http://wandb.me/artifacts-colab)や他のフレームワーク、あるいは純粋なPythonでも同じことができます。

最初に`データセット`を用意します：
- パラメーターを選ぶための`train`ingセット
- ハイパーパラメーターを選ぶための`validation`セット
- 最終モデルを評価するための`test`ingセット

以下のセルでは、これら3つのデータセットを定義します。

```python
import random 

import torch
import torchvision
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm

# 確定的な振る舞いを保証
torch.backends.cudnn.deterministic = True
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# デバイス設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# データパラメータ
num_classes = 10
input_shape = (1, 28, 28)

# MNISTミラリストから遅いミラーを削除
torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                      if not mirror.startswith("http://yann.lecun.com")]

def load(train_size=50_000):
    """
    タデータをロードする
    """

    # データをトレーニングセットとテストセットに分割
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

ここでは、データをArtifactとしてログに記録するコードが、データを生成するコードを包むパターンを設定します。この場合、データを`load`するコードは、データを`load_and_log`するコードから分離されています。

これは良いプラクティスです！

これらのデータセットをArtifactsとしてログに記録するには、以下が必要です：
1. `wandb.init`で`Run`を開始する（L4）
2. データセットのための`Artifact`を作成する（L10）
3. 関連する`file`を保存し、ログに記録する（L20, L23）

以下のコードセルの例を確認して、その後のセクションを展開して詳しい説明を見てください。

```python
def load_and_log():

    # 🚀 タイプとホームプロジェクトをラベルにしてrunを開始
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
            # 🐣 新しいファイルをartifactに保存し、その内容を書き込む
            with raw_data.new_file(name + ".pt", mode="wb") as file:
                x, y = data.tensors
                torch.save((x, y), file)

        # ✍️ ArtifactをW&Bに保存
        run.log_artifact(raw_data)

load_and_log()
```

### 🚀 `wandb.init`

Artifactsを生成する`Run`を作成する際、その`Run`がどの`project`に所属しているかを指定する必要があります。

ワークフローに応じて、プロジェクトは`car-that-drives-itself`のように大きいものから、`iterative-architecture-experiment-117`のように小さいものまで様々です。

> **Rule of 👍**: 可能であれば、共有する`Artifact`を持つすべての`Run`を1つのプロジェクト内に保持してください。これにより、シンプルに保つことができます。ただし、心配しないでください。`Artifact`はプロジェクト間でポータブルです！

異なる種類のジョブをトラックするのに役立つため、`Run`を作成する際に`job_type`を指定することが有益です。これにより、Artifactsのグラフが整然と保たれます。

> **Rule of 👍**: `job_type`は詳細であり、パイプラインの単一ステップに対応するべきです。ここでは、データの`load`とデータの`preprocess`を分離します。

### 🏺 `wandb.Artifact`

`sArtifact`として何かをログに記録するためには、まず`Artifact`オブジェクトを作成する必要があります。

すべての`Artifact`には`name`があります。これは最初の引数で設定されます。

> **Rule of 👍**: `name`は説明的でありながら、覚えやすくタイプしやすいものにしましょう。コード内の変数名に対応するハイフンで区切られた名前を使用することをお勧めします。

また、`type`もあります。これは`Run`の`job_type`sと同様に、`Run`と`Artifact`のグラフを組織するのに使用されます。

> **Rule of 👍**: `type`はシンプルにしてください。`dataset`や`model`のようなもので、`mnist-data-YYYYMMDD`のようなものではありません。

さらに、`description`や`metadata`も辞書として追加できます。`metadata`はJSONにシリアライズ可能である必要があります。

> **Rule of 👍**: `metadata`はできるだけ詳細にしてください。

### 🐣 `artifact.new_file` と ✍️ `run.log_artifact`

`Artifact`オブジェクトを作成したら、ファイルを追加する必要があります。

複数のファイルを追加する必要がある場合、`new_file`メソッドを使用します。これにより、ファイルを同時に書き込んで`Artifact`に追加します。以下では、`add_file`メソッドを使用します。これは、これらのステップを分離します。

すべてのファイルを追加したら、`log_artifact`を使用して[wandb.ai](https://wandb.ai)にログを記録します。

出力には、`Run`ページのURLなどのURLが表示されます。`Run`の結果やログに記録された`Artifact`などを見ることができます。

続いての例では、`Run`ページの他のコンポーネントをさらに効果的に利用する例を紹介します。

# 2️⃣ ログに記録されたデータセットArtifactの使用

博物館のartifactと異なり、W&B内の`Artifact`は保管されるだけでなく、_使用される_ことを目的としています。

具体的に見てみましょう。

以下のセルでは、RAWデータセットを読み込み、それを`preprocess`されたデータセット、つまり`normalize`され正しく形状が整えられたデータセットに変換するパイプラインステップを定義します。

ここでも、`preprocess`を担当するコードと、`wandb`と連携するコードを分離しています。

```python
def preprocess(dataset, normalize=True, expand_dims=True):
    """
    データの準備
    """
    x, y = dataset.tensors

    if normalize:
        # 画像を[0, 1]の範囲にスケーリング
        x = x.type(torch.float32) / 255

    if expand_dims:
        # 画像の形状を(1, 28, 28)に確保
        x = torch.unsqueeze(x, 1)
    
    return TensorDataset(x, y)
```

以下は、`wandb.Artifact`を使用してこの`preprocess`ステップを記録するコードです。

ここでは、新しい`job_type`である`preprocess-data`を使用して、前のステップとは異なる種類のジョブであることを明確にしています。

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

ここで注意すべきことは、前処理の`steps`が`preprocessed_data`の`metadata`として保存されていることです。

実験の再現性を高めるためには、多くのメタデータをキャプチャすることが良策ですね！

また、データセットは「大きなartifact」であっても、`download`ステップは1秒もかかりません。

下のマークダウンセルを展開して、詳細を確認してください。

```python
steps = {"normalize": True,
         "expand_dims": True}

preprocess_and_log(steps)
```

### ✔️ `run.use_artifact`

これらのステップはシンプルです。使用者は`Artifact`の`name`を知っているだけで十分です。ただし、もう少しだけ情報が必要です。

その「もう少し」の部分は、`Artifact`のバージョンの`alias`です。

デフォルトでは、最後にアップロードされたバージョンは`latest`タグが付けられます。その他に、`v0`や`v1`などで古いバージョンを選択することもできますし、カスタムのエイリアス（例：`best`や`jit-script`）を使用することもできます。エイリアスは名前の後ろに`:`で区切られますので、必要な`Artifact`は`mnist-raw:latest`です。

> **Rule of 👍**: エイリアスは短く分かりやすいものに保ちましょう。特定の属性を満たす`Artifact`にはカスタム`alias`（例：`latest`や`best`）を使いましょう。

### 📥 `artifact.download`

`download`コールについて心配しているかもしれません。
もう一度コピーをダウンロードすることで、メモリの負担が倍増しないでしょうか？

心配しないでください。何かを実際にダウンロードする前に、正しいバージョンがローカルにあるかどうかをチェックします。これは、[トレント](https://en.wikipedia.org/wiki/Torrent_file)や[gitによるバージョン管理](https://blog.thoughtram.io/git/2014/11/18/the-anatomy-of-a-git-commit.html)の基礎となる技術を使用しています：ハッシングです。

`Artifact`が作成されログに記録されると、作業ディレクトリ内に`artifacts`というフォルダが現れ、各`Artifact`に対応するサブディレクトリが追加され始めます。`!tree artifacts`を使ってその内容を確認してください：

```python
!tree artifacts
```

### 🌐 [wandb.ai](https://wandb.ai)のArtifactsページ

`Artifact`をログに記録し使用したので、RunページのArtifactsタブを確認してみましょう。

`wandb`の出力からRunページのURLに移動し、左サイドバーから「Artifacts」タブを選択します（これは三つのホッケーパックが積み重なったようなデータベースアイコンです）。

「Input Artifacts」テーブルまたは「Output Artifacts」テーブルの行をクリックし、「Overview」や「Metadata」タブを確認して、`Artifact`にログに記録されたすべての情報を見てみましょう。

私たちは特に「グラフ表示」が気に入っています。デフォルトでは、`Artifact`の`type`と`Run`の`job_type`の2種類のノードを持つグラフが表示され、消費と生成を表す矢印が表示されます。

# 3️⃣ モデルのログ

これで、`Artifact`APIの動作を理解するのに十分ですが、パイプラインの最後までこの例を追って、`Artifact`がMLワークフローをどのように改善できるかを確認してみましょう。

最初のセルでは、PyTorchで非常にシンプルなConvNetというDNN`model`を構築します。

最初に`model`を初期化するだけで、トレーニングはしません。すべての条件を一定に保ちながらトレーニングを繰り返すことができます。

```python
from math import floor

import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, hidden_layer_sizes=[32, 64],
                  kernel_sizes=[3],
                  activation="ReLU",
                  pool_sizes=[2],
                  dropout=0.5,
                  num