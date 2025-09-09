---
title: モデル と データセット をトラッキングする
menu:
  tutorials:
    identifier: ja-tutorials-artifacts
weight: 4
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Pipeline_Versioning_with_W&B_Artifacts.ipynb" >}}
このノートブックでは、W&B Artifacts を使って ML の 実験 パイプラインをトラッキングする方法を紹介します。

[ビデオチュートリアル](https://tiny.cc/wb-artifacts-video) もあわせてご覧ください。

## アーティファクトについて

アーティファクトは、ギリシャの [アンフォラ](https://en.wikipedia.org/wiki/Amphora) のように、プロセスの結果として生み出される オブジェクト です。
ML では、最も重要なアーティファクトは _データセット_ と _モデル_ です。

そして [コロナドの十字架](https://indianajones.fandom.com/wiki/Cross_of_Coronado) のように、重要なアーティファクトは博物館に収蔵されるべきものです。
つまり、あなたやチーム、そして広く ML コミュニティがそこから学べるように、カタログ化して整理しておくべきです。
トレーニング を記録しない者は、それを繰り返す運命にあるのですから。

Artifacts API を使うと、W&B の `Run` の出力として `Artifact` をログしたり、逆に `Run` の入力として `Artifact` を使ったりできます。次の図のように、トレーニングの Run は dataset を入力として受け取り、 model を出力します。
 
 {{< img src="/images/tutorials/artifacts-diagram.png" alt="Artifacts のワークフロー図" >}}

1 つの Run が別の Run の出力を入力として使えるため、`Artifact` と `Run` は合わせて有向グラフ（`Artifact` と `Run` がノードの二部 [DAG](https://en.wikipedia.org/wiki/Directed_acyclic_graph)）
を形成し、矢印で `Run` と、それが消費・生成する `Artifact` を結びます。

## Artifact を使って モデル と データセット をトラッキングする

### インストールとインポート

Artifacts は Python ライブラリの一部で、`0.9.2` から利用できます。

ほとんどの ML の Python スタックと同様に、`pip` でインストールできます。


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

まず、いくつかの Artifacts を定義します。

この例は PyTorch の
["Basic MNIST Example"](https://github.com/pytorch/examples/tree/master/mnist/)
をベースにしていますが、[TensorFlow](https://wandb.me/artifacts-colab) や他の任意のフレームワーク、
あるいは純粋な Python でも同様に実現できます。

まずは `Dataset` から始めます:
- パラメータ を選ぶための `train` セット
- ハイパーパラメータ を選ぶための `validation` セット
- 最終モデルを評価するための `test` セット

以下の最初のセルは、これら 3 つのデータセットを定義します。


```python
import random 

import torch
import torchvision
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm

# 決定論的な振る舞いを保証
torch.backends.cudnn.deterministic = True
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# デバイス設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# データのパラメータ
num_classes = 10
input_shape = (1, 28, 28)

# 遅い MNIST ミラーを候補から除外
torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                      if not mirror.startswith("http://yann.lecun.com")]

def load(train_size=50_000):
    """
    # Load the data
    """

    # データを学習用とテスト用に分割
    train = torchvision.datasets.MNIST("./", train=True, download=True)
    test = torchvision.datasets.MNIST("./", train=False, download=True)
    (x_train, y_train), (x_test, y_test) = (train.data, train.targets), (test.data, test.targets)

    # ハイパーパラメータ チューニング用に検証セットを分割
    x_train, x_val = x_train[:train_size], x_train[train_size:]
    y_train, y_val = y_train[:train_size], y_train[train_size:]

    training_set = TensorDataset(x_train, y_train)
    validation_set = TensorDataset(x_val, y_val)
    test_set = TensorDataset(x_test, y_test)

    datasets = [training_set, validation_set, test_set]

    return datasets
```

これは、この例の中で繰り返し登場するパターンを示しています。
データをアーティファクトとしてログするためのコードは、
そのデータを生成するコードの周りにラップされます。
この場合、データを `load` するコードと、
それを `load_and_log` するコードを分けています。

これは良いプラクティスです。

これらのデータセットを Artifacts としてログするには、
以下の手順だけです。
1. `wandb.init()` で `Run` を作成する（L4）
2. データセット用の `Artifact` を作る（L10）
3. 関連する `file` を保存してログする（L20, L23）

まずは下のコードセルの例を確認して、
その後のセクションで詳細を見ていきましょう。


```python
def load_and_log():

    # Run を開始。ラベル用に type を、保存先としての Project を指定
    with wandb.init(project="artifacts-example", job_type="load-data") as run:
        
        datasets = load()  # データセットを読み込むコードは分離
        names = ["training", "validation", "test"]

        # 🏺 Artifact を作成
        raw_data = wandb.Artifact(
            "mnist-raw", type="dataset",
            description="Raw MNIST dataset, split into train/val/test",
            metadata={"source": "torchvision.datasets.MNIST",
                      "sizes": [len(dataset) for dataset in datasets]})

        for name, data in zip(names, datasets):
            # 🐣 新しいファイルを Artifact に保存し、その内容に書き込む
            with raw_data.new_file(name + ".pt", mode="wb") as file:
                x, y = data.tensors
                torch.save((x, y), file)

        # ✍️ Artifact を W&B に保存
        run.log_artifact(raw_data)

load_and_log()
```

#### `wandb.init()`

`Artifact` を生成する `Run` を作る際には、
どの `project` に属するかを指定する必要があります。

ワークフロー によっては、
Project は `car-that-drives-itself` のように大きいものかもしれませんし、
`iterative-architecture-experiment-117` のように小さいものかもしれません。

> **ベストプラクティス**: 可能であれば、同じ `Artifact` を共有するすべての `Run` は 1 つの Project にまとめましょう。シンプルになります。とはいえ、`Artifact` は Project をまたいで持ち運べます。

実行するジョブの種類は多岐にわたるので、
`Run` を作るときに `job_type` を与えると整理に役立ちます。
Artifacts のグラフがすっきりと保てます。

> **ベストプラクティス**: `job_type` は説明的で、パイプラインの 1 ステップに対応させましょう。ここでは、データの `load` と `preprocess` を分けています。

#### `wandb.Artifact`

何かを `Artifact` としてログするには、まず `Artifact` オブジェクトを作成します。

すべての `Artifact` には `name` があり、これは第一引数で設定します。

> **ベストプラクティス**: `name` はわかりやすく、思い出しやすく、タイプしやすいものにしましょう。コード中の変数名に対応させ、ハイフン区切りを使うのがおすすめです。

`type` もあります。`Run` の `job_type` と同様に、`Run` と `Artifact` のグラフを整理するために使われます。

> **ベストプラクティス**: `type` はシンプルに。`mnist-data-YYYYMMDD` より、`dataset` や `model` のようなものを使いましょう。

さらに、`description` や `metadata` を辞書として付与できます。
`metadata` は JSON にシリアライズ可能である必要があります。

> **ベストプラクティス**: `metadata` はできるだけ詳細にしましょう。

#### `artifact.new_file` と `run.log_artifact`

`Artifact` オブジェクトを作ったら、そこにファイルを追加する必要があります。

そのとおり、_files_ に _s_ がついた複数形です。
`Artifact` はディレクトリーのような構造で、
ファイルやサブディレクトリーを含みます。

> **ベストプラクティス**: 可能な限り、`Artifact` の内容は複数ファイルに分割しましょう。スケールが必要になったときに役立ちます。

`new_file` メソッドを使うと、
ファイルの書き込みと `Artifact` への追加を同時に行えます。
この後では、2 ステップに分ける `add_file` メソッドも使います。

すべてのファイルを追加したら、[wandb.ai](https://wandb.ai) に `log_artifact` します。

出力にいくつかの URL が表示されるのがわかるはずです。
Run ページの URL も含まれます。
ここで `Run` の結果や、
ログされた `Artifact` を確認できます。

Run ページの他のコンポーネントの活用例は、この後で見ていきます。

### ログ済みの dataset Artifact を使う

博物館のアーティファクトと違い、W&B の `Artifact` は保管するだけでなく、実際に _使う_ ために設計されています。

その使い方を見ていきましょう。

次のセルでは、生のデータセットを受け取り、
`preprocess` して `normalize` と形状調整を行ったデータセットを生成する
パイプラインステップを定義します。

ここでも、`preprocess` の中身と、
`wandb` とのインターフェース部分のコードを分離しています。


```python
def preprocess(dataset, normalize=True, expand_dims=True):
    """
    ## Prepare the data
    """
    x, y = dataset.tensors

    if normalize:
        # 画像を [0, 1] の範囲にスケーリング
        x = x.type(torch.float32) / 255

    if expand_dims:
        # 画像の形状を (1, 28, 28) にそろえる
        x = torch.unsqueeze(x, 1)
    
    return TensorDataset(x, y)
```

次は、この `preprocess` ステップに `wandb.Artifact` のロギングを組み込むコードです。

以下の例では、新しく `Artifact` を `use` し、
さらに `log` します（ここは前のステップと同じ）。
`Artifact` は `Run` の入力にも出力にもなります。

今回は `job_type` に `preprocess-data` を使い、
前のステップとは別の種類のジョブであることを明確にします。


```python
def preprocess_and_log(steps):

    with wandb.init(project="artifacts-example", job_type="preprocess-data") as run:

        processed_data = wandb.Artifact(
            "mnist-preprocess", type="dataset",
            description="Preprocessed MNIST dataset",
            metadata=steps)
         
        # ✔️ 使用する Artifact を宣言
        raw_data_artifact = run.use_artifact('mnist-raw:latest')

        # 📥 必要であれば Artifact をダウンロード
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

ここで注目したいのは、前処理の `steps` を
`preprocessed_data` の `metadata` として保存している点です。

実験 を再現可能にしたいなら、
豊富なメタデータを記録するのは良い考えです。

また、今回のデータセットは「`large artifact`」ですが、
`download` ステップは 1 秒もかからずに完了します。

詳しくは、以下のマークダウンセルを展開してください。


```python
steps = {"normalize": True,
         "expand_dims": True}

preprocess_and_log(steps)
```

#### `run.use_artifact()`

このステップはシンプルです。利用側は `Artifact` の `name` を知っていればよく、少しだけ追加情報が必要です。

その「少しだけ」が、使いたい `Artifact` の特定のバージョンを表す `alias` です。

デフォルトでは、最後にアップロードされたバージョンに `latest` が付与されます。
それ以外にも `v0` / `v1` のように任意の過去バージョンを選べますし、
`best` や `jit-script` のように独自のエイリアスを付けることもできます。
[Docker Hub](https://hub.docker.com/) のタグと同様に、
エイリアスは `:` で名前と区切ります。
今回欲しい `Artifact` は `mnist-raw:latest` です。

> **ベストプラクティス**: エイリアスは短くてわかりやすく。ある条件を満たす `Artifact` が欲しいときは、`latest` や `best` のようなカスタム `alias` を使いましょう。

#### `artifact.download`

`download` の呼び出しでメモリ使用量が倍増しないか心配かもしれません。

ご心配なく。実際にダウンロードを行う前に、
同じバージョンがローカルにあるかどうかを確認します。
これは [torrent](https://en.wikipedia.org/wiki/Torrent_file) や [`git` のバージョン管理](https://blog.thoughtram.io/git/2014/11/18/the-anatomy-of-a-git-commit.html) を支えるのと同じ、ハッシュ化の仕組みを使っています。

`Artifact` が作成・ログされていくと、
作業ディレクトリー内の `artifacts` フォルダーに
各 `Artifact` ごとのサブディレクトリーが増えていきます。
`!tree artifacts` で中身を確認してみましょう:


```python
!tree artifacts
```

#### Artifacts ページ 

`Artifact` をログして使ったので、
Run ページの Artifacts タブを見てみましょう。 

`wandb` の出力にある Run ページの URL に移動し、
左サイドバーから「Artifacts」タブを選択します
（データベースのアイコンで、
ホッケーパックが三段に積み重なったような見た目です）。

**Input Artifacts** テーブルか
**Output Artifacts** テーブルのいずれかの行をクリックして、
タブ（**Overview**、**Metadata**）を見れば、
その `Artifact` に関してログされた内容を確認できます。

特に **Graph View** は便利です。
デフォルトでは、`Artifact` の `type` と
`Run` の `job_type` を 2 種類のノードとして、
消費と生成の関係を矢印で表したグラフが表示されます。

### モデルをログする

`Artifact` の API の使い方はここまでで十分ですが、
パイプラインの最後までこの例を追って、
`Artifact` が ML のワークフローをどう改善するかも見てみましょう。

最初のセルでは、PyTorch で DNN の `model`（とてもシンプルな ConvNet）を構築します。

まずは `model` の初期化だけを行い、学習はしません。
こうすることで、他を変えずに学習だけを繰り返せます。


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

        fc_input_dims = floor((input_shape[1] - kernel_sizes[0] + 1) / pool_sizes[0]) # レイヤー 1 の出力サイズ
        fc_input_dims = floor((fc_input_dims - kernel_sizes[-1] + 1) / pool_sizes[-1]) # レイヤー 2 の出力サイズ
        fc_input_dims = fc_input_dims*fc_input_dims*hidden_layer_sizes[-1] # レイヤー 3 の出力サイズ

        self.fc = nn.Linear(fc_input_dims, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc(x)
        return x
```

ここでは W&B で Run をトラッキングするので、
[`run.config`](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-config/Configs_in_W%26B.ipynb)
オブジェクトにすべてのハイパーパラメータ を保存します。

その `config` オブジェクトの `dict` 版はとても有用な `metadata` なので、必ず含めてください。


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
        # ➕ Artifact にファイルを追加するもう 1 つの方法
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

データセットのロギング例のように `new_file` で
ファイルの書き込みと `Artifact` への追加を同時に行う代わりに、
（ここでは `torch.save` のように）まずファイルを書き出し、
その後のステップで `add` して `Artifact` に追加することもできます。

> **ベストプラクティス**: 重複を防ぐため、可能なら `new_file` を使いましょう。

#### ログ済みの Model Artifact を使う

`dataset` に対して `use_artifact` を呼び出せたのと同様に、
`initialized_model` に対しても呼び出し、
別の `Run` で利用できます。

今回は `model` を `train` しましょう。

詳しくは、[PyTorch で W&B を計測する](https://wandb.me/pytorch-colab) Colab をご覧ください。


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

        # 各エポックごとに検証セットで評価
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
            pred = output.argmax(dim=1, keepdim=True)  # 予測確率が最大のインデックスを取得
            correct += pred.eq(target.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)
    
    return test_loss, accuracy


def train_log(loss, example_ct, epoch):
    loss = float(loss)

    # 魔法が起きる場所
    with wandb.init(project="artifacts-example", job_type="train") as run:
        run.log({"epoch": epoch, "train/loss": loss}, step=example_ct)
        print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")
    

def test_log(loss, accuracy, example_ct, epoch):
    loss = float(loss)
    accuracy = float(accuracy)

    # 魔法が起きる場所
    with wandb.init() as run:
        run.log({"epoch": epoch, "validation/loss": loss, "validation/accuracy": accuracy}, step=example_ct)
        print(f"Loss/accuracy after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}/{accuracy:.3f}")
```

今回は、`Artifact` を生成する `Run` を 2 つ実行します。

1 つ目が `model` を `train` し終えたら、
2 つ目が `trained-model` の `Artifact` を取り込み、
`test_dataset` で性能を `evaluate` します。

さらに、ネットワークが最も混乱した 32 個のサンプル、
すなわち `categorical_crossentropy` が最も高いサンプルも抽出します。

これはデータセットやモデルの問題を診断するのに有効です。


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

    # データセットの各アイテムに対する損失と予測を取得
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

これらのロギング関数は、新しい `Artifact` の機能を追加していないので、
特に解説はしません。
単に `use`、`download`、
そして `log` で `Artifact` を扱っているだけです。


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