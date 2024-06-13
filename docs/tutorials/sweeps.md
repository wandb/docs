


# ハイパーパラメーターをチューニングする

[**こちらのColabノートブックで試してみてください →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W&B.ipynb)

高次元のハイパーパラメータースペースを探索して最もパフォーマンスの良いモデルを見つけるのは非常に難しいことがあります。ハイパーパラメーター探索は、モデルのバトルロワイヤルを効率的に行い、最も正確なモデルを選ぶための整理された方法を提供します。これにより、学習率、バッチサイズ、隠れ層の数、オプティマイザータイプなどのハイパーパラメーター値の組み合わせを自動的に検索して最適な値を見つけることができます。

このチュートリアルでは、Weights & Biasesを使用して3つの簡単なステップで高度なハイパーパラメーター探索を実行する方法を見ていきます。

### [ビデオチュートリアル](http://wandb.me/sweeps-video)に沿って進めてください！

![](https://i.imgur.com/WVKkMWw.png)

# 🚀 設定

まずは実験追跡ライブラリをインストールし、無料のW&Bアカウントを設定しましょう：

1. `!pip install`でインストール
2. Pythonにライブラリを`import`
3. `.login()`してプロジェクトにメトリクスを記録する

初めてWeights & Biasesを使用する場合は、
`login`の呼び出しによりアカウントのサインアップリンクが表示されます。
W&Bは個人および学術プロジェクトで無料で使用できます！

```python
!pip install wandb -Uq
```

```python
import wandb

wandb.login()
```

# ステップ1️⃣: スイープを定義する

基本的に、スイープは一連のハイパーパラメーター値を試すための戦略とそれを評価するコードを組み合わせたものです。
[設定](https://docs.wandb.com/sweeps/configuration)の形で戦略を定義する必要があります。

上記のスイープをノートブックで設定する場合、
設定オブジェクトはネストされた辞書になります。
コマンドラインでスイープを実行する場合、
設定オブジェクトは
[YAMLファイル](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration)になります。

スイープ設定の定義を一緒に見ていきましょう。
各コンポーネントを説明するため、ゆっくり進めていきます。
通常のスイープ開発フローでは、
このステップは一つの課題で行われます。

### 👈 `method`を選択する

最初に定義する必要があるのは、新しいパラメーター値を選択するための`method`です。

以下の検索`methods`を提供しています：
* **`grid` Search** – ハイパーパラメーター値のすべての組み合わせを繰り返し試します。非常に効果的ですが、計算コストが高いです。
* **`random` Search** – 提供された`distribution`に従ってランダムに新しい組み合わせを選択します。意外に効果的です！
* **`bayes`ian Search** – ハイパーパラメーターの関数としてメトリクススコアの確率モデルを作成し、メトリクスの改善の確率が高いパラメーターを選択します。少数の連続パラメーターに対してはうまく機能しますが、スケールが悪いです。

ここでは`random`を使用します。

```python
sweep_config = {
    'method': 'random'
}
```

`bayes`ianスイープの場合、
`metric`についても少し教えてください。
モデルの出力で見つけられるように、名前が必要であり、
また、`goal`が`minimize`（例：二乗誤差の場合）
または`maximize`（例：精度の場合）であるかも教えてください。

```python
metric = {
    'name': 'loss',
    'goal': 'minimize'   
}

sweep_config['metric'] = metric
```

`bayes`ianスイープを実行していない場合でも、
後で変更する場合を考えて、`sweep_config`にこれを含めることをお勧めします。
また、再現性のためにこれらの情報をメモしておくのも良い習慣です。
6ヶ月後または6年後にスイープを再訪したとき、
`val_G_batch`が高い方が良いのか低い方が良いのかわからない場合に役立ちます。

### 📃 ハイパー`parameters`の名前を付ける

新しいハイパーパラメーターの値を試す`method`を選んだら、
その`parameters`を定義する必要があります。

このステップはほとんどの場合、簡単です。
`parameter`に名前を付け、
そのパラメーターの合法な`values`のリストを指定するだけです。

たとえば、ネットワークの`optimizer`を選ぶ際には、限られたオプションしかありません。
ここでは最も人気のある選択肢である`adam`と`sgd`に限定します。
無限の選択肢があるハイパーパラメーターでも、
いくつかの選択された`values`を試すことが意味をなす場合がほとんどです。
これは、ここでの隠れ`layer_size`と`dropout`の例です。

```python
parameters_dict = {
    'optimizer': {
        'values': ['adam', 'sgd']
    },
    'fc_layer_size': {
        'values': [128, 256, 512]
    },
    'dropout': {
        'values': [0.3, 0.4, 0.5]
    },
}

sweep_config['parameters'] = parameters_dict
```

スイープで変えたくないハイパーパラメーターもよくありますが、
それでも`sweep_config`に設定しておきたい場合があります。

その場合は、単に`value`を直接設定します。

```python
parameters_dict.update({
    'epochs': {
        'value': 1}
})
```

`grid`サーチの場合、これが必要なすべてです。

`random`検索の場合、パラメーターのすべての`values`が特定のrunで選ばれる確率は同じです。

それでは不十分であれば、
名前付き`distribution`とそのパラメーター（例えば、`normal`分布の平均`mu`と標準偏差`sigma`）を指定できます。

ランダム変数の分布の設定方法については[こちら](https://docs.wandb.com/sweeps/configuration#distributions)で詳しく説明しています。

```python
parameters_dict.update({
    'learning_rate': {
        # 0から0.1までのフラット分布
        'distribution': 'uniform',
        'min': 0,
        'max': 0.1
    },
    'batch_size': {
        # 32から256までの整数、対数が均等に分布
        'distribution': 'q_log_uniform_values',
        'q': 8,
        'min': 32,
        'max': 256,
    }
})
```

終わったら、`sweep_config`はネストされた辞書になり、
どの`parameters`を試してみたいか、
そして何の`method`で試すかを正確に指定することができます。

```python
import pprint

pprint.pprint(sweep_config)
```

しかし、これですべての設定オプションではありません！

例えば、[HyperBand](https://arxiv.org/pdf/1603.06560.pdf)スケジューリングアルゴリズムでrunを早期終了するオプションも提供しています。
詳細は[こちら](https://docs.wandb.com/sweeps/configuration#stopping-criteria)をご参照ください。

すべての設定オプションのリストは[こちら](https://docs.wandb.com/library/sweeps/configuration)で見つけることができ、
YAML形式の例の大集合は[こちら](https://github.com/wandb/examples/tree/master/examples/keras/keras-cnn-fashion)で見つけることができます。

# ステップ2️⃣: スイープを初期化する

検索戦略を定義したら、それを実装する何かを設定する時です。

スイープを管理するのは _Sweep Controller_ と呼ばれるものです。
各runが終了すると、新しいrunを実行するための新しい指示を出します。
この指示は_runを実行するエージェントにより拾われます。

通常のスイープでは、コントローラーは_自分のマシン_に住んでおり、
runを完了させるエージェントは_ユーザーのマシン_に住み、以下の図のようになります。
この労働の分割により、エージェントを実行するマシンを追加するだけでスイープを簡単にスケールアップできます！

<img src="https://i.imgur.com/zlbw3vQ.png" alt="sweeps-diagram" width="500"/>

`sweep_config`と`project`名を指定して`wandb.sweep`を呼び出すことでスイープコントローラーを設定できます。

この関数は後でエージェントをこのコントローラーに割り当てるために使用する`sweep_id`を返します。

> _サイドノート_: コマンドラインでは、この関数は次のように置き換えられます
```python
wandb sweep config.yaml
```
[コマンドラインでスイープを使用する方法について詳しくはこちら ➡](https://docs.wandb.ai/guides/sweeps/walkthrough)

```python
sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")
```

# ステップ3️⃣: スイープエージェントを実行する

### 💻 トレーニング手順を定義する

実際にスイープを実行する前に、
それらの値を使用するトレーニング手順を定義する必要があります。

以下の関数では、PyTorchでシンプルな全結合ニューラルネットワークを定義し、
以下の`wandb`ツールを追加してモデルメトリクスをログに記録し、パフォーマンスと出力を可視化し、実験を追跡するようにします：
* [**`wandb.init()`**](https://docs.wandb.com/library/init) – 新しいW&B Runを初期化します。各Runはトレーニング関数を一回実行するものです。
* [**`wandb.config`**](https://docs.wandb.com/library/config) – すべてのハイパーパラメーターを設定オブジェクトに保存し、それをログに記録できるようにします。`wandb.config`の使い方について詳しくは[こちら](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Configs_in_W%26B.ipynb)を参照してください。
* [**`wandb.log()`**](https://docs.wandb.com/library/log) – モデルの振る舞いをW&Bにログします。ここではパフォーマンスだけをログしますが、`wandb.log`でログできる他のリッチメディアについては[こちらのColab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Log_(Almost)_Anything_with_W%26B_Media.ipynb)を参照してください。

PyTorchとW&Bの統合についての詳細は[こちらのColab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb)を参照してください。

```python
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(config=None):
    # 新しいwandb runを初期化
    with wandb.init(config=config):
        # 下記のようにwandb.agentから呼び出された場合、
        # このconfigはスイープコントローラーにより設定されます
        config = wandb.config

        loader = build_dataset(config.batch_size)
        network = build_network(config.fc_layer_size, config.dropout)
        optimizer = build_optimizer(network, config.optimizer, config.learning_rate)

        for epoch in range(config.epochs):
            avg_loss = train_epoch(network, loader, optimizer)
            wandb.log({"loss": avg_loss, "epoch": epoch})           
```

このセルはトレーニング手順の4つの部分、`build_dataset`、`build_network`、`build_optimizer`、`train_epoch`を定義します。

これらはすべて基本的なPyTorch開発フローの標準的な部分であり、W&Bの使用によってその実装に影響はないため、コメントしません。

```python
def build_dataset(batch_size):
   
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
    # MNISTトレーニングデータセットをダウンロード
    dataset = datasets.MNIST(".", train=True, download=True,
                             transform=transform)
    sub_dataset = torch.utils.data.Subset(
        dataset, indices=range(0, len(dataset), 5))
    loader = torch.utils.data.DataLoader(sub_dataset, batch_size=batch_size)

    return loader


def build_network(fc_layer_size, dropout):
    network = nn.Sequential(  # 全結合、単一の隠れ層
        nn.Flatten(),
        nn.Linear(784, fc_layer_size), nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(fc_layer_size, 10),
        nn.LogSoftmax(dim=1))

    return network.to(device)
        

def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),
                               lr=learning_rate)
    return optimizer


def train_epoch(network, loader, optimizer):
    cumu_loss = 0
    for _, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # ➡ Forwardパス
        loss = F.nll_loss(network(data), target)
        cumu_loss += loss.item()

        # ⬅ バックワードパス + 重みの更新
        loss.backward()
        optimizer.step()

        wandb.log({"batch loss": loss.item()})

    return cumu_loss / len(loader)
```

さあ、スイープを始めましょう！🧹🧹🧹

`wandb.sweep`を実行して作成されたスイープコントローラーは、
誰かが`config`を試すように要求するのを待っています。

その誰かとは`agent`であり、`wandb.agent`で作成されます。
エージェントを起動するために知る必要があるのは
1. どのスイープの一部であるか（`sweep_id`）
2. 実行する関数（ここでは`train`）
3. （オプションで）コントローラーに要求するconfigの数（`count`）

FYI、同じ`sweep_id`を持つ複数の`agent`を異なるコンピュータリリソースで起動でき、
コントローラーは`sweep_config`に示された戦略に従って協力して動作するようにします。
これにより、使用可能なノードが多いほどスイープを簡単にスケールできます！

> _サイドノート:_ コマンドラインでは、この関数は次のように置き換えられます
```bash
wandb agent sweep_id
```
[コマンドラインでスイープを使用する方法について詳しくはこちら ➡](https://docs.wandb.com/sweeps/walkthrough)

以下のセルは、スイープコントローラーから返されたランダムに生成されたハイパーパラメータ値を使用して`train`を5回実行する`agent`を起動します。
実行には5分未満かかります。

```python
wandb.agent(sweep_id, train, count=5)
```

# 👀 スイープ結果を可視化する


## 🔀 パラレルコーディネートプロット
このプロットはハイパーパラメーター値をモデルメトリクスにマッピングします。最も良いモデル性能を引き出したハイパーパラメーターの組み合わせを特定するのに役立ちます。

![](https://assets.website-files.com/5ac6b7f2924c652fd013a