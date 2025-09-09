---
title: PyTorch
menu:
  tutorials:
    identifier: ja-tutorials-integration-tutorials-pytorch
    parent: integration-tutorials
weight: 1
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb" >}}

[W&B](https://wandb.ai) を使って、機械学習の 実験管理、データセット の バージョン管理、プロジェクトのコラボレーションを行いましょう。

{{< img src="/images/tutorials/huggingface-why.png" alt="W&B を使う利点" >}}

## このノートブックで扱う内容

PyTorch コードに W&B を統合して、パイプラインに 実験管理 を追加する方法を紹介します。

{{< img src="/images/tutorials/pytorch.png" alt="PyTorch と W&B のインテグレーション図" >}}

```python
# ライブラリをインポート
import wandb

# 新しい実験を開始
with wandb.init(project="new-sota-model") as run:
 
    # ハイパーパラメーターの辞書を config で記録
    run.config = {"learning_rate": 0.001, "epochs": 100, "batch_size": 128}

    # モデルとデータを用意
    model, dataloader = get_model(), get_data()

    # 任意: 勾配をトラッキング
    run.watch(model)

    for batch in dataloader:
    metrics = model.training_step()
    # トレーニングループの中でメトリクスをログして、モデル性能を可視化
    run.log(metrics)

    # 任意: 最後にモデルを保存
    model.to_onnx()
    run.save("model.onnx")
```

[動画チュートリアル](https://wandb.me/pytorch-video) もあわせてご覧ください。

注意: 先頭が _Step_ の セクションだけ読めば、既存のパイプラインに W&B を統合できます。残りはデータの読み込みとモデル定義です。

## Install、import、ログイン


```python
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm.auto import tqdm

# 決定論的な振る舞いを保証する
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# デバイスの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# MNIST ミラーのリストから遅いミラーを除外
torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                      if not mirror.startswith("http://yann.lecun.com")]
```

### Step 0: W&B をインストール

まずは ライブラリ を入れます。
`wandb` は `pip` で簡単にインストールできます。


```python
!pip install wandb onnx -Uq
```

### Step 1: W&B を import してログイン

Web サービスに データ をログするには、
ログインが必要です。

はじめて W&B を使う場合は、
表示されるリンクから無料アカウントに登録してください。


```
import wandb

wandb.login()
```

## 実験 と パイプラインを定義する

### `wandb.init` で メタデータ と ハイパーパラメーター を追跡

まずは実験を定義します。
どんな ハイパーパラメーター を使い、どんな メタデータ をこの run に関連付けるのか。

この情報を `config` 辞書（または同様の オブジェクト）に格納し、
必要に応じて参照するのが一般的な ワークフロー です。

この例では、一部の ハイパーパラメーター だけを可変にして、
残りは手書きで設定しています。
ただし、モデルのあらゆる部分を `config` に含められます。

メタデータもいくつか入れています。ここでは MNIST データセット と 畳み込み アーキテクチャー を使います。同じ プロジェクト でたとえば
CIFAR 上の 全結合 アーキテクチャーを扱うことになっても、
run を見分けるのに役立ちます。


```python
config = dict(
    epochs=5,
    classes=10,
    kernels=[16, 32],
    batch_size=128,
    learning_rate=0.005,
    dataset="MNIST",
    architecture="CNN")
```

次に、モデル学習の典型的な パイプライン 全体を定義します。

1. まずモデルと関連する データ と オプティマイザー を `make` で用意し、
2. それに従って `train` を実行し、最後に
3. `test` してトレーニングの結果を確認します。

これらの関数は後で実装します。


```python
def model_pipeline(hyperparameters):

    # wandb に開始を知らせる
    with wandb.init(project="pytorch-demo", config=hyperparameters) as run:
        # run.config を通じてすべての HP にアクセスし、実行内容とログを一致させる
        config = run.config

        # モデル、データ、最適化問題を作成
        model, train_loader, test_loader, criterion, optimizer = make(config)
        print(model)

        # それらを使ってモデルを学習
        train(model, train_loader, criterion, optimizer, config)

        # 最終的な性能をテスト
        test(model, test_loader)

    return model
```

ここで標準的な パイプライン と違うのは、
すべてが `wandb.init` のコンテキスト内で行われる点です。
この関数を呼ぶと、あなたの コード と当社の サーバー の間に
通信の経路が設定されます。

`config` 辞書を `wandb.init` に渡すと、
その情報はすぐにログされるので、
実験の ハイパーパラメーター の 値 をいつでも確認できます。

選んでログした 値 が、実際にモデルで使われる値と常に一致するようにするために、
オブジェクトの `run.config` 版を使うことをおすすめします。
例として、下の `make` の定義を確認してください。

> _補足_: 私たちの コード は別 プロセス で動かしており、
> こちら側の問題（巨大な海の怪物が データセンター を襲う、など）が
> あなたの コード をクラッシュさせないようにしています。
> 問題が解決したら（クラーケンが深海へ戻ったら）、
> `wandb sync` でデータをログできます。


```python
def make(config):
    # データを作成
    train, test = get_data(train=True), get_data(train=False)
    train_loader = make_loader(train, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)

    # モデルを作成
    model = ConvNet(config.kernels, config.classes).to(device)

    # 損失関数とオプティマイザーを作成
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, test_loader, criterion, optimizer
```

### データ読み込みとモデルを定義

次に、データの読み込み方法とモデルの形を指定します。

この部分は非常に重要ですが、
`wandb` を使わない場合と何も変わらないので、
細かくは立ち入りません。


```python
def get_data(slice=5, train=True):
    full_dataset = torchvision.datasets.MNIST(root=".",
                                              train=train, 
                                              transform=transforms.ToTensor(),
                                              download=True)
    #  [::slice] でのスライスと同等 
    sub_dataset = torch.utils.data.Subset(
      full_dataset, indices=range(0, len(full_dataset), slice))
    
    return sub_dataset


def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size, 
                                         shuffle=True,
                                         pin_memory=True, num_workers=2)
    return loader
```

モデル定義は本来いちばん楽しいところです。

ただし `wandb` を使っても何も変わらないので、
ここでは標準的な ConvNet アーキテクチャーでいきます。

ぜひいろいろ試してみてください。すべての 結果 は [wandb.ai](https://wandb.ai) にログされます。




```python
# 典型的な（畳み込み）ニューラルネットワーク

class ConvNet(nn.Module):
    def __init__(self, kernels, classes=10):
        super(ConvNet, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, kernels[0], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, kernels[1], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * kernels[-1], classes)
        
        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc(out)
            return out
```

### トレーニング処理を定義

`model_pipeline` の続きとして、`train` のやり方を指定します。

ここでは 2 つの `wandb` 関数、`watch` と `log` を使います。

## `run.watch()` で 勾配 を、`run.log()` で それ以外を追跡

`run.watch` は、トレーニングの各 `log_freq` ステップごとに、
モデルの 勾配 や パラメータ をログします。

トレーニングを始める前に呼び出すだけで OK です。

残りのトレーニング コード は同じです。
エポック と バッチ を反復し、
forward pass と backward pass を実行し、
`optimizer` を適用します。


```python
def train(model, loader, criterion, optimizer, config):
    # wandb にモデルの挙動（勾配・重みなど）を監視させる
    run = wandb.init(project="pytorch-demo", config=config)
    run.watch(model, criterion, log="all", log_freq=10)

    # トレーニングを実行し、wandb で追跡
    total_batches = len(loader) * config.epochs
    example_ct = 0  # 見たサンプル数
    batch_ct = 0
    for epoch in tqdm(range(config.epochs)):
        for _, (images, labels) in enumerate(loader):

            loss = train_batch(images, labels, model, optimizer, criterion)
            example_ct +=  len(images)
            batch_ct += 1

            # 25 バッチごとにメトリクスを報告
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)


def train_batch(images, labels, model, optimizer, criterion):
    images, labels = images.to(device), labels.to(device)
    
    # forward pass ➡
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # オプティマイザーで一歩進める
    optimizer.step()

    return loss
```

違いはロギング コード だけです。
これまでターミナルに メトリクス を print していたところを、
同じ情報を `run.log()` に渡します。

`run.log()` は、キーが文字列の 辞書 を受け取ります。
これらの文字列はログ対象を識別し、値にその オブジェクト を入れます。
任意で、トレーニングのどの `step` かも記録できます。

> _補足_: 私はモデルが見たサンプル数を使うのが好きです。
> バッチサイズが違っても比較しやすいからです。
> もちろん生のステップやバッチ数でも構いません。学習が長い場合は `epoch` ごとにログするのも妥当です。


```python
def train_log(loss, example_ct, epoch):
    with wandb.init(project="pytorch-demo") as run:
        # 損失とエポック番号をログ
        # ここで W&B にメトリクスをログします
        run.log({"epoch": epoch, "loss": loss}, step=example_ct)
        print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")
```

### テスト処理を定義

モデルの学習が終わったらテストします。
プロダクションの新しい データ に対して走らせたり、
手作業で用意した例に適用したりします。



## （任意）`run.save()` を呼ぶ

このタイミングで、モデルのアーキテクチャー と最終 パラメータ を
ディスクに保存しておくのも良いでしょう。
最大限の互換性のため、モデルを
[Open Neural Network eXchange (ONNX) 形式](https://onnx.ai/) で `export` します。

そのファイル名を `run.save()` に渡すと、
モデルのパラメータが W&B の サーバー に保存されます。
どの `.h5` や `.pb` がどのトレーニング run に対応するのか、
もう見失いません。

モデルの保存・バージョン管理・配布に関する、より高度な `wandb` の機能は、
[Artifacts tools](https://www.wandb.com/artifacts) をご覧ください。


```python
def test(model, test_loader):
    model.eval()

    with wandb.init(project="pytorch-demo") as run:
        # テスト用のサンプルでモデルを実行
        with torch.no_grad():
            correct, total = 0, 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f"Accuracy of the model on the {total} " +
                f"test images: {correct / total:%}")
            
            run.log({"test_accuracy": correct / total})

        # 交換可能な ONNX 形式でモデルを保存
        torch.onnx.export(model, images, "model.onnx")
        run.save("model.onnx")
```

### 学習を実行し、メトリクスを wandb.ai でリアルタイムに確認

これでパイプライン全体を定義し、W&B の数行を差し込めました。
完全にトラッキングされた実験を実行する準備が整いました。

いくつかのリンクが表示されます。
ドキュメント、
プロジェクト をまとめる Project ページ、
この run の 結果 が保存される Run ページ、などです。

Run ページに移動して、次のタブを確認しましょう。

1. **Charts**: ここでは、学習中の 勾配、パラメータ値、損失がログされます
2. **System**: Disk I/O 使用率、CPU と GPU のメトリクス（温度の上昇に注目）、その他さまざまなシステム メトリクスが見られます
3. **Logs**: 学習中に標準出力へ出した内容のコピーが表示されます
4. **Files**: 学習完了後、`model.onnx` をクリックすると [Netron model viewer](https://github.com/lutzroeder/netron) でネットワークを表示できます

run が終了して `with wandb.init` ブロックを抜けると、
セル出力に 結果 のサマリーも表示します。


```python
# パイプラインでモデルを構築・学習・解析
model = model_pipeline(config)
```

### Sweeps で ハイパーパラメーター を試す

この例ではハイパーパラメーターの組み合わせは 1 つだけでした。
しかし多くの ML ワークフローでは、
複数の ハイパーパラメーター を繰り返し試します。

W&B Sweeps を使えば、ハイパーパラメーターのテストを自動化し、
可能な モデル と最適化戦略の空間を探索できます。

W&B Sweeps でハイパーパラメーター最適化を行う [Colab ノートブック](https://wandb.me/sweeps-colab) をチェックしてください。

W&B でハイパーパラメーター スイープを走らせるのはとても簡単。わずか 3 ステップです。

1. **スイープを定義:** 探索する パラメータ、探索戦略、最適化メトリクス などを指定した 辞書 か [YAML ファイル]({{< relref path="/guides/models/sweeps/define-sweep-configuration" lang="ja" >}}) を作成します。

2. **スイープを初期化:** 
`sweep_id = wandb.sweep(sweep_config)`

3. **sweep agent を実行:** 
`wandb.agent(sweep_id, function=train)`

以上でハイパーパラメーター スイープの実行は完了です。

{{< img src="/images/tutorials/pytorch-2.png" alt="PyTorch トレーニング ダッシュボード" >}}


## 例のギャラリー

W&B でトラッキング・可視化されたプロジェクト例を [ギャラリー →](https://app.wandb.ai/gallery) で探してみてください。

## Advanced Setup
1. [Environment variables]({{< relref path="/guides/hosting/env-vars/" lang="ja" >}}): 環境 変数 に APIキー を設定して、マネージド クラスター 上でトレーニングを実行できるようにします。
2. [Offline mode]({{< relref path="/support/kb-articles/run_wandb_offline.md" lang="ja" >}}): `dryrun` モードでオフライン学習し、後から 結果 を同期します。
3. [オンプレミス]({{< relref path="/guides/hosting/hosting-options/self-managed" lang="ja" >}}): 自社の インフラストラクチャー の プライベートクラウド やエアギャップ環境の サーバー に W&B をインストールします。研究機関からエンタープライズのチームまで、ローカル導入の実績があります。
4. [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}): 軽量な ツール でハイパーパラメーター探索をすばやくセットアップ。