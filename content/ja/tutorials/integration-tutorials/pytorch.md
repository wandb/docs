---
title: PyTorch
menu:
  tutorials:
    identifier: ja-tutorials-integration-tutorials-pytorch
    parent: integration-tutorials
weight: 1
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb" >}}

[W&B](https://wandb.ai) を使って機械学習の実験管理・データセットのバージョン管理・プロジェクトでの共同作業を始めましょう。

{{< img src="/images/tutorials/huggingface-why.png" alt="W&B を使うメリット" >}}

## このノートブックで学べること

PyTorch コードに W&B を組み込んで、experiment tracking をパイプラインに追加する方法を紹介します。

{{< img src="/images/tutorials/pytorch.png" alt="PyTorch と W&B のインテグレーション図" >}}

```python
# ライブラリをインポート
import wandb

# 新しい実験を開始
with wandb.init(project="new-sota-model") as run:
 
    # ハイパーパラメーター辞書を config で記録
    run.config = {"learning_rate": 0.001, "epochs": 100, "batch_size": 128}

    # モデルとデータをセットアップ
    model, dataloader = get_model(), get_data()

    # オプション: 勾配を記録
    run.watch(model)

    for batch in dataloader:
        metrics = model.training_step()
        # トレーニングループ内でメトリクスを記録し、モデルの性能を可視化
        run.log(metrics)

    # オプション: 最後にモデルを保存
    model.to_onnx()
    run.save("model.onnx")
```

[ビデオチュートリアル](https://wandb.me/pytorch-video)を見ながら一緒に進めていきましょう。

**注**: _Step_ から始まるセクションは、既存のパイプラインへ W&B を統合するのに最低限必要な部分です。それ以外はデータのロードやモデルの定義です。

## インストール・インポート・ログイン

```python
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm.auto import tqdm

# 挙動の再現性を確保
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# デバイス設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 遅いミラーを MNIST ミラーリストから除外
torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                      if not mirror.startswith("http://yann.lecun.com")]
```

### ステップ 0: W&B のインストール

始めるには、まずライブラリを用意しましょう。
`wandb` は `pip` ですぐにインストールできます。

```python
!pip install wandb onnx -Uq
```

### ステップ 1: W&B のインポートとログイン

データを Web サービスに記録するには、ログインが必要です。

初めて W&B を使う場合は、表示されるリンクから無料アカウントを作成しましょう。

```
import wandb

wandb.login()
```

## 実験とパイプラインの定義

### `wandb.init` でメタデータとハイパーパラメーターを記録

まずプログラム的に実験を定義します:
どのハイパーパラメーターを使うか？どんなメタデータをこの run に紐付けるか？

設定情報は一般的に `config` 辞書
(または似たオブジェクト)に保存して、必要なときにアクセスするのが典型です。

この例では、ごく一部のハイパーパラメーターだけを可変にし、それ以外は手動でコーディングしています。
でも、どんなモデルのパラメーターでも `config` にまとめて管理できます。

今回はメタデータとして MNIST データセットと畳み込みアーキテクチャーを使うことも記録します。
もし同じ Project 内で、CIFAR の全結合アーキテクチャーなども試す場合、それぞれの run の切り分けに役立ちます。

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

それでは、パイプライン全体を定義しましょう。
モデルのトレーニングとしてよくある流れです：

1. まずモデル、関連するデータとオプティマイザーを `make` し
2. モデルをトレーニングし
3. 最後にテストして学習結果を確認します

この流れは、下記で関数として実装します。

```python
def model_pipeline(hyperparameters):

    # W&B で experiment を開始
    with wandb.init(project="pytorch-demo", config=hyperparameters) as run:
        # run.config から全ハイパーパラメータにアクセスし、記録内容と実行内容を対応させる
        config = run.config

        # モデル・データ・損失関数・オプティマイザーの生成
        model, train_loader, test_loader, criterion, optimizer = make(config)
        print(model)

        # モデルのトレーニング
        train(model, train_loader, criterion, optimizer, config)

        # 最終的な性能のテスト
        test(model, test_loader)

    return model
```

ここが標準的なパイプラインと違うのは、すべてが `wandb.init` のコンテキストの中で行われる点です。
この関数を呼ぶことでコードと W&B のサーバーとの間に通信を確立します。

`wandb.init` に `config` 辞書を渡すだけで、すべての情報が即座に記録されます。
これで実験に使ったハイパーパラメーターの値がいつでも確認できるようになります。

モデルで利用し、記録された値が必ず一致するよう担保するため、`run.config` のオブジェクトを利用することを推奨します。
`make` の定義を確認すると、具体的な使い方がわかります。

> *補足*: W&B ではコードを別プロセスで実行しているため、仮にサーバー側で問題が起こっても（例えば大きな海の怪物がデータセンターを襲っても）あなたのコードがクラッシュしないようになっています。問題が解消（例えばクラーケンが海に帰った）後は、`wandb sync` でデータを同期できます。

```python
def make(config):
    # データの用意
    train, test = get_data(train=True), get_data(train=False)
    train_loader = make_loader(train, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)

    # モデルの作成
    model = ConvNet(config.kernels, config.classes).to(device)

    # 損失関数とオプティマイザー
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, test_loader, criterion, optimizer
```

### データのロード・モデルの定義

次は、データのロード方法やモデルの構造を定義します。

この部分はとても重要ですが、
`wandb` を使わないケースと変わらないため、ここでは詳しく説明しません。

```python
def get_data(slice=5, train=True):
    full_dataset = torchvision.datasets.MNIST(root=".",
                                              train=train, 
                                              transform=transforms.ToTensor(),
                                              download=True)
    #  [::slice] と等価なサブセットを作成
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

モデル定義は、普段の機械学習の “楽しいところ” ですよね。

ですが、`wandb` を使ってもこの部分は変わりません。
今回は標準的な ConvNet アーキテクチャーにしています。

この部分は自由に変更して実験してみてください ― どんな結果も [wandb.ai](https://wandb.ai) で記録されます。

```python
# 標準的な畳み込みニューラルネットワーク

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

### トレーニングロジックの定義

`model_pipeline` の続きとして、トレーニング方法を定義しましょう。

ここで使う W&B の2つの関数は、`watch` と `log` です。

## `run.watch()` で勾配を、`run.log()` でメトリクスなどを記録

`run.watch` を使うと、トレーニングの `log_freq` ステップごとに、モデルの勾配やパラメーターを記録できます。

トレーニング開始前に呼び出すだけで OK。

トレーニングコード自体はこれまで通り ―
エポック・バッチをイテレートしながら forward/backward を回し、
オプティマイザーでパラメーターを更新します。

```python
def train(model, loader, criterion, optimizer, config):
    # W&B にモデルの挙動（勾配・重みなど）を監視させる
    run = wandb.init(project="pytorch-demo", config=config)
    run.watch(model, criterion, log="all", log_freq=10)

    # トレーニングを実行し、W&B で追跡
    total_batches = len(loader) * config.epochs
    example_ct = 0  # 観測したサンプル数
    batch_ct = 0
    for epoch in tqdm(range(config.epochs)):
        for _, (images, labels) in enumerate(loader):

            loss = train_batch(images, labels, model, optimizer, criterion)
            example_ct +=  len(images)
            batch_ct += 1

            # 25バッチごとにメトリクスを記録
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)


def train_batch(images, labels, model, optimizer, criterion):
    images, labels = images.to(device), labels.to(device)
    
    # フォワードパス ➡
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # バックワードパス ⬅
    optimizer.zero_grad()
    loss.backward()

    # オプティマイザーでパラメーター更新
    optimizer.step()

    return loss
```

違いは logging コードだけです。
以前ならターミナルに print していたようなメトリクスを、今度は同じ情報を `run.log()` で渡します。

`run.log()` には「キーが文字列、値がログ化するオブジェクト」でなる辞書を渡します。
キーはログの名前になり、値は記録対象です。
トレーニングの「どの step で」記録したかを指定することもできます。

> *補足*: サンプル数で「何例見せた時点か」を使うとバッチサイズをまたいで比較しやすいのでおすすめです。もちろんそのままステップ数やバッチ数で記録してもOK。トレーニングが長くなる場合は epoch ごとのロギングも便利です。

```python
def train_log(loss, example_ct, epoch):
    with wandb.init(project="pytorch-demo") as run:
        # 損失とエポック数を記録
        # ここで W&B にメトリクスを記録
        run.log({"epoch": epoch, "loss": loss}, step=example_ct)
        print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")
```

### テストロジックの定義

トレーニング完了後はモデルをテストしましょう。
プロダクションの新しいデータや、厳選した例で評価します。

## （オプション）`run.save()` の利用

このタイミングでモデルのアーキテクチャーや最終パラメータをディスクに保存するのもおすすめです。
最大限の互換性のため、モデルを[Open Neural Network eXchange（ONNX）フォーマット](https://onnx.ai/)でエクスポートします。

ファイル名を `run.save()` に渡すことで、モデルのパラメータが W&B サーバーに保存されます。
もうどの `.h5` や `.pb` がどのトレーニング run のものなのか分からなくなる心配もありません。

さらに詳しいモデルの保存・バージョン管理・配布方法については [Artifacts ツール](https://www.wandb.com/artifacts) をご覧ください。

```python
def test(model, test_loader):
    model.eval()

    with wandb.init(project="pytorch-demo") as run:
        # テスト用データでモデルを評価
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

        # 交換可能なONNX形式でモデルを保存
        torch.onnx.export(model, images, "model.onnx")
        run.save("model.onnx")
```

### トレーニングして wandb.ai でライブメトリクスを確認

ここまででパイプライン全体を用意し、W&B のコードも数行加えるだけで、
完全に管理された実験を実行できるようになりました。

完了後には次のリンクが表示されます:
ドキュメント・Project ページ（すべての run を整理）・Run ページ（個々の結果が保存）です。

Run ページでは以下のタブをチェックできます：

1. **Charts** ― モデルの勾配・パラメータ値・損失などがトレーニング全体で記録
2. **System** ― ディスク I/O 利用率や、CPU/GPU メトリクス（温度上昇なども）、その他多数のシステム情報
3. **Logs** ― トレーニング中に標準出力へ出したすべてのログ
4. **Files** ― トレーニング完了後は `model.onnx` をクリックして、[Netron モデルビューア](https://github.com/lutzroeder/netron)でネットワーク構造を可視化できます

Run が完了し `with wandb.init` ブロックを抜けると、
セルの出力に実験結果のサマリーが表示されます。

```python
# パイプラインでモデルを構築・トレーニング・分析
model = model_pipeline(config)
```

### Sweeps でハイパーパラメータ探索

この例ではハイパーパラメータを 1 セットしか使いませんでしたが、
多くの ML ワークフローではさまざまなハイパーパラメータの組み合わせを試す必要があります。

W&B Sweeps を使えば、ハイパーパラメータテストを自動化し、モデルやオプティマイザーの探索が簡単に行えます。

W&B Sweeps でのハイパーパラメータ最適化を紹介する [Colabノートブック](https://wandb.me/sweeps-colab) もご覧ください。

W&B でのハイパーパラメータスイープはとてもシンプル。たった3ステップです：

1. **スイープの定義:** 探索するパラメータ・サーチ戦略・最適化指標などを指定した辞書または [YAMLファイル]({{< relref path="/guides/models/sweeps/define-sweep-configuration" lang="ja" >}}) を作成します。

2. **スイープの初期化:**  
`sweep_id = wandb.sweep(sweep_config)`

3. **スイープエージェントの実行:**  
`wandb.agent(sweep_id, function=train)`

これだけで自動的にハイパーパラメータ探索が始められます。

{{< img src="/images/tutorials/pytorch-2.png" alt="PyTorch トレーニングダッシュボード" >}}

## 事例ギャラリー

[Gallery →](https://app.wandb.ai/gallery) では、W&B で管理・可視化した様々なプロジェクト事例を公開中です。

## 応用セットアップ
1. [環境変数]({{< relref path="/guides/hosting/env-vars/" lang="ja" >}}): APIキー を環境変数に設定して、マネージドクラスターでトレーニングを実行。
2. [オフラインモード]({{< relref path="/support/kb-articles/run_wandb_offline.md" lang="ja" >}}): `dryrun` モードでオフライン実行し、後で結果を同期。
3. [オンプレミス]({{< relref path="/guides/hosting/hosting-options/self-managed" lang="ja" >}}): プライベートクラウドや自社インフラのエアギャップサーバーに W&B をインストール。アカデミックからエンタープライズまで対応したローカルインストールが可能です。
4. [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}): 軽量ツールでハイパーパラメーター探索をすばやくセットアップできます。