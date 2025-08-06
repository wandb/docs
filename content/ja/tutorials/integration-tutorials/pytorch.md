---
title: PyTorch
menu:
  tutorials:
    identifier: pytorch
    parent: integration-tutorials
weight: 1
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb" >}}

[W&B](https://wandb.ai) を使って、機械学習の実験管理、データセットのバージョン管理、プロジェクトのコラボレーションを行いましょう。

{{< img src="/images/tutorials/huggingface-why.png" alt="Benefits of using W&B" >}}

## このノートブックで学べること

PyTorch コードに W&B を統合し、experiment tracking をパイプラインに追加する方法を紹介します。

{{< img src="/images/tutorials/pytorch.png" alt="PyTorch and W&B integration diagram" >}}

```python
# ライブラリをインポート
import wandb

# 新しい実験を開始
with wandb.init(project="new-sota-model") as run:
 
    # ハイパーパラメーターの辞書を config で記録
    run.config = {"learning_rate": 0.001, "epochs": 100, "batch_size": 128}

    # モデルとデータのセットアップ
    model, dataloader = get_model(), get_data()

    # 任意：勾配を記録
    run.watch(model)

    for batch in dataloader:
    metrics = model.training_step()
    # トレーニングループ内でメトリクスを記録し、モデル性能を可視化
    run.log(metrics)

    # 任意：最後にモデルを保存
    model.to_onnx()
    run.save("model.onnx")
```

[動画チュートリアル](https://wandb.me/pytorch-video)でも詳しく解説しています。

**注意**：_Step_ で始まるセクションは、W&B を既存のパイプラインへ統合するために必要な内容だけを紹介しています。それ以外はデータのロードやモデル定義についてです。

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

# 再現性を保証
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# デバイスの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# MNIST ミラーリストから遅いミラーを除外
torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                      if not mirror.startswith("http://yann.lecun.com")]
```

### ステップ0：W&Bのインストール

始めるには、ライブラリを入手します。
`wandb` は `pip` で簡単にインストールできます。

```python
!pip install wandb onnx -Uq
```

### ステップ1：W&Bのインポートとログイン

データをWebサービスに記録するには、ログインが必要です。

W&B を初めて使う場合は、
表示されるリンクから無料アカウントを作成してください。

```
import wandb

wandb.login()
```

## Experiment とパイプラインの定義

### `wandb.init` でメタデータとハイパーパラメーターの記録

まず最初に、experiment を定義します：
どのハイパーパラメーターを使うか？ この run にどんなメタデータを持たせるのか？

多くの場合、こうした情報は `config` 辞書型（または類似のオブジェクト）に保存し、
必要に応じて参照します。

この例では、いくつかのハイパーパラメーターだけを可変にして手動で設定していますが、
モデルのどんな部分でも `config` に含めることができます。

ここではメタデータとして MNIST データセットと
畳み込みアーキテクチャを利用している旨を記載しています。仮に後で同じプロジェクト内で
CIFAR の全結合アーキテクチャを使う場合にも、どの run が何だったか識別しやすくなります。

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

では、モデルのトレーニング用によくある全体のパイプラインを定義しましょう：

1. まずモデルと、関連するデータやオプティマイザーを `make` し、
2. モデルを `train` し
3. 最後に、トレーニングの成果を `test` します。

下でこれらの関数を実装していきます。

```python
def model_pipeline(hyperparameters):

    # wandb に開始を伝える
    with wandb.init(project="pytorch-demo", config=hyperparameters) as run:
        # すべての HP は run.config から参照。実行時のログと一致します。
        config = run.config

        # モデル・データ・最適化問題を生成
        model, train_loader, test_loader, criterion, optimizer = make(config)
        print(model)

        # 生成したものでモデルをトレーニング
        train(model, train_loader, criterion, optimizer, config)

        # 最終的な性能をテスト
        test(model, test_loader)

    return model
```

ここで標準的なパイプラインとの違いは、すべてが `wandb.init` の中で行われている点です。
この関数を呼ぶことで、コードと当社サーバー間の通信ラインがセットアップされます。

`config` 辞書を `wandb.init` に渡すと、
すべての情報が即座に記録され、experiment で使われたハイパーパラメーターの値を常に追跡できます。

常に記録した値がモデルで実際に使われるようにしたい場合は、
`run.config` のオブジェクトを使うことをおすすめします。
`make` の定義を次で確認してみてください。

> *補足*：私たちのコードは別プロセスで実行されるよう注意しています。
これにより、例えば巨大なシーモンスターがデータセンターを襲撃しても
あなたのコードがクラッシュすることはありません。
問題が解決した（たとえばクラーケンが深海に戻った）後は、`wandb sync` でデータ連携を再開できます。

```python
def make(config):
    # データの生成
    train, test = get_data(train=True), get_data(train=False)
    train_loader = make_loader(train, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)

    # モデルの生成
    model = ConvNet(config.kernels, config.classes).to(device)

    # 損失関数とオプティマイザーの生成
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, test_loader, criterion, optimizer
```

### データローディングとモデル定義

次はデータのロード方法と、どんなモデルかを指定します。

この部分はとても重要ですが、
`wandb` を使わなくても同じ内容なので、
詳しくは触れません。

```python
def get_data(slice=5, train=True):
    full_dataset = torchvision.datasets.MNIST(root=".",
                                              train=train, 
                                              transform=transforms.ToTensor(),
                                              download=True)
    #  [::slice] でスライスしたのと同じ
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

モデル定義は通常一番楽しい部分です。

ここでも `wandb` で何も変わらないため、
標準的な ConvNet アーキテクチャにしておきます。

ここは自由にいじって実験してみてください --
結果はすべて [wandb.ai](https://wandb.ai) 上に記録されます。

```python
# 一般的な畳み込みニューラルネットワーク

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

`model_pipeline` を進めると、次は `train` を定義します。

ここで `wandb` の2つの関数 `watch` と `log` が登場します。

## `run.watch()` で勾配・パラメータ追跡、`run.log()` でメトリクス記録

`run.watch` を使うと、トレーニング中に一定間隔（`log_freq` ステップ）で
モデルの重みや勾配などを自動で記録できます。

トレーニングを開始する前に呼び出してください。

それ以外は通常通り：エポックやバッチを回し、
forward / backward パスとオプティマイザーの適用を行います。

```python
def train(model, loader, criterion, optimizer, config):
    # wandb にモデルの振る舞い（勾配や重みなど）を監視させる
    run = wandb.init(project="pytorch-demo", config=config)
    run.watch(model, criterion, log="all", log_freq=10)

    # トレーニングの実行と wandb による追跡
    total_batches = len(loader) * config.epochs
    example_ct = 0  # 処理済みのサンプル数
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

    # オプティマイザーでステップ
    optimizer.step()

    return loss
```

唯一の違いは、ログ用コードです：
以前はメトリクスをターミナルに print していましたが、
今はその情報をそのまま `run.log()` に渡します。

`run.log()` にはキーを文字列とした辞書を渡します。
これが記録対象のオブジェクト名となり、値が内容です。
オプションでトレーニングの `step` を記載できます。

> *補足*：個人的には、モデルが見たサンプル数を step 指標に使っています。
バッチサイズが違っても比較しやすいからです。
step 数やバッチカウントでもOK。長いトレーニングの場合、エポック単位のログも有用です。

```python
def train_log(loss, example_ct, epoch):
    with wandb.init(project="pytorch-demo") as run:
        # 損失値・エポック番号を記録
        # ここで W&B にメトリクスをログ
        run.log({"epoch": epoch, "loss": loss}, step=example_ct)
        print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")
```

### テストロジックの定義

モデルのトレーニングが終わったら、テストしたいですよね。
例えば、実運用の新しいデータや、手作業で用意したサンプルに
適用してみる場面などです。

## (オプション) `run.save()` を呼び出す

このタイミングでモデルのアーキテクチャや最終パラメータを
ディスクに保存しておくとベストです。
最大限の互換性のため、モデルを
[Open Neural Network eXchange (ONNX) format](https://onnx.ai/) でエクスポートします。

そのファイル名を `run.save()` に渡すと、モデルパラメーターも
W&B サーバーに保存され、「どの .h5 や .pb がどの run なのか分からない」問題ともおさらばです。

モデルの保存・バージョン管理・配布をより高度に行う方法は [Artifacts ツール](https://www.wandb.com/artifacts) をご覧ください。

```python
def test(model, test_loader):
    model.eval()

    with wandb.init(project="pytorch-demo") as run:
        # テストデータでモデルを実行
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

        # モデルを ONNX 形式で保存
        torch.onnx.export(model, images, "model.onnx")
        run.save("model.onnx")
```

### トレーニング実行＆wandb.aiでライブで可視化

ここまででパイプライン全体ができ、その中に
W&B コードを数行追加できました。
これで実験管理が完全にトラッキングされた状態で実行できます。

いくつかリンクも合わせてレポートされます：
公式ドキュメント、
Project ページでプロジェクト内の run を整理し、
Run ページでは今回の run 結果一覧が公開されます。

Run ページでは以下のタブをぜひ確認してください：

1. **Charts**：トレーニング中のモデルの勾配、パラメータ値、損失値などが可視化されます
2. **System**：ディスクI/O、CPUやGPUなどシステムメトリクスが並びます（温度にもご注目！）
3. **Logs**：トレーニング中に標準出力した内容がコピーされています
4. **Files**：トレーニング完了後、`model.onnx` をクリックして [Netron モデルビューア](https://github.com/lutzroeder/netron) でネットワークが確認できます

run が終了し `with wandb.init` ブロックを抜けると、
セル出力にも結果のまとめが表示されます。

```python
# パイプラインでモデルを構築・学習・解析
model = model_pipeline(config)
```

### ハイパーパラメータ探索（Sweeps）の実行

今回の例では、1組のハイパーパラメーターのみ使いました。
しかし多くの ML ワークフローでは、
さまざまなハイパーパラメーターを試行錯誤することが重要です。

W&B Sweeps を使えば、ハイパーパラメーター探索と
モデルや最適化手法の空間探索を自動化できます。

[W&B Sweeps を使ったハイパーパラメータ最適化の Colab ノートブックデモ](https://wandb.me/sweeps-colab) をぜひご覧ください。

W&B でハイパーパラメーター探索を行うのはとても簡単。たった3つのステップです：

1. **スイープの定義:** 検索対象パラメータや探索戦略、最適化指標などを指定する [YAMLファイル]({{< relref "/guides/models/sweeps/define-sweep-configuration" >}}) または辞書を作成します。

2. **スイープの初期化:**  
`sweep_id = wandb.sweep(sweep_config)`

3. **sweep agent の実行:**  
`wandb.agent(sweep_id, function=train)`

これだけで簡単にハイパーパラメータ探索が実現できます。

{{< img src="/images/tutorials/pytorch-2.png" alt="PyTorch training dashboard" >}}

## 例ギャラリー

W&B でトラッキング＆可視化されているプロジェクト例は [Gallery →](https://app.wandb.ai/gallery) で公開中です。

## 上級設定
1. [環境変数]({{< relref "/guides/hosting/env-vars/" >}}): APIキーを環境変数に設定し、管理されたクラスター上でトレーニング実行も簡単に。
2. [オフラインモード]({{< relref "/support/kb-articles/run_wandb_offline.md" >}}): `dryrun` モードを使えばオフラインでトレーニング＆後で結果を同期。
3. [オンプレミス]({{< relref "/guides/hosting/hosting-options/self-managed" >}}): プライベートクラウドや自社インフラのエアギャップ環境にも W&B をインストールできます。学術用途からエンタープライズチームまで幅広く対応。
4. [Sweeps]({{< relref "/guides/models/sweeps/" >}}): ハイパーパラメータチューニング用の軽量ツールで簡単に探索を始められます。