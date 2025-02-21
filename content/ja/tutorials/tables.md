---
title: Visualize predictions with tables
menu:
  tutorials:
    identifier: ja-tutorials-tables
    parent: null
weight: 2
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W&B_Tables_Quickstart.ipynb" >}}

これは、MNIST データを用いた PyTorch によるトレーニングの過程でのモデル予測を追跡、可視化、比較する方法をカバーしています。

この学習で身につけること:
1. モデルトレーニングや評価の際に、`wandb.Table()` にメトリクスや画像、テキストなどをログする方法
2. これらのテーブルを表示、ソート、フィルタ、グループ、ジョイン、インタラクティブなクエリ、探求する方法
3. モデルの予測や結果を比較する: 特定の画像、ハイパーパラメーター/モデルバージョン、または時間のステップごとに動的に比較する方法

## Examples
### 特定の画像の予測スコアを比較する

[ライブ例: 1 エポックと 5 エポックのトレーニング後の予測を比較する →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#compare-predictions-after-1-vs-5-epochs)

{{< img src="/images/tutorials/tables-1.png" alt="1 epoch vs 5 epochs of training" >}}

ヒストグラムは2つのモデル間のクラス別スコアを比較しています。各ヒストグラムの上の緑のバーは、1 エポック (id 0) だけトレーニングされたモデル "CNN-2, 1 epoch" を表し、下の紫のバーは 5 エポック (id 1) トレーニングされたモデル "CNN-2, 5 epochs" を表します。画像は、モデルが異なる場合にフィルタされています。例えば、最初の行では、"4" が 1 エポック後にはすべての可能な数字で高スコアを得ていますが、5 エポック後には正しいラベルで最も高いスコアを得て、他のラベルでは非常に低いスコアになっています。

### 时间経過による主要エラーに焦点を当てる
[ライブ例 →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#top-errors-over-time)

全テストデータで､誤った予測 (「予測」と「真実」が異なる行にフィルタ) を参照します。1 エポックのトレーニング後には 229 の間違った予測がありましたが、5 エポック後には 98 しかありません。

{{< img src="/images/tutorials/tables-2.png" alt="side by side, 1 vs 5 epochs of training" >}}

### モデルパフォーマンスを比較し、パターンを見つける

[ライブ例の詳細を見る →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#false-positives-grouped-by-guess)

正解をフィルタした後、予測によってグループ化して、誤分類された画像の例と真のラベルの基礎分布を 2つのモデルでサイドバイサイドで参照します。左側には層サイズと学習率が 2倍になったモデルバリアントがあり、右側にはベースラインモデルがあります。ベースラインは、推測されたクラスごとにわずかに多くの誤りを犯すことに注意してください。

{{< img src="/images/tutorials/tables-3.png" alt="grouped errors for baseline vs double variant" >}}

## サインアップまたはログインする

[サインアップまたはログイン](https://wandb.ai/login)して、W&Bで自分の実験をブラウザで参照し、相互作用を持つことができます。

この例では、Google Colab を便利なホスティング環境として使用していますが、自分のトレーニングスクリプトをどこからでも実行し、W&B の実験管理ツールでメトリクスを可視化できます。

```python
!pip install wandb -qqq
```

自分のアカウントにログインする


```python

import wandb
wandb.login()

WANDB_PROJECT = "mnist-viz"
```

## 0. セットアップ

依存関係をインストールし、MNIST をダウンロードし、PyTorch を使用してトレインとテストのデータセットを作成します。

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T 
import torch.nn.functional as F


device = "cuda:0" if torch.cuda.is_available() else "cpu"

# トレインとテストのデータローダーを作成する
def get_dataloader(is_train, batch_size, slice=5):
    "トレーニングデータローダーを取得する"
    ds = torchvision.datasets.MNIST(root=".", train=is_train, transform=T.ToTensor(), download=True)
    loader = torch.utils.data.DataLoader(dataset=ds, 
                                         batch_size=batch_size, 
                                         shuffle=True if is_train else False, 
                                         pin_memory=True, num_workers=2)
    return loader
```

## 1. モデルとトレーニングスケジュールを定義する

* エポック数を設定し、各エポックはトレーニングステップと検証 (テスト) ステップで構成されます。オプションで、テストステップごとにログするデータの量を設定できます。このデモを簡単にするために、バッチ数とバッチごとに可視化する画像の数は低く設定されています。
* シンプルな畳み込みニューラルネットワークを定義する ([pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial) のコードに準拠)。
* PyTorch を使用してトレーニングとテストのセットをロードします。

```python
# 実行するエポック数
# 各エポックはトレーニングステップとテストステップを含むため、
# テスト予測をログするテーブルの数を設定します
EPOCHS = 1

# 各テストステップでテストデータからログするバッチの数
# (デモを簡略化するためにデフォルトで低く設定)
NUM_BATCHES_TO_LOG = 10 #79

# 各テストバッチでログする画像の数
# (デモを簡略化するためにデフォルトで低く設定)
NUM_IMAGES_PER_BATCH = 32 #128

# トレーニング設定とハイパーパラメーター
NUM_CLASSES = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
L1_SIZE = 32
L2_SIZE = 64
# これを変更する場合は、隣接する層の形を変更する必要があります
CONV_KERNEL_SIZE = 5

# 2 層の畳み込みニューラルネットワークを定義する
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, L1_SIZE, CONV_KERNEL_SIZE, stride=1, padding=2),
            nn.BatchNorm2d(L1_SIZE),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(L1_SIZE, L2_SIZE, CONV_KERNEL_SIZE, stride=1, padding=2),
            nn.BatchNorm2d(L2_SIZE),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*L2_SIZE, NUM_CLASSES)
        self.softmax = nn.Softmax(NUM_CLASSES)

    def forward(self, x):
        # 特定の層の形を確認するには以下を解除してください:
        #print("x: ", x.size())
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

train_loader = get_dataloader(is_train=True, batch_size=BATCH_SIZE)
test_loader = get_dataloader(is_train=False, batch_size=2*BATCH_SIZE)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

## 2. トレーニングを実行し、テスト予測をログする

各エポックについて、トレーニングステップとテストステップを実行します。各テストステップでは、`wandb.Table()` を作成してテスト予測を保存します。これらはブラウザで視覚化され、動的にクエリされ、並べて比較できます。

```python
# ✨ W&B: このモデルのトレーニングを追跡するために新しい run を初期化
wandb.init(project="table-quickstart")

# ✨ W&B: ハイパーパラメーターを config を用いてログ
cfg = wandb.config
cfg.update({"epochs" : EPOCHS, "batch_size": BATCH_SIZE, "lr" : LEARNING_RATE,
            "l1_size" : L1_SIZE, "l2_size": L2_SIZE,
            "conv_kernel" : CONV_KERNEL_SIZE,
            "img_count" : min(10000, NUM_IMAGES_PER_BATCH*NUM_BATCHES_TO_LOG)})

# モデル、損失、オプティマイザーを定義
model = ConvNet(NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# テスト画像のバッチに対する予測をログするための便利な関数
def log_test_predictions(images, labels, outputs, predicted, test_table, log_counter):
  # すべてのクラスに対する信頼スコアを取得
  scores = F.softmax(outputs.data, dim=1)
  log_scores = scores.cpu().numpy()
  log_images = images.cpu().numpy()
  log_labels = labels.cpu().numpy()
  log_preds = predicted.cpu().numpy()
  # 画像の順序に基づいて id を付ける
  _id = 0
  for i, l, p, s in zip(log_images, log_labels, log_preds, log_scores):
    # データテーブルに必要な情報を追加する:
    # id、画像ピクセル、モデルの推測、正解ラベル、すべてのクラスのスコア
    img_id = str(_id) + "_" + str(log_counter)
    test_table.add_data(img_id, wandb.Image(i), p, l, *s)
    _id += 1
    if _id == NUM_IMAGES_PER_BATCH:
      break

# モデルをトレーニングする
total_step = len(train_loader)
for epoch in range(EPOCHS):
    # トレーニングステップ
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        # backward と optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
  
        # ✨ W&B: トレーニングステップごとの損失をログし、UIでライブで可視化
        wandb.log({"loss" : loss})
        if (i+1) % 100 == 0:
            print ('エポック [{}/{}], ステップ [{}/{}], 損失: {:.4f}'
                .format(epoch+1, EPOCHS, i+1, total_step, loss.item()))
            

    # ✨ W&B: 各テストステップの予測を格納するためのテーブルを作成
    columns=["id", "image", "guess", "truth"]
    for digit in range(10):
      columns.append("score_" + str(digit))
    test_table = wandb.Table(columns=columns)

    # モデルをテスト
    model.eval()
    log_counter = 0
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            if log_counter < NUM_BATCHES_TO_LOG:
              log_test_predictions(images, labels, outputs, predicted, test_table, log_counter)
              log_counter += 1
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        # ✨ W&B: トレーニングエポック中の正確性をログして、UIで可視化
        wandb.log({"epoch" : epoch, "acc" : acc})
        print('モデルの10000のテスト画像における正確性: {} %'.format(acc))

    # ✨ W&B: 予測テーブルを wandb にログ
    wandb.log({"test_predictions" : test_table})

# ✨ W&B: run を完了としてマーク (マルチセルノートブックで便利)
wandb.finish()
```

## What's next?
次のチュートリアルでは、W&B Sweeps を使用してハイパーパラメーターを最適化する方法を学びます:
## 👉 [ハイパーパラメータを最適化する]({{< relref path="sweeps.md" lang="ja" >}})