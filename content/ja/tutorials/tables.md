---
title: 予測をテーブルで視覚化する
menu:
  tutorials:
    identifier: ja-tutorials-tables
    parent: null
weight: 2
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W&B_Tables_Quickstart.ipynb" >}}

これは PyTorch を使用して MNIST データ上でトレーニングの過程でモデルの予測を追跡、可視化、比較する方法をカバーしています。

次のことを学びます：
1. モデルトレーニングまたは評価中に `wandb.Table()` にメトリクス、画像、テキストなどを記録する
2. これらのテーブルを表示、ソート、フィルター、グループ化、結合、対話的にクエリし、探索する
3. モデルの予測または結果を比較する： 特定の画像、ハイパーパラメーター/モデルバージョン、またはタイムステップにわたって動的に比較する

## 例
### 特定の画像に対する予測スコアを比較する

[ライブ例: トレーニング1 エポック目と5 エポック目の予測を比較する →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#compare-predictions-after-1-vs-5-epochs)

{{< img src="/images/tutorials/tables-1.png" alt="1 epoch vs 5 epochs of training" >}}

ヒストグラムは、2つのモデル間のクラスごとのスコアを比較します。各ヒストグラムの上部の緑のバーは、1回のエポックしかトレーニングされていないモデル「CNN-2, 1 epoch」(ID 0)を表しています。下部の紫のバーは、5エポックでトレーニングされたモデル「CNN-2, 5 epochs」(ID 1)を表しています。画像は、モデル間で意見が分かれるケースにフィルターされています。例えば、最初の行では、「4」が1エポック後すべての可能な数字で高いスコアを得ていますが、5エポック後には正しいラベルで最も高いスコアを獲得し、他のものは非常に低いスコアを得ています。

### 時間経過に伴う主要なエラーに焦点を当てる
[ライブ例 →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#top-errors-over-time)

テストデータ全体で誤った予測（"guess" != "truth" でフィルター）を参照します。1回のトレーニングエポック後には229件の間違った予測がありますが、5エポック後には98件のみです。

{{< img src="/images/tutorials/tables-2.png" alt="side by side, 1 vs 5 epochs of training" >}}

### モデルのパフォーマンスを比較し、パターンを見つける

[ライブ例で詳細を確認する →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#false-positives-grouped-by-guess)

正解を除外し、推測ごとにグループ化して、画像の誤分類例と真のラベルの根底にある分布を参照することができます。2倍のレイヤーサイズと学習率のモデルバリアントが左にあり、ベースラインが右にあります。ベースラインは推測された各クラスの間違いをわずかに多くします。

{{< img src="/images/tutorials/tables-3.png" alt="grouped errors for baseline vs double variant" >}}

## サインアップまたはログイン

[サインアップまたはログイン](https://wandb.ai/login)して、ブラウザで実験を見て操作します。

この例では、Google Colab を便利なホスト環境として使用していますが、どこからでもトレーニングスクリプトを実行し、W&B の実験管理ツールを使用してメトリクスを視覚化することができます。

```python
!pip install wandb -qqq
```

アカウントにログインする

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

## 1. モデルとトレーニングスケジュールの定義

* 各エポックがトレーニングステップと検証（テスト）ステップで構成されるように、実行するエポック数を設定します。オプションで、各テストステップで記録するデータ量を設定します。ここでは、デモを簡略化するために、バッチ数とバッチごとに可視化する画像の数を少なく設定しています。
* シンプルな畳み込みニューラルネットワークを定義します（[pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial) のコードに従います）。
* PyTorch を使用してトレインとテストセットをロードする

```python
# 実行するエポック数
# 各エポックはトレーニングステップとテストステップを含みます。これで、ログするテスト予測のテーブル数が設定されます
EPOCHS = 1

# 各テストステップのテストデータからログするバッチ数
# (シンプルなデモのためにデフォルトを低く設定)
NUM_BATCHES_TO_LOG = 10 #79

# 各テストバッチごとにログする画像の数
# (シンプルなデモのためにデフォルトを低く設定)
NUM_IMAGES_PER_BATCH = 32 #128

# トレーニング設定とハイパーパラメーター
NUM_CLASSES = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
L1_SIZE = 32
L2_SIZE = 64
# これを変更すると、隣接するレイヤーの形状を変更する必要があります
CONV_KERNEL_SIZE = 5

# 二層の畳み込みニューラルネットワークを定義
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
        # 指定されたレイヤーの形状を確認するためにコメント解除:
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

## 2. トレーニングを実行し、テスト予測をログします

各エポックごとにトレーニングステップとテストステップを実行します。各テストステップごとに、テスト予測を保存する `wandb.Table()` を作成します。これらはブラウザで視覚化され、動的にクエリされ、並べて比較されます。

```python
# W&B: このモデルのトレーニングを追跡する新しい run を初期化します
wandb.init(project="table-quickstart")

# W&B: config を使用してハイパーパラメーターをログ
cfg = wandb.config
cfg.update({"epochs" : EPOCHS, "batch_size": BATCH_SIZE, "lr" : LEARNING_RATE,
            "l1_size" : L1_SIZE, "l2_size": L2_SIZE,
            "conv_kernel" : CONV_KERNEL_SIZE,
            "img_count" : min(10000, NUM_IMAGES_PER_BATCH*NUM_BATCHES_TO_LOG)})

# モデル、損失関数、オプティマイザーを定義
model = ConvNet(NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# テスト画像のバッチの予測をログするための便利な関数
def log_test_predictions(images, labels, outputs, predicted, test_table, log_counter):
  # すべてのクラスの信頼スコアを取得
  scores = F.softmax(outputs.data, dim=1)
  log_scores = scores.cpu().numpy()
  log_images = images.cpu().numpy()
  log_labels = labels.cpu().numpy()
  log_preds = predicted.cpu().numpy()
  # 画像の順序に基づいて ID を追加
  _id = 0
  for i, l, p, s in zip(log_images, log_labels, log_preds, log_scores):
    # データテーブルに必要な情報を追加:
    # ID、画像ピクセル、モデルの推測、真のラベル、すべてのクラスのスコア
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
        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
  
        # W&B: トレーニングステップでの損失をログし、UIでライブ視覚化
        wandb.log({"loss" : loss})
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                .format(epoch+1, EPOCHS, i+1, total_step, loss.item()))
            

    # W&B: 各テストステップでの予測を保存するためのテーブルを作成
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
        # W&B: トレーニングエポックの精度をログして、UIで可視化
        wandb.log({"epoch" : epoch, "acc" : acc})
        print('Test Accuracy of the model on the 10000 test images: {} %'.format(acc))

    # W&B: 予測テーブルを wandb にログ
    wandb.log({"test_predictions" : test_table})

# W&B: run を完了としてマークする（マルチセルノートブックに便利）
wandb.finish()
```

## 次は何ですか？
次のチュートリアルでは、[W&B Sweeps を使用してハイパーパラメーターを最適化する方法]({{< relref path="sweeps.md" lang="ja" >}})を学びます。