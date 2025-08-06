---
title: テーブルで予測を可視化する
menu:
  tutorials:
    identifier: tables
    parent: null
weight: 2
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W&B_Tables_Quickstart.ipynb" >}}

このクイックスタートでは、PyTorch で MNIST データを使い、トレーニング中にモデルの予測を追跡・可視化・比較する方法を説明します。

このチュートリアルで学べること:

1. モデルトレーニングや評価時に `wandb.Table()` にメトリクスや画像、テキストなどをログする方法
2. こうして記録したテーブルを表示・ソート・フィルタ・グループ化・結合・対話的に検索・探索する方法
3. モデルの予測や結果を比較する方法：特定の画像、ハイパーパラメーターやモデルバージョン、タイムステップごとにダイナミックに比較できます

## 例
### 特定の画像で予測スコアを比較する

[ライブ例：トレーニングエポック 1 回後と 5 回後での予測を比較する →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#compare-predictions-after-1-vs-5-epochs)

{{< img src="/images/tutorials/tables-1.png" alt="トレーニングエポック比較" >}}

このヒストグラムは、2つのモデル間のクラスごとのスコアを比較したものです。各ヒストグラムの上側の緑色のバーは「CNN-2, 1 epoch」（id 0）で、エポック1回だけトレーニングしたモデルを表します。下側の紫色のバーは「CNN-2, 5 epochs」（id 1）で、エポック5回トレーニングしたものです。画像は両モデルの予測が異なるケースだけに絞って表示しています。たとえば1行目では、「4」という数字が1エポック後は全ての数字で高いスコアを持っていますが、5エポック後には正しいラベルで高いスコアを、その他の数字では非常に低いスコアになっています。

### 時間経過でトップの誤分類に注目する
[ライブ例 →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#top-errors-over-time)

予測が間違っているもの（"guess" != "truth" の行でフィルタ）をテストデータ全体で表示します。たとえばエポック1回後には229件、エポック5回後には98件の誤った予測があることがわかります。

{{< img src="/images/tutorials/tables-2.png" alt="エポック比較サイドバイサイド" >}}

### モデルのパフォーマンスを並べて比較し、パターンを見つける

[ライブ例で詳細を見る →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#false-positives-grouped-by-guess)

正解だったデータを除外し、予測したクラスごとにグループ化することで、分類ミスの例や正解ラベルの分布が確認できます。左側はレイヤーサイズと学習率を2倍にしたモデルバリアント、右側はベースラインモデルです。ベースラインの方が、各推測クラスについてわずかにエラーが多いことがわかります。

{{< img src="/images/tutorials/tables-3.png" alt="エラー比較" >}}

## サインアップまたはログイン

[W&B にサインアップまたはログイン](https://wandb.ai/login) して、ブラウザから自分の Experiments を確認・操作してみましょう。

この例では Google Colab を使っていますが、自分の環境からトレーニングスクリプトを実行し、W&B の実験管理ツールでメトリクスを可視化することもできます。

```python
!pip install wandb -qqq
```

自分のアカウントにログインします

```python

import wandb
wandb.login()

WANDB_PROJECT = "mnist-viz"
```

## 0. セットアップ

依存関係のインストール、MNIST のダウンロード、PyTorch を使ったトレーニング・テストデータセットの作成を行います。

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T 
import torch.nn.functional as F


device = "cuda:0" if torch.cuda.is_available() else "cpu"

# トレーニング・テスト用のデータローダーを作成
def get_dataloader(is_train, batch_size, slice=5):
    "トレーニング用のデータローダを取得"
    ds = torchvision.datasets.MNIST(root=".", train=is_train, transform=T.ToTensor(), download=True)
    loader = torch.utils.data.DataLoader(dataset=ds, 
                                         batch_size=batch_size, 
                                         shuffle=True if is_train else False, 
                                         pin_memory=True, num_workers=2)
    return loader
```

## 1. モデルとトレーニングスケジュールの定義

* 実行するエポック数を設定します。各エポックはトレーニングステップと検証（テスト）ステップで構成されます。テストステップごとにログするデータ量もオプションで設定できます。ここではデモを簡単にするために、バッチ数とバッチごとに可視化する画像数を少なくしています。
* シンプルな畳み込みニューラルネットを定義します（[pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial) のコードに沿っています）。
* PyTorch を使ってトレーニングセットとテストセットをロードします

```python
# 実行するエポック数
# 各エポックではトレーニングステップ＋テストステップがあり、
# それぞれのテストステップごとにテスト予測用のテーブルをログします
EPOCHS = 1

# 各テストステップでテストデータからログするバッチ数
# （デモ簡略化のため少数に設定）
NUM_BATCHES_TO_LOG = 10 #79

# テストバッチごとにログする画像数
# （デモ簡略化のため少数に設定）
NUM_IMAGES_PER_BATCH = 32 #128

# トレーニング設定とハイパーパラメーター
NUM_CLASSES = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
L1_SIZE = 32
L2_SIZE = 64
# この値を変更する場合、隣接するレイヤーの形状も変更が必要です
CONV_KERNEL_SIZE = 5

# 2層の畳み込みニューラルネットワークを定義
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
        # レイヤーの形状を確認したい場合はコメントを外す
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

## 2. トレーニングの実行とテスト予測のログ

各エポックごとにトレーニングステップとテストステップを行います。各テストステップごとにテスト予測を格納するための `wandb.Table()` を作成します。これによって、ブラウザ上での動的な可視化や検索・比較が可能になります。

```python
# テスト画像バッチの予測をログするための関数
def log_test_predictions(images, labels, outputs, predicted, test_table, log_counter):
  # 全クラスの信頼度スコアを取得
  scores = F.softmax(outputs.data, dim=1)
  log_scores = scores.cpu().numpy()
  log_images = images.cpu().numpy()
  log_labels = labels.cpu().numpy()
  log_preds = predicted.cpu().numpy()
  # 画像の順番に合わせてIDを付与
  _id = 0
  for i, l, p, s in zip(log_images, log_labels, log_preds, log_scores):
    # 必要な情報をデータテーブルに追加
    # id, 画像ピクセル, モデルの予測, 正解ラベル, 全クラスのスコア
    img_id = str(_id) + "_" + str(log_counter)
    test_table.add_data(img_id, wandb.Image(i), p, l, *s)
    _id += 1
    if _id == NUM_IMAGES_PER_BATCH:
      break

# W&B: このモデルのトレーニングを追跡する run を初期化
with wandb.init(project="table-quickstart") as run:

    # W&B: ハイパーパラメーターを config でログ
    cfg = run.config
    cfg.update({"epochs" : EPOCHS, "batch_size": BATCH_SIZE, "lr" : LEARNING_RATE,
                "l1_size" : L1_SIZE, "l2_size": L2_SIZE,
                "conv_kernel" : CONV_KERNEL_SIZE,
                "img_count" : min(10000, NUM_IMAGES_PER_BATCH*NUM_BATCHES_TO_LOG)})

    # モデル・損失関数・オプティマイザーの定義
    model = ConvNet(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # モデルトレーニング
    total_step = len(train_loader)
    for epoch in range(EPOCHS):
        # トレーニングステップ
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # backward および最適化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            # W&B: トレーニングステップごとの損失をリアルタイム可視化用にログ
            run.log({"loss" : loss})
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, EPOCHS, i+1, total_step, loss.item()))
                

        # W&B: 各テストステップごとに予測を格納する Table を作成
        columns=["id", "image", "guess", "truth"]
        for digit in range(10):
        columns.append("score_" + str(digit))
        test_table = wandb.Table(columns=columns)

        # モデルのテスト
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
            # W&B: エポックごとの精度を UI 表示用にログ
            run.log({"epoch" : epoch, "acc" : acc})
            print('Test Accuracy of the model on the 10000 test images: {} %'.format(acc))

        # W&B: テスト予測テーブルを wandb にログ
        run.log({"test_predictions" : test_table})
```

## 次は？
次のチュートリアルでは、[W&B Sweeps を利用したハイパーパラメーター最適化の方法]({{< relref "sweeps.md" >}})を学びます。