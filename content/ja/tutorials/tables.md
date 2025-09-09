---
title: テーブル を使って 予測 を可視化する
menu:
  tutorials:
    identifier: ja-tutorials-tables
    parent: null
weight: 2
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W&B_Tables_Quickstart.ipynb" >}}

このチュートリアルでは、PyTorch を使って MNIST データ上で トレーニング の過程における モデル の 予測 を追跡・可視化・比較する方法を解説します。

学べること:
1. モデルトレーニングや評価の最中に `wandb.Table()` に メトリクス、画像、テキストなどを ログ する
2. これらのテーブルを表示、ソート、フィルター、グループ化、結合、対話的にクエリして探索する
3. モデルの 予測 や 結果 を比較する: 特定の画像、ハイパーパラメーター / モデル バージョン、または時系列で動的に

## 例
### 特定の画像に対する予測スコアを比較

[ライブ例: トレーニング 1 エポックと 5 エポック後の 予測 を比較 →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#compare-predictions-after-1-vs-5-epochs)

{{< img src="/images/tutorials/tables-1.png" alt="トレーニング エポックの比較" >}}

ヒストグラムは 2 つの モデル のクラス別スコアを比較しています。各ヒストグラムの上の緑のバーは 1 エポックだけ トレーニング した モデル "CNN-2, 1 epoch"（id 0）を表し、下の紫のバーは 5 エポック トレーニング した モデル "CNN-2, 5 epochs"（id 1）を表します。画像は 2 つの モデル の予測が不一致のケースにフィルターされています。例えば 1 行目では、1 エポック後は「4」がほぼすべての数字で高得点でしたが、5 エポック後には正しいラベルで最も高く、他は非常に低くなっています。

### 時間経過で主な誤りに注目
[ライブ例 →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#top-errors-over-time)

テストデータ全体に対する誤った 予測（"guess" != "truth" の行にフィルター）を確認します。1 回目の トレーニング エポック後には 229 件の誤答がありますが、5 エポック後は 98 件に減っています。

{{< img src="/images/tutorials/tables-2.png" alt="エポックの横並び比較" >}}

### モデル の性能を比較し、パターンを見つける

[ライブ例で詳細を見る →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#false-positives-grouped-by-guess)

正解を除外してから guess ごとにグループ化し、誤分類画像の例と真のラベル分布を、2 つの モデル を横並びで確認します。層のサイズと学習率が 2 倍の モデル バリアントが左、ベースラインが右です。ベースラインの方が各推定クラスでわずかに誤りが多い点に注目してください。

{{< img src="/images/tutorials/tables-3.png" alt="エラー比較" >}}

## サインアップまたはログイン

[Sign up or login](https://wandb.ai/login) して W&B でブラウザからあなたの 実験 を見たり操作したりしましょう。

この例では Google Colab を便利なホスト型の 環境 として使っていますが、どこからでも自分の トレーニングスクリプト を実行し、W&B の 実験管理 ツールで メトリクス を可視化できます。


```python
!pip install wandb -qqq
```

あなたのアカウントに ログ する


```python

import wandb
wandb.login()

WANDB_PROJECT = "mnist-viz"
```

## 0. セットアップ

依存関係をインストールし、MNIST をダウンロードして、PyTorch で トレーニング 用と テスト 用の データセット を作成します。 


```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T 
import torch.nn.functional as F


device = "cuda:0" if torch.cuda.is_available() else "cpu"

# トレーニング と テスト のデータローダーを作成
def get_dataloader(is_train, batch_size, slice=5):
    "Get a training dataloader"
    ds = torchvision.datasets.MNIST(root=".", train=is_train, transform=T.ToTensor(), download=True)
    loader = torch.utils.data.DataLoader(dataset=ds, 
                                         batch_size=batch_size, 
                                         shuffle=True if is_train else False, 
                                         pin_memory=True, num_workers=2)
    return loader
```

## 1. モデルと トレーニング スケジュールを定義

* 実行する エポック 数を設定します。各 エポック は トレーニング ステップと検証（テスト）ステップで構成されます。任意で各テストステップで ログ する データ 量を設定できます。ここでは、可視化するバッチ数と各バッチの画像枚数を小さくして デモ を簡単にしています。 
* 単純な畳み込み ニューラルネットワーク を定義します（[pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial) の コード に準拠）。
* PyTorch でトレインとテストのセットを読み込みます。



```python
# 実行する エポック 数
# 各 エポック はトレーニング ステップとテスト ステップを含むため、これは
# ログ するテスト 予測 テーブルの数を設定
EPOCHS = 1

# 各テストステップでテストデータから ログ するバッチ数
# （デモを簡単にするため既定値は小さめ）
NUM_BATCHES_TO_LOG = 10 #79

# 各テストバッチで ログ する画像枚数
# （デモを簡単にするため既定値は小さめ）
NUM_IMAGES_PER_BATCH = 32 #128

# トレーニング の 設定 と ハイパーパラメーター
NUM_CLASSES = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
L1_SIZE = 32
L2_SIZE = 64
# これを変更すると隣接レイヤーの形状も変更が必要な場合あり
CONV_KERNEL_SIZE = 5

# 2 層の畳み込み ニューラルネットワーク を定義
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
        # コメントアウトを外すと各層のテンソル形状を確認できます
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

## 2. トレーニングを実行し、テスト 予測 を ログ する

各 エポック ごとにトレーニング ステップとテスト ステップを実行します。各テスト ステップでは テスト 予測 を保存するための `wandb.Table()` を作成します。これらはブラウザで可視化・動的クエリ・横並び比較できます。


```python
# テスト画像のバッチに対する 予測 を ログ するためのユーティリティ関数
def log_test_predictions(images, labels, outputs, predicted, test_table, log_counter):
  # 全クラスの信頼度スコアを取得
  scores = F.softmax(outputs.data, dim=1)
  log_scores = scores.cpu().numpy()
  log_images = images.cpu().numpy()
  log_labels = labels.cpu().numpy()
  log_preds = predicted.cpu().numpy()
  # 画像の順序に基づく id を付与
  _id = 0
  for i, l, p, s in zip(log_images, log_labels, log_preds, log_scores):
    # データテーブルに必要な情報を追加:
    # id、画像、モデルの推測、正解ラベル、全クラスのスコア
    img_id = str(_id) + "_" + str(log_counter)
    test_table.add_data(img_id, wandb.Image(i), p, l, *s)
    _id += 1
    if _id == NUM_IMAGES_PER_BATCH:
      break

# W&B: この モデル の トレーニング を追跡する新しい Run を初期化
with wandb.init(project="table-quickstart") as run:

    # W&B: config を使って ハイパーパラメーター を ログ
    cfg = run.config
    cfg.update({"epochs" : EPOCHS, "batch_size": BATCH_SIZE, "lr" : LEARNING_RATE,
                "l1_size" : L1_SIZE, "l2_size": L2_SIZE,
                "conv_kernel" : CONV_KERNEL_SIZE,
                "img_count" : min(10000, NUM_IMAGES_PER_BATCH*NUM_BATCHES_TO_LOG)})

    # モデル、損失関数、オプティマイザー を定義
    model = ConvNet(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # モデルを トレーニング
    total_step = len(train_loader)
    for epoch in range(EPOCHS):
        # トレーニング ステップ
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # 逆伝播して最適化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            # W&B: トレーニング ステップにおける損失を ログ（UI にリアルタイム表示）
            run.log({"loss" : loss})
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, EPOCHS, i+1, total_step, loss.item()))
                

        # W&B: 各 テスト ステップの 予測 を保存する Table を作成
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
            # W&B: トレーニング エポック全体の精度を ログ（UI で可視化）
            run.log({"epoch" : epoch, "acc" : acc})
            print('Test Accuracy of the model on the 10000 test images: {} %'.format(acc))

        # W&B: 予測 テーブルを wandb に ログ
        run.log({"test_predictions" : test_table})
```

## 次のステップは？
次のチュートリアルでは、[W&B Sweeps を使って ハイパーパラメーター を最適化する方法]({{< relref path="sweeps.md" lang="ja" >}}) を学びます。