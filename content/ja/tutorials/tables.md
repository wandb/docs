---
title: テーブルで予測を可視化する
menu:
  tutorials:
    identifier: ja-tutorials-tables
    parent: null
weight: 2
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W&B_Tables_Quickstart.ipynb" >}}

このガイドでは、PyTorch と MNIST データを使って、トレーニングの過程でモデルの予測を追跡・可視化・比較する方法を解説します。

このチュートリアルで学べること：
1. モデルトレーニングまたは評価中に、メトリクス・画像・テキストなどを `wandb.Table()` にログする方法
2. これらのテーブルを表示、ソート、フィルタ、グループ化、結合、インタラクティブなクエリ、探索する方法
3. モデル予測や結果を比較する方法：特定の画像、ハイパーパラメーター／モデルバージョン、またはタイムステップ単位で動的に比較できます

## 例
### 特定の画像に対する予測スコアを比較する

[ライブ例：1 エポックと 5 エポックのトレーニング後の予測比較 →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#compare-predictions-after-1-vs-5-epochs)

{{< img src="/images/tutorials/tables-1.png" alt="トレーニングエポック比較" >}}

このヒストグラムは各クラスごとのスコアを2つのモデル間で比較しています。一番上の緑色のバーはモデル「CNN-2, 1 epoch」(id 0) で、1 エポックのみトレーニングされています。一番下の紫色のバーは5エポックトレーニングされたモデル「CNN-2, 5 epochs」(id 1) です。画像は2つのモデルの予測が異なるケースに絞り込んでいます。例えば 1 行目では「4」という画像が 1 エポック時点では全ての数字において高いスコアを持っていますが、5 エポック後には正しいラベルに最も高いスコアがつき、他の数字には非常に低いスコアがついています。

### 時間経過による最大の誤分類に注目する
[ライブ例はこちら →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#top-errors-over-time)

間違った予測（"guess" != "truth" の行でフィルタ）をテストデータ全体で確認できます。1 エポック後には 229 件の誤った予測がありますが、5 エポック後には 98 件に減少しています。

{{< img src="/images/tutorials/tables-2.png" alt="エポックの比較表示" >}}

### モデルの性能を比較し、パターンを見つける

[ライブ例で詳細を確認 →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#false-positives-grouped-by-guess)

正解を除外し、予測された数字ごとにグループ化して、誤分類された画像の例と本来のラベルの分布を比較できます。左のモデルはレイヤーサイズと学習率が2倍のバリアント、右はベースラインモデルです。ベースラインモデルは各クラスの予測でわずかに多くミスをしています。

{{< img src="/images/tutorials/tables-3.png" alt="エラー比較" >}}

## サインアップまたはログイン

[W&B へサインアップまたはログイン](https://wandb.ai/login) し、ブラウザ上で自分の Experiments を見て操作しましょう。

このチュートリアルでは Google Colab のホスト環境を使っていますが、お手持ちの環境からトレーニングスクリプトを実行し、W&B の実験管理ツールでメトリクスを可視化できます。

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

依存関係のインストール、MNIST データセットのダウンロード、PyTorch でトレーニング・テストデータセットの作成を行います。

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T 
import torch.nn.functional as F


device = "cuda:0" if torch.cuda.is_available() else "cpu"

# トレーニングおよびテストのデータローダを作成
def get_dataloader(is_train, batch_size, slice=5):
    "トレーニング用のデータローダ取得"
    ds = torchvision.datasets.MNIST(root=".", train=is_train, transform=T.ToTensor(), download=True)
    loader = torch.utils.data.DataLoader(dataset=ds, 
                                         batch_size=batch_size, 
                                         shuffle=True if is_train else False, 
                                         pin_memory=True, num_workers=2)
    return loader
```

## 1. モデルとトレーニングスケジュールの定義

* エポック数を指定します。各エポックはトレーニングとバリデーション（テスト）で構成されます。テストごとにログするデータ量も調整できます。ここではデモを簡略化するためバッチ数とイメージ数を少なめに設定しています。
* シンプルな畳み込みニューラルネットワークを定義します（[pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial) コードを参考）。
* PyTorch でトレーニングとテストセットをロードします。

```python
# 実行するエポック数
# 各エポックにトレーニングとテストが含まれるので、
# テスト予測用のテーブル数もこの回数だけ作成されます
EPOCHS = 1

# 各テスト時にログするバッチ数
# （デモを簡単にするためデフォルトは少なめ）
NUM_BATCHES_TO_LOG = 10 #79

# 各テストバッチにつきログする画像数
# （デモ用に少なめ）
NUM_IMAGES_PER_BATCH = 32 #128

# トレーニングの設定とハイパーパラメーター
NUM_CLASSES = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
L1_SIZE = 32
L2_SIZE = 64
# この値を変える場合は、隣接するレイヤーの形状も調整が必要です
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
        # レイヤーごとの形状を見たい場合はこのコメントアウトをはずす
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

各エポックごとにトレーニングとテストを行います。テストごとに `wandb.Table()` を作成して予測を保存します。これはブラウザ上で可視化や動的なクエリ・比較が可能です。

```python
# テスト画像バッチの予測をログするための補助関数
def log_test_predictions(images, labels, outputs, predicted, test_table, log_counter):
  # 全クラスの信頼度スコアを取得
  scores = F.softmax(outputs.data, dim=1)
  log_scores = scores.cpu().numpy()
  log_images = images.cpu().numpy()
  log_labels = labels.cpu().numpy()
  log_preds = predicted.cpu().numpy()
  # 画像順に id を付与
  _id = 0
  for i, l, p, s in zip(log_images, log_labels, log_preds, log_scores):
    # 必要な情報をテーブルに追加
    # id, 画像ピクセル, モデルの予測, 正解ラベル, 全クラスのスコア
    img_id = str(_id) + "_" + str(log_counter)
    test_table.add_data(img_id, wandb.Image(i), p, l, *s)
    _id += 1
    if _id == NUM_IMAGES_PER_BATCH:
      break

# W&B: このモデルのトレーニングをトラッキングするための新しい run を初期化
with wandb.init(project="table-quickstart") as run:

    # W&B: ハイパーパラメーターを config で記録
    cfg = run.config
    cfg.update({"epochs" : EPOCHS, "batch_size": BATCH_SIZE, "lr" : LEARNING_RATE,
                "l1_size" : L1_SIZE, "l2_size": L2_SIZE,
                "conv_kernel" : CONV_KERNEL_SIZE,
                "img_count" : min(10000, NUM_IMAGES_PER_BATCH*NUM_BATCHES_TO_LOG)})

    # モデル、損失関数、オプティマイザーの定義
    model = ConvNet(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # モデルのトレーニング実行
    total_step = len(train_loader)
    for epoch in range(EPOCHS):
        # トレーニング
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # backward と最適化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            # W&B: トレーニングステップごとの損失をログ（UI でリアルタイム表示）
            run.log({"loss" : loss})
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, EPOCHS, i+1, total_step, loss.item()))
                

        # W&B: 各テストごとに予測を保存するための Table を作成
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
            # W&B: エポックごとの正答率をログ（UI で可視化）
            run.log({"epoch" : epoch, "acc" : acc})
            print('Test Accuracy of the model on the 10000 test images: {} %'.format(acc))

        # W&B: 予測テーブルを wandb にログ
        run.log({"test_predictions" : test_table})
```

## 次は？
次のチュートリアルでは、[W&B Sweeps を使ったハイパーパラメーターの最適化方法]({{< relref path="sweeps.md" lang="ja" >}}) を学びます。