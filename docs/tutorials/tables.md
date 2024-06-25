
# 予測を可視化する

[**Colabノートブックで試す →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W&B_Tables_Quickstart.ipynb)

このセクションでは、PyTorchを使用してMNISTデータ上でトレーニングを行う過程で、モデルの予測を追跡し、可視化し、比較する方法について説明します。

以下のことを学びます:
1. モデルトレーニングや評価中に `wandb.Table()` にメトリクス、画像、テキストなどをログする方法
2. これらのテーブルを表示、ソート、フィルタ、グループ、結合、対話的にクエリ、探索する方法
3. モデルの予測や結果を比較する方法: 特定の画像、ハイパーパラメーター/モデルバージョン、または時間の経過で動的に比較

# 例
## 特定の画像に対する予測スコアを比較

[ライブ例: トレーニングの1エポック後と5エポック後の予測を比較 →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#compare-predictions-after-1-vs-5-epochs)
<img src="https://i.imgur.com/NMme6Qj.png" alt="1エポック vs 5エポックのトレーニング"/>
ヒストグラムは、2つのモデル間でクラスごとのスコアを比較しています。各ヒストグラムの上部の緑のバーは、1エポックのみトレーニングされたモデル「CNN-2, 1 epoch」（id 0）を表し、下部の紫のバーは5エポックトレーニングされたモデル「CNN-2, 5 epochs」（id 1）を表します。画像は、モデルが異なる予測をする場合にフィルタされています。例えば、最初の行では、「4」は1エポック後にはすべての可能な数字に対して高スコアを得ていますが、5エポック後には正しいラベルに最高スコアをつけ、他のラベルには非常に低いスコアをつけています。

## 時間の経過での主要なエラーに注目
[ライブ例 →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#top-errors-over-time)

全テストデータに対して誤った予測 (「guess」 != 「truth」でフィルタ) を確認します。トレーニング1エポック後には229の誤った予測がありますが、5エポック後には98のみです。
<img src="https://i.imgur.com/7g8nodn.png" alt="並んでいる、1エポック vs 5エポックのトレーニング"/>

## モデルの性能を比較してパターンを見つける

[ライブ例で詳細を見る →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#false-positives-grouped-by-guess)

正解をフィルタリングした後、予測ごとにグループ化して、誤分類された画像の例と真のラベルの基礎となる分布を確認します。2倍のレイヤーサイズと学習率を持つモデルのバリアントが左に、ベースラインが右にあります。ベースラインは各予測クラスごとに少し多くのミスをします。
<img src="https://i.imgur.com/i5PP9AE.png" alt="ベースラインと2倍のバリアントのグループ化されたエラー"/>

# サインアップまたはログイン

[サインアップまたはログイン](https://wandb.ai/login)して、ブラウザで自分のExperimentsを見たり操作したりしましょう。

この例では、Google Colabを便利なホスティング環境として使用していますが、自分のトレーニングスクリプトをどこからでも実行し、W&Bの実験管理ツールを使ってメトリクスを可視化できます。


```python
!pip install wandb -qqq
```

アカウントにログインします


```python

import wandb
wandb.login()

WANDB_PROJECT = "mnist-viz"
```

# 0. セットアップ

依存関係をインストールし、MNISTをダウンロードし、PyTorchを使用してトレーニングとテストデータセットを作成します。


```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T 
import torch.nn.functional as F


device = "cuda:0" if torch.cuda.is_available() else "cpu"

# トレーニングとテストのデータローダーを作成
def get_dataloader(is_train, batch_size, slice=5):
    "トレーニングデータローダーを取得"
    ds = torchvision.datasets.MNIST(root=".", train=is_train, transform=T.ToTensor(), download=True)
    loader = torch.utils.data.DataLoader(dataset=ds, 
                                         batch_size=batch_size, 
                                         shuffle=True if is_train else False, 
                                         pin_memory=True, num_workers=2)
    return loader
```

# 1. モデルとトレーニングスケジュールを定義する

* 各エポックがトレーニングステップと検証（テスト）ステップから成るエポック数を設定します。オプションで、各テストステップでログするデータ量を設定できます。デモを簡略化するために、ここではバッチ数と可視化する各バッチの画像数を少なく設定しています。
* 単純な畳み込みニューラルネット（[pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial)のコードに従う）を定義します。
* PyTorchを使用してトレーニングセットとテストセットをロードします


```python
# 実行するエポック数
# 各エポックにはトレーニングステップとテストステップが含まれるため、
# これによりログされるテスト予測のテーブル数が設定されます
EPOCHS = 1

# 各テストステップでテストデータからログするバッチ数
# （デモを簡略化するためにデフォルトで低く設定されています）
NUM_BATCHES_TO_LOG = 10 #79

# 各テストバッチでログする画像の数
# （デモを簡略化するためにデフォルトで低く設定されています）
NUM_IMAGES_PER_BATCH = 32 #128

# トレーニング設定とハイパーパラメーター
NUM_CLASSES = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
L1_SIZE = 32
L2_SIZE = 64
# これを変更すると隣接するレイヤーの形状を変更する必要がある場合があります
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
        # 与えられたレイヤーの形状を見るためにコメントを解除します:
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

# 2. トレーニングを実行し、テスト予測をログする

各エポックごとに、トレーニングステップとテストステップを実行します。各テストステップでは、wandb.Table()を作成してテスト予測を保存します。これらはブラウザで視覚化、動的クエリ、並べて比較できます。


```python
# ✨ W&B: このモデルのトレーニングを追跡する新しいrunを初期化
wandb.init(project="table-quickstart")

# ✨ W&B: 設定を使用してハイパーパラメーターをログ
cfg = wandb.config
cfg.update({"epochs" : EPOCHS, "batch_size": BATCH_SIZE, "lr" : LEARNING_RATE,
            "l1_size" : L1_SIZE, "l2_size": L2_SIZE,
            "conv_kernel" : CONV_KERNEL_SIZE,
            "img_count" : min(10000, NUM_IMAGES_PER_BATCH*NUM_BATCHES_TO_LOG)})

# モデル、損失関数、オプティマイザーを定義
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
  # 画像の順序に基づくIDを追加
  _id = 0
  for i, l, p, s in zip(log_images, log_labels, log_preds, log_scores):
    # 必要な情報をデータテーブルに追加:
    # id, 画像のピクセル, モデルの予測, 正解ラベル, すべてのクラスのスコア
    img_id = str(_id) + "_" + str(log_counter)
    test_table.add_data(img_id, wandb.Image(i), p, l, *s)
    _id += 1
    if _id == NUM_IMAGES_PER_BATCH:
      break

# モデルをトレーニング
total_step = len(train_loader)
for epoch in range(EPOCHS):
    # トレーニングステップ
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        # backwardと最適化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
  
        # ✨ W&B: トレーニングステップで損失をログし、UIでリアルタイムに視覚化
        wandb.log({"loss" : loss})
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                .format(epoch+1, EPOCHS, i+1, total_step, loss.item()))
            

    # ✨ W&B: 各テストステップの予測を保存するためのTableを作成
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
        # ✨ W&B: トレーニングエポック全体にわたる精度をログし、UIで視覚化
        wandb.log({"epoch" : epoch, "acc" : acc})
        print('10000のテスト画像に対するモデルのテスト精度: {} %'.format(acc))

    # ✨ W&B: wandbに予測テーブルをログ
    wandb.log({"test_predictions" : test_table})

# ✨ W&B: runを完了としてマーク (マルチセルノートブックに便利)
wandb.finish()
```

# 次は？
次のチュートリアルでは、W&B Sweepsを使用してハイパーパラメーターを最適化する方法を学びます:
## 👉 [ハイパーパラメーターの最適化](sweeps)