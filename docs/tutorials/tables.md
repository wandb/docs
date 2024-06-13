


# 予測の可視化

[**Colabノートブックで試す →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W&B_Tables_Quickstart.ipynb)

このセクションでは、PyTorchを使用してMNISTデータ上でトレーニングする過程でモデルの予測をトラッキングし、可視化し、比較する方法を説明します。

学ぶ内容:
1. モデルトレーニングや評価中に`wandb.Table()`にメトリクス、画像、テキストなどをログ
2. これらのテーブルを表示、ソート、フィルター、グループ、結合、インタラクティブなクエリ、探索
3. 特定の画像やハイパーパラメーター/モデルバージョン、時間ステップにわたる予測や結果の比較

# 例
## 特定の画像の予測スコアを比較

[ライブ例: トレーニング1エポック目と5エポック目の予測比較 →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#compare-predictions-after-1-vs-5-epochs)
<img src="https://i.imgur.com/NMme6Qj.png" alt="1エポック vs 5エポックのトレーニング"/>
ヒストグラムは2つのモデル間のクラス別スコアを比較しています。各ヒストグラムの上の緑色のバーは1エポックだけトレーニングしたモデル "CNN-2, 1エポック" (id 0) を示しています。下の紫色のバーは5エポックトレーニングしたモデル "CNN-2, 5エポック" (id 1) を示しています。画像はモデルが異なる予測をしたケースにフィルターされています。たとえば、最初の行では、1エポック後は「4」がすべての可能な数字に対して高いスコアを取得していますが、5エポック後は正しいラベルに最高のスコアを取得し、他のラベルには非常に低いスコアになります。

## 時間をかけたトップエラーに注目
[ライブ例 →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#top-errors-over-time)

全テストデータに対して間違った予測（"guess" != "truth" の行にフィルター）を確認できます。1エポック目のトレーニング後には229の間違った予測がありますが、5エポック後には98のみになります。
<img src="https://i.imgur.com/7g8nodn.png" alt="並べて比較、1エポック vs 5エポックのトレーニング"/>

## モデルのパフォーマンスを比較してパターンを見つける

[詳しくはライブ例を参照 →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#false-positives-grouped-by-guess)

正しい回答をフィルターアウトし、推測ごとにグループ化して誤分類された画像とその真のラベルの分布を2つのモデルで並べて確認します。左側は層のサイズと学習率が2倍のモデルのバリアント、右側はベースラインです。ベースラインモデルの方が各推測クラスでわずかに多くのミスをしています。
<img src="https://i.imgur.com/i5PP9AE.png" alt="ベースラインとダブルバリアントのエラーをグループ化"/>

# サインアップまたはログイン

[サインアップまたはログイン](https://wandb.ai/login)して、ブラウザで自分のExperimentsを見てインタラクトしましょう。

この例では、便利なホスティング環境としてGoogle Colabを使用していますが、どこからでも自分のトレーニングスクリプトを実行してW&BのExperimentトラッキングツールでメトリクスを可視化できます。

```python
!pip install wandb -qqq
```

アカウントにログイン


```python

import wandb
wandb.login()

WANDB_PROJECT = "mnist-viz"
```

# 0. セットアップ

依存関係をインストールし、MNISTをダウンロードし、PyTorchを使用してトレインデータセットとテストデータセットを作成します。

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T 
import torch.nn.functional as F


device = "cuda:0" if torch.cuda.is_available() else "cpu"

# トレインデータローダーとテストデータローダーを作成
def get_dataloader(is_train, batch_size, slice=5):
    "トレインデータローダーを取得"
    ds = torchvision.datasets.MNIST(root=".", train=is_train, transform=T.ToTensor(), download=True)
    loader = torch.utils.data.DataLoader(dataset=ds, 
                                         batch_size=batch_size, 
                                         shuffle=True if is_train else False, 
                                         pin_memory=True, num_workers=2)
    return loader
```

# 1. モデルとトレーニングスケジュールを定義

* エポック数を設定。各エポックはトレーニングステップとバリデーション（テスト）ステップで構成されます。オプションで、各テストステップごとにログするデータの量を設定できます。このデモを簡略化するために、バッチ数とバッチごとに可視化される画像数を低く設定しています。
* シンプルな畳み込みニューラルネットを定義（[pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial)コードに従う）。
* PyTorchを使用してトレインセットとテストセットをロード

```python
# 実行するエポック数
# 各エポックはトレーニングステップとテストステップで構成されるので、
# これによってログされるテスト予測のテーブル数が設定されます
EPOCHS = 1

# 各テストステップでテストデータからログするバッチ数
# （デモを簡略化するためにデフォルトでは低めに設定）
NUM_BATCHES_TO_LOG = 10 #79

# 各テストバッチでログする画像数
# （デモを簡略化するためにデフォルトでは低めに設定）
NUM_IMAGES_PER_BATCH = 32 #128

# トレーニング設定とハイパーパラメーター
NUM_CLASSES = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
L1_SIZE = 32
L2_SIZE = 64
# これを変更する場合、隣接レイヤーの形状も変更する必要あり
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
        # レイヤーの形状を確認するためのコメント解除:
        #print("x: ", x.size())
        out = self.layer1(x)
        out = self.layer2(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

train_loader = get_dataloader(is_train=True, batch_size=BATCH_SIZE)
test_loader = get_dataloader(is_train=False, batch_size=2*BATCH_SIZE)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

# 2. トレーニングを実行してテスト予測をログ

各エポックごとにトレーニングステップとテストステップを実行します。各テストステップでは、予測結果を保存するためにwandb.Table()を作成します。これらはブラウザで視覚化、動的にクエリ、並べて比較することができます。

```python
# ✨ W&B: このモデルのトレーニングを追跡する新しいrunを初期化
wandb.init(project="table-quickstart")

# ✨ W&B: ハイパーパラメーターをconfigでログ
cfg = wandb.config
cfg.update({"epochs" : EPOCHS, "batch_size": BATCH_SIZE, "lr" : LEARNING_RATE,
            "l1_size" : L1_SIZE, "l2_size": L2_SIZE,
            "conv_kernel" : CONV_KERNEL_SIZE,
            "img_count" : min(10000, NUM_IMAGES_PER_BATCH*NUM_BATCHES_TO_LOG)})

# モデル、損失関数、オプティマイザーを定義
model = ConvNet(NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 予測をログするための便利な関数
def log_test_predictions(images, labels, outputs, predicted, test_table, log_counter):
  # すべてのクラスに対する信頼度スコアを取得
  scores = F.softmax(outputs.data, dim=1)
  log_scores = scores.cpu().numpy()
  log_images = images.cpu().numpy()
  log_labels = labels.cpu().numpy()
  log_preds = predicted.cpu().numpy()
  # 画像の順序に基づいてIDを追加
  _id = 0
  for i, l, p, s in zip(log_images, log_labels, log_preds, log_scores):
    # データテーブルに必要な情報を追加:
    # id, 画像ピクセル, モデルの推測, 正しいラベル, すべてのクラスのスコア
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
        # forwardパス
        outputs = model(images)
        loss = criterion(outputs, labels)
        # backwardパスと最適化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
  
        # ✨ W&B: トレーニングステップの間の損失をログし、UIでライブで可視化
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
        # ✨ W&B: トレーニングエポック間の精度をログし、UIで可視化
        wandb.log({"epoch" : epoch, "acc" : acc})
        print('モデルの10000テスト画像に対するテスト精度: {} %'.format(acc))

    # ✨ W&B: 予測テーブルをwandbにログ
    wandb.log({"test_predictions" : test_table})

# ✨ W&B: runを完了としてマーク（マルチセルノートブックに便利）
wandb.finish()
```

# 次は？
次のチュートリアルでは、W&B Sweepsを使用してハイパーパラメーターを最適化する方法を学びます:
## 👉 [ハイパーパラメーターの最適化](sweeps)