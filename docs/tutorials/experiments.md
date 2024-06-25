
# 実験をトラッキングする

[**Colab ノートブックで試す →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_&_Biases.ipynb)

急速な実験は機械学習にとって基本的です。このチュートリアルでは、W&B を使用して実験をトラッキングおよび可視化し、結果を迅速に反復し理解します。

## 🤩 実験のための共有ダッシュボード

数行のコードで、
[あなた自身でここで見られる](https://wandb.ai/wandb/wandb_example)豊富でインタラクティブな共有可能なダッシュボードを手に入れることができます。
![](https://i.imgur.com/Pell4Oo.png)

## 🔒 データとプライバシー

私たちはセキュリティを非常に重視しており、クラウドホストのダッシュボードは暗号化の業界標準のベストプラクティスを使用しています。エンタープライズクラスターを外部に出られないデータセットを扱っている場合は、[オンプレミス](https://docs.wandb.com/self-hosted)インストールも利用可能です。

また、すべてのデータを簡単にダウンロードして他のツールにエクスポートすることもできます — Jupyterノートブックでのカスタム分析のように。こちらに[APIについての詳細](https://docs.wandb.com/library/api)があります。

---

## 🪄 `wandb` ライブラリをインストールしてログインする

まず、ライブラリをインストールして無料アカウントにログインします。

```python
!pip install wandb -qU
```

```python
# W&B アカウントにログイン
import wandb
wandb.login()
```

## 👟 実験を実行する
1️⃣. **新しい run を開始**し、トラッキングするハイパーパラメーターを渡す

2️⃣. **トレーニングまたは評価のメトリクスをログ**する

3️⃣. **結果をダッシュボードで可視化**する

```python
import random

# 5つのシミュレーション実験を開始
total_runs = 5
for run in range(total_runs):
  # 🐝 1️⃣ このスクリプトをトラッキングするために新しい run を開始
  wandb.init(
      # この run をログするプロジェクトを設定
      project="basic-intro", 
      # run の名前を設定（さもなければランダムに割り当てられます。例: sunshine-lollypop-10）
      name=f"experiment_{run}", 
      # ハイパーパラメーターと run のメタデータをトラッキング
      config={
      "learning_rate": 0.02,
      "architecture": "CNN",
      "dataset": "CIFAR-100",
      "epochs": 10,
      })

  # このシンプルなブロックはメトリクスをログするトレーニングループをシミュレートします
  epochs = 10
  offset = random.random() / 5
  for epoch in range(2, epochs):
      acc = 1 - 2 ** -epoch - random.random() / epoch - offset
      loss = 2 ** -epoch + random.random() / epoch + offset
      
      # 🐝 2️⃣ スクリプトから W&B にメトリクスをログ
      wandb.log({"acc": acc, "loss": loss})
      
  # run を終了としてマーク
  wandb.finish()
```

3️⃣ このコードを実行すると、上記のいずれかの wandb リンクをクリックしてインタラクティブなダッシュボードを見つけることができます。

# 🔥 シンプルな Pytorch ニューラルネットワーク

💪 このモデルを走らせてシンプルな MNIST クラス分類器を訓練し、結果が W&B プロジェクトにリアルタイムにストリーミングされるのをプロジェクトページリンクで確認してください。

`wandb` のいかなる run においても自動的に[メトリクス](https://docs.wandb.ai/ref/app/pages/run-page#charts-tab)、[システム情報](https://docs.wandb.ai/ref/app/pages/run-page#system-tab)、[ハイパーパラメーター](https://docs.wandb.ai/ref/app/pages/run-page#overview-tab)、[ターミナル出力](https://docs.wandb.ai/ref/app/pages/run-page#logs-tab) がログされ、モデル入力と出力が含まれた[インタラクティブな表](https://docs.wandb.ai/guides/tables)を見ることができます。

## Dataloader のセットアップ

この例を実行するには、PyTorch をインストールする必要があります。Google Colab を使用している場合は、すでにプリインストールされています。

```python
!pip install torch torchvision
```

```python
import wandb
import math
import random
import torch, torchvision
import torch.nn as nn
import torchvision.transforms as T

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_dataloader(is_train, batch_size, slice=5):
    "トレーニングデータローダーを取得"
    full_dataset = torchvision.datasets.MNIST(root=".", train=is_train, transform=T.ToTensor(), download=True)
    sub_dataset = torch.utils.data.Subset(full_dataset, indices=range(0, len(full_dataset), slice))
    loader = torch.utils.data.DataLoader(dataset=sub_dataset, 
                                         batch_size=batch_size, 
                                         shuffle=True if is_train else False, 
                                         pin_memory=True, num_workers=2)
    return loader

def get_model(dropout):
    "シンプルなモデル"
    model = nn.Sequential(nn.Flatten(),
                         nn.Linear(28*28, 256),
                         nn.BatchNorm1d(256),
                         nn.ReLU(),
                         nn.Dropout(dropout),
                         nn.Linear(256,10)).to(device)
    return model

def validate_model(model, valid_dl, loss_func, log_images=False, batch_idx=0):
    "モデルの性能を検証用データセットで評価し、wandb.Table にログ"
    model.eval()
    val_loss = 0.
    with torch.inference_mode():
        correct = 0
        for i, (images, labels) in enumerate(valid_dl):
            images, labels = images.to(device), labels.to(device)

            # Forward pass ➡
            outputs = model(images)
            val_loss += loss_func(outputs, labels)*labels.size(0)

            # 精度を計算して蓄積
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            # ダッシュボードに1バッチの画像をログ、常に同じ batch_idx
            if i==batch_idx and log_images:
                log_image_table(images, predicted, labels, outputs.softmax(dim=1))
    return val_loss / len(valid_dl.dataset), correct / len(valid_dl.dataset)

def log_image_table(images, predicted, labels, probs):
    "（画像、予測、ターゲット、スコア）を含む wandb.Table をログ"
    # 🐝 画像、ラベル、予測をログするための wandb Table を作成
    table = wandb.Table(columns=["image", "pred", "target"]+[f"score_{i}" for i in range(10)])
    for img, pred, targ, prob in zip(images.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")):
        table.add_data(wandb.Image(img[0].numpy()*255), pred, targ, *prob.numpy())
    wandb.log({"predictions_table":table}, commit=False)
```

## モデルのトレーニング

```python
# 5つの実験を開始し、異なるドロップアウト率を試す
for _ in range(5):
    # 🐝 wandb run を初期化
    wandb.init(
        project="pytorch-intro",
        config={
            "epochs": 10,
            "batch_size": 128,
            "lr": 1e-3,
            "dropout": random.uniform(0.01, 0.80),
            })
    
    # コンフィグをコピー
    config = wandb.config

    # データを取得
    train_dl = get_dataloader(is_train=True, batch_size=config.batch_size)
    valid_dl = get_dataloader(is_train=False, batch_size=2*config.batch_size)
    n_steps_per_epoch = math.ceil(len(train_dl.dataset) / config.batch_size)
    
    # シンプルなMLPモデル
    model = get_model(config.dropout)

    # 損失関数とオプティマイザーを作成
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

   # トレーニング
    example_ct = 0
    step_ct = 0
    for epoch in range(config.epochs):
        model.train()
        for step, (images, labels) in enumerate(train_dl):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            train_loss = loss_func(outputs, labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            example_ct += len(images)
            metrics = {"train/train_loss": train_loss, 
                       "train/epoch": (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch, 
                       "train/example_ct": example_ct}
            
            if step + 1 < n_steps_per_epoch:
                # 🐝 トレーニングメトリクスを wandb にログ
                wandb.log(metrics)
                
            step_ct += 1

        val_loss, accuracy = validate_model(model, valid_dl, loss_func, log_images=(epoch==(config.epochs-1)))

        # 🐝 トレーニングと検証メトリクスを wandb にログ
        val_metrics = {"val/val_loss": val_loss, 
                       "val/val_accuracy": accuracy}
        wandb.log({**metrics, **val_metrics})
        
        print(f"Train Loss: {train_loss:.3f}, Valid Loss: {val_loss:3f}, Accuracy: {accuracy:.2f}")

    # テストセットがあれば、これが Summary メトリクスとしてログする方法です
    wandb.summary['test_accuracy'] = 0.8

    # 🐝 wandb run を終了
    wandb.finish()
```

これで wandb を使って最初のモデルを訓練しました！ 👆上記の wandb リンクをクリックしてメトリクスを確認してください。

# 🔔 W&B Alerts を試してみる

**[W&B Alerts](https://docs.wandb.ai/guides/track/alert)**を使用すると、Python コードからトリガーされたアラートを Slack やメールに送信できます。コードからトリガーされた Slack やメールアラートを送信したい場合、初めて行う際には次の 2 ステップを実行します。

1) W&B の [ユーザー設定](https://wandb.ai/settings)でアラートをオンにする

2) コードに `wandb.alert()`を追加する:

```python
wandb.alert(
    title="Low accuracy", 
    text=f"Accuracy is below the acceptable threshold"
)
```

以下の最小例を参照して `wandb.alert` の使い方を確認してください。**[W&B Alerts](https://docs.wandb.ai/guides/track/alert)** の完全なドキュメントもご覧いただけます。

```python
# wandb run を開始
wandb.init(project="pytorch-intro")

# モデルトレーニングループをシミュレート
acc_threshold = 0.3
for training_step in range(1000):

    # 精度のためのランダム数を生成する
    accuracy = round(random.random() + random.random(), 3)
    print(f'Accuracy is: {accuracy}, {acc_threshold}')
    
    # 🐝 精度を wandb にログ
    wandb.log({"Accuracy": accuracy})

    # 🔔 精度が閾値を下回った場合、W&B アラートを発火させて run を停止
    if accuracy <= acc_threshold:
        # 🐝 wandb アラートを送信
        wandb.alert(
            title='Low Accuracy',
            text=f'Accuracy {accuracy} at step {training_step} is below the acceptable theshold, {acc_threshold}',
        )
        print('Alert triggered')
        break

# run を終了としてマーク（Jupyterノートブックでは有用）
wandb.finish()
```

# 次は？
次のチュートリアルでは、W&B Tables を使用してモデルの予測を表示および分析する方法を学びます：
## 👉 [View & Analyze Model Predictions](tables)