---
description: W&B Quickstart.
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# クイックスタート

W&Bをインストールして、あなたの機械学習プロジェクトのトラッキングをすぐ開始できます。

## 1. アカウント作成とW&Bのインストール
始める前にまず、アカウントを作成し、W&Bをインストールしてください

1. [ここから](https://wandb.ai/site)フリーアカウントにサインナップし、あなたのwandbアカウントにログインできます。
2. Python 3の環境にwandb ライブラリーをインストールするために[`pip`](https://pypi.org/project/wandb/)を使います。
<!-- 3. Login to the wandb library on your machine. You will find your API key here: [https://wandb.ai/authorize](https://wandb.ai/authorize).   -->

下記のコードを使ってコマンドラインもしくはPythonライブラリーからW&Bをインストールし、ログインすることができます：

<Tabs
  defaultValue="notebook"
  values={[
    {label: 'ノートブック', value: 'notebook'},
    {label: 'コマンドライン', value: 'cli'},
  ]}>
  <TabItem value="cli">

コマンドラインおよびパイソンライブラリーをインストールして、Wights and Biases APIを使う：

```
pip install wandb
```

  </TabItem>
  <TabItem value="notebook">

コマンドラインおよびパイソンライブラリーをインストールして、Wights and Biases APIを使う：

```python
!pip install wandb
```


  </TabItem>
</Tabs>

## 2. W&Bにログイン


<Tabs
  defaultValue="notebook"
  values={[
    {label: 'ノートブック', value: 'notebook'},
    {label: 'コマンドライン', value: 'cli'},
  ]}>
  <TabItem value="cli">

W&BのPython SDKをインポートし、ログインします

```
wandb login
```

Or if you're using [W&B Server:](./guides/hosting)

```
wandb login --host=http://wandb.your-shared-local-host.com
```

[ あなたのAPIキー](https://wandb.ai/authorize)をここから入手してください。

  </TabItem>
  <TabItem value="notebook">

W&BのPython SDKをインポートし、ログインします：

```python
wandb.login()
```

[ あなたのAPIキー](https://wandb.ai/authorize)をここから入手してください。
  </TabItem>
</Tabs>


## 3. 学習Runを開始し、ハイパーパラメーターをトラッキング

W&BのRunオブジェクトを初期化します。Pythonスクリプトもしくはノートブックで[`wandb.init()`](./ref/python/run.md) を使い `config` パラメーターに key-value ペアでハイパーパラメーターの名称と値を指定します：

```python
run = wandb.init(
    # Set the project where this run will be logged
    project="my-awesome-project",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": 0.01,
        "epochs": 10,
    })
```


<!-- ```python
run = wandb.init(project="my-awesome-project")
``` -->

[Run](./guides/runs)はW&Bの最も基本的なオブジェクトです。[メトリクスのトラッキング](./guides/track), [ログの生成](./guides/artifacts), [ジョブの生成](./guides/launch),などさまざまな用途で使われます。


<!-- ## Track metrics -->
<!-- Pass a dictionary to the `config` parameter with key-value pairs of hyperparameter name and values when you initialize a run object:

```python
  # Track hyperparameters and run metadata
  config={
      "learning_rate": lr,
      "epochs": epochs,
  }
``` -->


<!-- Use [`wandb.log()`](./ref/python/log.md) to track metrics:

```python
wandb.log({'accuracy': acc, 'loss': loss})
```

Anything you log with `wandb.log` is stored in the run object that was most recently initialized. -->



## 全体をつなげると

ここまでの過程をつなげて、あなたの学習スクリプトがどのようになるのか、下記のコード例で見ていきましょう。ハイライトされている部分がW&Bに特化した部分です。下記の例では機械学習のコードになぞらえた例を使っています

```python
# train.py
import wandb
import random # for demo script

# highlight-next-line
wandb.login()

epochs=10
lr=0.01

# highlight-start
run = wandb.init(
    # Set the project where this run will be logged
    project="my-awesome-project",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": lr,
        "epochs": epochs,
    })
# highlight-end    

offset = random.random() / 5
print(f"lr: {lr}")

# simulating a training run
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset
    print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
    # highlight-next-line
    wandb.log({"accuracy": acc, "loss": loss})

# run.log_code()
```

以上です。W&Bウェブアプリ[https://wandb.ai/home](https://wandb.ai/home) に行って、記録されたメトリック（AccuracyとLoss）をリアルタイムで見てみましょう。 

![Shows the loss and accuracy that was tracked from each time we ran the script above. ](/images/quickstart/quickstart_image.png)

上記のスクリーンショット（クリックして拡大できます）は上記スクリプトを複数回走らせた結果を示しています。毎回異なるRunオブジェクトが作られ、Runsという左側のカラムに並んでいます。Runの名前は指定しなければランダムに生成されます。


## ネクストステップ
Explore the rest of the W&B ecosystem.

1. Check out [W&B Integrations](guides/integrations) to learn how to integrate W&B with your ML framework such as PyTorch, ML library such as Hugging Face, or ML service such as SageMaker. 
2. Organize runs, embed and automate visualizations, describe your findings, and share updates with collaborators with [W&B Reports](./guides/reports).
2. Create [W&B Artifacts](./guides/artifacts) to track datasets, models, dependencies, and results through each step of your machine learning pipeline.
3. Automate hyperparameter search and explore the space of possible models with [W&B Sweeps](./guides/sweeps).
4. Understand your datasets, visualize model predictions, and share insights in a [central dashboard](./guides/data-vis).


![](/images/quickstart/wandb_demo_experiments.gif) 


## よくある質問

**APIキーはどこにありますか?**
W&Bにログイン（www.wandb.ai)し、[Authorize page](https://wandb.ai/authorize) に行くことで入手できます。

**自動化された環境でW&Bを使うことができますか？**
もし自動化された環境でモデルの学習をしていて、コマンドラインを叩くことができない場合（例えば Google's CloudML）[環境変数](guides/track/environment-variables)のページにある設定ガイドを参照してください。

**ローカルないしオンプレ環境での実行に対応していますか?**
はい、[プライベート環境でのホスティング](guides/hosting)でご自身のマシンやプライベートクラウドなどでW&Bを実行することができます。こちらの[チュートリアルノートブック](http://wandb.me/intro)をお試しください。　

wandbのローカルサーバーにログインするには、[ホストフラグ](guides/hosting/basic-setup#login)を、ローカルサーバーのアドレスに設定してください。 

**W&Bのログトラッキングを一時的に切ることはできますか?**
もしコードのテスト中など、トラッキングを行いたくない場合には、環境変数 [WANDB_MODE=offline](guides/track/environment-variables)を設定してください。

