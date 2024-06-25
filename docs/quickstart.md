---
description: W&B クイックスタート.
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# クイックスタート

W&Bをインストールして、数分で機械学習実験をトラッキングしましょう。

## 1. アカウントを作成してW&Bをインストールする
始める前に、アカウントを作成してW&Bをインストールしてください:

1. [サインアップ](https://wandb.ai/site)して、[https://wandb.ai/site](https://wandb.ai/site) で無料アカウントを作成し、wandbアカウントにログインしてください。
2. Python 3環境で[`pip`](https://pypi.org/project/wandb/)を使ってwandbライブラリをマシンにインストールします。

以下のコードスニペットは、W&B CLIおよびPythonライブラリを使ってW&Bにインストールしてログインする方法を示しています。

<Tabs
  defaultValue="notebook"
  values={[
    {label: 'Notebook', value: 'notebook'},
    {label: 'Command Line', value: 'cli'},
  ]}>
  <TabItem value="cli">

Weights & Biases APIと対話するためにCLIとPythonライブラリをインストールします。

```bash
pip install wandb
```

  </TabItem>
  <TabItem value="notebook">

Weights & Biases APIと対話するためにCLIとPythonライブラリをインストールします。

```notebook
!pip install wandb
```

  </TabItem>
</Tabs>

## 2. W&Bにログインする


<Tabs
  defaultValue="notebook"
  values={[
    {label: 'Notebook', value: 'notebook'},
    {label: 'Command Line', value: 'cli'},
  ]}>
  <TabItem value="cli">

次に、W&Bにログインします。

```bash
wandb login
```

または、[W&B Server](./guides/hosting)（**Dedicated Cloud** または **Self-managed** を含む）を使用している場合：

```bash
wandb login --relogin --host=http://your-shared-local-host.com
```

必要に応じて、デプロイメント管理者にホスト名を問い合わせてください。

[あなたのAPIキー](https://wandb.ai/authorize) を求められたら入力してください。

  </TabItem>
  <TabItem value="notebook">

次に、W&B Python SDKをインポートしてログインします。

```python
wandb.login()
```

[あなたのAPIキー](https://wandb.ai/authorize) を求められたら入力してください。
  </TabItem>
</Tabs>

## 3. Runを開始してハイパーパラメータをトラックする

Pythonスクリプトまたはノートブックで[`wandb.init()`](./ref/python/run.md)を使ってW&B Runオブジェクトを初期化し、ハイパーパラメータ名と値のペアを持つ辞書を`config`パラメータに渡します：

```python
run = wandb.init(
    # このrunがログされるプロジェクトを設定
    project="my-awesome-project",
    # ハイパーパラメータとrunメタデータをトラック
    config={
        "learning_rate": 0.01,
        "epochs": 10,
    },
)
```

[run](./guides/runs) はW&Bの基本的な構成要素です。メトリクスを[トラック](./guides/track)したり、[ログ](./guides/artifacts)を作成したり、[ジョブを作成](./guides/launch)したりする際に頻繁に使用します。

## すべてをまとめる

すべてをまとめると、トレーニングスクリプトは以下のコード例のようになります。強調したコードはW&B特有のコードです。機械学習のトレーニングを模擬するコードを追加しています。

```python
# train.py
import wandb
import random  # デモスクリプト用

# 次の行を強調
wandb.login()

epochs = 10
lr = 0.01

# ここから強調
run = wandb.init(
    # このrunがログされるプロジェクトを設定
    project="my-awesome-project",
    # ハイパーパラメータとrunメタデータをトラック
    config={
        "learning_rate": lr,
        "epochs": epochs,
    },
)
# ここまで強調

offset = random.random() / 5
print(f"lr: {lr}")

# トレーニングrunをシミュレーション
for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset
    print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
    # 次の行を強調
    wandb.log({"accuracy": acc, "loss": loss})

# run.log_code()
```

以上です！W&Bアプリの[https://wandb.ai/home](https://wandb.ai/home)に移動して、W&Bでログしたメトリクス（精度と損失）が各トレーニングステップでどのように改善されたかを確認してください。

![各スクリプト実行時にトラッキングされた損失と精度を示す。](/images/quickstart/quickstart_image.png)

上の画像（クリックで拡大）は、上記のスクリプトを実行した際にトラッキングされた損失と精度を示しています。作成された各runオブジェクトは**Runs**列内に表示されます。各run名はランダムに生成されます。

## 次に何をする？

W&Bのエコシステムを探索しましょう。

1. [W&B Integrations](guides/integrations)をチェックして、PyTorchのようなMLフレームワーク、Hugging FaceのようなMLライブラリ、SageMakerのようなMLサービスとW&Bを統合する方法を学びましょう。
2. runを整理し、可視化を埋め込み、自動化し、学びを記述し、コラボレーターとアップデートを共有しましょう。[W&B Reports](./guides/reports)を使ってください。
2. [W&B Artifacts](./guides/artifacts)を作成して、データセット、モデル、依存関係、結果を機械学習パイプラインの各ステップでトラッキングしましょう。
3. [W&B Sweeps](./guides/sweeps)を使ってハイパーパラメータ検索を自動化し、可能なモデルの空間を探索しましょう。
4. データセットを理解し、モデルの予測を可視化し、インサイトを[中央ダッシュボード](./guides/tables)で共有しましょう。

![](/images/quickstart/wandb_demo_experiments.gif)

## よくある質問

**APIキーはどこで見つけられますか？**
www.wandb.ai にサインインすると、[Authorizeページ](https://wandb.ai/authorize)にAPIキーがあります。

**自動化環境でW&Bを使用するにはどうすればよいですか？**
GoogleのCloudMLのように、シェルコマンドを実行するのが不便な自動化環境でモデルをトレーニングする場合、[環境変数](guides/track/environment-variables)を使った設定ガイドをご覧ください。

**ローカルのオンプレミスインストールは提供していますか？**
はい、W&Bを[ローカルホスト](guides/hosting/)したり、プライベートクラウドでプライベートにホストすることができます。[この簡単なチュートリアルノートブック](http://wandb.me/intro)をご覧ください。

**一時的にwandb loggingをオフにするにはどうすればよいですか？**
コードをテストしていて一時的にwandbの同期を無効にしたい場合、環境変数[`WANDB_MODE=offline`](./guides/track/environment-variables)を設定します。