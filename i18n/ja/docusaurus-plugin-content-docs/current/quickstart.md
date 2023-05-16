---
description: W&B クイックスタート.
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# クイックスタート

W&B をインストールして、すぐに機械学習実験をトラッキングし始めましょう。

## 1. アカウントを作成し、W&Bをインストール
まずはじめに、アカウントを作成して W&B をインストールしてください：

1. [https://wandb.ai/site](https://wandb.ai/site) で無料アカウントに[サインアップ](https://wandb.ai/site)し、wandb アカウントにログインします。  
2. Python 3 の環境があるマシンに[`pip`](https://pypi.org/project/wandb/)を使ってwandbライブラリをインストールします。 
<!-- 3. Login to the wandb library on your machine. You will find your API key here: [https://wandb.ai/authorize](https://wandb.ai/authorize).   -->

以下のコードスニペットは、W&B CLI と Python ライブラリを使って W&B にインストールし、ログインする方法を示しています:

<Tabs
  defaultValue="notebook"
  values={[
    {label: 'ノートブック', value: 'notebook'},
    {label: 'コマンドライン', value: 'cli'},
  ]}>
  <TabItem value="cli">

Weights and Biases API とやり取りするための CLI と Python ライブラリをインストールします：

```
pip install wandb
```
</TabItem>
  <TabItem value="notebook">

Weights and Biases APIとやり取りするためのCLIとPythonライブラリをインストールしてください：

```python
!pip install wandb
```


  </TabItem>
</Tabs>

## 2. W&Bにログインする


<Tabs
  defaultValue="notebook"
  values={[
    {label: 'ノートブック', value: 'notebook'},
    {label: 'コマンドライン', value: 'cli'},
  ]}>
  <TabItem value="cli">

次に、W&Bにログインします：

```
wandb login
```
また、[W&Bサーバ:](./guides/hosting) を利用している場合は、

```
wandb login --host=http://wandb.your-shared-local-host.com
```

プロンプトに表示されたときに[あなたのAPIキー](https://wandb.ai/authorize)を提供してください。

  </TabItem>
  <TabItem value="notebook">

次に、W&B Python SDKをインポートし、ログインしてください:

```python
wandb.login()
```

プロンプトに表示されたときに[あなたのAPIキー](https://wandb.ai/authorize)を提供してください。
  </TabItem>
</Tabs>


## 3. runを開始し、ハイパーパラメーターをトラッキングする

Pythonスクリプトやノートブックで、W&B Runオブジェクトを[`wandb.init()`](./ref/python/run.md)で初期化し、`config`パラメータにハイパーパラメータ名と値のキー・バリューのペアを持った辞書を渡してください:

```python
run = wandb.init(
    # このrunがログに記録されるプロジェクトを設定
    project="my-awesome-project",
    # ハイパーパラメーターとrunのメタデータをトラッキング
    config={
        "learning_rate": 0.01,
        "epochs": 10,
    })
```


<!-- ```python
run = wandb.init(project="my-awesome-project")
``` -->

[run](./guides/runs)はW&Bの基本的な構成要素です。[メトリクスをトラッキング](./guides/track)したり、[ログを作成](./guides/artifacts)したり、[ジョブを作成](./guides/launch)したり、その他の多くのことを行う際に、頻繁に使用します。

<!-- ## メトリクスをトラックする -->
<!-- ランオブジェクトを初期化する際、`config`パラメータにハイパーパラメータ名と値のキー・バリュー・ペアのディクショナリを渡します:

```python
  # ハイパーパラメータとランのメタデータを追跡
  config={
      "learning_rate": lr,
      "epochs": epochs,
  }
``` -->

<!-- [`wandb.log()`](./ref/python/log.md)を使ってメトリクスをトラックします:

```python
wandb.log({'accuracy': acc, 'loss': loss})
```

`wandb.log`でログを追加すると、最近初期化されたランオブジェクトにデータが保存されます。
## すべてをまとめる

すべてをまとめると、トレーニングスクリプトは次のコード例のようになるかもしれません。ハイライトされたコードは W&B 固有のコードです。
機械学習トレーニングを模倣するコードを追加したことに注意してください。

```python
# train.py
import wandb
import random # デモスクリプト用

# highlight-next-line
wandb.login()

epochs=10
lr=0.01

# highlight-start
run = wandb.init(
    # このrunが記録されるプロジェクトを設定
    project="my-awesome-project",
    # ハイパーパラメーターと run のメタデータをトラッキング
    config={
        "learning_rate": lr,
        "epochs": epochs,
    })
# highlight-end

offset = random.random() / 5
print(f"lr: {lr}")

# トレーニングランのシミュレーション
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset
    print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
    # highlight-next-line
    wandb.log({"accuracy": acc, "loss": loss})

# run.log_code()
```

以上です！W&Bアプリにアクセスして[https://wandb.ai/home](https://wandb.ai/home) にアクセスし、W&Bでログしたメトリクス（精度と損失）がトレーニングの各ステップでどのように向上したかを確認してください。

![上記のスクリプトを実行するたびにトラッキングされた損失と精度を示しています。](/images/quickstart/quickstart_image.png)

上の画像（クリックして拡大）は、上記のスクリプトを実行するたびにトラッキングされた損失と精度を示しています。作成された各run オブジェクトは**Runs**カラムに表示されます。各ランの名前はランダムに生成されます。

## 次に何をすべきか？
<!-- ### アラートを取得する

SlackやメールでW&B Runがクラッシュした場合や、カスタムトリガーで通知を受け取ります。例えば、損失が`NaN`となっている場合や、MLパイプラインのステップが完了した場合に通知を受け取るトリガーを作成できます。

以下の手順に従ってアラートを設定してください:

1. W&B [ユーザー設定](https://wandb.ai/settings)でアラートをオンにします。
2. コードに[`wandb.alert()`](./guides/runs/alert.md)を追加します。

```python
wandb.alert(
    title="Low accuracy", 
    text=f"Accuracy {acc} is below threshold {thresh}"
)
```
アラート条件が満たされた場合に、メールやSlackで通知が行われます。例えば、次の画像はSlackアラートの例です：
![W&BアラートをSlackチャンネルに表示](/images/quickstart/get_alerts.png)

[アラートのドキュメント](./guides/runs/alert.md)でアラートの設定方法について詳しく知ることができます。設定オプションについては、[設定](./guides/app/settings-page/intro.md)ページを参照してください。  -->

W&Bのエコシステムをさらに探索してみましょう。

1. [W&Bインテグレーション](guides/integrations)をチェックして、PyTorchやHugging Face、SageMakerなどのMLフレームワークやライブラリ、サービスとW&Bを統合する方法を学んでください。

2. [W&Bレポート](./guides/reports)を使用して、runの整理、可視化の埋め込み・自動化、発見の説明、コラボレーターとのアップデート共有を行います。

3. [W&Bアーティファクト](./guides/artifacts)を作成して、データセット、モデル、依存関係、結果を機械学習の各ステップでトラッキングします。

4. [W&Bスイープ](./guides/sweeps)でハイパーパラメーター探索を自動化し、可能なモデルのスペースを探索します。

5. [中央ダッシュボード](./guides/data-vis)でデータセットを理解し、モデルの予測を可視化し、洞察を共有します。

![](/images/quickstart/wandb_demo_experiments.gif) 

## よくある質問

**APIキーはどこで見つけることができますか？**

www.wandb.aiにサインインすると、[認証ページ](https://wandb.ai/authorize)にAPIキーがあります。

**W&Bを自動化された環境でどのように使用しますか？**

GoogleのCloudMLのようなシェルコマンドを実行するのが不便な自動化された環境でモデルをトレーニングしている場合は、[環境変数](guides/track/environment-variables)を使った設定に関するガイドを参照してください。

**オンプレミスのインストールは提供していますか？**

はい、独自のマシンやプライベートクラウドで[W&Bをプライベートにホスト](guides/hosting)することができます。試しに[このクイックチュートリアルノートブック](http://wandb.me/intro)を参照してください。注意: wandbのローカルサーバーにログインするには、[ホストフラグを設定](guides/hosting/how-to-guides/basic-setup)して、ローカルインスタンスのアドレスを指定する必要があります。

**wandbのログを一時的にオフにする方法は？**

コードをテストしている際にwandbの同期を無効にしたい場合は、環境変数[`WANDB_MODE=offline`](guides/track/environment-variables)を設定してください。