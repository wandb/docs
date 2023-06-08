---
description: W&B クイックスタート.
displayed_sidebar: ja
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
また、[W&Bサーバ:](./guides/hosting/intro.md) を利用している場合は、

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




W&Bのエコシステムをさらに探索してみましょう。

1. [W&Bインテグレーション](guides/integrations/intro.md)をチェックして、PyTorchやHugging Face、SageMakerなどのMLフレームワークやライブラリ、サービスとW&Bを統合する方法を学んでください。

2. [W&Bレポート](./guides/reports/intro.md)を使用して、runの整理、可視化の埋め込み・自動化、発見の説明、コラボレーターとのアップデート共有を行います。

3. [W&Bアーティファクト](./guides/artifacts/intro.md)を作成して、データセット、モデル、依存関係、結果を機械学習の各ステップでトラッキングします。

4. [W&Bスイープ](./guides/sweeps/intro.md)でハイパーパラメーター探索を自動化し、可能なモデルのスペースを探索します。

5. [中央ダッシュボード](./guides/data-vis/intro.md)でデータセットを理解し、モデルの予測を可視化し、洞察を共有します。

![](/images/quickstart/wandb_demo_experiments.gif) 

## よくある質問

**APIキーはどこで見つけることができますか？**

www.wandb.aiにサインインすると、[認証ページ](https://wandb.ai/authorize)にAPIキーがあります。

**W&Bを自動化された環境でどのように使用しますか？**

GoogleのCloudMLのようなシェルコマンドを実行するのが不便な自動化された環境でモデルをトレーニングしている場合は、[環境変数](guides/track/environment-variables.md)を使った設定に関するガイドを参照してください。

**オンプレミスのインストールは提供していますか？**

はい、独自のマシンやプライベートクラウドで[W&Bをプライベートにホスト](guides/hosting/intro.md)することができます。試しに[このクイックチュートリアルノートブック](http://wandb.me/intro)を参照してください。注意: wandbのローカルサーバーにログインするには、[ホストフラグを設定](guides/hosting/how-to-guides/basic-setup.md)して、ローカルインスタンスのアドレスを指定する必要があります。

**wandbのログを一時的にオフにする方法は？**

コードをテストしている際にwandbの同期を無効にしたい場合は、環境変数[`WANDB_MODE=offline`](guides/track/environment-variables)を設定してください。