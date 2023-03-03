---
description: W&B Quickstart.
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Quickstart
このページでは、あなたのコードにW&Bを組み込むために最もよく使われる手順を紹介します。主なステップは下記の通りです。
1. [Set up W&B](#set-up-wb)
2. [Track metrics](#track-metrics)
3. [Track hyperparameters](#track-hyperparameters)
4. [Get an alert](#get-alerts)

## W&Bのセットアップ
まず、W&Bを使い始めるには、下記の要求を満たす必要があります：

1. フリーアカウントに[サインアップ](https://wandb.ai/site)し、あなたのwandbアカウントにログイン。
2. `pip`コマンドで、Python 3向けのwandbライブラリーをインストール。  
3. あなたのマシンからwandbライブラリーを使ってログイン。APIキーはここで取得できます： [https://wandb.ai/authorize](https://wandb.ai/authorize).  

W&Bをインストールしてログインするためには、下記のコードを参考してください。ここでは二つの方法を見ていきます：

<Tabs
  defaultValue="cli"
  values={[
    {label: 'コマンドライン', value: 'cli'},
    {label: 'ノートブック', value: 'notebook'},
  ]}>
  <TabItem value="cli">

Weights & BiasesのAPIを使うために、ライブラリをインストールします

```
pip install wandb
```

次に、W&Bにログインします:

```
wandb login
```

もしくは[W&B サーバー](./guides/hosting/intro.md) を使う場合には：

```
wandb login --host=http://wandb.your-shared-local-host.com
```

  </TabItem>
  <TabItem value="notebook">

Install the CLI and Python library for interacting with the Weights and Biases API:

```python
!pip install wandb
```

Next, import the W&B Python SDK and log in:

```python
import wandb
wandb.login()
```

  </TabItem>
</Tabs>


## 新しいRunの開始

新しいRunを初期化するには、Pythonスクリプトないし、ノートブックで、[`wandb.init()`](./ref/python/run.md)を呼び出します。学習用スクリプトの初めの方で、下記のコードを使ってください：

```python
import wandb
wandb.init(project="my-awesome-project")
```

`wandb.init()` APIを呼び出すと、W&Bはシステム関連メトリクスとコンソールのログを自動的にトラッキング開始します。

コードを走らせて、求められたら[APIキー](https://wandb.ai/authorize)を入力してください。次のステップで、他のメトリクスの捕捉方法を説明します。


## メトリクスのトラッキング

Use [`wandb.log()`](./ref/python/log.md) to track metrics or a framework [integration](guides/integrations/intro.md) for easy instrumentation.

[`wandb.log()`](./ref/python/log.md) を使うことで、任意のメトリクスのトラッキングを開始します。各種開発フレームワークに既に[インテグレーション](guides/integrations/intro.md)が用意されていれば、そちらを使う方が簡単でしょう。

```python
wandb.log({'accuracy': train_acc, 'loss': train_loss})
```

W&Bは先ほど初期化したRunオブジェクトに`wandb.log`で指定されたメトリクスを保存します。つまり精度や損失関数などがW&BのRunに紐づく形になります。

![](/images/quickstart/wandb_demo_logging_metrics.png)


## ハイパーパラメーターのトラッキング

[`wandb.config`](./guides/track/config.md)を使えば、ハイパーパラメーターを保存することができます。ハイパーパラメーターをトラッキングすることで、複数の実験をW&B Webアプリの中から簡単に比較できるようになります。

```python
wandb.config.dropout = 0.2
```

`wandb.config`オブジェクト内の悪トリビュートは、最も最新のRunオブジェクトに紐づきます。

![](/images/quickstart/wandb_demo_experiments.gif)

## アラートの生成

もしもRunがクラッシュしたり、事前に設定したトリガー条件が満たされた時に、Slackかemailでアラートを受け取ることができます。

そのようなアラートを作るには、下記の手順に従ってください：

1. W&Bのユーザー設定の中で、アラートをオンにする
2. [`wandb.alert()`](./guides/runs/alert.md) をあなたのコードに付け加える。

```python
wandb.alert(
    title="Low accuracy", 
    text=f"Accuracy {acc} is below threshold {thresh}"
)
```
アラート条件が満たされた際に、Slackもしくはメールで通知を受け取れます。例えば、下記のスクリーンショットはSlackでのアラートの例です：

![W&B Alerts in a Slack channel](/images/quickstart/get_alerts.png)

See the [Alerts docs](./guides/runs/alert.md) for more information on how to set up an alert. For more information about setting options, see the [Settings](./guides/app/settings-page/intro.md) page. 

[アラートドキュメンテーション](./guides/runs/alert.mdにはより詳細な通知設定の方法が記載されています。また、ユーザー設定に関する詳細は[設定ページ](./guides/app/settings-page/intro.md)を参照してください。


## ネクストステップ

1. [**Collaborative Reports**](./guides/reports/intro.md): Snapshot results, take notes, and share findings
2. [**Data + Model Versioning**](./guides/data-and-model-versioning/intro.md): Track dependencies and results in your ML pipeline
3. [**Data Visualization**](guides/data-vis/intro.md): Visualize and query datasets and model evaluations
4. [**Hyperparameter Tuning**](guides/sweeps/intro.md): Quickly automate optimizing hyperparameters
5. [**Private-Hosting**](guides/hosting/intro.md): The enterprise solution for private cloud or on-prem hosting of W&B

## よくある質問

**Where do I find my API key?**
APIキーはどこにありますか？ W&Bにログイン（www.wandb.ai）し、Authorize pageに行くことで入手できます。

**自動化された環境でW&Bを使うことができますか？**
If you are training models in an automated environment where it's inconvenient to run shell commands, such as Google's CloudML, you should look at our guide to configuration with [Environment Variables](guides/track/environment-variables.md).

**ローカルないしオンプレ環境での実行に対応していますか？**
Yes, you can [privately host W&B](guides/hosting/intro.md) locally on your own machines or in a private cloud, try [this quick tutorial notebook](http://wandb.me/intro) to see how. Note, to login to wandb local server you can [set the host flag](./guides/hosting/basic-setup.md#login) to the address of the local instance.  **** 

**wandbのローカルサーバーにログインするには、ホストフラグを、ローカルサーバーのアドレスに設定してください。**
If you're testing code and want to disable wandb syncing, set the environment variable [`WANDB_MODE=offline`](guides/track/environment-variables.md).
