---
description: PythonコードからのアラートをSlackまたはメールで受け取る
---

# wandb.alertを使ってアラートを送信する

<head>
  <title>Pythonコードからアラートを送信する</title>
</head>


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://wandb.me/alerts-colab)

W&B Alertsを使用すると、W&B Runがクラッシュした場合や、損失がNaNになったり、MLパイプラインのステップが完了したなどのカスタムトリガーが達成された場合に、Slackまたはメールで通知を受け取ることができます。W&B Alertsは、個人プロジェクトとチームプロジェクトの両方を含む、runを起動するすべてのプロジェクトに適用されます。

以下のようにアラートを設定することができます:

```python
text = f"精度 {acc} は許容範囲 {thresh} を下回っています"

wandb.alert(
    title="低い精度", 
    text=text
)
```

そして、Slack（またはメール）でW&B Alertsメッセージを確認できます:

![](/images/track/send_alerts_slack.png)
## はじめに

:::info
この手順は、パブリッククラウドでのみアラートを有効にする方法です。

プライベートクラウドやW&B専用クラウドで[W&Bサーバー](../hosting/intro.md)を使用している場合は、[このドキュメント](../hosting/slack-alerts.md)を参照してSlackアラートのセットアップを行ってください。
:::

コードからSlackやメールでアラートを送信するためには、最初に以下の2つの手順を行ってください。

1. W&B [ユーザー設定](https://wandb.ai/settings)でアラートを有効にする
2. `wandb.alert()`をコードに追加する

### 1. W&Bユーザー設定でアラートを有効にする

[ユーザー設定](https://wandb.ai/settings)で:

* **アラート**セクションまでスクロールする
* \`wandb.alert()\`からアラートを受け取るために、**スクリプトで実行可能なアラート** を有効にする
* **Slackに接続**を使用して、アラートを投稿するSlackチャンネルを選択します。アラートをプライベートに保つために、**Slackbot**チャンネルをお勧めします。
* **メール**は、W&Bにサインアップした際に使用したメールアドレスに送信されます。すべてのこれらのアラートがフォルダに入り、受信トレイがいっぱいにならないように、メールでフィルタを設定することをお勧めします。

W&Bアラートの設定を初めて行う場合や、アラートの受信方法を変更したい場合に、この作業を行う必要があります。

![W&Bユーザー設定のアラート設定](/images/track/demo_connect_slack.png)

### 2. \`wandb.alert()\`をコードに追加する

コード（ノートブックまたはPythonスクリプト）に`wandb.alert()`を追加し、アラートがトリガーされる場所を選択します。
次のMarkdownテキストを日本語に翻訳してください。翻訳したテキストのみを返してください。テキスト：

```python
wandb.alert(
    title="High Loss", 
    text="Loss is increasing rapidly"
)
```

#### Slackまたはメールを確認してください

アラートメッセージがSlackまたはメールに届いているか確認してください。もし届いていない場合は、[ユーザー設定](https://wandb.ai/settings)で**スクリプト可能なアラート**に対してメールまたはSlackが有効になっているか確認してください。

## `wandb.alert()`の使用法

| 引数                         | 説明                                                                                                                                             |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `title` (string)          | アラートの短い説明。例：「精度が低い」                                                                                                           |
| `text` (string)            | アラートのトリガーとなった事象についてのより詳細な説明                                                                                              |
| `level` (任意)             | アラートの重要度。`AlertLevel.INFO`、`AlertLevel.WARN`、または`AlertLevel.ERROR`のいずれかである必要があります。`wandb`から`AlertLevel.xxx`をインポートできます。 |
|                            |                                                                                                                                                   |
| `wait_duration` (任意)     | 同じ**タイトル**で別のアラートを送信するまでの秒数。これにより、アラートのスパムが減少します。                                                       |

### 例

このシンプルなアラートは、精度が閾値を下回ったときに警告を送信します。この例では、少なくとも5分間隔でアラートを送信します。

[コードを実行 →](http://wandb.me/alerts)

```python
import wandb
from wandb import AlertLevel

if acc < threshold:
    wandb.alert(
        title="低い精度", 
        text=f"精度 {acc} は許容範囲 {threshold} を下回っています",
        level=AlertLevel.WARN,
        wait_duration=300
    )
```

## 詳細情報

### タグ付け/ユーザーのメンション

Slackでアラートを送る際、自分自身や同僚にメンションを送ることができます。そのためには、アラートのタイトルやテキスト内にSlackユーザーIDを `<@USER_ID>` として追加してください。SlackプロフィールページからSlackユーザーIDを見つけることができます。

```python
wandb.alert(
    title="Loss is NaN", 
    text=f"Hey <@U1234ABCD> lossがNaNになってしまいました"
)
```

### W&Bチームアラート

チームの管理者は、チーム設定ページでチームに対するアラートを設定できます: wandb.ai/teams/ `your-team`。これらのアラートは、チーム内の全員に適用されます。アラートをプライベートに保つために、**Slackbot**チャンネルの使用をお勧めします。

### Slackチャンネルの変更

投稿先のチャンネルを変更するには、**Slackの接続解除**をクリックしてから、再接続して別の宛先チャンネルを選択してください。
## FAQ(よくある質問)



#### Jupyterノートブックでは「Run Finished」アラートは機能しますか？



ユーザー設定で「Run Finished」設定をオンにすることで有効になる **「Run Finished」** アラートは、Pythonスクリプトでのみ動作し、Jupyter Notebooks環境では各セルの実行ごとにアラート通知が発生しないように無効にされています。代わりに、Jupyter Notebook環境で `wandb.alert()` を使用してください。



#### [W&Bサーバー](../hosting/intro.md)でアラートを有効にする方法は？



<!-- 自分でW&Bサーバーをホストしている場合は、Slackアラートを有効にする前に、[この手順](../../hosting/setup/configuration#slack)に従って設定する必要があります。 -->