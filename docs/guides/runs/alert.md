---
description: PythonコードからトリガーされたアラートをSlackやメールに送信
displayed_sidebar: default
---


# wandb.alert でアラートを送信

<head>
  <title>Pythonコードからアラートを送信</title>
</head>

[**Colabノートブックで試す →**](http://wandb.me/alerts-colab)

W&B アラートを使用すると、W&B Run がクラッシュした場合や、NaN になった損失やMLパイプラインのステップが完了した場合などのカスタムトリガーが達成された場合に、Slackやメールで通知を受け取ることができます。W&B アラートは個人プロジェクトやチームプロジェクトを含むすべてのプロジェクトに適用されます。

次のようにアラートを設定できます：

```python
text = f"Accuracy {acc} is below acceptable threshold {thresh}"

wandb.alert(title="Low accuracy", text=text)
```

そして、Slack（またはメール）でW&Bアラートメッセージを確認できます：

![](/images/track/send_alerts_slack.png)

## はじめに

:::情報
以下の手順はパブリッククラウドでのみアラートを有効にするためのものです。

プライベートクラウドやW&B専用クラウドで [W&B Server](../hosting/intro.md) を使用している場合は、Slackアラートの設定については[こちらのドキュメント](../hosting/monitoring-usage/slack-alerts.md)を参照してください。
:::

コードからトリガーされるSlackまたはメールアラートを初めて送信する場合は、以下の2つの手順を実行します：

1. W&Bの[ユーザー設定](https://wandb.ai/settings)でアラートを有効にする
2. コードに `wandb.alert()` を追加する

### 1. W&Bのユーザー設定でアラートを有効にする

[ユーザー設定](https://wandb.ai/settings)で：

* **Alerts** セクションまでスクロール
* **Scriptable run alerts** をオンにして、`wandb.alert()` からのアラートを受信する設定にする
* **Connect Slack** を使用して、アラートを投稿するSlackチャンネルを選択。**Slackbot** チャンネルがおすすめです。アラートがプライベートに保持されるためです。
* **メール** は、W&Bに登録したメールアドレスに送信されます。これらのアラートが受信トレイを埋めないように、メールのフィルターを設定することをおすすめします。

これは、W&Bアラートを初めて設定する場合や、アラートの受信方法を変更したい場合にのみ行う必要があります。

![W&Bユーザー設定のアラート設定](/images/track/demo_connect_slack.png)

### 2. コードに `wandb.alert()` を追加する

`wandb.alert()` をノートブックまたはPythonスクリプト内のトリガーしたい場所に追加します

```python
wandb.alert(title="High Loss", text="Loss is increasing rapidly")
```

#### Slack またはメールを確認

Slackまたはメールでアラートメッセージを確認します。アラートが届かない場合、ユーザー設定で **Scriptable Alerts** がオンになっているか確認してください。

## `wandb.alert()` の使用

| 引数                      | 説明                                                                                                                                                 |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `title` (string)          | アラートの簡単な説明。例："Low accuracy"                                                                                                              |
| `text` (string)           | アラートがトリガーされた詳細な説明                                                                                                                     |
| `level` (optional)        | アラートの重要度。`AlertLevel.INFO`、`AlertLevel.WARN`、または`AlertLevel.ERROR`のいずれかである必要があります。`wandb` から `AlertLevel.xxx` をインポート可能です |
|                           |                                                                                                                                                       |
| `wait_duration` (optional)| 同じ**タイトル**のアラートを再送信する前に待機する秒数。この設定によりアラートスパムを減らすことができます                                           |

### 例

この簡単なアラートは、精度が閾値を下回ると警告を送信します。この例では、少なくとも5分ごとにしかアラートを送りません。

[コードを実行する →](http://wandb.me/alerts)

```python
import wandb
from wandb import AlertLevel

if acc < threshold:
    wandb.alert(
        title="Low accuracy",
        text=f"Accuracy {acc} is below the acceptable threshold {threshold}",
        level=AlertLevel.WARN,
        wait_duration=300,
    )
```

## その他の情報

### ユーザーのタグ付け / メンション

Slackでアラートを送信する際、アラートのタイトルまたはテキスト内にSlackのユーザーIDを `<@USER_ID>` として追加することで、自分自身や同僚のメンションができます。SlackユーザーIDはそのユーザープロフィールページから見つけることができます。

```python
wandb.alert(title="Loss is NaN", text=f"Hey <@U1234ABCD> loss has gone to NaN")
```

### W&B チームアラート

チーム管理者は、チーム設定ページでチームのためのアラートを設定できます：wandb.ai/teams/`your-team`。これらのアラートはチーム内の全員に適用されます。**Slackbot** チャンネルを使用することをおすすめします。アラートがプライベートに保持されるためです。

### Slackチャンネルの変更

投稿先のチャンネルを変更するには、**Disconnect Slack** をクリックしてから再接続し、別のチャンネルを選択します。

## FAQ(よくある質問)

#### "Run Finished" アラートはJupyterノートブックで動作しますか？

**"Run Finished"** アラート（ユーザー設定の **"Run Finished"** 設定でオンにする）はPythonスクリプトでのみ動作し、Jupyterノートブック環境では各セルの実行ごとに通知が来ないよう無効化されています。Jupyterノートブック環境では `wandb.alert()` を使用してください。

#### [W&B Server](../hosting/intro.md) でアラートを有効にする方法は？