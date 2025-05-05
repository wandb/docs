---
title: アラートを送信する
description: Python コードからトリガーされたアラートを Slack またはメールに送信する
menu:
  default:
    identifier: ja-guides-models-track-runs-alert
    parent: what-are-runs
---

{{< cta-button colabLink="http://wandb.me/alerts-colab" >}}

run がクラッシュしたり、カスタムトリガーで Slack やメールでアラートを作成します。例えば、トレーニングループの勾配が膨らみ始めた場合（NaNを報告）や、ML パイプライン内のステップが完了した場合などです。アラートは、個人およびチームプロジェクトの両方を含む、run を初期化するすべての Projects に適用されます。

その後、Slack（またはメール）で W&B Alerts メッセージを確認します。

{{< img src="/images/track/send_alerts_slack.png" alt="" >}}

## アラートの作成方法

{{% alert %}}
以下のガイドは、マルチテナントクラウドでのアラートにのみ適用されます。

プライベートクラウドまたは W&B 専用クラウドで [W&B Server]({{< relref path="/guides/hosting/" lang="ja" >}}) を使用している場合は、Slack アラートの設定については[こちらのドキュメント]({{< relref path="/guides/hosting/monitoring-usage/slack-alerts.md" lang="ja" >}})を参照してください。
{{% /alert %}}

アラートを設定するための主なステップは2つあります。

1. W&B の [User Settings](https://wandb.ai/settings) で Alerts をオンにする
2. コードに `run.alert()` を追加する
3. アラートが正しく設定されているか確認する

### 1. W&B User Settings でアラートをオンにする

[User Settings](https://wandb.ai/settings) の中で：

* **Alerts** セクションまでスクロールします
* **Scriptable run alerts** をオンにして `run.alert()` からのアラートを受け取ります
* **Connect Slack** を使用して、アラートを投稿する Slack チャンネルを選択します。アラートを非公開に保持するため、**Slackbot** チャンネルをお勧めします。
* **Email** は W&B にサインアップしたときに使用したメールアドレスに送られます。これらのアラートがフォルダに入り、受信トレイを埋めないようにメールにフィルターを設定することをお勧めします。

W&B Alerts を初めて設定する際、またはアラートの受け取り方を変更したい場合にのみ、これを行う必要があります。

{{< img src="/images/track/demo_connect_slack.png" alt="Alerts settings in W&B User Settings" >}}

### 2. コードに `run.alert()` を追加する

ノートブックや Python スクリプトのどこでトリガーしたいかに `run.alert()` をコードに追加します。

```python
import wandb

run = wandb.init()
run.alert(title="High Loss", text="Loss is increasing rapidly")
```

### 3. Slack またはメールを確認する

アラートメッセージのために Slack またはメールをチェックします。受信していない場合は、[User Settings](https://wandb.ai/settings) で **Scriptable Alerts** 用のメールまたは Slack がオンになっていることを確認してください

### 例

このシンプルなアラートは、精度が閾値を下回ると警告を送信します。この例では、少なくとも 5 分おきにアラートを送信します。

```python
import wandb
from wandb import AlertLevel

run = wandb.init()

if acc < threshold:
    run.alert(
        title="Low accuracy",
        text=f"Accuracy {acc} is below the acceptable threshold {threshold}",
        level=AlertLevel.WARN,
        wait_duration=300,
    )
```

## ユーザーをタグ付けまたはメンションする方法

アットマーク `@` に続けて Slack ユーザー ID を使用して、アラートのタイトルまたはテキストで自身または同僚をタグ付けします。Slack ユーザー ID は、彼らの Slack プロフィールページから見つけることができます。

```python
run.alert(title="Loss is NaN", text=f"Hey <@U1234ABCD> loss has gone to NaN")
```

## チームアラート

チーム管理者は、チームの設定ページでチーム用のアラートを設定できます：`wandb.ai/teams/your-team`。

チームアラートは、チームの全員に適用されます。W&B は、アラートを非公開に保持するために **Slackbot** チャンネルを使用することをお勧めします。

## アラート送信先の Slack チャンネルを変更する

アラートの送信先を変更するには、**Disconnect Slack** をクリックして、再接続してください。再接続後、別の Slack チャンネルを選択します。