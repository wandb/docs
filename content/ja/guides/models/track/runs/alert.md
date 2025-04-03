---
title: Send an alert
description: Python コードからトリガーされたアラートを Slack またはメールに送信します。
menu:
  default:
    identifier: ja-guides-models-track-runs-alert
    parent: what-are-runs
---

{{< cta-button colabLink="http://wandb.me/alerts-colab" >}}

run がクラッシュした場合、またはカスタムトリガーで Slack またはメールでアラートを作成します。たとえば、トレーニングループの勾配が爆発し始めた場合 (NaN を報告)、または ML パイプライン のステップが完了した場合にアラートを作成できます。アラートは、個人 プロジェクト と Team プロジェクト の両方を含む、run を初期化するすべての プロジェクト に適用されます。

次に、Slack (またはメール) で W&B Alerts メッセージを確認します。

{{< img src="/images/track/send_alerts_slack.png" alt="" >}}

## アラートの作成方法

{{% alert %}}
次の ガイド は、マルチテナント cloud のアラートにのみ適用されます。

[W&B Server]({{< relref path="/guides/hosting/" lang="ja" >}}) を Private Cloud または W&B Dedicated Cloud で使用している場合は、[このドキュメント]({{< relref path="/guides/hosting/monitoring-usage/slack-alerts.md" lang="ja" >}}) を参照して、Slack アラートを設定してください。
{{% /alert %}}

アラートを設定するには、主に次の 2 つのステップがあります。

1. W&B の [ユーザー 設定](https://wandb.ai/settings) でアラートをオンにします
2. `run.alert()` を code に追加します
3. アラートが適切に設定されていることを確認します

### 1. W&B ユーザー 設定でアラートをオンにする

[ユーザー 設定](https://wandb.ai/settings) で:

* **アラート** セクションまでスクロールします
* **スクリプト可能な run アラート** をオンにして、`run.alert()` からアラートを受信します
* **Slack に接続** を使用して、アラートを投稿する Slack channel を選択します。アラートを非公開にするため、**Slackbot** channel をお勧めします。
* **メール** は、W&B にサインアップしたときに使用したメール アドレスに送信されます。これらのアラートがすべてフォルダーに移動し、受信トレイがいっぱいにならないように、メールでフィルターを設定することをお勧めします。

W&B Alerts を初めて設定する場合、またはアラートの受信方法を変更する場合は、これを行う必要があります。

{{< img src="/images/track/demo_connect_slack.png" alt="W&B ユーザー設定のアラート設定" >}}

### 2. `run.alert()` を code に追加する

アラートをトリガーする場所に、`run.alert()` を code ( notebook または Python スクリプト のいずれか) に追加します

```python
import wandb

run = wandb.init()
run.alert(title="High Loss", text="Loss is increasing rapidly")
```

### 3. Slack またはメールを確認する

Slack またはメールでアラート メッセージを確認します。何も受信しなかった場合は、[ユーザー 設定](https://wandb.ai/settings) で **スクリプト可能なアラート** のメールまたは Slack がオンになっていることを確認してください。

### 例

このシンプルなアラートは、精度がしきい値を下回ると警告を送信します。この例では、少なくとも 5 分間隔でアラートを送信します。

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

アラートのタイトルまたはテキストのいずれかで、@ 記号の後に Slack ユーザー ID を続けて入力して、自分自身または同僚をタグ付けします。Slack ユーザー ID は、Slack プロフィール ページから確認できます。

```python
run.alert(title="Loss is NaN", text=f"Hey <@U1234ABCD> loss has gone to NaN")
```

## Team アラート

Team 管理者は、Team 設定ページ `wandb.ai/teams/your-team` で Team のアラートを設定できます。

Team アラートは、Team の全員に適用されます。W&B は、アラートを非公開にするため、**Slackbot** channel を使用することをお勧めします。

## アラートの送信先 Slack channel の変更

アラートの送信先 channel を変更するには、**Slack の接続を解除** をクリックしてから、再接続します。再接続後、別の Slack channel を選択します。
