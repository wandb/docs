---
title: Send an alert
description: Python コードからトリガーされたアラートを Slack またはメールに送信します。
menu:
  default:
    identifier: ja-guides-models-track-runs-alert
    parent: what-are-runs
---

{{< cta-button colabLink="http://wandb.me/alerts-colab" >}}

run がクラッシュした場合、またはカスタムトリガーで Slack またはメールでアラートを作成します。たとえば、トレーニング ループの 勾配 が急上昇 (NaN を報告) し始めた場合、または ML パイプライン のステップが完了した場合にアラートを作成できます。アラートは、個人 プロジェクト と チーム プロジェクト の両方を含む、run を初期化するすべての プロジェクト に適用されます。

次に、Slack (またはメール) で W&B Alerts メッセージを確認します。

{{< img src="/images/track/send_alerts_slack.png" alt="" >}}

## アラートの作成方法

{{% alert %}}
次の ガイド は、マルチテナント クラウド のアラートにのみ適用されます。

[W&B サーバー ]({{< relref path="/guides/hosting/" lang="ja" >}})を プライベートクラウド または W&B 専用クラウド で使用している場合は、[この ドキュメント ]({{< relref path="/guides/hosting/monitoring-usage/slack-alerts.md" lang="ja" >}})を参照して、Slack アラート を設定してください。
{{% /alert %}}

アラートを設定するには、主に次の 2 つの手順があります。

1. W&B [ユーザー 設定 ](https://wandb.ai/settings)でアラートをオンにします
2. `run.alert()` を コード に追加します
3. アラートが正しく設定されていることを確認します

### 1. W&B ユーザー 設定 でアラートをオンにします

[ユーザー 設定 ](https://wandb.ai/settings)で:

* **アラート** セクションまでスクロールします
* `run.alert()` からアラートを受信するには、[**スクリプト化可能な run アラート**] をオンにします
* [**Slack に接続**] を使用して、アラートを投稿する Slack チャンネル を選択します。アラートを非公開にするため、[**Slackbot**] チャンネル をお勧めします。
* [**メール**] は、W&B にサインアップしたときに使用した メールアドレス に送信されます。これらのアラートがすべてフォルダに移動し、受信トレイがいっぱいにならないように、メール でフィルタを設定することをお勧めします。

これは、W&B Alerts を初めて設定するとき、またはアラートの受信方法を変更する場合にのみ行う必要があります。

{{< img src="/images/track/demo_connect_slack.png" alt="W&B ユーザー 設定 のアラート 設定 " >}}

### 2. `run.alert()` を コード に追加します

`run.alert()` を、トリガーする Notebook または Python スクリプト の コード に追加します。

```python
import wandb

run = wandb.init()
run.alert(title="High Loss", text="Loss is increasing rapidly")  # 損失が急速に増加している
```

### 3. Slack またはメール を確認してください

Slack または メール でアラート メッセージ を確認してください。何も受信しなかった場合は、[ユーザー 設定 ](https://wandb.ai/settings)で [**スクリプト化可能なアラート**] の メール または Slack がオンになっていることを確認してください。

### 例

このシンプルな アラート は、精度がしきい値を下回った場合に警告を送信します。この例では、少なくとも 5 分間隔で アラート を送信します。

```python
import wandb
from wandb import AlertLevel

run = wandb.init()

if acc < threshold:
    run.alert(
        title="Low accuracy",
        text=f"Accuracy {acc} is below the acceptable threshold {threshold}", # 精度 {acc} が許容できるしきい値 {threshold} を下回っています
        level=AlertLevel.WARN,
        wait_duration=300,
    )
```

## ユーザー をタグ付けまたはメンションする方法

アラート のタイトルまたはテキストで、自分自身または同僚をタグ付けするには、アットマーク `@` の後に Slack ユーザー ID を使用します。Slack ユーザー ID は、Slack プロファイル ページから確認できます。

```python
run.alert(title="Loss is NaN", text=f"Hey <@U1234ABCD> loss has gone to NaN") # Loss が NaN になりました
```

## チーム アラート

チーム 管理者 は、チーム 設定 ページ: `wandb.ai/teams/your-team` で チーム の アラート を設定できます。

チーム アラート は、チーム の全員に適用されます。W&B は、アラート を非公開にするため、[**Slackbot**] チャンネル を使用することをお勧めします。

## アラート の送信先 Slack チャンネル を変更する

アラート の送信先 チャンネル を変更するには、[**Slack の接続を解除**] をクリックしてから、再接続します。再接続後、別の Slack チャンネル を選択します。
