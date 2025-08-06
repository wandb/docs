---
title: アラートを送信
description: Python コードからトリガーされるアラートを、Slack やメールに送信できます
menu:
  default:
    identifier: alert
    parent: what-are-runs
---

{{< cta-button colabLink="https://wandb.me/alerts-colab" >}}

run がクラッシュした際やカスタムトリガーで、Slack やメールでアラートを送信できます。例えば、トレーニングループ内で勾配が急激に大きくなった場合（NaN になる）や、ML パイプライン内のあるステップが完了したときにアラートを作成することが可能です。アラートは、個人プロジェクト・チームプロジェクト問わず、run を初期化するすべての Projects で有効です。

そして、Slack（またはメール）で W&B Alerts のメッセージを受け取ることができます：

{{< img src="/images/track/send_alerts_slack.png" alt="Slack alert setup" >}}

{{% alert %}}
W&B Alerts を利用するには、コードに `run.alert()` を追加する必要があります。コードの修正なしで通知を送りたい場合は、[Automations]({{< relref "/guides/core/automations/" >}}) を使って、W&B 上のイベント（たとえば、[artifact]({{< relref "/guides/core/artifacts" >}}) のバージョン作成時や [run metric]({{< relref "/guides/models/track/runs.md" >}}) が特定のしきい値に達した際など）をトリガーに Slack へ通知できます。

たとえば、新しいバージョンが作成された時に Slack チャンネルへ通知したり、artifact に `production` エイリアスが追加された場合に自動テスト用の webhook を実行したり、run の `loss` が許容範囲内にある場合だけバリデーションジョブを開始したり、といった自動化が可能です。

詳しくは [Automations overview]({{< relref "/guides/core/automations/" >}}) や [create an automation]({{< relref "/guides/core/automations/create-automations/" >}}) をご覧ください。
{{% /alert %}}


## アラートを作成する

{{% alert %}}
このガイドは、マルチテナントクラウドでのアラートについて説明しています。

[W&B Server]({{< relref "/guides/hosting/" >}}) を Private Cloud や W&B Dedicated Cloud で利用している場合は、[W&B Server での Slack アラートの設定方法]({{< relref "/guides/hosting/monitoring-usage/slack-alerts.md" >}}) をご確認ください。
{{% /alert %}}

アラートの設定手順は以下のとおりです。各項目については後述します：

1. W&B の [User Settings](https://wandb.ai/settings) で Alerts を有効にする
2. コードに `run.alert()` を追加する
3. 設定をテストする

### 1. W&B User Settings で Alerts を有効にする

[User Settings](https://wandb.ai/settings) で以下を行います：

* **Alerts** セクションまでスクロールします
* **Scriptable run alerts** を有効にして `run.alert()` からのアラートを受信できるようにします
* **Connect Slack** でアラートを投稿する Slack チャンネルを選択します。**Slackbot** チャンネルの利用を推奨します（アラートがプライベートに保たれます）。
* **Email** は W&B 登録時に利用したメールアドレスに届きます。メールボックスが埋まらないよう専用フォルダへの自動振り分けをおすすめします。

これらの設定は初回の W&B Alerts 利用時、または受信方法を変更したい場合のみ必要です。

{{< img src="/images/track/demo_connect_slack.png" alt="Alerts settings in W&B User Settings" >}}

### 2. コードに `run.alert()` を追加する

アラートを発火させたい場所（ノートブックまたは Python スクリプト）に `run.alert()` を追加します

```python
import wandb

run = wandb.init()
run.alert(title="High Loss", text="Loss is increasing rapidly")
```

### 3. 設定をテストする

Slack やメールにアラートメッセージが届くか確認してください。届かない場合は、[User Settings](https://wandb.ai/settings) 内 **Scriptable Alerts** のメール・Slack 設定がオンになっているか確認しましょう。

## 例

このシンプルなアラートは、accuracy がしきい値を下回った場合に警告を送信します。この例では、アラートの間隔を最低 5 分空けるようにしています。

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


## ユーザーをタグ・メンションする

アラートのタイトルや本文内で、`@` の後に Slack ユーザー ID を記載することで、自分自身や同僚をタグ付けできます。Slack ユーザー ID は、Slack のプロフィールページから確認できます。

```python
run.alert(title="Loss is NaN", text=f"Hey <@U1234ABCD> loss has gone to NaN")
```

## チームアラートの設定

チーム管理者は `wandb.ai/teams/your-team` のチーム settings ページでチーム全体のアラートを設定できます。

チームアラートは同じチーム全員に適用されます。アラートをプライベートに保つため、**Slackbot** チャンネルの利用を推奨します。

## アラート送信先の Slack チャンネルを変更する

アラートの送信先チャンネルを変更するには、**Disconnect Slack** をクリックして一度接続を解除し、再度接続します。その後、別の Slack チャンネルを選択してください。