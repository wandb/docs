---
title: アラートを送信する
description: Python コード からトリガーされるアラートを Slack やメールに送信する
menu:
  default:
    identifier: ja-guides-models-track-runs-alert
    parent: what-are-runs
---

{{< cta-button colabLink="https://wandb.me/alerts-colab" >}}

run がクラッシュした場合や、カスタムトリガーで Slack やメールにアラートを送ることができます。たとえば、トレーニングループの勾配が発散した場合（NaN を報告）や、ML パイプラインのステップが完了した際にアラートを送信することができます。アラートは、run を初期化したすべての Projects（個人・チーム問わず）に適用されます。

そして、Slack（またはメール）で W&B Alerts のメッセージを確認できます。

{{< img src="/images/track/send_alerts_slack.png" alt="Slack alert setup" >}}

{{% alert %}}
W&B Alerts を使うには、コード内で `run.alert()` を追加する必要があります。コードを修正せずに、[Automations]({{< relref path="/guides/core/automations/" lang="ja" >}}) を使えば、W&B 内のイベント（たとえば [artifact]({{< relref path="/guides/core/artifacts" lang="ja" >}}) のバージョン作成時や [run metric]({{< relref path="/guides/models/track/runs.md" lang="ja" >}}) のしきい値到達・変化時）をトリガーに Slack へ通知を送ることもできます。

たとえば、Automations を使って新しいバージョンが作成された時に Slack チャンネルに通知したり、`production` エイリアスがアーティファクトに追加された時に自動テスト用の Webhook を実行したり、run の `loss` が許容範囲内に収まった時だけバリデーションジョブを実行したりできます。

[Automations の概要]({{< relref path="/guides/core/automations/" lang="ja" >}})や、[Automations の作成方法]({{< relref path="/guides/core/automations/create-automations/" lang="ja" >}})もご参照ください。
{{% /alert %}}


## アラートを作成する

{{% alert %}}
以下の手順は、マルチテナントクラウドでのアラート設定に適用されます。

Private Cloud や W&B Dedicated Cloud 上で [W&B Server]({{< relref path="/guides/hosting/" lang="ja" >}}) を使っている場合は、[W&B Server での Slack アラート設定]({{< relref path="/guides/hosting/monitoring-usage/slack-alerts.md" lang="ja" >}}) をご参照ください。
{{% /alert %}}

アラートの設定は、次の手順で行います。（詳細は後述）

1. W&B の [User Settings](https://wandb.ai/settings) でアラート機能を有効にします。
2. コードに `run.alert()` を追加します。
3. 設定をテストします。

### 1. W&B の User Settings でアラートを有効化

[User Settings](https://wandb.ai/settings) にアクセスし、

* **Alerts** セクションまでスクロールします
* **Scriptable run alerts** を有効にして、`run.alert()` からアラートを受け取れるようにします
* **Connect Slack** を利用して、アラートを投稿する Slack チャンネルを選択します。**Slackbot** チャンネルの利用を推奨します（アラートが非公開で通知されます）
* **Email** は W&B 登録時のメールアドレスに送信されます。アラート専用のフォルダを作成して、受信箱が埋まらないようにフィルターを設定することをおすすめします。

この設定は、W&B Alerts を初めて使う時、または通知手段を変更したいときだけ必要です。

{{< img src="/images/track/demo_connect_slack.png" alt="Alerts settings in W&B User Settings" >}}

### 2. コードに `run.alert()` を追加

アラートをトリガーしたい場所（ノートブックや Python スクリプト内）で `run.alert()` を追加します。

```python
import wandb

run = wandb.init()
run.alert(title="High Loss", text="Loss is increasing rapidly")
```

### 3. 設定をテストする

Slack またはメールにアラートが届くか確認しましょう。もし届かない場合は、[User Settings](https://wandb.ai/settings) で **Scriptable Alerts** 用のメールまたは Slack 通知が有効になっているかを確認してください。

## 例

このシンプルなアラートは、accuracy がしきい値を下回った場合に警告を送ります。この例ではアラート送信間隔が最低 5 分に制限されています。

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

## ユーザーをタグ付け・メンションする

アラートのタイトルまたはテキストで、アットマーク `@` に続けて Slack ユーザー ID を入力すれば、あなた自身や同僚をメンションできます。Slack ユーザー ID は、その人の Slack プロフィールページから取得できます。

```python
run.alert(title="Loss is NaN", text=f"Hey <@U1234ABCD> loss has gone to NaN")
```

## チーム アラートの設定

チームの管理者は、`wandb.ai/teams/your-team` のチーム設定ページでアラートを設定できます。

チームアラートは、チームの全員に適用されます。W&B では、アラートを非公開にできる **Slackbot** チャンネルの利用をおすすめします。

## アラートを送信する Slack チャンネルを変更する

アラートを送るチャンネルを変更したい場合は、**Disconnect Slack** をクリックしてから再度接続しなおしてください。再接続後、お好きな Slack チャンネルを選び直せます。