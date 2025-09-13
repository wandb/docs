---
title: アラートを送信する
description: Python コードからトリガーされるアラートを Slack または メールに送信します
menu:
  default:
    identifier: ja-guides-models-track-runs-alert
    parent: what-are-runs
---

{{< cta-button colabLink="https://wandb.me/alerts-colab" >}}
run がクラッシュしたとき、またはカスタム トリガーに基づいて、Slack や メール でアラートを作成できます。たとえば、トレーニング ループの 勾配 が発散し始めた（NaN を報告した）ときや、ML パイプライン のステップが完了したときにアラートを作成できます。Alerts は、個人 Projects と Team の Projects の両方を含む、run を初期化したすべての Projects に適用されます。
その後、Slack（またはメール）で W&B Alerts のメッセージを確認できます:
{{< img src="/images/track/send_alerts_slack.png" alt="Slack アラートの設定" >}}
{{% alert %}}
W&B Alerts を使うには、コードに `run.alert()` を追加する必要があります。コードを変更したくない場合は、W&B 上のイベント（たとえば [artifact]({{< relref path="/guides/core/artifacts" lang="ja" >}}) の バージョン が作成されたときや、[run の指標]({{< relref path="/guides/models/track/runs.md" lang="ja" >}}) がしきい値に達した／変化したとき）に基づいて Slack に通知できる別の方法として、[Automations]({{< relref path="/guides/core/automations/" lang="ja" >}}) を利用できます。
たとえば、Automation で新しい バージョン が作成されたときに Slack の チャンネル へ通知したり、artifact に `production` エイリアス が追加されたときに自動テスト用の webhook を実行したり、run の `loss` が許容範囲にある場合にのみ検証ジョブを開始したりできます。
[Automations の概要]({{< relref path="/guides/core/automations/" lang="ja" >}})を読む、または[Automation を作成する]({{< relref path="/guides/core/automations/create-automations/" lang="ja" >}})をご覧ください。
{{% /alert %}}
## アラートを作成する
{{% alert %}}
以下のガイドは、マルチテナント クラウドでのアラートにのみ適用されます。
プライベートクラウド 上または W&B 専用クラウド 上の [W&B Server]({{< relref path="/guides/hosting/" lang="ja" >}}) をお使いの場合は、Slack アラートの設定について [W&B Server で Slack アラートを設定する]({{< relref path="/guides/hosting/monitoring-usage/slack-alerts.md" lang="ja" >}}) を参照してください。
{{% /alert %}}
アラートを設定するには、次の手順に従います。各手順の詳細は後述します。
1. W&B の [User Settings](https://wandb.ai/settings) で Alerts をオンにする。
2. コードに `run.alert()` を追加する。
3. 設定をテストする。
### 1. W&B の User Settings で Alerts をオンにする
[User Settings](https://wandb.ai/settings) で次を行います:
* **Alerts** セクションまでスクロールします
* `run.alert()` からのアラートを受け取るために **Scriptable run alerts** をオンにします
* **Connect Slack** で、アラートの投稿先となる Slack の チャンネル を選びます。非公開にできるため **Slackbot** チャンネルを推奨します。
* **Email** は、W&B にサインアップしたときの メール アドレス宛に送信されます。受信箱が埋まらないよう、これらのアラートがフォルダに振り分けられるようなフィルタ設定をおすすめします。
これらの設定は、初めて W&B Alerts をセットアップするとき、またはアラートの受け取り方法を変更したいときだけ行えば大丈夫です。
{{< img src="/images/track/demo_connect_slack.png" alt="W&B の User Settings にある Alerts の設定" >}}
### 2. コードに `run.alert()` を追加する
`run.alert()` を、トリガーさせたい場所でコード（Notebook または Python スクリプト）に追加します。
```python
import wandb

run = wandb.init()
run.alert(title="High Loss", text="Loss is increasing rapidly")
```
### 3. 設定をテストする
アラート メッセージが届いているか、Slack または メール を確認してください。届いていない場合は、[User Settings](https://wandb.ai/settings) の **Scriptable Alerts** で Email または Slack がオンになっているかを確認してください。
## 例
次のシンプルなアラートは、精度がしきい値を下回ったときに警告を送ります。この例では、少なくとも 5 分の間隔をあけてから次のアラートを送ります。
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
## ユーザーをタグ付けまたはメンションする
アラートのタイトルまたはテキスト内で、自分や同僚をタグ付けするには、アットマーク `@` の後に Slack の ユーザー ID を記述します。Slack のプロフィール ページで ユーザー ID を確認できます。
```python
run.alert(title="Loss is NaN", text=f"Hey <@U1234ABCD> loss has gone to NaN")
```
## チーム用のアラートを設定する
Team の管理者は、チームの設定ページ `wandb.ai/teams/your-team` でチーム向けのアラートを設定できます。
チーム アラートは、Team の全員に適用されます。アラートを非公開に保てるため、W&B は **Slackbot** チャンネルの使用を推奨します。
## アラートの送信先 Slack チャンネルを変更する
送信先のチャンネルを変更するには、**Disconnect Slack** をクリックしてから再接続します。再接続後、別の Slack チャンネルを選択してください。