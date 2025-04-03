---
title: Create an automation
menu:
  default:
    identifier: ja-guides-core-automations-create-automations-_index
    parent: automations
weight: 1
---

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

このページでは、W&B の[オートメーション]({{< relref path="/guides/core/automations/" lang="ja" >}})の作成と管理の概要について説明します。詳細な手順については、[Slack オートメーションの作成]({{< relref path="/guides/core/automations/create-automations/slack.md" lang="ja" >}})または[Webhook オートメーションの作成]({{< relref path="/guides/core/automations/create-automations/webhook.md" lang="ja" >}})を参照してください。

{{% alert %}}
オートメーションに関するチュートリアルをお探しですか？
- [モデルの評価とデプロイメントのために Github Action を自動的にトリガーする方法](https://wandb.ai/wandb/wandb-model-cicd/reports/Model-CI-CD-with-W-B--Vmlldzo0OTcwNDQw)をご覧ください。
- [モデルを Sagemaker エンドポイントに自動的にデプロイするデモビデオ](https://www.youtube.com/watch?v=s5CMj_w3DaQ)をご覧ください。
- [オートメーションを紹介するビデオシリーズ](https://youtube.com/playlist?list=PLD80i8An1OEGECFPgY-HPCNjXgGu-qGO6&feature=shared)をご覧ください。
{{% /alert %}}

## 要件
- Team admin は、チームの Projects のオートメーション、および Webhook、シークレット、Slack 接続などのオートメーションのコンポーネントを作成および管理できます。[Team settings]({{< relref path="/guides/models/app/settings-page/team-settings/" lang="ja" >}})を参照してください。
- Registry automation を作成するには、Registry へのアクセス権が必要です。[Registry アクセスの設定]({{< relref path="/guides/core/registry/configure_registry.md#registry-roles" lang="ja" >}})を参照してください。
- Slack オートメーションを作成するには、選択した Slack インスタンスと channel に投稿する権限が必要です。

## オートメーションの作成
Project または Registry の [**Automations**] タブからオートメーションを作成します。大まかに言って、オートメーションを作成するには、次の手順に従います。

1. 必要に応じて、アクセス・トークン、パスワード、SSH キーなど、オートメーションに必要な機密文字列ごとに[W&B シークレットを作成]({{< relref path="/guides/core/secrets.md" lang="ja" >}})します。シークレットは [**Team Settings**] で定義します。シークレットは、Webhook オートメーションで最も一般的に使用されます。
2. W&B が Slack に投稿したり、代わりに Webhook を実行したりできるように、Webhook または Slack 通知を設定して W&B を承認します。1 つのオートメーションアクション（Webhook または Slack 通知）を複数のオートメーションで使用できます。これらのアクションは [**Team Settings**] で定義します。
3. Project または Registry で、監視するイベントと実行するアクション（Slack への投稿や Webhook の実行など）を指定するオートメーションを作成します。Webhook オートメーションを作成する場合は、送信するペイロードを設定します。

詳細については、以下を参照してください。

- [Slack オートメーションの作成]({{< relref path="slack.md" lang="ja" >}})
- [Webhook オートメーションの作成]({{< relref path="webhook.md" lang="ja" >}})

## オートメーションの表示と管理
Project または Registry の [**Automations**] タブからオートメーションを表示および管理します。

- オートメーションの詳細を表示するには、その名前をクリックします。
- オートメーションを編集するには、そのアクション `...` メニューをクリックし、[**Edit automation**] をクリックします。
- オートメーションを削除するには、そのアクション `...` メニューをクリックし、[**Delete automation**] をクリックします。

## 次のステップ
- [Slack オートメーションの作成]({{< relref path="slack.md" lang="ja" >}})。
- [Webhook オートメーションの作成]({{< relref path="webhook.md" lang="ja" >}})。
- [シークレットの作成]({{< relref path="/guides/core/secrets.md" lang="ja" >}})。
