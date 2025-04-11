---
title: 自動化の作成
menu:
  default:
    identifier: ja-guides-core-automations-create-automations-_index
    parent: automations
weight: 1
---

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

このページでは、W&B の [オートメーション]({{< relref path="/guides/core/automations/" lang="ja" >}}) を作成および管理する概要を示します。詳細な手順については [Slack オートメーションを作成する]({{< relref path="/guides/core/automations/create-automations/slack.md" lang="ja" >}}) または [Webhook オートメーションを作成する]({{< relref path="/guides/core/automations/create-automations/webhook.md" lang="ja" >}}) を参照してください。

{{% alert %}}
オートメーションに関するチュートリアルをお探しですか？
- [モデルの評価とデプロイメントのための Github アクションを自動的にトリガーする方法を学ぶ](https://wandb.ai/wandb/wandb-model-cicd/reports/Model-CI-CD-with-W-B--Vmlldzo0OTcwNDQw)。
- [モデルを Sagemaker エンドポイントに自動的にデプロイするデモ動画を見る](https://www.youtube.com/watch?v=s5CMj_w3DaQ)。
- [オートメーションを紹介する動画シリーズを見る](https://youtube.com/playlist?list=PLD80i8An1OEGECFPgY-HPCNjXgGu-qGO6&feature=shared)。
{{% /alert %}}

## 要件
- チーム管理者は、チームの Projects やオートメーションのコンポーネント（Webhook、秘密情報、Slack 接続など）のオートメーションを作成および管理できます。[チーム設定]({{< relref path="/guides/models/app/settings-page/team-settings/" lang="ja" >}})を参照してください。
- レジストリオートメーションを作成するには、レジストリへのアクセスが必要です。[レジストリアクセスの設定]({{< relref path="/guides/core/registry/configure_registry.md#registry-roles" lang="ja" >}})を参照してください。
- Slack オートメーションを作成するには、選択した Slack インスタンスとチャンネルに投稿する権限が必要です。

## オートメーションを作成する
プロジェクトまたはレジストリの **Automations** タブからオートメーションを作成します。オートメーションを作成するには、以下のステップに従います：

1. 必要に応じて、オートメーションに必要な各機密文字列（アクセストークン、パスワード、SSH キーなど）のために [W&B の秘密情報を作成する]({{< relref path="/guides/core/secrets.md" lang="ja" >}})。秘密情報は **Team Settings** で定義されます。秘密情報は主に Webhook オートメーションで使用されます。
1. Webhook または Slack 通知を設定して、W&B が Slack に投稿するか、代わりに Webhook を実行できるようにします。単一のオートメーションアクション（Webhook または Slack 通知）は、複数のオートメーションによって使用できます。これらのアクションは **Team Settings** で定義されます。
1. プロジェクトまたはレジストリで、監視するイベントと実行するアクション（Slack への投稿や Webhook の実行など）を指定するオートメーションを作成します。Webhook オートメーションを作成するときは、送信するペイロードを設定します。

詳細については、以下を参照してください：

- [Slack オートメーションを作成する]({{< relref path="slack.md" lang="ja" >}})
- [Webhook オートメーションを作成する]({{< relref path="webhook.md" lang="ja" >}})

## オートメーションを表示および管理する
プロジェクトまたはレジストリの **Automations** タブからオートメーションを表示および管理します。

- オートメーションの詳細を表示するには、その名前をクリックします。
- オートメーションを編集するには、そのアクションの `…` メニューをクリックし、**Edit automation** をクリックします。
- オートメーションを削除するには、そのアクションの `…` メニューをクリックし、**Delete automation** をクリックします。

## 次のステップ
- [Slack オートメーションを作成する]({{< relref path="slack.md" lang="ja" >}})。
- [Webhook オートメーションを作成する]({{< relref path="webhook.md" lang="ja" >}})。
- [秘密情報を作成する]({{< relref path="/guides/core/secrets.md" lang="ja" >}})。