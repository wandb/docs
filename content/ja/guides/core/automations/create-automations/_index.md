---
title: オートメーションを作成する
cascade:
- url: /guides/automations/create-automations/:filename
menu:
  default:
    identifier: ja-guides-core-automations-create-automations-_index
    parent: automations
url: /guides/automations/create-automations/
weight: 1
---

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

このページでは、W&B の [オートメーション]({{< relref path="/guides/core/automations/" lang="ja" >}}) の作成と管理について概要を説明します。詳細な手順については、[Slack オートメーションを作成する]({{< relref path="/guides/core/automations/create-automations/slack.md" lang="ja" >}}) または [Webhook オートメーションを作成する]({{< relref path="/guides/core/automations/create-automations/webhook.md" lang="ja" >}}) を参照してください。

{{% alert %}}
オートメーションの関連チュートリアルをお探しですか？
- [モデルの評価とデプロイメントのために GitHub Actions を自動的にトリガーする方法を学ぶ](https://wandb.ai/wandb/wandb-model-cicd/reports/Model-CI-CD-with-W-B--Vmlldzo0OTcwNDQw)。
- [モデルを SageMaker エンドポイントに自動的にデプロイする動画を見る](https://www.youtube.com/watch?v=s5CMj_w3DaQ)。
- [オートメーションを紹介する動画シリーズを見る](https://youtube.com/playlist?list=PLD80i8An1OEGECFPgY-HPCNjXgGu-qGO6&feature=shared)。
{{% /alert %}}

## 要件
- チームの管理者は、チームの Project におけるオートメーションや、Webhook、シークレット、Slack インテグレーションなどのオートメーションのコンポーネントを作成および管理できます。[チーム設定]({{< relref path="/guides/models/app/settings-page/team-settings/" lang="ja" >}}) を参照してください。
- Registry オートメーションを作成するには、Registry へのアクセスが必要です。[Registry アクセスを構成する]({{< relref path="/guides/core/registry/configure_registry.md#registry-roles" lang="ja" >}}) を参照してください。
- Slack オートメーションを作成するには、選択した Slack インスタンスとチャンネルに投稿する権限が必要です。

## オートメーションを作成する
Project または Registry の **Automations** タブからオートメーションを作成します。大まかな手順は次のとおりです。

1. 必要に応じて、アクセストークン、パスワード、SSH キーなど、オートメーションに必要な機密文字列ごとに [W&B シークレットを作成する]({{< relref path="/guides/core/secrets.md" lang="ja" >}}) 必要があります。シークレットは **Team Settings** で定義されます。シークレットは、Webhook オートメーションで最も一般的に使用されます。
1. Webhook または Slack インテグレーションを設定し、W&B が Slack に投稿したり、ユーザーに代わって Webhook を実行したりすることを承認します。単一の Webhook または Slack インテグレーションは、複数のオートメーションで使用できます。これらのアクションは **Team Settings** で定義されます。
1. Project または Registry でオートメーションを作成します。監視するイベントと実行するアクション（Slack への投稿や Webhook の実行など）を指定します。Webhook オートメーションを作成する場合は、送信するペイロードを設定します。

あるいは、Workspace の線形プロットから、表示されるメトリックの [run メトリックのオートメーション]({{< relref path="/guides/core/automations/automation-events.md#run-events" lang="ja" >}}) をすばやく作成できます。

1. パネルにカーソルを合わせ、パネル上部のベルアイコンをクリックします。

    {{< img src="/images/automations/run_metric_automation_from_panel.png" alt="オートメーションのベルアイコンの位置" >}}
1. 基本または詳細の設定コントロールを使ってオートメーションを設定します。たとえば、オートメーションの範囲を絞るために run フィルターを適用したり、絶対しきい値を設定したりします。

詳細については、以下を参照してください。

- [Slack オートメーションを作成する]({{< relref path="slack.md" lang="ja" >}})
- [Webhook オートメーションを作成する]({{< relref path="webhook.md" lang="ja" >}})

## オートメーションを表示および管理する
Project または Registry の **Automations** タブからオートメーションを表示および管理します。

- オートメーションの詳細を表示するには、その名前をクリックします。
- オートメーションを編集するには、そのアクション `...` メニューをクリックし、**Edit automation** をクリックします。
- オートメーションを削除するには、そのアクション `...` メニューをクリックし、**Delete automation** をクリックします。

## 次のステップ
- [オートメーション イベントとスコープ]({{< relref path="/guides/core/automations/automation-events.md" lang="ja" >}}) について詳しく学ぶ
- [Slack オートメーションを作成する]({{< relref path="slack.md" lang="ja" >}})。
- [Webhook オートメーションを作成する]({{< relref path="webhook.md" lang="ja" >}})。
- [シークレットを作成する]({{< relref path="/guides/core/secrets.md" lang="ja" >}})。