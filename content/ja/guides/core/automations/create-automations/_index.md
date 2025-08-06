---
title: 自動化を作成する
menu:
  default:
    identifier: create-automations
    parent: automations
weight: 1
---

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

このページでは、W&B の [オートメーション]({{< relref "/guides/core/automations/">}}) を作成および管理するための概要を説明します。より詳細な手順については、[Slack オートメーションの作成]({{< relref "/guides/core/automations/create-automations/slack.md" >}}) または [Webhook オートメーションの作成]({{< relref "/guides/core/automations/create-automations/webhook.md" >}}) をご覧ください。

{{% alert %}}
オートメーションに関するチュートリアルをお探しですか？
- [Github Action を自動でトリガーしてモデルの評価やデプロイを行う方法を学ぶ](https://wandb.ai/wandb/wandb-model-cicd/reports/Model-CI-CD-with-W-B--Vmlldzo0OTcwNDQw)
- [モデルを Sagemaker エンドポイントに自動デプロイするデモ動画を視聴する](https://www.youtube.com/watch?v=s5CMj_w3DaQ)
- [オートメーションのイントロ動画シリーズを視聴する](https://youtube.com/playlist?list=PLD80i8An1OEGECFPgY-HPCNjXgGu-qGO6&feature=shared)
{{% /alert %}}

## 必要条件
- チームの管理者は、チームの Projects のためのオートメーションや、その構成要素（Webhook、Secret、Slack インテグレーションなど）を作成・管理できます。詳細は [Team settings]({{< relref "/guides/models/app/settings-page/team-settings/" >}}) をご覧ください。
- Registry オートメーションを作成するには、その Registry へのアクセス権が必要です。[Configure Registry access]({{< relref "/guides/core/registry/configure_registry.md#registry-roles" >}}) を参照してください。
- Slack オートメーションを作成するには、選択した Slack インスタンスやチャンネルへ投稿する権限が必要です。

## オートメーションの作成
Project または Registry の **Automations** タブからオートメーションを作成します。大まかな流れは以下の通りです。

1. オートメーションに必要なアクセストークン、パスワード、SSH キーなどの機密文字列ごとに、必要に応じて [W&B Secret を作成]({{< relref "/guides/core/secrets.md" >}}) します。Secret は **Team Settings** で定義します。Secret は主に Webhook オートメーションで使われます。
1. Webhook または Slack インテグレーションを設定し、W&B が Slack への投稿や Webhook の実行を代行できるようにします。1つの Webhook または Slack インテグレーションを複数のオートメーションで利用できます。これらの操作も **Team Settings** で設定します。
1. Project または Registry でオートメーションを作成し、監視するイベントや実行するアクション（Slack への投稿や Webhook の実行など）を指定します。Webhook オートメーションを作成する場合は、送信するペイロードも設定します。

または、Workspace のラインプロット上から該当メトリクス用の [run Metric オートメーション]({{< relref "/guides/core/automations/automation-events.md#run-events" >}}) を素早く作成できます。

1. パネルにカーソルを合わせて、パネル上部のベルアイコンをクリックします。

    {{< img src="/images/automations/run_metric_automation_from_panel.png" alt="Automation bell icon location" >}}
1. 基本設定または詳細設定コントロールを使ってオートメーションを設定します。例えば、run フィルタを適用してオートメーションの範囲を絞ったり、絶対値の閾値を設定したりできます。

詳しくは、以下を参照してください。

- [Slack オートメーションの作成]({{< relref "slack.md" >}})
- [Webhook オートメーションの作成]({{< relref "webhook.md" >}})

## オートメーションの確認と管理
Project もしくは Registry の **Automations** タブから、オートメーションの確認・管理ができます。

- オートメーションの詳細を確認するには、その名前をクリックします。
- オートメーションを編集するには、アクションの `...` メニューから **Edit automation** をクリックします。
- オートメーションを削除するには、アクションの `...` メニューから **Delete automation** をクリックします。

## 次のステップ
- [オートメーションのイベントとスコープ]({{< relref "/guides/core/automations/automation-events.md" >}}) について詳しく学ぶ
- [Slack オートメーションの作成]({{< relref "slack.md" >}})
- [Webhook オートメーションの作成]({{< relref "webhook.md" >}})
- [Secret の作成]({{< relref "/guides/core/secrets.md" >}})