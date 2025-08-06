---
title: オートメーションを作成
menu:
  default:
    identifier: ja-guides-core-automations-create-automations-_index
    parent: automations
weight: 1
---

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

このページでは、W&Bの[オートメーション]({{< relref path="/guides/core/automations/" lang="ja" >}})の作成と管理の概要を紹介します。より詳しい手順については、[Slack オートメーションの作成]({{< relref path="/guides/core/automations/create-automations/slack.md" lang="ja" >}})や[Webhook オートメーションの作成]({{< relref path="/guides/core/automations/create-automations/webhook.md" lang="ja" >}})をご覧ください。

{{% alert %}}
オートメーションのチュートリアルをお探しですか？
- [Github Action を自動でトリガーしてモデルの評価・デプロイを行う方法](https://wandb.ai/wandb/wandb-model-cicd/reports/Model-CI-CD-with-W-B--Vmlldzo0OTcwNDQw) を学ぶ。
- [モデルを Sagemaker エンドポイントに自動デプロイする動画を見る](https://www.youtube.com/watch?v=s5CMj_w3DaQ)。
- [オートメーションの紹介動画シリーズを見る](https://youtube.com/playlist?list=PLD80i8An1OEGECFPgY-HPCNjXgGu-qGO6&feature=shared)。
{{% /alert %}}

## 必要条件
- チーム管理者は、そのチームの Projects およびオートメーションの構成要素（Webhook、シークレット、Slack インテグレーションなど）の作成と管理ができます。詳細は[チーム設定]({{< relref path="/guides/models/app/settings-page/team-settings/" lang="ja" >}})を参照してください。
- レジストリ オートメーションを作成するには、そのレジストリへのアクセス権限が必要です。[Registry アクセスの設定]({{< relref path="/guides/core/registry/configure_registry.md#registry-roles" lang="ja" >}})を参照してください。
- Slack オートメーションを作成するには、選択した Slack インスタンスおよびチャンネルへの投稿権限が必要です。

## オートメーションの作成
Project または Registry の **Automations** タブからオートメーションを作成できます。大まかな流れは、以下の手順となります。

1. オートメーションで必要とされるアクセストークン、パスワード、SSHキーなど、各種機密文字列ごとに[W&B シークレットを作成]({{< relref path="/guides/core/secrets.md" lang="ja" >}})します。シークレットは**Team Settings**で管理され、Webhook オートメーションで特によく利用されます。
1. Webhook または Slack インテグレーションを設定し、W&B が Slack への投稿または Webhook の実行をできるように認可します。Webhook や Slack インテグレーションは複数のオートメーションで共有できます。これらの操作も**Team Settings**で管理されます。
1. Project または Registry 内でオートメーションを作成し、監視したいイベントおよびアクション内容（Slack への投稿や Webhook の実行など）を指定します。Webhook オートメーション作成時には、送信されるペイロードの設定も行います。

また、Workspace の折れ線グラフパネルから、表示している指標に対して素早く[run メトリクス オートメーション]({{< relref path="/guides/core/automations/automation-events.md#run-events" lang="ja" >}})を作成することもできます。

1. パネルの上にカーソルを重ね、パネル上部のベルアイコンをクリックします。

    {{< img src="/images/automations/run_metric_automation_from_panel.png" alt="Automation bell icon location" >}}
1. ベーシックまたはアドバンスト設定でオートメーション内容を調整します。たとえば run フィルターを適用して対象範囲を絞り込んだり、絶対値のしきい値を設定したりできます。

詳しい手順は以下もご覧ください。

- [Slack オートメーションの作成]({{< relref path="slack.md" lang="ja" >}})
- [Webhook オートメーションの作成]({{< relref path="webhook.md" lang="ja" >}})

## オートメーションの確認・管理
Project または Registry の **Automations** タブで、オートメーションの確認および管理ができます。

- オートメーションの詳細を確認するには、その名前をクリックします。
- 編集する場合は、アクションの `...` メニューをクリックし、**Edit automation** を選択します。
- 削除する場合も、アクションの `...` メニューから **Delete automation** を選択します。

## 次のステップ
- [オートメーションのイベントやスコープについて]({{< relref path="/guides/core/automations/automation-events.md" lang="ja" >}})さらに詳しく学ぶ
- [Slack オートメーションの作成]({{< relref path="slack.md" lang="ja" >}})。
- [Webhook オートメーションの作成]({{< relref path="webhook.md" lang="ja" >}})。
- [シークレットの作成]({{< relref path="/guides/core/secrets.md" lang="ja" >}})方法を確認する。