---
title: オートメーション
menu:
  default:
    identifier: ja-guides-core-automations-_index
    parent: core
weight: 4
---

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

このページでは、W&B の _オートメーション_ について説明します。ワークフローステップをトリガーする [オートメーションの作成]({{< relref path="create-automations/" lang="ja" >}}) は、W&B 内のイベント（例えば、アーティファクトバージョンが作成されたとき）に基づいて、自動モデルテストやデプロイメントなどを行います。

例えば、新しいバージョンが作成されたときに Slack のチャンネルに投稿したり、アーティファクトに `production` エイリアスが追加されたときに自動テストをトリガーするためにウェブフックを実行することができます。

## 概要
オートメーションは、レジストリやプロジェクトで特定の[イベント]({{< relref path="automation-events.md" lang="ja" >}})が発生したときに実行できます。

[レジストリ]({{< relref path="/guides/core/registry/" lang="ja" >}})内のアーティファクトの場合、オートメーションを設定して次の場合に実行することができます：
- 新しいアーティファクトバージョンがコレクションにリンクされたとき。例えば、新しい候補モデルに対するテストおよび検証ワークフローをトリガーします。
- アーティファクトバージョンにエイリアスが追加されたとき。例えば、モデルバージョンにエイリアスが追加されたときにデプロイメントワークフローをトリガーします。

[プロジェクト]({{< relref path="/guides/models/track/project-page.md" lang="ja" >}})内のアーティファクトの場合、オートメーションを設定して次の場合に実行することができます：
- 新しいバージョンがアーティファクトに追加されたとき。例えば、指定されたコレクションにデータセットアーティファクトの新しいバージョンが追加されたときにトレーニングジョブを開始します。
- アーティファクトバージョンにエイリアスが追加されたとき。例えば、データセットアーティファクトにエイリアス「redaction」が追加されたときに PII 編集ワークフローをトリガーします。

詳細については、[オートメーションイベントとスコープ]({{< relref path="automation-events.md" lang="ja" >}})を参照してください。

[オートメーションを作成するには]({{< relref path="create-automations/" lang="ja" >}})、以下を行います：

1. 必要に応じて、自動化に必要なアクセス トークン、パスワード、またはセンシティブな設定の詳細などの機密文字列のために、[シークレット]({{< relref path="/guides/core/secrets.md" lang="ja" >}})を設定します。シークレットは **Team Settings** で定義されます。シークレットは、資格情報やトークンをプレーン テキストで公開することなく、Webhook のペイロードにハードコードすることなく、安全に外部サービスに渡すために Webhook オートメーションで最も一般的に使用されます。
1. Webhook または Slack 通知を設定して、Slack に投稿したり、ユーザーに代わって Webhook を実行するように W&B を承認します。単一のオートメーションアクション（Webhookまたは Slack 通知）は、複数のオートメーションで使用できます。これらのアクションは、**Team Settings** で定義されます。
1. プロジェクトまたはレジストリでオートメーションを作成します：
    1. 新しいアーティファクトバージョンが追加されたときなど、監視する[イベント]({{< relref path="#automation-events" lang="ja" >}})を定義します。
    1. イベントが発生したときに取るアクション（Slack チャンネルへの投稿またはウェブフックの実行）を定義します。ウェブフックの場合、必要に応じてペイロードと共に送信するアクセス トークンやシークレットを使用するためのシークレットを指定します。

## 次のステップ
- [オートメーションを作成する]({{< relref path="create-automations/" lang="ja" >}})。
- [オートメーションのイベントとスコープ]({{< relref path="automation-events.md" lang="ja" >}})について学ぶ。
- [シークレットを作成する]({{< relref path="/guides/core/secrets.md" lang="ja" >}})。