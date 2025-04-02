---
title: Automations
menu:
  default:
    identifier: ja-guides-core-automations-_index
    parent: core
weight: 4
---

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

このページでは、W&B の _automations_ について説明します。W&B のイベント（[artifact]({{< relref path="/guides/core/artifacts" lang="ja" >}}) Artifacts のバージョンが作成されたときなど）に基づいて、自動モデルテストやデプロイメントなどのワークフローステップをトリガーする [オートメーションの作成]({{< relref path="create-automations/" lang="ja" >}}) を行います。

たとえば、新しいバージョンが作成されたときに Slack チャンネルに投稿したり、`production` エイリアスが Artifacts に追加されたときに webhook を実行して自動テストをトリガーしたりできます。

## Overview
Automation は、特定の [event]({{< relref path="automation-events.md" lang="ja" >}}) が Registry または Project で発生したときに実行できます。

[Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) 内の Artifacts の場合、Automation の実行を次のように設定できます。
- 新しい Artifacts バージョンがコレクションにリンクされたとき。たとえば、新しい候補 Models のテストと検証のワークフローをトリガーします。
- エイリアスが Artifacts バージョンに追加されたとき。たとえば、エイリアスが Model バージョンに追加されたときに、デプロイメント ワークフローをトリガーします。

[Project]({{< relref path="/guides/models/track/project-page.md" lang="ja" >}}) 内の Artifacts の場合、Automation の実行を次のように設定できます。
- 新しいバージョンが Artifacts に追加されたとき。たとえば、Dataset Artifacts の新しいバージョンが特定のコレクションに追加されたときに、Training ジョブを開始します。
- エイリアスが Artifacts バージョンに追加されたとき。たとえば、エイリアス「redaction」が Dataset Artifacts に追加されたときに、PII 編集ワークフローをトリガーします。

詳細については、[オートメーションイベントとスコープ]({{< relref path="automation-events.md" lang="ja" >}}) を参照してください。

[オートメーションを作成]({{< relref path="create-automations/" lang="ja" >}}) するには、次の手順を実行します。

1. 必要に応じて、アクセストークン、パスワード、または機密性の高い設定の詳細など、Automation で必要な機密文字列の [secrets]({{< relref path="/guides/core/secrets.md" lang="ja" >}}) を設定します。Secrets は、**Team Settings** で定義されます。Secrets は、webhook Automation で最も一般的に使用され、認証情報またはトークンをプレーンテキストで公開したり、webhook のペイロードにハードコーディングしたりすることなく、webhook の外部サービスに安全に渡すために使用されます。
1. W&B が Slack に投稿したり、ユーザーに代わって webhook を実行したりすることを承認するように、webhook または Slack 通知を設定します。単一の Automation アクション（webhook または Slack 通知）を複数の Automation で使用できます。これらのアクションは、**Team Settings** で定義されます。
1. Project または Registry で、Automation を作成します。
    1. 監視する [event]({{< relref path="#automation-events" lang="ja" >}}) （新しい Artifacts バージョンが追加されたときなど）を定義します。
    1. イベントが発生したときに実行するアクション（Slack チャンネルへの投稿または webhook の実行）を定義します。Webhook の場合は、アクセストークンに使用する secret、および必要に応じてペイロードとともに送信する secret を指定します。

## 次のステップ
- [オートメーションの作成]({{< relref path="create-automations/" lang="ja" >}})。
- [オートメーションイベントとスコープ]({{< relref path="automation-events.md" lang="ja" >}}) について学びます。
- [secret の作成]({{< relref path="/guides/core/secrets.md" lang="ja" >}})。
