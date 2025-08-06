---
title: オートメーション
menu:
  default:
    identifier: automations
    parent: core
weight: 4
---

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

このページでは、W&B の _automations_ について説明します。[Automation を作成]({{< relref "create-automations/" >}}) すると、W&B 内のイベント（例: [artifact]({{< relref "/guides/core/artifacts" >}}) のアーティファクトバージョン作成や [run metric]({{< relref "/guides/models/track/runs.md" >}}) のしきい値変化時）をトリガーに、ワークフローステップ（モデルの自動テストやデプロイメントなど）を実行できます。

たとえば、新しいバージョンが作成された際に Slack チャンネルへ通知したり、`production` エイリアスがアーティファクトに追加されたときに自動テストの webhook を実行したり、run の `loss` が許容範囲内のときのみバリデーションジョブを開始したりできます。

## 概要
automation は、レジストリやプロジェクト内で特定の [イベント]({{< relref "automation-events.md" >}}) が発生した際に実行できます。

[Registry]({{< relref "/guides/core/registry/">}}) の場合、automation は以下のタイミングで開始できます:
- 新しいアーティファクトバージョンがコレクションにリンクされたとき。例: 新規モデル候補に対してテストやバリデーションワークフローをトリガー。
- エイリアスがアーティファクトバージョンに追加されたとき。例: モデルバージョンにエイリアスが追加された場合にデプロイメントワークフローを実行。

[project]({{< relref "/guides/models/track/project-page.md" >}}) の場合、automation は以下のタイミングで開始できます:
- アーティファクトに新しいバージョンが追加されたとき。例: 特定のコレクションにデータセットアーティファクトの新バージョンが追加された際にトレーニングジョブを実行。
- エイリアスがアーティファクトバージョンに追加されたとき。例: データセットアーティファクトに「redaction」というエイリアスが付与されたときに PII マスキングワークフローをトリガー。
- タグがアーティファクトバージョンに追加されたとき。例: アーティファクトバージョンに「europe」タグが追加された際に、地域別ワークフローを起動。
- run のメトリクスが設定したしきい値を満たす、または超えたとき。
- run のメトリクスが設定したしきい値分変化したとき。
- run のステータスが **Running**、**Failed**、**Finished** に変化したとき。

さらに、ユーザーや run 名で対象の run をフィルタすることもできます。

詳細については [Automation events and scopes]({{< relref "automation-events.md" >}}) をご覧ください。

[automation を作成する]({{< relref "create-automations/" >}})には、以下の手順を踏みます。

1. 必要であれば、automation で利用する機密文字列（アクセストークンやパスワード、機密設定など）のために [secrets]({{< relref "/guides/core/secrets.md" >}}) を設定します。Secrets は **Team Settings** で管理できます。Secrets は、webhook automation で外部サービスへ安全に認証情報やトークンを渡す場合などによく利用されます（平文やコード内に含めずに安全に連携可能）。
1. Webhook や Slack 通知アクションを設定し、W&B が代理で Slack への投稿や webhook の実行を行えるようにします。Webhook や Slack 通知のアクションは複数の automation で共用できます。これらのアクションも **Team Settings** で管理します。
1. プロジェクトまたはレジストリで automation を作成します:
    1. 監視したい [イベント]({{< relref "#automation-events" >}}) を定義します（例: アーティファクトの新規バージョン追加時など）。
    1. イベント発生時のアクション（Slack への投稿や webhook の実行）を定義します。Webhook の場合、アクセストークン用の secret や、必要に応じてペイロードに含める secret を指定します。

## 制限事項
[Run metric automations]({{< relref "automation-events.md#run-metrics-events">}}) は現在 [W&B Multi-tenant Cloud]({{< relref "/guides/hosting/#wb-multi-tenant-cloud" >}}) のみでサポートされています。

## 次のステップ
- [Automation を作成]({{< relref "create-automations/" >}})
- [Automation events and scopes]({{< relref "automation-events.md" >}}) を詳しく知る
- [Secret を作成]({{< relref "/guides/core/secrets.md" >}})