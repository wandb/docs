---
title: Automation
menu:
  default:
    identifier: ja-guides-core-automations-_index
    parent: core
weight: 4
---

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

このページでは、W&B における _Automation_ について説明します。[Automationを作成する]({{< relref path="create-automations/" lang="ja" >}})ことで、モデルの自動テストやデプロイメントなどのワークフローステップを、W&B 上でのイベント（例えば [artifact]({{< relref path="/guides/core/artifacts" lang="ja" >}}) のバージョン作成や [run metric]({{< relref path="/guides/models/track/runs.md" lang="ja" >}}) の値が指定したしきい値を満たした場合または変化した場合など）に基づいてトリガーできます。

例えば、Automationを使えば、新しいバージョンが作成された際に Slack チャンネルへ通知したり、`production` エイリアスが artifact に追加されたタイミングで自動テスト用の webhook を発火したり、run の `loss` が許容範囲にある場合のみバリデーションジョブを開始したりできます。

## 概要
Automationは、レジストリやプロジェクト内で特定の [イベント]({{< relref path="automation-events.md" lang="ja" >}}) が発生した時に開始できます。

[Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) では、次のタイミングでAutomationを開始できます：
- 新しい artifact バージョンがコレクションにリンクされたとき。例：新しい候補モデルに対してテストやバリデーションのワークフローをトリガー。
- エイリアスが artifact バージョンに追加されたとき。例：モデルバージョンにエイリアスが追加されたタイミングでデプロイメント用ワークフローを実行。

[project]({{< relref path="/guides/models/track/project-page.md" lang="ja" >}}) では、次のタイミングでAutomationを開始できます：
- 新しいバージョンが artifact に追加された時。例：特定のコレクションにデータセット artifact の新バージョンが追加されたタイミングでトレーニングジョブを開始。
- エイリアスが artifact バージョンに追加されたとき。例："redaction" エイリアスがデータセット artifact に付与された時、PII（個人情報）マスキングワークフローを実行。
- artifact バージョンにタグが追加されたとき。例："europe" タグが artifact バージョンに追加された時、地域別のワークフローを実行。
- run のメトリクスが設定したしきい値を満たした、または超えたとき。
- run のメトリクスが設定したしきい値分変化したとき。
- run のステータスが **Running**、**Failed**、**Finished** のいずれかに変化したとき。

オプションで、ユーザーや run 名で run を絞り込むことも可能です。

詳細については、[Automation のイベントと範囲]({{< relref path="automation-events.md" lang="ja" >}})をご覧ください。

[Automationを作成する]({{< relref path="create-automations/" lang="ja" >}})には、以下のステップを行います：

1. 必要に応じて、Automationで利用する認証トークンやパスワードなどの機密文字列用に [secrets]({{< relref path="/guides/core/secrets.md" lang="ja" >}}) を設定します。Secrets は **Team Settings** で定義します。Secrets は主に webhook Automationで、外部サービスへ安全に認証情報やトークンを渡す用途に使われます（プレーンテキストやコード内への埋め込みなしで安全に管理）。
1. webhook や Slack 通知の設定を行い、W&B が Slack への投稿や webhook の実行権限を持てるようにします。これらのアクション（webhook や Slack 通知）は複数のAutomationから共通利用できます。アクションも **Team Settings** で定義します。
1. プロジェクトまたはレジストリ内で、Automationを作成します：
    1. 監視する [イベント]({{< relref path="#automation-events" lang="ja" >}}) を定義します（例：新しい artifact バージョン追加時）。
    1. イベント発生時のアクションを指定します（Slack に通知、webhook の実行など）。webhook を使う場合には、アクセス用トークンやペイロード送信用に利用する secret も必要に応じて指定します。

## 制限事項
[Run metric automations]({{< relref path="automation-events.md#run-metrics-events" lang="ja" >}}) は現在 [W&B Multi-tenant Cloud]({{< relref path="/guides/hosting/#wb-multi-tenant-cloud" lang="ja" >}}) のみ対応しています。

## 次のステップ
- [Automationを作成する]({{< relref path="create-automations/" lang="ja" >}})。
- [Automation のイベントと範囲について学ぶ]({{< relref path="automation-events.md" lang="ja" >}})。
- [Secret を作成する]({{< relref path="/guides/core/secrets.md" lang="ja" >}})。