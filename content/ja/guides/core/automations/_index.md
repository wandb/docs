---
title: オートメーション
aliases:
- /guides/core/automations/
cascade:
- url: guides/automations/:filename
menu:
  default:
    identifier: ja-guides-core-automations-_index
    parent: core
url: guides/automations
weight: 4
---

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

このページでは、W&B の _オートメーション_ について説明します。[オートメーションを作成する]({{< relref path="create-automations/" lang="ja" >}}) と、Artifact バージョン が作成されたときや、run のメトリクスが特定のしきい値に達したとき、または変化したときなど、W&B でのイベントに基づいて、自動モデル テストやデプロイメントなどのワークフロー ステップをトリガーします。

たとえば、オートメーションは、新しい バージョン が作成されたときに Slack チャンネルに通知したり、`production` エイリアスが Artifact に追加されたときに自動テスト webhook をトリガーしたり、run の `loss` が許容範囲内にある場合にのみ検証ジョブを開始したりできます。

## 概要
オートメーションは、Registry または Projects で特定の [イベント]({{< relref path="automation-events.md" lang="ja" >}}) が発生したときに開始できます。

[Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) では、オートメーションは次の場合に開始できます。
- 新しい Artifact バージョン がコレクションにリンクされたとき。たとえば、新しい候補モデルのテストおよび検証ワークフローをトリガーします。
- Artifact バージョン に エイリアス が追加されたとき。たとえば、モデル バージョン に エイリアス が追加されたときにデプロイメント ワークフローをトリガーします。

[Projects]({{< relref path="/guides/models/track/project-page.md" lang="ja" >}}) では、オートメーションは次の場合に開始できます。
- 新しい バージョン が Artifact に追加されたとき。たとえば、特定のコレクションに Dataset Artifact の新しい バージョン が追加されたときに、トレーニング ジョブを開始します。
- Artifact バージョン に エイリアス が追加されたとき。たとえば、Dataset Artifact に "redaction" エイリアスが追加されたときに PII 削除ワークフローをトリガーします。
- Artifact バージョン にタグが追加されたとき。たとえば、Artifact バージョン に "europe" タグが追加されたときに地域固有のワークフローをトリガーします。
- run のメトリクスが設定されたしきい値に達するか、超えたとき。
- run のメトリクスが設定されたしきい値によって変化したとき。
- run のステータスが **Running**、**Failed**、または **Finished** に変更されたとき。

オプションで、ユーザーまたは run 名で run をフィルタリングできます。

詳細については、[オートメーションのイベントとスコープ]({{< relref path="automation-events.md" lang="ja" >}}) を参照してください。

[オートメーションを作成する]({{< relref path="create-automations/" lang="ja" >}}) には、次の手順を実行します。

1. 必要に応じて、アクセストークン、パスワード、機密の設定の詳細など、オートメーションで必要となる機密文字列の [シークレット]({{< relref path="/guides/core/secrets.md" lang="ja" >}}) を設定します。シークレットは **Team Settings** で定義されます。シークレットは、webhook オートメーションで最も一般的に使用され、平文で公開したり、webhook のペイロードにハードコーディングしたりすることなく、認証情報やトークンを webhook の外部サービスに安全に渡します。
2. W&B が Slack に投稿したり、ユーザーに代わって webhook を実行したりすることを承認するように、webhook または Slack 通知を設定します。1 つのオートメーション アクション (webhook または Slack 通知) は、複数のオートメーションで使用できます。これらのアクションは **Team Settings** で定義されます。
3. Projects または Registry でオートメーションを作成します。
    1. 新しい Artifact バージョン が追加されたときなど、監視する [イベント]({{< relref path="#automation-events" lang="ja" >}}) を定義します。
    2. イベントが発生したときに実行するアクション (Slack チャンネルへの投稿または webhook の実行) を定義します。webhook の場合は、必要に応じて、アクセストークンに使用するシークレット、および/またはペイロードで送信するシークレットを指定します。

## 制限事項
[run メトリクスのオートメーション]({{< relref path="automation-events.md#run-metrics-events" lang="ja" >}}) は現在、[W&B マルチテナント クラウド]({{< relref path="/guides/hosting/#wb-multi-tenant-cloud" lang="ja" >}}) でのみサポートされています。

## 次のステップ
- [オートメーションを作成する]({{< relref path="create-automations/" lang="ja" >}})。
- [オートメーションのイベントとスコープ]({{< relref path="automation-events.md" lang="ja" >}}) について学習する。
- [シークレットを作成する]({{< relref path="/guides/core/secrets.md" lang="ja" >}})。