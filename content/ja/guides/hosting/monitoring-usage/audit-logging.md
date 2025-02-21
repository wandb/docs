---
title: Track user activity with audit logs
menu:
  default:
    identifier: ja-guides-hosting-monitoring-usage-audit-logging
    parent: monitoring-and-usage
weight: 1
---

W&B の監査 ログを使用して、組織内の ユーザー のアクティビティを追跡し、エンタープライズ ガバナンスの要件に準拠します。監査 ログ は JSON 形式で利用できます。監査 ログ へのアクセス 方法は、W&B プラットフォーム のデプロイメント タイプによって異なります。

| W&B Platform Deployment type | 監査 ログ のアクセス メカニズム |
|----------------------------|--------------------------------|
| [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) | インスタンスレベル の bucket に10分ごとに同期されます。また、[API]({{< relref path="#fetch-audit-logs-using-api" lang="ja" >}}) を使用して利用することもできます。 |
| [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) with [secure storage connector (BYOB)]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}}) | インスタンスレベル の bucket (BYOB) に10分ごとに同期されます。また、[API]({{< relref path="#fetch-audit-logs-using-api" lang="ja" >}}) を使用して利用することもできます。 |
| [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) with W&B managed storage (without BYOB) | [API]({{< relref path="#fetch-audit-logs-using-api" lang="ja" >}}) を使用してのみ利用可能です。 |

{{% alert %}}
監査 ログ は、[SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) では利用できません。
{{% /alert %}}

監査 ログ にアクセスしたら、[Pandas](https://pandas.pydata.org/docs/index.html), [Amazon Redshift](https://aws.amazon.com/redshift/), [Google BigQuery](https://cloud.google.com/bigquery), [Microsoft Fabric](https://www.microsoft.com/en-us/microsoft-fabric) などの好みの ツール を使用して分析します。分析を行う前に、JSON 形式の監査 ログ を ツール に関連する形式に変換する必要がある場合があります。特定の ツール 用に監査 ログ を変換する方法に関する情報は、W&B ドキュメント の範囲外です。

{{% alert %}}
**監査 ログ の保持:** 組織内のコンプライアンス、セキュリティ、またはリスク チーム が特定の期間監査 ログ を保持する必要がある場合、W&B は、インスタンスレベル の bucket から長期保持 ストレージ に定期的に ログ を転送することをお勧めします。代わりに API を使用して監査 ログ にアクセスする場合は、最後の スクリプト の 実行 時から生成された可能性のある ログ を取得し、分析のために 短期 ストレージ に保存するか、長期保持 ストレージ に直接転送するために、定期的に (毎日または数日ごとなど) 実行される簡単な スクリプト を実装できます。
{{% /alert %}}

HIPAA コンプライアンス には、監査 ログ を最低6年間保持する必要があります。HIPAA 準拠の [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) インスタンス と [BYOB]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}}) では、長期保持 ストレージ を含むマネージド ストレージ の ガードレール を構成して、内部または外部の ユーザー が必須保持期間の終了前に監査 ログ を削除できないようにする必要があります。

## 監査 ログ スキーマ
次の 表 に、監査 ログ に存在する可能性のあるすべての異なる キー を示します。各 ログ には、対応する アクション に関連する アセット のみが含まれており、他の アセット は ログ から除外されます。

| Key | Definition |
|---------| -------|
|timestamp               | [RFC3339 format](https://www.rfc-editor.org/rfc/rfc3339) の タイム スタンプ 。例: `2023-01-23T12:34:56Z` は、2023年1月23日の `12:34:56 UTC` 時間を表します。
|action                  | ユーザー が実行した [アクション]({{< relref path="#actions" lang="ja" >}})。
|actor_user_id           | 存在する場合、 アクション を実行した ログイン ユーザー の ID。
|response_code           | アクション の HTTP レスポンス コード。
|artifact_asset          | 存在する場合、 アクション はこの Artifact ID に対して実行されました。
|artifact_sequence_asset | 存在する場合、 アクション はこの Artifact Sequence ID に対して実行されました。
|entity_asset            | 存在する場合、 アクション はこの Entity または Team ID に対して実行されました。
|project_asset           | 存在する場合、 アクション はこの Project ID に対して実行されました。
|report_asset            | 存在する場合、 アクション はこの Report ID に対して実行されました。
|user_asset              | 存在する場合、 アクション はこの ユーザー アセット に対して実行されました。
|cli_version             | アクション が Python SDK 経由で実行された場合、これには バージョン が含まれます。
|actor_ip                | ログイン ユーザー の IP アドレス 。
|actor_email             | 存在する場合、 アクション はこの アクター の メール に対して実行されました。
|artifact_digest         | 存在する場合、 アクション はこの Artifact ダイジェスト に対して実行されました。
|artifact_qualified_name | 存在する場合、 アクション はこの Artifact に対して実行されました。
|entity_name             | 存在する場合、 アクション はこの Entity または Team 名 に対して実行されました。
|project_name            | 存在する場合、 アクション はこの Project 名 に対して実行されました。
|report_name             | 存在する場合、 アクション はこの Report 名 に対して実行されました。
|user_email              | 存在する場合、 アクション はこの ユーザー メール に対して実行されました。

メール ID や Projects、Teams、Reports の名前などの個人を特定できる情報 (PII) は、API エンドポイント オプション を使用してのみ利用でき、[下記の説明]({{< relref path="#fetch-audit-logs-using-api" lang="ja" >}}) に従ってオフにすることができます。

## API を使用して監査 ログ を取得する
インスタンス 管理者 は、次の API を使用して W&B インスタンス の監査 ログ を取得できます。
1. ベース エンドポイント `<wandb-platform-url>/admin/audit_logs` と次の URL パラメータ を組み合わせて、完全な API エンドポイント を構築します。
    - `numDays`: ログ は `今日 - numdays` から最新のものまで取得されます。デフォルト は `0` で、`今日` の ログ のみ返します。
    - `anonymize`: `true` に設定すると、PII を削除します。デフォルト は `false` です。
2. 構築された完全な API エンドポイント で HTTP GET リクエスト を実行します。最新の ブラウザ 内で直接実行するか、[Postman](https://www.postman.com/downloads/)、[HTTPie](https://httpie.io/)、cURL コマンド などの ツール を使用します。

組織またはインスタンス 管理者 は、 APIキー を使用して基本的な認証を行い、監査 ログ API にアクセスできます。HTTP リクエスト の `Authorization` ヘッダー を 文字列 `Basic` に設定し、その後にスペース、そして `username:API-KEY` 形式の base-64 エンコードされた 文字列 を続けます。つまり、 ユーザー 名と APIキー を `:` 文字 で区切って置き換え、その 結果 を base-64 でエンコードします。たとえば、`demo:p@55w0rd` として認証するには、ヘッダー は `Authorization: Basic ZGVtbzpwQDU1dzByZA==` にする必要があります。

W&B インスタンス の URL が `https://mycompany.wandb.io` であり、過去1週間の ユーザー アクティビティ の PII なしで監査 ログ を取得する場合は、API エンドポイント `https://mycompany.wandb.io/admin/audit_logs?numDays=7&anonymize=true` を使用する必要があります。

{{% alert %}}
W&B [インスタンス 管理者]({{< relref path="/guides/hosting/iam/access-management/" lang="ja" >}}) のみが API を使用して監査 ログ を取得できます。インスタンス 管理者 でない場合、または組織に ログイン していない場合は、`HTTP 403 Forbidden` エラー が表示されます。
{{% /alert %}}

API レスポンス には、改行で区切られた JSON オブジェクト が含まれます。オブジェクト には、スキーマ で説明されている フィールド が含まれます。これは、監査 ログ ファイル をインスタンスレベル の bucket に同期する際に使用されるものと同じ形式です (前述のように該当する場合)。これらの場合、監査 ログ は bucket の `/wandb-audit-logs` ディレクトリー にあります。

## アクション
次の 表 に、W&B によって記録できる可能性のある アクション について説明します。

|Action | Definition |
|-----|-----|
| artifact:create             | Artifact が作成されました。
| artifact:delete             | Artifact が削除されました。
| artifact:read               | Artifact が読み取られました。
| project:delete              | Project が削除されました。
| project:read                | Project が読み取られました。
| report:read                 | Report が読み取られました。
| run:delete                  | Run が削除されました。
| run:delete_many             | 複数の Runs がバッチ で削除されました。
| run:update_many             | 複数の Runs がバッチ で更新されました。
| run:stop                    | Run が停止しました。
| run:undelete_many           | 複数の Runs が ゴミ箱 からバッチ で戻されました。
| run:update                  | Run が更新されました。
| sweep:create_agent          | Sweep agent が作成されました。
| team:invite_user            | ユーザー が Team に招待されました。
| team:create_service_account | サービス アカウント が Team に作成されました。
| team:create                 | Team が作成されました。
| team:uninvite               | ユーザー または サービス アカウント が Team から招待解除されました。
| team:delete                 | Team が削除されました。
| user:create                 | ユーザー が作成されました。
| user:delete_api_key         | ユーザー の APIキー が削除されました。
| user:deactivate             | ユーザー が非アクティブ化されました。
| user:create_api_key         | ユーザー の APIキー が作成されました。
| user:permanently_delete     | ユーザー が完全に削除されました。
| user:reactivate             | ユーザー が再アクティブ化されました。
| user:update                  | ユーザー が更新されました。
| user:read                   | ユーザー プロファイル が読み取られました。
| user:login                  | ユーザー が ログイン しました。
| user:initiate_login         | ユーザー が ログイン を開始しました。
| user:logout                 | ユーザー が ログアウト しました。
