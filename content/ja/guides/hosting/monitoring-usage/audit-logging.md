---
title: Track user activity with audit logs
menu:
  default:
    identifier: ja-guides-hosting-monitoring-usage-audit-logging
    parent: monitoring-and-usage
weight: 1
---

W&B 監査ログを使用して、組織内のユーザー活動を追跡し、企業ガバナンス要件に準拠します。監査ログは JSON 形式で利用可能です。監査ログへのアクセス方法は、W&B プラットフォームのデプロイメントタイプによって異なります。

| W&B プラットフォームのデプロイメントタイプ | 監査ログへのアクセス方法 |
|----------------------------|--------------------------------|
| [セルフ管理]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) | インスタンスレベルのバケットに10分ごとに同期されます。また、[API]({{< relref path="#fetch-audit-logs-using-api" lang="ja" >}})を使用しても利用可能です。 |
| [専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) と [セキュアストレージコネクタ (BYOB)]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}}) | インスタンスレベルのバケット (BYOB) に10分ごとに同期されます。また、[API]({{< relref path="#fetch-audit-logs-using-api" lang="ja" >}})を使用しても利用可能です。 |
| [専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) と W&B 管理ストレージ (BYOBなし) | [API]({{< relref path="#fetch-audit-logs-using-api" lang="ja" >}})を使用してのみ利用可能です。 |

{{% alert %}}
監査ログは、[SaaS クラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) では利用できません。
{{% /alert %}}

監査ログにアクセスしたら、好みのツールを使用してそれを分析します。例えば、[Pandas](https://pandas.pydata.org/docs/index.html)、[Amazon Redshift](https://aws.amazon.com/redshift/)、[Google BigQuery](https://cloud.google.com/bigquery)、[Microsoft Fabric](https://www.microsoft.com/en-us/microsoft-fabric) などがあります。分析を行う前に、JSON形式の監査ログをツールに適した形式に変換する必要があるかもしれません。特定のツール向けに監査ログを変換する方法についての情報は、W&B ドキュメントの範囲外です。

{{% alert %}}
**監査ログの保持:** 組織内のコンプライアンス、セキュリティ、またはリスクチームが特定の期間監査ログを保持することを要求する場合、W&B は、インスタンスレベルのバケットからログを定期的に長期保管ストレージに転送することを推奨します。また、監査ログに API を使用する場合、簡単なスクリプトを定期的に (毎日、または数日ごと) 実行して、最後のスクリプト実行以降に生成された可能性のあるログを取得し、それらを短期間のストレージに保存し、分析のためまたは直接長期保管ストレージに転送することができます。
{{% /alert %}}

HIPAA コンプライアンスでは、監査ログを最低6年間保持する必要があります。HIPAAに準拠した [専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) インスタンスで [BYOB]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}}) を使用している場合、管理されたストレージには、内部または外部のユーザーが必須の保持期間が終了する前に監査ログを削除できないようにするガードレールを設定する必要があります。

## 監査ログスキーマ
以下の表は、監査ログに存在する可能性があるさまざまなキーをリストしています。各ログには、対応するアクションに関連する資産のみが含まれ、他はログから省略されます。

| キー | 定義 |
|---------| -------|
|timestamp               | [RFC3339形式](https://www.rfc-editor.org/rfc/rfc3339)のタイムスタンプ。例: `2023-01-23T12:34:56Z` は、2023年1月23日 `12:34:56 UTC` の時間を表します。
|action                  | [ユーザーが取ったアクション]({{< relref path="#actions" lang="ja" >}})。
|actor_user_id           | 存在する場合、アクションを実行したログインユーザーのID。
|response_code           | アクションのHTTPレスポンスコード。
|artifact_asset          | 存在する場合、このアーティファクトIDに対してアクションが実行された。
|artifact_sequence_asset | 存在する場合、このアーティファクトシーケンスIDに対してアクションが実行された。
|entity_asset            | 存在する場合、このエンティティまたはチームIDに対してアクションが実行された。
|project_asset           | 存在する場合、このプロジェクトIDに対してアクションが実行された。
|report_asset            | 存在する場合、このレポートIDに対してアクションが実行された。
|user_asset              | 存在する場合、このユーザーアセットに対してアクションが実行された。
|cli_version             | Python SDK 経由でアクションが実行された場合、バージョンが含まれます。
|actor_ip                | ログインユーザーのIPアドレス。
|actor_email             | 存在する場合、このアクターのメールでアクションが実行された。
|artifact_digest         | 存在する場合、このアーティファクトダイジェストに対してアクションが実行された。
|artifact_qualified_name | 存在する場合、このアーティファクトに対してアクションが実行された。
|entity_name             | 存在する場合、このエンティティまたはチーム名に対してアクションが実行された。
|project_name            | 存在する場合、このプロジェクト名に対してアクションが実行された。
|report_name             | 存在する場合、このレポート名に対してアクションが実行された。
|user_email              | 存在する場合、このユーザーメールに対してアクションが実行された。

メールIDやプロジェクト、チーム、レポートの名前などの個人を特定できる情報（PII）は、APIエンドポイントオプションを使用した場合に限り利用でき、[以下に説明されているように]({{< relref path="#fetch-audit-logs-using-api" lang="ja" >}})オフにすることができます。

## API を使用した監査ログの取得
インスタンス管理者は次の API を使用して W&B インスタンスの監査ログを取得できます:
1. 基本エンドポイント `<wandb-platform-url>/admin/audit_logs` と次の URL パラメータの組み合わせを使用して完全な API エンドポイントを構築します:
   - `numDays`: `today - numdays` から最新までのログが取得されます。デフォルトは `0` で、`today` のみのログを返します。
   - `anonymize`: `true` に設定すると、PIIが削除されます。デフォルトは `false` です。
2. 現代のブラウザで直接実行するか、[Postman](https://www.postman.com/downloads/)、[HTTPie](https://httpie.io/)、cURL コマンドなどのツールを使用して、構築した完全な API エンドポイントへの HTTP GET リクエストを実行します。

組織またはインスタンス管理者は、APIキーを使用した基本認証を使用して監査ログ API にアクセスできます。HTTPリクエストの `Authorization` ヘッダーを文字列 `Basic` その後にスペースと、形式 `username:API-KEY` の base-64 エンコードされた文字列に設定します。つまり、ユーザー名と API キーを `:` 文字で区切って、それを base-64 エンコードします。たとえば、`demo:p@55w0rd`として認証するには、ヘッダーは `Authorization: Basic ZGVtbzpwQDU1dzByZA==` でなければなりません。

もし W&B インスタンスの URL が `https://mycompany.wandb.io` で、過去1週間のユーザー活動に関するPIIがない監査ログを取得したい場合、APIエンドポイント `https://mycompany.wandb.io/admin/audit_logs?numDays=7&anonymize=true` を使用しなければなりません。

{{% alert %}}
W&Bの[インスタンス管理者]({{< relref path="/guides/hosting/iam/access-management/" lang="ja" >}})のみがAPIを使用して監査ログを取得できます。もしインスタンス管理者でないか、組織にログインしていない場合、`HTTP 403 Forbidden` エラーが発生します。
{{% /alert %}}

APIレスポンスには、JSONオブジェクトが改行で区切られて含まれています。オブジェクトにはスキーマで説明されているフィールドが含まれます。このフォーマットは、監査ログファイルをインスタンスレベルのバケットに同期する際に使用されるフォーマットと同じです（ここで前述した通りの場合に適用されます）。その場合、監査ログはバケットの `/wandb-audit-logs` ディレクトリーに配置されます。

## アクション
以下の表は、W&Bによって記録される可能性のあるアクションを説明しています。

| アクション | 定義 |
|-----|-----|
| artifact:create             | アーティファクトが作成されます。
| artifact:delete             | アーティファクトが削除されます。
| artifact:read               | アーティファクトが読まれます。
| project:delete              | プロジェクトが削除されます。
| project:read                | プロジェクトが読まれます。
| report:read                 | レポートが読まれます。
| run:delete                  | Run が削除されます。
| run:delete_many             | Runs がバッチで削除されます。
| run:update_many             | Runs がバッチで更新されます。
| run:stop                    | Run が停止されます。
| run:undelete_many           | Runs がバッチでゴミ箱から復元されます。
| run:update                  | Run が更新されます。
| sweep:create_agent          | Sweep agent が作成されます。
| team:invite_user            | ユーザーがチームに招待されます。
| team:create_service_account | チーム用のサービスアカウントが作成されます。
| team:create                 | チームが作成されます。
| team:uninvite               | チームからユーザーまたはサービスアカウントの招待が取り消されます。
| team:delete                 | チームが削除されます。
| user:create                 | ユーザーが作成されます。
| user:delete_api_key         | ユーザーのAPIキーが削除されます。
| user:deactivate             | ユーザーが非アクティブ化されます。
| user:create_api_key         | ユーザーのAPIキーが作成されます。
| user:permanently_delete     | ユーザーが完全に削除されます。
| user:reactivate             | ユーザーが再アクティブ化されます。
| user:update                 | ユーザーが更新されます。
| user:read                   | ユーザープロファイルが読まれます。
| user:login                  | ユーザーがログインします。
| user:initiate_login         | ユーザーがログインを開始します。
| user:logout                 | ユーザーがログアウトします。