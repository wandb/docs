---
title: Track user activity with audit logs
menu:
  default:
    identifier: ja-guides-hosting-monitoring-usage-audit-logging
    parent: monitoring-and-usage
weight: 1
---

W&B の監査ログを使用すると、組織内のユーザーアクティビティを追跡し、企業のガバナンス要件に準拠できます。監査ログは JSON 形式で利用できます。[監査ログスキーマ]({{< relref path="#audit-log-schema" lang="ja" >}})を参照してください。

監査ログへのアクセス方法は、W&B プラットフォームのデプロイメントタイプによって異なります。

| W&B Platform Deployment type | 監査ログアクセス方法 |
|----------------------------|--------------------------------|
| [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) | インスタンスレベルのバケットに10分ごとに同期されます。また、[API]({{< relref path="#fetch-audit-logs-using-api" lang="ja" >}})を使用しても利用できます。 |
| [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) with [secure storage connector (BYOB)]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}}) | インスタンスレベルのバケット (BYOB) に10分ごとに同期されます。また、[API]({{< relref path="#fetch-audit-logs-using-api" lang="ja" >}})を使用しても利用できます。 |
| [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) with W&B managed storage (without BYOB) | [API]({{< relref path="#fetch-audit-logs-using-api" lang="ja" >}})を使用するとのみ利用できます。 |
| [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) | Enterprise プランでのみ利用できます。[API]({{< relref path="#fetch-audit-logs-using-api" lang="ja" >}})を使用するとのみ利用できます。

監査ログを取得した後、[Pandas](https://pandas.pydata.org/docs/index.html)、[Amazon Redshift](https://aws.amazon.com/redshift/)、[Google BigQuery](https://cloud.google.com/bigquery)、または [Microsoft Fabric](https://www.microsoft.com/en-us/microsoft-fabric) などのツールを使用して分析できます。一部の監査ログ分析ツールは JSON をサポートしていません。分析ツールに関するドキュメントを参照して、分析の前に JSON 形式の監査ログを変換するためのガイドラインと要件を確認してください。

{{% alert title="監査ログの保持" %}}
監査ログを特定の期間保持する必要がある場合、W&B は、ストレージバケットまたは Audit Logging API を使用して、ログを長期ストレージに定期的に転送することをお勧めします。

[Health Insurance Portability and Accountability Act of 1996 (HIPAA)](https://www.hhs.gov/hipaa/for-professionals/index.html) の対象となる場合、監査ログは、義務的な保持期間が終了する前に、内部または外部の行為者によって削除または変更できない環境で、最低 6 年間保持する必要があります。HIPAA 準拠の [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) インスタンス ( [BYOB]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}}) ) の場合、長期保持ストレージを含む、管理対象ストレージのガードレールを構成する必要があります。
{{% /alert %}}

## 監査ログスキーマ
この表は、監査ログエントリに表示される可能性のあるすべてのキーをアルファベット順に示しています。アクションと状況に応じて、特定ログエントリには、可能なフィールドのサブセットのみが含まれる場合があります。

| キー | 定義 |
|---------| -------|
|`action`                  | イベントの [アクション]({{< relref path="#actions" lang="ja" >}})。
|`actor_email`             | アクションを開始したユーザーのメールアドレス (該当する場合)。
|`actor_ip`                | アクションを開始したユーザーの IP アドレス。
|`actor_user_id`           | アクションを実行したログインユーザーの ID (該当する場合)。
|`artifact_asset`          | アクションに関連付けられた Artifact ID (該当する場合)。
|`artifact_digest`         | アクションに関連付けられた Artifact ダイジェスト (該当する場合)。
|`artifact_qualified_name` | アクションに関連付けられた Artifact の完全名 (該当する場合)。
|`artifact_sequence_asset` | アクションに関連付けられた Artifact シーケンス ID (該当する場合)。
|`cli_version`             | アクションを開始した Python SDK のバージョン (該当する場合)。
|`entity_asset`            | アクションに関連付けられた Entity または Team ID (該当する場合)。
|`entity_name`             | Entity または Team 名 (該当する場合)。
|`project_asset`           | アクションに関連付けられた Project (該当する場合)。
|`project_name`            | アクションに関連付けられた Project の名前 (該当する場合)。
|`report_asset`            | アクションに関連付けられた Report ID (該当する場合)。
|`report_name`             | アクションに関連付けられた Report の名前 (該当する場合)。
|`response_code`           | アクションの HTTP レスポンスコード (該当する場合)。
|`timestamp`               | [RFC3339 形式](https://www.rfc-editor.org/rfc/rfc3339)でのイベントの時刻。たとえば、`2023-01-23T12:34:56Z` は、2023 年 1 月 23 日の 12:34:56 UTC を表します。
|`user_asset`              | アクションが影響を与える User アセット (アクションを実行するユーザーではなく) (該当する場合)。
|`user_email`              | アクションが影響を与える User のメールアドレス (アクションを実行するユーザーのメールアドレスではなく) (該当する場合)。

### 個人情報 (PII)

メールアドレスや Projects、Teams、Reports の名前などの個人情報 (PII) は、API エンドポイントオプションでのみ利用できます。
- [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) および [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) の場合、組織管理者は、監査ログの取得時に [PII を除外]({{< relref path="#exclude-pii" lang="ja" >}}) できます。
- [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) の場合、API エンドポイントは常に、PII を含む監査ログの関連フィールドを返します。これは構成できません。

## 監査ログの取得
組織またはインスタンス管理者は、エンドポイント `audit_logs/` で Audit Logging API を使用して、W&B インスタンスの監査ログを取得できます。

{{% alert %}}
- 管理者以外のユーザーが監査ログを取得しようとすると、HTTP `403` エラーが発生し、アクセスが拒否されたことを示します。

- 複数の Enterprise [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) 組織の管理者である場合は、監査ログ API リクエストの送信先となる組織を構成する必要があります。プロファイル画像をクリックし、[User Settings] をクリックします。設定の名前は [Default API organization] です。
{{% /alert %}}

1. インスタンスの正しい API エンドポイントを決定します。

    - [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}): `<wandb-platform-url>/admin/audit_logs`
    - [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}): `<wandb-platform-url>/admin/audit_logs`
    - [SaaS Cloud (Enterprise required)]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}): `https://api.wandb.ai/audit_logs`

    以下の手順では、`<API-endpoint>` を API エンドポイントに置き換えます。
2. ベースエンドポイントから完全な API エンドポイントを作成し、オプションで URL パラメータを含めます。
    - `anonymize`: `true` に設定すると、PII が削除されます。デフォルトは `false` です。[監査ログの取得時に PII を除外]({{< relref path="#exclude-pii" lang="ja" >}}) を参照してください。SaaS Cloud ではサポートされていません。
    - `numDays`: ログは `today - numdays` から最新のものまで取得されます。デフォルトは `0` で、`today` のログのみが返されます。SaaS Cloud の場合、過去最大 7 日間の監査ログを取得できます。
    - `startDate`: オプションの日付で、形式は `YYYY-MM-DD` です。[SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) でのみサポートされています。

      `startDate` と `numDays` は相互に作用します。
        - `startDate` と `numDays` の両方を設定した場合、ログは `startDate` から `startDate` + `numDays` まで返されます。
        - `startDate` を省略して `numDays` を含めた場合、ログは `today` から `numDays` まで返されます。
        - `startDate` も `numDays` も設定しない場合、ログは `today` に対してのみ返されます。

3. Web ブラウザーまたは [Postman](https://www.postman.com/downloads/)、[HTTPie](https://httpie.io/)、cURL などのツールを使用して、構築された完全修飾 API エンドポイントに対して HTTP `GET` リクエストを実行します。

API レスポンスには、改行で区切られた JSON オブジェクトが含まれています。オブジェクトには、監査ログがインスタンスレベルのバケットに同期される場合と同様に、[スキーマ]({{< relref path="#audit-log-schemag" lang="ja" >}}) で説明されているフィールドが含まれます。これらの場合、監査ログはバケットの `/wandb-audit-logs` ディレクトリーにあります。

### 基本認証の使用
API キーで基本認証を使用して監査ログ API にアクセスするには、HTTP リクエストの `Authorization` ヘッダーを文字列 `Basic` の後にスペース、次に `username:API-KEY` 形式の base-64 エンコードされた文字列に設定します。つまり、ユーザー名と API キーを `:` 文字で区切られた値に置き換え、その結果を base-64 エンコードします。たとえば、`demo:p@55w0rd` として認証する場合、ヘッダーは `Authorization: Basic ZGVtbzpwQDU1dzByZA==` にする必要があります。

### 監査ログの取得時に PII を除外 {#exclude-pii}
[Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) および [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) の場合、W&B 組織またはインスタンス管理者は、監査ログの取得時に PII を除外できます。[SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) の場合、API エンドポイントは常に、PII を含む監査ログの関連フィールドを返します。これは構成できません。

PII を除外するには、`anonymize=true` URL パラメータを渡します。たとえば、W&B インスタンスの URL が `https://mycompany.wandb.io` で、過去 1 週間のユーザーアクティビティの監査ログを取得し、PII を除外する場合は、次のような API エンドポイントを使用します。

```text
https://mycompany.wandb.io/admin/audit_logs?numDays=7&anonymize=true.
```

## アクション
次の表は、W&B によって記録できるアクションをアルファベット順に説明しています。

|アクション | 定義 |
|-----|-----|
| `artifact:create`             | Artifact が作成されました。
| `artifact:delete   `          | Artifact が削除されました。
| `artifact:read`               | Artifact が読み取られました。
| `project:delete`              | Project が削除されました。
| `project:read`                | Project が読み取られました。
| `report:read`                 | Report が読み取られました。<sup><a href="#1">1</a></sup>
| `run:delete_many`             | Run のバッチが削除されました。
| `run:delete`                  | Run が削除されました。
| `run:stop`                    | Run が停止されました。
| `run:undelete_many`           | Run のバッチがゴミ箱から復元されました。
| `run:update_many`             | Run のバッチが更新されました。
| `run:update`                  | Run が更新されました。
| `sweep:create_agent`          | sweep agent が作成されました。
| `team:create_service_account` | サービスアカウントが Team に対して作成されました。
| `team:create`                 | Team が作成されました。
| `team:delete`                 | Team が削除されました。
| `team:invite_user`            | User が Team に招待されました。
| `team:uninvite`               | User またはサービスアカウントが Team から招待解除されました。
| `user:create_api_key`         | User の APIキー が作成されました。<sup><a href="#1">1</a></sup>
| `user:create`                 | User が作成されました。<sup><a href="#1">1</a></sup>
| `user:deactivate`             | User が非アクティブ化されました。<sup><a href="#1">1</a></sup>
| `user:delete_api_key`         | User の APIキー が削除されました。<sup><a href="#1">1</a></sup>
| `user:initiate_login`         | User がログインを開始しました。<sup><a href="#1">1</a></sup>
| `user:login`                  | User がログインしました。<sup><a href="#1">1</a></sup>
| `user:logout`                 | User がログアウトしました。<sup><a href="#1">1</a></sup>
| `user:permanently_delete`     | User が完全に削除されました。<sup><a href="#1">1</a></sup>
| `user:reactivate`             | User が再アクティブ化されました。<sup><a href="#1">1</a></sup>
| `user:read`                   | User プロファイルが読み取られました。<sup><a href="#1">1</a></sup>
| `user:update`                 | User が更新されました。<sup><a href="#1">1</a></sup>

<a id="1">1</a>: [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) では、次の監査ログは収集されません。
- オープンまたはパブリック Projects。
- `report:read` アクション。
- 特定の組織に関連付けられていない `User` アクション。
