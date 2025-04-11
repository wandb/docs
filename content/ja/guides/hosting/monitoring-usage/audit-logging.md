---
title: ユーザーのアクティビティを監査ログで追跡する
menu:
  default:
    identifier: ja-guides-hosting-monitoring-usage-audit-logging
    parent: monitoring-and-usage
weight: 1
---

W&B の監査ログを使用して、組織内のユーザー活動を追跡し、企業のガバナンス要件に準拠します。監査ログは JSON フォーマットで利用可能です。[監査ログスキーマ]({{< relref path="#audit-log-schema" lang="ja" >}}) を参照してください。

監査ログへのアクセス方法は、W&B プラットフォームのデプロイメントタイプによって異なります:

| W&B プラットフォームデプロイメントタイプ | 監査ログアクセス機構 |
|----------------------------|--------------------------------|
| [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) | 10 分ごとにインスタンスレベルのバケットに同期されます。また、[API]({{< relref path="#fetch-audit-logs-using-api" lang="ja" >}}) を使用しても利用可能です。 |
| [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) with [secure storage connector (BYOB)]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}}) | インスタンスレベルのバケット (BYOB) に 10 分ごとに同期されます。また、[API]({{< relref path="#fetch-audit-logs-using-api" lang="ja" >}}) を使用しても利用可能です。 |
| [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) with W&B managed storage (without BYOB) | [API]({{< relref path="#fetch-audit-logs-using-api" lang="ja" >}}) を使用してのみ利用可能です。 |
| [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) | エンタープライズプランのみで利用可能です。[API]({{< relref path="#fetch-audit-logs-using-api" lang="ja" >}}) を使用してのみ利用可能です。

監査ログを取得した後、[Pandas](https://pandas.pydata.org/docs/index.html)、[Amazon Redshift](https://aws.amazon.com/redshift/)、[Google BigQuery](https://cloud.google.com/bigquery)、[Microsoft Fabric](https://www.microsoft.com/en-us/microsoft-fabric) などのツールを使用して分析できます。監査ログの分析ツールによっては JSON をサポートしていないものもあります。分析ツールのドキュメントを参照して、分析前に JSON 形式の監査ログを変換するためのガイドラインと要件をご確認ください。

{{% alert title="監査ログの保存" %}}
特定の期間に渡って監査ログを保存する必要がある場合、W&B はログを長期保存場所に定期的に転送することを推奨しています。ストレージバケットや監査ログ API を使用できます。

[Health Insurance Portability and Accountability Act of 1996 (HIPAA)](https://www.hhs.gov/hipaa/for-professionals/index.html) に準拠する必要がある場合、監査ログは最低 6 年間保存され、保存期間が終了するまで内部または外部のいずれのアクターによっても削除または変更できない環境に保存されなければなりません。HIPAA に準拠した [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) インスタンスには、長期保存ストレージを含むマネージドストレージのためのガードレールを設定する必要があります。
{{% /alert %}}

## 監査ログスキーマ
この表は、監査ログエントリに現れる可能性のあるすべてのキーをアルファベット順に示しています。アクションと状況によって、特定のログエントリには、可能なフィールドのサブセットのみが含まれる場合があります。

| キー | 定義 |
|---------| -------|
|`action`                  | イベントの[アクション]({{< relref path="#actions" lang="ja" >}})。
|`actor_email`             | アクションを開始したユーザーのメールアドレス（該当する場合）。
|`actor_ip`                | アクションを開始したユーザーの IP アドレス。
|`actor_user_id`           | アクションを実施したログインユーザーの ID（該当する場合）。
|`artifact_asset`          | アクションに関連するアーティファクト ID（該当する場合）。
|`artifact_digest`         | アクションに関連するアーティファクトダイジェスト（該当する場合）。
|`artifact_qualified_name` | アクションに関連するアーティファクトの完全名（該当する場合）。
|`artifact_sequence_asset` | アクションに関連するアーティファクトシーケンス ID（該当する場合）。
|`cli_version`             | アクションを開始した Python SDK のバージョン（該当する場合）。
|`entity_asset`            | アクションに関連するエンティティまたはチーム ID（該当する場合）。
|`entity_name`             | アクションに関連するエンティティまたはチーム名（該当する場合）。
|`project_asset`           | アクションに関連するプロジェクト（該当する場合）。
|`project_name`            | アクションに関連するプロジェクトの名前（該当する場合）。
|`report_asset`            | アクションに関連するレポート ID（該当する場合）。
|`report_name`             | アクションに関連するレポートの名前（該当する場合）。
|`response_code`           | アクションの HTTP レスポンスコード（該当する場合）。
|`timestamp`               | イベントの時間を [RFC3339 形式](https://www.rfc-editor.org/rfc/rfc3339) で示します。例えば、`2023-01-23T12:34:56Z` は 2023 年 1 月 23 日 12 時 34 分 56 秒 UTC を示します。
|`user_asset`              | アクションが影響を与えるユーザーアセット（アクションを実行するユーザーではなく）（該当する場合）。
|`user_email`              | アクションが影響を与えるユーザーのメールアドレス（アクションを実行するユーザーのメールアドレスではなく）（該当する場合）。

### 個人を特定できる情報 (PII)

個人を特定できる情報 (PII) は API エンドポイントオプションを使用することによってのみ利用可能です。
- [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) および
  [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) では、組織の管理者は監査ログを取得する際に [PII を除外]({{< relref path="#exclude-pii" lang="ja" >}}) できます。
- [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) では、監査ログの関連フィールドを常に API エンドポイントが返します。PII を含むこれらのフィールドは設定変更できません。

## 監査ログの取得
W&B インスタンスの監査ログは、Audit Logging API を使用して、組織またはインスタンス管理者が取得できます。そのエンドポイントは `audit_logs/` です。

{{% alert %}}
- 管理者以外のユーザーが監査ログを取得しようとすると、HTTP `403` エラーが発生し、アクセスが拒否されたことが示されます。

- 複数のエンタープライズ [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) 組織の管理者である場合、監査ログ API リクエストが送信される組織を設定する必要があります。プロフィール画像をクリックし、**ユーザー設定** をクリックしてください。その設定は **デフォルト API 組織** と呼ばれます。
{{% /alert %}}

1. インスタンスに対する正しい API エンドポイントを決定します：

    - [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}): `<wandb-platform-url>/admin/audit_logs`
    - [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}): `<wandb-platform-url>/admin/audit_logs`
    - [SaaS Cloud (エンタープライズ 必須)]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}): `https://api.wandb.ai/audit_logs`

    次のステップで、`<API-endpoint>` を実際の API エンドポイントで置き換えてください。
1. 基本エンドポイントから完全な API エンドポイントを構築し、必要に応じて URL パラメータを含めます：
    - `anonymize`: `true` に設定すると、PII を削除します。デフォルトは `false` です。[監査ログ取得時の PII を除外]({{< relref path="#exclude-pii" lang="ja" >}}) を参照してください。SaaS Cloud ではサポートされていません。
    - `numDays`: `today - numdays` から最新のログまで取得されます。デフォルトは `0` で、今日のログのみを返します。SaaS Cloud では過去最大 7 日分の監査ログを取得できます。
    - `startDate`: オプションの日付、`YYYY-MM-DD` 形式で指定します。 [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) でのみサポートされています。

      `startDate` と `numDays` の相互作用：
        - `startDate` と `numDays` の両方を設定すると、`startDate` から `startDate` + `numDays` の範囲でログが返されます。
        - `startDate` を省略して `numDays` を含めると、`today` から `numDays` までの範囲でログが返されます。
        - `startDate` と `numDays` のどちらも設定しないと、今日のログのみが返されます。

1. Web ブラウザーや [Postman](https://www.postman.com/downloads/)、[HTTPie](https://httpie.io/)、cURL などのツールを使用して、構築した完全修飾 API エンドポイントに対して HTTP `GET` リクエストを実行します。

API レスポンスには、新しい行で区切られた JSON オブジェクトが含まれます。監査ログがインスタンスレベルのバケットに同期される場合と同じように、そのオブジェクトには [スキーマ]({{< relref path="#audit-log-schema" lang="ja" >}}) に記載されたフィールドが含まれます。その場合、監査ログはバケット内の `/wandb-audit-logs` ディレクトリー内に配置されます。

### 基本認証を使用する
Audit logs API に アクセスするために基本認証を API キーで使用するには、HTTP リクエストの `Authorization` ヘッダーを `Basic` という文字列の後にスペースを置き、 その後にフォーマット `username:API-KEY` で base-64 エンコードされた文字列を設定します。すなわち、ユーザー名と API キーを `:` 文字で区切って、その結果を base-64 エンコードします。例えば、`demo:p@55w0rd` として認証するには、ヘッダーは `Authorization: Basic ZGVtbzpwQDU1dzByZA==` となります。

### 監査ログ取得時の PII を除外する {#exclude-pii}
[Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) および [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) では、W&B の組織やインスタンス管理者が監査ログを取得する際に PII を除外できます。[SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) では、監査ログの関連フィールドを常に API エンドポイントが返します。この設定は変更できません。

PII を除外するには、URL パラメータ `anonymize=true` を渡します。例えば、W&B インスタンスの URL が `https://mycompany.wandb.io` で、過去 1 週間のユーザー活動の監査ログを取得し、PII を除外したい場合、以下のような API エンドポイントを使用します：

https://mycompany.wandb.io/admin/audit_logs?numDays=7&anonymize=true.

## アクション
この表は、W&B によって記録される可能性のあるアクションをアルファベット順で説明しています。

| アクション | 定義 |
|-----|-----|
| `artifact:create`             | アーティファクトが作成されます。
| `artifact:delete   `          | アーティファクトが削除されます。
| `artifact:read`               | アーティファクトが読み取られます。
| `project:delete`              | プロジェクトが削除されます。
| `project:read`                | プロジェクトが読み取られます。
| `report:read`                 | レポートが読み取られます。 <sup><a href="#1">1</a></sup>
| `run:delete_many`             | 複数の run が削除されます。
| `run:delete`                  | run が削除されます。
| `run:stop`                    | run が停止されます。
| `run:undelete_many`           | 複数の run がゴミ箱から復元されます。
| `run:update_many`             | 複数の run が更新されます。
| `run:update`                  | run が更新されます。
| `sweep:create_agent`          | sweep agent が作成されます。
| `team:create_service_account` | チーム用のサービスアカウントが作成されます。
| `team:create`                 | チームが作成されます。
| `team:delete`                 | チームが削除されます。
| `team:invite_user`            | ユーザーがチームに招待されます。
| `team:uninvite`               | ユーザーまたはサービスアカウントがチームから招待取り消されます。
| `user:create_api_key`         | ユーザーの API キーが作成されます。<sup><a href="#1">1</a></sup>
| `user:create`                 | ユーザーが作成されます。 <sup><a href="#1">1</a></sup>
| `user:deactivate`             | ユーザーが無効化されます。<sup><a href="#1">1</a></sup>
| `user:delete_api_key`         | ユーザーの API キーが削除されます。<sup><a href="#1">1</a></sup>
| `user:initiate_login`         | ユーザーがログインを開始します。<sup><a href="#1">1</a></sup>
| `user:login`                  | ユーザーがログインします。<sup><a href="#1">1</a></sup>
| `user:logout`                 | ユーザーがログアウトします。<sup><a href="#1">1</a></sup>
| `user:permanently_delete`     | ユーザーが完全に削除されます。<sup><a href="#1">1</a></sup>
| `user:reactivate`             | ユーザーが再活性化されます。<sup><a href="#1">1</a></sup>
| `user:read`                   | ユーザーのプロフィールが読み取られます。<sup><a href="#1">1</a></sup>
| `user:update`                 | ユーザーが更新されます。<sup><a href="#1">1</a></sup>

<a id="1">1</a>: [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) では、監査ログは次の項目で収集されません:
- オープンまたはパブリックプロジェクトの場合。
- `report:read` のアクション。
- 特定の組織に関連付けられていない `User` のアクション。