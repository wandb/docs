---
title: 監査ログでユーザーのアクティビティを追跡する
menu:
  default:
    identifier: ja-guides-hosting-monitoring-usage-audit-logging
    parent: monitoring-and-usage
weight: 1
---

W&B の監査ログを使って、組織内のユーザー アクティビティを追跡し、エンタープライズのガバナンス要件に準拠しましょう。監査ログは JSON 形式で提供されます。詳しくは [監査ログ スキーマ]({{< relref path="#audit-log-schema" lang="ja" >}}) を参照してください。

監査ログへのアクセス方法は、W&B プラットフォームのデプロイメント形態によって異なります。

| W&B Platform Deployment type | 監査ログへのアクセス方法 |
|----------------------------|--------------------------------|
| [Self-Managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) | 10 分ごとにインスタンス レベルのバケットへ同期。加えて [API]({{< relref path="#fetch-audit-logs-using-api" lang="ja" >}}) でも取得可能。 |
| [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) with [secure storage connector (BYOB)]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}}) | 10 分ごとにインスタンス レベルのバケット（BYOB）へ同期。加えて [API]({{< relref path="#fetch-audit-logs-using-api" lang="ja" >}}) でも取得可能。 |
| [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) with W&B managed storage (without BYOB) | [API]({{< relref path="#fetch-audit-logs-using-api" lang="ja" >}}) からのみ取得可能。 |
| [Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) | Enterprise プランのみ利用可能。[API]({{< relref path="#fetch-audit-logs-using-api" lang="ja" >}}) からのみ取得可能。 |

監査ログを取得したら、[Pandas](https://pandas.pydata.org/docs/index.html)、[Amazon Redshift](https://aws.amazon.com/redshift/)、[Google BigQuery](https://cloud.google.com/bigquery)、[Microsoft Fabric](https://www.microsoft.com/microsoft-fabric) などのツールで分析できます。監査ログの分析ツールの中には JSON をサポートしないものもあります。分析前に JSON 形式の監査ログを変換する際のガイドラインや要件については、利用する分析ツールのドキュメントを参照してください。

{{% alert title="監査ログの保持" %}}
特定の期間、監査ログの保持が必要な場合は、ストレージ バケットや Audit Logging API を用いて、定期的に長期保存ストレージへ転送することを W&B は推奨します。

[HIPAA（医療保険の相互運用性と説明責任に関する法律）](https://www.hhs.gov/hipaa/for-professionals/index.html) の適用を受ける場合、監査ログは、義務的な保持期間が終了する前に、内部・外部のいかなるアクターによっても削除・変更できない 環境で、最低 6 年間保持しなければなりません。HIPAA 準拠の [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) インスタンスで [BYOB]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}}) を利用する場合は、長期保持ストレージを含む管理ストレージに対してガードレールを設定する必要があります。
{{% /alert %}}

## 監査ログ スキーマ
この表は、監査ログ エントリに現れる可能性のあるすべてのキーをアルファベット順に示します。アクションや状況に応じて、特定のログ エントリには可能なフィールドのサブセットのみが含まれます。

| Key | 定義 |
|---------| -------|
|`action`                  | イベントの [action]({{< relref path="#actions" lang="ja" >}})。 |
|`actor_email`             | 対象となる場合、アクションを開始したユーザーのメール アドレス。 |
|`actor_ip`                | アクションを開始したユーザーの IP アドレス。 |
|`actor_user_id`           | 対象となる場合、アクションを実行したログイン中のユーザーの ID。 |
|`artifact_asset`          | 対象となる場合、アクションに関連する Artifact の ID。 |
|`artifact_digest`         | 対象となる場合、アクションに関連する Artifact のダイジェスト。 |
|`artifact_qualified_name` | 対象となる場合、アクションに関連する Artifact のフルネーム。 |
|`artifact_sequence_asset` | 対象となる場合、アクションに関連する Artifact シーケンスの ID。 |
|`cli_version`             | 対象となる場合、アクションを開始した Python SDK のバージョン。 |
|`entity_asset`            | 対象となる場合、アクションに関連する Entity または Team の ID。 |
|`entity_name`             | 対象となる場合、アクションに関連する Entity または Team の名前。 |
|`project_asset`           | 対象となる場合、アクションに関連する Project。 |
|`project_name`            | 対象となる場合、アクションに関連する Project の名前。 |
|`report_asset`            | 対象となる場合、アクションに関連する Report の ID。 |
|`report_name`             | 対象となる場合、アクションに関連する Report の名前。 |
|`response_code`           | 対象となる場合、アクションの HTTP レスポンス コード。 |
|`timestamp`               | [RFC3339 形式](https://www.rfc-editor.org/rfc/rfc3339) のイベント時刻。例えば、`2023-01-23T12:34:56Z` は 2023 年 1 月 23 日 12:34:56 UTC を表します。 |
|`user_asset`              | 対象となる場合、（アクションを実行するユーザーではなく）アクションの影響を受ける User アセット。 |
|`user_email`              | 対象となる場合、（アクションを実行するユーザーではなく）アクションの影響を受ける User のメール アドレス。 |

### 個人を特定できる情報（PII）
メール アドレス、Project・Team・Report 名などの PII は、API エンドポイントによる取得でのみ利用可能です。
- [Self-Managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) および 
  [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) では、組織の管理者が監査ログを取得する際に [PII を除外]({{< relref path="#exclude-pii" lang="ja" >}}) できます。
- [Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) では、API エンドポイントは PII を含む、監査ログに関連するフィールドを常に返します。これは設定で変更できません。

## 監査ログの取得
組織またはインスタンスの管理者は、エンドポイント `audit_logs/` にある Audit Logging API を使用して W&B インスタンスの監査ログを取得できます。

{{% alert %}}
- 管理者以外のユーザーが監査ログの取得を試みると、HTTP `403` エラーが発生し、アクセス拒否が示されます。
- 複数の Enterprise 向け [Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) 組織の管理者である場合は、監査ログ API リクエストの送信先となる組織を設定する必要があります。プロフィール画像をクリックし、次に **User Settings** をクリックします。設定名は **Default API organization** です。
{{% /alert %}}

1. ご利用のインスタンスに対する正しい API エンドポイントを確認します。
    - [Self-Managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}): `<wandb-platform-url>/admin/audit_logs`
    - [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}): `<wandb-platform-url>/admin/audit_logs`
    - [Multi-tenant Cloud (Enterprise 必須)]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}): `https://api.wandb.ai/audit_logs`

    以降の手順では、`<API-endpoint>` をあなたの API エンドポイントに置き換えてください。
1. ベース エンドポイントから完全な API エンドポイントを組み立て、必要に応じて URL パラメータを追加します。
    - `anonymize`: `true` の場合、PII を削除します。デフォルトは `false`。詳しくは [監査ログ取得時に PII を除外]({{< relref path="#exclude-pii" lang="ja" >}}) を参照。Multi-tenant Cloud では非対応。
    - `numDays`: `today - numdays` から最新までのログを取得します。デフォルトは `0`（`today` のログのみを返す）。Multi-tenant Cloud では、過去 7 日分が上限です。
    - `startDate`: `YYYY-MM-DD` 形式の任意の日付。[Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) のみ対応。

      `startDate` と `numDays` の関係:
        - `startDate` と `numDays` の両方を指定した場合、`startDate` から `startDate + numDays` までのログが返されます。
        - `startDate` を省略して `numDays` のみ指定した場合、`today` から `numDays` までのログが返されます。
        - `startDate` と `numDays` のどちらも指定しない場合、`today` のログのみが返されます。

1. 作成した完全な API エンドポイントに対し、Web ブラウザまたは [Postman](https://www.postman.com/downloads/)、[HTTPie](https://httpie.io/)、cURL などのツールで HTTP `GET` リクエストを実行します。

API のレスポンスは、改行区切りの JSON オブジェクトです。オブジェクトには、インスタンス レベルのバケットへ監査ログを同期する場合と同様に、[スキーマ]({{< relref path="#audit-log-schemag" lang="ja" >}}) に記載されたフィールドが含まれます。その場合、監査ログはバケット内の `/wandb-audit-logs` ディレクトリーに配置されます。

### Basic 認証を使う
API で監査ログにアクセスする際に APIキー を使って Basic 認証を行うには、HTTP リクエストの `Authorization` ヘッダーに、`Basic`（後に半角スペース）と、`username:API-KEY` 形式の文字列を Base64 エンコードしたものを設定します。つまり、ユーザー名と APIキー を `:` で連結した文字列を Base64 エンコードして指定します。例えば `demo:p@55w0rd` で認証する場合、ヘッダーは `Authorization: Basic ZGVtbzpwQDU1dzByZA==` となります。

### 監査ログ取得時に PII を除外 {#exclude-pii}
[Self-Managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) と [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) では、W&B の組織またはインスタンス管理者が監査ログ取得時に PII を除外できます。[Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) では、API エンドポイントは PII を含む監査ログの関連フィールドを常に返し、変更はできません。

PII を除外するには、URL パラメータとして `anonymize=true` を指定します。例えば、W&B インスタンスの URL が `https://mycompany.wandb.io` で、直近 1 週間のユーザー アクティビティに対する監査ログを PII 抜きで取得したい場合、次のような API エンドポイントを使用します。

```text
https://mycompany.wandb.io/admin/audit_logs?numDays=7&anonymize=true.
```

## Actions
この表は、W&B が記録する可能性のあるアクションをアルファベット順に示したものです。

|Action | 定義 |
|-----|-----|
| `artifact:create`             | Artifact is created. |
| `artifact:delete   `          | Artifact is deleted. |
| `artifact:read`               | Artifact is read. |
| `project:delete`              | Project is deleted. |
| `project:read`                | Project is read. |
| `report:read`                 | Report is read. <sup><a href="#1">1</a></sup> |
| `run:delete_many`             | Batch of runs is deleted. |
| `run:delete`                  | Run is deleted. |
| `run:stop`                    | Run is stopped. |
| `run:undelete_many`           | Batch of runs is restored from trash. |
| `run:update_many`             | Batch of runs is updated. |
| `run:update`                  | Run is updated. |
| `sweep:create_agent`          | Sweep agent is created. |
| `team:create_service_account` | Service account is created for the team. |
| `team:create`                 | Team is created. |
| `team:delete`                 | Team is deleted. |
| `team:invite_user`            | User is invited to team. |
| `team:uninvite`               | User or service account is uninvited from team. |
| `user:create_api_key`         | API key for the user is created. <sup><a href="#1">1</a></sup> |
| `user:create`                 | User is created. <sup><a href="#1">1</a></sup> |
| `user:deactivate`             | User is deactivated. <sup><a href="#1">1</a></sup> |
| `user:delete_api_key`         | API key for the user is deleted. <sup><a href="#1">1</a></sup> |
| `user:initiate_login`         | User initiates log in. <sup><a href="#1">1</a></sup> |
| `user:login`                  | User logs in. <sup><a href="#1">1</a></sup> |
| `user:logout`                 | User logs out. <sup><a href="#1">1</a></sup> |
| `user:permanently_delete`     | User is permanently deleted. <sup><a href="#1">1</a></sup> |
| `user:reactivate`             | User is reactivated. <sup><a href="#1">1</a></sup> |
| `user:read`                   | User profile is read. <sup><a href="#1">1</a></sup> |
| `user:update`                 | User is updated. <sup><a href="#1">1</a></sup> |

<a id="1">1</a>: [Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) では、以下について監査ログは収集されません。
- Open または Public な Project。
- `report:read` アクション。
- 特定の組織に紐づかない `User` アクション。