---
title: 監査ログで ユーザー のアクティビティを追跡する
menu:
  default:
    identifier: ja-guides-hosting-monitoring-usage-audit-logging
    parent: monitoring-and-usage
weight: 1
---

W&B の監査ログを活用することで、組織内のユーザー活動を追跡し、エンタープライズ向けガバナンス要件への準拠を実現できます。監査ログは JSON 形式で提供されます。詳細は[監査ログスキーマ]({{< relref path="#audit-log-schema" lang="ja" >}})をご参照ください。

監査ログへのアクセス方法は、W&B プラットフォームのデプロイタイプによって異なります。

| W&B プラットフォームのデプロイタイプ | 監査ログへのアクセス方法 |
|----------------------------|--------------------------------|
| [Self-Managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) | インスタンスレベルのバケットに 10 分ごとに同期されます。さらに[API を利用]({{< relref path="#fetch-audit-logs-using-api" lang="ja" >}})して取得することも可能です。 |
| [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) + [セキュアストレージコネクタ（BYOB）]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}}) | インスタンスレベルのバケット（BYOB）に 10 分ごとに同期されます。さらに[API を利用]({{< relref path="#fetch-audit-logs-using-api" lang="ja" >}})して取得することも可能です。 |
| [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) （W&B 管理ストレージ利用時 / BYOBなし） | [API のみ]({{< relref path="#fetch-audit-logs-using-api" lang="ja" >}})で取得可能です。 |
| [Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) | Enterprise プランのみ利用可能。[API のみ]({{< relref path="#fetch-audit-logs-using-api" lang="ja" >}})で取得可能です。

監査ログ取得後は、[Pandas](https://pandas.pydata.org/docs/index.html)、[Amazon Redshift](https://aws.amazon.com/redshift/)、[Google BigQuery](https://cloud.google.com/bigquery)、[Microsoft Fabric](https://www.microsoft.com/microsoft-fabric) などのツールで分析が可能です。一部の監査ログ分析ツールは JSON をサポートしていません。ご利用の分析ツールのドキュメントをご確認いただき、分析前に JSON 形式の監査ログ整形が必要かご確認ください。

{{% alert title="監査ログの保持" %}}
特定期間、監査ログの保持が必要な場合は、ストレージバケットまたは Audit Logging API を利用して、定期的に長期保存用ストレージへ転送することを推奨します。

[医療保険の携帯性と責任に関する法律（HIPAA）](https://www.hhs.gov/hipaa/for-professionals/index.html) の対象となる場合、監査ログは最短 6 年間、保持期間満了まで内部・外部の関係者による削除・変更ができない環境で管理される必要があります。HIPAA 準拠の [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) インスタンス（[BYOB]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}})利用時）は、長期保存ストレージも含めストレージのガードレールを設定してください。
{{% /alert %}}

## 監査ログスキーマ
この表は、監査ログエントリに現れる可能性のあるすべてのキーとその定義をアルファベット順に示しています。アクションや状況によって、ログエントリには以下のキーの一部のみが含まれる場合があります。

| Key | 定義 |
|---------| -------|
|`action`                  | イベントの [action]({{< relref path="#actions" lang="ja" >}})。
|`actor_email`             | アクションを実行したユーザーのメールアドレス（該当する場合）。
|`actor_ip`                | アクションを実行したユーザーの IP アドレス。
|`actor_user_id`           | アクションを実行したログインユーザーのID（該当する場合）。
|`artifact_asset`          | アクションに関連する Artifacts のID（該当する場合）。
|`artifact_digest`         | アクションに関連する Artifacts のダイジェスト（該当する場合）。
|`artifact_qualified_name` | アクションに関連する Artifacts のフルネーム（該当する場合）。
|`artifact_sequence_asset` | アクションに関連する Artifacts のシーケンスID（該当する場合）。
|`cli_version`             | アクションを実行した Python SDK のバージョン（該当する場合）。
|`entity_asset`            | アクションに関連する Entity または Team のID（該当する場合）。
|`entity_name`             | アクションに関連する Entity または Team の名称（該当する場合）。
|`project_asset`           | アクションに関連する Projects（該当する場合）。
|`project_name`            | アクションに関連する Projects の名称（該当する場合）。
|`report_asset`            | アクションに関連する Reports のID（該当する場合）。
|`report_name`             | アクションに関連する Reports の名称（該当する場合）。
|`response_code`           | アクションに対する HTTP レスポンスコード（該当する場合）。
|`timestamp`               | イベント発生時刻（[RFC3339 形式](https://www.rfc-editor.org/rfc/rfc3339)）。例: `2023-01-23T12:34:56Z` は 2023 年 1 月 23 日 12:34:56 UTC を表します。
|`user_asset`              | アクションの影響対象となるユーザーアセット（実行ユーザーとは異なる場合）。
|`user_email`              | アクションの影響対象となるユーザーのメールアドレス（実行ユーザーとは異なる場合）。

### 個人を特定できる情報（PII）について

メールアドレスや Projects・Teams・Reports の名称などの個人を特定できる情報（PII）は、API エンドポイント利用時のみアクセスできます。
- [Self-Managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) および 
  [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) の場合、組織管理者は監査ログ取得時に [PII を除外]({{< relref path="#exclude-pii" lang="ja" >}})できます。
- [Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) の場合、API エンドポイントは監査ログ取得時に常に関連フィールド（PII含む）を返します（設定変更不可）。

## 監査ログの取得
W&B 組織またはインスタンス管理者は、Audit Logging API の `audit_logs/` エンドポイントを通じて監査ログを取得できます。

{{% alert %}}
- 管理者以外のユーザーが監査ログの取得を試みた場合、HTTP `403` エラー（アクセス拒否）が発生します。

- 複数の Enterprise [Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}})組織の管理者の場合は、どの組織に対して Audit Logging API リクエストを送信するかを設定してください。プロフィール画像をクリックし、**User Settings** をクリックします。設定項目名は **Default API organization** です。
{{% /alert %}}

1. インスタンスごとに正しい API エンドポイントを確認します。

    - [Self-Managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}): `<wandb-platform-url>/admin/audit_logs`
    - [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}): `<wandb-platform-url>/admin/audit_logs`
    - [Multi-tenant Cloud (Enterprise required)]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}): `https://api.wandb.ai/audit_logs`

    以降の手順では `<API-endpoint>` をご自身の API エンドポイントに置き換えてください。
1. ベースエンドポイントからフル API エンドポイントを作成し、必要に応じてURL パラメータを追加します。
    - `anonymize`: `true` に設定すると PII を削除します（デフォルトは `false`）。[PII を除外して監査ログを取得]({{< relref path="#exclude-pii" lang="ja" >}})を参照。Multi-tenant Cloud では非対応。
    - `numDays`: `today - numdays` から直近のログまで取得します（デフォルトは `0` で本日分のみ）。Multi-tenant Cloud の場合、最大過去 7 日分まで取得できます。
    - `startDate`: `YYYY-MM-DD` 形式で指定する任意の日付（[Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) のみ対応）。

      `startDate` と `numDays` の関係：
        - 両方指定時：`startDate` から `startDate + numDays` まで取得
        - `startDate` は省略・`numDays` のみ指定：`today` から `numDays` まで取得
        - どちらも指定しない場合：今日のみ取得

1. 作成した API エンドポイントに対し [Postman](https://www.postman.com/downloads/)、[HTTPie](https://httpie.io/)、cURL 等のツールやブラウザで HTTP `GET` リクエストを実行します。

API のレスポンスは、改行区切りの JSON オブジェクトとなって返却されます。各オブジェクトには、[スキーマ]({{< relref path="#audit-log-schemag" lang="ja" >}}) で示したフィールドが含まれます。インスタンスレベルのバケットへ同期される監査ログも同様です。同期時はバケット内 `/wandb-audit-logs` ディレクトリーに格納されます。

### ベーシック認証の利用
API キーで監査ログ API にアクセスする際は、HTTP リクエストの `Authorization` ヘッダーに `Basic`と半角スペース、続けて `username:API-KEY` 形式の文字列を base64 エンコードしたものを設定します。ユーザー名とAPIキーを `:` でつなげ、base64 エンコードしてください。例：`demo:p@55w0rd` で認証する場合、ヘッダーには `Authorization: Basic ZGVtbzpwQDU1dzByZA==` を指定します。

### 監査ログ取得時に PII を除外する {#exclude-pii}
[Self-Managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}})、[Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) の場合、W&B 組織またはインスタンス管理者は監査ログ取得時に PII を除外できます。[Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) では、常に該当フィールド（PII含む）が返るため設定変更はできません。

PII を除外する場合は、URL パラメータ `anonymize=true` を指定してください。例えば、W&B インスタンス URL が `https://mycompany.wandb.io` で、過去 1 週間分のユーザーアクティビティ監査ログを PII 除外で取得する場合、エンドポイントは以下のようになります。

```text
https://mycompany.wandb.io/admin/audit_logs?numDays=7&anonymize=true.
```

## Actions
この表は、W&B で記録される可能性のあるアクションをアルファベット順で説明します。

|アクション | 定義 |
|-----|-----|
| `artifact:create`             | Artifact が作成される。
| `artifact:delete   `          | Artifact が削除される。
| `artifact:read`               | Artifact が参照される。
| `project:delete`              | Project が削除される。
| `project:read`                | Project が参照される。
| `report:read`                 | Report が参照される。<sup><a href="#1">1</a></sup>
| `run:delete_many`             | 複数の Run が削除される。
| `run:delete`                  | Run が削除される。
| `run:stop`                    | Run が停止される。
| `run:undelete_many`           | 複数の Run がゴミ箱から復元される。
| `run:update_many`             | 複数の Run が更新される。
| `run:update`                  | Run が更新される。
| `sweep:create_agent`          | sweep agent が作成される。
| `team:create_service_account` | Team 用サービスアカウントが作成される。
| `team:create`                 | Team が作成される。
| `team:delete`                 | Team が削除される。
| `team:invite_user`            | Team にユーザーが招待される。
| `team:uninvite`               | Team からユーザーまたはサービスアカウントの招待が取り消される。
| `user:create_api_key`         | ユーザーの APIキーが作成される。<sup><a href="#1">1</a></sup>
| `user:create`                 | ユーザーが作成される。<sup><a href="#1">1</a></sup>
| `user:deactivate`             | ユーザーが無効化される。<sup><a href="#1">1</a></sup>
| `user:delete_api_key`         | ユーザーの APIキーが削除される。<sup><a href="#1">1</a></sup>
| `user:initiate_login`         | ユーザーがログインを開始する。<sup><a href="#1">1</a></sup>
| `user:login`                  | ユーザーがログインする。<sup><a href="#1">1</a></sup>
| `user:logout`                 | ユーザーがログアウトする。<sup><a href="#1">1</a></sup>
| `user:permanently_delete`     | ユーザーが完全に削除される。<sup><a href="#1">1</a></sup>
| `user:reactivate`             | ユーザーが再有効化される。<sup><a href="#1">1</a></sup>
| `user:read`                   | ユーザーのプロフィールが参照される。<sup><a href="#1">1</a></sup>
| `user:update`                 | ユーザーが更新される。<sup><a href="#1">1</a></sup>

<a id="1">1</a>: [Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) では次の項目で監査ログが収集されません：
- オープンまたは公開 Projects。
- `report:read` アクション。
- 特定の組織に紐づかない `User` アクション。