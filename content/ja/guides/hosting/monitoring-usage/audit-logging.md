---
title: 監査ログでユーザーのアクティビティを追跡する
menu:
  default:
    identifier: audit-logging
    parent: monitoring-and-usage
weight: 1
---

W&B の監査ログを活用することで、組織内のユーザーアクティビティを追跡したり、エンタープライズのガバナンス要件への準拠を実現できます。監査ログは JSON 形式で提供されます。[監査ログスキーマ]({{< relref "#audit-log-schema" >}})もご参照ください。

監査ログへのアクセス方法は、ご利用中の W&B プラットフォームのデプロイメントタイプによって異なります。

| W&B プラットフォームデプロイメントタイプ | 監査ログへのアクセス方法 |
|----------------------------|--------------------------------|
| [Self-Managed]({{< relref "/guides/hosting/hosting-options/self-managed.md" >}}) | インスタンスレベルのバケットに10分ごとに同期されます。[API]({{< relref "#fetch-audit-logs-using-api" >}})でも取得可能です。 |
| [Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}}) ＋ [セキュアストレージコネクタ（BYOB）]({{< relref "/guides/hosting/data-security/secure-storage-connector.md" >}}) | インスタンスレベルのバケット（BYOB）に10分ごとに同期されます。[API]({{< relref "#fetch-audit-logs-using-api" >}})でも取得可能です。 |
| [Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}}) ＋ W&B 管理ストレージ（BYOBなし） | [API]({{< relref "#fetch-audit-logs-using-api" >}})でのみ取得可能です。|
| [Multi-tenant Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}) | エンタープライズプランのみ利用可能です。[API]({{< relref "#fetch-audit-logs-using-api" >}})でのみ取得可能です。|

監査ログ取得後は、[Pandas](https://pandas.pydata.org/docs/index.html)、[Amazon Redshift](https://aws.amazon.com/redshift/)、[Google BigQuery](https://cloud.google.com/bigquery)、[Microsoft Fabric](https://www.microsoft.com/microsoft-fabric) などのツールを用いて分析できます。一部の監査ログ分析ツールは JSON 非対応のため、分析前にログを変換する方法と要件についてはご利用中の分析ツールのドキュメントをご確認ください。

{{% alert title="監査ログの保管期間" %}}
一定期間監査ログの保持が必要な場合、ストレージバケットや Audit Logging API を利用して定期的にログを長期保管先に移動することをおすすめします。

[医療保険の携行性と責任に関する法律 (HIPAA)](https://www.hhs.gov/hipaa/for-professionals/index.html) の適用を受ける場合、監査ログは、強制保管期間終了前に社内・外部のいかなる人物によっても削除・変更ができない環境で、最低6年間保管する必要があります。HIPAA準拠の [Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}}) インスタンス（[BYOB]({{< relref "/guides/hosting/data-security/secure-storage-connector.md" >}}) 利用時）では、長期保管先を含めてマネージドストレージのガードレール設定が必要です。
{{% /alert %}}

## 監査ログスキーマ
この表は、監査ログエントリに含まれる可能性のある全てのキーをアルファベット順に示しています。アクションや状況によって、実際のエントリには一部のみ表示されます。

| キー | 定義 |
|---------| -------|
|`action`                  | イベントの [アクション]({{< relref "#actions" >}})。
|`actor_email`             | アクションを実行したユーザーのメールアドレス（該当する場合）。
|`actor_ip`                | アクションを実行したユーザーの IP アドレス。
|`actor_user_id`           | アクションを実行したログインユーザーの ID（該当する場合）。
|`artifact_asset`          | アクションに関連する Artifact の ID（該当する場合）。
|`artifact_digest`         | アクションに関連する Artifact のダイジェスト（該当する場合）。
|`artifact_qualified_name` | アクションに関連する Artifact のフルネーム（該当する場合）。
|`artifact_sequence_asset` | アクションに関連する Artifact シーケンスの ID（該当する場合）。
|`cli_version`             | アクションを発生させた Python SDK のバージョン（該当する場合）。
|`entity_asset`            | アクションに関連する Entity または Team の ID（該当する場合）。
|`entity_name`             | アクションに関連する Entity または Team の名前（該当する場合）。
|`project_asset`           | アクションに関連する Project（該当する場合）。
|`project_name`            | アクションに関連する Project の名前（該当する場合）。
|`report_asset`            | アクションに関連する Report の ID（該当する場合）。
|`report_name`             | アクションに関連する Report の名前（該当する場合）。
|`response_code`           | アクションの HTTP レスポンスコード（該当する場合）。
|`timestamp`               | イベント時刻 ([RFC3339 形式](https://www.rfc-editor.org/rfc/rfc3339))。例: `2023-01-23T12:34:56Z` は 2023年1月23日12:34:56 UTC を表します。
|`user_asset`              | アクションが影響を与えるユーザーアセット（アクションを実行するユーザーではなく）、該当する場合。
|`user_email`              | アクションが影響を与えるユーザーのメールアドレス（アクションを実行するユーザーのアドレスではなく）、該当する場合。

### 個人を特定できる情報（PII）について

メールアドレスや Project・Team・Report の名前などの個人を特定できる情報（PII）は、API エンドポイントオプションを利用する場合のみ取得可能です。
- [Self-Managed]({{< relref "/guides/hosting/hosting-options/self-managed.md" >}}) および 
  [Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}}) では、監査ログ取得時に [PII を除外]({{< relref "#exclude-pii" >}}) できます（組織管理者のみ）。
- [Multi-tenant Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}) では、API エンドポイントは監査ログ用に常に該当フィールド（PII含む）を返します。この設定は変更できません。

## 監査ログの取得
W&B インスタンスの組織管理者やインスタンス管理者は、Audit Logging API の `audit_logs/` エンドポイントを使って監査ログを取得できます。

{{% alert %}}
- 一般ユーザー（管理者以外）が監査ログを取得しようとすると、HTTP `403` エラーでアクセス拒否されます。

- 複数エンタープライズ [Multi-tenant Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}) 組織の管理者の場合、Audit Logging API リクエストを送信する組織を設定する必要があります。プロフィール画像をクリックし、**User Settings** を選択してください。設定名は **Default API organization** です。
{{% /alert %}}

1. ご自身のインスタンスに合った API エンドポイントを確認します:

    - [Self-Managed]({{< relref "/guides/hosting/hosting-options/self-managed.md" >}}): `<wandb-platform-url>/admin/audit_logs`
    - [Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}}): `<wandb-platform-url>/admin/audit_logs`
    - [Multi-tenant Cloud（エンタープライズ必須）]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}): `https://api.wandb.ai/audit_logs`

    続く手順では `<API-endpoint>` をご自身の API エンドポイントに置き換えてください。
1. ベースエンドポイントに URL パラメータを必要に応じて追加して、完全な API エンドポイントを作成します:
    - `anonymize`: `true` に設定すると PII 情報を除外（デフォルトは `false`）。[PII 除外について]({{< relref "#exclude-pii" >}}) 参照。Multi-tenant Cloud では非対応です。
    - `numDays`: `today - numdays` から直近までのログを取得（デフォルトは `0` で当日のみ）。Multi-tenant Cloud では 7 日前までが取得可能です。
    - `startDate`: `YYYY-MM-DD` 形式の日付。Multi-tenant Cloud のみ対応。

      `startDate` と `numDays` の関係は以下の通りです：
        - 両方指定した場合：`startDate` から `startDate + numDays` までのログを取得
        - `startDate` なし/`numDays` あり：`today` から `numDays` 分のログ
        - どちらも未指定：`today` のみ取得

1. 作成した API エンドポイントに対して、Web ブラウザや [Postman](https://www.postman.com/downloads/)、[HTTPie](https://httpie.io/)、cURL などのツールで HTTP `GET` リクエストを実行します。

API レスポンスは改行区切りの JSON オブジェクトになります。オブジェクト内のフィールドは [スキーマ]({{< relref "#audit-log-schemag" >}})と同様です。インスタンスレベルのバケットに同期される場合、監査ログはバケット内の `/wandb-audit-logs` ディレクトリーに保存されます。

### ベーシック認証の利用方法
API キーで監査ログ API にベーシック認証を使うには、HTTP リクエストの `Authorization` ヘッダーに `Basic` と半角スペース、続けて `username:API-KEY` を base64 エンコードした文字列を指定します。つまり、ユーザー名と API キーを「:」で区切って結合し、それを base64 でエンコードしてください。  
例：`demo:p@55w0rd` で認証する場合は、`Authorization: Basic ZGVtbzpwQDU1dzByZA==` となります。

### 監査ログ取得時にPIIを除外する {#exclude-pii}
[Self-Managed]({{< relref "/guides/hosting/hosting-options/self-managed.md" >}})、[Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}}) では、W&B 組織管理者またはインスタンス管理者が監査ログ取得時にPIIを除外できます。[Multi-tenant Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}) では、API エンドポイントが常にPIIを含む関連フィールドを返すため、設定変更はできません。

PII を除外するには、URL パラメータとして `anonymize=true` を渡します。  
たとえば W&B インスタンスの URL が `https://mycompany.wandb.io` で、過去1週間のユーザーアクティビティ監査ログを取得し、PII を除外したい場合、下記のような API エンドポイントになります。

```text
https://mycompany.wandb.io/admin/audit_logs?numDays=7&anonymize=true.
```

## アクション一覧
W&B で記録される可能性があるアクションをアルファベット順に示します。

|アクション | 説明 |
|-----|-----|
| `artifact:create`             | Artifact の作成。
| `artifact:delete   `          | Artifact の削除。
| `artifact:read`               | Artifact の参照。
| `project:delete`              | Project の削除。
| `project:read`                | Project の参照。
| `report:read`                 | Report の参照。<sup><a href="#1">1</a></sup>
| `run:delete_many`             | 複数の Run の削除。
| `run:delete`                  | Run の削除。
| `run:stop`                    | Run の停止。
| `run:undelete_many`           | 複数の Run をゴミ箱から復元。
| `run:update_many`             | 複数の Run を更新。
| `run:update`                  | Run を更新。
| `sweep:create_agent`          | Sweep agent の作成。
| `team:create_service_account` | Team 用のサービスアカウント作成。
| `team:create`                 | Team の作成。
| `team:delete`                 | Team の削除。
| `team:invite_user`            | Team へのユーザー招待。
| `team:uninvite`               | Team からユーザーまたはサービスアカウントを招待解除。
| `user:create_api_key`         | ユーザーの APIキー作成。<sup><a href="#1">1</a></sup>
| `user:create`                 | ユーザーの作成。<sup><a href="#1">1</a></sup>
| `user:deactivate`             | ユーザーの無効化。<sup><a href="#1">1</a></sup>
| `user:delete_api_key`         | ユーザーの APIキー削除。<sup><a href="#1">1</a></sup>
| `user:initiate_login`         | ユーザーがログイン操作を開始。<sup><a href="#1">1</a></sup>
| `user:login`                  | ユーザーがログイン。<sup><a href="#1">1</a></sup>
| `user:logout`                 | ユーザーがログアウト。<sup><a href="#1">1</a></sup>
| `user:permanently_delete`     | ユーザーを完全に削除。<sup><a href="#1">1</a></sup>
| `user:reactivate`             | ユーザーの再有効化。<sup><a href="#1">1</a></sup>
| `user:read`                   | ユーザープロファイルの参照。<sup><a href="#1">1</a></sup>
| `user:update`                 | ユーザー情報の更新。<sup><a href="#1">1</a></sup>

<a id="1">1</a>: [Multi-tenant Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}) では、以下の監査ログは収集されません:
- Open または Public な Project
- `report:read` アクション
- 特定の組織に紐づかない `User` アクション