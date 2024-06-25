---
displayed_sidebar: default
---


# Audit logs
W&B サーバーの監査ログを使用して、チーム内のユーザー活動を追跡し、企業のガバナンス要件に準拠します。監査ログは JSON 形式で書かれており、アクセス メカニズムは W&B サーバーデプロイメントの種類に依存します。

| W&B サーバーデプロイメントの種類 | 監査ログアクセスメカニズム |
|----------------------------|--------------------------------|
| セルフマネージド | 10 分ごとにインスタンスレベルのバケットに同期されます。API を使用しても利用可能です。|
| 専用クラウド (BYOB を含む) | 10 分ごとにインスタンスレベルのバケット (BYOB) に同期されます。API を使用しても利用可能です。|
| W&B 管理ストレージ専用クラウド (BYOB を含まない) | API を使用してのみ利用可能です。|

監査ログにアクセスしたら、お気に入りのツール (例: [Pandas](https://pandas.pydata.org/docs/index.html), [Amazon Redshift](https://aws.amazon.com/redshift/), [Google BigQuery](https://cloud.google.com/bigquery), [Microsoft Fabric](https://www.microsoft.com/en-us/microsoft-fabric) など) を使用して分析します。監査ログを分析する前に、ツールに関連する形式に変換する必要がある場合があります。特定のツールのための監査ログの変換方法に関する情報は、W&B ドキュメントの範囲外です。

:::tip
**監査ログの保持:** 組織のコンプライアンス、セキュリティ、またはリスクチームが、特定の期間監査ログを保持する必要がある場合、W&B はログを定期的にインスタンスレベルのバケットから長期保存ストレージに転送することを推奨します。API を使用して監査ログにアクセスしている場合は、簡単なスクリプトを実装して、定期的に (例えば毎日や数日ごとに) 実行し、前回のスクリプト実行以来生成された可能性のあるログを取得し、それらを短期保存ストレージに保存するか、直接長期保存ストレージに転送します。
:::

:::note
監査ログは、W&B Multi-tenant Cloud ではまだ利用できません。
:::

## 監査ログスキーマ
以下の表は、監査ログに存在する可能性のあるすべての異なるキーを一覧にしています。各ログには、対応するアクションに関連するアセットのみが含まれ、その他はログから省略されます。

| Key | 定義 |
|---------| -------|
|timestamp               | [RFC3339 フォーマット](https://www.rfc-editor.org/rfc/rfc3339) のタイムスタンプ。例: `2023-01-23T12:34:56Z` は、2023 年 1 月 23 日の `12:34:56 UTC` を表します。
|action                  | ユーザーが取った[アクション](#actions)。
|actor_user_id           | 存在する場合、アクションを実行したログインユーザーの ID。
|response_code           | アクションに対する HTTP レスポンスコード。
|artifact_asset          | 存在する場合、このアーティファクト ID に対してアクションが実行されました。
|artifact_sequence_asset | 存在する場合、このアーティファクトシーケンス ID に対してアクションが実行されました。
|entity_asset            | 存在する場合、このエンティティまたはチーム ID に対してアクションが実行されました。
|project_asset           | 存在する場合、このプロジェクト ID に対してアクションが実行されました。
|report_asset            | 存在する場合、このレポート ID に対してアクションが実行されました。
|user_asset              | 存在する場合、このユーザーアセットに対してアクションが実行されました。
|cli_version             | Python SDK を介してアクションが実行された場合、そのバージョン。
|actor_ip                | ログインユーザーの IP アドレス。
|actor_email             | 存在する場合、このアクターのメールアドレスに対してアクションが実行されました。
|artifact_digest         | 存在する場合、このアーティファクトのダイジェストに対してアクションが実行されました。
|artifact_qualified_name | 存在する場合、このアーティファクトに対してアクションが実行されました。
|entity_name             | 存在する場合、このエンティティまたはチーム名に対してアクションが実行されました。
|project_name            | 存在する場合、このプロジェクト名に対してアクションが実行されました。
|report_name             | 存在する場合、このレポート名に対してアクションが実行されました。
|user_email              | 存在する場合、このユーザーメールに対してアクションが実行されました。

メールアドレス、プロジェクト、チーム、レポート名などの個人を特定できる情報 (PII) は、API エンドポイントのオプションを使用した場合にのみ利用可能であり、以下で説明されているようにオフにすることもできます。

## API を使用した監査ログの取得
インスタンス管理者は、次の API を使用して W&B サーバーインスタンスの監査ログを取得できます。
1. 基本エンドポイント `<wandb-server-url>/admin/audit_logs` と次の URL パラメータを組み合わせてフル API エンドポイントを構築します。
    - `numDays` : `today - numdays` から最新までのログを取得します。デフォルトは `0` で、`today` のみのログが返されます。
    - `anonymize` : `true` に設定すると、PII を削除します。デフォルトは `false`。
2. 構築したフル API エンドポイントで HTTP GET リクエストを実行し、モダンブラウザ内で直接実行するか、[Postman](https://www.postman.com/downloads/), [HTTPie](https://httpie.io/), cURL コマンド などのツールを使用します。

W&B サーバーインスタンスの URL が `https://mycompany.wandb.io` で、過去 1 週間のユーザー活動の PII なしの監査ログを取得したい場合、API エンドポイントは `https://mycompany.wandb.io?numDays=7&anonymize=true` になります。

:::note
W&B サーバーの [インスタンス管理者](../iam/manage-users.md#instance-admins) のみが、API を使用して監査ログを取得することができます。インスタンス管理者でない場合や、組織にログインしていない場合、`HTTP 403 Forbidden` エラーが発生します。
:::

API のレスポンスには、新しい行で区切られた JSON オブジェクトが含まれています。オブジェクトには、スキーマで説明されているフィールドが含まれます。これは、監査ログ ファイルが (前述されたように適用される場合は) インスタンスレベルのバケットに同期される場合と同じ形式です。この場合、監査ログはバケットの `/wandb-audit-logs` ディレクトリーに配置されます。

## Actions
W&B によって記録される可能性があるアクションを以下の表に示します。

|アクション | 定義 |
|-----|-----|
| artifact:create             | Artifact が作成されます。
| artifact:delete             | Artifact が削除されます。
| artifact:read               | Artifact が読み取られます。
| project:delete              | Project が削除されます。
| project:read                | Project が読み取られます。
| report:read                 | Report が読み取られます。
| run:delete                  | Run が削除されます。
| run:delete_many             | 複数の Runs が一括削除されます。
| run:update_many             | 複数の Runs が一括更新されます。
| run:stop                    | Run が停止されます。
| run:undelete_many           | 複数の Runs が一括でゴミ箱から戻されます。
| run:update                  | Run が更新されます。
| sweep:create_agent          | Sweep agent が作成されます。
| team:invite_user            | User がチームに招待されます。
| team:create_service_account | チームのサービスアカウントが作成されます。
| team:create                 | Team が作成されます。
| team:uninvite               | チームからユーザーまたはサービスアカウントが招待取り消しされます。
| team:delete                 | Team が削除されます。
| user:create                 | User が作成されます。
| user:delete_api_key         | ユーザーの APIキーが削除されます。
| user:deactivate             | User が非アクティブ化されます。
| user:create_api_key         | ユーザーの APIキーが作成されます。
| user:permanently_delete     | User が永久に削除されます。
| user:reactivate             | User が再アクティブ化されます。
| user:update                 | User が更新されます。
| user:read                   | ユーザープロファイルが読み取られます。
| user:login                  | User がログインします。
| user:initiate_login         | User がログインを開始します。
| user:logout                 | User がログアウトします。