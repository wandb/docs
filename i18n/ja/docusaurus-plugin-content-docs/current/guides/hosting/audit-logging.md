---
description: The Prompts Quickstart shows how to visualise and debug the execution flow of your LLM chains and pipelines
displayed_sidebar: ja
---
# 監査ログ
監査ログを使用して、チーム内のアクティビティを追跡し、理解します。監査ログは、10秒ごとにバケットストアに同期されます。オプションで監査ログをダウンロードし、[Pandas](https://pandas.pydata.org/docs/index.html)、[BigQuery](https://cloud.google.com/bigquery)など、選択したツールで表示できます。

:::info
この機能は現在プライベートプレビュー中です。
:::

## 監査ログのスキーマ

| キー | 定義 |
|---------| -------|
|timestamp | [RFC3339形式](https://www.rfc-editor.org/rfc/rfc3339)でのタイムスタンプ。例: `2023-01-23T12:34:56Z` は、2023年1月23日の `12:34:56 UTC` を表しています。|
|action | ユーザーが何の[アクション](#actions) を行ったか。|
|actor_user_id| ある場合、アクションを実行したログインユーザーのID。|
|response_code |アクションに対するHttpレスポンスコード。|
|user_asset | ある場合、アクションがこのユーザーアセットを返します。|
|project_asset | ある場合、アクションがこのプロジェクトアセットを返します。|
|run_asset| ある場合、アクションがこのrunアセットを返します。|
|artifact_asset| ある場合、アクションがこのアーティファクトアセットを返します。|


## 監査ログの表示
W&Bサーバーインストールをサポートするバケット内で監査ログを表示します。

1. バケット内の `/wandb-audit-logs` ディレクトリーに移動してください。
2. 興味のある期間のファイルをダウンロードします。

1日に1つのファイルがアップロードされます。ファイルには、改行で区切られたJSONオブジェクトが含まれます。ファイルに書き込まれたオブジェクトには、スキーマで説明されているフィールドが含まれます。
## アクション
以下の表は、W&Bで記録できる可能なアクションを説明しています：

|アクション | 定義 |
|-----|-----|
|||
|"artifact:create" | |
|"artifact:read" | |
|"artifact:read" | |
|"project:delete"  | |
|"project:read" | |
|"project:read"  | |
|"report:read" | |
|"run:delete" | |
|"run:delete_many" | |
|"run:update_many" | |
|"run:stop" | |
|"run:undelete_many" | |
|"run:update" | |
|"sweep:create_agent" | |
|"team:invite_user" | |
|"team:create_service_account" | |
|"team:create" | |
|"team:uninvite" | |
|"team:delete" | |
|"user:create" | |
|"user:delete_api_key" | |
|"user:deactivate" | |
|"user:create_api_key" | |
|"user:permanently_delete" | |
|"user:reactivate" | |
|"user:update" | |
|"user:read" | |
|"user:login" | |
|"user:initiate_login" | |
|"user:logout" | |