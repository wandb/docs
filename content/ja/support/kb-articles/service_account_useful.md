---
title: サービスアカウントとは何か、そしてなぜ便利なのか？
url: /support/:filename
toc_hide: true
type: docs
support:
- 管理者
---

サービスアカウント（エンタープライズ限定機能）は、非人間もしくはマシンユーザーを表し、チームや Projects をまたいだ共通タスクの自動化や、特定の人間ユーザーに紐づかないタスクの自動化に利用できます。チーム内でサービスアカウントを作成し、その APIキー を使って、そのチーム内の Projects に読み書きできます。

他にも、サービスアカウントは wandb への自動化されたジョブのログ（定期的な再トレーニング、ナイトリービルドなど）に便利です。必要に応じて、[環境変数]({{< relref "/guides/models/track/environment-variables.md" >}}) `WANDB_USERNAME` または `WANDB_USER_EMAIL` を使って、これらのマシン実行 run にユーザー名を関連付けることも可能です。

[Team Service Account Behavior]({{< relref "/guides/models/app/settings-page/teams.md#team-service-account-behavior" >}}) もご参照ください。

チーム内のサービスアカウントの APIキー は `<WANDB_HOST_URL>/<your-team-name>/service-accounts` から取得できます。もしくは、対象チームの **Team settings** から **Service Accounts** タブを開いても確認できます。

新しいサービスアカウントをチームに作成する手順:
* チームの **Service Accounts** タブで **+ New service account** ボタンをクリック
* **Name** フィールドに名前を入力
* 認証方法として **Generate API key (Built-in)** を選択
* **Create** ボタンを押す
* 作成したサービスアカウントの **Copy API key** ボタンをクリックし、APIキー をシークレットマネージャや安全かつアクセス可能な場所に保管してください

{{% alert %}}
**Built-in** サービスアカウントのほかに、W&B は [SDKおよびCLI向けアイデンティティフェデレーション]({{< relref "/guides/hosting/iam/authentication/identity_federation.md#external-service-accounts" >}}) を利用した **External service accounts** もサポートしています。アイデンティティプロバイダーで管理され、JSON Web Token (JWT) を発行できるサービスIDを使って W&B のタスクを自動化したい場合は、External service accounts をご利用ください。
{{% /alert %}}