---
title: サービスアカウントとは何か、それはなぜ役立つのか？
menu:
  support:
    identifier: ja-support-kb-articles-service_account_useful
support:
- administrator
toc_hide: true
type: docs
url: /support/:filename
---

サービスアカウント (エンタープライズ専用機能) は、人間ではないユーザーまたは機械がチームやProjects全体で一般的なタスクを自動化するためのもので、特定の人間ユーザーに特化しないタスクに使用されます。チーム内でサービスアカウントを作成し、そのAPIキーを使用してチーム内のProjectsを読み書きすることができます。

他にも、サービスアカウントは、wandb にログを記録する自動化されたジョブ、例えば定期的な再トレーニングやナイトリービルドなどを追跡するのに便利です。必要に応じて、これらの機械がローンンチした Runs に `WANDB_USERNAME` や `WANDB_USER_EMAIL` などの [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) を関連付けることができます。

[チームサービスアカウントの振る舞い]({{< relref path="/guides/models/app/settings-page/teams.md#team-service-account-behavior" lang="ja" >}}) を参照して詳細を確認してください。

チームのサービスアカウントのAPIキーは `<WANDB_HOST_URL>/<your-team-name>/service-accounts` で取得できます。また、チームの **Team settings** に移動して **Service Accounts** タブを参照することもできます。

チームの新しいサービスアカウントを作成するには:
* チームの **Service Accounts** タブで **+ 新しいサービスアカウント** ボタンを押します
* **Name** フィールドで名前を指定します
* 認証メソッドとして **Generate API key (Built-in)** を選択します
* **Create** ボタンを押します
* 新しく作成したサービスアカウントの **Copy API key** ボタンをクリックし、それを秘密の管理者や他の安全でアクセス可能な場所に保存します

{{% alert %}}
**Built-in** サービスアカウントの他に、W&B は [SDKとCLI用のアイデンティティフェデレーション]({{< relref path="/guides/hosting/iam/authentication/identity_federation.md#external-service-accounts" lang="ja" >}}) を使用した **外部サービスアカウント** もサポートしています。サービスアイデンティティを使用して W&B タスクを自動化したい場合は、JSON Web Tokens (JWT) を発行できるあなたのアイデンティティプロバイダで管理される外部サービスアカウントを使用してください。
{{% /alert %}}