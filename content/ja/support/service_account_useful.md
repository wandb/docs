---
title: What is a service account, and why is it useful?
menu:
  support:
    identifier: ja-support-service_account_useful
tags:
- administrator
toc_hide: true
type: docs
---

サービスアカウント (エンタープライズ限定機能) は、人間でないまたは機械的なユーザーを表し、チームやプロジェクト全体で一般的なタスクを自動化したり、特定のユーザーに依存しないタスクを実行することができます。サービスアカウントをチーム内で作成し、そのAPIキーを用いてそのチーム内のProjectsを読み書きすることができます。

サービスアカウントは、例えば定期的な再トレーニングやナイトリービルドなど、wandbにログされた自動化ジョブを追跡するのに便利です。ご希望であれば、これらの機械でローンンチされたRunsに [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) `WANDB_USERNAME` または `WANDB_USER_EMAIL` を使用してユーザー名を関連付けることができます。

詳しくは [Team Service Account Behavior]({{< relref path="/guides/models/app/settings-page/teams.md#team-service-account-behavior" lang="ja" >}}) を参照してください。

チームのサービスアカウントのAPIキーは `<WANDB_HOST_URL>/<your-team-name>/service-accounts` で取得できます。あるいは、チームの **Team settings** に行き、**Service Accounts** タブを参照してください。

新しいサービスアカウントをチームに作成するには:
* チームの **Service Accounts** タブで **+ New service account** ボタンを押します
* **Name** フィールドに名前を入力します
* 認証メソッドとして **Generate API key (Built-in)** を選択します
* **Create** ボタンを押します
* 新しく作成したサービスアカウントの **Copy API key** ボタンをクリックして、シークレットマネージャまたは他の安全でアクセス可能な場所に保存します

{{% alert %}}
**Built-in** サービスアカウントに加えて、W&Bは [SDKとCLIのためのアイデンティティフェデレーション]({{< relref path="/guides/hosting/iam/authentication/identity_federation.md#external-service-accounts" lang="ja" >}}) を使用した **External service accounts** もサポートしています。アイデンティティプロバイダで管理され、JSON Web Tokens (JWT) を発行できるサービスアイデンティティを使用してW&Bタスクを自動化しようとしている場合は、External service accountsを使用してください。
{{% /alert %}}