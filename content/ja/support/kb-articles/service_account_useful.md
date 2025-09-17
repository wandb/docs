---
title: サービス アカウントとは何ですか、なぜ便利なのですか？
menu:
  support:
    identifier: ja-support-kb-articles-service_account_useful
support:
- 管理者
toc_hide: true
type: docs
url: /support/:filename
---

A サービス アカウント は、人ではない、つまりマシンのアイデンティティを表し、Teams や Projects をまたいだ一般的なタスクを自動化できます。サービス アカウント は、CI/CD パイプライン、自動トレーニング ジョブ、その他のマシン間ワークフローに最適です。

{{< readfile file="/content/en/_includes/service-account-benefits.md" >}}

そのほかにも、サービス アカウント は、定期的な再トレーニングやナイトリー ビルドなど、wandb に ログ される自動ジョブの追跡に役立ちます。必要に応じて、これらのマシンが起動した Runs のいずれかに、[環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) `WANDB_USERNAME` または `WANDB_USER_EMAIL` を使ってユーザー名を関連付けることができます。

ベストプラクティスや詳細なセットアップ手順を含む サービス アカウント の包括的な情報は、[サービス アカウントでワークフローを自動化する]({{< relref path="/guides/hosting/iam/authentication/service-accounts.md" lang="ja" >}}) を参照してください。チームのコンテキストにおける サービス アカウント の振る舞いについては、[Team サービス アカウントの挙動]({{< relref path="/guides/models/app/settings-page/teams.md#team-service-account-behavior" lang="ja" >}}) を参照してください。

チームの サービス アカウント 用の APIキー は `<WANDB_HOST_URL>/<your-team-name>/service-accounts` で取得できます。別の方法として、チームの **Team settings** に移動し、**Service Accounts** タブを参照してください。
To create a new service account for your team:
* チームの **Service Accounts** タブで **+ New service account** ボタンを押す
* **Name** フィールドに名前を入力する
* 認証メソッドとして **Generate API key (Built-in)** を選択する
* **Create** ボタンを押す
* 作成した サービス アカウント の **Copy API key** ボタンをクリックし、シークレット マネージャーや、その他の安全かつアクセス可能な場所に保管する

{{% alert %}}
**Built-in** な サービス アカウント に加えて、W&B は [SDK と CLI のアイデンティティ フェデレーション]({{< relref path="/guides/hosting/iam/authentication/identity_federation.md#external-service-accounts" lang="ja" >}}) を使用する **External service accounts** にも対応しています。JSON Web Token (JWT) を発行できるアイデンティティ プロバイダで管理されたサービス アイデンティティを使って W&B のタスクを自動化したい場合は、External service accounts を使用してください。
{{% /alert %}}