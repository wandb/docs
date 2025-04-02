---
title: What is a service account, and why is it useful?
menu:
  support:
    identifier: ja-support-kb-articles-service_account_useful
support:
- administrator
toc_hide: true
type: docs
url: /support/:filename
---

サービスアカウント（エンタープライズ限定の機能）は、人間ではない、または機械の ユーザー を表します。 チーム や Projects 全体、または特定の ユーザー に固有ではない一般的なタスクを自動化できます。 チーム 内にサービスアカウントを作成し、その APIキー を使用して、その チーム 内の Projects からの読み取りと書き込みを行うことができます。

とりわけ、サービスアカウントは、定期的な再 トレーニング 、夜間のビルドなど、wandb に ログ 記録された自動化されたジョブを追跡するのに役立ちます。 必要に応じて、[環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) `WANDB_USERNAME` または `WANDB_USER_EMAIL` を使用して、これらの機械で ローンチ された Runs に ユーザー 名を関連付けることができます。

詳細については、[Team Service Account Behavior]({{< relref path="/guides/models/app/settings-page/teams.md#team-service-account-behavior" lang="ja" >}})を参照してください。

`<WANDB_HOST_URL>/<your-team-name>/service-accounts` で、 チーム 内のサービスアカウントの APIキー を取得できます。 または、 チーム の [**Team settings（ チーム の 設定 ）**] に移動し、[**Service Accounts（サービスアカウント）**] タブを参照することもできます。

チーム の新しいサービスアカウントを作成するには:
* チーム の [**Service Accounts（サービスアカウント）**] タブにある [**+ New service account（+ 新しいサービスアカウント）**] ボタンを押します。
* [**Name（名前）**] フィールドに名前を入力します。
* 認証 方法として [**Generate API key (Built-in)（APIキー の生成 (組み込み)）**] を選択します。
* [**Create（作成）**] ボタンを押します。
* 新しく作成されたサービスアカウントの [**Copy API key（APIキー のコピー）**] ボタンをクリックし、秘密 マネージャー またはその他の安全でアクセス可能な場所に保存します。

{{% alert %}}
[**組み込み**] のサービスアカウントとは別に、W&B は [SDK および CLI の ID フェデレーション]({{< relref path="/guides/hosting/iam/authentication/identity_federation.md#external-service-accounts" lang="ja" >}}) を使用した [**外部** サービスアカウント] もサポートしています。 JSON Web Tokens (JWT) を発行できる ID プロバイダーで管理されるサービス ID を使用して W&B タスクを自動化する場合は、外部サービスアカウントを使用してください。
{{% /alert %}}
