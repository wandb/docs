---
title: 高度な IAM 設定
menu:
  default:
    identifier: advanced_env_vars
    parent: identity-and-access-management-iam
---

基本的な[環境変数]({{< relref "../env-vars.md" >}})に加えて、[Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}}) または [Self-managed]({{< relref "/guides/hosting/hosting-options/self-managed.md" >}}) インスタンスの IAM オプションを設定するためにも環境変数を利用できます。

ご利用の IAM ニーズに応じて、以下のいずれかの環境変数をインスタンスで設定してください。

| 環境変数 | 説明 |
|----------------------|-------------|
| `DISABLE_SSO_PROVISIONING` | これを `true` に設定すると、W&B インスタンスでのユーザー自動プロビジョニングを無効化します。|
| `SESSION_LENGTH` | デフォルトのユーザーセッション有効期限を変更したい場合、この変数に希望する時間（時間単位）を設定します。たとえば `SESSION_LENGTH` を `24` に設定すると、セッション有効期限が 24 時間になります。デフォルト値は 720 時間です。|
| `GORILLA_ENABLE_SSO_GROUP_CLAIMS` | OIDC ベースの SSO を利用している場合、この変数を `true` に設定すると、OIDC グループに基づいて W&B のチームメンバーシップを自動化できます。ユーザー OIDC トークンに `groups` クレームを追加してください。これは文字列の配列で、ユーザーが所属する W&B チーム名がそれぞれ格納されている必要があります。配列にはユーザーが所属するすべてのチームが含まれる必要があります。|
| `GORILLA_LDAP_GROUP_SYNC` | LDAP を使った SSO を利用している場合、これを `true` に設定すると LDAP グループに基づいて W&B のチームメンバーシップを自動化できます。|
| `GORILLA_OIDC_CUSTOM_SCOPES` | OIDC ベースの SSO をご利用の場合、W&B インスタンスがアイデンティティプロバイダーからリクエストする追加の [スコープ](https://auth0.com/docs/get-started/apis/scopes/openid-connect-scopes) を指定できます。これらのカスタムスコープ設定によって SSO の基本動作が変更されることはありません。|
| `GORILLA_USE_IDENTIFIER_CLAIMS` | OIDC ベースの SSO を利用している場合、この変数を `true` に設定すると、アイデンティティプロバイダーからの特定 OIDC クレームでユーザー名およびフルネームを強制できます。設定する場合は、ユーザー名とフルネームをそれぞれ `preferred_username` および `name` OIDC クレームに設定してください。ユーザー名には英数字とアンダースコア、ハイフンのみ使用可能です。|
| `GORILLA_DISABLE_PERSONAL_ENTITY` | true に設定すると、[personal entities]({{< relref "/support/kb-articles/difference_team_entity_user_entity_mean_me.md" >}}) を無効化します。これにより、ユーザーが personal entities 内の新しい personal projects を作成できず、既存の personal projects への書き込みもできなくなります。|
| `GORILLA_DISABLE_ADMIN_TEAM_ACCESS` | これを `true` に設定すると、Organization や Instance の管理者が自身でチームに参加したり、自分を W&B チームに追加することができなくなり、Data & AI 関連の担当者のみがチーム内のプロジェクトへアクセスできるようになります。|
| `WANDB_IDENTITY_TOKEN_FILE`        | [identity federation]({{< relref "/guides/hosting/iam/authentication/identity_federation.md" >}}) を利用する場合、Java Web Token (JWT) を保存するローカルディレクトリーの絶対パスを指定します。|

{{% alert color="secondary" %}}
`GORILLA_DISABLE_ADMIN_TEAM_ACCESS` など、これらの設定を有効にする際は影響をよくご理解のうえ慎重にご対応ください。ご不明な点は W&B チームまでお問い合わせください。
{{% /alert %}}