---
title: 高度な IAM 設定
menu:
  default:
    identifier: ja-guides-hosting-iam-advanced_env_vars
    parent: identity-and-access-management-iam
---

基本的な [環境変数]({{< relref path="../env-vars.md" lang="ja" >}}) に加えて、[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) や [セルフマネージド]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) のインスタンスで IAM オプションを環境変数で設定できます。

IAM のニーズに応じて、インスタンスでは以下の環境変数から選んで使用してください。

| 環境変数 | 説明 |
|----------------------|-------------|
| `DISABLE_SSO_PROVISIONING` | これを `true` に設定すると、W&B インスタンスでユーザーの自動プロビジョニングを無効にします。 |
| `SESSION_LENGTH` | デフォルトのセッション有効期限を変更したい場合、この変数に希望する時間数を設定します。たとえば `SESSION_LENGTH` を `24` にすると、有効期限は 24 時間になります。デフォルト値は 720 時間です。 |
| `GORILLA_ENABLE_SSO_GROUP_CLAIMS` | OIDC ベースの SSO を使用している場合、OIDC のグループに基づいてインスタンス内の W&B Team のメンバーシップを自動化するために、この変数を `true` に設定します。ユーザーの OIDC トークンに `groups` クレームを追加してください。各要素が、そのユーザーが所属すべき W&B Team の名前である文字列配列である必要があります。配列には、そのユーザーが所属するすべての Team を含めてください。 |
| `GORILLA_LDAP_GROUP_SYNC` | LDAP ベースの SSO を使用している場合、LDAP のグループに基づいてインスタンス内の W&B Team のメンバーシップを自動化するため、`true` に設定します。 |
| `GORILLA_OIDC_CUSTOM_SCOPES` | OIDC ベースの SSO を使用している場合、W&B インスタンスがアイデンティティ プロバイダーから要求すべき追加の[スコープ](https://auth0.com/docs/get-started/apis/scopes/openid-connect-scopes)を指定できます。W&B は、これらのカスタム スコープによって SSO の機能を変更することはありません。 |
| `GORILLA_USE_IDENTIFIER_CLAIMS` | OIDC ベースの SSO を使用している場合、IdP からの特定の OIDC クレームを用いてユーザー名と氏名を強制するには、この変数を `true` に設定します。有効にした場合は、強制するユーザー名と氏名がそれぞれ `preferred_username` と `name` の OIDC クレームで提供されるように設定してください。ユーザー名に使用できるのは英数字と、特殊文字としてアンダースコアとハイフンのみです。 |
| `GORILLA_DISABLE_PERSONAL_ENTITY` | true に設定すると、[personal entities]({{< relref path="/support/kb-articles/difference_team_entity_user_entity_mean_me.md" lang="ja" >}}) をオフにします。personal Entities 内で新しい personal Projects を作成できなくなり、既存の personal Projects への書き込みも防止します。 |
| `GORILLA_DISABLE_ADMIN_TEAM_ACCESS` | これを `true` に設定すると、Organization またはインスタンスの管理者が、自分で W&B Team に参加したり自分自身を追加したりすることを制限します。これにより、Teams 内の Projects には Data & AI のペルソナのみがアクセスできます。 |
| `WANDB_IDENTITY_TOKEN_FILE`        | [identity federation]({{< relref path="/guides/hosting/iam/authentication/identity_federation.md" lang="ja" >}}) の場合、Java Web Token (JWT) が保存されるローカル ディレクトリーへの絶対パスです。 |

{{% alert color="secondary" %}}
W&B は、`GORILLA_DISABLE_ADMIN_TEAM_ACCESS` のようないくつかの設定を有効化する前に、十分に注意し、影響を理解することを推奨します。ご不明な点は W&B チームまでお問い合わせください。
{{% /alert %}}