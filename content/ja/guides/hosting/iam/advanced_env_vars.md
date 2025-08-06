---
title: 高度な IAM 設定
menu:
  default:
    identifier: ja-guides-hosting-iam-advanced_env_vars
    parent: identity-and-access-management-iam
---

基本的な[環境変数]({{< relref path="../env-vars.md" lang="ja" >}})に加えて、[Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) や [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) インスタンスの IAM オプションを設定するためにも環境変数を利用できます。

IAM の要件に応じて、下記の環境変数から必要なものを選んでインスタンスに設定してください。

| 環境変数 | 説明 |
|----------------------|-------------|
| `DISABLE_SSO_PROVISIONING` | W&B インスタンスでユーザーの自動プロビジョニングを無効化する場合は `true` に設定してください。 |
| `SESSION_LENGTH` | デフォルトのユーザーセッション有効期限を変更したい場合、この変数に希望する時間（単位：時間）を設定してください。例えば、`SESSION_LENGTH` を `24` に設定するとセッションの有効期限が 24 時間になります。デフォルト値は 720 時間です。 |
| `GORILLA_ENABLE_SSO_GROUP_CLAIMS` | OIDC ベースの SSO を利用している場合、この変数を `true` に設定すると、OIDC グループに基づいて W&B チームのメンバーシップが自動で管理されます。ユーザーの OIDC トークンに `groups` クレーム（配列、各要素がユーザーが所属する W&B チーム名の文字列）を追加してください。この配列にはユーザーが所属するすべてのチームを含める必要があります。 |
| `GORILLA_LDAP_GROUP_SYNC` | LDAP ベースの SSO を利用している場合、この変数を `true` に設定すると、LDAP グループに基づき W&B チームのメンバーシップを自動で管理します。 |
| `GORILLA_OIDC_CUSTOM_SCOPES` | OIDC ベースの SSO を利用している場合、W&B インスタンスがアイデンティティプロバイダーにリクエストする追加の[スコープ](https://auth0.com/docs/get-started/apis/scopes/openid-connect-scopes)を指定できます。W&B の SSO 機能自体にこれらカスタムスコープが影響を及ぼすことはありません。 |
| `GORILLA_USE_IDENTIFIER_CLAIMS` | OIDC ベースの SSO を利用している場合、この変数を `true` に設定すると、アイデンティティプロバイダーから特定の OIDC クレームを用いて、ユーザー名およびフルネームを強制的に割り当てます。設定する場合は `preferred_username` および `name` OIDC クレームをそれぞれ対応する値に設定する必要があります。ユーザー名には英数字、アンダースコア、ハイフンが使用可能です。 |
| `GORILLA_DISABLE_PERSONAL_ENTITY` | true を設定すると[パーソナルエンティティ]({{< relref path="/support/kb-articles/difference_team_entity_user_entity_mean_me.md" lang="ja" >}})が無効化されます。パーソナルエンティティ内で新規のパーソナルプロジェクトの作成や既存プロジェクトへの書き込みができなくなります。 |
| `GORILLA_DISABLE_ADMIN_TEAM_ACCESS` | 組織やインスタンスの管理者による W&B チームへの自己追加や自分自身の所属設定を制限したい場合、`true` に設定してください。これにより Data & AI 関連の担当者のみチーム内プロジェクトへアクセスできるようになります。|
| `WANDB_IDENTITY_TOKEN_FILE`        | [アイデンティティフェデレーション]({{< relref path="/guides/hosting/iam/authentication/identity_federation.md" lang="ja" >}})用に、Java Web Token (JWT) を格納するローカルディレクトリーへの絶対パスを指定します。 |

{{% alert color="secondary" %}}
W&B では、`GORILLA_DISABLE_ADMIN_TEAM_ACCESS` など一部の設定を有効にする前に、十分な注意と影響の理解を推奨しています。不明点があれば W&B チームまでご相談ください。
{{% /alert %}}