---
title: 高度な IAM 設定
menu:
  default:
    identifier: ja-guides-hosting-iam-advanced_env_vars
    parent: identity-and-access-management-iam
---

基本的な[環境変数]({{< relref path="../env-vars.md" lang="ja" >}})に加えて、環境変数を使用して、[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}})または[自己管理]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}})インスタンスの IAM オプションを設定できます。

お使いのインスタンスにおける IAM のニーズに応じて、以下の環境変数のいずれかを選んでください。

| Environment variable | 説明 |
|----------------------|-----|
| DISABLE_SSO_PROVISIONING | W&B インスタンスでのユーザー自動プロビジョニングを無効にするには、これを `true` に設定します。 |
| SESSION_LENGTH | デフォルトのユーザーセッション有効期限を変更したい場合は、この変数を希望する時間数に設定します。例えば、SESSION_LENGTH を `24` に設定すると、セッション有効期限が 24 時間に設定されます。デフォルト値は 720 時間です。 |
| GORILLA_ENABLE_SSO_GROUP_CLAIMS | OIDC ベースの SSO を使用している場合、この変数を `true` に設定すると、OIDC グループに基づいて W&B チームメンバーシップが自動化されます。ユーザー OIDC トークンに `groups` クレームを追加してください。これは、ユーザーが所属するべき W&B チームの名前をそれぞれのエントリーとして含む文字列配列であるべきです。配列には、ユーザーが所属するすべてのチームを含める必要があります。 |
| GORILLA_LDAP_GROUP_SYNC | LDAP ベースの SSO を使用している場合、これを `true` に設定すると、LDAP グループに基づいて W&B チームメンバーシップが自動化されます。 |
| GORILLA_OIDC_CUSTOM_SCOPES | OIDC ベースの SSO を使用している場合、W&B インスタンスがアイデンティティプロバイダーから要求する追加の[スコープ](https://auth0.com/docs/get-started/apis/scopes/openid-connect-scopes)を指定できます。W&B は、これらのカスタムスコープによって SSO 機能をいかなる形でも変更しません。 |
| GORILLA_USE_IDENTIFIER_CLAIMS | OIDC ベースの SSO を使用している場合、この変数を `true` に設定すると、特定の OIDC クレームを使用してユーザーのユーザー名とフルネームを強制します。設定する場合は、`preferred_username` と `name` の OIDC クレームで強制されるユーザー名とフルネームを設定してください。ユーザー名には、英数字とアンダースコア、ハイフンの特殊文字のみを含めることができます。 |
| GORILLA_DISABLE_PERSONAL_ENTITY | W&B インスタンスでの個人用ユーザープロジェクトを無効にするには、これを `true` に設定します。設定すると、ユーザーは個人用 Entities 内で新しい個人用プロジェクトを作成できなくなり、既存の個人用プロジェクトへの書き込みも無効になります。 |
| GORILLA_DISABLE_ADMIN_TEAM_ACCESS | これを `true` に設定すると、組織またはインスタンスの管理者が自分で W&B チームに参加したり、自分を追加したりすることを制限します。これにより、Data & AI の関係者のみがチーム内のプロジェクトにアクセスできるようになります。 |

{{% alert color="secondary" %}}
W&B は、`GORILLA_DISABLE_ADMIN_TEAM_ACCESS` などの設定を有効にする前にあらゆる影響を理解し、注意を払うことを推奨します。ご質問があれば、W&B チームにお問い合わせください。
{{% /alert %}}