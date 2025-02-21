---
title: Advanced IAM configuration
menu:
  default:
    identifier: ja-guides-hosting-iam-advanced_env_vars
    parent: identity-and-access-management-iam
---

基本的な[環境変数]({{< relref path="../env-vars.md" lang="ja" >}})に加えて、環境変数を使用して [専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}})や[セルフマネージド]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}})インスタンスの IAM オプションを設定できます。

IAM のニーズに応じて、インスタンス用に次の環境変数から選択してください。

| 環境変数 | 説明 |
|----------------------|-------------|
| DISABLE_SSO_PROVISIONING | W&B インスタンスでユーザーの自動プロビジョニングを無効にするには、これを `true` に設定します。 |
| SESSION_LENGTH | デフォルトのユーザーセッションの有効期限を変更したい場合は、この変数を希望の時間数に設定します。たとえば、SESSION_LENGTH を `24` に設定すると、セッションの有効期限は 24 時間になります。デフォルト値は 720 時間です。 |
| GORILLA_ENABLE_SSO_GROUP_CLAIMS | OIDC ベースの SSO を使用している場合、これを `true` に設定すると、OIDC グループに基づいて W&B チームのメンバーシップがインスタンス内で自動化されます。ユーザーの OIDC トークンに `groups` クレームを追加します。これは、ユーザーが所属するべき W&B チームの名前が各項目となる文字列配列でなければなりません。この配列には、ユーザーが所属するすべてのチームを含める必要があります。 |
| GORILLA_LDAP_GROUP_SYNC | LDAP ベースの SSO を使用している場合、LDAP グループに基づいて W&B チームのメンバーシップをインスタンス内で自動化するには、これを `true` に設定します。 |
| GORILLA_OIDC_CUSTOM_SCOPES | OIDC ベースの SSO を使用している場合、W&B インスタンスがアイデンティティ・プロバイダーから要求する追加の[スコープ](https://auth0.com/docs/get-started/apis/scopes/openid-connect-scopes)を指定できます。W&B は、これらのカスタムスコープによって SSO の機能を変更することはありません。 |
| GORILLA_USE_IDENTIFIER_CLAIMS | OIDC ベースの SSO を利用している場合、ユーザー名とフルネームの特定の OIDC クレームを使用して、ユーザーの名前とフルネームを強制するために、この変数を `true` に設定します。設定する場合は、強制されるユーザー名とフルネームをそれぞれ `preferred_username` と `name` の OIDC クレームに設定してください。ユーザー名には、特殊文字としてアンダースコアとハイフンを含む英数字のみを使用できます。 |
| GORILLA_DISABLE_PERSONAL_ENTITY | W&B インスタンスで個人ユーザーのプロジェクトを無効にするには、これを `true` に設定します。設定されている場合、ユーザーは個人エンティティ内で新しい個人プロジェクトを作成できず、既存の個人プロジェクトへの書き込みも無効になります。 |
| GORILLA_DISABLE_ADMIN_TEAM_ACCESS | 組織またはインスタンスの管理者が W&B チームへの自己参加や自身の追加を制限するには、これを `true` に設定し、データ & AI 担当者だけがチーム内のプロジェクトにアクセスできるようにします。 |

{{% alert color="secondary" %}}
W&B は、一部の設定（例えば `GORILLA_DISABLE_ADMIN_TEAM_ACCESS`）を有効にする前に注意を払い、そのすべての影響を理解するようにアドバイスします。ご質問がある場合は、W&B チームにお問い合わせください。
{{% /alert %}}