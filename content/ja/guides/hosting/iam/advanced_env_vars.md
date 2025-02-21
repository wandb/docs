---
title: Advanced IAM configuration
menu:
  default:
    identifier: ja-guides-hosting-iam-advanced_env_vars
    parent: identity-and-access-management-iam
---

基本的な[環境変数]({{< relref path="../env-vars.md" lang="ja" >}})に加えて、環境変数を使用すると、[Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}})または[Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}})インスタンスの IAM オプションを構成できます。

IAM のニーズに応じて、インスタンスに次のいずれかの環境変数を選択してください。

| 環境変数 | 説明 |
|----------------------|-------------|
| DISABLE_SSO_PROVISIONING | これを `true` に設定すると、W&B インスタンスでの ユーザー の自動プロビジョニングが無効になります。 |
| SESSION_LENGTH | デフォルトの ユーザー セッションの有効期限を変更する場合は、この変数を目的の時間数に設定します。たとえば、SESSION_LENGTH を `24` に設定すると、セッションの有効期限が 24 時間に構成されます。デフォルト 値 は 720 時間です。 |
| GORILLA_ENABLE_SSO_GROUP_CLAIMS | OIDC ベースの SSO を使用している場合は、この変数を `true` に設定すると、OIDC グループに基づいてインスタンス内の W&B Team メンバーシップが自動化されます。`groups` クレームを ユーザー OIDC トークンに追加します。これは、各エントリが ユーザー が所属する W&B Team の名前である文字列配列である必要があります。配列には、 ユーザー が所属するすべての Team が含まれている必要があります。 |
| GORILLA_LDAP_GROUP_SYNC | LDAP ベースの SSO を使用している場合は、これを `true` に設定すると、LDAP グループに基づいてインスタンス内の W&B Team メンバーシップが自動化されます。 |
| GORILLA_OIDC_CUSTOM_SCOPES | OIDC ベースの SSO を使用している場合、W&B インスタンスが ID プロバイダーに要求する必要がある追加の[スコープ](https://auth0.com/docs/get-started/apis/scopes/openid-connect-scopes)を指定できます。W&B は、これらのカスタム スコープによって SSO 機能を変更することはありません。 |
| GORILLA_USE_IDENTIFIER_CLAIMS | OIDC ベースの SSO を使用している場合は、この変数を `true` に設定して、ID プロバイダーからの特定の OIDC クレームを使用して ユーザー の ユーザー 名とフル ネームを強制します。設定する場合は、`preferred_username` および `name` OIDC クレームで、強制される ユーザー 名とフル ネームを構成していることを確認してください。ユーザー 名には、英数字と、特殊文字としてアンダースコアとハイフンのみを含めることができます。 |
| GORILLA_DISABLE_PERSONAL_ENTITY | これを `true` に設定すると、W&B インスタンスで個人の ユーザー Projects がオフになります。設定した場合、 ユーザー は個人の Entities で新しい個人の Projects を作成できなくなり、既存の個人の Projects への書き込みもオフになります。 |
| GORILLA_DISABLE_ADMIN_TEAM_ACCESS | これを `true` に設定すると、Organization または Instance の管理者が W&B Team に自己参加または追加することを制限し、Data & AI ペルソナのみが Team 内の Projects に アクセス できるようにします。 |

{{% alert color="secondary" %}}
W&B では、`GORILLA_DISABLE_ADMIN_TEAM_ACCESS` などの一部の 設定 を有効にする前に、注意してすべての影響を理解することをお勧めします。ご不明な点がございましたら、W&B Team までお問い合わせください。
{{% /alert %}}
