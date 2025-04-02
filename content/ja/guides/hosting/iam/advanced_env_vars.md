---
title: Advanced IAM configuration
menu:
  default:
    identifier: ja-guides-hosting-iam-advanced_env_vars
    parent: identity-and-access-management-iam
---

基本的な[環境変数]({{< relref path="../env-vars.md" lang="ja" >}})に加えて、環境変数を使用して、[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}})または[セルフマネージド]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}})インスタンスのIAMオプションを構成できます。

IAMのニーズに応じて、インスタンスに対して次のいずれかの環境変数を選択してください。

| 環境変数 | 説明 |
|----------------------|-------------|
| DISABLE_SSO_PROVISIONING | これを `true` に設定すると、W&Bインスタンスでの ユーザー の自動プロビジョニングがオフになります。 |
| SESSION_LENGTH | デフォルトの ユーザー セッションの有効期限を変更する場合は、この変数を目的の時間数に設定します。たとえば、SESSION_LENGTHを `24` に設定すると、セッションの有効期限が24時間に構成されます。デフォルト値は720時間です。 |
| GORILLA_ENABLE_SSO_GROUP_CLAIMS | OIDCベースのSSOを使用している場合は、この変数を `true` に設定すると、OIDCグループに基づいてインスタンス内のW&B team メンバーシップが自動化されます。 ユーザー OIDCトークンに `groups` クレームを追加します。各エントリが ユーザー が所属するW&B teamの名前である文字列配列である必要があります。配列には、 ユーザー が所属するすべての team が含まれている必要があります。 |
| GORILLA_LDAP_GROUP_SYNC | LDAPベースのSSOを使用している場合は、`true` に設定すると、LDAPグループに基づいてインスタンス内のW&B team メンバーシップが自動化されます。 |
| GORILLA_OIDC_CUSTOM_SCOPES | OIDCベースのSSOを使用している場合は、W&BインスタンスがIDプロバイダーに要求する必要がある追加の[スコープ](https://auth0.com/docs/get-started/apis/scopes/openid-connect-scopes)を指定できます。W&Bは、これらのカスタムスコープによってSSO機能を変更することはありません。 |
| GORILLA_USE_IDENTIFIER_CLAIMS | OIDCベースのSSOを使用している場合は、この変数を `true` に設定して、IDプロバイダーからの特定のOIDCクレームを使用して ユーザー の ユーザー 名とフルネームを強制します。設定する場合は、`preferred_username` および `name` OIDCクレームで、強制される ユーザー 名とフルネームを構成していることを確認してください。 ユーザー 名には、英数字と、特殊文字としてアンダースコアとハイフンのみを含めることができます。 |
| GORILLA_DISABLE_PERSONAL_ENTITY | これを `true` に設定すると、W&Bインスタンス内の個人の ユーザー project がオフになります。設定すると、 ユーザー は個人の entity に新しい個人 project を作成できなくなり、既存の個人 project への書き込みもオフになります。 |
| GORILLA_DISABLE_ADMIN_TEAM_ACCESS | これを `true` に設定すると、組織またはインスタンス管理者がW&B teamに自己参加または自己追加することを制限し、Data ＆ AIペルソナのみが team 内の project にアクセスできるようにします。 |

{{% alert color="secondary" %}}
W&Bは、`GORILLA_DISABLE_ADMIN_TEAM_ACCESS` など、これらの 設定 の一部を有効にする前に、注意を払い、すべての影響を理解することをお勧めします。ご不明な点がございましたら、W&B teamにお問い合わせください。
{{% /alert %}}
