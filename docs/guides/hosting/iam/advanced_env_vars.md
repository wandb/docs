---
displayed_sidebar: default
---

# Advanced configuration

基本的な[環境変数](../env-vars.md)に加えて、環境変数を使用して[Dedicated Cloud](../hosting-options/dedicated_cloud.md)または[Self-managed](../hosting-options/self-managed.md)インスタンスのIAMオプションを設定することができます。

IAMのニーズに応じて、インスタンス用に次の環境変数から選択してください。

| Environment variable | Description |
|----------------------|-------------|
| DISABLE_SSO_PROVISIONING | これを `true` に設定すると、W&Bインスタンスでユーザーの自動プロビジョニングが無効になります。 |
| SESSION_LENGTH | デフォルトのユーザーセッションの有効期限を変更したい場合、この変数を希望する時間数に設定します。例えば、SESSION_LENGTHを`24`に設定すると、セッションの有効期限が24時間に設定されます。デフォルトの値は720時間です。 |
| GORILLA_ENABLE_SSO_GROUP_CLAIMS | OIDCベースのSSOを使用している場合、この変数を `true` に設定すると、OIDCグループに基づいてW&Bチームメンバーシップが自動化されます。ユーザーのOIDCトークンに `groups` クレームを追加します。これは、ユーザーが所属すべきW&Bチームの名前を各エントリとして含む文字列配列である必要があります。配列には、ユーザーが所属するすべてのチームが含まれている必要があります。 |
| GORILLA_LDAP_GROUP_SYNC | LDAPベースのSSOを使用している場合、これを `true` に設定すると、LDAPグループに基づいてW&Bチームメンバーシップが自動化されます。 |
| GORILLA_OIDC_CUSTOM_SCOPES | OIDCベースのSSOを使用している場合、W&Bインスタンスがアイデンティティプロバイダから要求する追加の[スコープ](https://auth0.com/docs/get-started/apis/scopes/openid-connect-scopes)を指定できます。W&Bは、これらのカスタムスコープによってSSOの機能を変更することはありません。 |
| GORILLA_USE_IDENTIFIER_CLAIMS | OIDCベースのSSOを使用している場合、特定のOIDCクレームを使用してユーザー名およびフルネームを強制するには、この変数を `true` に設定します。設定する場合、強制されるユーザー名およびフルネームをそれぞれ `preferred_username` と `name` のOIDCクレームに設定してください。ユーザー名には英数字とアンダースコア、およびハイフンを含む特殊文字を使用できます。 |
| GORILLA_DISABLE_PERSONAL_ENTITY | これを `true` に設定すると、W&Bインスタンスで個人ユーザープロジェクトが無効になります。設定すると、ユーザーは個人のエンティティ内で新しい個人プロジェクトを作成できなくなり、既存の個人プロジェクトへの書き込みも無効になります。 |
| GORILLA_DISABLE_ADMIN_TEAM_ACCESS | これを `true` に設定すると、組織またはインスタンス管理者が自分自身をW&Bチームに追加することや、自己参加することを制限し、データとAIのパーソナのみがチーム内のプロジェクトにアクセスできるようにします。 |

:::caution
W&Bは、`GORILLA_DISABLE_ADMIN_TEAM_ACCESS` などの設定を有効にする前に、そのすべての影響を理解し、注意を払うことを推奨します。質問がある場合は、W&Bチームにお問い合わせください。
:::