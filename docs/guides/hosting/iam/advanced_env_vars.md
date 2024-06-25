---
displayed_sidebar: default
---


# Advanced configuration

基本的な[環境変数](../env-vars.md)に加えて、[Dedicated Cloud](../hosting-options/dedicated_cloud.md)や[Self-managed](../hosting-options/self-managed.md)インスタンスのIAMオプションを設定するために環境変数を使用できます。

IAMのニーズに応じて、以下の環境変数をインスタンスに選択してください。

| Environment variable | Description |
|----------------------|-------------|
| DISABLE_SSO_PROVISIONING | この値を `true` に設定すると、W&Bインスタンスでのユーザー自動プロビジョニングが無効になります。 |
| GORILLA_ENABLE_SSO_GROUP_CLAIMS | OIDCベースのSSOを使用している場合、この変数を `true` に設定すると、OIDCグループに基づいてインスタンス内でW&Bチームメンバーシップを自動化できます。ユーザーのOIDCトークンに `groups` クレームを追加します。これは、ユーザーが所属するW&Bチームの名前がそれぞれエントリに含まれる文字列配列である必要があります。配列には、ユーザーが所属する全てのチームを含める必要があります。 |
| GORILLA_LDAP_GROUP_SYNC | LDAPベースのSSOを使用している場合、これを `true` に設定すると、LDAPグループに基づいてインスタンス内でW&Bチームメンバーシップを自動化できます。 |
| GORILLA_OIDC_CUSTOM_SCOPES | OIDCベースのSSOを使用している場合、W&Bインスタンスがアイデンティティプロバイダーから要求する追加の[スコープ](https://auth0.com/docs/get-started/apis/scopes/openid-connect-scopes)を指定できます。W&BはこれらのカスタムスコープによってSSO機能を変更することはありません。 |
| GORILLA_USE_IDENTIFIER_CLAIMS | OIDCベースのSSOを使用している場合、この変数を `true` に設定すると、アイデンティティプロバイダーからの特定のOIDCクレームを使用してユーザー名とフルネームを強制します。設定する場合、`preferred_username` と `name` のOIDCクレームに強制したユーザー名とフルネームをそれぞれ設定してください。 |
| GORILLA_DISABLE_PERSONAL_ENTITY | この値を `true` に設定すると、W&Bインスタンスでの個人ユーザープロジェクトが無効になります。設定されると、ユーザーは個人のエンティティで新しい個人プロジェクトを作成できませんし、既存の個人プロジェクトへの書き込みも無効になります。 |
| GORILLA_DISABLE_ADMIN_TEAM_ACCESS | この値を `true` に設定すると、組織またはインスタンスの管理者が自己参加したり、自分をW&Bチームに追加したりすることを制限し、データとAIのペルソナのみがチーム内のプロジェクトにアクセスできるようにします。 |

:::caution
W&Bは、`GORILLA_DISABLE_ADMIN_TEAM_ACCESS` のような設定を有効にする前に、すべての影響を理解するために慎重に対処することをお勧めします。質問があればW&Bチームにお問い合わせください。
:::