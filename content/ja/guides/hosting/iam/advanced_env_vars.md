---
menu:
  default:
    identifier: ja-guides-hosting-iam-advanced_env_vars
    parent: identity-and-access-management-iam
title: Advanced IAM configuration
---

In addition to basic [environment variables]({{< relref path="../env-vars.md" lang="ja" >}}), you can use environment variables to configure IAM options for your [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) or [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) instance.

Choose any of the following environment variables for your instance depending on your IAM needs.

| Environment variable | Description |
|----------------------|-------------|
| `DISABLE_SSO_PROVISIONING` | Set this to `true` to turn off user auto-provisioning in your W&B instance. |
| `SESSION_LENGTH` | If you would like to change the default user session expiry time, set this variable to the desired number of hours. For example, set SESSION_LENGTH to `24` to configure session expiry time to 24 hours. The default value is 720 hours. |
| `GORILLA_ENABLE_SSO_GROUP_CLAIMS` | If you are using OIDC based SSO, set this variable to `true` to automate W&B team membership in your instance based on your OIDC groups. Add a `groups` claim to user OIDC token. It should be a string array where each entry is the name of a W&B team that the user should belong to. The array should include all the teams that a user is a part of. |
| `GORILLA_LDAP_GROUP_SYNC` | If you are using LDAP based SSO, set it to `true` to automate W&B team membership in your instance based on your LDAP groups. |
| `GORILLA_OIDC_CUSTOM_SCOPES` | If you are using OIDC based SSO, you can specify additional [scopes](https://auth0.com/docs/get-started/apis/scopes/openid-connect-scopes) that W&B instance should request from your identity provider. W&B does not change the SSO functionality due to these custom scopes in any way. |
| `GORILLA_USE_IDENTIFIER_CLAIMS` | If you are using OIDC based SSO, set this variable to `true` to enforce username and full name of your users using specific OIDC claims from your identity provider. If set, ensure that you configure the enforced username and full name in the `preferred_username` and `name` OIDC claims respectively. Usernames can only contain alphanumeric characters along with underscores and hyphens as special characters. |
| `GORILLA_DISABLE_PERSONAL_ENTITY` | When set to true, turns off [personal entities]({{< relref path="/support/kb-articles/difference_team_entity_user_entity_mean_me.md" lang="ja" >}}). Prevents creation of new personal projects in their personal entities and prevents writing to existing personal projects.
| `GORILLA_DISABLE_ADMIN_TEAM_ACCESS` | Set this to `true` to restrict Organization or Instance Admins from self-joining or adding themselves to a W&B team, thus ensuring that only Data & AI personas have access to the projects within the teams. |
| `WANDB_IDENTITY_TOKEN_FILE`        | For [identity federation]({{< relref path="/guides/hosting/iam/authentication/identity_federation.md" lang="ja" >}}), the absolute path to the local directory where Java Web Tokens (JWTs) are stored. |

{{% alert color="secondary" %}}
W&B advises to exercise caution and understand all implications before enabling some of these settings, like `GORILLA_DISABLE_ADMIN_TEAM_ACCESS`. Reach out to your W&B team for any questions.
{{% /alert %}}