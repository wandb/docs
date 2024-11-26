---
description: Manage automated or non-interactive workflows using org and team scoped service accounts
displayed_sidebar: default
title: Use service accounts to automate workflows
---

A service account represents a non-human or machine user that can automate common tasks across projects within a team or across teams. As an org admin, you can create a service account at the scope of your overall organization such that it could be used across all teams. Or as a team admin, you could create a service account within the scope of your specific team. You can then use the service account's API key to read from or write to projects within the team(s).

Service accounts are an alternative to using user-specific API keys to automate experiment tracking for W&B Models or to log traces for W&B Weave. They are more useful when you're looking to centralize your workflows across multiple users or teams. You have the option to associate the username for a human user with a service account managed workflow by using either of the the [environment variables](../../track/environment-variables.md) `WANDB_USERNAME` or `WANDB_USER_EMAIL`.

:::info
Service accounts are available on [Dedicated Cloud](../hosting-options/dedicated_cloud.md), [Self-managed instances](../hosting-options/self-managed.md) with enterprise license, and enterprise accounts in [SaaS Cloud](../hosting-options/saas_cloud.md).
:::

## Organization scoped service accounts

Organization scoped service accounts have read and write permissions in all projects within all teams in that organization, except to [restricted projects](./restricted-projects.md#visibility-scopes) in any team. The admin of a restricted project must specifically add a org-scoped service account to the project in order to use it.

As a org admin, you can get the API key for a org scoped service account in the **Service Accounts** tab of your organization or account dashboard.

To create a new org scoped service account for your organization:

* Press the **+ New service account** button in the **Service Accounts** tab of your organization dashboard
* Provide a name in the **Name** field
* Select **Generate API key (Built-in)** as the authentication method
* Select a default team from the available teams
* Press the **Create** button
* Click the **Copy API key** button for the newly created service account and store it in a secret manager or another safe but accessible location

:::info
Organization scoped service accounts require a default team even though they have access to all teams within the organization. This is to ensure that W&B Models or Weave workflows do not fail if the `WANDB_ENTITY` variable is not available in your model training or generative AI application envrionment. To use a org scoped service account for a project in a team that's different from the service account's default team, ensure that you configure the relevant team using the `WANDB_ENTITY` variable in your environment.
:::

## Team scoped service accounts

Team scoped service accounts have read and write permissions in all projects within the team they have been created in, except to [restricted projects](./restricted-projects.md#visibility-scopes) in that team. The admin of a restricted project must specifically add a team-scoped service account to the project in order to use it.

As a team admin, you can get the API key for a team scoped service account in your team at `<WANDB_HOST_URL>/<your-team-name>/service-accounts`. Alternatively you can go to the **Team settings** for your team and then refer to the **Service Accounts** tab.

To create a new team scoped service account for your team:

* Press the **+ New service account** button in the **Service Accounts** tab of your team
* Provide a name in the **Name** field
* Select **Generate API key (Built-in)** as the authentication method
* Press the **Create** button
* Click the **Copy API key** button for the newly created service account and store it in a secret manager or another safe but accessible location

:::info
When you do not configure a team in your model training or generative AI application environment and use a team scoped service account, the model runs or weave traces log to the named project within the service account's parent team. In such a scenario, user attribution using the `WANDB_USERNAME` or `WANDB_USER_EMAIL` variables work only if the referenced user is part of the service account's parent team.
:::

:::warning
A team scoped service account can not log runs to a private project in a team different from its parent team, but it can log runs to [open visibility](./restricted-projects.md#visibility-scopes) projects within other teams.
:::

### External service accounts

Apart from the **Built-in** service accounts, W&B also supports team scoped **External service accounts** using [identity federation for SDK and CLI](./identity_federation.md#external-service-accounts). Use external service accounts if you are looking to automate W&B workflows using service identities managed in your identity provider that can issue JSON Web Tokens (JWT).