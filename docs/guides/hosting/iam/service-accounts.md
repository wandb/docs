---
description: Manage automated or non-interactive workflows using org and team scoped service accounts
displayed_sidebar: default
title: Use service accounts to automate workflows
---

A service account represents a non-human or machine user that can automate common tasks across projects within a team or across teams. 

- An org admin can create a service account at the scope of the organization.
- A team admin can create a service account at the scope of that team.
	
A service account's API key allows the caller to read from or write to projects within the service account's scope.

Service accounts allow for centralized management of workflows by multiple users or teams, to automate experiment tracking for W&B Models or to log traces for W&B Weave. You have the option to associate a human user's identity with a workflow managed by a service account, by using the [environment variables](../../track/environment-variables.md) `WANDB_USERNAME` or `WANDB_USER_EMAIL`.

:::info
Service accounts are available on [Dedicated Cloud](../hosting-options/dedicated_cloud.md), [Self-managed instances](../hosting-options/self-managed.md) with an enterprise license, and enterprise accounts in [SaaS Cloud](../hosting-options/saas_cloud.md).
:::

## Organization-scoped service accounts

Service accounts scoped to an organization have permissions to read and write in all projects in the organization, regardless of the team, with the exception of [restricted projects](./restricted-projects.md#visibility-scopes). Before an organization-scoped service account can access a restricted project, an admin of that project must explicitly add the service account to the project.

An organization admin can obtain the API key for an organization-scoped service account from the **Service Accounts** tab of the organization or account dashboard.

To create a new organization-scoped service account:

* Press the **+ New service account** button in the **Service Accounts** tab of your organization dashboard.
* Enter a **Name**.
* Select a default team for the service account.
* Click **Create**.
* Next to the newly created service account, click **Copy API key**.
* Store the copied API key in a secret manager or another secure but accessible location.

:::info
An organization-scoped service account requires a default team, even though it has access to non-restricted projects owned by all teams within the organization. This helps to prevent a workload from failing if the `WANDB_ENTITY` variable is not set in the environment for your model training or generative AI app. To use an organization-scoped service account for a project in a different team, ensure that you configure the `WANDB_ENTITY` environment variable to that team.
:::

## Team-scoped service accounts

A team-scoped service account can read and write in all projects within its team, except to [restricted projects](./restricted-projects.md#visibility-scopes) in that team. Before a team-scoped service account can access a restricted project, an admin of that project must explicitly add the service account to the project.

As a team admin, you can get the API key for a team-scoped service account in your team at `<WANDB_HOST_URL>/<your-team-name>/service-accounts`. Alternatively you can go to the **Team settings** for your team and then refer to the **Service Accounts** tab.

To create a new team scoped service account for your team:

* Press the **+ New service account** button in the **Service Accounts** tab of your team.
* Enter a **Name**.
* Select **Generate API key (Built-in)** as the authentication method.
* Click **Create**.
* Next to the newly created service account, click **Copy API key**.
* Store the copied API key in a secret manager or another secure but accessible location.

If you do not configure a team in your model training or generative AI app environment that uses a team-scoped service account, the model runs or weave traces log to the named project within the service account's parent team. In such a scenario, user attribution using the `WANDB_USERNAME` or `WANDB_USER_EMAIL` variables _do not work_ unless the referenced user is part of the service account's parent team.

:::warning
A team-scoped service account cannot log runs to a [team or restricted-scoped project](./restricted-projects.md#visibility-scopes) in a team different from its parent team, but it can log runs to an open visibility project within another team.
:::

### External service accounts

In addition to **Built-in** service accounts, W&B also supports team-scoped **External service accounts** with the W&B SDK and CLI using [Identity federation](./identity_federation.md#external-service-accounts) with identity providers (IdPs) that can issue JSON Web Tokens (JWTs).