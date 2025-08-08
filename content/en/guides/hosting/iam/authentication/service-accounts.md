---
description: Manage automated or non-interactive workflows using org and team scoped service accounts
displayed_sidebar: default
title: Use service accounts to automate workflows
---

A service account represents a non-human or machine user that can automatically perform common tasks across projects within a team or across teams. Service accounts are ideal for CI/CD pipelines, automated training jobs, and other machine-to-machine workflows.

## Key benefits

{{< readfile file="/content/en/_includes/service-account-benefits.md" >}}

## Overview

Service accounts provide a secure way to automate W&B workflows without using personal user credentials or hard-coded credentials. They can be created at two scopes:

- **Organization-scoped**: Created by org admins, with access across all teams (not available in W&B Multi-tenant Cloud)
- **Team-scoped**: Created by team admins, with access limited to a specific team
	
A service account's API key allows the caller to read from or write to projects within the service account's scope. This enables centralized management of automated workflows for experiment tracking in W&B Models or logging traces in W&B Weave.

Service accounts are particularly useful for:
- **CI/CD pipelines**: Automatically log model training runs from GitHub Actions, GitLab CI, or Jenkins
- **Scheduled jobs**: Nightly model retraining, periodic evaluation runs, or data validation workflows
- **Production monitoring**: Log inference metrics and model performance from production systems
- **Jupyter notebooks**: Shared notebooks in JupyterHub or Google Colab environments
- **Kubernetes jobs**: Automated workflows running in K8s clusters
- **Airflow/Prefect/Dagster**: ML pipeline orchestration tools

{{% alert %}}
Service accounts are available on [Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}}), [Self-managed instances]({{< relref "/guides/hosting/hosting-options/self-managed.md" >}}) with an enterprise license, and enterprise accounts in [SaaS Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}).
{{% /alert %}}

## Organization-scoped service accounts

Service accounts scoped to an organization have permissions to read and write in all projects in the organization, regardless of the team, with the exception of [restricted projects]({{< relref "../access-management/restricted-projects.md#visibility-scopes" >}}). Before an organization-scoped service account can access a restricted project, an admin of that project must explicitly add the service account to the project.

An organization admin can obtain the API key for an organization-scoped service account from the **Service Accounts** tab of the organization or account dashboard.

To create a new organization-scoped service account:

* Click **New service account** button in the **Service Accounts** tab of your organization dashboard.
* Enter a **Name**.
* Select a default team for the service account.
* Click **Create**.
* Next to the newly created service account, click **Copy API key**.
* Store the copied API key in a secret manager or another secure but accessible location.

{{% alert %}}
An organization-scoped service account requires a default team, even though it has access to non-restricted projects owned by all teams within the organization. This helps to prevent a workload from failing if the `WANDB_ENTITY` variable is not set in the environment for your model training or generative AI app. To use an organization-scoped service account for a project in a different team, you must set the `WANDB_ENTITY` environment variable to that team.
{{% /alert %}}

## Team-scoped service accounts

A team-scoped service account can read and write in all projects within its team, except to [restricted projects]({{< relref "../access-management/restricted-projects.md#visibility-scopes" >}}) in that team. Before a team-scoped service account can access a restricted project, an admin of that project must explicitly add the service account to the project.

As a team admin, you can get the API key for a team-scoped service account in your team at `<WANDB_HOST_URL>/<your-team-name>/service-accounts`. Alternatively you can go to the **Team settings** for your team and then refer to the **Service Accounts** tab.

To create a new team scoped service account for your team:

* Click **New service account** button in the **Service Accounts** tab of your team.
* Enter a **Name**.
* Select **Generate API key (Built-in)** as the authentication method.
* Click **Create**.
* Next to the newly created service account, click **Copy API key**.
* Store the copied API key in a secret manager or another secure but accessible location.

If you do not configure a team in your model training or generative AI app environment that uses a team-scoped service account, the model runs or weave traces log to the named project within the service account's parent team. In such a scenario, user attribution using the `WANDB_USERNAME` or `WANDB_USER_EMAIL` variables _do not work_ unless the referenced user is part of the service account's parent team.

{{% alert color="warning" %}}
A team-scoped service account cannot log runs to a [team or restricted-scoped project]({{< relref "../access-management/restricted-projects.md#visibility-scopes" >}}) in a team different from its parent team, but it can log runs to an open visibility project within another team.
{{% /alert %}}

### External service accounts

In addition to built-in service accounts, W&B also supports team-scoped external service accounts with the W&B SDK and CLI using [Identity federation]({{< relref "./identity_federation.md#external-service-accounts" >}}) with identity providers (IdPs) that can issue JSON Web Tokens (JWTs).

## Best practices

Follow these recommendations to ensure secure and efficient use of service accounts in your organization:

- **Use a secrets manager**: Store service account API keys in a secure secrets management system (e.g., AWS Secrets Manager, HashiCorp Vault, Azure Key Vault) rather than in plain text configuration files.

- **Principle of least privilege**: Create team-scoped service accounts when possible, rather than organization-scoped accounts, to limit access to only necessary projects.

- **Unique service accounts per use case**: Create separate service accounts for different automation workflows (e.g., one for CI/CD, another for scheduled retraining) to improve auditability and enable granular access control.

- **Regular audits**: Periodically review active service accounts and remove those no longer in use. Check the audit logs to monitor service account activity.

- **Secure API key handling**: 
  - Never commit API keys to version control
  - Use environment variables to pass keys to applications
  - Rotate keys if they are accidentally exposed

- **Naming conventions**: Use descriptive names that indicate the service account's purpose:
  - Good: `ci-model-training`, `nightly-eval-pipeline`, `prod-inference-monitor`
  - Avoid: `service-account-1`, `test-sa`, `temp`

- **User attribution**: When multiple team members use the same automation workflow, set `WANDB_USERNAME` or `WANDB_USER_EMAIL` to track who triggered each run:
  ```bash
  export WANDB_API_KEY="<service_account_key>"
  export WANDB_USERNAME="john.doe@company.com"
  ```

- **Environment configuration**: For team-scoped service accounts, always set the `WANDB_ENTITY` to ensure runs log to the correct team:
  ```bash
  export WANDB_ENTITY="ml-team"
  export WANDB_PROJECT="production-models"
  ```

- **Error handling**: Implement proper error handling and alerts for failed authentication to quickly identify issues with service account credentials.

- **Documentation**: Maintain documentation of:
  - Which service accounts exist and their purposes
  - Which systems/workflows use each service account
  - Contact information for the team responsible for each account

## Troubleshooting

Common issues and solutions:

- **"Unauthorized" errors**: Verify the API key is correctly set and the service account has access to the target project
- **Runs not appearing**: Check that `WANDB_ENTITY` is set to the correct team name
- **User attribution not working**: Ensure the user specified in `WANDB_USERNAME` is a member of the team
- **Access denied to restricted projects**: Explicitly add the service account to the restricted project's access list