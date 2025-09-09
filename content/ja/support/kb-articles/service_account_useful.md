---
menu:
  support:
    identifier: ja-support-kb-articles-service_account_useful
support:
- administrator
title: What is a service account, and why is it useful?
toc_hide: true
type: docs
url: /support/:filename
---

A **service account** represents a non-human or machine identity, which can automate common tasks across teams and projects. Service accounts are ideal for CI/CD pipelines, automated training jobs, and other machine-to-machine workflows.

{{< readfile file="/content/en/_includes/service-account-benefits.md" >}}

Among other things, service accounts are useful for tracking automated jobs logged to wandb, like periodic retraining, nightly builds, and so on. If you'd like, you can associate a username with one of these machine-launched runs with the [environment variables]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) `WANDB_USERNAME` or `WANDB_USER_EMAIL`.

For comprehensive information about service accounts, including best practices and detailed setup instructions, refer to [Use service accounts to automate workflows]({{< relref path="/guides/hosting/iam/authentication/service-accounts.md" lang="ja" >}}). For information about how service accounts behave in team contexts, refer to [Team Service Account Behavior]({{< relref path="/guides/models/app/settings-page/teams.md#team-service-account-behavior" lang="ja" >}}).

You can get the API key for a service account in your team at `<WANDB_HOST_URL>/<your-team-name>/service-accounts`. Alternatively you can go to the **Team settings** for your team and then refer to the **Service Accounts** tab. 

To create a new service account for your team:
* Press the **+ New service account** button in the **Service Accounts** tab of your team
* Provide a name in the **Name** field
* Select **Generate API key (Built-in)** as the authentication method
* Press the **Create** button
* Click the **Copy API key** button for the newly created service account and store it in a secret manager or another safe but accessible location

{{% alert %}}
Apart from the **Built-in** service accounts, W&B also supports **External service accounts** using [identity federation for SDK and CLI]({{< relref path="/guides/hosting/iam/authentication/identity_federation.md#external-service-accounts" lang="ja" >}}). Use external service accounts if you are looking to automate W&B tasks using service identities managed in your identity provider that can issue JSON Web Tokens (JWT).
{{% /alert %}}