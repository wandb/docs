---
title: "What is a service account, and why is it useful?"
toc_hide: true
type: docs
tags:
   - administrator
---


A service account (Enterprise-only feature) represents a non-human or machine user, which can automate common tasks across teams and projects or ones that are not specific to a particular human user. You can create a service account within a team and use its API key to read from and write to projects within that team.

Among other things, service accounts are useful for tracking automated jobs logged to wandb, like periodic retraining, nightly builds, and so on. If you'd like, you can associate a username with one of these machine-launched runs with the [environment variables](../guides/track/environment-variables.md) `WANDB_USERNAME` or `WANDB_USER_EMAIL`.



Refer to [Team Service Account Behavior](../guides/app/features/teams.md#team-service-account-behavior) for more information.

You can get the API key for a service account in your team at `<WANDB_HOST_URL>/<your-team-name>/service-accounts`. Alternatively you can go to the **Team settings** for your team and then refer to the **Service Accounts** tab. 

To create a new service account for your team:
* Press the **+ New service account** button in the **Service Accounts** tab of your team
* Provide a name in the **Name** field
* Select **Generate API key (Built-in)** as the authentication method
* Press the **Create** button
* Click the **Copy API key** button for the newly created service account and store it in a secret manager or another safe but accessible location

{{% alert %}}
Apart from the **Built-in** service accounts, W&B also supports **External service accounts** using [identity federation for SDK and CLI](../guides/hosting/iam/identity_federation.md#external-service-accounts). Use external service accounts if you are looking to automate W&B tasks using service identities managed in your identity provider that can issue JSON Web Tokens (JWT).
{{% /alert %}}