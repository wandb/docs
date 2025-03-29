---
description: Overview W&B secrets, how they work, and how to get started using them.
menu:
  default:
    identifier: secrets
    parent: core
title: Secrets
url: guides/secrets
weight: 1
---

W&B Secret Manager allows you to securely and centrally store, manage, and inject _secrets_, which are sensitive strings such as access tokens, bearer tokens, API keys, or passwords. W&B Secret Manager removes the need to add sensitive strings directly to your code or when configuring a webhook's header or payload({{< relref "/guides/core/automations/" >}}).

Secrets are stored and managed in each team's Secret Manager, in the **Team secrets** section of the [team settings]({{< relref "/guides/models/app/settings-page/team-settings/" >}}).

{{% alert %}}
* Only W&B Admins can create, edit, or delete a secret.
* Secrets are included as a core part of W&B, including in [W&B Server deployments]({{< relref "/guides/hosting/" >}}) that you host in Azure, GCP, or AWS. Connect with your W&B account team to discuss how you can use secrets in W&B if you use a different deployment type.
* In W&B Server, you are responsible for configuring security measures that satisfy your security needs. 

  - W&B strongly recommends that you store secrets in a W&B instance of a cloud provider's secrets manager provided by AWS, GCP, or Azure, which are configured with advanced security capabilities.

  - W&B recommends against using a Kubernetes cluster as the backend of your secrets store unless you are unable to use a W&B instance of a cloud secrets manager (AWS, GCP, or Azure), and you understand how to prevent security vulnerabilities that can occur if you use a cluster.
{{% /alert %}}

## Add a secret
To add a secret:

1. If the receiving service requires it to authenticate incoming webhooks, generate the required token or API key. If necessary, save the sensitive string securely, such as in a password manager.
1. Log in to W&B and go to the team's **Settings** page.
1. In the **Team Secrets** section, click **New secret**.
1. Using letters, numbers, and `_`, provide a name for the secret.
1. Paste the sensitive string into the **Secret** field.
1. Click **Add secret**.

Specify the secrets you want to use for your webhook automation when you configure the webhook. See the [Configure a webhook]({{< relref "#configure-a-webhook" >}}) section for more information. 

{{% alert %}}
Once you create a secret, you can access that secret in a [webhook automation's payload]({{< relref "/guides/core/automations/create-automations/webhook.md" >}}) using the format `${SECRET_NAME}`.
{{% /alert %}}

## Rotate a secret
To rotate a secret and update its value:
1. Click the pencil icon in the secret's row to open the secret's details.
1. Set **Secret** to the new value. Optionally click **Reveal secret** to verify the new value.
1. Click **Add secret**. The secret's value updates and no longer resolves to the previous value.

{{% alert %}}
After a secret is created or updated, you can no longer reveal its current value. Instead, rotate the secret to a new value.
{{% /alert %}}

## Delete a secret
To delete a secret:
1. Click the trash icon in the secret's row.
1. Read the confirmation dialog, then click **Delete**. The secret is deleted immediately and permanently.

## Manage access to secrets
A team's automations can use the team's secrets. Before you remove a secret, update or remove automations that use it so they don't stop working.