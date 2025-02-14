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

A W&B secret is a team-level variable that lets you obfuscate a sensitive string such as a credential, API key, password, or token. W&B recommends you use secrets to store any string that you want to protect the plain text content of.

You add a secret to your team's secret manager.

{{% alert %}}
* Only W&B Admins can create, edit, or delete a secret.
* Secrets are also available if you use [W&B Server]({{< relref "/guides/hosting/" >}}) in an Azure, GCP, or AWS deployment. Connect with your W&B account team to discuss how you can use secrets in W&B if you use a different deployment type.
* If you use secrets in W&B Server, you are responsible for configuring security measures that satisfy your security needs. 

  - W&B strongly recommends that you store secrets in a W&B instance of a cloud secrets manager provided by AWS, GCP, or Azure. Secret managers provided by AWS, GCP, and Azure are configured with advanced security capabilities.

  - W&B does not recommend that you use a Kubernetes cluster as the backend of your secrets store. Consider a Kubernetes cluster only if you are not able to use a W&B instance of a cloud secrets manager (AWS, GCP, or Azure), and you understand how to prevent security vulnerabilities that can occur if you use a cluster.
{{% /alert %}}

## Add a secret
To add a secret:

1. If necessary, generate the sensitive string in the webhook's service. For example, generate an API key or set a password. If necessary, save the sensitive string securely, such as in a password manager.
1. Log in to W&B and go to the **Settings** page.
1. In the **Team Secrets** section, click **New secret**.
1. Using letters, numbers, and `_`, provide a name for the secret.
1. Paste the sensitive string into the **Secret** field.
1. Click **Add secret**.

Specify the secrets you want to use for your webhook automation when you configure the webhook. See the [Configure a webhook]({{< relref "#configure-a-webhook" >}}) section for more information. 

{{% alert %}}
Once you create a secret, you can access that secret in your W&B workflows with `$`.
{{% /alert %}}

## Manage secrets
TODO

## Manage access to secrets
TODO