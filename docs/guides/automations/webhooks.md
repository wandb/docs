---
description: Use webhooks to trigger W&B automations.
title:  Create webhooks 
displayed_sidebar: default
---

Automate a webhook based on an action with the W&B App UI. To do this, first establish a webhook, then configure the webhook automation. 

:::info
Specify an endpoint for your webhook that has an Address record (A record). W&B does not support connecting to endpoints that are exposed directly with IP addresses such as `[0-255].[0-255].[0-255].[0.255]` or endpoints exposed as `localhost`. This restriction helps protect against server-side request forgery (SSRF) attacks and other related threat vectors.
:::

## Add a secret for authentication or authorization
Secrets are team-level variables that let you obfuscate private strings such as credentials, API keys, passwords, tokens, and more. W&B recommends you use secrets to store any string that you want to protect the plain text content of.

To use a secret in your webhook, you must first add that secret to your team's secret manager.

:::info
* Only W&B Admins can create, edit, or delete a secret.
* Skip this section if the external server you send HTTP POST requests to does not use secrets.  
* Secrets are also available if you use [W&B Server](../hosting/intro.md) in an Azure, GCP, or AWS deployment. Connect with your W&B account team to discuss how you can use secrets in W&B if you use a different deployment type.
:::

There are two types of secrets W&B suggests that you create when you use a webhook automation:

* **Access tokens**: Authorize senders to help secure webhook requests 
* **Secret**: Ensure the authenticity and integrity of data transmitted from payloads

Follow the instructions below to create a webhook:

1. Navigate to the W&B App UI.
2. Click on **Team Settings**.
3. Scroll down the page until you find the **Team secrets** section.
4. Click on the **New secret** button.
5. A modal will appear. Provide a name for your secret in the **Secret name** field.
6. Add your secret into the **Secret** field. 
7. (Optional) Repeat steps 5 and 6 to create another secret (such as an access token) if your webhook requires additional secret keys or tokens to authenticate your webhook.

Specify the secrets you want to use for your webhook automation when you configure the webhook. See the [Configure a webhook](#configure-a-webhook) section for more information. 

:::tip
Once you create a secret, you can access that secret in your W&B workflows with `$`.
:::

:::caution
Considerations if you use secrets in W&B Server:

You are responsible for configuring security measures that satisfy your security needs. 

W&B strongly recommends that you store secrets in a W&B instance of a cloud secrets manager provided by AWS, GCP, or Azure. Secret managers provided by AWS, GCP, and Azure are configured with advanced security capabilities.  

W&B does not recommend that you use a Kubernetes cluster as the backend of your secrets store. Consider a Kubernetes cluster only if you are not able to use a W&B instance of a cloud secrets manager (AWS, GCP, or Azure), and you understand how to prevent security vulnerabilities that can occur if you use a cluster.
:::

## Configure a webhook
Before you can use a webhook, first configure that webhook in the W&B App UI. 

:::info
* Only W&B Admins can configure a webhook for a W&B Team.
* Ensure you already [created one or more secrets](#add-a-secret-for-authentication-or-authorization) if your webhook requires additional secret keys or tokens to authenticate your webhook.
:::

1. Navigate to the W&B App UI.
2. Click on **Team Settings**.
4. Scroll down the page until you find the **Webhooks** section.
5. Click on the **New webhook** button.  
6. Provide a name for your webhook in the **Name** field.
7. Provide the endpoint URL for the webhook in the **URL** field.
8. (Optional) From the **Secret** dropdown menu, select the secret you want to use to authenticate the webhook payload.
9. (Optional) From the **Access token** dropdown menu, select the access token you want to use to authorize the sender.
9. (Optional) From the **Access token** dropdown menu select additional secret keys or tokens required to authenticate a webhook  (such as an access token).

:::note
See the [Troubleshoot your webhook](#troubleshoot-your-webhook) section to view where the secret and access token are specified in
the POST request.
:::


### Add a webhook 
Once you have a webhook configured and (optionally) a secret, navigate to the Model Registry App at [https://wandb.ai/registry/model](https://wandb.ai/registry/model).

1. From the **Event type** dropdown, select an [event type](#event-types).
![](/images/models/webhook_select_event.png)
2. (Optional) If you selected **A new version is added to a registered model** event, provide the name of a registered model from the **Registered model** dropdown. 
![](/images/models/webhook_new_version_reg_model.png)
3. Select **Webhooks** from the **Action type** dropdown. 
4. Click on the **Next step** button.
5. Select a webhook from the **Webhook** dropdown.
![](/images/models/webhooks_select_from_dropdown.png)
6. (Optional) Provide a payload in the JSON expression editor. See the [Example payload](#example-payloads) section for common use case examples.
7. Click on **Next step**.
8. Provide a name for your webhook automation in the **Automation name** field. 
![](/images/models/webhook_name_automation.png)
9. (Optional) Provide a description for your webhook. 
10. Click on the **Create automation** button.
