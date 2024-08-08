---
description: Use webhooks to trigger W&B automations.
title:  Create webhooks 
displayed_sidebar: default
---

Automate a webhook based on an action with the W&B App UI. To do this, first establish a webhook, then configure the webhook automation. 

:::info
Specify an endpoint for your webhook that has an Address record (A record). W&B does not support connecting to endpoints that are exposed directly with IP addresses such as `[0-255].[0-255].[0-255].[0.255]` or endpoints exposed as `localhost`. This restriction helps protect against server-side request forgery (SSRF) attacks and other related threat vectors.
:::

## POST Requests
When you set up a webhook in Weights & Biases, you'll define several key components that are used to construct a secure POST request. Understanding how these components are used in the request can help you configure your webhook correctly and ensure secure data transmission. Below is an outline of how each component you define is utilized in the POST request:

### Access Token (`$ACCESS_TOKEN`)
An Access Token authenticates the request to the target API or service. It is included in the HTTP headers as a bearer token. For example:

```bash
-H "Authorization: Bearer $ACCESS_TOKEN"
```

### Secret (`$SECRET`)
A secret generates a cryptographic HMAC signature that verifies the integrity and authenticity of the payload. The secret itself is not transmitted, but is used to create an HMAC signature of the payload, which is included in the HTTP headers. For example:

```bash
-H "X-Wandb-Signature: $SIGNATURE"
```

### Payload (`$PAYLOAD`)
A payload contains the data you want to send to the endpoint. This can be customized according to the requirements of the receiving API. The payload is included in the body of the POST request. For example:

```bash
-d "$PAYLOAD" API_ENDPOINT
```

### API Endpoint (`$API_ENDPOINT`)
The API Endpoint specifies the URL to which the POST request is sent. The API endpoint is the destination URL for the curl command. For example:

```bash
-d "$PAYLOAD" API_ENDPOINT
```

### Example Bash Script
The following script mimics the POST request that W&B sends on your behalf when the webhook is triggered. Make sure to replace `your_api_key`, `your_api_secret`, the JSON payload, and API_ENDPOINT with your actual data:

```bash
#!/bin/bash

# Your access token and secret
ACCESS_TOKEN="your_api_key" 
SECRET="your_api_secret"

# The data you want to send (for example, in JSON format)
PAYLOAD='{"key1": "value1", "key2": "value2"}'

# Generate the HMAC signature
# For security, W&B includes the X-Wandb-Signature in the header computed 
# from the payload and the shared secret key associated with the webhook 
# using the HMAC with SHA-256 algorithm.
SIGNATURE=$(echo -n "$PAYLOAD" | openssl dgst -sha256 -hmac "$SECRET" -binary | base64)

# Make the cURL request
curl -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "X-Wandb-Signature: $SIGNATURE" \
  -d "$PAYLOAD" API_ENDPOINT
```

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
