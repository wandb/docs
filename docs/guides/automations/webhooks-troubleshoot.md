---
description: Troubleshoot your webhook programmatically or with the W&B App UI.
title:  Troubleshoot webhooks
displayed_sidebar: default
---

Interactively troubleshoot your webhook with the W&B App UI or programmatically with a Bash script. You can troubleshoot a webhook when you create a new webhook or edit an existing webhook.

<Tabs
  defaultValue="app"
  values={[
    {label: 'W&B App UI', value: 'app'},
    {label: 'Bash script', value: 'bash'},
  ]}>
  <TabItem value="app">

Interactively test a webhook with the W&B App UI. 

1. Navigate to your W&B Team Settings page.
2. Scroll to the **Webhooks** section.
3. Click on the horizontal three docs (meatball icon) next to the name of your webhook.
4. Select **Test**.
5. From the UI panel that appears, paste your POST request to the field that appears. 
![](/images/models/webhook_ui.png)
6. Click on **Test webhook**.

Within the W&B App UI, W&B posts the response made by your endpoint.

![](/images/models/webhook_ui_testing.gif)

See [Testing Webhooks in Weights & Biases](https://www.youtube.com/watch?v=bl44fDpMGJw&ab_channel=Weights%26Biases) YouTube video to view a real-world example.

  </TabItem>
  <TabItem value="bash">

The following bash script generates a POST request similar to the POST request W&B sends to your webhook automation when it is triggered.

Copy and paste the code below into a shell script to troubleshoot your webhook. Specify your own values for the following:

* `ACCESS_TOKEN`
* `SECRET`
* `PAYLOAD`
* `API_ENDPOINT`


```sh title="webhook_test.sh"
#!/bin/bash

# Your access token and secret
ACCESS_TOKEN="your_api_key" 
SECRET="your_api_secret"

# The data you want to send (for example, in JSON format)
PAYLOAD='{"key1": "value1", "key2": "value2"}'

# Generate the HMAC signature
# For security, Wandb includes the X-Wandb-Signature in the header computed 
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

  </TabItem>
</Tabs>