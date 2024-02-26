---
description: Get Slack notifications when a new model version is linked to the model registry.
displayed_sidebar: default
---

# Create alerts and notifications

<!-- # Notifications for new model versions -->
Get Slack notifications when a new model version is linked to the model registry. 


1. Navigate to the W&B Model Registry app at [https://wandb.ai/registry/model](https://wandb.ai/registry/model).
2. Select the registered model you want to receive notifications from.
3. Click on the **Connect Slack** button.
    ![](/images/models/connect_to_slack.png)
4. This will redirect you to an OAuth page with instructions on how to enable W&B in your Slack workspace.


Once you have configured Slack notifications for your team, you can pick and choose registered models to get notifications from. 

:::info
A toggle that reads **New model version linked to...** will appear instead of a **Connect Slack** button if you already have Slack notifications configured for your team.
:::

The screenshot below shows a FMNIST Classifier registered model that has Slack notifications. 

![](/images/models/conect_to_slack_fmnist.png)

A message is automatically posted to the connected Slack channel each time a new model version is linked to the FMNIST Classifer registered model.