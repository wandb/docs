---
description: Send alerts, triggered from your Python code, to your Slack or email
title: Send an alert
---
{{< cta-button colabLink="http://wandb.me/alerts-colab" >}}

Create alerts with Slack or email if your run crashes or with a custom trigger. For example, you can create an alert if the gradient of your training loop starts to blow up (reports NaN) or a step in your ML pipeline completes. Alerts apply to all projects where you initialize runs, including both personal and team projects.


And then see W&B Alerts messages in Slack (or your email):

{{< img src="/images/track/send_alerts_slack.png" alt="" >}}

## How to create an alert

{{% alert %}}
The following guide only applies to alerts in multi-tenant cloud.

If you're using [W&B Server](../hosting/intro.md) in your Private Cloud or on W&B Dedicated Cloud, then please refer to [this documentation](../hosting/monitoring-usage/slack-alerts.md) to setup Slack alerts.
{{% /alert %}}


There are two main steps to set up an alert:

1. Turn on Alerts in your W&B [User Settings](https://wandb.ai/settings)
2. Add `run.alert()` to your code
3. Confirm alert is set up properly
### 1. Turn on alerts in your W&B User Settings

In your [User Settings](https://wandb.ai/settings):

* Scroll to the **Alerts** section
* Turn on **Scriptable run alerts** to receive alerts from `run.alert()`
* Use **Connect Slack** to pick a Slack channel to post alerts. We recommend the **Slackbot** channel because it keeps the alerts private.
* **Email** will go to the email address you used when you signed up for W&B. We recommend setting up a filter in your email so all these alerts go into a folder and don't fill up your inbox.

You will only have to do this the first time you set up W&B Alerts, or when you'd like to modify how you receive alerts.

{{< img src="/images/track/demo_connect_slack.png" alt="Alerts settings in W&B User Settings" >}}

### 2. Add `run.alert()` to your code

Add `run.alert()` to your code (either in a Notebook or Python script) wherever you'd like it to be triggered

```python
import wandb

run = wandb.init()
run.alert(title="High Loss", text="Loss is increasing rapidly")
```

### 3. Check your Slack or email

Check your Slack or emails for the alert message. If you didn't receive any, make sure you've got emails or Slack turned on for **Scriptable Alerts** in your [User Settings](https://wandb.ai/settings)

### Example

This simple alert sends a warning when accuracy falls below a threshold. In this example, it only sends alerts at least 5 minutes apart.

```python
import wandb
from wandb import AlertLevel

run = wandb.init()

if acc < threshold:
    run.alert(
        title="Low accuracy",
        text=f"Accuracy {acc} is below the acceptable threshold {threshold}",
        level=AlertLevel.WARN,
        wait_duration=300,
    )
```


## How to tag or mention users

Use the at sign `@` followed by the Slack user ID to tag yourself or your colleagues in either the title or the text of the alert. You can find a Slack user ID from their Slack profile page.

```python
run.alert(title="Loss is NaN", text=f"Hey <@U1234ABCD> loss has gone to NaN")
```

## Team alerts

Team admins can set up alerts for the team on the team settings page: `wandb.ai/teams/your-team`. 

Team alerts apply to everyone on your team. W&B recommends using the **Slackbot** channel because it keeps alerts private.

## Change Slack channel to send alerts to

To change what channel alerts are sent to, click **Disconnect Slack** and then reconnect. After you reconnect, pick a different Slack channel.