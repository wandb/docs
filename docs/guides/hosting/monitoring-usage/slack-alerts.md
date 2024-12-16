---
title: Configure Slack alerts
---

Integrate W&B Server with [Slack](https://slack.com/).

## Create the Slack application

Follow the procedure below to create a Slack application.

1. Visit https://api.slack.com/apps and select **Create an App**.

{{< img src="/images/hosting/create_an_app.png" alt="" >}}

2. Provide a name for your app in the **App Name** field.
3. Select a Slack workspace where you want to develop your app in. Ensure that the Slack workspace you use is the same workspace you intend to use for alerts.

{{< img src="/images/hosting/name_app_workspace.png" alt="" >}}

## Configure the Slack application

1. On the left sidebar, select **OAth & Permissions**.

{{< img src="/images/hosting/add_an_oath.png" alt="" >}}

2. Within the Scopes section, provide the bot with the **incoming_webhook** scope. Scopes give your app permission to perform actions in your development workspace.

   For more information about OAuth scopes for Bots, see the Understanding OAuth scopes for Bots tutorial in the Slack api documentation.

{{< img src="/images/hosting/save_urls.png" alt="" >}}

3. Configure the Redirect URL to point to your W&B installation. Use the same URL that your host URL is set to in your local system settings. You can specify multiple URLs if you have different DNS mappings to your instance.

{{< img src="/images/hosting/redirect_urls.png" alt="" >}}

4. Select **Save URLs**.
5. You can optionally specify an IP range under **Restrict API Token Usage**, allow-list the IP or IP range of your W&B instance(s). Limiting the allowed IP address helps further secure your Slack application.

## Register your Slack application with W&B

1. Navigate to the **System Settings** or **System Console** page of your W&B instance, depending on your deployment

2. Depending on the System page you are on follow one of the below options:

- If you are in the **System Console**: go to **Settings** then to **Notifications**

{{< img src="/images/hosting/register_slack_app_console.png" alt="" >}}

- If you are in the **System Settings**: toggle the **Enable a custom Slack application to dispatch alerts** to enable a custom Slack application

{{< img src="/images/hosting/register_slack_app.png" alt="" >}}

3. Supply your **Slack client ID** and **Slack secret** then click **Save**. Navigate to Basic Information in Settings to find your application’s client ID and secret.

4. Verify that everything is working by setting up a Slack integration in the W&B app.
