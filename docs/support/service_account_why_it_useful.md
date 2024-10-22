---
title: "What is a service account, and why is it useful?"
tags:
   - None
---
A service account (Enterprise-only feature) acts as an API key with permissions to write to your team without being tied to a specific user. Service accounts track automated tasks logged to W&B, such as periodic retraining and nightly builds. To associate a username with a machine-launched run, set the environment variable `WANDB_USERNAME`.

For further details, see [Team Service Account Behavior](../guides/app/features/teams.md#team-service-account-behavior).

Obtain the API key from your Team Settings page at `/teams/<your-team-name>`, where new team members are invited. Select "service" and click "create" to add a service account.

![Create a service account on your team settings page for automated jobs](/images/technical_faq/what_is_service_account.png)