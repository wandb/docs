---
title: "What is a service account, and why is it useful?"
tags:
   - 
---

A service account (Enterprise-only feature) is an API key that has permissions to write to your team, but is not associated with a particular user. Among other things, service accounts are useful for tracking automated jobs logged to wandb, like periodic retraining, nightly builds, and so on. If you'd like, you can associate a username with one of these machine-launched runs with the [environment variable](../track/environment-variables.md) `WANDB_USERNAME`.

Refer to [Team Service Account Behavior](../app/features/teams.md#team-service-account-behavior) for more information.

You can get the API key in your Team Settings page `/teams/<your-team-name>` where you invite new team members. Select service and click create to add a service account.

![Create a service account on your team settings page for automated jobs](/images/technical_faq/what_is_service_account.png)